"""
File: Ctrl_test.py
Description: World Models "C" stage. A controller reads the frozen vision latent (V) and the frozen memory state (M)
of every UGV cluster head and decides, each time a UAV is ready to leave a charger, WHERE it flies next and
WHETHER (or how long) it charges before leaving. Trained with PPO in the real simulator (not in M's dreams), so
M is only used one step at a time, as the Lyapunov tests showed is enough. Uses the episodes VAE_test.py collected
(same --out folder) to fit V and M, and the world settings recorded there for every new world.

Pipeline
  fit     V: token PCA (--k-token) + the map transform (--linear-map, default the best linear method in
          summary_map.csv), both 8-bit quantised, fitted on the training episodes and frozen.
          M: one model per memory arm and seed, trained exactly as in Lyap_test.py (teacher forcing, early
          stopping on validation NLL), then frozen. Saved under <ctrl-out>/models/.
  train   one controller per (memory arm, charging mode, seed): PPO, V and M frozen, fresh worlds every episode
          (world seeds disjoint from the collected data and from the evaluation worlds). Memory seed s pairs with
          controller seed s, and seed s sees the same sequence of training worlds in every arm.
  eval    every controller (greedy action, no sampling) and the three behaviour policies on the same
          --eval-worlds held-out worlds, so every comparison is paired by world.
  report  summary per arm and paired gaps.

Memory arms (--memories, ablation over everything Lyap_test.py compared)
  none            no memory: the controller sees the vision latent only (World Models' "V only" ablation)
  lstm            plain LSTM (most accurate, no certificate)
  lstm_diss@r     LSTM with the projected stability penalty
  ren@r           REN, contraction certified by construction
  attn            L2 self-attention over a 2 h window
  lren@rgG        Lipschitz-bounded REN: contraction r and input -> prediction gain <= G certified

Charging (--charging binary, default): leave now, or charge here first (until 80 % or 20 min, the
  simulator's charge exit). --charging duration (0 / 5 / 10 / 15 / 20 min) is kept for comparison.

Knowledge (the episodes must be collected with VAE_test.py --knowledge local)
  Each UAV runs its own world model on what it alone knows: its own contacts, charges and uploads, the state of
  the charger it is at, and records swapped with another UAV while both are airborne within PLoRa range with
  line of sight (VAE_test.UAVKnowledge). Every charger, the base station included, gets a token from that UAV's
  view (its last reading of the charger, minutes since, AoI, its own data share, the time, its distance to the
  charger and its own battery), compressed by V; M keeps one memory state per (UAV, UGV) (the base station has
  no hidden state: its memory state is M's zero state). A charger's map latent (its local canopy/shade and
  terrain, which that charger holds) is known to a UAV once it has heard from that charger, directly or
  relayed; until then it is 0 (the mean). Terrain and charger locations are known to all (every CH holds them).
  Destination predictor (DestNet, part of the world model): for every other UAV j it forecasts where j is
  heading, from what this UAV knows of j (destination last heard, how long ago, j's position and battery then)
  and its own latent of each charger. A record heard this minute is the truth. Trained on the collected
  episodes (true destinations); accuracy by record age, against "last heard" and uniform guesses, is in
  dest_pred.csv.

Decision (one per visit, when the UAV is at a charger and has no sensor sortie left there)
  The controller sees only the world model's output: for every charger c, [z_c, z_map_c, h_c, o_c] from this
  UAV's world model, o_c being the forecast share of the other UAVs heading to c (the destination predictor).
  Nothing else (no raw distances, batteries or other UAVs' targets): the UAV's battery and its distance to
  each charger reach it only through z. z is re-encoded from the UAV's knowledge when it decides, and its
  choice is announced to the UAVs it is linked with. A shared encoder embeds every charger (so the number of
  chargers can change per world); the destination is a softmax over chargers (the current one masked); the
  charge choice is conditioned on the chosen destination and the current charger. h_c comes from M after it
  consumed the tokens and the UAV's docking up to the previous 5-minute step (World Models: h_t summarises
  z_<t, a_<t); z_c is the latest token.

Memory state (--h-dim 16): the controller gets the first 16 principal components of M's standardised state
  (fitted on the training sequences), so every memory arm hands it the same small input; 0 = the full state.

Warm start (--warm-start greedy): before PPO the controller imitates a teacher acting through the same
  interface on the same UAV-local knowledge: greedy on the UAV's own estimate of each charger's peak AoI minus
  flight time, skipping chargers its world model forecasts another UAV is heading to, charging when the leg
  would leave less than --teacher-soc %. Values are fitted to
  the teacher's returns; imitation accuracy is in ctrl_bc.csv. PPO then improves on it, gently (--lr 1e-4,
  --ent 0.001), and every --val-every episodes the greedy-action policy is scored on --val-worlds held-out
  worlds; the best one (the warm start included) is kept.

Safety shield (on): the simulator's reserve check and charger diversion stay active underneath the controller
  (a UAV whose next leg would breach the reserve tops up first; a UAV whose UGV cannot supply diverts to the best
  reachable charger). The controller's choice is what PPO is trained on; the shield is part of the environment.
  Sensor sorties (cached nearest-neighbour + 2-opt tour) are unchanged. Every world uses the settings the
  episodes were collected with, including the supply margin (VAE_test.py --supply-margin, 5 by default: a UGV
  supplies only above 15 %).

Rewards (the paper's equations, fleet view): every minute
      R_peak = max(1 - max_c peak AoI_c / 120, 0)   R_avg = max(1 - mean_c avg AoI_c / t, 0)   (Eqs. 18, 19)
  with each charger's AoI the fleet's freshest (minimum over UAVs), and per UAV R_chg from the simulator
  (Eq. 20). Destination head: 0.7 R_peak + 0.3 R_avg (Eq. 23); charge head: 0.25 R_peak + 0.25 R_avg + 0.5 R_chg
  (Eq. 22). A dead UAV earns 0 from then on (Eq. 21). Training only: the charge head loses --ugv-penalty (1.0)
  for every minute its UAV sits on a UGV at or below the 10 % reserve (ugv_drain_min); reported rewards
  exclude it. Each head has its own value and advantage (semi-MDP
  returns between a UAV's decisions, discount --gamma per minute).
  Credit (--credit difference, default): six UAVs share one fleet reward, so one UAV's choice barely moves it
  (its correlation with the chosen charger's staleness was ~0.03). Each UAV is instead given the paper's AoI
  terms minus the same terms computed without that UAV (a difference reward: the freshness only it provides;
  correlation ~0.3), plus its own R_chg. --credit team gives every UAV the fleet reward. Logged and evaluated
  rewards are always the fleet (team) values.

Baselines (always evaluated): greedy (highest AoI minus flight time), random sequence and rotation, each with
  the rule-based dynamic charging and the same shield: the behaviour policies the data was collected with.

Outputs (in --ctrl-out, default <out>/controller)
  models/vision.pkl, models/mem_<arm>_s<seed>.pt      frozen V and M
  models/dest.pt, dest_pred.csv                        destination predictor and its accuracy
  ctrl/<arm>__<charging>__s<seed>.pt                    trained controller (ckpt_*.pt while training)
  ctrl_bc.csv         warm start: imitation accuracy and the teacher's own scores
  ctrl_train.csv      learning curve: one row per PPO iteration (val_r_tgt when validated; best = kept)
  ctrl_eval.csv       one row per (arm, charging, seed, world); the eval step resumes from it
  ctrl_summary.csv    per arm: mean and 95 % CI over seeds (baselines: over worlds)
  ctrl_paired.csv     paired gaps per seed vs the greedy baseline and vs the lstm arm with the same charging

Run
  python Ctrl_test.py all --out vae_test_out --k-token 16 --workers 16    # fit, train, eval, report
  python Ctrl_test.py all --out vae_test_out --seeds 3 --memories none lstm lren@0.97g100   # a first pass
  python Ctrl_test.py train --memories lren@0.97g100 --force   # retrain those controllers
"""
import argparse
import math
import os
import pickle
import random
import shutil
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import Lyap_test as L
M = L.M
V = M.V

CHARGE_SETS = {"duration": [0, 5, 10, 15, 20], "binary": [0, 20]}
BASELINES = ("greedy", "random", "rotation")
DEFAULT_MEMORIES = ["none", "lstm", "lstm_diss@0.97", "ren@0.97", "attn", "lren@0.97g30", "lren@0.97g100"]
TRAIN_WORLD_BASE, VAL_WORLD_BASE, EVAL_WORLD_BASE = 200_000, 800_000, 900_000   # data: seed_base (1000) + episode
AOI_SAT = 120.0                                            # Eq. (19): AoI_sat
METRICS = ["r_tgt", "r_chg", "peak_aoi", "avg_aoi", "data_mb", "deaths", "ugv_out_min", "ugv_drain_min", "sensor_out", "diversions",
           "decisions", "charge_min", "ms_decision", "ms_wm_step"]
CTRL_KEYS = ("k_token", "k_map", "linear_map", "h_dim", "warm_start", "bc_episodes", "bc_epochs", "teacher_soc",
             "ugv_penalty", "credit", "gamma", "lam", "clip", "ppo_epochs", "minibatch", "lr", "ent", "vcoef",
             "ctrl_hidden", "train_episodes", "episodes_per_iter", "val_worlds", "val_every")
MEM_KEYS = ("hidden", "ren_state", "ren_neurons", "attn_window", "attn_dim", "attn_heads", "attn_layers", "attn_c",
            "penalty", "mixtures", "mem_epochs", "mem_patience", "mem_batch", "mem_lr", "dock_m", "map_max_train",
            "lren_state", "lren_neurons", "lren_head", "lren_epochs", "lren_patience", "token_every", "history",
            "history_stride", "z_clip")


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("fit", "train", "eval", "report", "all"))
    ap.add_argument("--ctrl-out", default=None, help="results folder (default <out>/controller)")
    ap.add_argument("--memories", nargs="+", default=DEFAULT_MEMORIES,
                    help="memory arms: none, lstm, lstm_diss@r, ren@r, attn, lren@<r>g<gamma>")
    ap.add_argument("--charging", nargs="+", default=["binary"], choices=list(CHARGE_SETS))
    ap.add_argument("--baselines", nargs="*", default=list(BASELINES), choices=BASELINES)
    ap.add_argument("--force", action="store_true",
                    help="delete the trained controllers and eval rows of the selected memories x charging modes "
                         "(not the fitted V and M) and redo them")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="simulator processes")
    p = ap.add_argument_group("PPO")
    p.add_argument("--train-episodes", type=int, default=600, help="training worlds per controller")
    p.add_argument("--episodes-per-iter", type=int, default=12)
    p.add_argument("--gamma", type=float, default=0.99, help="discount per minute (half-life about 69 min)")
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--credit", choices=("difference", "team"), default="difference",
                   help="difference: each UAV's AoI terms are the paper reward minus the reward without that UAV "
                        "(its own contribution); team: every UAV gets the fleet reward")
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ent", type=float, default=0.001)
    p.add_argument("--vcoef", type=float, default=0.5)
    p.add_argument("--ctrl-hidden", type=int, default=64)
    p.add_argument("--h-dim", type=int, default=16,
                   help="memory state given to the controller: its first N principal components (fitted on the "
                        "training sequences, standardised); 0 = the full state")
    p.add_argument("--ugv-penalty", type=float, default=1.0,
                   help="charge-head penalty per minute a UAV sits on a UGV that is at or below the 10%% reserve")
    p.add_argument("--warm-start", choices=("greedy", "none"), default="greedy",
                   help="greedy: before PPO, imitate the greedy policy (destination) and a reserve rule (charge)")
    p.add_argument("--bc-episodes", type=int, default=48, help="teacher episodes for the warm start")
    p.add_argument("--bc-epochs", type=int, default=30)
    p.add_argument("--teacher-soc", type=float, default=40.0,
                   help="teacher charge rule: charge until the leg to the chosen charger leaves this %% battery")
    p.add_argument("--val-worlds", type=int, default=8, help="held-out worlds for picking the best checkpoint")
    p.add_argument("--val-every", type=int, default=60, help="training episodes between validations")
    p.add_argument("--eval-worlds", type=int, default=30)
    p.add_argument("--dest-max-train", type=int, default=300_000, help="pairs for the destination predictor")
    p.add_argument("--quick", action="store_true", help="tiny settings to check the pipeline end to end")
    a, rest = ap.parse_known_args(argv)
    # V and M settings: Lyap_test.py's parser (and through it VAE_test.py's)
    la, b = L.parse(["run", "--out", "vae_test_out"] + rest)
    for k, v in vars(la).items():
        if not hasattr(a, k):
            setattr(a, k, v)
    a.ctrl_out = a.ctrl_out or os.path.join(a.out, "controller")
    if a.quick:
        a.train_episodes, a.episodes_per_iter, a.eval_worlds, a.ppo_epochs = 4, 2, 2, 2
        a.val_worlds, a.val_every, a.bc_episodes, a.bc_epochs = 2, 2, 2, 2
        a.mem_epochs = a.lren_epochs = 2
        a.max_train, a.max_eval, a.map_max_train = 5000, 5000, 3000
        b.max_train, b.max_eval = 5000, 5000
    world_settings(a, b)
    return a, b


def world_settings(a, b):
    """New worlds must be built with the settings the episodes were collected with."""
    import json
    p = os.path.join(a.out, "data", "world_config.json")
    if os.path.exists(p):
        cfg = {**V.LEGACY, **json.load(open(p))}
        for k, v in cfg.items():
            setattr(b, k, tuple(v) if isinstance(v, list) and k not in ("policy_cycle",) else v)
    if getattr(b, "knowledge", "fleet") != "local":
        raise SystemExit(f"{p}: the controller needs episodes collected with --knowledge local (each UAV's own "
                         "knowledge); this folder has the fleet-shared view. Collect into a new --out.")
    b.verbose = False


def arm_tag(mem, charging, seed):
    return f"{mem}__{charging}__s{seed}"


# ==============================================================================================
# fit: frozen V and M
# ==============================================================================================
def cmd_fit(a, b):
    V.resolve_device(b)
    mdir = os.path.join(a.ctrl_out, "models")
    os.makedirs(mdir, exist_ok=True)
    V.check_config(os.path.join(a.ctrl_out, "fit_config.json"), argparse.Namespace(**{**vars(b), **vars(a)}),
                   ("k_token", "k_map", "linear_map") + MEM_KEYS, "fitted V and M")
    mems = [m for m in a.memories if m != "none"]
    todo = [(m, s) for m in mems for s in range(a.seeds)
            if not os.path.exists(os.path.join(mdir, f"mem_{m}_s{s}.pt"))]
    backfill = [(m, s) for m in mems for s in range(a.seeds) if (m, s) not in todo and
                "h_pc" not in torch.load(os.path.join(mdir, f"mem_{m}_s{s}.pt"), weights_only=False)]
    vpath = os.path.join(mdir, "vision.pkl")
    if not todo and not backfill and os.path.exists(vpath) and os.path.exists(os.path.join(mdir, "dest.pt")):
        print("fit: V and every M already fitted", flush=True)
        return
    lin = a.linear_map or M.default_linear(a)[0]
    ch = V.load_track("ch", b)
    mt = b.max_train
    b.max_train = a.map_max_train or mt
    mp = V.load_track("map", b)
    b.max_train = mt
    S = M.load_sequences(a, b)
    tok, mapc = M.vision_pair(lin, 0, a, b, ch, mp, {})
    D, (zmu, zsd) = L.prepare(tok, mapc, S, a)
    zm = mapc.enc(S["train"]["P"])
    if not os.path.exists(vpath):
        bundle = dict(tok=tok, mapc=mapc, linear_map=lin, ch_stats=b._ch_stats, map_stats=b._map_stats,
                      z_clip=b.z_clip, zmu=zmu, zsd=zsd, zm_mu=zm.mean(0), zm_sd=zm.std(0) + 1e-9,
                      token_every=b.token_every, history=b.history, history_stride=b.history_stride, dock_m=a.dock_m,
                      k_token=a.k_token, k_map=a.k_map)
        pickle.dump(bundle, open(vpath, "wb"))
        print(f"fit: vision saved (token PCA {a.k_token} + map {lin} {a.k_map})", flush=True)
    din, dz = D["train"]["inp"].shape[2], D["train"]["zs"].shape[2]
    R = np.abs(D["train"]["inp"]).reshape(-1, din).max(0) + 1e-6
    margs = {k: getattr(a, k) for k in vars(a) if isinstance(getattr(a, k), (int, float, str, bool, type(None)))}
    dpath = os.path.join(mdir, "dest.pt")
    if not os.path.exists(dpath):
        state, f = fit_dest(a, b, pickle.load(open(vpath, "rb")), torch.device(b.device))
        torch.save(dict(state=state, f=f), dpath)
    for mem, seed in backfill:                               # fitted before the principal components were stored
        f = os.path.join(mdir, f"mem_{mem}_s{seed}.pt")
        m, d = load_memory(f, b.device)
        d.update(h_stats(m, D))
        torch.save(d, f)
        print(f"fit: {mem} seed {seed}: memory-state principal components added", flush=True)
    for mem, seed in todo:
        t0 = time.time()
        m = L.train(mem, seed, D, a, b)
        cert = m.certificate() if hasattr(m, "certificate") else (float("nan"),) * 3
        torch.save(dict(arm=mem, seed=seed, state={k: v.cpu() for k, v in m.state_dict().items()}, din=din, dz=dz, R=R,
                        args=margs, val_nll=m.val_nll, epochs=m.epochs, **h_stats(m, D),
                        cert=dict(rho=cert[0], L=cert[1], **getattr(m, "extra", {}))),
                   os.path.join(mdir, f"mem_{mem}_s{seed}.pt"))
        print(f"fit: {mem} seed {seed}: {m.epochs} epochs, val NLL {m.val_nll:.3f}, certificate "
              f"rho={cert[0]:.3g} L={cert[1]:.4g}  ({time.time() - t0:.0f}s)", flush=True)


# ==============================================================================================
# Destination predictor (part of the world model)
# ==============================================================================================
class DestNet(nn.Module):
    """Scores every charger as the destination of another UAV j, from what UAV i knows: i's latent of the charger
    and i's record of j (destination heard, how long ago, j's position and battery then)."""

    def __init__(s, f, H=64):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(f, H), nn.Tanh(), nn.Linear(H, H), nn.Tanh(), nn.Linear(H, 1))
        s.stay = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1))   # trust in the destination heard
        with torch.no_grad():
            s.stay[2].bias.fill_(3.0)

    def forward(s, F, valid):
        """Starts from the destination last heard (a learned trust that depends on its age and the battery
        then) and adds the network's correction for every charger."""
        trust = s.stay(F[..., 0, -2:]).squeeze(-1)                     # [age, battery] are the same for every c
        logit = s.net(F).squeeze(-1) + trust[..., None] * F[..., -3]   # F[..., -3]: heard destination == c
        return logit.masked_fill(~valid, -1e9)


def dest_features(z, peer, ch_xy):
    """z (..., K, k) UAV i's latents of the chargers; peer (..., 5) its record of UAV j; ch_xy (K, 2) / dim.
    -> (..., K, k + 4): [z, distance from j's last known position, heard destination == charger, age, battery]."""
    K = z.shape[-2]
    d = np.hypot(peer[..., None, 2] - ch_xy[:, 0], peer[..., None, 3] - ch_xy[:, 1])
    known = (peer[..., None, 0] == np.arange(K)).astype(np.float32)
    rep = lambda v: np.broadcast_to(v[..., None], d.shape)
    return np.concatenate([z, d[..., None], known[..., None], rep(peer[..., 1])[..., None],
                           rep(peer[..., 4])[..., None]], -1).astype(np.float32)


def dest_data(files, vis, b, cap, rng):
    """Training pairs (UAV i's view, UAV j's true destination) every token step: features (N, Kmax, f), valid,
    target, and the age of i's record of j."""
    mu, sd = vis["ch_stats"]
    out = []
    for f in files:
        d = dict(np.load(f))
        X, _, _, ugv, uav = V.token_grid(d, b, include_base=True)
        n, R, D = X.shape
        U, K = d["uv_peer"].shape[1], len(d["ch_xy"])
        z = ((vis["tok"].enc(np.clip((X.reshape(n * R, D) - mu) / sd, -b.z_clip, b.z_clip)) - vis["zmu"]) / vis["zsd"])
        z = z.reshape(n, U, K, -1).astype(np.float32)
        steps = np.arange(0, d["ch_obs"].shape[0], b.token_every)
        peer, dest = d["uv_peer"][steps], d["uav_dest"][steps]                    # (n, U, U, 5), (n, U)
        alive = d["uav_alive"][steps]
        s_, i_, j_ = np.nonzero(alive[:, :, None] & alive[:, None, :] & ~np.eye(U, dtype=bool)[None]
                                & (dest[:, None, :] >= 0))
        if cap and len(s_) > cap:
            k = rng.choice(len(s_), cap, replace=False)
            s_, i_, j_ = s_[k], i_[k], j_[k]
        Fe = dest_features(z[s_, i_], peer[s_, i_, j_], d["ch_xy"] / float(d["dim"]))
        out.append((Fe, dest[s_, j_].astype(int), peer[s_, i_, j_, 1] * V.EPISODE_STEPS, peer[s_, i_, j_, 0]))
    Kmax = max(o[0].shape[1] for o in out)
    pad = lambda F: np.pad(F, ((0, 0), (0, Kmax - F.shape[1]), (0, 0)))
    F = np.concatenate([pad(o[0]) for o in out])
    valid = np.concatenate([np.arange(Kmax)[None] < o[0].shape[1] for o in out for _ in range(len(o[0]))])
    return F, valid, np.concatenate([o[1] for o in out]), np.concatenate([o[2] for o in out]), \
        np.concatenate([o[3] for o in out])


def fit_dest(a, b, vis, dev):
    """Train the destination predictor on the training episodes, early-stop on validation, report on test."""
    files = sorted(V.glob.glob(os.path.join(a.out, "data", "ep_*.npz")))
    split = dict(zip(("train", "val", "test"), V.split_episodes(files, b)))
    rng = np.random.default_rng(0)
    cap = max(a.dest_max_train // max(len(split["train"]), 1), 50)
    data = {k: dest_data(split[k], vis, b, cap if k == "train" else cap // 2, rng) for k in ("train", "val", "test")}
    tt = lambda x, dt=torch.float32: torch.tensor(x, dtype=dt)
    F, Vd, y = tt(data["train"][0]), tt(data["train"][1], torch.bool), tt(data["train"][2], torch.long)
    net = DestNet(F.shape[2]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()

    def nll(k):
        Fk, Vk, yk = (tt(data[k][0]).to(dev), tt(data[k][1], torch.bool).to(dev), tt(data[k][2], torch.long).to(dev))
        with torch.no_grad():
            lg = torch.cat([net(Fk[i:i + 4096], Vk[i:i + 4096]) for i in range(0, len(Fk), 4096)])
        return ce(lg, yk).item(), lg.argmax(1).cpu().numpy(), torch.log_softmax(lg, 1).cpu().numpy()

    best, bad, state = np.inf, 0, None
    for ep in range(60):
        net.train()
        for idx in torch.randperm(len(F)).split(1024):
            loss = ce(net(F[idx].to(dev), Vd[idx].to(dev)), y[idx].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        v = nll("val")[0]
        if v < best - 1e-4:
            best, bad, state = v, 0, {k: x.detach().cpu().clone() for k, x in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 5:
                break
    net.load_state_dict(state)
    rows = []
    for k in ("val", "test"):
        _, pred, lp = nll(k)
        yk, age, heard = data[k][2], data[k][3], data[k][4]
        for lo, hi, name in ((0, 1, "0 min"), (1, 6, "1-5 min"), (6, 16, "6-15 min"), (16, 31, "16-30 min"),
                             (31, 10_000, ">30 min")):
            sel = (age >= lo) & (age < hi)
            if sel.sum() == 0:
                continue
            nv = data[k][1][sel].sum(1)
            rows.append(dict(split=k, age=name, n=int(sel.sum()), acc=float((pred[sel] == yk[sel]).mean()),
                             nll=float(-lp[sel, yk[sel]].mean()), acc_last_heard=float((heard[sel] == yk[sel]).mean()),
                             acc_uniform=float((1.0 / nv).mean()), nll_uniform=float(np.log(nv).mean())))
    pd.DataFrame(rows).to_csv(os.path.join(a.ctrl_out, "dest_pred.csv"), index=False)
    t = pd.DataFrame(rows)
    t = t[t.split == "test"]
    print("fit: destination predictor (test): " + "; ".join(
        f"{r.age} acc {r.acc:.2f} (last heard {r.acc_last_heard:.2f})" for r in t.itertuples()), flush=True)
    return {k: v.cpu() for k, v in net.state_dict().items()}, F.shape[2]


def dest_occupancy(net, z_i, peer_i, i, ch_xy):
    """Expected share of the other UAVs heading to each charger, as UAV i's world model forecasts it. A record
    heard this minute is the truth: that UAV's destination is taken as known."""
    U, K = len(peer_i), len(z_i)
    others = [j for j in range(U) if j != i]
    if not others:
        return np.zeros(K, np.float32)
    P = peer_i[others]
    Fe = torch.from_numpy(dest_features(np.broadcast_to(z_i, (len(others),) + z_i.shape), P, ch_xy))
    with torch.no_grad():
        p = torch.softmax(net(Fe, torch.ones(Fe.shape[:2], dtype=torch.bool)), 1).numpy()
    fresh = (P[:, 1] == 0) & (P[:, 0] >= 0)
    p[fresh] = np.eye(K, dtype=np.float32)[P[fresh, 0].astype(int)]
    return (p.sum(0) / len(others)).astype(np.float32)


def h_stats(m, D):
    """Statistics of the memory state the controller reads, over the training sequences (teacher forced):
    per-dimension mean and sd, then the principal components of the standardised state."""
    dev = next(m.parameters()).device
    with torch.no_grad():
        X = torch.tensor(D["train"]["inp"])
        H = np.concatenate([m(X[i:i + 64].to(dev))[0].flatten(0, 1).cpu().numpy() for i in range(0, len(X), 64)])
    mu, sd = H.mean(0), H.std(0) + 1e-6
    Hs = (H - mu) / sd
    val, vec = np.linalg.eigh(Hs.T @ Hs / len(Hs))
    o = np.argsort(val)[::-1]
    return dict(h_mu=mu, h_sd=sd, h_pc=vec[:, o].astype(np.float32), h_pc_sd=np.sqrt(np.maximum(val[o], 1e-12)).astype(np.float32),
                h_pc_var=(val[o] / val.sum()).astype(np.float32))


def h_width(a, mem, seed):
    if mem == "none":
        return 0
    n = len(torch.load(os.path.join(a.ctrl_out, "models", f"mem_{mem}_s{seed}.pt"), weights_only=False)["h_mu"])
    return min(a.h_dim, n) if a.h_dim > 0 else n


def load_memory(path, dev="cpu"):
    d = torch.load(path, map_location=dev, weights_only=False)
    ns = argparse.Namespace(**d["args"])
    m = L.build(d["arm"], d["din"], d["dz"], ns, d["R"])
    m.load_state_dict(d["state"])
    m.eval()
    return m, d


# ==============================================================================================
# Online world model: V and M running inside the simulator
# ==============================================================================================
class OnlineWM:
    """Every UAV's own world model, run inside the simulator. Each --token-every minutes it builds, for each UAV
    and each charger, the token VAE_test.token_grid builds from a finished episode (--knowledge local), encodes it
    with the frozen V, and steps the frozen M once per (UAV, UGV) with the previous token and the share of the
    previous 5 minutes that UAV was landed on that UGV (the input M was trained on)."""

    def __init__(s, vis, mem=None, mem_meta=None, h_dim=0):
        s.v, s.mem = vis, mem
        s.hd = 0
        if mem is not None:
            n = len(mem_meta["h_mu"])
            s.hd = min(h_dim, n) if h_dim > 0 else n
            s.h_mu, s.h_sd = mem_meta["h_mu"].astype(np.float32), mem_meta["h_sd"].astype(np.float32)
            s.pc = (mem_meta["h_pc"][:, :s.hd] / mem_meta["h_pc_sd"][None, :s.hd]).astype(np.float32) if h_dim > 0 else None
        s.lags = np.arange(vis["history"]) * vis["history_stride"]

    def h_feat(s, raw):
        """Memory state as the controller sees it: standardised, then (with --h-dim) its leading components."""
        z = (raw - s.h_mu) / s.h_sd
        return (z if s.pc is None else z @ s.pc).astype(np.float32)

    def start(s, env, chs, uavs, doy, T, cluster_size, charger_m, n_feat):
        v = s.v
        s.T, s.U, s.K = T, len(uavs), len(chs)
        s.is_base = np.array([getattr(ch, "is_base", False) for ch in chs])
        s.ugv = np.flatnonzero(~s.is_base)
        s.xy = np.array([(ch.id_x, ch.id_y) for ch in chs], np.float32)
        s.dim = float(env.dim)
        s.sun = np.clip(90.0 - env.sun_zenith, 0.0, 90.0) / 90.0
        s.season = (math.sin(2 * math.pi * doy / 365), math.cos(2 * math.pi * doy / 365))
        s.size = (cluster_size / 10.0).astype(np.float32)
        s.to_base = (charger_m / s.dim).astype(np.float32)
        P = V.node_patches(env, s.xy).astype(np.float64)
        mmu, msd = v["map_stats"]
        s.zm = ((v["mapc"].enc((P - mmu[0]) / msd[0]) - v["zm_mu"]) / v["zm_sd"]).astype(np.float32)   # (K, k_map)
        s.obs = np.zeros((T, s.U, s.K, n_feat), np.float32)
        s.known = np.repeat(s.is_base[None], s.U, 0)         # which chargers' maps each UAV has
        s.z = np.zeros((s.U, s.K, v["k_token"]), np.float32)
        s.z_prev = None
        s.h = np.zeros((s.U, s.K, s.hd), np.float32)
        if s.mem is not None:
            s.h[:] = s.h_feat(np.zeros((1, len(s.h_mu)), np.float32))[0]      # the zero state, transformed
        s.state = None
        s.dock = np.zeros((s.U, s.K))
        s.wm_time, s.wm_steps = 0.0, 0
        s.t_obs = -1

    def token_now(s, i, t, u):
        """Fresh token for UAV i deciding during minute t: its knowledge up to the end of minute t - 1 with its
        current position and battery (V only; M steps on the grid)."""
        tn = max(min(t - 1, s.t_obs), 0)
        return s.token(tn, np.array([[u.id_x, u.id_y]], np.float32), np.array([u.state[0][2]], np.float32), [i])[0]

    def token(s, t, uav_xy, uav_soc, rows=None):
        """Tokens for every UAV (or just rows) from knowledge up to the end of minute t."""
        v = s.v
        rows = np.arange(s.U) if rows is None else np.asarray(rows)
        U, K = len(rows), s.K
        frames = np.clip(t - s.lags, 0, s.T - 1)
        hist = s.obs[frames][:, rows].transpose(1, 2, 0, 3).reshape(U, K, -1)
        tm = np.broadcast_to(np.array([t / s.T, s.season[0], s.season[1], s.sun[t]], np.float32), (U, K, 4))
        dxy = uav_xy[:, None, :] - s.xy[None, :, :]
        dist = (np.hypot(dxy[..., 0], dxy[..., 1]) / s.dim).astype(np.float32)
        batt = np.repeat((uav_soc / 100.0)[:, None], K, 1).astype(np.float32)
        X = np.concatenate([hist, tm, np.broadcast_to(s.size, (U, K))[..., None], dist[..., None],
                            np.broadcast_to(s.to_base, (U, K))[..., None], batt[..., None]],
                           2).astype(np.float64)
        mu, sd = v["ch_stats"]
        X = np.clip((X - mu) / sd, -v["z_clip"], v["z_clip"]).reshape(U * K, -1)
        return ((v["tok"].enc(X) - v["zmu"]) / v["zsd"]).astype(np.float32).reshape(U, K, -1)

    def minute(s, t, obs_t, known_t, uav_xy, uav_soc, docked):
        """End of minute t: record each UAV's knowledge; on token minutes update h, then z. known_t (U, K): the
        chargers each UAV has heard from (their maps are known); docked (U,): the charger each UAV is landed on."""
        s.obs[t] = obs_t
        s.t_obs = t
        te = s.v["token_every"]
        if t % te == 0:
            t0 = time.perf_counter()
            if s.z_prev is not None and s.mem is not None:
                g = s.ugv
                zm = np.broadcast_to(s.zm[g], (s.U, len(g), s.zm.shape[1])) * s.known[:, g, None]
                x = np.concatenate([s.z_prev[:, g], zm, (s.dock[:, g] / te)[..., None]],
                                   2).reshape(s.U * len(g), -1).astype(np.float32)
                with torch.no_grad():
                    out = s.mem(torch.from_numpy(x)[:, None, :], s.state)
                s.state = out[1]
                s.h[:, g] = s.h_feat(out[0][:, -1].numpy()).reshape(s.U, len(g), -1)
            s.dock[:] = 0.0
            s.known = known_t.copy()
            s.z = s.token(t, uav_xy, uav_soc)
            s.z_prev = s.z
            s.wm_time += time.perf_counter() - t0
            s.wm_steps += 1
        for i, c in enumerate(docked):
            if c >= 0:
                s.dock[i, c] += 1.0


# ==============================================================================================
# Controller
# ==============================================================================================
class Controller(nn.Module):
    """Shared per-charger encoder over the world model's output; destination = softmax over chargers;
    charge | destination; two values. No other input."""

    def __init__(s, dc, nC, H=64):
        super().__init__()
        s.enc = nn.Sequential(nn.Linear(dc, 2 * H), nn.Tanh(), nn.Linear(2 * H, H), nn.Tanh())
        s.ctx = nn.Sequential(nn.Linear(2 * H, H), nn.Tanh())
        s.dest = nn.Sequential(nn.Linear(2 * H, H), nn.Tanh(), nn.Linear(H, 1))
        s.chg = nn.Sequential(nn.Linear(3 * H, H), nn.Tanh(), nn.Linear(H, nC))
        s.val = nn.Sequential(nn.Linear(H, H), nn.Tanh(), nn.Linear(H, 2))

    def forward(s, F, mask, here):
        """F (B, K, dc), mask (B, K) True = allowed destination, here (B,). Rows that are neither allowed nor
        the current charger are padding."""
        e = s.enc(F)
        ar = torch.arange(F.shape[1], device=F.device)[None]
        valid = (mask | (ar == here[:, None])).float()
        pool = (e * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp(min=1.0)
        eh = e[torch.arange(len(F)), here]
        q = s.ctx(torch.cat([eh, pool], -1))
        logit = s.dest(torch.cat([e, q[:, None].expand_as(e)], -1)).squeeze(-1)
        return logit.masked_fill(~mask, -1e9), q, e, eh

    def charge_logits(s, q, e, eh, d):
        return s.chg(torch.cat([q, e[torch.arange(len(d)), d], eh], -1))

    def value(s, q):
        return s.val(q)


class CtrlAgent:
    """VisitPolicy-compatible: the simulator calls act() while a UAV sits at a charger with nothing to sweep.
    The first call of a visit decides (destination, charge) from this UAV's world model only; later calls in that
    visit return the commitment."""

    def __init__(s, net, wm, charges, sample, dest=None):
        s.net, s.wm, s.charges, s.sample, s.dest = net, wm, charges, sample, dest
        s.uav, s.commit, s.records, s.dec_time = None, {}, [], 0.0

    def start(s, chs, uavs, is_base, t_ref, know):
        s.chs, s.uavs, s.is_base, s.t_ref, s.know = chs, uavs, is_base, t_ref, know
        s.K = len(chs)
        s.xy = np.array([(c.id_x, c.id_y) for c in chs], float)

    def features(s, i, here):
        """For every charger, UAV i's world model output: [z, z_map, h, forecast share of the other UAVs heading
        there]. z is re-encoded from its knowledge now; a charger's map is known once it has heard from it."""
        wm, t = s.wm, s.t_ref[0]
        z = wm.token_now(i, t, s.uavs[i])
        known = (s.know.ref[i] > -10_000.0) | s.is_base
        occ = dest_occupancy(s.dest, z, s.know.peers(t)[i], i, s.xy / wm.dim)
        F = np.concatenate([z, wm.zm * known[:, None], wm.h[i], occ[:, None]], 1).astype(np.float32)
        mask = np.ones(s.K, bool)
        mask[here] = False
        return F, mask

    def decide(s, F, mask, here):
        with torch.no_grad():
            logit, q, e, eh = s.net(torch.from_numpy(F)[None], torch.from_numpy(mask)[None], torch.tensor([here]))
            pd_ = torch.distributions.Categorical(logits=logit[0])
            d = int(pd_.sample()) if s.sample else int(logit[0].argmax())
            cl = s.net.charge_logits(q, e, eh, torch.tensor([d]))[0]
            pc = torch.distributions.Categorical(logits=cl)
            c = int(pc.sample()) if s.sample else int(cl.argmax())
            return d, c, float(pd_.log_prob(torch.tensor(d))), float(pc.log_prob(torch.tensor(c))), s.net.value(q)[0].numpy()

    def act(s, state):
        u = s.uav
        here = u.target.cluster_num
        com = s.commit.get(id(u))
        if com is not None and com["here"] is u.target:
            return com["dest"]
        t0 = time.perf_counter()
        i = s.uavs.index(u)
        F, mask = s.features(i, here)
        if not mask.any():
            return here
        d, c, lpd, lpc, v = s.decide(F, mask, here)
        s.dec_time += time.perf_counter() - t0
        return s.commit_to(u, i, here, F, mask, d, c, lpd, lpc, v)

    def commit_to(s, u, i, here, F, mask, d, c, lpd, lpc, v):
        minutes = s.charges[c]
        if minutes > 0:
            u.is_charging = True
            u.p_count = max(u.p_count, minutes + 1)     # set_dest takes one off before the exit check
        s.commit[id(u)] = dict(here=u.target, dest=d)
        s.know.announce(i, d, s.t_ref[0])
        s.records.append(dict(F=F, mask=mask, here=here, d=d, c=c, lpd=lpd, lpc=lpc, v=v, t=s.t_ref[0], uav=i,
                              minutes=minutes))
        return d


class TeacherAgent(CtrlAgent):
    """Warm-start teacher acting through the controller's interface (same features recorded, same commitment),
    deciding from the same UAV-local knowledge uncompressed: destination = greedy on this UAV's own estimate of
    each charger's peak AoI minus the flight minutes, skipping chargers its world model forecasts another UAV is
    heading to; charge when the leg there would leave less than
    --teacher-soc %, unless the UAV's own reading of this UGV is at or below 20 %."""

    def __init__(s, wm, charges, teacher_soc, dest):
        super().__init__(None, wm, charges, False, dest)
        s.teacher_soc = teacher_soc

    def act(s, state):
        from sim_mods import ablation
        from sim_mods.ablation import CFG
        u = s.uav
        here = u.target.cluster_num
        com = s.commit.get(id(u))
        if com is not None and com["here"] is u.target:
            return com["dest"]
        i = s.uavs.index(u)
        F, mask = s.features(i, here)
        if not mask.any():
            return here
        mins = np.array([ablation.leg_minutes(math.hypot(s.xy[c, 0] - u.id_x, s.xy[c, 1] - u.id_y)) for c in range(s.K)])
        free = mask & (F[:, -1] * (len(s.uavs) - 1) < 0.5)    # skip chargers it forecasts others heading to
        score = np.where(free if free.any() else mask, s.know.estP[i] - mins, -np.inf)
        d = int(np.argmax(score))
        after = u.state[0][2] - CFG.tour_margin * ablation.minutes_pct(mins[d])
        low = not s.is_base[here] and u.known_charge(s.chs[here]) <= 20.0
        need = after < s.teacher_soc and not low
        c = len(s.charges) - 1 if need else 0
        if need and len(s.charges) > 2:                   # duration mode: shortest option that is enough
            want = math.ceil((s.teacher_soc - after) / 10.0)
            c = next((k for k, m in enumerate(s.charges) if m >= want), len(s.charges) - 1)
        return s.commit_to(u, i, here, F, mask, d, c, 0.0, 0.0, np.zeros(2))


class BaselineAgent:
    """The data-collection behaviour policies (VisitPolicy) with dynamic charging."""

    def __init__(s, kind):
        s.kind, s.records, s.dec_time, s.uav = kind, [], 0.0, None

    def make(s, chs, uavs, rng):
        from sim_mods import ablation
        s.pol = V.VisitPolicy(s.kind, chs, uavs, rng, ablation.UAV_SPEED)

    def act(s, state):
        s.pol.uav = s.uav
        t0 = time.perf_counter()
        r = s.pol.act(state)
        s.dec_time += time.perf_counter() - t0
        return r


# ==============================================================================================
# One episode in a new world
# ==============================================================================================
def run_world(world_seed, b, agent, wm=None):
    """Mirrors VAE_test.run_episode (same resets, same order of simulator calls, same fleet observation) with the
    agent choosing destinations. Returns per-minute rewards, the agent's records and episode metrics."""
    from sim_mods import ablation
    from sim_mods.ablation import CFG
    env, info = V.build_world(world_seed, b)
    CFG.charging = "external" if isinstance(agent, CtrlAgent) else "dynamic"
    ep_rng = np.random.default_rng(world_seed + 1)
    random.seed(world_seed)
    np.random.seed(world_seed % (2 ** 32))
    sensors = [env.sensor_table.iat[i, 0] for i in range(len(env.sensor_table))]
    chs = [env.ch_table.iat[i, 0] for i in range(len(env.ch_table))]
    uavs = [env.uav_table.iat[i, 0] for i in range(len(env.uav_table))]
    K, U, S, T = len(chs), len(uavs), len(sensors), V.EPISODE_STEPS
    start = [(u.id_x, u.id_y) for u in uavs]
    for s_ in sensors:
        s_.reset()
        s_.stored_energy = float(ep_rng.uniform(*b.sensor_start_soc))
        s_.harvested_mAh = 0.0
    is_base = np.array([getattr(ch, "is_base", False) for ch in chs])
    for ch in chs:
        ch.reset()
        if not getattr(ch, "is_base", False):
            ch.stored_energy = float(ep_rng.uniform(*b.ch_start_soc))
    for u, (x, y) in zip(uavs, start):
        u.reset()
        u.id_x, u.id_y = x, y
        u.target_x, u.target_y = x, y
        u.stored_energy = float(ep_rng.uniform(*b.uav_start_soc))
        u.state[0][2] = u.stored_energy
    sid = {id(s_): i for i, s_ in enumerate(sensors)}
    cluster = np.zeros(S, int)
    for c, ch in enumerate(chs):
        for j in range(len(ch.sens_table)):
            cluster[sid[id(ch.sens_table.iat[j, 0])]] = c
    cluster_size = np.bincount(cluster, minlength=K).astype(np.float32)
    charger_m = np.array([getattr(ch, "charger_m", 0.0) for ch in chs], np.float32)
    t_ref = [0]
    know = V.UAVKnowledge(env, chs, uavs, T, b.uav_alt, b.plora_m, b.intent_min)   # what each UAV knows (as collected)
    if wm is not None:
        wm.start(env, chs, uavs, info["doy"], T, cluster_size, charger_m, len(V.CH_FEATS))
    if isinstance(agent, CtrlAgent):
        agent.start(chs, uavs, is_base, t_ref, know)
    else:
        agent.make(chs, uavs, ep_rng)
    power = V.NullPower()

    last_contact = np.full(K, -10_000.0)
    last_short = np.full(K, -10_000.0)
    short_seen = np.array([ch.outage_minutes for ch in chs], float)
    dead = np.zeros(U, bool)
    died_at = np.full(U, T)
    r_tgt = np.zeros(T)                       # fleet (team) destination reward, Eq. (23)
    d_tgt = np.zeros((U, T))                  # each UAV's difference reward: fleet reward minus fleet-without-it
    d_aoi = np.zeros((U, T))                  # the same for the AoI part of the charge reward, Eq. (22)
    r_chg = np.zeros((U, T))
    peak_t, avg_t = np.zeros(T), np.zeros(T)
    ugv_out, sens_out, diversions = 0, 0.0, 0
    drain = np.zeros((U, T))                  # UAV sat on a UGV at or below the reserve this minute
    seek = np.zeros(U, bool)
    for t in range(T):
        t_ref[0] = t
        for s_ in sensors:
            s_.harvest_energy(env, t)
            sens_out += not s_.harvest_data()
        for ch in chs:
            ch.harvest_energy(env, t)
            ch.download(t)
        for i, u in enumerate(uavs):
            if dead[i]:
                continue
            agent.uav = u
            u.set_dest(agent, power, t)
            u.navigate_step(env)
            u.receive_data(t)
            rc = u.receive_energy()
            u.step_battery()
            if u.crash or u.stored_energy <= 0:
                dead[i] = u.crash = True
                died_at[i] = t
                continue
            r_chg[i, t] = min(max(rc, 0.0), 1.0)
            if u.seek_charge and not seek[i]:
                diversions += 1
            seek[i] = u.seek_charge
            tgt = u.target
            if tgt is not None and tgt.type == 2 and u.inRange:
                last_contact[tgt.cluster_num] = t
        tel_soc = np.array([ch.perceived_soc / 100.0 for ch in chs])
        now_short = np.array([ch.outage_minutes for ch in chs], float)
        last_short[now_short > short_seen] = t
        short_seen = now_short
        for s_ in sensors:
            s_.step_battery()
        for ch in chs:
            ch.step_battery()
        ugv_out += sum(ch.stored_energy <= V.CH_OUT_SOC for c, ch in enumerate(chs) if not is_base[c])
        for i, u in enumerate(uavs):
            tg = u.target
            if (not dead[i] and tg is not None and tg.type == 2 and not is_base[tg.cluster_num] and u.h == 0
                    and tg.stored_energy <= V.CH_OUT_SOC):
                drain[i, t] = 1.0
        data_tot = max(sum(u.state[0][1] for u in uavs), 1)
        age = np.where(dead, t - died_at, 0.0)
        obs = np.zeros((K, len(V.CH_FEATS)), np.float32)
        Pk = np.array([[u.state[c + 1][2] for c in range(K)] for u in uavs], float) + age[:, None]   # (U, K)
        Av = np.array([[u.state[c + 1][3] for c in range(K)] for u in uavs], float) + age[:, None]
        peaks, avgs = Pk.min(0), Av.min(0)
        for c in range(K):
            obs[c] = [tel_soc[c], min(t - last_contact[c], T) / T, min(t - last_short[c], T) / T,
                      min(peaks[c], T) / T, min(avgs[c], T) / T, sum(u.state[c + 1][1] for u in uavs) / data_tot]
        rp = lambda pk: max(1.0 - pk.max() / AOI_SAT, 0.0)
        ra = lambda av: max(1.0 - av.mean() / max(t, 1), 0.0)
        r_peak, r_avg = rp(peaks), ra(avgs)
        r_tgt[t] = 0.7 * r_peak + 0.3 * r_avg
        if U > 1:                                   # fleet view without UAV i: min over the other UAVs
            for i in range(U):
                if dead[i]:
                    continue
                o = np.delete(np.arange(U), i)
                p_i, a_i = rp(Pk[o].min(0)), ra(Av[o].min(0))
                d_tgt[i, t] = 0.7 * (r_peak - p_i) + 0.3 * (r_avg - a_i)
                d_aoi[i, t] = 0.25 * (r_peak - p_i) + 0.25 * (r_avg - a_i)
        else:
            d_tgt[0, t], d_aoi[0, t] = r_tgt[t], 0.25 * (r_peak + r_avg)
        d_chg = d_aoi[:, t] + 0.5 * r_chg[:, t]
        r_chg[:, t] = np.where(dead, 0.0, 0.25 * r_peak + 0.25 * r_avg + 0.5 * r_chg[:, t])
        d_aoi[:, t] = np.where(dead, 0.0, d_chg)
        peak_t[t], avg_t[t] = peaks.max(), avgs.mean()
        know.update(t, dead, age)                    # same place in the minute as VAE_test.run_episode
        if wm is not None:
            soc = np.array([0.0 if dead[i] else u.state[0][2] for i, u in enumerate(uavs)], np.float32)
            dk = np.array([u.target.cluster_num if (not dead[i] and u.target is not None and u.target.type == 2
                                                    and u.h == 0) else -1 for i, u in enumerate(uavs)])
            wm.minute(t, know.obs(t), (know.ref > -10_000.0) | is_base[None], np.array([(u.id_x, u.id_y) for u in uavs], np.float32), soc, dk)
    n_dec = len(agent.records) if isinstance(agent, CtrlAgent) else 0
    met = dict(r_tgt=r_tgt.mean(), r_chg=r_chg.mean(), peak_aoi=peak_t.mean(), avg_aoi=avg_t.mean(),
               data_mb=sum(u.state[0][1] for u in uavs) / 1e6, deaths=int(dead.sum()),
               ugv_out_min=float(ugv_out), ugv_drain_min=float(drain.sum()), sensor_out=sens_out / (S * T), diversions=diversions, decisions=n_dec,
               charge_min=float(np.mean([r["minutes"] for r in agent.records])) if n_dec else float("nan"),
               ms_decision=1e3 * agent.dec_time / max(n_dec, 1) if n_dec else float("nan"),
               ms_wm_step=1e3 * wm.wm_time / max(wm.wm_steps, 1) if wm is not None and wm.mem is not None else float("nan"),
               n_chargers=K, n_ugv=int((~is_base).sum()))
    return dict(r_tgt=r_tgt, r_chg=r_chg, d_tgt=d_tgt, d_chg=d_aoi, drain=drain, died_at=died_at, records=getattr(agent, "records", []), metrics=met)


# ==============================================================================================
# Worker processes
# ==============================================================================================
_W = {}


def _init_worker(ctrl_out, mem, seed, charging, b, hidden, h_dim=0):
    torch.set_num_threads(1)
    _W.update(b=b, charges=CHARGE_SETS.get(charging))
    if mem is None:
        return
    vis = pickle.load(open(os.path.join(ctrl_out, "models", "vision.pkl"), "rb"))
    m, meta = (None, None) if mem == "none" else load_memory(os.path.join(ctrl_out, "models", f"mem_{mem}_s{seed}.pt"))
    _W["wm"] = OnlineWM(vis, m, meta, h_dim)
    dc = vis["k_token"] + vis["k_map"] + _W["wm"].hd + 1
    dd = torch.load(os.path.join(ctrl_out, "models", "dest.pt"), weights_only=False)
    _W["dest"] = DestNet(dd["f"])
    _W["dest"].load_state_dict(dd["state"])
    _W["dest"].eval()
    _W["net"] = Controller(dc, len(_W["charges"]), hidden)
    _W["net"].eval()


def _rollout(job):
    world, weights, sample, kind = job
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):            # the simulator prints
        if kind == "ctrl":
            _W["net"].load_state_dict({k: torch.from_numpy(v) for k, v in weights.items()})
            agent = CtrlAgent(_W["net"], _W["wm"], _W["charges"], sample, _W["dest"])
            torch.manual_seed(world)
            out = run_world(world, _W["b"], agent, _W["wm"])
        elif kind == "teacher":                                # weights = --teacher-soc here
            out = run_world(world, _W["b"], TeacherAgent(_W["wm"], _W["charges"], weights, _W["dest"]), _W["wm"])
        else:
            out = run_world(world, _W["b"], BaselineAgent(kind))
    out["world"] = world
    return out


def pool(a, mem, seed, charging, b):
    import multiprocessing as mp
    return ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn"), initializer=_init_worker,
                               initargs=(a.ctrl_out, mem, seed, charging, b, a.ctrl_hidden, a.h_dim))


# ==============================================================================================
# PPO
# ==============================================================================================
def advantages(ep, a):
    """Semi-MDP returns between each UAV's decisions, one stream per head, then GAE."""
    recs, out = ep["records"], []
    g = a.gamma
    T = len(ep["r_tgt"])
    disc = g ** np.arange(T + 1)
    for u in sorted({r["uav"] for r in recs}):
        rs = sorted([r for r in recs if r["uav"] == u], key=lambda r: r["t"])
        end_u = min(int(ep["died_at"][u]) + 1, T)
        Rs, Ds = [], []
        for k, r in enumerate(rs):
            t0 = r["t"]
            t1 = rs[k + 1]["t"] if k + 1 < len(rs) else end_u
            w = disc[:max(t1 - t0, 0)]
            if a.credit == "difference":
                rt, rc = ep["d_tgt"][u, t0:t1], ep["d_chg"][u, t0:t1]
            else:
                rt, rc = ep["r_tgt"][t0:t1], ep["r_chg"][u, t0:t1]
            rc = rc - a.ugv_penalty * ep["drain"][u, t0:t1]
            Rs.append((1 - g) * np.array([(w * rt).sum(), (w * rc).sum()]))
            Ds.append(g ** (t1 - t0) if k + 1 < len(rs) else 0.0)
        A = np.zeros((len(rs), 2))
        nxt = np.zeros(2)
        for k in reversed(range(len(rs))):
            vn = rs[k + 1]["v"] if k + 1 < len(rs) else np.zeros(2)
            delta = Rs[k] + Ds[k] * vn - rs[k]["v"]
            nxt = delta + Ds[k] * a.lam * (nxt if k + 1 < len(rs) else 0.0)
            A[k] = nxt
        for k, r in enumerate(rs):
            out.append(dict(r, adv=A[k], ret=A[k] + r["v"]))
    return out


def batch_tensors(rows, dev):
    Kmax = max(len(r["mask"]) for r in rows)
    dc = rows[0]["F"].shape[1]
    n = len(rows)
    F = np.zeros((n, Kmax, dc), np.float32)
    Mk = np.zeros((n, Kmax), bool)
    for i, r in enumerate(rows):
        k = len(r["mask"])
        F[i, :k], Mk[i, :k] = r["F"], r["mask"]
    st = lambda key, dt=np.float32: torch.tensor(np.array([r[key] for r in rows], dt), device=dev)
    return dict(F=torch.tensor(F, device=dev), mask=torch.tensor(Mk, device=dev), here=st("here", np.int64),
                d=st("d", np.int64), c=st("c", np.int64), lpd=st("lpd"), lpc=st("lpc"), adv=st("adv"), ret=st("ret"))


def ppo_update(net, opt, rows, a, dev):
    B = batch_tensors(rows, dev)
    adv = B["adv"]
    adv = (adv - adv.mean(0)) / (adv.std(0) + 1e-8)
    n = len(rows)
    stats = []
    for _ in range(a.ppo_epochs):
        for idx in torch.randperm(n, device=dev).split(a.minibatch):
            logit, q, e, eh = net(B["F"][idx], B["mask"][idx], B["here"][idx])
            pd_ = torch.distributions.Categorical(logits=logit)
            cl = net.charge_logits(q, e, eh, B["d"][idx])
            pc = torch.distributions.Categorical(logits=cl)
            loss = 0.0
            for lp_new, lp_old, A, ent in ((pd_.log_prob(B["d"][idx]), B["lpd"][idx], adv[idx, 0], pd_.entropy()),
                                           (pc.log_prob(B["c"][idx]), B["lpc"][idx], adv[idx, 1], pc.entropy())):
                ratio = (lp_new - lp_old).exp()
                loss = loss - torch.min(ratio * A, ratio.clamp(1 - a.clip, 1 + a.clip) * A).mean() - a.ent * ent.mean()
            vloss = ((net.value(q) - B["ret"][idx]) ** 2).mean()
            loss = loss + a.vcoef * vloss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            stats.append((vloss.item(), pd_.entropy().mean().item(), pc.entropy().mean().item()))
    return np.mean(stats, 0)


def warm_start(net, a, ex, seed, dev, tag):
    """Imitate the teacher (greedy destination, reserve-rule charge) and fit the values to its returns."""
    worlds = [TRAIN_WORLD_BASE + seed * 100_000 + 90_000 + i for i in range(a.bc_episodes)]
    eps = list(ex.map(_rollout, [(w, a.teacher_soc, False, "teacher") for w in worlds]))
    rows = [r for ep in eps for r in advantages(ep, a)]
    B = batch_tensors(rows, dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    net.train()
    for _ in range(a.bc_epochs):
        for idx in torch.randperm(len(rows), device=dev).split(a.minibatch):
            logit, q, e, eh = net(B["F"][idx], B["mask"][idx], B["here"][idx])
            cl = net.charge_logits(q, e, eh, B["d"][idx])
            loss = ce(logit, B["d"][idx]) + ce(cl, B["c"][idx]) + a.vcoef * ((net.value(q) - B["ret"][idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
    net.eval()
    with torch.no_grad():
        logit, q, e, eh = net(B["F"], B["mask"], B["here"])
        acc_d = (logit.argmax(1) == B["d"]).float().mean().item()
        acc_c = (net.charge_logits(q, e, eh, B["d"]).argmax(1) == B["c"]).float().mean().item()
    m = pd.DataFrame([ep["metrics"] for ep in eps]).mean(numeric_only=True)
    L.append(os.path.join(a.ctrl_out, "ctrl_bc.csv"), dict(
        arm=tag.split("__")[0], charging=tag.split("__")[1], seed=seed, episodes=len(eps), decisions=len(rows),
        acc_dest=acc_d, acc_charge=acc_c, **{f"teacher_{k}": m[k] for k in ("r_tgt", "r_chg", "peak_aoi", "ugv_out_min",
                                                                            "ugv_drain_min", "deaths", "charge_min")}))
    print(f"train {tag}: warm start on {len(eps)} teacher episodes ({len(rows)} decisions): match greedy "
          f"destination {acc_d:.0%}, charge {acc_c:.0%}; teacher r_tgt {m['r_tgt']:.3f}, peak AoI {m['peak_aoi']:.0f}",
          flush=True)


def train_one(a, b, mem, charging, seed):
    cdir = os.path.join(a.ctrl_out, "ctrl")
    os.makedirs(cdir, exist_ok=True)
    tag = arm_tag(mem, charging, seed)
    final, ckpt = os.path.join(cdir, tag + ".pt"), os.path.join(cdir, "ckpt_" + tag + ".pt")
    if os.path.exists(final):
        return
    dev = torch.device(b.device)
    torch.manual_seed(seed)
    vis = pickle.load(open(os.path.join(a.ctrl_out, "models", "vision.pkl"), "rb"))
    hd = h_width(a, mem, seed)
    net = Controller(vis["k_token"] + vis["k_map"] + hd + 1, len(CHARGE_SETS[charging]), a.ctrl_hidden).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    done = 0
    if os.path.exists(ckpt):
        c = torch.load(ckpt, map_location=dev, weights_only=False)
        net.load_state_dict(c["net"])
        opt.load_state_dict(c["opt"])
        done = c["episodes"]
        print(f"train {tag}: resuming after {done} episodes", flush=True)
    log = os.path.join(a.ctrl_out, "ctrl_train.csv")
    vworlds = [VAL_WORLD_BASE + i for i in range(a.val_worlds)]
    weights = lambda: {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}

    def validate(ex):
        outs = list(ex.map(_rollout, [(wd, weights(), False, "ctrl") for wd in vworlds]))
        return float(np.mean([o["metrics"]["r_tgt"] for o in outs]))

    best = dict(score=-np.inf, net=None, episodes=-1)
    if os.path.exists(ckpt):
        best = c.get("best", best)
    with pool(a, mem, seed, charging, b) as ex:
        if a.warm_start == "greedy" and not os.path.exists(ckpt):
            warm_start(net, a, ex, seed, dev, tag)
            score = validate(ex)
            best = dict(score=score, net=weights(), episodes=0)
            L.append(log, dict(arm=mem, charging=charging, seed=seed, episodes=0, val_r_tgt=score, best=True))
            print(f"train {tag}: warm start validation r_tgt {score:.3f}", flush=True)
            torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), episodes=0, best=best), ckpt)
        while done < a.train_episodes:
            t0 = time.time()
            n = min(a.episodes_per_iter, a.train_episodes - done)
            worlds = [TRAIN_WORLD_BASE + seed * 100_000 + done + i for i in range(n)]
            eps = list(ex.map(_rollout, [(wd, weights(), True, "ctrl") for wd in worlds]))
            rows = [r for ep in eps for r in advantages(ep, a)]
            net.train()
            vl, hd_, hc = ppo_update(net, opt, rows, a, dev) if rows else (np.nan,) * 3
            net.eval()
            prev, done = done, done + n
            m = pd.DataFrame([ep["metrics"] for ep in eps]).mean(numeric_only=True)
            row = dict(arm=mem, charging=charging, seed=seed, episodes=done, decisions=len(rows), value_loss=vl,
                       ent_dest=hd_, ent_charge=hc, seconds=time.time() - t0,
                       **{k: m[k] for k in ("r_tgt", "r_chg", "peak_aoi", "deaths", "diversions", "charge_min")})
            note = ""
            if done // a.val_every > prev // a.val_every or done >= a.train_episodes:
                score = validate(ex)
                row.update(val_r_tgt=score, best=score > best["score"])
                if score > best["score"]:
                    best = dict(score=score, net=weights(), episodes=done)
                note = f"  validation {score:.3f} (best {best['score']:.3f} at {best['episodes']})"
            L.append(log, row)
            print(f"train {tag}: {done}/{a.train_episodes} episodes  r_tgt {m['r_tgt']:.3f}  r_chg {m['r_chg']:.3f}  "
                  f"peak AoI {m['peak_aoi']:.0f}  deaths {m['deaths']:.2f}  charge {m['charge_min']:.1f} min  "
                  f"({time.time() - t0:.0f}s){note}", flush=True)
            torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), episodes=done, best=best), ckpt)
    keep = best["net"] if best["net"] is not None else weights()
    torch.save(dict(net={k: torch.as_tensor(v) for k, v in keep.items()}, mem=mem, charging=charging, seed=seed,
                    hidden=a.ctrl_hidden, best_episodes=best["episodes"], val_r_tgt=best["score"]), final)
    print(f"train {tag}: kept the policy from episode {best['episodes']} (validation r_tgt {best['score']:.3f})",
          flush=True)
    os.remove(ckpt)


def force_clear(a):
    cdir = os.path.join(a.ctrl_out, "ctrl")
    tags = {(m, c) for m in a.memories for c in a.charging}
    if os.path.isdir(cdir):
        for f in os.listdir(cdir):
            parts = f.replace("ckpt_", "").rsplit(".pt", 1)[0].split("__")
            if len(parts) == 3 and (parts[0], parts[1]) in tags:
                os.remove(os.path.join(cdir, f))
                print(f"--force: removed ctrl/{f}", flush=True)
    for f in ("ctrl_eval.csv", "ctrl_train.csv"):
        p = os.path.join(a.ctrl_out, f)
        if os.path.exists(p):
            d = pd.read_csv(p)
            drop = np.array([(m, c) in tags for m, c in zip(d.arm, d.charging)])
            if drop.any():
                shutil.copy(p, p + ".bak")
                d[~drop].to_csv(p, index=False)
                print(f"--force: removed {int(drop.sum())} rows from {f} (backup {f}.bak)", flush=True)


def cmd_train(a, b):
    V.resolve_device(b)
    os.makedirs(a.ctrl_out, exist_ok=True)
    V.check_config(os.path.join(a.ctrl_out, "ctrl_config.json"), a, CTRL_KEYS, "controller results")
    for seed in range(a.seeds):
        for mem in a.memories:
            for charging in a.charging:
                train_one(a, b, mem, charging, seed)


# ==============================================================================================
# Evaluation and report
# ==============================================================================================
def cmd_eval(a, b):
    path = os.path.join(a.ctrl_out, "ctrl_eval.csv")
    have = set()
    if os.path.exists(path):
        d = pd.read_csv(path)
        have = set(zip(d.arm, d.charging, d.seed, d.world))
    worlds = [EVAL_WORLD_BASE + i for i in range(a.eval_worlds)]
    jobs = [(k, "dynamic", -1) for k in a.baselines]
    jobs += [(m, c, s) for s in range(a.seeds) for m in a.memories for c in a.charging]
    for mem, charging, seed in jobs:
        todo = [w for w in worlds if (mem, charging, seed, w) not in have]
        if not todo:
            continue
        if mem in BASELINES:
            with pool(a, None, 0, charging, b) as ex:
                outs = list(ex.map(_rollout, [(w, None, False, mem) for w in todo]))
        else:
            f = os.path.join(a.ctrl_out, "ctrl", arm_tag(mem, charging, seed) + ".pt")
            if not os.path.exists(f):
                print(f"eval: no trained controller {arm_tag(mem, charging, seed)}; skipped", flush=True)
                continue
            w = {k: v.cpu().numpy() for k, v in torch.load(f, weights_only=False)["net"].items()}
            with pool(a, mem, seed, charging, b) as ex:
                outs = list(ex.map(_rollout, [(wd, w, False, "ctrl") for wd in todo]))
        for o in outs:
            L.append(path, dict(arm=mem, charging=charging, seed=seed, world=o["world"], **o["metrics"]))
        m = pd.DataFrame([o["metrics"] for o in outs]).mean(numeric_only=True)
        print(f"eval {mem} {charging} seed {seed}: r_tgt {m['r_tgt']:.3f}  peak AoI {m['peak_aoi']:.0f}  "
              f"deaths {m['deaths']:.2f}  UGV-out {m['ugv_out_min']:.0f} min  on drained UGVs {m['ugv_drain_min']:.0f} min",
              flush=True)


def cmd_report(a, b):
    d = pd.read_csv(os.path.join(a.ctrl_out, "ctrl_eval.csv"))
    rows = []
    for (arm, ch), g in d.groupby(["arm", "charging"], sort=False):
        per = g.groupby("seed")[METRICS].mean() if g.seed.nunique() > 1 else g[METRICS]   # baselines: over worlds
        r = dict(arm=arm, charging=ch, n_seeds=g.seed.nunique(), n_worlds=g.world.nunique())
        for k in METRICS:
            x = per[k].dropna()
            r[k], r[k + "_ci95"] = (x.mean(), M.ci95(x) if len(x) > 1 else float("nan")) if len(x) else (np.nan, np.nan)
        rows.append(r)
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(a.ctrl_out, "ctrl_summary.csv"), index=False)
    paired = []
    ctrl = d[~d.arm.isin(BASELINES)]
    greedy = d[d.arm == "greedy"].set_index("world")
    for (arm, ch), g in ctrl.groupby(["arm", "charging"], sort=False):
        refs = [("greedy", greedy)]
        lstm = ctrl[(ctrl.arm == "lstm") & (ctrl.charging == ch)].set_index(["seed", "world"])
        if arm != "lstm" and len(lstm):
            refs.append(("lstm", lstm))
        for name, ref in refs:
            for k in METRICS[:10]:
                diffs = []
                for s, gs in g.groupby("seed"):
                    gs = gs.set_index("world")
                    rv = ref[k] if name == "greedy" else ref.loc[s][k] if s in ref.index.get_level_values(0) else None
                    if rv is None:
                        continue
                    common = gs.index.intersection(rv.index)
                    if len(common):
                        diffs.append(float((gs.loc[common, k] - rv.loc[common]).mean()))
                if diffs:
                    paired.append(dict(arm=arm, charging=ch, vs=name, metric=k, diff=np.mean(diffs),
                                       diff_ci95=M.ci95(np.array(diffs)) if len(diffs) > 1 else np.nan, seeds=len(diffs)))
    pd.DataFrame(paired).to_csv(os.path.join(a.ctrl_out, "ctrl_paired.csv"), index=False)
    pd.set_option("display.width", 250)
    print(summ[["arm", "charging", "n_seeds", "r_tgt", "r_tgt_ci95", "r_chg", "peak_aoi", "avg_aoi", "deaths",
                "ugv_out_min", "ugv_drain_min", "diversions", "charge_min", "ms_decision"]].round(3).to_string(index=False))


def main(argv=None):
    a, b = parse(argv)
    if a.force:
        force_clear(a)
    if a.command in ("fit", "all"):
        cmd_fit(a, b)
    if a.command in ("train", "all"):
        cmd_train(a, b)
    if a.command in ("eval", "all"):
        V.resolve_device(b)
        cmd_eval(a, b)
    if a.command in ("report", "all"):
        cmd_report(a, b)


if __name__ == "__main__":
    main()
