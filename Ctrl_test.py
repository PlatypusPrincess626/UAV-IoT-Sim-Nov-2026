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

Charging modes (--charging)
  duration        charge here for 0 / 5 / 10 / 15 / 20 min before leaving (the paper's power-DDQN actions)
  binary          leave now, or charge before leaving (until 80 % or 20 min, the simulator's charge exit)

Decision (one per visit, when the UAV is at a charger and has no sensor sortie left there)
  For every charger c (UGV cluster heads and the base station) the controller builds
      [z_c (k_token), z_map_c (k_map), h_c (memory state, standardised), is_base,
       flight hours here -> c, battery after that leg / 100, other UAVs already heading to c / U]
  The base station has no hidden state and is not compressed, so its z, z_map and h are zeros. The UAV's own
  battery, the leg lengths and the other UAVs' targets are not in the world model (it describes UGVs), so they
  are given to the controller directly. Context: [UAV battery / 100, time of day, sun elevation].
  A shared encoder embeds every charger (so the number of chargers can change per world); the destination is
  a softmax over chargers (the current one masked); the charge choice is conditioned on the chosen
  destination and the current charger. h_c comes from M after it has consumed the UGV's tokens and docking up
  to the previous 5-minute step (World Models: h_t summarises z_<t, a_<t); z_c is the latest token.

Safety shield (on): the simulator's reserve check and charger diversion stay active underneath the controller
  (a UAV whose next leg would breach the reserve tops up first; a UAV whose UGV cannot supply diverts to the best
  reachable charger). The controller's choice is what PPO is trained on; the shield is part of the environment.
  Sensor sorties (cached nearest-neighbour + 2-opt tour) are unchanged.

Rewards (the paper's equations, fleet view): every minute
      R_peak = max(1 - max_c peak AoI_c / 120, 0)   R_avg = max(1 - mean_c avg AoI_c / t, 0)   (Eqs. 18, 19)
  with each charger's AoI the fleet's freshest (minimum over UAVs), and per UAV R_chg from the simulator
  (Eq. 20). Destination head: 0.7 R_peak + 0.3 R_avg (Eq. 23); charge head: 0.25 R_peak + 0.25 R_avg + 0.5 R_chg
  (Eq. 22). A dead UAV earns 0 from then on (Eq. 21). Each head has its own value and advantage (semi-MDP
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
  ctrl/<arm>__<charging>__s<seed>.pt                    trained controller (ckpt_*.pt while training)
  ctrl_train.csv      learning curve: one row per PPO iteration
  ctrl_eval.csv       one row per (arm, charging, seed, world); the eval step resumes from it
  ctrl_summary.csv    per arm: mean and 95 % CI over seeds (baselines: over worlds)
  ctrl_paired.csv     paired gaps per seed vs the greedy baseline and vs the lstm arm with the same charging

Run
  python Ctrl_test.py all --out vae_test_out --workers 16              # fit, train, eval, report
  python Ctrl_test.py all --out vae_test_out --seeds 3 --memories none lstm lren@0.97g100   # a first pass
  python Ctrl_test.py train --memories lren@0.97g100 --charging binary --force   # retrain those controllers
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
TRAIN_WORLD_BASE, EVAL_WORLD_BASE = 200_000, 900_000       # VAE_test data uses seed_base (1000) + episode
AOI_SAT = 120.0                                            # Eq. (19): AoI_sat
METRICS = ["r_tgt", "r_chg", "peak_aoi", "avg_aoi", "data_mb", "deaths", "ugv_out_min", "sensor_out", "diversions",
           "decisions", "charge_min", "ms_decision", "ms_wm_step"]
CTRL_KEYS = ("k_token", "k_map", "linear_map", "credit", "gamma", "lam", "clip", "ppo_epochs", "minibatch", "lr", "ent", "vcoef",
             "ctrl_hidden", "train_episodes", "episodes_per_iter")
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
    ap.add_argument("--charging", nargs="+", default=["duration", "binary"], choices=list(CHARGE_SETS))
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
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent", type=float, default=0.01)
    p.add_argument("--vcoef", type=float, default=0.5)
    p.add_argument("--ctrl-hidden", type=int, default=64)
    p.add_argument("--eval-worlds", type=int, default=30)
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
        for k, v in json.load(open(p)).items():
            setattr(b, k, tuple(v) if isinstance(v, list) and k not in ("policy_cycle",) else v)
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
    vpath = os.path.join(mdir, "vision.pkl")
    if not todo and os.path.exists(vpath):
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
    for mem, seed in todo:
        t0 = time.time()
        m = L.train(mem, seed, D, a, b)
        dev = next(m.parameters()).device
        with torch.no_grad():                               # statistics of the state the controller will read
            X = torch.tensor(D["train"]["inp"])
            H = np.concatenate([m(X[i:i + 64].to(dev))[0].flatten(0, 1).cpu().numpy() for i in range(0, len(X), 64)])
        cert = m.certificate() if hasattr(m, "certificate") else (float("nan"),) * 3
        torch.save(dict(arm=mem, seed=seed, state={k: v.cpu() for k, v in m.state_dict().items()}, din=din, dz=dz, R=R,
                        args=margs, h_mu=H.mean(0), h_sd=H.std(0) + 1e-6, val_nll=m.val_nll, epochs=m.epochs,
                        cert=dict(rho=cert[0], L=cert[1], **getattr(m, "extra", {}))),
                   os.path.join(mdir, f"mem_{mem}_s{seed}.pt"))
        print(f"fit: {mem} seed {seed}: {m.epochs} epochs, val NLL {m.val_nll:.3f}, certificate "
              f"rho={cert[0]:.3g} L={cert[1]:.4g}  ({time.time() - t0:.0f}s)", flush=True)


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
    """Builds each UGV's token every --token-every minutes exactly as VAE_test.ch_tokens does from a finished
    episode, encodes it with the frozen V, and steps the frozen M once per token with the previous token and the
    docking during the previous 5 minutes (the input M was trained on)."""

    def __init__(s, vis, mem=None, mem_meta=None):
        s.v, s.mem = vis, mem
        s.hd = 0 if mem is None else len(mem_meta["h_mu"])
        if mem is not None:
            s.h_mu, s.h_sd = mem_meta["h_mu"].astype(np.float32), mem_meta["h_sd"].astype(np.float32)
        s.lags = np.arange(vis["history"]) * vis["history_stride"]

    def start(s, env, chs, uavs, doy, T, cluster_size, charger_m, n_feat):
        v = s.v
        s.T, s.U = T, len(uavs)
        s.ugv = np.array([c for c, ch in enumerate(chs) if not getattr(ch, "is_base", False)])
        s.Ku = len(s.ugv)
        s.xy = np.array([(chs[c].id_x, chs[c].id_y) for c in s.ugv], float)
        s.dim = float(env.dim)
        s.sun = np.clip(90.0 - env.sun_zenith, 0.0, 90.0) / 90.0
        s.season = (math.sin(2 * math.pi * doy / 365), math.cos(2 * math.pi * doy / 365))
        s.size = cluster_size[s.ugv] / 10.0
        s.to_base = charger_m[s.ugv] / s.dim
        P = V.node_patches(env, s.xy).astype(np.float64)
        mmu, msd = v["map_stats"]
        s.zm = ((v["mapc"].enc((P - mmu[0]) / msd[0]) - v["zm_mu"]) / v["zm_sd"]).astype(np.float32)
        s.obs = np.zeros((T, len(chs), n_feat), np.float32)
        s.z = np.zeros((s.Ku, v["k_token"]), np.float32)
        s.z_prev = None
        s.h = np.zeros((s.Ku, s.hd), np.float32)
        if s.mem is not None:
            s.h = (0.0 - s.h_mu)[None].repeat(s.Ku, 0) / s.h_sd     # standardised zero state
        s.state = None
        s.dock = np.zeros(s.Ku)
        s.wm_time, s.wm_steps = 0.0, 0

    def token(s, t, uav_xy, alive):
        v = s.v
        frames = np.clip(t - s.lags, 0, s.T - 1)
        hist = s.obs[frames][:, s.ugv].transpose(1, 0, 2).reshape(s.Ku, -1)
        tm = np.repeat(np.array([[t / s.T, s.season[0], s.season[1], s.sun[t]]], np.float32), s.Ku, 0)
        d = np.hypot(*(uav_xy[None, :, :] - s.xy[:, None, :].astype(np.float32)).transpose(2, 0, 1)) / np.float32(s.dim)
        d = np.where(alive[None, :], d, 2.0).min(1)
        X = np.concatenate([hist, tm, s.size[:, None].astype(np.float32), d[:, None].astype(np.float32),
                            s.to_base[:, None].astype(np.float32)], 1).astype(np.float64)
        mu, sd = v["ch_stats"]
        X = np.clip((X - mu) / sd, -v["z_clip"], v["z_clip"])
        return ((v["tok"].enc(X) - v["zmu"]) / v["zsd"]).astype(np.float32)

    def minute(s, t, obs_t, uav_xy, alive):
        """End of minute t: record the fleet's observation; on token minutes update h, then z."""
        s.obs[t] = obs_t
        te = s.v["token_every"]
        if t % te == 0:
            t0 = time.perf_counter()
            if s.z_prev is not None and s.mem is not None:
                x = np.concatenate([s.z_prev, s.zm, (s.dock / te)[:, None]], 1).astype(np.float32)
                with torch.no_grad():
                    out = s.mem(torch.from_numpy(x)[:, None, :], s.state)
                s.state = out[1]
                s.h = ((out[0][:, -1].numpy() - s.h_mu) / s.h_sd).astype(np.float32)
            s.dock[:] = 0.0
            s.z = s.token(t, uav_xy, alive)
            s.z_prev = s.z
            s.wm_time += time.perf_counter() - t0
            s.wm_steps += 1
        dd = np.hypot(*(uav_xy[None, :, :] - s.xy[:, None, :]).transpose(2, 0, 1))
        s.dock += ((dd < s.v["dock_m"]) & alive[None, :]).sum(1)


# ==============================================================================================
# Controller
# ==============================================================================================
class Controller(nn.Module):
    """Shared per-charger encoder; destination = softmax over chargers; charge | destination; two values."""

    def __init__(s, dc, nC, H=64):
        super().__init__()
        s.enc = nn.Sequential(nn.Linear(dc, 2 * H), nn.Tanh(), nn.Linear(2 * H, H), nn.Tanh())
        s.ctx = nn.Sequential(nn.Linear(3 + 2 * H, H), nn.Tanh())
        s.dest = nn.Sequential(nn.Linear(2 * H, H), nn.Tanh(), nn.Linear(H, 1))
        s.chg = nn.Sequential(nn.Linear(3 * H, H), nn.Tanh(), nn.Linear(H, nC))
        s.val = nn.Sequential(nn.Linear(H, H), nn.Tanh(), nn.Linear(H, 2))

    def forward(s, F, mask, ctx, here):
        """F (B, K, dc), mask (B, K) True = allowed, ctx (B, 3), here (B,). Returns dest logits, q, embeddings."""
        e = s.enc(F)
        valid = (mask | (torch.arange(F.shape[1], device=F.device)[None] == here[:, None])).float()   # not padding
        pool = (e * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp(min=1.0)
        eh = e[torch.arange(len(F)), here]
        q = s.ctx(torch.cat([ctx, eh, pool], -1))
        logit = s.dest(torch.cat([e, q[:, None].expand_as(e)], -1)).squeeze(-1)
        logit = logit.masked_fill(~mask, -1e9)
        return logit, q, e, eh

    def charge_logits(s, q, e, eh, d):
        return s.chg(torch.cat([q, e[torch.arange(len(d)), d], eh], -1))

    def value(s, q):
        return s.val(q)


class CtrlAgent:
    """VisitPolicy-compatible: the simulator calls act() while a UAV sits at a charger with nothing to sweep.
    The first call of a visit decides (destination, charge); later calls in that visit return the commitment."""

    def __init__(s, net, wm, charges, sample, rng):
        s.net, s.wm, s.charges, s.sample, s.rng = net, wm, charges, sample, rng
        s.uav, s.commit, s.records, s.dec_time = None, {}, [], 0.0

    def start(s, chs, uavs, is_base, t_ref):
        s.chs, s.uavs, s.is_base, s.t_ref = chs, uavs, is_base, t_ref
        s.K = len(chs)
        s.xy = np.array([(c.id_x, c.id_y) for c in chs], float)
        s.row = {int(c): r for r, c in enumerate(s.wm.ugv)}

    def features(s, u, here):
        from sim_mods import ablation
        from sim_mods.ablation import CFG
        wm = s.wm
        dz, dm, hd = wm.z.shape[1], wm.zm.shape[1], wm.hd
        F = np.zeros((s.K, dz + dm + hd + 4), np.float32)
        soc = float(u.state[0][2])
        claimed = np.zeros(s.K)
        for o in s.uavs:
            if o is not u and not o.crash and o.targetHead is not None:
                claimed[o.targetHead.cluster_num] += 1
        for c in range(s.K):
            if c in s.row:
                r = s.row[c]
                F[c, :dz], F[c, dz:dz + dm], F[c, dz + dm:dz + dm + hd] = wm.z[r], wm.zm[r], wm.h[r]
            else:
                F[c, dz + dm + hd] = 1.0                            # base station
            d = math.hypot(s.xy[c, 0] - u.id_x, s.xy[c, 1] - u.id_y)
            mins = ablation.leg_minutes(d)
            F[c, -3] = mins / 60.0
            F[c, -2] = (soc - CFG.tour_margin * ablation.minutes_pct(mins)) / 100.0
            F[c, -1] = claimed[c] / max(len(s.uavs), 1)
        F[here, -3:] = 0.0
        F[here, -2] = soc / 100.0
        mask = np.ones(s.K, bool)
        mask[here] = False
        t = s.t_ref[0]
        ctx = np.array([soc / 100.0, t / wm.T, wm.sun[min(t, wm.T - 1)]], np.float32)
        return F, mask, ctx

    def act(s, state):
        u = s.uav
        here = u.target.cluster_num
        com = s.commit.get(id(u))
        if com is not None and com["here"] is u.target:
            return com["dest"]
        t0 = time.perf_counter()
        F, mask, ctx = s.features(u, here)
        if not mask.any():
            return here
        with torch.no_grad():
            Ft, mt, ct, ht = (torch.from_numpy(F)[None], torch.from_numpy(mask)[None], torch.from_numpy(ctx)[None],
                              torch.tensor([here]))
            logit, q, e, eh = s.net(Ft, mt, ct, ht)
            pd_ = torch.distributions.Categorical(logits=logit[0])
            d = int(pd_.sample()) if s.sample else int(logit[0].argmax())
            cl = s.net.charge_logits(q, e, eh, torch.tensor([d]))[0]
            pc = torch.distributions.Categorical(logits=cl)
            c = int(pc.sample()) if s.sample else int(cl.argmax())
            v = s.net.value(q)[0].numpy()
            lpd, lpc = float(pd_.log_prob(torch.tensor(d))), float(pc.log_prob(torch.tensor(c)))
        s.dec_time += time.perf_counter() - t0
        minutes = s.charges[c]
        if minutes > 0:
            u.is_charging = True
            u.p_count = max(u.p_count, minutes + 1)     # set_dest takes one off before the exit check
        s.commit[id(u)] = dict(here=u.target, dest=d)
        s.records.append(dict(F=F, mask=mask, ctx=ctx, here=here, d=d, c=c, lpd=lpd, lpc=lpc, v=v, t=s.t_ref[0],
                              uav=s.uavs.index(u), minutes=minutes))
        return d


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
    if wm is not None:
        wm.start(env, chs, uavs, info["doy"], T, cluster_size, charger_m, len(V.CH_FEATS))
    if isinstance(agent, CtrlAgent):
        agent.start(chs, uavs, is_base, t_ref)
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
        if wm is not None:
            wm.minute(t, obs, np.array([(u.id_x, u.id_y) for u in uavs], np.float32), ~dead)
    n_dec = len(agent.records) if isinstance(agent, CtrlAgent) else 0
    met = dict(r_tgt=r_tgt.mean(), r_chg=r_chg.mean(), peak_aoi=peak_t.mean(), avg_aoi=avg_t.mean(),
               data_mb=sum(u.state[0][1] for u in uavs) / 1e6, deaths=int(dead.sum()),
               ugv_out_min=float(ugv_out), sensor_out=sens_out / (S * T), diversions=diversions, decisions=n_dec,
               charge_min=float(np.mean([r["minutes"] for r in agent.records])) if n_dec else float("nan"),
               ms_decision=1e3 * agent.dec_time / max(n_dec, 1) if n_dec else float("nan"),
               ms_wm_step=1e3 * wm.wm_time / max(wm.wm_steps, 1) if wm is not None and wm.mem is not None else float("nan"),
               n_chargers=K, n_ugv=int((~is_base).sum()))
    return dict(r_tgt=r_tgt, r_chg=r_chg, d_tgt=d_tgt, d_chg=d_aoi, died_at=died_at, records=getattr(agent, "records", []), metrics=met)


# ==============================================================================================
# Worker processes
# ==============================================================================================
_W = {}


def _init_worker(ctrl_out, mem, seed, charging, b, hidden):
    torch.set_num_threads(1)
    _W.update(b=b, charges=CHARGE_SETS.get(charging))
    if mem is None:
        return
    vis = pickle.load(open(os.path.join(ctrl_out, "models", "vision.pkl"), "rb"))
    m, meta = (None, None) if mem == "none" else load_memory(os.path.join(ctrl_out, "models", f"mem_{mem}_s{seed}.pt"))
    _W["wm"] = OnlineWM(vis, m, meta)
    dc = vis["k_token"] + vis["k_map"] + _W["wm"].hd + 4
    _W["net"] = Controller(dc, len(_W["charges"]), hidden)
    _W["net"].eval()


def _rollout(job):
    world, weights, sample, kind = job
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):            # the simulator prints
        if kind == "ctrl":
            _W["net"].load_state_dict({k: torch.from_numpy(v) for k, v in weights.items()})
            agent = CtrlAgent(_W["net"], _W["wm"], _W["charges"], sample, np.random.default_rng(world))
            torch.manual_seed(world)
            out = run_world(world, _W["b"], agent, _W["wm"])
        else:
            out = run_world(world, _W["b"], BaselineAgent(kind))
    out["world"] = world
    return out


def pool(a, mem, seed, charging, b):
    import multiprocessing as mp
    return ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn"), initializer=_init_worker,
                               initargs=(a.ctrl_out, mem, seed, charging, b, a.ctrl_hidden))


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
    return dict(F=torch.tensor(F, device=dev), mask=torch.tensor(Mk, device=dev), ctx=st("ctx"), here=st("here", np.int64),
                d=st("d", np.int64), c=st("c", np.int64), lpd=st("lpd"), lpc=st("lpc"), adv=st("adv"), ret=st("ret"))


def ppo_update(net, opt, rows, a, dev):
    B = batch_tensors(rows, dev)
    adv = B["adv"]
    adv = (adv - adv.mean(0)) / (adv.std(0) + 1e-8)
    n = len(rows)
    stats = []
    for _ in range(a.ppo_epochs):
        for idx in torch.randperm(n, device=dev).split(a.minibatch):
            logit, q, e, eh = net(B["F"][idx], B["mask"][idx], B["ctx"][idx], B["here"][idx])
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
    hd = 0 if mem == "none" else len(torch.load(os.path.join(a.ctrl_out, "models", f"mem_{mem}_s{seed}.pt"),
                                                weights_only=False)["h_mu"])
    net = Controller(vis["k_token"] + vis["k_map"] + hd + 4, len(CHARGE_SETS[charging]), a.ctrl_hidden).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    done = 0
    if os.path.exists(ckpt):
        c = torch.load(ckpt, map_location=dev, weights_only=False)
        net.load_state_dict(c["net"])
        opt.load_state_dict(c["opt"])
        done = c["episodes"]
        print(f"train {tag}: resuming after {done} episodes", flush=True)
    log = os.path.join(a.ctrl_out, "ctrl_train.csv")
    with pool(a, mem, seed, charging, b) as ex:
        while done < a.train_episodes:
            t0 = time.time()
            n = min(a.episodes_per_iter, a.train_episodes - done)
            w = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
            worlds = [TRAIN_WORLD_BASE + seed * 100_000 + done + i for i in range(n)]
            eps = list(ex.map(_rollout, [(wd, w, True, "ctrl") for wd in worlds]))
            rows = [r for ep in eps for r in advantages(ep, a)]
            net.train()
            vl, hd_, hc = ppo_update(net, opt, rows, a, dev) if rows else (np.nan,) * 3
            net.eval()
            done += n
            m = pd.DataFrame([ep["metrics"] for ep in eps]).mean(numeric_only=True)
            row = dict(arm=mem, charging=charging, seed=seed, episodes=done, decisions=len(rows), value_loss=vl,
                       ent_dest=hd_, ent_charge=hc, seconds=time.time() - t0,
                       **{k: m[k] for k in ("r_tgt", "r_chg", "peak_aoi", "deaths", "diversions", "charge_min")})
            L.append(log, row)
            print(f"train {tag}: {done}/{a.train_episodes} episodes  r_tgt {m['r_tgt']:.3f}  r_chg {m['r_chg']:.3f}  "
                  f"peak AoI {m['peak_aoi']:.0f}  deaths {m['deaths']:.2f}  charge {m['charge_min']:.1f} min  "
                  f"({time.time() - t0:.0f}s)", flush=True)
            torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), episodes=done), ckpt)
    torch.save(dict(net={k: v.cpu() for k, v in net.state_dict().items()}, mem=mem, charging=charging, seed=seed,
                    hidden=a.ctrl_hidden), final)
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
            w = {k: v.numpy() for k, v in torch.load(f, weights_only=False)["net"].items()}
            with pool(a, mem, seed, charging, b) as ex:
                outs = list(ex.map(_rollout, [(wd, w, False, "ctrl") for wd in todo]))
        for o in outs:
            L.append(path, dict(arm=mem, charging=charging, seed=seed, world=o["world"], **o["metrics"]))
        m = pd.DataFrame([o["metrics"] for o in outs]).mean(numeric_only=True)
        print(f"eval {mem} {charging} seed {seed}: r_tgt {m['r_tgt']:.3f}  peak AoI {m['peak_aoi']:.0f}  "
              f"deaths {m['deaths']:.2f}  UGV-out {m['ugv_out_min']:.0f} min", flush=True)


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
            for k in METRICS[:9]:
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
                "ugv_out_min", "diversions", "charge_min", "ms_decision"]].round(3).to_string(index=False))


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
