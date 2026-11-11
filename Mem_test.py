"""
File: Mem_test.py
Description: World Models "M" stage on a frozen "V": does a linear observation code support the memory model
as well as the VAE?

Uses the episodes VAE_test.py collected (same --out folder, same train/val/test split by episode).

Vision (V), fitted once on the training episodes and then frozen (8-bit quantised latents, as in VAE_test.py):
  linear  UGV token -> PCA (k_token dims)  +  UGV map patch -> one linear map transform (k_map dims).
          Three map transforms: by default the top three linear methods on validation sunlight R^2 at
          k_map in <out>/summary_map.csv (VAE_test.py report), or --linear-maps.
  vae     UGV token -> VAE (k_token)  +  map patch -> VAE (k_map); VAE seed s pairs with memory seed s.

Controls (--controls, on by default), noise added only while training M; scoring always uses clean latents:
  vae_sampled         the same VAE as "vae", but M trains on z = mu + sigma * eps (fresh each epoch, inputs and
                      targets, token and map), as World Models trained it.
  pca+<map>+noise     the best linear configuration (first in the list) trained on its latents plus Gaussian
                      noise rank-matched to that seed's VAE: the VAE's per-dimension noise-to-signal ratios
                      (mean sigma / spread of mu) are sorted from smallest to largest and given to the linear
                      dimensions in order of their variance, so the most informative linear dimensions get the
                      least noise, as in the VAE.
  Read: vae_sampled - vae = cost of the stochastic encoder; pca+<map>+noise vs vae_sampled = codes compared
  under equal training noise; pca+<map> - pca+<map>+noise = what determinism itself is worth to M.

Memory (M): an MDN-RNN as in World Models. One sequence per UGV per episode, one step every --token-every
minutes (720-minute day -> 144 steps). Input at step t: token latent z_t, the UGV's map latent (static over the
day) and the action a_t (UAVs docked at this UGV during the step). Output: a per-dimension mixture of
--mixtures Gaussians over z_{t+1}. Latents are standardised with training statistics before M (an invertible
affine map: no information added or removed). Trained by teacher forcing; early stopping on validation NLL.

Scores (validation and test episodes, never seen by V or M):
  next_fvu      next-step token prediction, decoded to the 55-number token by the frozen V decoder,
                fraction of variance unexplained. Comparable across V because it is in observation space.
  floor_fvu     V's own reconstruction error of the true next token (what M could reach at best).
  added_fvu     next_fvu - floor_fvu: the error the memory model itself adds.
  latent_r2     next-step R^2 in M's own (standardised) latent space: how predictable the code is.
  persist_fvu   "nothing changes" baseline: decode(z_t) as the prediction of x_{t+1}.
  roll{H}_fvu   open-loop dream: from the true z at t0, M feeds back its own predictions for H steps
                (H x 5 min) with the true actions; error of the token at t0 + H.
  mem_*         ridge probes on [z_t, h_t] (what the World Models controller sees): hidden UGV charge R^2,
                outage within 60 min AUC, and outage onset AUC (UGVs not out yet). zonly_* = probes on z_t
                alone, so mem_* - zonly_* is what the memory adds.
  paired gaps   linear minus VAE per seed (same seed), mean and 95 % CI over seeds.

Run (CPU-sized example):
  python Mem_test.py all --out out --seeds 9 --map-max-train 10000 --map-max-epochs 40
"""
import argparse
import math
import os
import time

import numpy as np
import pandas as pd

import VAE_test as V

METRICS = ["next_fvu", "floor_fvu", "added_fvu", "persist_fvu", "latent_r2", "mem_soc_r2", "mem_out60_auc", "mem_onset_auc",
           "zonly_soc_r2", "zonly_onset_auc", "mem_gain_soc_r2", "mem_gain_onset_auc"]


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("run", "report", "all"))
    ap.add_argument("--out", default="vae_test_out", help="folder VAE_test.py collected into")
    ap.add_argument("--mem-out", default=None, help="results folder (default <out>/memory)")
    ap.add_argument("--k-token", type=int, default=8, help="token latent size (8 = 64 bits)")
    ap.add_argument("--k-map", type=int, default=32, help="map latent size (32 = 256 bits)")
    ap.add_argument("--linear-maps", nargs="+", default=None, choices=V.MAP_METHODS)
    ap.add_argument("--seeds", type=int, default=9)
    ap.add_argument("--hidden", type=int, default=256, help="LSTM units (World Models: 256)")
    ap.add_argument("--mixtures", type=int, default=5, help="Gaussians per latent dimension")
    ap.add_argument("--mem-epochs", type=int, default=60)
    ap.add_argument("--mem-patience", type=int, default=8)
    ap.add_argument("--mem-batch", type=int, default=32)
    ap.add_argument("--mem-lr", type=float, default=1e-3)
    ap.add_argument("--dock-m", type=float, default=50.0, help="a UAV this close to the UGV counts as docked")
    ap.add_argument("--rollout", type=int, nargs="+", default=[6, 12], help="dream horizons in steps")
    ap.add_argument("--map-max-train", type=int, default=None, help="cap on map patches for the map VAE")
    ap.add_argument("--map-max-epochs", type=int, default=None, help="epoch cap for the map VAE")
    ap.add_argument("--min-onset", type=int, default=500)
    ap.add_argument("--controls", action=argparse.BooleanOptionalAction, default=True,
                    help="also run vae_sampled and the best linear configuration with matched noise")
    ap.add_argument("--device", default="auto")
    a, rest = ap.parse_known_args(argv)
    b = V.parse(["run", "--out", a.out, "--device", a.device] + rest)   # VAE_test settings for V training
    a.mem_out = a.mem_out or os.path.join(a.out, "memory")
    return a, b


# ----------------------------------------------------------------------------------------------
# Sequences
# ----------------------------------------------------------------------------------------------
def episode_sequences(f, a, b):
    d = dict(np.load(f))
    ugv = ~d["is_base"] if "is_base" in d else np.ones(d["ch_obs"].shape[1], bool)
    X, Y = V.ch_tokens(d, b, None, None)                       # rows ordered step-major, then UGV
    K = int(ugv.sum())
    n = len(X) // K
    X = X.reshape(n, K, -1).transpose(1, 0, 2)
    Y = Y.reshape(n, K, 4).transpose(1, 0, 2)
    T = d["ch_obs"].shape[0]
    steps = np.arange(0, T, b.token_every)
    dxy = d["uav_xy"][:, None, :, :] - d["ch_xy"][ugv][None, :, None, :]
    docked = ((np.hypot(dxy[..., 0], dxy[..., 1]) < a.dock_m) & d["uav_alive"][:, None, :]).sum(2)   # (T, K)
    A = np.stack([docked[s:s + b.token_every].mean(0) for s in steps], 1)[..., None]                 # (K, n, 1)
    P = d["patches"][d["kinds"] == 2][ugv].astype(np.float64)
    return X, Y, A.astype(np.float64), P


def load_sequences(a, b):
    files = sorted(V.glob.glob(os.path.join(a.out, "data", "ep_*.npz")))
    parts = V.split_episodes(files, b)
    mu, sd = b._ch_stats
    mmu, msd = b._map_stats
    out = {}
    for name, fs in zip(("train", "val", "test"), parts):
        Xs, Ys, As, Ps = [], [], [], []
        for f in fs:
            X, Y, A, P = episode_sequences(f, a, b)
            Xs.append(np.clip((X - mu) / sd, -b.z_clip, b.z_clip))
            Ys.append(Y)
            As.append(A)
            Ps.append((P - mmu[0]) / msd[0])
        out[name] = dict(X=np.concatenate(Xs), Y=np.concatenate(Ys), A=np.concatenate(As), P=np.concatenate(Ps))
        Y = out[name]["Y"].reshape(-1, 4)
        onset = int((Y[Y[:, 3] == 0, 1]).sum())
        print(f"{name}: {len(out[name]['X'])} UGV-day sequences x {out[name]['X'].shape[1]} steps "
              f"from {len(fs)} episodes; outage-onset samples {onset}", flush=True)
        out[name]["onset"] = onset
    return out


# ----------------------------------------------------------------------------------------------
# Vision (frozen)
# ----------------------------------------------------------------------------------------------
class Frozen:
    """Encoder + decoder + 8-bit quantiser, fitted on training data only."""
    def __init__(self, comp, Xtr):
        self.c = comp
        self.q = V.Quantizer().fit(comp.encode(Xtr))

    def enc(self, X):
        return self.q(self.c.encode(X))

    def dec(self, Z):
        return self.c.decode(Z)


def vae_sigma(comp, X):
    """Posterior standard deviation of a fitted VAE (VAE_test.neural) for inputs X."""
    import torch
    X = torch.tensor(X.reshape(len(X), -1), dtype=torch.float32)
    with torch.no_grad():
        return np.concatenate([torch.exp(0.5 * comp.enc(X[i:i + 4096].to(comp.dev))[1]).cpu().numpy()
                               for i in range(0, len(X), 4096)]).astype(np.float64)


def noise_ratio(comp, X):
    """Per-dimension noise-to-signal ratio of a VAE: mean sigma / spread of mu."""
    return vae_sigma(comp, X).mean(0) / (comp.encode(X).std(0) + 1e-9)


def vision_pair(kind, seed, a, b, ch, mp, cache):
    """kind = 'vae' or a map method name (token side is PCA)."""
    key = (kind, seed if kind == "vae" else 0)
    if key in cache:
        return cache[key]
    (Xtr, Ytr, _), (Xva, _), _ = ch
    (Mtr, Mytr, _), (Mva, _), _ = mp
    t0 = time.time()
    if kind == "vae":
        tok = Frozen(V.neural("ch", "vae", a.k_token, seed, b).fit(Xtr, Ytr, Xva), Xtr)
        e0 = b.max_epochs
        b.max_epochs = a.map_max_epochs or e0
        mapc = Frozen(V.neural("map", "vae", a.k_map, seed, b).fit(Mtr, Mytr, Mva), Mtr)
        b.max_epochs = e0
        cache[("vae-ratio", seed)] = (noise_ratio(tok.c, Xtr), noise_ratio(mapc.c, Mtr))
    else:
        if ("pca-token",) not in cache:
            cache[("pca-token",)] = Frozen(V.PCAComp(a.k_token).fit(Xtr), Xtr)
        tok = cache[("pca-token",)]
        mapc = Frozen(V.LINEAR[("map", kind)](a.k_map, 0).fit(Mtr, Mytr), Mtr)
    print(f"vision {kind} seed {key[1]}: fitted in {time.time() - t0:.0f}s", flush=True)
    cache[key] = (tok, mapc)
    return cache[key]


# ----------------------------------------------------------------------------------------------
# Memory (MDN-RNN)
# ----------------------------------------------------------------------------------------------
def build_memory(din, dz, a):
    import torch
    import torch.nn as nn

    class MDNRNN(nn.Module):
        def __init__(s):
            super().__init__()
            s.dz, s.M = dz, a.mixtures
            s.lstm = nn.LSTM(din, a.hidden, batch_first=True)
            s.head = nn.Linear(a.hidden, 3 * dz * a.mixtures)

        def forward(s, x, state=None):
            h, st = s.lstm(x, state)
            p = s.head(h).view(*h.shape[:2], s.dz, 3, s.M)
            logpi = torch.log_softmax(p[..., 0, :], -1)
            return h, st, logpi, p[..., 1, :], p[..., 2, :].clamp(-7.0, 5.0)

        @staticmethod
        def nll(logpi, mu, logsig, z):
            lp = -0.5 * ((z.unsqueeze(-1) - mu) / logsig.exp()) ** 2 - logsig - 0.5 * math.log(2 * math.pi)
            return -torch.logsumexp(logpi + lp, -1).mean()

        @staticmethod
        def mean(logpi, mu):
            return (logpi.exp() * mu).sum(-1)

    return MDNRNN()


def to_inputs(Zt, Zm, A):
    """(N, n, dz), (N, dm), (N, n, da) -> (N, n, dz + dm + da)."""
    return np.concatenate([Zt, np.repeat(Zm[:, None, :], Zt.shape[1], 1), A], 2).astype(np.float32)


def train_memory(seed, D, a, b, noise=None):
    """noise = (token sd, map sd) in standardised latent units, broadcastable to (N, n, dz) and (N, dm)."""
    import torch
    torch.manual_seed(seed)
    dev = torch.device(b.device)
    tr, va = D["train"], D["val"]
    din, dz = tr["inp"].shape[2], tr["zs"].shape[2]
    m = build_memory(din, dz, a).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=a.mem_lr)
    Xi, Zt = torch.tensor(tr["inp"][:, :-1]), torch.tensor(tr["zs"][:, 1:].astype(np.float32))
    if noise is not None:
        Zall = torch.tensor(tr["zs"].astype(np.float32))
        tsd = torch.tensor(np.broadcast_to(noise[0], tr["zs"].shape).astype(np.float32))
        msd = torch.tensor(np.broadcast_to(noise[1], (len(tr["zs"]), tr["inp"].shape[2] - dz - 1)).astype(np.float32))
    Vi, Vz = torch.tensor(va["inp"][:, :-1]).to(dev), torch.tensor(va["zs"][:, 1:].astype(np.float32)).to(dev)
    g = torch.Generator().manual_seed(seed)
    best, best_state, bad = float("inf"), None, 0
    for ep in range(a.mem_epochs):
        m.train()
        for idx in torch.randperm(len(Xi), generator=g).split(a.mem_batch):
            x, y = Xi[idx], Zt[idx]
            if noise is not None:                       # a fresh noisy copy of each sequence, every epoch
                z = Zall[idx] + torch.randn_like(Zall[idx]) * tsd[idx]
                x = x.clone()
                x[..., :dz] = z[:, :-1]
                x[..., dz:-1] += (torch.randn_like(msd[idx]) * msd[idx])[:, None, :]
                y = z[:, 1:]
            _, _, lp, mu, ls = m(x.to(dev))
            loss = m.nll(lp, mu, ls, y.to(dev))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        m.eval()
        with torch.no_grad():
            vl = float(sum(m.nll(*m(Vi[i:i + 256])[2:], Vz[i:i + 256]) * len(Vi[i:i + 256])
                           for i in range(0, len(Vi), 256)) / len(Vi))
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= a.mem_patience:
                break
    m.load_state_dict(best_state)
    m.eval()
    m.epochs, m.val_nll = ep + 1, best
    return m


def evaluate(m, split, tok, zstats, a, b, dreams=True):
    """Observation-space prediction errors, dream rollouts, and hidden-state features for probes."""
    import torch
    dev = next(m.parameters()).device
    zmu, zsd = zstats
    inp, zs, X = split["inp"], split["zs"], split["X"]
    N, n, D = X.shape
    dec = lambda Zstd: tok.dec((Zstd * zsd + zmu).reshape(-1, len(zmu))).reshape(*Zstd.shape[:-1], D)

    def fvu(P, T):
        T2 = T.reshape(-1, D)
        return float(((P.reshape(-1, D) - T2) ** 2).sum() / ((T2 - T2.mean(0)) ** 2).sum())

    H, preds = [], []
    with torch.no_grad():
        for i in range(0, N, 128):
            h, _, lp, mu, _ = m(torch.tensor(inp[i:i + 128]).to(dev))
            H.append(h.cpu().numpy())
            preds.append(m.mean(lp, mu).cpu().numpy())
    H, pred = np.concatenate(H), np.concatenate(preds)
    res = dict(next_fvu=fvu(dec(pred[:, :-1]), X[:, 1:]), floor_fvu=fvu(dec(zs[:, 1:]), X[:, 1:]),
               persist_fvu=fvu(dec(zs[:, :-1]), X[:, 1:]))
    res["added_fvu"] = res["next_fvu"] - res["floor_fvu"]                 # error M adds on top of V
    err = ((pred[:, :-1] - zs[:, 1:]) ** 2).reshape(-1, zs.shape[2])
    res["latent_r2"] = float(1 - (err.mean(0) / zs[:, 1:].reshape(-1, zs.shape[2]).var(0)).mean())
    # dreams: from the true z at t0, feed predictions back for max(H) steps with the true actions
    hmax = max(a.rollout)
    starts = list(range(12, n - hmax, 12)) if dreams else []
    got = {h: ([], []) for h in a.rollout}
    dz = zs.shape[2]
    with torch.no_grad():
        for t0 in starts:
            x = torch.tensor(inp[:, :t0 + 1]).to(dev)
            _, st, lp, mu, _ = m(x)
            z = m.mean(lp[:, -1:], mu[:, -1:])                                 # prediction of z_{t0+1}
            for j in range(1, hmax + 1):
                if j in got:
                    got[j][0].append(z[:, 0].cpu().numpy())
                    got[j][1].append(X[:, t0 + j])
                if j == hmax:
                    break
                step = torch.tensor(inp[:, t0 + j:t0 + j + 1]).to(dev).clone()
                step[..., :dz] = z
                _, st, lp, mu, _ = m(step, st)
                z = m.mean(lp, mu)
    for h, (P, T) in (got.items() if dreams else ()):
        res[f"roll{h}_fvu"] = fvu(dec(np.concatenate(P)[:, None]), np.concatenate(T)[:, None])
    return res, H


def config_name(kind, mode):
    if kind == "vae":
        return "vae_sampled" if mode == "sampled" else "vae"
    return f"pca+{kind}" + ("+noise" if mode == "noise" else "")


def run_config(kind, seed, a, b, ch, mp, S, cache, mode="clean"):
    tok, mapc = vision_pair(kind, seed, a, b, ch, mp, cache)
    t0 = time.time()
    D, Ztr = {}, None
    for name in ("train", "val", "test"):
        s = S[name]
        N, n, _ = s["X"].shape
        Zt = tok.enc(s["X"].reshape(N * n, -1)).reshape(N, n, -1)
        Zm = mapc.enc(s["P"])
        D[name] = dict(Zt=Zt, Zm=Zm)
    zmu, zsd = D["train"]["Zt"].reshape(-1, a.k_token).mean(0), D["train"]["Zt"].reshape(-1, a.k_token).std(0) + 1e-9
    mmu, msd = D["train"]["Zm"].mean(0), D["train"]["Zm"].std(0) + 1e-9
    for name in D:
        zs = (D[name]["Zt"] - zmu) / zsd
        D[name].update(zs=zs, inp=to_inputs(zs, (D[name]["Zm"] - mmu) / msd, S[name]["A"]), X=S[name]["X"])
    noise = None
    if mode == "sampled":                                # per-sample posterior sigma, in standardised units
        N, n, _ = S["train"]["X"].shape
        noise = (vae_sigma(tok.c, S["train"]["X"].reshape(N * n, -1)).reshape(N, n, -1) / zsd,
                 vae_sigma(mapc.c, S["train"]["P"]) / msd)
    elif mode == "noise":                                # VAE ratios, rank-matched to the linear dimensions
        if ("vae-ratio", seed) not in cache:
            vision_pair("vae", seed, a, b, ch, mp, cache)
        rt, rm = cache[("vae-ratio", seed)]
        tsd, msd_ = np.zeros(a.k_token), np.zeros(a.k_map)
        tsd[np.argsort(-D["train"]["Zt"].reshape(-1, a.k_token).var(0))] = np.sort(rt)
        msd_[np.argsort(-D["train"]["Zm"].var(0))] = np.sort(rm)
        noise = (tsd, msd_)
    m = train_memory(seed, D, a, b, noise)
    row = dict(config=config_name(kind, mode), seed=seed, k_token=a.k_token, k_map=a.k_map,
               bits=8 * (a.k_token + a.k_map), epochs=m.epochs, val_nll=m.val_nll,
               noise_token=float(np.mean(noise[0])) if noise is not None else 0.0,
               noise_map=float(np.mean(noise[1])) if noise is not None else 0.0)
    feats = {}
    for name in ("train", "val", "test"):
        r, Hh = evaluate(m, D[name], tok, (zmu, zsd), a, b, dreams=name != "train")
        if name != "train":
            row.update({f"{name}_{k}": v for k, v in r.items()})
        N, n, _ = Hh.shape
        z = D[name]["zs"].reshape(N * n, -1)
        zm = np.repeat((D[name]["Zm"] - mmu) / msd, n, 0)
        feats[name] = (np.concatenate([z, zm, Hh.reshape(N * n, -1)], 1), np.concatenate([z, zm], 1),
                       S[name]["Y"].reshape(N * n, 4))
    for tag, col in (("mem", 0), ("zonly", 1)):
        pr = V.probe("ch", feats["train"][col], feats["train"][2],
                     {n_: (feats[n_][col], feats[n_][2]) for n_ in ("val", "test")}, seed, b)
        row.update({f"{k.split('_', 1)[0]}_{tag}_{k.split('_', 1)[1]}": v for k, v in pr.items()})
    for sp in ("val", "test"):
        row[f"{sp}_mem_gain_soc_r2"] = row[f"{sp}_mem_soc_r2"] - row[f"{sp}_zonly_soc_r2"]
        row[f"{sp}_mem_gain_onset_auc"] = row[f"{sp}_mem_onset_auc"] - row[f"{sp}_zonly_onset_auc"]
    row["seconds"] = round(time.time() - t0, 1)
    return row


def default_linear(a):
    p = os.path.join(a.out, "summary_map.csv")
    if os.path.exists(p):
        s = pd.read_csv(p)
        s = s[s.method.isin(V.MAP_METHODS) & (s.k == a.k_map)].sort_values("val_sun_r2", ascending=False)
        if len(s) >= 3:
            return list(s.method[:3])
    return ["dct", "pca", "db2"]


def cmd_run(a, b):
    V.resolve_device(b)
    os.makedirs(a.mem_out, exist_ok=True)
    V.check_config(os.path.join(a.mem_out, "memory_config.json"), argparse.Namespace(**{**vars(b), **vars(a)}),
                   ("k_token", "k_map", "hidden", "mixtures", "mem_epochs", "mem_patience", "mem_batch", "mem_lr",
                    "dock_m", "rollout", "map_max_train", "map_max_epochs", "max_train", "max_epochs", "patience",
                    "token_every", "history", "history_stride", "z_clip", "beta", "kl_tol"), "memory results")
    linear = a.linear_maps or default_linear(a)
    print(f"linear configurations: token PCA + map {linear}; compared with token VAE + map VAE "
          f"({a.k_token} + {a.k_map} dims = {8 * (a.k_token + a.k_map)} bits)", flush=True)
    ch = V.load_track("ch", b)
    mt = b.max_train
    b.max_train = a.map_max_train or mt
    mp = V.load_track("map", b)
    b.max_train = mt
    S = load_sequences(a, b)
    for sp in ("val", "test"):
        if S[sp]["onset"] < a.min_onset:
            print(f"WARNING: only {S[sp]['onset']} outage-onset samples in {sp} (want {a.min_onset})", flush=True)
    path = os.path.join(a.mem_out, "memory_runs.csv")
    done = set()
    if os.path.exists(path):
        old = pd.read_csv(path)
        done = set(zip(old.config, old.seed))
    cache = {}
    for seed in range(a.seeds):
        jobs = [(k, "clean") for k in linear + ["vae"]]
        if a.controls:
            jobs += [("vae", "sampled"), (linear[0], "noise")]
        for kind, mode in jobs:
            if (config_name(kind, mode), seed) in done:
                continue
            row = run_config(kind, seed, a, b, ch, mp, S, cache, mode)
            print("memory", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
            pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
        cache = {k: v for k, v in cache.items() if k[0] not in ("vae", "vae-ratio")}       # VAE seeds are not reused
    meta = dict(linear=linear, onset={sp: S[sp]["onset"] for sp in S},
                sequences={sp: int(len(S[sp]["X"])) for sp in S}, steps=int(S["train"]["X"].shape[1]))
    pd.Series(meta).to_json(os.path.join(a.mem_out, "memory_meta.json"))


def ci95(x):
    x = np.asarray(x, float)
    return float(2.306 * x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan")  # t(8) for 9 seeds


def cmd_report(a, b):
    d = pd.read_csv(os.path.join(a.mem_out, "memory_runs.csv"))
    cols = [f"{sp}_{m}" for sp in ("val", "test") for m in
            METRICS + [f"roll{h}_fvu" for h in a.rollout]] + ["epochs"]
    cols = [c for c in cols if c in d]
    rows = []
    for cfg, g in d.groupby("config", sort=False):
        r = dict(config=cfg, n_seeds=len(g))
        for c in cols:
            r[c], r[c + "_ci95"] = g[c].mean(), ci95(g[c])
        rows.append(r)
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(a.mem_out, "memory_summary.csv"), index=False)
    v = d[d.config == "vae"].set_index("seed")
    paired = []
    for cfg, g in d[d.config != "vae"].groupby("config", sort=False):
        g = g.set_index("seed")
        s = g.index.intersection(v.index)
        for c in cols:
            if c == "epochs":
                continue
            diff = g.loc[s, c] - v.loc[s, c]
            paired.append(dict(config=cfg, metric=c, linear=g.loc[s, c].mean(), vae=v.loc[s, c].mean(),
                               diff=diff.mean(), diff_ci95=ci95(diff), seeds=len(s)))
    paired = pd.DataFrame(paired)
    paired.to_csv(os.path.join(a.mem_out, "memory_paired.csv"), index=False)
    pd.set_option("display.width", 220)
    show = ["config", "n_seeds"] + [c for c in cols if c.startswith("val_")]
    print(summ[show].round(3).to_string(index=False))
    print(paired[paired.metric.str.startswith("val_")].round(4).to_string(index=False))


def main(argv=None):
    a, b = parse(argv)
    if a.command in ("run", "all"):
        cmd_run(a, b)
    if a.command in ("report", "all"):
        cmd_report(a, b)


if __name__ == "__main__":
    main()
