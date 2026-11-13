"""
File: Lyap_test.py
Description: Can the memory model (World Models "M") carry Lyapunov properties and a Lipschitz bound, and what
does that cost?

Vision is frozen and linear (token PCA + one linear map transform, 8-bit, as in Mem_test.py), so the observation
code is 1-Lipschitz by construction. Memory arms, all trained the same way (teacher forcing, NLL, early stopping):

  lstm            the Mem_test MDN-RNN, unchanged (256-unit LSTM, mixture weights depend on the state). Baseline.
  lstm_diss@r     the same LSTM made contracting (incremental input-to-state stable) with certified rate r.
                  Inputs are scaled to the box [-1, 1] (by their training maxima, then clipped). A Bonassi-style
                  bound on the gates gives a 2x2 non-negative matrix A on (|dc|_inf, |dh|_inf); its spectral
                  radius rho(A) is the certified contraction rate. Training adds a penalty on rho(A) > r and, after
                  each step, shrinks the LSTM weights (bisection on one scale factor) until rho(A) <= r.
  ren@r           a contracting Recurrent Equilibrium Network (Revay, Wang & Manchester) with a prescribed rate r,
                  by construction: free parameters are mapped to matrices that satisfy the contraction LMI, so plain
                  Adam training keeps the guarantee. Explicit form (D11 = 0), tanh neurons.
  lren@rgG        a Lipschitz-bounded REN (rate r = --lren-rate, gain G from --gammas) certified end to end: its output
                  IS the predicted mean, and one incremental dissipation inequality (an LMI satisfied for every
                  parameter value) gives contraction at rate r, input -> prediction gain <= G, initial state ->
                  prediction (cert_Lx) and input -> memory state (cert_Lh). Mixture weights and scales may depend on
                  the state: the component means are re-centred so the mixture mean is exactly the certified output.
                  Defaults planned from the ren results: rate 0.97, 256 states / 256 neurons, up to 120 epochs.
                  Its settings live in lyap_config_lren.json, so it can be added to an existing results folder.
  attn            a 2-layer L2 self-attention transformer (tied query/key, Kim, Papamakarios & Mnih 2021) over a
                  sliding window of --attn-window steps; every matrix's spectral norm is capped at --attn-c.
                  No recurrent state: memory is the window itself.

Every certified arm uses a certifiable mixture head: mixture weights are learned constants per latent dimension
(the means and scales still depend on the state), so the predicted mean is a linear map of the features with an
exactly computable spectral norm. The lstm baseline keeps the Mem_test head (state-dependent weights), which has
no simple Lipschitz bound.

Certificates (computed from the trained weights):
  cert_rho      contraction rate of the memory state (lstm: rho(A) on the training input box, usually > 1).
  cert_L        Lipschitz bound, l2 in the standardised inputs -> l2 in the predicted next latent, over whole
                sequences: if every input step moves by at most delta, every prediction moves by at most
                cert_L * delta. inf when no bound exists.
Measured (held-out validation sequences):
  emp_lip       largest response found by projected gradient ascent on the input sequence (per-step l2 ball of
                radius --eps): max_t |d pred_t| / delta. A lower bound on the true constant; must be <= cert_L.
  imp_k         impulse response: perturb only step --impulse-at (worst direction found by gradient ascent) and
                measure |d pred_{t+k}| / delta for k = 0..--impulse-len. imp_half = steps for it to halve.
  emp_rho       state contraction: two copies of the memory start from different states and get identical
                inputs; emp_rho is the per-step shrink factor of their distance, state_half the steps to halve.
lren only (bound_ok also requires both):
  cert_Lx/emp_Lx  initial state -> prediction: |dy_t| <= cert_Lx r^(t+1) |dx_0|; emp_Lx from random state pairs.
  cert_Lh/emp_Lh  input -> memory state (what a controller would read), same per-step l2 convention as cert_L.
  alpha           input scale the construction applied (1 = the gain bound did not bind).
  lmi_min         smallest eigenvalue of the certifying LMI / its largest (>= 0 up to rounding).
Plus every Mem_test metric (next_fvu, added_fvu, latent_r2, dreams, hidden-charge R^2, outage AUCs).

Run:
  python Lyap_test.py all --out vae_test_out                 (GPU, full specification)
  python Lyap_test.py all --out vae_test_out --seeds 3 --map-max-train 10000 --device cpu   (CPU-sized)
"""
import argparse
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    import Mem_test as M
except ImportError:                                   # development copy
    import memory_test as M
V = M.V

KEY = ["val_next_fvu", "val_added_fvu", "val_latent_r2", "val_roll12_fvu", "val_mem_soc_r2", "val_mem_onset_auc",
       "val_mem_out60_auc", "val_mem_gain_soc_r2", "test_mem_soc_r2", "test_mem_onset_auc", "cert_rho", "cert_L",
       "head_lip", "emp_lip", "tightness", "emp_rho", "state_half", "imp_k0", "imp_k12", "imp_k48", "imp_half", "epochs",
       "alpha", "cert_Lx", "emp_Lx", "cert_Lh", "emp_Lh", "lmi_min"]


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("run", "report", "all"))
    ap.add_argument("--out", default="vae_test_out", help="folder VAE_test.py collected into")
    ap.add_argument("--lyap-out", default=None, help="results folder (default <out>/lyapunov)")
    ap.add_argument("--k-token", type=int, default=8)
    ap.add_argument("--k-map", type=int, default=32)
    ap.add_argument("--linear-map", default=None, choices=V.MAP_METHODS,
                    help="map transform (default: best linear method in summary_map.csv)")
    ap.add_argument("--arms", nargs="+", default=["lstm", "lstm_diss", "ren", "attn", "lren"],
                    choices=["lstm", "lstm_diss", "ren", "attn", "lren"])
    ap.add_argument("--gammas", type=float, nargs="+", default=[3.0, 10.0, 30.0],
                    help="lren: certified input -> prediction gain (per-step l2 in, per-step l2 out)")
    ap.add_argument("--lren-rate", type=float, default=0.97, help="lren: certified contraction rate")
    ap.add_argument("--lren-state", type=int, default=256)
    ap.add_argument("--lren-neurons", type=int, default=256)
    ap.add_argument("--lren-head", default="mix", choices=["mix", "const"],
                    help="mix: mixture weights depend on the state (mean still certified); const: learned constants")
    ap.add_argument("--lren-epochs", type=int, default=120)
    ap.add_argument("--lren-patience", type=int, default=12)
    ap.add_argument("--rates", type=float, nargs="+", default=[0.9, 0.97, 0.99],
                    help="certified contraction rates for lstm_diss and ren (per 5-minute step)")
    ap.add_argument("--seeds", type=int, default=9)
    ap.add_argument("--hidden", type=int, default=256, help="LSTM units")
    ap.add_argument("--ren-state", type=int, default=128)
    ap.add_argument("--ren-neurons", type=int, default=128)
    ap.add_argument("--attn-window", type=int, default=24, help="steps the transformer sees (24 = 2 h)")
    ap.add_argument("--attn-dim", type=int, default=64)
    ap.add_argument("--attn-heads", type=int, default=4)
    ap.add_argument("--attn-layers", type=int, default=2)
    ap.add_argument("--attn-c", type=float, default=1.0, help="spectral-norm cap on every transformer matrix")
    ap.add_argument("--penalty", type=float, default=10.0, help="weight on rho(A) above the target (lstm_diss)")
    ap.add_argument("--mixtures", type=int, default=5)
    ap.add_argument("--mem-epochs", type=int, default=60)
    ap.add_argument("--mem-patience", type=int, default=8)
    ap.add_argument("--mem-batch", type=int, default=32)
    ap.add_argument("--mem-lr", type=float, default=1e-3)
    ap.add_argument("--dock-m", type=float, default=50.0)
    ap.add_argument("--rollout", type=int, nargs="+", default=[6, 12])
    ap.add_argument("--probe-seqs", type=int, default=64, help="validation sequences for the empirical tests")
    ap.add_argument("--eps", type=float, default=0.02, help="perturbation size per step (standardised units)")
    ap.add_argument("--pgd-steps", type=int, default=20)
    ap.add_argument("--impulse-at", type=int, default=24)
    ap.add_argument("--impulse-len", type=int, default=60)
    ap.add_argument("--map-max-train", type=int, default=None, help="cap on map patches loaded (linear fit)")
    ap.add_argument("--map-max-epochs", type=int, default=None)
    ap.add_argument("--min-onset", type=int, default=500)
    ap.add_argument("--device", default="auto")
    a, rest = ap.parse_known_args(argv)
    b = V.parse(["run", "--out", a.out, "--device", a.device] + rest)
    a.lyap_out = a.lyap_out or os.path.join(a.out, "lyapunov")
    return a, b


# ----------------------------------------------------------------------------------------------
# Shared pieces
# ----------------------------------------------------------------------------------------------
def snorm(W):
    return torch.linalg.matrix_norm(W, 2)


class Memory(nn.Module):
    """Common interface: forward(x, state) -> features, state, log mixture weights, means, log scales."""
    recurrent = True

    @staticmethod
    def nll(logpi, mu, logsig, z):
        lp = -0.5 * ((z.unsqueeze(-1) - mu) / logsig.exp()) ** 2 - logsig - 0.5 * math.log(2 * math.pi)
        return -torch.logsumexp(logpi + lp, -1).mean()

    @staticmethod
    def mean(logpi, mu):
        return (logpi.exp() * mu).sum(-1)

    def penalty(self):
        return None

    def project(self):
        pass


class MDNHead(nn.Module):
    """Mem_test head: mixture weights, means and scales all depend on the state."""
    def __init__(s, F, dz, K):
        super().__init__()
        s.dz, s.K = dz, K
        s.lin = nn.Linear(F, 3 * dz * K)

    def forward(s, f):
        p = s.lin(f).view(*f.shape[:-1], s.dz, 3, s.K)
        return torch.log_softmax(p[..., 0, :], -1), p[..., 1, :], p[..., 2, :].clamp(-7.0, 5.0)

    def lip(s):
        return float("inf")


class CertHead(nn.Module):
    """Mixture weights are learned constants, so the predicted mean is W_bar f + b_bar (exact Lipschitz)."""
    def __init__(s, F, dz, K):
        super().__init__()
        s.dz, s.K = dz, K
        s.mu, s.ls = nn.Linear(F, dz * K), nn.Linear(F, dz * K)
        s.pi = nn.Parameter(torch.zeros(dz, K))

    def forward(s, f):
        sh = f.shape[:-1]
        mu = s.mu(f).view(*sh, s.dz, s.K)
        ls = s.ls(f).view(*sh, s.dz, s.K).clamp(-7.0, 5.0)
        return torch.log_softmax(s.pi, -1).expand(*sh, s.dz, s.K), mu, ls

    def lip(s):
        W = s.mu.weight.view(s.dz, s.K, -1)
        return float(snorm((torch.softmax(s.pi, -1)[..., None] * W).sum(1)))


# ----------------------------------------------------------------------------------------------
# LSTM: incremental-stability bound (derived here, in the style of Bonassi et al.)
# ----------------------------------------------------------------------------------------------
def lstm_bound(lstm, H, in_scale=None):
    """Two runs with the same inputs: e_{t+1} <= A e_t + b |dx|_inf, e = (|dc|_inf, |dh|_inf), for inputs in the
    box |x_j| <= 1 (or |x_j| <= in_scale_j) and c_0 = 0. Gate maxima s_k = max_row sigmoid(|W|1 + |U|1 + |b|),
    |c| <= c_bar = s_i / (1 - s_f), |g| <= 1, sigmoid is 1/4-Lipschitz, tanh 1-Lipschitz:
        dc' <= s_f dc + (U_f c_bar / 4 + s_i U_c + U_i / 4) dh + (W_f c_bar / 4 + s_i W_c + W_i / 4) dx
        dh' <= s_o dc' + (U_o tanh(c_bar) / 4) dh + (W_o tanh(c_bar) / 4) dx
    with induced inf-norms of the weight blocks. rho(A) < 1 means contraction; with Perron vector v,
    m_t = max_k e_k / v_k obeys m_{t+1} <= rho m_t + beta |dx|, beta = max_k b_k / v_k, so over a whole
    sequence |dh|_inf <= v_h beta / (1 - rho) sup_t |dx_t|_inf."""
    Wih, Whh = lstm.weight_ih_l0.double(), lstm.weight_hh_l0.double()        # float64: gates near 1 matter
    bias = (lstm.bias_ih_l0 + lstm.bias_hh_l0).double().abs()
    Wa = Wih.abs() * (in_scale.double()[None, :] if in_scale is not None else 1.0)
    Ua = Whh.abs()
    blk = {k: (Wa[i * H:(i + 1) * H], Ua[i * H:(i + 1) * H], bias[i * H:(i + 1) * H]) for i, k in enumerate("ifgo")}
    ninf = lambda X: X.sum(1).max()
    sbar = lambda k: torch.sigmoid((blk[k][0].sum(1) + blk[k][1].sum(1) + blk[k][2]).max())
    sf, si, so = sbar("f"), sbar("i"), sbar("o")
    cbar = si / (1 - sf)
    tc = torch.tanh(cbar)
    a11 = sf
    a12 = 0.25 * ninf(blk["f"][1]) * cbar + si * ninf(blk["g"][1]) + 0.25 * ninf(blk["i"][1])
    a21 = so * a11
    a22 = so * a12 + 0.25 * ninf(blk["o"][1]) * tc
    tr, det = a11 + a22, a11 * a22 - a12 * a21
    rho = 0.5 * (tr + torch.sqrt(torch.clamp(tr * tr - 4 * det, min=0.0)))
    bc = 0.25 * ninf(blk["f"][0]) * cbar + si * ninf(blk["g"][0]) + 0.25 * ninf(blk["i"][0])
    bh = so * bc + 0.25 * ninf(blk["o"][0]) * tc
    v0, v1 = a12 + 1e-12, torch.clamp(rho - a11, min=1e-12)
    beta = torch.maximum(bc / v0, bh / v1)
    if not bool(torch.isfinite(rho)) or float(sf) >= 1.0:
        return torch.tensor(float("inf"), dtype=torch.float64), torch.tensor(float("inf")), cbar
    return rho, (v1 * beta / (1 - rho) if float(rho) < 1 else torch.tensor(float("inf"))), cbar


class LSTMMemory(Memory):
    def __init__(s, din, dz, a, R, rate=None):
        super().__init__()
        s.H, s.rate = a.hidden, rate
        s.lstm = nn.LSTM(din, a.hidden, batch_first=True)
        s.head = (CertHead if rate is not None else MDNHead)(a.hidden, dz, a.mixtures)
        s.register_buffer("R", torch.tensor(R, dtype=torch.float32))
        s.pen = a.penalty
        if rate is not None:
            s.project()

    def forward(s, x, state=None):
        if s.rate is not None:                       # certified arm: inputs scaled into the box and clipped
            x = (x / s.R).clamp(-1.0, 1.0)
        h, st = s.lstm(x, state)
        return (h, st) + s.head(h)

    def bound(s):
        rho, Lh, cbar = lstm_bound(s.lstm, s.H, None if s.rate is not None else s.R)
        return rho, Lh, cbar

    def penalty(s):
        if s.rate is None:
            return None
        return s.pen * torch.relu(s.bound()[0] - s.rate) ** 2

    @torch.no_grad()
    def project(s):
        if s.rate is None or float(s.bound()[0]) <= s.rate:
            return
        # Only what rho(A) depends on is shrunk: every recurrent weight, and the input weights and biases of the
        # i, f, o gates. The cell-input path (W_c, b_c) only enters the input gain, so it is left alone.
        # At scale 0 every gate sits at 1/2 and there is no recurrence, so rho = 1/2: a feasible scale exists.
        H = s.H
        mask = torch.ones(4 * H, 1, device=s.R.device)
        mask[2 * H:3 * H] = 0.0                      # PyTorch gate order i, f, g, o
        ps = [(s.lstm.weight_hh_l0, None), (s.lstm.weight_ih_l0, mask), (s.lstm.bias_ih_l0, mask[:, 0]),
              (s.lstm.bias_hh_l0, mask[:, 0])]
        orig = [p.detach().clone() for p, _ in ps]

        def put(scale):
            for (p, mk), o in zip(ps, orig):
                p.copy_(o * scale if mk is None else o * (mk * scale + (1 - mk)))
        lo, hi = 0.0, 1.0
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            put(mid)
            lo, hi = (mid, hi) if float(s.bound()[0]) <= s.rate else (lo, mid)
        put(lo)

    def certificate(s):
        with torch.no_grad():
            rho, Lh, _ = s.bound()
            if s.rate is None:                       # bound on the training box; MDN head has no bound
                return float(rho), float("inf"), s.head.lip()
            L_state = math.sqrt(s.H) * float(Lh) / float(s.R.min())      # inf-norms -> l2 in raw inputs
            return float(rho), s.head.lip() * L_state, s.head.lip()

    def random_state(s, B, dev):
        g = torch.Generator().manual_seed(1)
        r = lambda: (torch.rand(1, B, s.H, generator=g) - 0.5).to(dev)
        return (torch.tanh(r()), r())


# ----------------------------------------------------------------------------------------------
# Contracting REN (explicit, D11 = 0)
# ----------------------------------------------------------------------------------------------
class RENMemory(Memory):
    """E x+ = F x + B1 w + B2 u + bx,  w = tanh(C1 x + D12 u + bv),  features x+.
    H = X'X + eps I is split into blocks; with P = H33, F = H31, B1 = H32, Lambda = diag(row sums |H22|) / 2,
    C1 = -Lambda^-1 H21 and E = (H11 / r^2 + P + Y - Y') / 2, the matrix
        [[r^2 (E + E' - P), -C1' Lambda, F'], [-Lambda C1, 2 Lambda, B1'], [F, B1, P]] = H + diag(0, 2Lambda - H22, 0)
    is positive definite (2 Lambda - H22 is diagonally dominant). With the incremental sector condition of tanh
    this gives |dx+|_V <= r |dx|_V in the metric V = E' P^-1 E, for every parameter value. With inputs:
    |dx+|_V <= r |dx|_V + kappa |du|_2, kappa = |P^-1/2 B1| |D12| + |P^-1/2 B2|."""
    def __init__(s, din, dz, a, rate):
        super().__init__()
        n, q = a.ren_state, a.ren_neurons
        s.n, s.q, s.rate = n, q, rate
        m = 2 * n + q
        s.X = nn.Parameter(torch.randn(m, m) / math.sqrt(m))
        s.Y = nn.Parameter(torch.zeros(n, n))
        s.B2 = nn.Parameter(torch.randn(n, din) / math.sqrt(din))
        s.D12 = nn.Parameter(torch.randn(q, din) / math.sqrt(din))
        s.bx, s.bv = nn.Parameter(torch.zeros(n)), nn.Parameter(torch.zeros(q))
        s.head = CertHead(n, dz, a.mixtures)

    def mats(s):
        n, q = s.n, s.q
        Hm = s.X.T @ s.X + 1e-3 * torch.eye(s.X.shape[0], device=s.X.device)
        H11, H21, H22 = Hm[:n, :n], Hm[n:n + q, :n], Hm[n:n + q, n:n + q]
        F, B1, P = Hm[n + q:, :n], Hm[n + q:, n:n + q], Hm[n + q:, n + q:]
        lam = 0.5 * H22.abs().sum(1)
        C1 = -H21 / lam[:, None]
        E = 0.5 * (H11 / s.rate ** 2 + P + s.Y - s.Y.T)
        return E, F, B1, C1, P

    def forward(s, x, state=None):
        E, F, B1, C1, _ = s.mats()
        Ei = torch.linalg.inv(E)
        A, B1t, B2t, bxt = Ei @ F, Ei @ B1, Ei @ s.B2, Ei @ s.bx
        Bn, T, _ = x.shape
        xs = state if state is not None else x.new_zeros(Bn, s.n)
        Du, Bu = x @ s.D12.T + s.bv, x @ B2t.T + bxt
        out = []
        for t in range(T):
            w = torch.tanh(xs @ C1.T + Du[:, t])
            xs = xs @ A.T + w @ B1t.T + Bu[:, t]
            out.append(xs)
        h = torch.stack(out, 1)
        return (h, xs) + s.head(h)

    def certificate(s):
        with torch.no_grad():
            E, F, B1, C1, P = s.mats()
            ev, U = torch.linalg.eigh(P)
            Pm, Pp = U @ torch.diag(ev ** -0.5) @ U.T, U @ torch.diag(ev ** 0.5) @ U.T
            kappa = float(snorm(Pm @ B1) * snorm(s.D12) + snorm(Pm @ s.B2))
            L_state = float(snorm(torch.linalg.inv(E) @ Pp)) * kappa / (1 - s.rate)
            return s.rate, s.head.lip() * L_state, s.head.lip()

    def random_state(s, B, dev):
        g = torch.Generator().manual_seed(1)
        return torch.randn(B, s.n, generator=g).to(dev)


# ----------------------------------------------------------------------------------------------
# Lipschitz-bounded REN: contraction rate r and input -> predicted-mean gain <= gamma, head included
# ----------------------------------------------------------------------------------------------
class LRENMemory(Memory):
    """Explicit REN whose OUTPUT is the predicted mean, so the certificate runs from the inputs (and the initial
    state) all the way to the prediction, with no separate head bound multiplied on.

        E x+ = F x + B1 w + B2 u + bx,   w = tanh(C1 x + D12 u + bv),   y = C2 x + D21 w + D22 u + by

    Incremental dissipation (Delta between any two runs; V = |E dx|^2 in the metric P^-1):
        V(dx+) + |dy|^2 / nu  <=  r^2 V(dx) + mu |du|^2                                           (*)
    holds when, with blocks ordered (x, w, x+, u),
        Psi = [[Psi0, N], [N', mu I]] - G' G / nu  >= 0,   G = [C2, D21, 0, D22],   N = [0; -Lambda D12; B2]
    and Psi0 the contraction matrix of RENMemory (built from H = X'X + eps I, positive definite for every
    parameter value). For fixed nu the smallest feasible mu is a closed-form eigenvalue (Schur complement).
    Iterating (*) from equal initial states with every |du_t| <= delta gives the per-step (sup-sup) bound
        |dy_t| <= sqrt(nu mu / (1 - r^2)) delta = L* delta
    (1/sqrt(1-r^2), not the 1/(1-r) of a triangle-inequality bound). Scaling the input channel by alpha scales the
    gain by alpha exactly (a congruence of Psi), so the model feeds alpha u with alpha = min(1, gamma / L*): every
    parameter value gives a model with gain <= gamma, and plain Adam keeps the certificate.
    Also certified: initial state -> output |dy_t| <= cert_Lx r^(t+1) |dx0|, cert_Lx = sqrt(nu lambda_max(E'P^-1 E)),
    and input -> memory state |dx_t| <= alpha sqrt(mu / (1 - r^2)) / sqrt(lambda_min(E'P^-1 E)) (cert_Lh).
    Mixture head: component means are y + o_k - sum_j pi_j o_j, so the mixture mean is exactly y whatever the
    offsets o, weights pi and scales do; those may all depend on the state (as in the Mem_test head)."""

    def __init__(s, din, dz, a, rate, gamma):
        super().__init__()
        n, q = a.lren_state, a.lren_neurons
        s.n, s.q, s.din, s.dz, s.K = n, q, din, dz, a.mixtures
        s.rate, s.gamma = rate, gamma
        m = 2 * n + q
        s.X = nn.Parameter(torch.randn(m, m) / math.sqrt(m))
        s.Y = nn.Parameter(torch.zeros(n, n))
        s.B2 = nn.Parameter(0.1 * torch.randn(n, din) / math.sqrt(din))      # small: start with alpha near 1
        s.D12 = nn.Parameter(0.1 * torch.randn(q, din) / math.sqrt(din))
        s.C2, s.D21 = nn.Parameter(torch.zeros(dz, n)), nn.Parameter(torch.zeros(dz, q))
        D22 = torch.zeros(dz, din)
        D22[:, :dz] = torch.eye(dz)                     # start at persistence: predict z_{t+1} = z_t
        s.D22 = nn.Parameter(D22)
        s.bx, s.bv, s.by = nn.Parameter(torch.zeros(n)), nn.Parameter(torch.zeros(q)), nn.Parameter(torch.zeros(dz))
        s.state_head = a.lren_head == "mix"
        s.aux = nn.Linear(n, (3 if s.state_head else 2) * dz * a.mixtures)
        if not s.state_head:
            s.pi = nn.Parameter(torch.zeros(dz, a.mixtures))
        s.register_buffer("tau", torch.tensor(1.0))    # nu = nu_min (1 + 1e-4) + tau, tau re-chosen every 50 steps
        s._cache, s._steps = None, 0

    def train(s, mode=True):
        s._cache = None
        return super().train(mode)

    # -- the LMI pieces -------------------------------------------------------------------------------
    def blocks(s, dtype=None):
        n, q = s.n, s.q
        X, Y = (s.X, s.Y) if dtype is None else (s.X.to(dtype), s.Y.to(dtype))
        cv = (lambda T: T) if dtype is None else (lambda T: T.to(dtype))
        Hm = X.T @ X + 1e-3 * torch.eye(X.shape[0], device=X.device, dtype=X.dtype)
        H11, H21, H22 = Hm[:n, :n], Hm[n:n + q, :n], Hm[n:n + q, n:n + q]
        F, B1, P = Hm[n + q:, :n], Hm[n + q:, n:n + q], Hm[n + q:, n + q:]
        lam = 0.5 * H22.abs().sum(1)
        C1 = -H21 / lam[:, None]
        E = 0.5 * (H11 / s.rate ** 2 + P + Y - Y.T)
        Psi0 = Hm.clone()
        Psi0[n:n + q, n:n + q] = torch.diag(2 * lam)
        Gz = torch.cat([cv(s.C2), cv(s.D21), torch.zeros(s.dz, n, device=X.device, dtype=X.dtype)], 1)
        N = torch.cat([torch.zeros(n, s.din, device=X.device, dtype=X.dtype), -lam[:, None] * cv(s.D12), cv(s.B2)], 0)
        return dict(E=E, F=F, B1=B1, C1=C1, P=P, lam=lam, Psi0=Psi0, Gz=Gz, N=N, D22=cv(s.D22))

    @staticmethod
    def nu_min(b):
        Lc = torch.linalg.cholesky(b["Psi0"])
        Z = torch.linalg.solve_triangular(Lc, b["Gz"].T, upper=False)
        return torch.linalg.eigvalsh(Z.T @ Z)[-1]

    @staticmethod
    def mu_star(b, nu):
        S = b["Psi0"] - b["Gz"].T @ b["Gz"] / nu
        Lc = torch.linalg.cholesky(S)
        Mx = b["N"] - b["Gz"].T @ b["D22"] / nu
        Z = torch.linalg.solve_triangular(Lc, Mx, upper=False)
        return torch.linalg.eigvalsh(b["D22"].T @ b["D22"] / nu + Z.T @ Z)[-1]

    def gain(s, b, nu):
        return torch.sqrt(nu * s.mu_star(b, nu) / (1 - s.rate ** 2))

    def mats(s):
        if s._cache is not None:
            return s._cache
        b = s.blocks()
        nu = s.nu_min(b).detach() * (1 + 1e-4) + s.tau
        alpha = torch.clamp(s.gamma / s.gain(b, nu), max=1.0)
        Ei = torch.linalg.inv(b["E"])
        out = dict(A=Ei @ b["F"], B1=Ei @ b["B1"], B2=Ei @ s.B2, bx=Ei @ s.bx, C1=b["C1"], alpha=alpha)
        if not s.training:
            s._cache = {k: v.detach() for k, v in out.items()}
        return out

    @torch.no_grad()
    def project(s):
        """Not a projection (the certificate holds for every parameter value): every 50 steps re-pick nu, the free
        multiplier of the output term, so that alpha restricts the model as little as possible."""
        s._steps += 1
        if s._steps % 50 != 1:
            return
        b = s.blocks(torch.float64)
        nm = float(s.nu_min(b)) * (1 + 1e-4)
        ts = np.logspace(-6, 4, 41)
        g = [float(s.gain(b, nm + t)) for t in ts]
        s.tau.fill_(float(ts[int(np.argmin(g))]))
        s._cache = None

    # -- forward --------------------------------------------------------------------------------------
    def forward(s, x, state=None):
        k = s.mats()
        u = x * k["alpha"]
        Bn, T, _ = x.shape
        xs = state if state is not None else x.new_zeros(Bn, s.n)
        Du, Bu, Yu = u @ s.D12.T + s.bv, u @ k["B2"].T + k["bx"], u @ s.D22.T + s.by
        hs, ys = [], []
        for t in range(T):
            w = torch.tanh(xs @ k["C1"].T + Du[:, t])
            ys.append(xs @ s.C2.T + w @ s.D21.T + Yu[:, t])
            xs = xs @ k["A"].T + w @ k["B1"].T + Bu[:, t]
            hs.append(xs)
        h, y = torch.stack(hs, 1), torch.stack(ys, 1)
        sh = h.shape[:-1]
        p = s.aux(h).view(*sh, s.dz, -1, s.K)
        logpi = torch.log_softmax(p[..., 2, :], -1) if s.state_head else \
            torch.log_softmax(s.pi, -1).expand(*sh, s.dz, s.K)
        o = p[..., 0, :]
        mu = y.unsqueeze(-1) + o - (logpi.exp() * o).sum(-1, keepdim=True)    # mixture mean == y exactly
        return h, xs, logpi, mu, p[..., 1, :].clamp(-7.0, 5.0)

    # -- certificate (float64, nu optimised) ------------------------------------------------------------
    def certificate(s):
        with torch.no_grad():
            alpha = float(s.mats()["alpha"])
            b = s.blocks(torch.float64)
            nm = float(s.nu_min(b)) * (1 + 1e-6)
            ts = np.logspace(-9, 6, 151)
            gs = [float(s.gain(b, nm + t)) for t in ts]
            i = int(np.argmin(gs))
            for _ in range(40):                                  # refine around the grid minimum
                lo, hi = ts[max(i - 1, 0)], ts[min(i + 1, len(ts) - 1)]
                ts = np.linspace(lo, hi, 7)
                gs = [float(s.gain(b, nm + t)) for t in ts]
                i = int(np.argmin(gs))
            nu = nm + ts[i]
            mu = float(s.mu_star(b, nu)) * (1 + 1e-6)
            Lstar = math.sqrt(nu * mu / (1 - s.rate ** 2))
            W = b["E"].T @ torch.linalg.solve(b["P"], b["E"])     # V = dx' E' P^-1 E dx
            ev = torch.linalg.eigvalsh(0.5 * (W + W.T))
            # numerical check of the scaled LMI: lambda_min(Psi) / |Psi| should be >= 0 (up to rounding)
            m = b["Psi0"].shape[0]
            Psi = torch.zeros(m + s.din, m + s.din, dtype=torch.float64)
            Psi[:m, :m], Psi[:m, m:], Psi[m:, :m] = b["Psi0"], alpha * b["N"], alpha * b["N"].T
            Psi[m:, m:] = alpha ** 2 * mu * torch.eye(s.din, dtype=torch.float64)
            G = torch.cat([b["Gz"], alpha * b["D22"]], 1)
            Psi -= G.T @ G / nu
            pe = torch.linalg.eigvalsh(Psi)
            s.extra = dict(cert_Lx=math.sqrt(nu) * math.sqrt(float(ev[-1])),
                           cert_Lh=alpha * math.sqrt(mu / (1 - s.rate ** 2)) / math.sqrt(float(ev[0])),
                           lmi_min=float(pe[0] / pe.abs().max()), alpha=alpha, gamma=s.gamma)
            return s.rate, alpha * Lstar, float("nan")

    def random_state(s, B, dev):
        g = torch.Generator().manual_seed(1)
        return torch.randn(B, s.n, generator=g).to(dev)


# ----------------------------------------------------------------------------------------------
# L2 self-attention transformer over a sliding window
# ----------------------------------------------------------------------------------------------
def capped(W, c):
    return W / torch.clamp(snorm(W) / c, min=1.0)


class L2MHA(nn.Module):
    """F(X) = [P^h X A_h W^{V,h}]_h W^O, A_h = W^{Q,h} W^{Q,h}' / sqrt(D/H), P^h_ij ~ exp(-|x_i W^Q - x_j W^Q|^2 /
    sqrt(D/H)) (Kim et al. 2021, tied query/key). Lip_2 <= sqrt(N)/sqrt(D/H) (4 W0(N/e) + 1)
    sqrt(sum_h |W^Q,h|^2 |W^V,h|^2) |W^O|."""
    def __init__(s, D, nh, c):
        super().__init__()
        s.nh, s.dh, s.c = nh, D // nh, c
        s.WQ = nn.Parameter(torch.randn(nh, D, s.dh) / math.sqrt(D))
        s.WV = nn.Parameter(torch.randn(nh, D, s.dh) / math.sqrt(D))
        s.WO = nn.Parameter(torch.randn(D, D) / math.sqrt(D))

    def weights(s):
        return (torch.stack([capped(w, s.c) for w in s.WQ]), torch.stack([capped(w, s.c) for w in s.WV]),
                capped(s.WO, s.c))

    def forward(s, X):                                    # X (..., N, D)
        WQ, WV, WO = s.weights()
        q = torch.einsum("...nd,hde->...hne", X, WQ)
        sq = q.pow(2).sum(-1)
        dist = (sq.unsqueeze(-1) + sq.unsqueeze(-2) - 2 * q @ q.transpose(-1, -2)).clamp(min=0.0)
        P = torch.softmax(-dist / math.sqrt(s.dh), -1)
        Mh = WQ.transpose(1, 2) @ WV / math.sqrt(s.dh)       # X A_h W^V = q_h (W^Q' W^V) / sqrt(D/H)
        o = torch.einsum("...hnm,...hmd->...hnd", P, q)
        o = torch.einsum("...hnd,hde->...hne", o, Mh)
        o = o.transpose(-3, -2).reshape(*X.shape[:-1], -1)
        return o @ WO

    def lip(s, N):
        from scipy.special import lambertw
        WQ, WV, WO = s.weights()
        ss = sum(float(snorm(WQ[h])) ** 2 * float(snorm(WV[h])) ** 2 for h in range(s.nh))
        return math.sqrt(N) / math.sqrt(s.dh) * (4 * float(lambertw(N / math.e).real) + 1) * math.sqrt(ss) * float(snorm(WO))


class AttnMemory(Memory):
    recurrent = False

    def __init__(s, din, dz, a):
        super().__init__()
        D, s.W, s.c = a.attn_dim, a.attn_window, a.attn_c
        s.inp = nn.Parameter(torch.randn(din, D) / math.sqrt(din))
        s.pos = nn.Parameter(torch.zeros(s.W, D))           # constant shift: does not change the Lipschitz bound
        s.att = nn.ModuleList([L2MHA(D, a.attn_heads, s.c) for _ in range(a.attn_layers)])
        s.f1 = nn.ParameterList([nn.Parameter(torch.randn(D, 2 * D) / math.sqrt(D)) for _ in range(a.attn_layers)])
        s.f2 = nn.ParameterList([nn.Parameter(torch.randn(2 * D, D) / math.sqrt(2 * D)) for _ in range(a.attn_layers)])
        s.fb = nn.ParameterList([nn.Parameter(torch.zeros(2 * D)) for _ in range(a.attn_layers)])
        s.head = CertHead(D, dz, a.mixtures)

    def forward(s, x, state=None):
        Bn, T, din = x.shape
        prev = state if state is not None else x.new_zeros(Bn, s.W - 1, din)
        full = torch.cat([prev, x], 1)
        win = full.unfold(1, s.W, 1).permute(0, 1, 3, 2)     # (B, T, W, din): full attention inside each window
        z = win @ capped(s.inp, s.c) + s.pos
        for att, f1, f2, fb in zip(s.att, s.f1, s.f2, s.fb):
            z = z + att(z)
            z = z + torch.relu(z @ capped(f1, s.c) + fb) @ capped(f2, s.c)
        h = z[..., -1, :]                                    # the current step's output
        return (h, full[:, -(s.W - 1):]) + s.head(h)

    def certificate(s):
        with torch.no_grad():
            L = float(snorm(capped(s.inp, s.c)))
            for att, f1, f2 in zip(s.att, s.f1, s.f2):
                L *= (1 + att.lip(s.W)) * (1 + float(snorm(capped(f1, s.c)) * snorm(capped(f2, s.c))))
            return float("nan"), s.head.lip() * math.sqrt(s.W) * L, s.head.lip()   # window of W inputs


def build(arm, din, dz, a, R):
    name, rate = (arm.split("@") + [None])[:2]
    rate = float(rate) if rate and name != "lren" else None
    if name == "lstm":
        return LSTMMemory(din, dz, a, R)
    if name == "lstm_diss":
        return LSTMMemory(din, dz, a, R, rate)
    if name == "ren":
        return RENMemory(din, dz, a, rate)
    if name == "lren":                                       # lren@<rate>g<gamma>
        r, g = arm.split("@")[1].split("g")
        return LRENMemory(din, dz, a, float(r), float(g))
    return AttnMemory(din, dz, a)


# ----------------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------------
def train(arm, seed, D, a, b):
    torch.manual_seed(seed)
    dev = torch.device(b.device)
    tr, va = D["train"], D["val"]
    din, dz = tr["inp"].shape[2], tr["zs"].shape[2]
    R = np.abs(tr["inp"]).reshape(-1, din).max(0) + 1e-6
    m = build(arm, din, dz, a, R).to(dev)
    m.project()
    opt = torch.optim.Adam(m.parameters(), lr=a.mem_lr)
    epochs, patience = (a.lren_epochs, a.lren_patience) if arm.startswith("lren") else (a.mem_epochs, a.mem_patience)
    Xi, Zt = torch.tensor(tr["inp"][:, :-1]), torch.tensor(tr["zs"][:, 1:].astype(np.float32))
    Vi, Vz = torch.tensor(va["inp"][:, :-1]).to(dev), torch.tensor(va["zs"][:, 1:].astype(np.float32)).to(dev)
    g = torch.Generator().manual_seed(seed)
    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        m.train()
        for idx in torch.randperm(len(Xi), generator=g).split(a.mem_batch):
            _, _, lp, mu, ls = m(Xi[idx].to(dev))
            loss = m.nll(lp, mu, ls, Zt[idx].to(dev))
            pen = m.penalty()
            if pen is not None:
                loss = loss + pen
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            m.project()                                     # keeps the certificate after every step
        m.eval()
        with torch.no_grad():
            vl = float(sum(m.nll(*m(Vi[i:i + 64])[2:], Vz[i:i + 64]) * len(Vi[i:i + 64])
                           for i in range(0, len(Vi), 64)) / len(Vi))
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(best_state)
    m.eval()
    m.epochs, m.val_nll = ep + 1, best
    return m


# ----------------------------------------------------------------------------------------------
# Empirical tests
# ----------------------------------------------------------------------------------------------
def pred(m, x):
    _, _, lp, mu, _ = m(x)
    return m.mean(lp, mu)


def ball(d, eps):                                            # per-step l2 ball of radius eps
    n = d.norm(dim=-1, keepdim=True)
    return d * torch.clamp(eps / (n + 1e-12), max=1.0)


def emp_lipschitz(m, X, eps, steps):
    g = torch.Generator().manual_seed(0)
    with torch.backends.cudnn.flags(enabled=False):
        base = pred(m, X).detach()
        d = ball(torch.randn(X.shape, generator=g).to(X.device), eps)
        best = 0.0
        for _ in range(steps + 1):
            d = d.detach().requires_grad_(True)
            r = (pred(m, X + d) - base).norm(dim=-1).amax(1)          # per sequence, worst step
            best = max(best, float(r.detach().max()) / eps)
            if _ == steps:
                break
            gr, = torch.autograd.grad(r.sum(), d)
            d = ball(d + 0.25 * eps * gr / (gr.norm(dim=-1, keepdim=True) + 1e-12), eps)
    return best


def impulse(m, X, at, L, eps, steps=8):
    g = torch.Generator().manual_seed(0)
    T = X.shape[1]
    L = min(L, T - 1 - at)
    with torch.backends.cudnn.flags(enabled=False):
        base = pred(m, X).detach()
        d = torch.randn(X.shape[0], 1, X.shape[2], generator=g).to(X.device)
        d = eps * d / d.norm(dim=-1, keepdim=True)
        mask = torch.zeros(1, T, 1, device=X.device)
        mask[:, at] = 1.0
        for k in range(steps + 1):
            d = d.detach().requires_grad_(True)
            resp = (pred(m, X + d * mask) - base)[:, at:at + L + 1].norm(dim=-1) / eps    # (B, L+1)
            if k == steps:
                break
            gr, = torch.autograd.grad(resp.sum(), d)
            d = d + 0.25 * eps * gr / (gr.norm(dim=-1, keepdim=True) + 1e-12)
            d = eps * d / d.norm(dim=-1, keepdim=True)
    return resp.detach().mean(0).cpu().numpy()


def contraction(m, X, L=48):
    if not m.recurrent:
        return float("nan"), float("nan")
    with torch.no_grad():
        h0, _, _, _, _ = m(X[:, :L])
        h1, _, _, _, _ = m(X[:, :L], m.random_state(X.shape[0], X.device))
        gap = (h0 - h1).norm(dim=-1).mean(0).cpu().numpy()          # (L,)
    gap = np.maximum(gap, 1e-30)
    rho = float((gap[-1] / gap[0]) ** (1.0 / (len(gap) - 1)))
    below = np.where(gap <= 0.5 * gap[0])[0]
    return rho, float(below[0]) if len(below) else float("nan")


def state_tests(m, X, eps, steps, L=48):
    """lren only. emp_Lx: two runs from different initial states, same inputs:
    max_t |dy_t| / (r^(t+1) |dx0|), must be <= cert_Lx. emp_Lh: worst input perturbation (per-step l2 ball, gradient
    ascent) of the memory state: max_t |dx_t| / delta, must be <= cert_Lh."""
    with torch.no_grad():
        X = X[:, :L]
        x0 = m.random_state(X.shape[0], X.device)
        _, _, lp0, mu0, _ = m(X)
        _, _, lp1, mu1, _ = m(X, x0)
        dy = (m.mean(lp1, mu1) - m.mean(lp0, mu0)).norm(dim=-1)                     # (B, L)
        decay = m.rate ** torch.arange(1, X.shape[1] + 1, device=X.device, dtype=dy.dtype)
        emp_x = float((dy / decay / x0.norm(dim=-1, keepdim=True)).max())
    g = torch.Generator().manual_seed(0)
    base = m(X)[0].detach()
    d = ball(torch.randn(X.shape, generator=g).to(X.device), eps)
    best = 0.0
    for k in range(steps + 1):
        d = d.detach().requires_grad_(True)
        r = (m(X + d)[0] - base).norm(dim=-1).amax(1)
        best = max(best, float(r.detach().max()) / eps)
        if k == steps:
            break
        gr, = torch.autograd.grad(r.sum(), d)
        d = ball(d + 0.25 * eps * gr / (gr.norm(dim=-1, keepdim=True) + 1e-12), eps)
    return emp_x, best


# ----------------------------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------------------------
def prepare(tok, mapc, S, a):
    D = {}
    for name in ("train", "val", "test"):
        s = S[name]
        N, n, _ = s["X"].shape
        D[name] = dict(Zt=tok.enc(s["X"].reshape(N * n, -1)).reshape(N, n, -1), Zm=mapc.enc(s["P"]))
    zt = D["train"]["Zt"].reshape(-1, a.k_token)
    zmu, zsd = zt.mean(0), zt.std(0) + 1e-9
    mmu, msd = D["train"]["Zm"].mean(0), D["train"]["Zm"].std(0) + 1e-9
    for name in D:
        zs = (D[name]["Zt"] - zmu) / zsd
        D[name].update(zs=zs, inp=M.to_inputs(zs, (D[name]["Zm"] - mmu) / msd, S[name]["A"]), X=S[name]["X"],
                       zm=(D[name]["Zm"] - mmu) / msd)
    return D, (zmu, zsd)


def run_arm(arm, seed, a, b, D, zstats, tok, S):
    t0 = time.time()
    m = train(arm, seed, D, a, b)
    row = dict(arm=arm, seed=seed, epochs=m.epochs, val_nll=m.val_nll)
    feats = {}
    for name in ("train", "val", "test"):
        r, Hh = M.evaluate(m, D[name], tok, zstats, a, b, dreams=name != "train")
        if name != "train":
            row.update({f"{name}_{k}": v for k, v in r.items()})
        N, n, _ = Hh.shape
        z = D[name]["zs"].reshape(N * n, -1)
        zm = np.repeat(D[name]["zm"], n, 0)
        feats[name] = (np.concatenate([z, zm, Hh.reshape(N * n, -1)], 1), np.concatenate([z, zm], 1),
                       S[name]["Y"].reshape(N * n, 4))
    for tag, col in (("mem", 0), ("zonly", 1)):
        pr = V.probe("ch", feats["train"][col], feats["train"][2],
                     {n_: (feats[n_][col], feats[n_][2]) for n_ in ("val", "test")}, seed, b)
        row.update({f"{k.split('_', 1)[0]}_{tag}_{k.split('_', 1)[1]}": v for k, v in pr.items()})
    for sp in ("val", "test"):
        row[f"{sp}_mem_gain_soc_r2"] = row[f"{sp}_mem_soc_r2"] - row[f"{sp}_zonly_soc_r2"]
        row[f"{sp}_mem_gain_onset_auc"] = row[f"{sp}_mem_onset_auc"] - row[f"{sp}_zonly_onset_auc"]
    # certificate and empirical tests
    row["cert_rho"], row["cert_L"], row["head_lip"] = m.certificate()
    dev = next(m.parameters()).device
    sel = np.random.default_rng(seed).choice(len(D["val"]["inp"]), min(a.probe_seqs, len(D["val"]["inp"])), replace=False)
    X = torch.tensor(D["val"]["inp"][np.sort(sel)]).to(dev)
    chunks = X.split(32)
    row["emp_lip"] = max(emp_lipschitz(m, x, a.eps, a.pgd_steps) for x in chunks)
    row["tightness"] = row["emp_lip"] / row["cert_L"] if np.isfinite(row["cert_L"]) else float("nan")
    row["bound_ok"] = bool(row["emp_lip"] <= row["cert_L"] * (1 + 1e-6))
    curves = [impulse(m, x, a.impulse_at, a.impulse_len, a.eps) * len(x) for x in chunks]
    curve = sum(curves) / len(X)
    peak = int(np.argmax(curve))
    after = np.where(curve[peak:] <= 0.5 * curve[peak])[0]
    row["imp_half"] = float(after[0]) if len(after) else float("nan")
    for k in (0, 12, 48):
        row[f"imp_k{k}"] = float(curve[k]) if k < len(curve) else float("nan")
    row["emp_rho"], row["state_half"] = contraction(m, X)
    for k in ("cert_Lx", "cert_Lh", "lmi_min", "alpha", "gamma", "emp_Lx", "emp_Lh"):
        row[k] = getattr(m, "extra", {}).get(k, float("nan"))
    if isinstance(m, LRENMemory):
        row["emp_Lx"], row["emp_Lh"] = state_tests(m, X, a.eps, a.pgd_steps)
        row["bound_ok"] = bool(row["bound_ok"] and row["emp_Lx"] <= row["cert_Lx"] * (1 + 1e-6)
                               and row["emp_Lh"] <= row["cert_Lh"] * (1 + 1e-6))
    row["seconds"] = round(time.time() - t0, 1)
    return row, curve


def append(path, row):
    new = pd.DataFrame([row])
    if os.path.exists(path):
        prev = pd.read_csv(path)
        if list(prev.columns) != list(new.columns):
            pd.concat([prev, new], ignore_index=True).to_csv(path, index=False)
            return
    new.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def arm_list(a):
    arms = []
    for k in a.arms:
        if k in ("lstm_diss", "ren"):
            arms += [f"{k}@{r:g}" for r in a.rates]
        elif k == "lren":
            arms += [f"lren@{a.lren_rate:g}g{g:g}" for g in a.gammas]
        else:
            arms.append(k)
    return arms


def cmd_run(a, b):
    V.resolve_device(b)
    os.makedirs(a.lyap_out, exist_ok=True)
    V.check_config(os.path.join(a.lyap_out, "lyap_config.json"), argparse.Namespace(**{**vars(b), **vars(a)}),
                   ("k_token", "k_map", "linear_map", "hidden", "ren_state", "ren_neurons", "attn_window", "attn_dim",
                    "attn_heads", "attn_layers", "attn_c", "penalty", "mixtures", "mem_epochs", "mem_patience",
                    "mem_batch", "mem_lr", "dock_m", "eps", "pgd_steps", "impulse_at", "impulse_len", "probe_seqs",
                    "map_max_train", "token_every", "history", "history_stride", "z_clip"), "Lyapunov results")
    lin = a.linear_map or M.default_linear(a)[0]
    arms = arm_list(a)
    if any(x.startswith("lren") for x in arms):     # own file, so lren can be added to an existing results folder
        V.check_config(os.path.join(a.lyap_out, "lyap_config_lren.json"), a,
                       ("lren_state", "lren_neurons", "lren_head", "lren_epochs", "lren_patience"), "lren results")
    print(f"vision: token PCA ({a.k_token}) + map {lin} ({a.k_map}), frozen and 8-bit; memory arms: {arms}", flush=True)
    ch = V.load_track("ch", b)
    mt = b.max_train
    b.max_train = a.map_max_train or mt
    mp = V.load_track("map", b)
    b.max_train = mt
    S = M.load_sequences(a, b)
    for sp in ("val", "test"):
        if S[sp]["onset"] < a.min_onset:
            print(f"WARNING: only {S[sp]['onset']} outage-onset samples in {sp} (want {a.min_onset})", flush=True)
    tok, mapc = M.vision_pair(lin, 0, a, b, ch, mp, {})
    D, zstats = prepare(tok, mapc, S, a)
    path, ipath = os.path.join(a.lyap_out, "lyap_runs.csv"), os.path.join(a.lyap_out, "lyap_impulse.csv")
    done = set(zip(*pd.read_csv(path)[["arm", "seed"]].values.T)) if os.path.exists(path) else set()
    for seed in range(a.seeds):
        for arm in arms:
            if (arm, seed) in done:
                continue
            row, curve = run_arm(arm, seed, a, b, D, zstats, tok, S)
            row["linear_map"] = lin
            print("lyap", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
            append(path, row)
            pd.DataFrame(dict(arm=arm, seed=seed, k=np.arange(len(curve)), response=curve)).to_csv(
                ipath, mode="a", header=not os.path.exists(ipath), index=False)
    meta = dict(linear_map=lin, arms=arms, onset={sp: S[sp]["onset"] for sp in S},
                sequences={sp: int(len(S[sp]["X"])) for sp in S}, steps=int(S["train"]["X"].shape[1]))
    pd.Series(meta).to_json(os.path.join(a.lyap_out, "lyap_meta.json"))


def cmd_report(a, b):
    d = pd.read_csv(os.path.join(a.lyap_out, "lyap_runs.csv"))
    cols = [c for c in KEY + [c for c in d if c.startswith(("val_", "test_")) and c not in KEY] if c in d]
    rows = []
    for arm, g in d.groupby("arm", sort=False):
        r = dict(arm=arm, n_seeds=len(g), bound_ok=int(g.bound_ok.sum()))
        for c in cols:
            x = g[c].replace([np.inf, -np.inf], np.nan).dropna()
            r[c] = float("inf") if np.isinf(g[c]).any() else (x.mean() if len(x) else float("nan"))
            r[c + "_ci95"] = M.ci95(x) if len(x) > 1 and np.isfinite(r[c]) else float("nan")
        rows.append(r)
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(a.lyap_out, "lyap_summary.csv"), index=False)
    paired = []
    if "lstm" in set(d.arm):
        base = d[d.arm == "lstm"].set_index("seed")
        for arm, g in d[d.arm != "lstm"].groupby("arm", sort=False):
            g = g.set_index("seed")
            s = g.index.intersection(base.index)
            for c in [c for c in cols if c.startswith(("val_", "test_"))]:
                diff = g.loc[s, c] - base.loc[s, c]
                paired.append(dict(arm=arm, metric=c, arm_mean=g.loc[s, c].mean(), lstm_mean=base.loc[s, c].mean(),
                                   diff=diff.mean(), diff_ci95=M.ci95(diff), seeds=len(s)))
    pd.DataFrame(paired).to_csv(os.path.join(a.lyap_out, "lyap_paired.csv"), index=False)
    pd.set_option("display.width", 250)
    print(summ[["arm", "n_seeds", "bound_ok"] + [c for c in KEY if c in summ]].round(4).to_string(index=False))


def main(argv=None):
    a, b = parse(argv)
    if a.command in ("run", "all"):
        cmd_run(a, b)
    if a.command in ("report", "all"):
        cmd_report(a, b)


if __name__ == "__main__":
    main()
