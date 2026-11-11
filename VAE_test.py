"""
File: VAE_test.py
Description: Checks whether the VAE latent of the UAV-IoT world model can be replaced by a linear
compressor without losing what the world model needs. Standalone test, separate from the full
world-model project. Needs the sim_mods package (the simulator) next to this file.

Scenario (every episode is a NEW world, so no landscape can be memorised)
    * a farm (default --layout farm, 36 km^2): crop fields 400-800 m on a side with no trees inside,
      hedgerows along field borders, 10-30 % of blocks left as woodlots, one farmstead near a map edge;
      200 sensors in instrumented fields (3-8 per field, the farmstead's field first);
    * a grid-powered BASE STATION at the farmstead: the cluster head of the sensors within 500 m,
      never drains, always charges a docked UAV fully; UAVs start the day there. It is not compressed
      (it has no hidden state) and does not count toward the CH range below;
    * UGV cluster heads parked on open field ground at least --bs-exclusion-m (1.5 km) from the base;
    * fresh terrain, foliage, layout, date (Apr-Sep, all-daylight 720-min episode) and hidden starting
      charge for every device in every episode;
    * terrain drawn at the LTAC map's physical scale ("fixed" mode), so slopes are as steep as on the
      800 m UGV map at any area size;
    * UGV cluster heads placed by K-means on the sensors the base does not serve. K is chosen per world by the best silhouette score among the
      K in [max(--min-ch, number of UAVs), --max-ch] (default 6-10 with 6 UAVs) that pass a flight
      check: a UAV leaving a CH at 80 % charge must reach the CH's farthest sensor and return with the
      20 % reserve, with the 25 % leg margin of the simulator's charging rule. If no K in the range
      passes, K grows until one does (with a warning). Sensors out of a CH's backscatter range are
      reached by UAV tours. The CH count therefore varies by world;
    * sensor tours (default --sweep cached): each UGV caches a nearest-neighbour + 2-opt order of its
      sensors once; a sortie walks that loop from the stalest sensor, skipping sensors collected in
      the last --tour-fresh-min minutes and any that would breach the battery reserve (no splitting);
      the UAV tops up first when the battery, not the need, shortened the sortie;
    * charger diversion (default on): a UAV whose UGV supplies no charge for 2 minutes, or whose next
      leg would breach the reserve, flies to the best reachable charger (distance / perceived charge
      rating; the base station is always rated highest);
    * UAV visitation (no UABO component) cycles by episode through three behaviour policies: greedy
      (highest AoI minus flight time), random sequence (each UAV follows its own random permutation of
      the chargers), and rotation (all UAVs cycle one fixed ring, spread apart). Charging is the
      rule-based dynamic heuristic plus charger diversion;

What is compressed
    ch    one token per CH per step: the fleet's view of that CH (telemetry SoC at last contact,
          minutes since contact, since a charge shortfall, peak/avg AoI, data share) over the last
          --history frames spaced --history-stride minutes, plus time of day/year, sun elevation,
          cluster size, distance to the nearest UAV and distance to the base station (the
          guaranteed charger, whose rating is always the highest). The SoC entry is the fleet's
          shared perceived charge, the same reading UAVs use to choose a charger. Tokens make the test independent of the CH
          count and carry no CH identity.
    map   one 2 x 32 x 32 patch per node (320 m x 320 m: canopy height, terrain relative to the node).

Compressors (k real numbers kept, every latent quantised to 8 bits, bits = 8k)
    linear  pca, rp (orthonormal random projection), pls (ch only; fitted to the hidden truth, so it is a
            privileged reference and never counted as the best deployable method), and for maps the fixed
            transforms dct, fft, haar, db2, block. All orthonormal ones are exactly 1-Lipschitz.
    learned vae (World Models recipe: SSE + KL with 0.5 nat/dim tolerance) and ae (same net, no KL),
            trained with early stopping on the validation episodes.

Episodes are split into train / validation / test by a fixed per-episode draw (15 % val, 25 % test), so
adding episodes never moves one between splits and no test world is seen in training. Every metric is
computed on validation AND test episodes; all comparisons and choices (early stopping, the best linear
method at each budget) use validation, and test is reported for the chosen methods only.

Enough training data: 'all' runs a learning curve for the VAE baseline (retrained on 25/50/75/100 %
of the training episodes, scored on validation) and keeps doubling the episode count until the last
step improves validation reconstruction by < 2 % and the probe score by < 0.01 (or --max-episodes).

Scores (prefixed val_ / test_)
    fvu         reconstruction error / variance
    ch:  soc_r2   linear-probe R^2 of each CH's TRUE charge (hidden from the fleet)
         out60_auc  linear-probe ROC-AUC for "this CH goes out within 60 min"
         sens_r2    linear-probe R^2 of the share of the CH's sensors in outage
    map: sun_r2   MLP-probe R^2 of ray-marched sunlight (daily mean + four time windows)
    lipschitz   exact for linear encoders; for vae/ae the largest encoder Jacobian norm over test
                inputs (a lower bound)

Usage
    python VAE_test.py all --device cuda           # collect, add data until sufficient, run, report
    python VAE_test.py collect --episodes 120 --workers 8
    python VAE_test.py sufficiency                 # learning curve only
    python VAE_test.py run --seeds 9 --device cuda
    python VAE_test.py report
    python VAE_test.py all --quick                 # minutes-long smoke run of the whole pipeline

Outputs (in --out, default vae_test_out/)
    data/ep_XXXX.npz       one file per episode
    runs_{ch,map}.csv      one row per (method, k, seed); the run step resumes from these
    summary_{ch,map}.csv   mean, sd and 95 % CI over seeds
    sufficiency.csv        VAE learning curve on validation, per round of collection
    gaps.csv               best deployable linear (chosen on validation) vs VAE, val and test
    report.png             four panels: information kept and reconstruction error vs bits
"""
import argparse
import contextlib
import glob
import io
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

EPISODE_STEPS = 720
SENSOR_OUT_SOC = 10.0          # Sensor.harvest_data stops below this when not on solar
CH_OUT_SOC = 10.0              # ClusterHead.charge_reserve: below this the CH cannot charge a UAV
CH_FEATS = ["tel_soc", "since_contact", "since_shortfall", "peak_aoi", "avg_aoi", "data_share"]
PATCH, CELL_M, SUN_CHECK_MIN = 32, 10, 15
MAP_METHODS = ("dct", "fft", "haar", "db2", "block", "pca", "rp")
CH_METHODS = ("pca", "rp", "pls")
STOCHASTIC = {"rp", "ae", "vae"}


# =====================================================================================
# Simulation
# =====================================================================================
class NullPower:
    """Power-agent stand-in: charging uses the 'dynamic' heuristic, which ignores it."""
    def act(self, state):
        return 0


class VisitPolicy:
    """
    Behaviour policies for data collection (no UABO component: no trained agent, no projected state).
    The simulator asks for a destination on every decision while a UAV sits at a charger, so each
    policy answers from the charger the UAV is at, giving the same answer when asked again.
    Choices are indices into the charger list (UGV cluster heads and, if present, the base station).

      greedy    highest peak AoI minus flight minutes, skipping chargers another UAV is heading to
      random    each UAV works through its own random permutation of the chargers; a new
                permutation is drawn when it reaches the end
      rotation  all UAVs cycle one fixed ring (nearest-neighbour + 2-opt over the charger positions),
                spread out by starting UAV i at position i * K / U of the ring
    """
    def __init__(self, kind, chs, uavs, rng, uav_speed):
        self.kind, self.chs, self.uavs, self.rng, self.speed = kind, chs, uavs, rng, uav_speed
        self.uav = None
        self.perm = {}
        self.started = set()
        if kind == "rotation":
            self.ring = self._ring()
            self.pos = {c: k for k, c in enumerate(self.ring)}

    def _ring(self):
        P = np.array([(ch.id_x, ch.id_y) for ch in self.chs], float)
        D = np.hypot(*(P[:, None] - P[None]).transpose(2, 0, 1))
        n = len(P)
        tour, left = [0], set(range(1, n))
        while left:
            j = min(left, key=lambda k: D[tour[-1], k])
            tour.append(j)
            left.remove(j)
        t = np.array(tour + [0])
        improved = True
        while improved and n > 3:
            improved = False
            for i in range(1, n - 1):
                a, b, c, d = t[i - 1], t[i], t[i + 1:n], t[i + 2:n + 1]
                delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]
                j = int(np.argmin(delta))
                if delta[j] < -1e-9:
                    t[i:i + 2 + j] = t[i:i + 2 + j][::-1]
                    improved = True
        return [int(k) for k in t[:n]]

    def _here(self):
        return self.uav.target.cluster_num if self.uav.target is not None and self.uav.target.type == 2 else -1

    def act(self, state):
        here, K = self._here(), len(self.chs)
        if self.kind == "greedy":
            claimed = {u.targetHead.cluster_num for u in self.uavs
                       if u is not self.uav and u.targetHead is not None and not u.crash}
            best, best_s = (here + 1) % K, -1e18
            for c, ch in enumerate(self.chs):
                if c == here or c in claimed:
                    continue
                fly_min = math.hypot(ch.id_x - self.uav.id_x, ch.id_y - self.uav.id_y) / (self.speed * 60)
                s = state[c + 1][2] - fly_min
                if s > best_s:
                    best, best_s = c, s
            return best
        key = id(self.uav)
        if self.kind == "random":
            perm = self.perm.get(key)
            if perm is None or here not in perm or perm.index(here) == K - 1:
                perm = [int(c) for c in self.rng.permutation(K)]
                if here in perm:                        # current charger first, then follow the draw
                    perm.remove(here)
                    perm.insert(0, here)
                self.perm[key] = perm
            return perm[perm.index(here) + 1] if here in perm else perm[0]
        # rotation: the next charger on the ring; on its first leg UAV i jumps i * K / U positions ahead
        # (remembered while it is still at its starting charger, so repeated asks agree)
        if key not in self.started:
            i = self.uavs.index(self.uav)
            target = self.ring[(self.pos.get(here, 0) + 1 + i * K // len(self.uavs)) % K]
            if target == here:
                target = self.ring[(self.pos.get(here, 0) + 1) % K]
            self.started.add(key)
            self.perm[key] = (here, target)
        start = self.perm.get(key)
        if start is not None:
            if here == start[0]:
                return start[1]
            self.perm[key] = None
        return self.ring[(self.pos.get(here, -1) + 1) % K]


def flight_check(a):
    """
    CH-layout check used when K-means picks K: a UAV leaving a CH at --depart-charge % must be able to fly
    to that CH's farthest member sensor and back and still land with --uav-reserve %, with the 25 % leg
    margin of the simulator's dynamic charging rule. Flight cost per metre comes from the simulator
    (45 min endurance at 15 m/s). Returns (ok, info) for sim_env's ch_check hook.
    """
    from sim_mods import ablation
    pct_per_m = ablation.flight_cost_per_s(ablation.UAV_FLIGHT_HOURS) / ablation.UAV_SPEED
    budget = a.depart_charge - a.uav_reserve

    def check(pts, centres, labels):
        farthest = float(np.hypot(*(pts - centres[labels]).T).max())
        need = a.flight_margin * 2.0 * farthest * pct_per_m
        return need <= budget, dict(farthest_m=farthest, round_trip_pct=need, budget_pct=budget)
    return check


def build_world(ep_seed, a):
    """A brand-new world for one episode: landscape, placement, date, CH count."""
    from sim_mods.ablation import CFG
    from sim_mods.environments import sim_env
    rng = np.random.default_rng(ep_seed)
    doy = int(rng.integers(91, 254))                       # Apr 1 .. Sep 10: whole episode in daylight
    date = (np.datetime64("2021-01-01") + np.timedelta64(doy - 1, "D")).astype(str)
    cover = float(rng.uniform(*a.tree_cover))
    sigma = float(rng.uniform(*a.stand_size_m))
    woodlot = float(rng.uniform(*a.woodlot_frac))
    # No UABO component: rule-based charging, the raw (unprojected) decision state, and the
    # behaviour policy below for targeting
    CFG.charging, CFG.sensor_sweep, CFG.decision_state = "dynamic", a.sweep, "raw"
    CFG.charger_divert, CFG.tour_fresh_min = a.charger_divert, a.tour_fresh_min
    # K-means picks K in [max(--min-ch, number of UAVs), --max-ch]: at least one CH per UAV
    ch_range = None if a.num_ch is not None or a.ch_select != "silhouette" else (max(a.min_ch, a.uavs), a.max_ch)
    auto = a.num_ch is None and a.ch_select == "coverage"
    env = sim_env("test", a.sensors, a.uavs, a.num_ch or 1, EPISODE_STEPS, map_seed=ep_seed,
                  tree_cover=cover, patch_sigma_m=sigma, area_km2=a.area_km2, auto_clusterheads=auto,
                  ch_range=ch_range, ch_check=flight_check(a) if ch_range else None,
                  placement_seed=f"episode-{ep_seed}", episode_date=date, terrain=a.terrain,
                  terrain_feature_m=a.terrain_feature_m, layout=a.layout, field_m=a.field_m,
                  woodlot_frac=woodlot, hedgerow_m=a.hedgerow_m,
                  base_station=a.base_station and a.layout == "farm", bs_exclusion_m=a.bs_exclusion_m,
                  sensors_per_field=a.sensors_per_field)
    sel = env.ch_selection or {}
    return env, dict(doy=doy, date=date, tree_cover=cover, stand_size_m=sigma, woodlot_frac=woodlot,
                     farthest_m=(sel.get("info") or {}).get("farthest_m", np.nan),
                     round_trip_pct=(sel.get("info") or {}).get("round_trip_pct", np.nan))


def run_episode(env, info, kind, ep_seed, a):
    from sim_mods import ablation
    ep_rng = np.random.default_rng(ep_seed + 1)
    random.seed(ep_seed)
    np.random.seed(ep_seed % (2 ** 32))
    sensors = [env.sensor_table.iat[i, 0] for i in range(len(env.sensor_table))]
    chs = [env.ch_table.iat[i, 0] for i in range(len(env.ch_table))]
    uavs = [env.uav_table.iat[i, 0] for i in range(len(env.uav_table))]
    K, U, S, T = len(chs), len(uavs), len(sensors), EPISODE_STEPS

    # Device reset as in make_env.reset, then the hidden overnight state: random starting charge
    start = [(u.id_x, u.id_y) for u in uavs]
    for s in sensors:
        s.reset()
        s.stored_energy = float(ep_rng.uniform(*a.sensor_start_soc))
        s.harvested_mAh = 0.0
    is_base = np.array([getattr(ch, "is_base", False) for ch in chs])
    for ch in chs:
        ch.reset()
        if not getattr(ch, "is_base", False):       # the grid-powered base station stays at 100 %
            ch.stored_energy = float(ep_rng.uniform(*a.ch_start_soc))
    for u, (x, y) in zip(uavs, start):
        u.reset()
        u.id_x, u.id_y = x, y
        u.target_x, u.target_y = x, y
        u.stored_energy = float(ep_rng.uniform(*a.uav_start_soc))
        u.state[0][2] = u.stored_energy

    sid = {id(s): i for i, s in enumerate(sensors)}
    cluster = np.zeros(S, int)
    for c, ch in enumerate(chs):
        for j in range(len(ch.sens_table)):
            cluster[sid[id(ch.sens_table.iat[j, 0])]] = c
    policy = VisitPolicy(kind, chs, uavs, ep_rng, ablation.UAV_SPEED)
    power = NullPower()
    sun_elev = np.clip(90.0 - env.sun_zenith, 0.0, 90.0) / 90.0

    last_contact = np.full(K, -10_000.0)
    last_short = np.full(K, -10_000.0)
    short_seen = np.array([ch.outage_minutes for ch in chs], float)
    dead = np.zeros(U, bool)
    died_at = np.zeros(U)                          # step a UAV died (its AoI table stops ageing then)
    died_docked = np.zeros(U, bool)                # died while trying to charge at a CH
    ch_obs = np.zeros((T, K, len(CH_FEATS)), np.float32)
    ch_soc = np.zeros((T, K), np.float32)
    sens_out = np.zeros((T, S), bool)
    uav_xy = np.zeros((T, U, 2), np.float32)
    uav_alive = np.zeros((T, U), bool)

    for t in range(T):
        sampled = np.zeros(S, bool)
        for j, s in enumerate(sensors):
            s.harvest_energy(env, t)
            sampled[j] = s.harvest_data()            # False = sensor outage (no solar, battery <= 10 %)
        for ch in chs:
            ch.harvest_energy(env, t)
            ch.download(t)
        for i, u in enumerate(uavs):
            if dead[i]:
                continue
            policy.uav = u
            u.set_dest(policy, power, t)
            u.navigate_step(env)
            u.receive_data(t)
            u.receive_energy()
            u.step_battery()
            if u.crash or u.stored_energy <= 0:
                dead[i] = u.crash = True
                died_at[i] = t
                died_docked[i] = bool(u.is_charging)
                continue
            tgt = u.target                          # CH telemetry piggybacks on a data upload
            if tgt is not None and tgt.type == 2 and u.inRange:
                last_contact[tgt.cluster_num] = t
        # The fleet's perceived charge of each CH (the simulator's shared reading, used for diversions)
        tel_soc = np.array([ch.perceived_soc / 100.0 for ch in chs])
        now_short = np.array([ch.outage_minutes for ch in chs], float)
        last_short[now_short > short_seen] = t      # a docked UAV got less charge than it asked for
        short_seen = now_short
        for s in sensors:
            s.step_battery()
        for ch in chs:
            ch.step_battery()

        ch_soc[t] = [ch.stored_energy for ch in chs]
        sens_out[t] = ~sampled
        uav_alive[t] = ~dead
        uav_xy[t] = [(u.id_x, u.id_y) for u in uavs]
        data_tot = max(sum(u.state[0][1] for u in uavs), 1)
        # A UAV ages its AoI table only inside its own receive_data, so a dead UAV's table freezes.
        # The fleet's knowledge from it keeps ageing: add the minutes since it died.
        age = np.where(dead, t - died_at, 0.0)
        for c in range(K):
            peak = min(u.state[c + 1][2] + age[i] for i, u in enumerate(uavs))
            avg = min(u.state[c + 1][3] + age[i] for i, u in enumerate(uavs))
            ch_obs[t, c] = [tel_soc[c], min(t - last_contact[c], T) / T, min(t - last_short[c], T) / T,
                            min(peak, T) / T, min(avg, T) / T, sum(u.state[c + 1][1] for u in uavs) / data_tot]

    ch_out = ch_soc <= CH_OUT_SOC
    ch_out_60 = np.stack([ch_out[t:t + 61].any(0) for t in range(T)])
    sens_frac = np.stack([sens_out[:, cluster == c].mean(1) if (cluster == c).any() else np.zeros(T)
                          for c in range(K)], 1).astype(np.float32)
    time_feats = np.column_stack([np.arange(T) / T, np.full(T, math.sin(2 * math.pi * info["doy"] / 365)),
                                  np.full(T, math.cos(2 * math.pi * info["doy"] / 365)), sun_elev]).astype(np.float32)
    return dict(ch_obs=ch_obs, ch_soc=ch_soc, ch_out=ch_out, ch_out_60=ch_out_60, sens_frac=sens_frac, time=time_feats,
                ch_xy=np.array([(ch.id_x, ch.id_y) for ch in chs], np.float32),
                charger_m=np.array([getattr(ch, "charger_m", 0.0) for ch in chs], np.float32),
                cluster_size=np.bincount(cluster, minlength=K).astype(np.float32),
                uav_xy=uav_xy, uav_alive=uav_alive, uav_died_docked=died_docked, is_base=is_base,
                ch_out_any=ch_out[:, ~is_base].any(0), sens_out_any=sens_out.any(0),
                dim=np.float32(env.dim))


def node_patches(env, pts):
    """(N, 2, 32, 32): canopy height / 32 m, terrain relative to the node / 10 m, at 10 m cells."""
    res, P = env.map_res, env.pad_cells
    f = CELL_M // res
    half = PATCH * f // 2
    out = np.zeros((len(pts), 2, PATCH, PATCH), np.float32)
    for i, (x, y) in enumerate(pts):
        cx, cy = int(x) // res + P, int(y) // res + P
        t = env.topo_mask[cy - half:cy + half, cx - half:cx + half].reshape(PATCH, f, PATCH, f).mean((1, 3))
        g = env.foliage_mask[cy - half:cy + half, cx - half:cx + half].reshape(PATCH, f, PATCH, f).mean((1, 3))
        out[i, 0] = g / 32.0
        out[i, 1] = (t - env.topo_mask[cy, cx]) / 10.0
    return out


def sunlight(env, pts):
    """(N, 5): ray-marched, elevation-weighted daily mean transmittance, then 4 time-window means."""
    res = env.map_res
    checks = np.arange(0, len(env.sun_zenith), SUN_CHECK_MIN)
    cx = np.clip(pts[:, 0].astype(int) // res, 0, env.map_cells - 1)
    cy = np.clip(pts[:, 1].astype(int) // res, 0, env.map_cells - 1)
    n, m = len(pts), len(checks)
    T = env._ray_march(np.repeat(cx, m), np.repeat(cy, m), np.tile(env.sun_azimuth[checks], n),
                       np.tile(env.sun_zenith[checks], n)).reshape(n, m)
    w = np.clip(np.sin(np.radians(90.0 - env.sun_zenith[checks])), 0.0, None)
    q = [T[:, i].mean(1) for i in np.array_split(np.arange(m), 4)]
    return np.column_stack([(T * w).sum(1) / max(w.sum(), 1e-9)] + q).astype(np.float32)


def collect_one(job):
    ep, a = job
    path = os.path.join(a.out, "data", f"ep_{ep:04d}.npz")
    if os.path.exists(path):
        return ep, "exists"
    t0 = time.time()
    ep_seed = a.seed_base + ep
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink if not a.verbose else sys.stdout):
        env, info = build_world(ep_seed, a)
        d = run_episode(env, info, a.policy_cycle[ep % len(a.policy_cycle)], ep_seed, a)
        rng = np.random.default_rng(ep_seed + 2)
        sens = np.array([(env.sensor_table.iat[i, 0].id_x, env.sensor_table.iat[i, 0].id_y)
                         for i in range(len(env.sensor_table))], float)
        margin = PATCH * CELL_M / 2 + 10
        rnd = rng.uniform(margin, env.dim - margin, size=(a.random_patches, 2))
        pts = np.vstack([sens, d["ch_xy"], rnd])
        kinds = np.concatenate([np.ones(len(sens), int), np.full(len(d["ch_xy"]), 2), np.zeros(len(rnd), int)])
        P = env.pad_cells
        h = env.topo_mask[P:P + env.map_cells, P:P + env.map_cells]
        gy, gx = np.gradient(h, env.map_res)
        slope = np.degrees(np.arctan(np.hypot(gx, gy)))
        d.update(patches=node_patches(env, pts), sun=sunlight(env, pts), kinds=kinds,
                 policy=a.policy_cycle[ep % len(a.policy_cycle)], doy=info["doy"], tree_cover=info["tree_cover"],
                 stand_size_m=info["stand_size_m"], farthest_m=np.float32(info["farthest_m"]),
                 round_trip_pct=np.float32(info["round_trip_pct"]), slope_p50_p90=np.percentile(slope, [50, 90]).astype(np.float32),
                 relief_m=np.float32(np.ptp(h)))
    tmp = path[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, **d)
    os.replace(tmp, path)                          # never leave a half-written episode behind
    K = int((~d["is_base"]).sum())
    return ep, (f"{K:2d} UGV CHs{' + base' if d['is_base'].any() else ''}  CH-out(any) {d['ch_out_any'].mean():.2f}  sensor-out(any) {d['sens_out_any'].mean():.2f}  "
                f"UAVs alive {d['uav_alive'][-1].mean():.2f}  farthest sensor {info['farthest_m'] / 1000:.1f} km "
                f"({info['round_trip_pct']:.0f} % round trip)  slope p50/p90 {d['slope_p50_p90'][0]:.0f}/"
                f"{d['slope_p50_p90'][1]:.0f} deg  {time.time() - t0:.0f}s")


WORLD_KEYS = ("area_km2", "sensors", "uavs", "num_ch", "ch_select", "min_ch", "max_ch", "depart_charge",
              "uav_reserve", "flight_margin", "val_frac", "test_frac", "terrain", "policy_cycle", "ch_start_soc",
              "sensor_start_soc", "uav_start_soc", "layout", "field_m", "woodlot_frac", "hedgerow_m",
              "sensors_per_field", "base_station", "sweep", "charger_divert", "tour_fresh_min", "bs_exclusion_m",
              "terrain_feature_m", "tree_cover", "stand_size_m", "random_patches", "seed_base")
RUN_KEYS = ("n_episodes", "val_frac", "test_frac", "history", "history_stride", "token_every", "z_clip", "max_train", "max_eval",
            "max_probe", "max_epochs", "patience", "batch", "lr", "beta", "kl_tol")


def check_config(path, a, keys, what):
    """Refuse to mix outputs made under different settings (they would be silently reused)."""
    import json
    cfg = {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(a).items() if k in keys}
    if os.path.exists(path):
        old = json.load(open(path))
        diff = {k: (old.get(k), cfg[k]) for k in cfg if old.get(k) != cfg[k]}
        if diff:
            raise SystemExit(f"{what} in {os.path.dirname(path) or '.'} were made with different settings "
                             f"(old, new): {diff}. Use a new --out, or delete them to start over.")
    else:
        json.dump(cfg, open(path, "w"), indent=1)


def cmd_collect(a):
    os.makedirs(os.path.join(a.out, "data"), exist_ok=True)
    check_config(os.path.join(a.out, "data", "world_config.json"), a, WORLD_KEYS, "episodes")
    jobs = [(ep, a) for ep in range(a.episodes)]           # existing episodes are skipped
    print(f"collecting {a.episodes} episodes ({a.area_km2:g} km^2, {a.sensors} sensors, {a.uavs} UAVs, "
          f"CHs {a.num_ch if a.num_ch is not None else ('K-means, silhouette K in [%d, %d] + flight check' % (max(a.min_ch, a.uavs), a.max_ch) if a.ch_select == 'silhouette' else 'K-means, 500 m coverage')}, terrain {a.terrain}) "
          f"with {a.workers} worker(s)", flush=True)
    if a.workers <= 1:
        for j in jobs:
            print("episode %d: %s" % collect_one(j), flush=True)
    else:
        # "spawn", not fork: after the sufficiency check has loaded PyTorch, forked workers can deadlock
        import multiprocessing as mp
        with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as ex:
            for ep, msg in ex.map(collect_one, jobs):
                print(f"episode {ep}: {msg}", flush=True)


# =====================================================================================
# Datasets
# =====================================================================================
def episode_id(path):
    return int(os.path.basename(path)[3:7])


def split_of(ep, a):
    """Fixed per episode (not by position), so collecting more episodes never moves one between splits."""
    u = np.random.default_rng([ep, 7919]).random()
    return "test" if u < a.test_frac else "val" if u < a.test_frac + a.val_frac else "train"


def split_episodes(files, a):
    parts = {"train": [], "val": [], "test": []}
    for f in files:
        parts[split_of(episode_id(f), a)].append(f)
    return parts["train"], parts["val"], parts["test"]


def ch_tokens(d, a, rng, cap):
    """Tokens (N, D) and targets (N, 3) for one episode; steps subsampled every --token-every minutes."""
    if "is_base" in d and d["is_base"].any():         # base station: no hidden state, not compressed
        keep = ~d["is_base"]
        d = dict(d, ch_obs=d["ch_obs"][:, keep], ch_xy=d["ch_xy"][keep], cluster_size=d["cluster_size"][keep],
                 ch_soc=d["ch_soc"][:, keep], ch_out=d["ch_out"][:, keep], ch_out_60=d["ch_out_60"][:, keep],
                 sens_frac=d["sens_frac"][:, keep], charger_m=d["charger_m"][keep])
    T, K, F = d["ch_obs"].shape
    steps = np.arange(0, T, a.token_every)
    lags = np.arange(a.history) * a.history_stride
    frames = np.clip(steps[:, None] - lags[None, :], 0, T - 1)              # (n_steps, H)
    hist = d["ch_obs"][frames]                                              # (n_steps, H, K, F)
    hist = hist.transpose(0, 2, 1, 3).reshape(len(steps), K, a.history * F)
    tm = np.repeat(d["time"][steps][:, None, :], K, 1)
    dim = float(d["dim"])
    alive = d["uav_alive"][steps]                                           # (n, U)
    dxy = d["uav_xy"][steps][:, None, :, :] - d["ch_xy"][None, :, None, :]   # (n, K, U, 2)
    dist = np.hypot(dxy[..., 0], dxy[..., 1]) / dim
    dist = np.where(alive[:, None, :], dist, 2.0).min(2)
    size = np.repeat((d["cluster_size"] / 10.0)[None, :], len(steps), 0)
    to_base = np.repeat((d["charger_m"] / dim)[None, :], len(steps), 0)      # 0 when there is no base
    X = np.concatenate([hist, tm, size[..., None], dist[..., None], to_base[..., None]], 2).reshape(-1, a.history * F + 7)
    # targets: true charge, out now or within 60 min, sensor-outage share; col 3 = out now (scoring only)
    Y = np.stack([d["ch_soc"][steps] / 100.0, d["ch_out_60"][steps].astype(np.float32),
                  d["sens_frac"][steps], d["ch_out"][steps].astype(np.float32)], 2).reshape(-1, 4)
    if cap and len(X) > cap:
        idx = rng.choice(len(X), cap, replace=False)
        X, Y = X[idx], Y[idx]
    return X.astype(np.float64), Y.astype(np.float64)


def load_track(track, a):
    """
    Returns ((Xtr, Ytr, Gtr), (Xva, Yva), (Xte, Yte)), standardised with training statistics.
    Gtr is each training row's episode id (used to subsample by episode in the sufficiency check).
    """
    files = sorted(glob.glob(os.path.join(a.out, "data", "ep_*.npz")))
    parts = split_episodes(files, a)
    if min(len(p) for p in parts) == 0:
        raise SystemExit(f"need episodes in every split (have {[len(p) for p in parts]} train/val/test in "
                         f"{a.out}/data); collect more")
    rng = np.random.default_rng(0)
    out = []
    for fs, cap in zip(parts, (a.max_train, a.max_eval, a.max_eval)):
        Xs, Ys, Gs = [], [], []
        per_ep = int(math.ceil(cap / len(fs))) if cap else None
        for f in fs:
            d = dict(np.load(f))
            if track == "ch":
                X, Y = ch_tokens(d, a, rng, per_ep)
            else:
                X, Y = d["patches"].astype(np.float64), d["sun"].astype(np.float64)
                if per_ep and len(X) > per_ep:                 # cap per episode, like the CH tokens
                    keep = np.sort(rng.choice(len(X), per_ep, replace=False))
                    X, Y = X[keep], Y[keep]
            Xs.append(X)
            Ys.append(Y)
            Gs.append(np.full(len(X), episode_id(f)))
        out.append((np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Gs)))
    (Xtr, Ytr, Gtr), (Xva, Yva, _), (Xte, Yte, _) = out
    if track == "ch":
        mu, sd = Xtr.mean(0), Xtr.std(0)
    else:
        mu, sd = Xtr.mean((0, 2, 3), keepdims=True), Xtr.std((0, 2, 3), keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    setattr(a, f"_{track}_stats", (mu, sd))         # reused by Mem_test.py for whole-episode sequences
    # Rare, nearly constant features (e.g. minutes since a charge shortfall) give huge z-scores on held-out
    # episodes; clip so a handful of tokens cannot dominate any method's latent or probe.
    clip = (lambda Z: np.clip(Z, -a.z_clip, a.z_clip)) if track == "ch" else (lambda Z: Z)
    print(f"{track}: train {Xtr.shape} / val {Xva.shape} / test {Xte.shape} from "
          f"{len(parts[0])}/{len(parts[1])}/{len(parts[2])} episodes", flush=True)
    return (clip((Xtr - mu) / sd), Ytr, Gtr), (clip((Xva - mu) / sd), Yva), (clip((Xte - mu) / sd), Yte)


# =====================================================================================
# Linear compressors
# =====================================================================================
def _flat(X):
    return X.reshape(len(X), -1)


class Linear:
    """z = W (x - mu), x_hat = W^T z + mu with orthonormal rows: exactly 1-Lipschitz."""
    lipschitz = 1.0

    def __init__(self, k, seed=0):
        self.k, self.seed = k, seed

    def encode(self, X):
        return (_flat(X) - self.mu) @ self.W.T

    def decode(self, Z):
        return Z @ self.W + self.mu


class PCAComp(Linear):
    def fit(self, X, Y=None):
        from sklearn.decomposition import PCA
        p = PCA(n_components=self.k, random_state=self.seed).fit(_flat(X))
        self.mu, self.W = p.mean_, p.components_
        return self


class RPComp(Linear):
    def fit(self, X, Y=None):
        X = _flat(X)
        q, _ = np.linalg.qr(np.random.default_rng(self.seed).standard_normal((X.shape[1], self.k)))
        self.mu, self.W = X.mean(0), q.T
        return self


class PLSComp:
    """Partial least squares toward the hidden targets; reports its own Lipschitz constant."""
    def __init__(self, k, seed=0):
        self.k = k

    def fit(self, X, Y):
        from sklearn.cross_decomposition import PLSRegression
        p = PLSRegression(n_components=self.k, scale=False).fit(_flat(X), Y)
        self.mu, self.R, self.P = p._x_mean, p.x_rotations_, p.x_loadings_
        self.lipschitz = float(np.linalg.norm(self.R, 2))
        return self

    def encode(self, X):
        return (_flat(X) - self.mu) @ self.R

    def decode(self, Z):
        return Z @ self.P.T + self.mu


class MapTransform:
    """2-D orthonormal transform per channel, keeping k/2 fixed (low-frequency / coarse) coefficients."""
    lipschitz = 1.0

    def __init__(self, k, seed=0):
        if k % 2:
            raise ValueError("map budgets must be even (k/2 per channel)")
        self.k, self.kc = k, k // 2

    def fit(self, X, Y=None):
        self.shape, self.mu = X.shape[1:], X.mean(0)
        self._setup()
        return self

    def encode(self, X):
        X = X - self.mu
        return np.concatenate([self._enc(X[:, c]) for c in range(X.shape[1])], 1)

    def decode(self, Z):
        n = len(Z)
        out = np.stack([self._dec(Z[:, c * self.kc:(c + 1) * self.kc], n) for c in range(self.shape[0])], 1)
        return (out + self.mu).reshape(n, -1)


class DCTComp(MapTransform):
    def _setup(self):
        h, w = self.shape[1:]
        ii, jj = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        self.idx = np.lexsort((ii.ravel(), (ii + jj).ravel()))[:self.kc]

    def _enc(self, x):
        from scipy.fft import dctn
        return dctn(x, axes=(1, 2), norm="ortho").reshape(len(x), -1)[:, self.idx]

    def _dec(self, z, n):
        from scipy.fft import idctn
        h, w = self.shape[1:]
        full = np.zeros((n, h * w))
        full[:, self.idx] = z
        return idctn(full.reshape(n, h, w), axes=(1, 2), norm="ortho")


class FFTComp(MapTransform):
    """Real 2-D FFT, lowest radial frequencies; conjugate-redundant entries skipped, pairs scaled by sqrt 2."""
    def _setup(self):
        h, w = self.shape[1:]
        wr = w // 2 + 1
        KY, KX = np.meshgrid(np.fft.fftfreq(h) * h, np.arange(wr), indexing="ij")
        edge = (KX == 0) | (KX == w / 2)
        selfconj = ((KY == 0) | (np.abs(KY) == h / 2)) & edge
        redundant = edge & (KY < 0) & ~selfconj
        order = np.lexsort((np.abs(KY).ravel(), (KY ** 2 + KX ** 2).ravel()))
        slots, budget = [], self.kc
        for o in order:
            if redundant.ravel()[o]:
                continue
            cost = 1 if selfconj.ravel()[o] else 2
            if cost > budget:
                continue
            slots.append((o, bool(selfconj.ravel()[o])))
            budget -= cost
            if budget == 0:
                break
        self.slots, self.wr, self.pad = slots, wr, budget

    def _enc(self, x):
        F = np.fft.rfft2(x, norm="ortho").reshape(len(x), -1)
        cols = []
        for o, sc in self.slots:
            cols += [F[:, o].real] if sc else [np.sqrt(2) * F[:, o].real, np.sqrt(2) * F[:, o].imag]
        return np.stack(cols + [np.zeros(len(x))] * self.pad, 1)

    def _dec(self, z, n):
        h, w = self.shape[1:]
        F = np.zeros((n, h * self.wr), complex)
        j = 0
        for o, sc in self.slots:
            if sc:
                F[:, o] = z[:, j]
                j += 1
            else:
                F[:, o] = (z[:, j] + 1j * z[:, j + 1]) / np.sqrt(2)
                j += 2
        F = F.reshape(n, h, self.wr)
        for col in {0, self.wr - 1}:
            for ky in range(1, h // 2):
                F[:, h - ky, col] = np.conj(F[:, ky, col])
        return np.fft.irfft2(F, s=(h, w), norm="ortho")


class WaveletComp(MapTransform):
    wavelet = "haar"

    def _setup(self):
        import pywt
        h = self.shape[1]
        self.level = pywt.dwtn_max_level((h, h), self.wavelet)
        self.template = pywt.wavedec2(np.zeros(self.shape[1:]), self.wavelet, mode="periodization", level=self.level)

    def _vec(self, co):
        return np.concatenate([co[0].ravel()] + [d.ravel() for lvl in co[1:] for d in lvl])

    def _unvec(self, v):
        co, j = [], self.template[0].size
        co.append(v[:j].reshape(self.template[0].shape))
        for lvl in self.template[1:]:
            ds = []
            for d in lvl:
                ds.append(v[j:j + d.size].reshape(d.shape))
                j += d.size
            co.append(tuple(ds))
        return co

    def _enc(self, x):
        import pywt
        return np.stack([self._vec(pywt.wavedec2(xi, self.wavelet, mode="periodization", level=self.level))[:self.kc]
                         for xi in x])

    def _dec(self, z, n):
        import pywt
        D = int(np.prod(self.shape[1:]))
        out = []
        for zi in z:
            v = np.zeros(D)
            v[:self.kc] = zi
            out.append(pywt.waverec2(self._unvec(v), self.wavelet, mode="periodization"))
        return np.stack(out)


class HaarComp(WaveletComp):
    wavelet = "haar"


class DB2Comp(WaveletComp):
    wavelet = "db2"


class BlockComp(MapTransform):
    """Block means (downsampling), scaled by the block side so the map stays orthonormal."""
    def _setup(self):
        s = int(round(math.sqrt(self.kc)))
        if s * s != self.kc:
            raise ValueError("block averaging needs k/2 to be a perfect square")
        self.s, self.f = s, self.shape[1] // s

    def _enc(self, x):
        n = len(x)
        return (x.reshape(n, self.s, self.f, self.s, self.f).sum((2, 4)) / self.f).reshape(n, -1)

    def _dec(self, z, n):
        b = z.reshape(n, self.s, self.s) / self.f
        return np.repeat(np.repeat(b, self.f, 1), self.f, 2)


LINEAR = {("map", "dct"): DCTComp, ("map", "fft"): FFTComp, ("map", "haar"): HaarComp, ("map", "db2"): DB2Comp,
          ("map", "block"): BlockComp, ("map", "pca"): PCAComp, ("map", "rp"): RPComp,
          ("ch", "pca"): PCAComp, ("ch", "rp"): RPComp, ("ch", "pls"): PLSComp}


class Quantizer:
    """Uniform 8-bit quantisation per latent dimension, range taken from the training latents."""
    def fit(self, Z, bits=8):
        self.levels = 2 ** bits - 1
        self.lo, self.hi = Z.min(0), Z.max(0)
        self.span = np.where(self.hi - self.lo > 1e-12, self.hi - self.lo, 1.0)
        return self

    def __call__(self, Z):
        q = np.round((np.clip(Z, self.lo, self.hi) - self.lo) / self.span * self.levels)
        return q / self.levels * self.span + self.lo


# =====================================================================================
# Learned compressors (PyTorch)
# =====================================================================================
def neural(track, kind, k, seed, a):
    import torch
    import torch.nn as nn

    class MLPEnc(nn.Module):
        def __init__(s, d, var):
            super().__init__()
            s.body = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
            s.mu, s.lv = nn.Linear(128, k), (nn.Linear(128, k) if var else None)

        def forward(s, x):
            h = s.body(x)
            return s.mu(h), (s.lv(h) if s.lv is not None else None)

    class ConvEnc(nn.Module):
        def __init__(s, var):
            super().__init__()
            s.body = nn.Sequential(nn.Conv2d(2, 32, 4, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
                                   nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(), nn.Flatten())
            s.mu, s.lv = nn.Linear(2048, k), (nn.Linear(2048, k) if var else None)

        def forward(s, x):
            h = s.body(x.view(-1, 2, PATCH, PATCH))
            return s.mu(h), (s.lv(h) if s.lv is not None else None)

    class ConvDec(nn.Module):
        def __init__(s):
            super().__init__()
            s.fc = nn.Linear(k, 2048)
            s.net = nn.Sequential(nn.ReLU(), nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
                                  nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(), nn.ConvTranspose2d(32, 2, 4, 2, 1))

        def forward(s, z):
            return s.net(s.fc(z).view(-1, 128, 4, 4)).reshape(len(z), -1)

    class Model:
        lipschitz_kind = "lower_bound"

        def fit(self, X, Y=None, Xv=None):
            torch.manual_seed(seed)
            np.random.seed(seed)
            dev = torch.device(a.device)
            self.dev = dev
            X = torch.tensor(_flat(X), dtype=torch.float32).to(dev)
            Xv = _flat(Xv)
            if len(Xv) > 4096:                              # early-stopping subset (scoring uses all of val/test)
                Xv = Xv[np.sort(np.random.default_rng(0).choice(len(Xv), 4096, replace=False))]
            Xv = torch.tensor(Xv, dtype=torch.float32).to(dev)
            d, var = X.shape[1], kind == "vae"
            if track == "map":
                self.enc, self.dec = ConvEnc(var).to(dev), ConvDec().to(dev)
            else:
                self.enc = MLPEnc(d, var).to(dev)
                self.dec = nn.Sequential(nn.Linear(k, 128), nn.ReLU(), nn.Linear(128, 256), nn.ReLU(),
                                         nn.Linear(256, d)).to(dev)
            opt = torch.optim.Adam(list(self.enc.parameters()) + list(self.dec.parameters()), lr=a.lr)
            g = torch.Generator().manual_seed(seed)
            best, best_state, bad = float("inf"), None, 0
            for ep in range(a.max_epochs):
                self.enc.train(); self.dec.train()
                for idx in torch.randperm(len(X), generator=g).split(a.batch):
                    x = X[idx.to(dev)]
                    mu, lv = self.enc(x)
                    if var:
                        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
                        kl = (-0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1)).mean()
                        kl = a.beta * torch.clamp(kl, min=a.kl_tol * k)   # World Models: on the batch mean
                    else:
                        z, kl = mu, 0.0
                    loss = ((self.dec(z) - x) ** 2).sum(1).mean() + kl
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                val = float(((self.decode_t(self.encode_t(Xv)) - Xv.cpu()) ** 2).mean())
                if val < best - 1e-6:
                    best, bad = val, 0
                    best_state = ({n: p.detach().clone() for n, p in self.enc.state_dict().items()},
                                  {n: p.detach().clone() for n, p in self.dec.state_dict().items()})
                else:
                    bad += 1
                    if bad >= a.patience:
                        break
            self.epochs = ep + 1
            self.enc.load_state_dict(best_state[0])
            self.dec.load_state_dict(best_state[1])
            self.enc.eval(); self.dec.eval()
            return self

        @torch.no_grad()
        def encode_t(self, X):
            self.enc.eval()
            return torch.cat([self.enc(X[i:i + 4096].to(self.dev))[0].cpu() for i in range(0, len(X), 4096)])

        @torch.no_grad()
        def decode_t(self, Z):
            self.dec.eval()
            return torch.cat([self.dec(Z[i:i + 4096].to(self.dev)).cpu() for i in range(0, len(Z), 4096)])

        def encode(self, X):
            return self.encode_t(torch.tensor(_flat(X), dtype=torch.float32)).numpy().astype(np.float64)

        def decode(self, Z):
            return self.decode_t(torch.tensor(Z, dtype=torch.float32)).numpy().astype(np.float64)

        def jacobian_bound(self, X, n=64):
            X = _flat(X)
            idx = np.random.default_rng(0).choice(len(X), min(n, len(X)), replace=False)
            best = 0.0
            for i in idx:
                x = torch.tensor(X[i:i + 1], dtype=torch.float32, device=self.dev)
                J = torch.autograd.functional.jacobian(lambda v: self.enc(v)[0], x).reshape(k, -1)
                best = max(best, float(torch.linalg.matrix_norm(J, 2)))
            return best

    return Model()


# =====================================================================================
# Probes and the grid
# =====================================================================================
def probe(track, Ztr, Ytr, evals, seed, a):
    """
    Fits the probes once on training latents and scores them on each held-out set in `evals`
    ({"val": (Z, Y), "test": (Z, Y)}). Returns metrics prefixed with the set name.
    ch:  ridge probe -> soc_r2, out60_auc, onset_auc (CHs not out yet), sens_r2
    map: MLP probe -> sun_r2 (and ridge -> sun_r2_lin)
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score, roc_auc_score
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Ztr), min(a.max_probe, len(Ztr)), replace=False)
    zm, zs = Ztr.mean(0), Ztr.std(0) + 1e-9
    Yfit = Ytr[:, :3] if track == "ch" else Ytr           # the ch out-now flag is for scoring only
    ym, ys = Yfit.mean(0), Yfit.std(0) + 1e-9
    lo, hi = Yfit.min(0), Yfit.max(0)                      # targets are bounded: keep predictions in range
    A, T = (Ztr[idx] - zm) / zs, (Yfit[idx] - ym) / ys
    ridge = Ridge(alpha=1.0).fit(A, T)
    mlp = None
    if track == "map":
        from sklearn.neural_network import MLPRegressor
        mlp = MLPRegressor(hidden_layer_sizes=(128, 128), early_stopping=True, max_iter=200, random_state=seed,
                           learning_rate_init=1e-3).fit(A, T)
    out = {}
    for name, (Z, Y) in evals.items():
        B = (Z - zm) / zs
        lin = np.clip(ridge.predict(B) * ys + ym, lo, hi)

        def r2(P, cols):
            ok = [c for c in cols if Y[:, c].std() > 1e-9]
            return float(np.mean([r2_score(Y[:, c], P[:, c]) for c in ok])) if ok else float("nan")

        if track == "ch":
            def auc(mask):
                y = Y[mask, 1]
                return float(roc_auc_score(y, lin[mask, 1])) if 0 < y.mean() < 1 else float("nan")
            up = Y[:, 3] == 0                              # onset: not out yet, goes out within 60 min
            out.update({f"{name}_soc_r2": r2(lin, [0]), f"{name}_out60_auc": auc(np.ones(len(Y), bool)),
                        f"{name}_onset_auc": auc(up), f"{name}_onset_pos": int(Y[up, 1].sum()),
                        f"{name}_sens_r2": r2(lin, [2])})
        else:
            P = np.clip(mlp.predict(B) * ys + ym, lo, hi)
            out.update({f"{name}_sun_r2": r2(P, range(5)), f"{name}_sun_r2_lin": r2(lin, range(5))})
    return out


def run_one(track, name, k, seed, data, a, evals=("val", "test")):
    (Xtr, Ytr, _), (Xva, Yva), (Xte, Yte) = data
    held = {"val": (Xva, Yva), "test": (Xte, Yte)}
    t0 = time.time()
    epochs = None
    if name == "raw":
        c, lip, kind = None, 1.0, "exact"
    elif name in ("ae", "vae"):
        c = neural(track, name, k, seed, a).fit(Xtr, Ytr, Xva)
        lip, kind, epochs = c.jacobian_bound(Xva), "lower_bound", c.epochs
    else:
        c = LINEAR[(track, name)](k, seed).fit(Xtr, Ytr[:, :3] if track == "ch" else Ytr)
        lip, kind = float(c.lipschitz), "exact"
    enc = (lambda X: _flat(X)) if c is None else c.encode
    Ztr = enc(Xtr)
    Z = {n: enc(X) for n, (X, _) in held.items() if n in evals}
    # Scale-free gain on validation: Lipschitz bound / typical gain (RMS latent spread over RMS input
    # spread). Rescaling an encoder changes `lipschitz` but not this; 1 = no direction amplified beyond
    # the average one.
    Xv = _flat(Xva)
    Zv = enc(Xva)
    typical = math.sqrt(((Zv - Zv.mean(0)) ** 2).sum(1).mean() / ((Xv - Xv.mean(0)) ** 2).sum(1).mean())
    row = dict(track=track, method=name, k=int(Ztr.shape[1]), seed=seed, bits=8 * int(Ztr.shape[1]),
               lipschitz=lip, lipschitz_rel=lip / max(typical, 1e-12), lipschitz_kind=kind, epochs=epochs)
    q = Quantizer().fit(Ztr)
    Ztr = q(Ztr)
    for n in Z:
        Z[n] = q(Z[n])
        Xf = _flat(held[n][0])
        rec = Z[n] if c is None else c.decode(Z[n])
        row[f"{n}_fvu"] = float(np.mean((rec - Xf) ** 2) / Xf.var())
    row.update(probe(track, Ztr, Ytr, {n: (Z[n], held[n][1]) for n in Z}, seed, a))
    row["seconds"] = round(time.time() - t0, 1)
    return row


def cmd_sufficiency(a):
    """
    Learning curve for the VAE baseline: retrain it on growing shares of the training episodes and score
    it on the validation episodes. Training data is enough when going from the second-largest share to
    all of it improves validation reconstruction by less than --suff-tol (relative) and the primary
    validation probe score by less than --suff-probe-tol. Writes sufficiency.csv; returns True if enough.
    """
    resolve_device(a)
    rows, enough = [], True
    for track in a.tracks:
        (Xtr, Ytr, Gtr), va, te = load_track(track, a)
        eps = np.unique(Gtr)
        k = a.suff_k_ch if track == "ch" else a.suff_k_map
        primary = "val_soc_r2" if track == "ch" else "val_sun_r2"
        curve = []
        for frac in a.suff_fracs:
            keep = np.isin(Gtr, eps[:max(1, int(round(frac * len(eps))))])
            sub = ((Xtr[keep], Ytr[keep], Gtr[keep]), va, te)
            res = [run_one(track, "vae", k, s, sub, a, evals=("val",)) for s in range(a.suff_seeds)]
            r = dict(track=track, k=k, train_episodes=int(round(frac * len(eps))), train_rows=int(keep.sum()),
                     val_fvu=float(np.mean([x["val_fvu"] for x in res])),
                     val_primary=float(np.mean([x[primary] for x in res])), primary=primary,
                     epochs=float(np.mean([x["epochs"] for x in res])))
            curve.append(r)
            print("sufficiency", {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in r.items()}, flush=True)
        a2, b2 = curve[-2], curve[-1]
        rel = (a2["val_fvu"] - b2["val_fvu"]) / max(a2["val_fvu"], 1e-12)
        dp = b2["val_primary"] - a2["val_primary"]
        ok = rel < a.suff_tol and dp < a.suff_probe_tol
        for r in curve:
            r.update(last_step_fvu_gain=rel, last_step_probe_gain=dp, enough=ok)
        rows += curve
        print(f"sufficiency {track}: last step improved val FVU by {100 * rel:.1f} % and {primary} by {dp:+.3f} "
              f"-> {'enough training data' if ok else 'still improving'}", flush=True)
        enough &= ok
    path = os.path.join(a.out, "sufficiency.csv")
    pd.concat([pd.read_csv(path), pd.DataFrame(rows)]) .to_csv(path, index=False) if os.path.exists(path) \
        else pd.DataFrame(rows).to_csv(path, index=False)
    return enough


def outage_counts(a):
    """
    Outage-onset samples per split, counted the way the CH tokens sample them (every --token-every
    minutes, UGV cluster heads only): a UGV not out yet that goes out within 60 minutes. Also the
    count per behaviour policy, to show where outages come from.
    """
    files = sorted(glob.glob(os.path.join(a.out, "data", "ep_*.npz")))
    out = {"train": 0, "val": 0, "test": 0}
    by_policy = {}
    for f in files:
        d = np.load(f)
        ugv = ~d["is_base"] if "is_base" in d else np.ones(d["ch_out"].shape[1], bool)
        steps = np.arange(0, d["ch_out"].shape[0], a.token_every)
        n = int((d["ch_out_60"][steps][:, ugv] & ~d["ch_out"][steps][:, ugv]).sum())
        out[split_of(episode_id(f), a)] += n
        pol = str(d["policy"]) if "policy" in d else "?"
        by_policy[pol] = by_policy.get(pol, 0) + n
    return out, by_policy


def outages_enough(a):
    counts, by_policy = outage_counts(a)
    ok = counts["val"] >= a.min_onset and counts["test"] >= a.min_onset
    print(f"outage-onset samples: {counts} (need {a.min_onset} in val and test) by policy {by_policy} "
          f"-> {'enough' if ok else 'need more episodes'}", flush=True)
    return ok


def resolve_device(a):
    if a.device == "auto":
        try:
            import torch
            a.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            a.device = "cpu"


def cmd_run(a):
    resolve_device(a)
    a.n_episodes = len(glob.glob(os.path.join(a.out, "data", "ep_*.npz")))
    print(f"device {a.device}, {a.seeds} seeds for {sorted(STOCHASTIC)}", flush=True)
    check_config(os.path.join(a.out, "run_config.json"), a, RUN_KEYS, "results")
    for track in a.tracks:
        data = load_track(track, a)
        budgets = a.ch_budgets if track == "ch" else a.map_budgets
        methods = [m for m in (CH_METHODS if track == "ch" else MAP_METHODS) + ("ae", "vae") if m in a.methods]
        path = os.path.join(a.out, f"runs_{track}.csv")
        done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
        raw_k = int(np.prod(data[0][0].shape[1:]))
        jobs = [("raw", raw_k, 0)] + [(m, k, s) for k in budgets for m in methods
                                       for s in (range(a.seeds) if m in STOCHASTIC else [0])]
        for m, k, s in jobs:
            if len(done) and ((done.method == m) & (done.k == k) & (done.seed == s)).any():
                continue
            row = run_one(track, m, k, s, data, a)
            done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
            done.to_csv(path, index=False)
            show = {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in row.items() if kk != "track"}
            print(track, show, flush=True)


# =====================================================================================
# Report
# =====================================================================================
PRIMARY = {"ch": ["soc_r2", "onset_auc", "out60_auc", "sens_r2"], "map": ["sun_r2", "sun_r2_lin"]}
DEPLOYABLE = {"ch": {"pca", "rp"}, "map": set(MAP_METHODS)}   # PLS needs the hidden truth: reference only


def summarise(track, a):
    from scipy import stats
    path = os.path.join(a.out, f"runs_{track}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    cols = ["lipschitz", "lipschitz_rel"] + [f"{n}_{m}" for n in ("val", "test") for m in ["fvu"] + PRIMARY[track]]
    rows = []
    for (m, k), g in df.groupby(["method", "k"]):
        r = dict(method=m, k=int(k), bits=int(8 * k), n_seeds=len(g))
        for col in cols:
            v = g[col].dropna().to_numpy() if col in g else np.array([])
            r[col] = v.mean() if len(v) else np.nan
            r[col + "_sd"] = v.std(ddof=1) if len(v) > 1 else np.nan
            r[col + "_ci95"] = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else np.nan
        rows.append(r)
    s = pd.DataFrame(rows).sort_values(["k", "method"])
    s.to_csv(os.path.join(a.out, f"summary_{track}.csv"), index=False)
    return s


def cmd_report(a):
    """
    Comparisons are made on VALIDATION episodes: the best deployable linear method at each budget is
    chosen by its validation score. Test scores are then reported for the chosen method, the VAE and
    the PLS reference, untouched by any choice.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sums = {t: summarise(t, a) for t in ("ch", "map")}
    gaps = []
    for t, s in sums.items():
        if s is None:
            continue
        for metric in PRIMARY[t][:2]:
            for k in sorted(s.k.unique()):
                sub = s[s.k == k]
                lin = sub[sub.method.isin(DEPLOYABLE[t]) & sub[f"val_{metric}"].notna()]
                vae = sub[sub.method == "vae"]
                if lin.empty or vae.empty:
                    continue
                b = lin.loc[lin[f"val_{metric}"].idxmax()]
                v = vae.iloc[0]
                pls = sub[sub.method == "pls"]
                gaps.append(dict(track=t, metric=metric, k=k, bits=8 * k, best_linear=b.method,
                                 val_linear=b[f"val_{metric}"], val_vae=v[f"val_{metric}"],
                                 val_vae_ci95=v[f"val_{metric}_ci95"], val_gap=b[f"val_{metric}"] - v[f"val_{metric}"],
                                 test_linear=b[f"test_{metric}"], test_vae=v[f"test_{metric}"],
                                 test_gap=b[f"test_{metric}"] - v[f"test_{metric}"],
                                 val_pls_reference=float(pls[f"val_{metric}"].iloc[0]) if len(pls) else np.nan,
                                 vae_seeds=int(v.n_seeds), linear_lipschitz_rel=b.lipschitz_rel,
                                 vae_lipschitz_rel_lb=v.lipschitz_rel))
    if sums.get("ch") is not None:
        runs = pd.read_csv(os.path.join(a.out, "runs_ch.csv"))
        for n in ("val", "test"):
            pos = runs[f"{n}_onset_pos"].max()
            if pos < 200:
                print(f"WARNING: only {pos} {n} tokens see a CH go out within 60 min; {n}_onset_auc is not "
                      f"reliable. Collect more episodes.")
    g = pd.DataFrame(gaps)
    g.to_csv(os.path.join(a.out, "gaps.csv"), index=False)
    pd.set_option("display.width", 220)
    if len(g):
        print(g.round(3).to_string(index=False))

    colors = {"vae": "#2a78d6", "ae": "#2a78d6", "pca": "#eb6834", "pls": "#1baf7a", "dct": "#1baf7a"}
    panels = [("ch", "val_soc_r2", "CH true charge, linear probe R² (validation)"),
              ("ch", "val_onset_auc", "CH outage onset within 60 min, AUC (validation)"),
              ("map", "val_sun_r2", "Sunlight from map latent, MLP probe R² (validation)"),
              ("map", "val_fvu", "Map reconstruction error, FVU (validation)")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (t, metric, title) in zip(axes.ravel(), panels):
        s = sums.get(t)
        ax.set_title(title, fontsize=10, loc="left")
        if s is None:
            ax.axis("off")
            continue
        raw = s[s.method == "raw"]
        for m in [x for x in s.method.unique() if x != "raw"]:
            r = s[s.method == m].sort_values("bits")
            hi = m in colors
            ax.errorbar(r.bits, r[metric], yerr=r.get(metric + "_ci95"), label=m + (" (privileged)" if m == "pls" else ""),
                        color=colors.get(m, "#a3a8b0"), lw=2 if hi else 1, ls="--" if m in ("ae", "pls") else "-",
                        marker="o", ms=4 if hi else 2.5, capsize=2, zorder=3 if hi else 2)
        if len(raw) and not metric.endswith("fvu"):
            ax.axhline(raw[metric].iloc[0], color="#7a818c", ls=":", lw=1, label="uncompressed")
        ax.set_xscale("log", base=2)
        ticks = sorted(s[s.method != "raw"].bits.unique())
        ax.set_xticks(ticks, [str(int(b)) for b in ticks])
        ax.minorticks_off()
        ax.set_ylim(*({"val_onset_auc": (0.5, 1.01), "val_fvu": (0, 1.05)}.get(metric, (-0.2, 1.0))))
        ax.set_xlabel("bits per CH token" if t == "ch" else "bits per map patch")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "report.png"), dpi=150)
    print(f"wrote {a.out}/summary_ch.csv, summary_map.csv, gaps.csv, report.png")


# =====================================================================================
# CLI
# =====================================================================================
def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("collect", "sufficiency", "run", "report", "all"))
    ap.add_argument("--out", default="vae_test_out")
    ap.add_argument("--quick", action="store_true", help="tiny run of the whole pipeline to check it works")
    w = ap.add_argument_group("world")
    w.add_argument("--episodes", type=int, default=120,
                   help="episodes to start with (each one a new world); 'all' adds more until the VAE is "
                        "fully trained (see --max-episodes)")
    w.add_argument("--max-episodes", type=int, default=960, help="ceiling for the sufficiency loop")
    w.add_argument("--area-km2", type=float, default=36.0, help="farm area; 36 km^2 chosen by the harshness sweep")
    w.add_argument("--sensors", type=int, default=200)
    w.add_argument("--uavs", type=int, default=6)
    w.add_argument("--ch-select", choices=("silhouette", "coverage"), default="silhouette",
                   help="how K-means picks K: best silhouette in [--min-ch, --max-ch], or the smallest K "
                        "putting every sensor within 500 m of a CH (gives ~75-80 CHs at 64 km^2)")
    w.add_argument("--min-ch", type=int, default=4, help="fewest CHs; the floor is also the number of UAVs")
    w.add_argument("--max-ch", type=int, default=10, help="most CHs a site would buy")
    w.add_argument("--depart-charge", type=float, default=80.0,
                   help="UAV charge (%%) assumed when it leaves a CH, for the flight check")
    w.add_argument("--uav-reserve", type=float, default=20.0, help="charge (%%) the UAV must land with")
    w.add_argument("--flight-margin", type=float, default=1.25, help="leg margin of the dynamic charging rule")
    w.add_argument("--num-ch", type=int, default=None, help="fixed CH count (overrides --ch-select)")
    w.add_argument("--policy-cycle", nargs="+", choices=("greedy", "random", "rotation"),
                   default=["greedy", "random", "rotation"],
                   help="UAV visitation policy per episode, cycled: greedy (AoI), random sequence, rotation")
    w.add_argument("--ch-start-soc", type=float, nargs=2, default=(15.0, 70.0), help="hidden start charge, %%")
    w.add_argument("--sensor-start-soc", type=float, nargs=2, default=(10.0, 60.0))
    w.add_argument("--uav-start-soc", type=float, nargs=2, default=(50.0, 100.0))
    w.add_argument("--layout", choices=("farm", "open"), default="farm",
                   help="farm: crop fields (no trees) on a grid, hedgerows, woodlots, a farmstead; sensors and "
                        "UGVs on open field ground. open: the original uniform layout")
    w.add_argument("--field-m", type=float, nargs=2, default=(400.0, 800.0), help="field side range (m)")
    w.add_argument("--woodlot-frac", type=float, nargs=2, default=(0.1, 0.3), help="share of blocks that are woodlots")
    w.add_argument("--hedgerow-m", type=float, default=15.0)
    w.add_argument("--sensors-per-field", type=int, nargs=2, default=(3, 8), help="sensors per instrumented field")
    w.add_argument("--base-station", action=argparse.BooleanOptionalAction, default=True,
                   help="grid-powered base station at the farmstead (farm layout)")
    w.add_argument("--bs-exclusion-m", type=float, default=1500.0, help="UGVs park at least this far from the base")
    w.add_argument("--sweep", choices=("cached", "christofides"), default="cached",
                   help="sensor tours: cached nearest-neighbour + 2-opt order, skipping fresh sensors and "
                        "truncated to what the battery allows (default), or the paper's Christofides split")
    w.add_argument("--tour-fresh-min", type=float, default=30.0, help="skip sensors refreshed this recently")
    w.add_argument("--charger-divert", action=argparse.BooleanOptionalAction, default=True,
                   help="UAVs divert to the best reachable charger when their CH cannot supply")
    w.add_argument("--terrain", choices=("fixed", "stretched"), default="fixed")
    w.add_argument("--terrain-feature-m", type=float, default=12.5, help="LTAC: 800 m / 64 noise cells")
    w.add_argument("--tree-cover", type=float, nargs=2, default=(0.15, 0.50), help="uniform range per world")
    w.add_argument("--stand-size-m", type=float, nargs=2, default=(30.0, 90.0), help="uniform range per world")
    w.add_argument("--random-patches", type=int, default=300, help="extra map patches per world")
    w.add_argument("--seed-base", type=int, default=1000)
    w.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    w.add_argument("--verbose", action="store_true", help="show the simulator's own prints")
    d = ap.add_argument_group("data")
    d.add_argument("--val-frac", type=float, default=0.15)
    d.add_argument("--test-frac", type=float, default=0.25)
    d.add_argument("--history", type=int, default=8, help="frames per CH token")
    d.add_argument("--history-stride", type=int, default=15, help="minutes between frames")
    d.add_argument("--token-every", type=int, default=5, help="use every n-th minute")
    d.add_argument("--z-clip", type=float, default=8.0, help="clip standardised CH-token features")
    d.add_argument("--max-train", type=int, default=200_000, help="cap on training samples per track")
    d.add_argument("--max-eval", type=int, default=150_000, help="cap on val/test samples per track")
    d.add_argument("--min-onset", type=int, default=500,
                   help="outage-onset samples needed in each of val and test before running the grid")
    d.add_argument("--max-probe", type=int, default=20_000, help="probe training subsample")
    f = ap.add_argument_group("sufficiency (is there enough training data for the VAE baseline?)")
    f.add_argument("--suff-fracs", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    f.add_argument("--suff-seeds", type=int, default=3)
    f.add_argument("--suff-k-ch", type=int, default=8)
    f.add_argument("--suff-k-map", type=int, default=32)
    f.add_argument("--suff-tol", type=float, default=0.02, help="relative val-FVU gain counted as converged")
    f.add_argument("--suff-probe-tol", type=float, default=0.01, help="val probe gain counted as converged")
    r = ap.add_argument_group("grid")
    r.add_argument("--tracks", nargs="+", choices=("ch", "map"), default=["ch", "map"])
    r.add_argument("--seeds", type=int, default=9, help="seeds for vae, ae and rp (minimum 9 for the paper)")
    r.add_argument("--methods", nargs="+", default=list(dict.fromkeys(CH_METHODS + MAP_METHODS + ("ae", "vae"))))
    r.add_argument("--ch-budgets", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    r.add_argument("--map-budgets", type=int, nargs="+", default=[8, 32, 128, 512])
    r.add_argument("--device", default="auto")
    r.add_argument("--max-epochs", type=int, default=100)
    r.add_argument("--patience", type=int, default=8, help="early stopping on validation reconstruction")
    r.add_argument("--batch", type=int, default=256)
    r.add_argument("--lr", type=float, default=1e-3)
    r.add_argument("--beta", type=float, default=1.0, help="KL weight")
    r.add_argument("--kl-tol", type=float, default=0.5, help="free nats per latent dimension (World Models)")
    a = ap.parse_args(argv)
    if a.quick:
        a.out = a.out if a.out != "vae_test_out" else "vae_test_quick"
        a.episodes, a.max_episodes, a.random_patches, a.seeds, a.suff_seeds = 12, 24, 60, 2, 1
        a.suff_fracs = [0.5, 1.0]
        a.ch_budgets, a.map_budgets = [4, 16], [8, 32]
        a.max_train, a.max_eval, a.max_probe, a.max_epochs, a.patience = 20_000, 5_000, 5_000, 5, 2
    return a


def main(argv=None):
    a = parse(argv)
    if a.seeds < 9 and not a.quick:
        print(f"note: --seeds {a.seeds} is below the 9 the study calls for", flush=True)
    if a.command in ("collect", "all"):
        cmd_collect(a)
    if a.command == "sufficiency":
        cmd_sufficiency(a)
    if a.command == "all":
        # Add episodes until there are enough outage samples AND the VAE baseline stops improving on
        # validation (or --max-episodes is hit)
        while True:
            outs_ok = outages_enough(a)
            vae_ok = cmd_sufficiency(a) if outs_ok or a.episodes >= a.max_episodes else False
            if outs_ok and vae_ok:
                break
            if a.episodes >= a.max_episodes:
                print(f"WARNING: reached --max-episodes {a.max_episodes} with "
                      f"{'too few outage samples' if not outs_ok else 'the VAE still improving'}; "
                      f"results may be unreliable", flush=True)
                break
            a.episodes = min(2 * a.episodes, a.max_episodes)
            print(f"collecting more: up to {a.episodes} episodes", flush=True)
            cmd_collect(a)
    if a.command in ("run", "all"):
        cmd_run(a)
    if a.command in ("report", "all"):
        cmd_report(a)


if __name__ == "__main__":
    main()
