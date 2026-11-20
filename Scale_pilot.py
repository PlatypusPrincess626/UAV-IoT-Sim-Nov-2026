"""
File: Scale_pilot.py
Description: Pilot for scaling the farm (Phase 2 of the scaling game plan). Before collecting full data sets at a
new farm size, run a few worlds per setting and measure whether the setting is viable and what a full run costs.

Settings (every combination is run; 36 km^2 is the anchor, where both fleet rules give the same 6 UAVs)
  --areas 36 120 400       farm area in km^2 (36 ~ 8,900 acres, 120 ~ 30,000, 400 ~ 100,000)
  --fleets density sqrt    density: sensors, UAVs and CH cap all grow with the area (6 / 20 / 67 UAVs);
                           sqrt: sensors and CH cap grow with the area, the fleet with its square root (6 / 11 / 20)
  --docks 0 2              grid-powered docks at outbuildings besides the farmhouse base (0 = farmhouse only)
Each world is built and run exactly as VAE_test.py collects an episode (same world settings as a new folder:
UAV-local knowledge, visit credit, supply margin 5), with the --policy behaviour policy (greedy by default), then
measured. Worlds use seeds 500000 + i, disjoint from the collected data and the controller's worlds, and the
same seeds in every setting.

What it reports per world (pilot_runs.csv) and per setting (pilot_summary.csv, mean and 95 % CI over worlds)
  viability   deaths (UAVs lost), ugv_out_min (UGV-minutes at or below the outage level), ugv_out_per_ugv,
              sensor_out (share of sensor-minutes without power), peak_aoi / avg_aoi (fleet view, minutes),
              flight_ok (a CH layout within [min K, max K] passed the flight check; 0 = K had to grow past
              the cap), farthest_km and round_trip_pct (farthest sensor from its CH and the round trip it costs),
              base_km_mean / base_km_max (UGV to the nearest guaranteed charger)
  knowledge   heard_60 (share of UAV-UGV pairs heard from, directly or relayed, within the last 60 min),
              heard_end (share heard from at least once by the end of the day), links_per_uav_h (PLoRa
              exchanges per UAV per hour)
  compute     build_s, episode_s, extras_s (map patches and sunlight), pairs (UAVs x chargers),
              file_mb (the episode file as collect writes it, compressed), peak_rss_mb (memory of the worker)
The summary also extrapolates a full collection: hours for --plan-worlds worlds with --workers workers, and disk.

Run
  python Scale_pilot.py --out scale_pilot --workers 16                       # all 36 / 120 / 400 settings, 24 worlds
  python Scale_pilot.py --out scale_pilot --areas 400 --fleets sqrt --worlds 8   # a quick look at one setting
  python Scale_pilot.py --out scale_pilot --report                           # summary only
Finished worlds are kept in pilot_runs.csv and skipped when rerun.
"""
import argparse
import contextlib
import io
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import VAE_test as V

WORLD_BASE = 500_000
KEYS = ["area_km2", "fleet", "docks"]


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="scale_pilot")
    ap.add_argument("--areas", type=float, nargs="+", default=[36.0, 120.0, 400.0])
    ap.add_argument("--fleets", nargs="+", choices=("density", "sqrt"), default=["density", "sqrt"])
    ap.add_argument("--docks", type=int, nargs="+", default=[0, 2])
    ap.add_argument("--worlds", type=int, default=24)
    ap.add_argument("--policy", choices=("greedy", "random", "rotation"), default="greedy")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--plan-worlds", type=int, default=240, help="worlds of a full collection, for the estimates")
    ap.add_argument("--report", action="store_true", help="only rebuild the summary")
    a, rest = ap.parse_known_args(argv)
    a.world_args = rest                                    # any other VAE_test.py world option, passed through
    return a


def settings(a):
    out = []
    for area in a.areas:
        for fleet in a.fleets:
            for docks in a.docks:
                if area == 36.0 and fleet != a.fleets[0]:
                    continue                               # the anchor: both rules give 6 UAVs
                out.append((area, fleet, docks))
    return out


def world_args(area, fleet, docks, extra):
    a = V.parse(["collect", "--out", os.path.join("__pilot_no_folder__"), "--area-km2", str(area), "--scale", fleet,
                 "--satellite-docks", str(docks)] + list(extra))
    a.verbose = False
    return a


def run_one(job):
    area, fleet, docks, w, policy, extra = job
    import resource
    a = world_args(area, fleet, docks, extra)
    seed = WORLD_BASE + w
    tm = {}
    with contextlib.redirect_stdout(io.StringIO()):
        d = V.make_episode(seed, a, policy, tm)
    t0 = time.time()
    buf = io.BytesIO()
    np.savez_compressed(buf, **d)
    save_s = time.time() - t0
    env, info = tm["env"], tm["info"]
    T = V.EPISODE_STEPS
    base = d["is_base"]
    ugv = ~base
    U = d["uav_alive"].shape[1]
    sel = env.ch_selection or {}
    tried = sel.get("tried") or []
    flight_ok = any(ok for K, sc, ok, inf in tried if sc is not None)
    since = d["uv_obs"][:, :, ugv, 1] * T                    # minutes since the newest reading, per UAV and UGV
    row = dict(area_km2=area, fleet=fleet, docks=docks, world=w, policy=policy,
               sensors=a.sensors, uavs=U, ugvs=int(ugv.sum()), bases=int(base.sum()), pairs=int(U * len(base)),
               deaths=int(U - d["uav_alive"][-1].sum()),
               ugv_out_min=float(d["ch_out"][:, ugv].sum()),
               ugv_out_per_ugv=float(d["ch_out"][:, ugv].sum() / max(ugv.sum(), 1)),
               sensor_out=float(d["sens_frac"].mean()),
               peak_aoi=float((d["ch_obs"][:, :, 3] * T).max(1).mean()),
               avg_aoi=float((d["ch_obs"][:, :, 4] * T).mean()),
               flight_ok=int(flight_ok), farthest_km=float(info["farthest_m"]) / 1000,
               round_trip_pct=float(info["round_trip_pct"]),
               base_km_mean=float(d["charger_m"][ugv].mean() / 1000), base_km_max=float(d["charger_m"][ugv].max() / 1000),
               heard_60=float((since <= 60).mean()), heard_end=float((since[-1] < T).mean()),
               links_per_uav_h=float(d["air_links"]) / U / (T / 60),
               build_s=tm["build_s"], episode_s=tm["episode_s"], extras_s=tm["extras_s"], save_s=save_s,
               file_mb=buf.tell() / 1e6, raw_mb=sum(v.nbytes for v in d.values() if isinstance(v, np.ndarray)) / 1e6,
               peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    return row


def ci95(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return 1.96 * x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else float("nan")


def report(a):
    path = os.path.join(a.out, "pilot_runs.csv")
    if not os.path.exists(path):
        raise SystemExit(f"no {path} yet")
    d = pd.read_csv(path)
    cols = ["uavs", "ugvs", "bases", "pairs", "deaths", "ugv_out_min", "ugv_out_per_ugv", "sensor_out", "peak_aoi",
            "avg_aoi", "flight_ok", "farthest_km", "round_trip_pct", "base_km_mean", "base_km_max", "heard_60",
            "heard_end", "links_per_uav_h", "build_s", "episode_s", "extras_s", "save_s", "file_mb", "peak_rss_mb"]
    rows = []
    for key, g in d.groupby(KEYS, sort=False):
        r = dict(zip(KEYS, key), worlds=len(g), acres=round(key[0] * 247.105))
        for c in cols:
            r[c], r[c + "_ci95"] = g[c].mean(), ci95(g[c])
        per_world_s = (g.build_s + g.episode_s + g.extras_s + g.save_s).mean()
        r["collect_h"] = per_world_s * a.plan_worlds / max(a.workers, 1) / 3600
        r["disk_gb"] = g.file_mb.mean() * a.plan_worlds / 1000
        r["ram_gb_all_workers"] = g.peak_rss_mb.max() * a.workers / 1024
        rows.append(r)
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(a.out, "pilot_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    show = ["area_km2", "acres", "fleet", "docks", "worlds", "uavs", "ugvs", "bases", "deaths", "ugv_out_per_ugv",
            "peak_aoi", "avg_aoi", "flight_ok", "base_km_max", "heard_60", "episode_s", "file_mb", "peak_rss_mb",
            "collect_h", "disk_gb", "ram_gb_all_workers"]
    print(s[show].round(2).to_string(index=False))
    print(f"\ncollect_h / disk_gb: a {a.plan_worlds}-world collection with {a.workers} workers on this machine; "
          f"ram_gb_all_workers: worst worker x {a.workers}", flush=True)


def main(argv=None):
    a = parse(argv)
    os.makedirs(a.out, exist_ok=True)
    if a.report:
        report(a)
        return
    path = os.path.join(a.out, "pilot_runs.csv")
    have = set()
    if os.path.exists(path):
        d = pd.read_csv(path)
        have = set(zip(d.area_km2, d.fleet, d.docks, d.world))
    jobs = [(area, fleet, docks, w, a.policy, a.world_args) for area, fleet, docks in settings(a)
            for w in range(a.worlds) if (area, fleet, docks, w) not in have]
    jobs.sort(key=lambda j: -j[0])                         # largest farms first: they set the wall time
    print(f"pilot: {len(jobs)} worlds to run over {len(settings(a))} settings with {a.workers} worker(s)", flush=True)
    import multiprocessing as mp
    kw = dict(mp_context=mp.get_context("spawn"))
    if sys.version_info >= (3, 11):
        kw["max_tasks_per_child"] = 1                       # a fresh process per world: peak memory per world
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, **kw) as ex:
        for n, row in enumerate(ex.map(run_one, jobs), 1):
            pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
            print(f"[{n}/{len(jobs)}] {row['area_km2']:g} km2 {row['fleet']} docks {row['docks']} world {row['world']}: "
                  f"{row['uavs']} UAVs {row['ugvs']} UGVs  deaths {row['deaths']}  UGV-out {row['ugv_out_per_ugv']:.0f} "
                  f"min/UGV  peak AoI {row['peak_aoi']:.0f}  heard<60min {row['heard_60']:.2f}  "
                  f"episode {row['episode_s']:.0f}s  file {row['file_mb']:.0f} MB  RAM {row['peak_rss_mb']:.0f} MB  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    report(a)


if __name__ == "__main__":
    main()
