"""
File: ablation.py
Description: Single source of truth for which strategy each component of the
UAV-IoT simulation uses. run.py fills this in from the command line via
configure(args); devices_iot, devices_uav and model_utils read it.

Every default reproduces the behaviour of the code as it stood before the
ablation flags were added, so a run with no new flags matches earlier results.
"""
import math
from dataclasses import dataclass

# ---- Choices for each ablation axis -------------------------------------
TARGET_AGENTS = ("ddaqn", "ddqn", "ppo", "ql", "gann", "random")
CHARGING_POLICIES = ("model", "dynamic", "static")
SENSOR_SWEEPS = ("christofides", "legacy", "none", "cached")
DECISION_STATES = ("projected", "raw")
ENERGY_REWARDS = ("paper", "overflow", "level")

@dataclass
class AblationConfig:
    target_agent: str = "ddaqn"
    charging: str = "model"
    sensor_sweep: str = "christofides"
    decision_state: str = "projected"
    energy_reward: str = "paper"
    # "cached" sweep: each CH orders its sensors once (nearest-neighbour + 2-opt). A visit flies that
    # order, skipping sensors refreshed in the last tour_fresh_min minutes, truncated to the longest
    # prefix whose sortie still leaves the reserve plus the trip to a guaranteed charger.
    tour_fresh_min: float = 30.0
    tour_margin: float = 1.1
    # Charger diversion (off = paper behaviour): a UAV whose docked CH has supplied no charge for
    # divert_after minutes, or whose next leg would breach the reserve while its CH cannot supply,
    # flies to the best reachable charger: rating = perceived usable charge (base station = 1).
    charger_divert: bool = False
    divert_after: int = 2
    divert_min_soc: float = 30.0


CFG = AblationConfig()


def configure(args) -> AblationConfig:
    """Copy the relevant argparse fields into the shared config."""
    CFG.target_agent = args.model
    CFG.charging = args.charging
    CFG.sensor_sweep = args.sensor_sweep
    CFG.decision_state = args.decision_state
    CFG.energy_reward = args.energy_reward
    return CFG


# ---- Battery helpers ------------------------------------------------------
# Every device stores its battery as a percentage (0-100); state[0][2] is the UAV's.
# These constants mirror QuadUAV so the cluster heads can reason about the UAV's battery.
UAV_STEP_CHARGE = 100 / ((1 / 3) * 60)   # QuadUAV.step_charge: 5 % per minute (20 min full charge)
UAV_FLIGHT_HOURS = 0.75                   # QuadUAV.flight_discharge: 45 min of flight
UAV_SPEED = 15                            # QuadUAV.maxSpd, m/s


def batt_full() -> float:
    """Value of state[0][2] that means a full battery."""
    return 100.0


def charge_projection(p_count: float, battery: float) -> float:
    """
    Battery gained over p_count minutes docked, following QuadUAV.receive_energy's curve:
    2x step_charge (10 %/min) below 80 %, 0.5x (2.5 %/min) above, capped at 100 %.
    Assumes the cluster head can supply full charge the whole time.
    """
    b = battery
    for _ in range(max(0, int(p_count))):
        b = min(100.0, b + (2.0 if b < 80 else 0.5) * UAV_STEP_CHARGE)
    return b - battery


def tour_discharge(dist: float, speed: float = UAV_SPEED) -> float:
    """Battery (percentage points) a sensor tour of length dist costs in flight time."""
    return (dist / speed) * 100.0 / (UAV_FLIGHT_HOURS * 3600)


def flight_cost_per_s(flight_discharge_h: float = UAV_FLIGHT_HOURS) -> float:
    """Battery (percentage points) spent per second of flight."""
    return 100.0 / (flight_discharge_h * 3600)


STEP_METRES = UAV_SPEED * 60               # one simulation minute of flight


def leg_minutes(dist: float) -> int:
    """Minutes a leg takes in the simulator: every stop costs at least one 1-minute step."""
    return max(1, math.ceil(dist / STEP_METRES))


def minutes_pct(minutes: float) -> float:
    """Battery (percentage points) for that many minutes airborne (flight and hover cost the same)."""
    return minutes * 60.0 * flight_cost_per_s()


def charger_rating(ch) -> float:
    """Perceived usable charge of a charger, 0..1 above the 10 % reserve; the base station is always 1."""
    if getattr(ch, "is_base", False):
        return 1.0
    return float(min(max((ch.perceived_soc - 10.0) / 90.0, 0.0), 1.0))
