"""
File: environments.py
Author: Mason Conkel
Creation Date: 2026-02-10
Description: This script establishes the environment

Landscape and solar model (after the LTAC environment's terrain/foliage generator and ray march):
    * One procedurally generated landscape per run: rolling terrain (multi-octave Gaussian-smoothed
      noise; by default at the LTAC map's physical scale, so slopes match the 800 m UGV map at any
      area size, ~0-70 m over 8 km) plus a foliage-height map (tree stands of 8/16/24/32 m trees).
    * Shadows move with the sun. The share of direct sunlight reaching a spot at a given minute is
      found by ray-marching from that spot toward that minute's sun position through terrain and
      canopy (LTAC _compute_obfuscation), so shadows lengthen, shorten and swing with the sun.
    * Power at a spot is  cell_current(t) * (1 - obfuscation(x, y, t)).
    * Nothing is stored per map cell per time. Each device's sunlight over the whole episode is
      ray-marched once (all minutes at once, vectorised) and cached; the cache holds one 720-value
      series per occupied map cell. The map itself is the terrain and foliage fields (about 11 MB).

The previous version stored 48 random shadow maps (48 x 5000 x 5000 float64, about 9.6 GB).
"""
# Import Dependencies
from pvlib import spectrum, solarposition, irradiance, atmosphere
import pandas as pd
from numpy.typing import NDArray
import numpy as np
import random
import time
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter, zoom
import math

# Custom Packages
from sim_mods import devices_iot, devices_uav


# np.trapz was removed in NumPy 2.4 in favour of np.trapezoid
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

###############################################################
# Landscape constants (from the LTAC environment)
###############################################################
MIN_USABLE_ELEVATION = 12.0     # deg; sets how far past the map edge terrain/foliage are generated
MIN_SUN_ELEVATION = 0.5         # deg; below this the sun is treated as set (no direct light)
PATH_CHECKPOINT = 15            # min; time resolution used when planning cluster-head movement
MAX_TERRAIN_HEIGHT = 50.0       # m; relief of one LTAC-sized (800 m) map
# Terrain feature scale. "fixed" (default) draws the hills at the LTAC map's physical size (its 64-cell
# noise grid spread over 800 m = 12.5 m per noise cell) and tiles them across the whole area, so slopes
# stay as steep as on the 800 m UGV map however large the area is (median ~13 deg, p90 ~23 deg at 5 m
# cells). "stretched" is the original behaviour: one 64x64 noise grid stretched over the whole map,
# so a 5 km map is ~6x flatter and an 8 km map ~10x flatter than LTAC (median ~1.6 deg at 8 km).
TERRAIN_MODES = ("fixed", "stretched")
TERRAIN_FEATURE_M = 800.0 / 64  # m per noise cell in "fixed" mode (LTAC: 800 cells of 1 m, 64 noise cells)
TERRAIN_M_PER_UNIT = 297.6      # m per unit of the octave field: median 50 m / range of a 64x64 LTAC field
TERRAIN_RELIEF_CAP = 100.0      # m; "fixed" terrain is clipped here (8 km maps measure 60-70 m total)
FOLIAGE_HEIGHTS = np.array([0, 8, 16, 24, 32], dtype=np.float32)   # m
TREE_HEIGHT_MIX = [0.25, 0.25, 0.20, 0.10]    # relative shares of the 8/16/24/32 m classes (LTAC)
# Defaults calibrated so sunlight over the day, weighted by sun height, matches the old random-shadow
# maps (with the shade fix applied): mean sunlight 0.875, 82 % of it at >= 95 % sun, 14 % at <= 50 %.
# LTAC's own woodland settings are tree_cover=0.8, patch_sigma_m=5.
TREE_COVER = 0.215              # share of the map with trees
FOLIAGE_PATCH_SIGMA_M = 50.0    # m; spatial scale of tree stands
MAX_FOLIAGE_HEIGHT = float(FOLIAGE_HEIGHTS.max())

# Canopy attenuation calibrated to a physical transmittance, as in LTAC: a ray crossing the centre
# of a REFERENCE_CANOPY_HEIGHT tree at REFERENCE_ELEVATION_DEG keeps TARGET_CANOPY_TRANSMITTANCE.
# MAX_FOLIAGE_ATTENUATION is the resulting attenuation per metre of horizontal ray travel.
TARGET_CANOPY_TRANSMITTANCE = 0.15
REFERENCE_ELEVATION_DEG = 30.0
REFERENCE_CANOPY_HEIGHT = 15.0
MAX_FOLIAGE_ATTENUATION = (
    -math.log(TARGET_CANOPY_TRANSMITTANCE)
    * 3.0
    * math.tan(math.radians(REFERENCE_ELEVATION_DEG))
    / REFERENCE_CANOPY_HEIGHT
)

MAX_DIST_AMBC = 500   # m; backscatter range with line of sight (PLoRa, spreading factor 8)
NLOS_FACTOR_AMBC = 0.2  # range multiplier without line of sight (500 m -> 100 m)
ANTENNA_HEIGHT = 1.5  # m; sensor and cluster-head antennas above the ground, for line of sight
AMBC_LOS_MODES = ("map", "nlos", "los")

###############################################################
# Episode date (from the LTAC environment)
#
# The summer solstice, with the episode centred on solar noon, so a
# 720-step (12 h) episode is daylight from start to finish. The sun is
# up about 15.5 h at this latitude on this date.
###############################################################
EPISODE_DATE = "2021-06-21"
EPISODE_TZ = "MST"


def episode_times(lat, long, max_num_steps, freq='min', date=None):
    """One timestamp per step, centred on solar noon (transit) of `date` (default EPISODE_DATE) at (lat, long)."""
    date = EPISODE_DATE if date is None else date
    transit = solarposition.sun_rise_set_transit_spa(
        pd.DatetimeIndex([pd.Timestamp(f"{date} 12:00", tz=EPISODE_TZ)]), lat, long
    )["transit"].iloc[0]
    start = (transit - pd.Timedelta(minutes=max_num_steps // 2)).round("min")
    return pd.date_range(start, freq=freq, periods=max_num_steps, tz=EPISODE_TZ)


def dist(pt1: NDArray[np.int32], pt2: NDArray[np.int32]):
    assert pt1.shape == (2,)
    assert pt2.shape == (2,)
    return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


class sim_env:
    def __init__(self, scene, num_sensors, num_uav, num_ch, max_num_steps,
                 map_seed: int = 0, map_res: int = 5, tree_cover: float = TREE_COVER,
                 patch_sigma_m: float = FOLIAGE_PATCH_SIGMA_M, area_km2: float = 25.0,
                 auto_clusterheads: bool = False, ambc_los: str = "map",
                 placement_seed=None, episode_date=None, terrain: str = "fixed",
                 terrain_feature_m: float = TERRAIN_FEATURE_M, ch_range=None, ch_check=None,
                 layout: str = "open", field_m=(400.0, 800.0), woodlot_frac: float = 0.2,
                 hedgerow_m: float = 15.0, base_station: bool = False, bs_exclusion_m: float = 1500.0,
                 sensors_per_field=(3, 8)):
        """
        area_km2           side of the square monitoring area is sqrt(area_km2) km (9 -> 3 km, 25 -> 5 km)
        num_ch             number of cluster heads K-means places (the paper uses 5)
        auto_clusterheads  instead, grow K until every sensor is within line-of-sight backscatter
                           range (500 m) of its cluster head
        ambc_los           sensor-to-CH line of sight: "map" (from the terrain and trees), "nlos"
                           (never: every ground link has the 100 m range), "los" (always: 500 m)
        placement_seed     seed for sensor/UAV placement (None keeps the original fixed seed)
        episode_date       YYYY-MM-DD of the (daylight-only) episode (None keeps EPISODE_DATE)
        terrain            "fixed" (LTAC slopes at any map size) or "stretched" (original, flattens
                           as the area grows)
        terrain_feature_m  metres per noise cell in "fixed" mode; larger values give gentler hills
        layout             "open" (original) or "farm": a grid of crop fields (side drawn from field_m,
                           no trees inside) with hedgerows (hedgerow_m wide, with gaps) along field
                           borders and a woodlot_frac share of blocks left as woodland (LTAC woodland
                           density); one field block near a map edge is the farmstead. Sensors are
                           placed only on open field ground.
        base_station       farm only: a grid-powered BaseStation at the farmstead. Sensors within
                           backscatter range (500 m) report to it as their cluster head; UAVs start the
                           day there and can always charge there. UGV cluster heads are placed at least
                           bs_exclusion_m from it and on open field ground (nearest such spot to their
                           K-means centroid).
        sensors_per_field  farm only: sensors go into instrumented fields, randint(lo, hi) per field,
                           the farmstead's field first, then fields picked at random.
        ch_range           (min_K, max_K): K-means with K chosen by the best silhouette score in that
                           range (an affordable CH count that still follows the sensor layout).
                           Overrides num_ch and auto_clusterheads when given.
        ch_check           optional callable(sensor_pts, centroids, labels) -> (ok, info). With ch_range,
                           only K values passing it are eligible; if none in the range passes, K keeps
                           growing past max_K until one does (with a printed warning).
        """
        self.ch_check = ch_check
        self.ch_selection = None
        if layout not in ("open", "farm"):
            raise ValueError("layout must be 'open' or 'farm'")
        if base_station and layout != "farm":
            raise ValueError("the base station sits at the farmstead: use layout='farm'")
        self.layout, self.field_m, self.woodlot_frac = layout, tuple(field_m), float(woodlot_frac)
        self.hedgerow_m, self.base_station, self.bs_exclusion_m = float(hedgerow_m), bool(base_station), float(bs_exclusion_m)
        self.bs_xy = None
        self.sensors_per_field = (int(sensors_per_field[0]), int(sensors_per_field[1]))
        self.ch_range = None if ch_range is None else (int(ch_range[0]), int(ch_range[1]))
        if self.ch_range is not None and not 2 <= self.ch_range[0] <= self.ch_range[1] < num_sensors:
            raise ValueError("ch_range must satisfy 2 <= min_K <= max_K < num_sensors")
        if terrain not in TERRAIN_MODES:
            raise ValueError(f"terrain must be one of {TERRAIN_MODES}")
        self.terrain_mode = terrain
        self.terrain_feature_m = float(terrain_feature_m)
        if area_km2 <= 0:
            raise ValueError("area_km2 must be positive")
        if ch_range is None and not auto_clusterheads and not 1 <= num_ch <= num_sensors:
            raise ValueError("num_ch must be between 1 and the number of sensors")
        if ambc_los not in AMBC_LOS_MODES:
            raise ValueError(f"ambc_los must be one of {AMBC_LOS_MODES}")
        self.area_km2 = float(area_km2)
        self.auto_clusterheads = bool(auto_clusterheads)
        self.ambc_los = ambc_los

        self.uav_table = None
        self.ch_table = None
        self.sensor_table = None
        self.total_sensors = num_sensors
        self.total_uav = num_uav
        self.total_clusterheads = num_ch
        self.max_num_steps = max_num_steps

        self.sensor_pts = []
        self.ch_pts = []

        if (scene == "test"):
            # Test scene will be Yellowstone National Park
            self.lat_center = 44.424  # Latitude
            self.long_center = -110.589  # Longitude
            self.stp = 0.000009  # 1 degree lat/long is ~111km
            self.pressure = 101253  # Sea Level is 1013.25 mb, Average Pressure in Yellowstone is +4.09mb
            self.water_vapor_content = 0.35  # Roughly 0.35 cm in Yellowstone
            self.tau500 = 0.75  # Aerosol Turbidity 500nm
            self.ozone = 0.23  # Ozone in atm-cm
            self.albedo = 0.2  # Bare Ground and Grassy
            self.dim = int(round(math.sqrt(self.area_km2) * 1_000))  # Map side in metres
            self.stepSize = 'min'  # Frequency of time steps
            self.episode_date = EPISODE_DATE if episode_date is None else episode_date
            self.times = episode_times(self.lat_center, self.long_center, self.max_num_steps, self.stepSize,
                                       date=self.episode_date)
            # Device placement seed: the original fixed seed unless one is given (scaled worlds vary it)
            random.seed('2021-01-01 8:00' if placement_seed is None else placement_seed)

        # Fix the map size before anything is built on it (the old code built the shadow maps at 5000
        # and then bumped dim to 5001, so every later lookup was misaligned).
        if self.dim % 2 == 0:
            self.dim += 1

        self._cell_cache = {}
        self.map_seed = map_seed
        self.map_res = int(map_res)
        # Share of the map with trees; the rest is open ground. The trees keep LTAC's height mix.
        self.patch_sigma_m = patch_sigma_m
        mix = np.array(TREE_HEIGHT_MIX) / sum(TREE_HEIGHT_MIX)
        self.foliage_probs = [1.0 - tree_cover] + list(tree_cover * mix)
        self.make_landscape()

        self.envMap = self.place_devices()

    # ------------------------------------------------------------------
    # Coordinates
    # ------------------------------------------------------------------
    def latlong(self, x, y):
        """
        Geographic position of map point (x, y): x runs east, y runs north, and the map is centred on
        (lat_center, long_center). A degree of longitude is shorter than a degree of latitude by
        cos(latitude). (The old code passed the latitude-based value in the longitude slot and offset
        by -dim instead of -dim/2, so pvlib was asked for the sun at latitude -110.6 deg.)
        """
        lat = self.lat_center + self.stp * (y - self.dim / 2)
        long = self.long_center + self.stp * (x - self.dim / 2) / math.cos(math.radians(self.lat_center))
        return lat, long

    # ------------------------------------------------------------------
    # Landscape
    # ------------------------------------------------------------------
    def make_landscape(self):
        """Generates terrain and foliage once per run, plus the sun's position at every step."""
        t0 = time.time()
        rng = np.random.default_rng(self.map_seed)   # own stream: never disturbs `random` / np.random
        res = self.map_res

        self.map_cells = math.ceil(self.dim / res)
        relief_bound = MAX_TERRAIN_HEIGHT if self.terrain_mode == "stretched" else TERRAIN_RELIEF_CAP
        pad_m = (relief_bound + MAX_FOLIAGE_HEIGHT) / math.tan(math.radians(MIN_USABLE_ELEVATION))
        self.pad_cells = math.ceil(pad_m / res)
        padded = self.map_cells + 2 * self.pad_cells

        print("Making Happy Trees")
        if self.terrain_mode == "fixed":
            self.topo_mask = self._generate_terrain_fixed((padded, padded), rng).astype(np.float32)
        else:
            self.topo_mask = self._generate_terrain((padded, padded), rng).astype(np.float32)
        self.foliage_mask = self._generate_foliage((padded, padded), rng)
        if self.layout == "farm":
            self._make_farm(rng, padded)
        self._max_canopy_top = float((self.topo_mask + self.foliage_mask).max())

        # Sun position for every step (scene centre; over 5 km the difference is negligible)
        solpos = solarposition.get_solarposition(self.times, self.lat_center, self.long_center)
        self.sun_azimuth = solpos['azimuth'].to_numpy()
        self.sun_zenith = solpos['apparent_zenith'].to_numpy()
        self._sun_series = {}     # (cell_x, cell_y) -> transmittance at every step
        print(f"Landscape ready in {time.time() - t0:.1f}s: {self.map_cells}x{self.map_cells} cells of {res} m, "
              f"{100 * (self.foliage_mask > 0).mean():.0f}% wooded")

    def _generate_terrain(self, shape, rng, base_res=64, num_octaves=5, persistence=0.55):
        """
        Rolling terrain from multiple octaves of isotropic Gaussian-smoothed noise, generated on a
        small grid and upsampled (LTAC _generate_woodland_terrain). Normalised to 0-50 m.
        """
        result = np.zeros((base_res, base_res))
        amplitude, total_amplitude = 1.0, 0.0
        for octave in range(num_octaves):
            sigma = max(base_res / (4.0 * (2 ** octave)), 1.0)
            result += amplitude * gaussian_filter(rng.standard_normal((base_res, base_res)), sigma=sigma)
            total_amplitude += amplitude
            amplitude *= persistence
        result /= total_amplitude

        data = zoom(result, (shape[0] / base_res, shape[1] / base_res), order=1)
        data = data - data.min()
        if data.max() > 0:
            data = data / data.max() * MAX_TERRAIN_HEIGHT
        return data

    def _generate_terrain_fixed(self, shape, rng, ref_res=64, num_octaves=5, persistence=0.55):
        """
        Same octave noise as _generate_terrain, but at a fixed physical scale: one noise cell is
        terrain_feature_m metres (LTAC's 12.5 m by default) and the octave widths are LTAC's
        (16, 8, 4, 2, 1 noise cells), so every 800 m window looks like an LTAC map. Heights use LTAC's
        metres-per-unit scaling instead of being normalised per map (normalising a large map to 0-50 m
        would flatten it again), then are shifted to start at 0 m and clipped at TERRAIN_RELIEF_CAP.
        """
        side_m = shape[0] * self.map_res
        nb = max(int(math.ceil(side_m / self.terrain_feature_m)) + 1, 2)
        result = np.zeros((nb, nb))
        amplitude, total_amplitude = 1.0, 0.0
        for octave in range(num_octaves):
            sigma = max(ref_res / (4.0 * (2 ** octave)), 1.0)
            result += amplitude * gaussian_filter(rng.standard_normal((nb, nb)), sigma=sigma)
            total_amplitude += amplitude
            amplitude *= persistence
        result = result / total_amplitude * TERRAIN_M_PER_UNIT
        f = self.terrain_feature_m / self.map_res          # map cells per noise cell
        data = zoom(result, (f, f), order=1)
        data = data[:shape[0], :shape[1]]
        if data.shape != tuple(shape):                     # zoom rounding: pad by edge values
            data = np.pad(data, [(0, shape[0] - data.shape[0]), (0, shape[1] - data.shape[1])], mode="edge")
        data = data - data.min()
        return np.clip(data, 0.0, TERRAIN_RELIEF_CAP)

    def _generate_foliage(self, shape, rng):
        """
        Patchy foliage heights: smoothed noise, rank-binned so each height class covers exactly its
        target share of the map (LTAC reset_foliage).
        """
        sigma = max(self.patch_sigma_m / self.map_res, 0.5)
        field = gaussian_filter(rng.random(shape), sigma=sigma)
        ranks = field.ravel().argsort().argsort().reshape(shape)
        quantiles = ranks / (field.size - 1)
        return FOLIAGE_HEIGHTS[np.digitize(quantiles, np.cumsum(self.foliage_probs)[:-1])]

    def _make_farm(self, rng, n):
        """Field grid, hedgerows, woodlots and farmstead on the padded map (see __init__ 'layout')."""
        from scipy.ndimage import binary_erosion
        res, P = self.map_res, self.pad_cells
        side = float(rng.uniform(*self.field_m))
        off = rng.uniform(0.0, side, 2)
        coord = (np.arange(n) - P) * res + res / 2.0                       # metres, map frame
        bx = np.floor((coord + off[0]) / side).astype(int)
        by = np.floor((coord + off[1]) / side).astype(int)
        nbx, nby = bx.max() - bx.min() + 1, by.max() - by.min() + 1
        wood_block = rng.random((nby, nbx)) < self.woodlot_frac
        # Farmstead: a block whose centre lies inside the map, near an edge
        cxs = (np.arange(nbx) + bx.min()) * side - off[0] + side / 2.0
        cys = (np.arange(nby) + by.min()) * side - off[1] + side / 2.0
        CY, CX = np.meshgrid(cys, cxs, indexing="ij")
        edge = np.minimum.reduce([CX, CY, self.dim - CX, self.dim - CY])
        cand = np.argwhere((edge >= side / 2.0) & (edge <= max(side, 0.15 * self.dim)))
        if len(cand) == 0:
            cand = np.argwhere(edge >= side / 2.0)
        fy, fx = cand[rng.integers(len(cand))]
        wood_block[fy, fx] = False
        self.bs_xy = (float(CX[fy, fx]), float(CY[fy, fx]))
        woodlot = wood_block[(by - by.min())[:, None], (bx - bx.min())[None, :]]
        farmstead = ((by - by.min())[:, None] == fy) & ((bx - bx.min())[None, :] == fx)
        # Hedgerows along block borders, with gaps
        ex = np.minimum((coord + off[0]) % side, side - (coord + off[0]) % side)
        ey = np.minimum((coord + off[1]) % side, side - (coord + off[1]) % side)
        hedge = (np.minimum(ey[:, None], ex[None, :]) < self.hedgerow_m / 2.0) & ~farmstead
        gaps = gaussian_filter(rng.random((n, n)), sigma=max(20.0 / res, 0.5))
        hedge &= gaps > np.quantile(gaps, 0.35)                            # ~65 % of the hedge line planted
        # Woodland density in woodlots (LTAC woodland cover 0.8), field crops have no trees
        mix = np.array(TREE_HEIGHT_MIX) / sum(TREE_HEIGHT_MIX)
        dense = gaussian_filter(rng.random((n, n)), sigma=max(self.patch_sigma_m / res, 0.5))
        q = dense.ravel().argsort().argsort().reshape(n, n) / (dense.size - 1)
        dense = FOLIAGE_HEIGHTS[np.digitize(q, np.cumsum([0.2] + list(0.8 * mix))[:-1])]
        hedge_h = FOLIAGE_HEIGHTS[1 + (rng.random((n, n)) * 3).astype(int)]    # 8, 16 or 24 m
        fol = np.zeros((n, n), dtype=self.foliage_mask.dtype)
        fol[woodlot] = dense[woodlot]
        fol[hedge & ~woodlot] = hedge_h[hedge & ~woodlot]
        self.foliage_mask = fol
        field = ~woodlot & ~hedge
        self.field_mask = binary_erosion(field, iterations=2)               # keep sensors off the hedges
        block = (by - by.min())[:, None] * nbx + (bx - bx.min())[None, :]
        self.field_block = np.where(self.field_mask, block, -1)
        self.farmstead_block = int(fy * nbx + fx)
        self.farm_info = dict(field_side_m=side, woodlot_blocks=float(wood_block.mean()))

    def _ray_march(self, cx, cy, azimuth, zenith):
        """
        Share of direct sunlight reaching the ground for a batch of (cell, sun position) pairs.
        cx, cy are map-cell indices and azimuth, zenith are sun angles in degrees; all are 1-D
        arrays of the same length (one entry per ray), so many points and many minutes are marched
        together.

        Same model as the LTAC environment's _compute_obfuscation. From each cell, step toward the
        sun one cell (map_res metres) at a time:
          * terrain at or above the sunline blocks the ray completely;
          * foliage whose canopy (from 1/3 of tree height to the top, crown radius = height / 4)
            contains the sunline attenuates it, strongest at the crown centre and deeper in the canopy;
          * the ray stops once the sunline clears the tallest canopy, leaves the padded map, or is
            dimmed below 1 %.
        Heights are measured from the cell's own ground level. Attenuation is calibrated per metre,
        so each map_res-metre step compounds it map_res times.

        Unlike LTAC, which treats the sun below 12 deg as fully blocked, rays are marched down to
        MIN_SUN_ELEVATION: the January sun here is below 12 deg for most of the day. Rays that reach
        the edge of the padded map count as unobstructed from there on.
        """
        res, P = self.map_res, self.pad_cells
        topo, fol = self.topo_mask, self.foliage_mask
        H, W = topo.shape

        n = len(cx)
        elevation = 90.0 - np.asarray(zenith, dtype=float)
        up = elevation >= MIN_SUN_ELEVATION
        transmittance = np.where(up, 1.0, 0.0)

        idx = np.flatnonzero(up)                     # rays still being marched
        if len(idx) == 0:
            return transmittance
        ax = np.asarray(cx, dtype=np.int64)[idx] + P
        ay = np.asarray(cy, dtype=np.int64)[idx] + P
        tan_el = np.tan(np.radians(elevation[idx]))
        az_rad = np.radians(90.0 - np.asarray(azimuth, dtype=float)[idx])   # compass -> math angle
        step_x, step_y = np.cos(az_rad), np.sin(az_rad)
        perp_x, perp_y = -step_y, step_x
        observer = topo[ay, ax]
        h_limit = self._max_canopy_top - observer
        max_half_width = math.ceil(MAX_FOLIAGE_HEIGHT / 4.0 / res)
        T = np.ones(len(idx))
        alive = np.ones(len(idx), dtype=bool)

        d = 0
        while True:
            d += 1
            live = np.flatnonzero(alive)
            if len(live) == 0:
                break
            h_min = d * res * tan_el[live]
            cleared = h_min > h_limit[live]          # sunline above everything on the map
            alive[live[cleared]] = False
            live, h_min = live[~cleared], h_min[~cleared]
            if len(live) == 0:
                break

            sunline = observer[live] + h_min
            rx = ax[live] + np.rint(d * step_x[live]).astype(np.int64)
            ry = ay[live] + np.rint(d * step_y[live]).astype(np.int64)
            inside = (rx >= 0) & (rx < W) & (ry >= 0) & (ry < H)
            alive[live[~inside]] = False
            live, sunline, rx, ry = live[inside], sunline[inside], rx[inside], ry[inside]

            ground = topo[ry, rx]
            blocked = ground >= sunline
            T[live[blocked]] = 0.0
            alive[live[blocked]] = False
            keep = ~blocked
            live, sunline, rx, ry, ground = live[keep], sunline[keep], rx[keep], ry[keep], ground[keep]

            height = fol[ry, rx]
            canopy_start = ground + height / 3.0
            canopy_top = ground + height
            in_band = (height > 0) & (sunline >= canopy_start) & (sunline <= canopy_top)
            if not in_band.any():
                continue
            live, sunline, rx, ry = live[in_band], sunline[in_band], rx[in_band], ry[in_band]
            canopy_start, canopy_top, height = canopy_start[in_band], canopy_top[in_band], height[in_band]

            radius = height / 4.0 / res                          # crown radius in cells
            depth_frac = (sunline - canopy_start) / np.maximum(canopy_top - canopy_start, 1e-6)
            half_width = min(int(math.ceil(float(radius.max()))), max_half_width)
            local = np.zeros(len(live))
            for k in range(-half_width, half_width + 1):
                r = abs(k)
                ok = r <= radius
                if not ok.any():
                    continue
                kx = rx + np.rint(k * perp_x[live]).astype(np.int64)
                ky = ry + np.rint(k * perp_y[live]).astype(np.int64)
                ok &= (kx >= 0) & (kx < W) & (ky >= 0) & (ky < H)
                ok &= fol[np.clip(ky, 0, H - 1), np.clip(kx, 0, W - 1)] > 0
                density = np.exp(-2.0 * (r / np.where(radius > 0, radius, 1.0)) ** 2)
                attenuation = 1.0 - (1.0 - density * MAX_FOLIAGE_ATTENUATION * depth_frac) ** res
                local = np.maximum(local, np.where(ok, attenuation, 0.0))

            T[live] *= 1.0 - local
            dark = T[live] < 0.01
            T[live[dark]] = 0.0
            alive[live[dark]] = False

        transmittance[idx] = T
        return transmittance

    def line_of_sight(self, ax, ay, bx, by):
        """
        Whether a ground link from (ax, ay) to (bx, by) is clear: no tree in any map cell along the
        straight line (including the end cells), and no terrain above the line between the two
        antennas (ANTENNA_HEIGHT above the ground at each end). Overridden by ambc_los "nlos"/"los".
        """
        if self.ambc_los == "los":
            return True
        if self.ambc_los == "nlos":
            return False
        length = math.hypot(bx - ax, by - ay)
        t = np.linspace(0.0, 1.0, max(2, int(length / (self.map_res / 2.0)) + 1))
        cx = np.clip(((ax + t * (bx - ax)) // self.map_res).astype(int), 0, self.map_cells - 1) + self.pad_cells
        cy = np.clip(((ay + t * (by - ay)) // self.map_res).astype(int), 0, self.map_cells - 1) + self.pad_cells
        if (self.foliage_mask[cy, cx] > 0).any():
            return False
        ground = self.topo_mask[cy, cx]
        line = (ground[0] + ANTENNA_HEIGHT) + t * ((ground[-1] + ANTENNA_HEIGHT) - (ground[0] + ANTENNA_HEIGHT))
        return not bool((ground[1:-1] > line[1:-1]).any())

    def _cell(self, x, y):
        return (min(max(int(x) // self.map_res, 0), self.map_cells - 1),
                min(max(int(y) // self.map_res, 0), self.map_cells - 1))

    def precompute_sunlight(self, points):
        """Ray-marches every step for all given (x, y) points in one batch and caches the results."""
        cells = sorted({self._cell(x, y) for x, y in points} - set(self._sun_series))
        if not cells:
            return
        steps = len(self.sun_zenith)
        cx = np.repeat([c[0] for c in cells], steps)
        cy = np.repeat([c[1] for c in cells], steps)
        T = self._ray_march(cx, cy, np.tile(self.sun_azimuth, len(cells)), np.tile(self.sun_zenith, len(cells)))
        for i, c in enumerate(cells):
            self._sun_series[c] = T[i * steps:(i + 1) * steps].astype(np.float32)

    # ------------------------------------------------------------------
    # Queries used by the devices
    # ------------------------------------------------------------------
    def get_obfuscation(self, x, y, step):
        """
        Share of direct sunlight blocked at map point (x, y) at a given step: 0 = full sun,
        1 = fully shaded or the sun is down. The first query for a map cell ray-marches the whole
        episode for that cell; later queries are lookups.
        """
        cell = self._cell(x, y)
        series = self._sun_series.get(cell)
        if series is None:
            self.precompute_sunlight([(x, y)])
            series = self._sun_series[cell]
        return 1.0 - float(series[int(step)])

    def plan_legs(self, x, y, r_max, num_legs):
        """
        Where a mobile device should park during the day to harvest the most sun, as (legs, times).

        The episode is split into num_legs equal time windows. For each window, pick the spot within
        r_max of (x, y) that gets the most direct sunlight over that window, weighted by how high the
        sun is (a proxy for irradiance); ties go to the spot nearest the previous leg. Candidates are
        a grid over the disc (about 150 spots), and sunlight is evaluated every PATH_CHECKPOINT minutes.
        times[i] is the step at which leg i ends.
        """
        res = self.map_res
        r_max = max(float(r_max), 0.0)
        spacing = max(res, r_max / 7.0)
        offsets = np.arange(-r_max, r_max + 1e-9, spacing) if r_max > 0 else np.array([0.0])
        ox, oy = np.meshgrid(offsets, offsets)
        keep = ox ** 2 + oy ** 2 <= r_max ** 2
        px = np.clip(np.concatenate([[x], x + ox[keep]]), 0, self.dim - 1)
        py = np.clip(np.concatenate([[y], y + oy[keep]]), 0, self.dim - 1)
        cx = np.clip(px.astype(int) // res, 0, self.map_cells - 1)
        cy = np.clip(py.astype(int) // res, 0, self.map_cells - 1)

        checks = np.arange(0, len(self.sun_zenith), PATH_CHECKPOINT)
        n_c, n_t = len(cx), len(checks)
        T = self._ray_march(np.repeat(cx, n_t), np.repeat(cy, n_t),
                            np.tile(self.sun_azimuth[checks], n_c),
                            np.tile(self.sun_zenith[checks], n_c)).reshape(n_c, n_t)
        weight = np.clip(np.sin(np.radians(90.0 - self.sun_zenith[checks])), 0.0, None)

        steps = len(self.sun_zenith)
        bounds = np.linspace(0, steps, num_legs + 1).round().astype(int)
        legs, times = [], []
        prev = np.array([x, y], dtype=float)
        for i in range(num_legs):
            window = (checks >= bounds[i]) & (checks < bounds[i + 1])
            score = (T[:, window] * weight[window]).sum(axis=1)
            dist2 = (px - prev[0]) ** 2 + (py - prev[1]) ** 2
            best = np.lexsort((dist2, -np.round(score, 6)))[0]
            legs.append([int(px[best]), int(py[best])])
            times.append(int(bounds[i + 1]))
            prev = np.array(legs[-1], dtype=float)
        return legs, times

    def get_spectrum(self, lat, long, tilt, azimuth, step):
        solpos = solarposition.get_solarposition(self.times[step], lat, long)
        aoi = irradiance.aoi(tilt, azimuth, solpos.apparent_zenith, solpos.azimuth)
        relative_airmass = atmosphere.get_relative_airmass(solpos.apparent_zenith, model='kasten1966')
        spectra = spectrum.spectrl2(
            apparent_zenith=solpos.apparent_zenith,
            aoi=aoi,
            surface_tilt=tilt,
            ground_albedo=self.albedo,
            surface_pressure=self.pressure,
            relative_airmass=relative_airmass,
            precipitable_water=self.water_vapor_content,
            ozone=self.ozone,
            aerosol_turbidity_500nm=self.tau500,
        )
        return spectra

    def poa_irradiance(self, lat, long, tilt, azimuth, step):
        """
        Plane-of-array irradiance (W/m^2) on a panel at (lat, long, tilt, azimuth) at a given step,
        before shading: the pvlib spectrum integrated over wavelength. Computed for all steps in one
        vectorised call the first time a panel asks, then cached. 0 when the sun is down.
        """
        key = ("poa", lat, long, tilt, azimuth)
        cached = self._cell_cache.get(key)
        if cached is None:
            spectra = self.get_spectrum(lat, long, tilt, azimuth, slice(None))
            cached = _trapezoid(spectra['poa_global'], spectra['wavelength'], axis=0)
            cached = np.nan_to_num(np.asarray(cached, dtype=float), nan=0.0)
            self._cell_cache[key] = np.maximum(cached, 0.0)
        return float(self._cell_cache[key][step])

    def cell_current(self, lat, long, tilt, azimuth, step, spectral_response):
        """
        Solar-cell current for a panel at (lat, long, tilt, azimuth) at a given step, before shading.
        Computed for all steps in one vectorised pvlib call the first time a panel asks, then cached.
        Night-time steps (sun below the horizon) come back as 0.
        """
        key = (lat, long, tilt, azimuth)
        cached = self._cell_cache.get(key)
        if cached is None:
            spectra = self.get_spectrum(lat, long, tilt, azimuth, slice(None))
            cached = _trapezoid(spectra['poa_global'] * spectral_response, spectra['wavelength'], axis=0)
            cached = np.nan_to_num(np.asarray(cached, dtype=float), nan=0.0)
            self._cell_cache[key] = cached
        return cached[step]

    # ------------------------------------------------------------------
    # Devices
    # ------------------------------------------------------------------
    def _ugv_sites(self, centres):
        """UGV parking spots: with a base station, the nearest open-field spot outside its exclusion zone."""
        centres = np.asarray(centres, float)
        if not self.base_station:
            return centres
        res = self.map_res
        cx = np.clip((centres[:, 0] // res).astype(int), 0, self.map_cells - 1)
        cy = np.clip((centres[:, 1] // res).astype(int), 0, self.map_cells - 1)
        iy, ix = self._ugv_site_idx[0][cy, cx], self._ugv_site_idx[1][cy, cx]
        return np.column_stack([ix * res + res // 2, iy * res + res // 2]).astype(float)

    def _check_layout(self, all_pts, ugv_pts, km, bs_members):
        """Points, centres and labels of the final layout (UGV sites + base station) for ch_check."""
        sites = self._ugv_sites(km.cluster_centers_)
        if not self.base_station:
            return ugv_pts, sites, km.labels_
        pts = np.vstack([ugv_pts, all_pts[bs_members]])
        labels = np.concatenate([km.labels_, np.full(len(bs_members), len(sites))])
        return pts, np.vstack([sites, [self.bs_xy]]), labels

    def place_devices(self) -> bool:
        sensor_pts = np.array([[0, 0]] * self.total_sensors, np.int32)
        sensor_list = []

        print("Placing Sensors")
        P, res = self.pad_cells, self.map_res
        if self.layout == "farm":
            # Instrumented fields: the farmstead's field first, then random fields, randint(lo, hi) each
            inner = self.field_block[P:P + self.map_cells, P:P + self.map_cells]
            ys, xs = np.nonzero(inner >= 0)
            ids = inner[ys, xs]
            order = np.argsort(ids, kind="stable")
            ids, ys, xs = ids[order], ys[order], xs[order]
            uniq, start = np.unique(ids, return_index=True)
            cells = {int(b): (ys[i:j], xs[i:j]) for b, i, j in zip(uniq, start, list(start[1:]) + [len(ids)])}
            fields = [b for b in cells if len(cells[b][0]) >= 20]
            random.shuffle(fields)
            if self.farmstead_block in fields:
                fields.remove(self.farmstead_block)
                fields.insert(0, self.farmstead_block)
            spots, fi = [], 0
            while len(spots) < self.total_sensors:
                b = fields[fi % len(fields)]
                fi += 1
                fy_, fx_ = cells[b]
                for _ in range(random.randint(*self.sensors_per_field)):
                    j = random.randrange(len(fy_))
                    spots.append((int(fx_[j] * res + random.randrange(res)), int(fy_[j] * res + random.randrange(res))))
            spots = spots[:self.total_sensors]
        for sensor in range(self.total_sensors):
            if self.layout == "farm":
                x, y = spots[sensor]
            else:
                position = random.randint(0, self.dim * self.dim - 1)
                x, y = int(position % self.dim), int(position / self.dim)
            lat, long = self.latlong(x, y)
            sensor_pts[sensor] = [x, y]
            sensor_list.append([devices_iot.Sensor(x, y, long, lat, random.randint(1, 3))])

        # Base station: sensors within backscatter range report to it (at least the nearest one)
        bs_members = np.array([], int)
        ugv_idx = np.arange(self.total_sensors)
        if self.base_station:
            d_bs = np.hypot(*(sensor_pts - np.array(self.bs_xy)).T)
            bs_members = np.flatnonzero(d_bs <= MAX_DIST_AMBC)
            if len(bs_members) == 0:
                bs_members = np.array([int(np.argmin(d_bs))])
            ugv_idx = np.setdiff1d(np.arange(self.total_sensors), bs_members)
            # Where a UGV may park: open field ground inside the map, at least bs_exclusion_m away
            from scipy.ndimage import distance_transform_edt
            ok = self.field_mask[P:P + self.map_cells, P:P + self.map_cells].copy()
            cy, cx = np.mgrid[0:self.map_cells, 0:self.map_cells]
            ok &= np.hypot(cx * res + res / 2 - self.bs_xy[0], cy * res + res / 2 - self.bs_xy[1]) >= self.bs_exclusion_m
            if not ok.any():
                raise ValueError("no field ground outside the base-station exclusion zone; shrink bs_exclusion_m")
            _, self._ugv_site_idx = distance_transform_edt(~ok, return_indices=True)
        all_pts = sensor_pts
        sensor_pts = all_pts[ugv_idx]                       # the sensors K-means clusters for the UGVs

        if self.ch_range is not None:
            # K-means with K picked by silhouette within an affordable range. Sensors outside a CH's
            # backscatter range are reached by UAV tours, as in the paper setup.
            from sklearn.metrics import silhouette_score
            best, tried = None, []
            for K in range(self.ch_range[0], self.ch_range[1] + 1):
                km = KMeans(n_clusters=K, random_state=0, n_init=10).fit(sensor_pts)
                ok, info = (True, None) if self.ch_check is None else \
                    self.ch_check(*self._check_layout(all_pts, sensor_pts, km, bs_members))
                score = silhouette_score(sensor_pts, km.labels_)
                tried.append((K, round(float(score), 3), ok, info))
                if ok and (best is None or score > best[0]):
                    best = (score, K, km, info)
            K = self.ch_range[1]
            while best is None and K < self.total_sensors - 1:
                K += 1
                km = KMeans(n_clusters=K, random_state=0, n_init=10).fit(sensor_pts)
                ok, info = self.ch_check(*self._check_layout(all_pts, sensor_pts, km, bs_members))
                tried.append((K, None, ok, info))
                if ok:
                    best = (None, K, km, info)
                    print(f"WARNING: no K in {self.ch_range} passed the CH check; using K = {K}")
            _, K, k_means, info = best
            self.ch_selection = dict(K=K, tried=tried, info=info)
        elif self.auto_clusterheads:
            # Option: smallest K for which every sensor is within line-of-sight backscatter range
            # of its cluster head. (The original loop had the condition inverted and stopped at K = 1.)
            K = 1
            while True:
                k_means = KMeans(n_clusters=K, random_state=0, n_init=10).fit(sensor_pts)
                centres = k_means.cluster_centers_[k_means.labels_]
                if np.all(np.hypot(*(sensor_pts - centres).T) <= MAX_DIST_AMBC) or K >= self.total_sensors:
                    break
                K += 1
        else:
            # Paper setup (Sec. 4.1): K-means splits the sensors into a fixed number of clusters, with a
            # cluster head at each centroid. Sensors out of a CH's range are reached by UAV tours.
            K = int(self.total_clusterheads)
            k_means = KMeans(n_clusters=K, random_state=0, n_init=10).fit(sensor_pts)

        centroids = self._ugv_sites(k_means.cluster_centers_)
        labels = np.full(self.total_sensors, -1)
        labels[ugv_idx] = k_means.labels_
        labels[bs_members] = K                               # base-station cluster (if any)
        sensor_pts = all_pts
        self.total_clusterheads = K + int(self.base_station)
        print(f"Placing {K} Cluster Heads" + (" + 1 Base Station" if self.base_station else ""))

        ch_list = []
        for index, cluster in enumerate(centroids):
            members = np.flatnonzero(labels == index)
            ch_sensor_list = [sensor_list[i] for i in members]
            pt_max_dist = max(dist(sensor_pts[i].astype(float), cluster) for i in members)

            # How far the CH may move and still reach all its sensors
            r_move = max(MAX_DIST_AMBC - pt_max_dist, 0.0)
            lat, long = self.latlong(cluster[0], cluster[1])
            ch = devices_iot.ClusterHead(self, cluster[0], cluster[1], long, lat,
                                         ch_sensor_list, r_move, index)
            ch.sensor_los = [self.line_of_sight(sensor_pts[i][0], sensor_pts[i][1], cluster[0], cluster[1])
                             for i in members]
            ch_list.append(ch)
        if self.base_station:
            bx_, by_ = int(round(self.bs_xy[0])), int(round(self.bs_xy[1]))
            lat, long = self.latlong(bx_, by_)
            bs = devices_iot.BaseStation(self, bx_, by_, long, lat, [sensor_list[i] for i in bs_members], K)
            bs.sensor_los = [self.line_of_sight(sensor_pts[i][0], sensor_pts[i][1], bx_, by_) for i in bs_members]
            ch_list.append(bs)
            centroids = np.vstack([centroids, [bx_, by_]])
            for ch in ch_list:                               # trip to the guaranteed charger
                ch.charger_m = 0.0 if ch.is_base else math.hypot(ch.id_x - bx_, ch.id_y - by_)

        uav_list = []
        for count in range(self.total_uav):
            if self.base_station:                             # UAVs start the day at the base
                x, y = int(round(self.bs_xy[0])), int(round(self.bs_xy[1]))
            else:
                place = random.randint(0, self.dim * self.dim - 1)
                x, y = int(place % self.dim), int(place / self.dim)
            lat, long = self.latlong(x, y)
            uav_list.append([devices_uav.QuadUAV(x, y, long, lat, count, ch_list)])

        self.sensor_pts = sensor_pts
        self.ch_pts = centroids

        links = [(ch.sensor_los[j], math.hypot(s.id_x - ch.id_x, s.id_y - ch.id_y))
                 for ch in ch_list for j, s in enumerate(ch.sens_table.iloc[:, 0])]
        in_range = sum(d <= MAX_DIST_AMBC * (1.0 if los else NLOS_FACTOR_AMBC) for los, d in links)
        print(f"Sensor-CH links: {sum(los for los, _ in links)}/{len(links)} with line of sight, "
              f"{in_range}/{len(links)} sensors in their CH's backscatter range")

        # Ray-march every step for all sensor spots and cluster-head parking spots in one batch
        t0 = time.time()
        spots = [tuple(p) for p in sensor_pts] + [tuple(leg) for ch in ch_list for leg in ch.legs]
        self.precompute_sunlight(spots)
        print(f"Shadows traced for {len(self._sun_series)} device spots in {time.time() - t0:.1f}s")

        self.sensor_table = pd.DataFrame(sensor_list).rename(columns={0: "Sensor"})
        self.ch_table = pd.DataFrame(ch_list).rename(columns={0: "CH"})
        self.uav_table = pd.DataFrame(uav_list).rename(columns={0: "UAV"})
        return True
