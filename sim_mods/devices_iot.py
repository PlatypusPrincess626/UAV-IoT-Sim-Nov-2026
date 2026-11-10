"""
File: devices_iot.py
Author: Mason Conkel
Creation Date: 2025-05-29
Description: This script simulates the edge devices to be used in the simulation of a remote IoT network.
"""
# Import Dependencies
from scipy.interpolate import InterpolatedUnivariateSpline
import pandas as pd
import numpy as np
import random
import math
import copy
import networkx as nx

from sim_mods import ablation
from sim_mods.ablation import CFG

# ---------------------------------------------------------------------------
# Solar power model, after the UGV simulator (ugv_simulator.py):
#   P = POA irradiance x panel area x cell efficiency x fill factor x (1 - shade)
#       x panel soiling x wiring x MPPT efficiency
# The UGV's spectral-response factor is left out to keep the model simple (it is close to 1).
# Harvested power charges the battery at its voltage, limited by the maximum charge current and
# a coulombic charge efficiency. Loads are drawn from the battery as currents (mA).
# ---------------------------------------------------------------------------
CELL_EFFICIENCY = 0.20
PANEL_FILL_FACTOR = 0.75
PANEL_SOILING = 0.98
WIRING_EFFICIENCY = 0.99
MPPT_EFFICIENCY = 0.96
SYSTEM_EFFICIENCY = (CELL_EFFICIENCY * PANEL_FILL_FACTOR * PANEL_SOILING
                     * WIRING_EFFICIENCY * MPPT_EFFICIENCY)          # about 0.14
CHARGE_EFFICIENCY = 0.97

class EdgeDevice:
    """
    Parent class for the construction of an edge device
    """
    def __init__(self, x: int, y: int, longitude: float, latitude: float, max_data: int):
        # Assign positional information
        self.id_x = x  # X index
        self.id_y = y  # Y index
        self.id_z = 0  # Z index
        self.long = longitude  # Longitude
        self.lat = latitude  # Latitude

        # PLoRa specs
        self._comms = {
            "max_dist_lora": 5_000,  # LoRa maximum distance (m)
            "max_bitrate_lora": 24_975,  # LoRa maximum bitrate (bps)
            "current_active_lora": 4_200,  # LoRa active current (muA)
            "current_sleep_lora": 1_200,  # LoRa dormant current (muA)
            "voltage_lora": 3.7,  # LoRa voltage requirements (V)
            "pow_active_lora": 20,  # LoRa active power (muW)
            "pow_sleep_lora": 4,  # LoRa dormant power (muW)

            "max_dist_ambc": 500,  # AmBC maximum distance with line of sight (m), per PLoRa
            "nlos_factor_ambc": 0.2,  # range multiplier without line of sight (500 m -> 100 m)
            "max_bitrate_ambc": 1_592,  # AmBC maximum bitrate (bps)
            "pow_ambc": 3,  # AmBC upkeep power (muW)
            "voltage_ambc": 3.3,  # AmBC voltage requirements (V)
            "current_ambc": 785  # AmBC upkeep current (muA)
        }

        # CPU Specs
        self.pow_cpu = 3.7  # CPU power (muW)
        self.current_cpu = 1_000  # CPU current drain (muA)

        # Solar Specs
        self.spectral_low = 0  # Lower bandwidth bound
        self.spectral_high = np.inf  # Upper bandwidth bound
        self.is_solar = True  # Flag for solar powered (panel covers the device's load this minute)
        self.harvested_mAh = 0.0  # solar charge gained this minute

        # Sensor data storage
        self.max_data = max_data  # Sensor maximum data storage
        self.reset_min = round(self.max_data * 0.10)  # Minimum data amount on reset
        self.reset_max = round(self.max_data * 0.25)  # Maximum data amount on reset

        self.spectral_response = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                                           [0.23], [0.25], [0.27], [0.29], [0.31], [0.33], [0.3376], [0.3452], [0.3529],
                                           [0.3605], [0.3681], [0.3757], [0.3833], [0.3910], [0.3986], [0.4062],
                                           [0.4138], [0.4214], [0.4290], [0.4367], [0.4443], [0.4595], [0.4694],
                                           [0.4824], [0.4976], [0.5174], [0.5263], [0.5433], [0.5586], [0.5647],
                                           [0.5695], [0.5814], [0.5910], [0.5948], [0.5986], [0.6024], [0.6119],
                                           [0.6271], [0.6393], [0.6427], [0.6274], [0.6107], [0.5714], [0.5321],
                                           [0.4830], [0.4634], [0.4437], [0.4339], [0.4202], [0.3986], [0.3652],
                                           [0.3357], [0.3092], [0.2179], [0.1589], [10.0], [0.0], [0.0], [0.0], [0.0],
                                           [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                                           [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                                           [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                                           [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                                           [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])

    # Data fetching functions
    def get_indicies(self):
        return self.id_x, self.id_y

    def get_coords(self):
        return self.lat, self.long

    def step_battery(self):
        """
        One-minute battery update in percent: + solar charge harvested this minute (mAh, from
        harvest_energy) - loads drawn this minute (total_amp_spent is the average current in mA).
        Then clears both accumulators.
        """
        spent_mAh = self.total_amp_spent / 60.0
        self.stored_energy += 100.0 * (self.harvested_mAh - spent_mAh) / self.max_energy
        self.stored_energy = float(min(100.0, max(0.0, self.stored_energy)))
        self.total_amp_spent = 0.0
        self.harvested_mAh = 0.0

    def solar_charge_mAh(self, power_w: float) -> float:
        """Battery charge (mAh) from one minute of panel power, at the battery voltage and charge limit."""
        current_a = min(max(power_w, 0.0) / self.battery_voltage, self.max_charge_current)
        return current_a * 1_000.0 / 60.0 * CHARGE_EFFICIENCY

    def optimal_move(self, env, r_max, x_ch, y_ch, alpha, move_type, step, speed):
        x_c, y_c = self.get_indicies()
        pt_optimal = [x_c, y_c]

        # Determine starting point (higher alpha)
        if move_type == 0:
            obfuscation_potential = 100.0
            for x in range(2 * math.floor(r_max) + 1):
                for y in range(2 * math.floor(r_max) + 1):
                    x_try = x_c + x - math.floor(r_max)
                    y_try = y_c + y - math.floor(r_max)
                    if math.sqrt((x_c - x_try) ** 2 + (y_c - y_try) ** 2) < r_max:
                        curr_obfuscation = env.get_obfuscation(x_try, y_try, step)
                        avg_nxt_obfuscation = 0
                        for w in range(3):
                            for v in range(3):
                                avg_nxt_obfuscation += env.get_obfuscation(x_try + math.floor((w - 1) * speed),
                                                                           y_try + math.floor((v - 1) * speed),
                                                                           step + 1)
                        avg_nxt_obfuscation = 1/9 * avg_nxt_obfuscation
                        avg_stl_obfuscation = 0
                        for timestep in range(env.max_num_steps - step):
                            avg_stl_obfuscation += env.get_obfuscation(x_try, y_try, timestep)
                        avg_stl_obfuscation = avg_stl_obfuscation / (env.max_num_steps - step)
                        if (curr_obfuscation + alpha * avg_nxt_obfuscation + (1 - alpha) * avg_stl_obfuscation
                                < obfuscation_potential):
                            obfuscation_potential = (curr_obfuscation + alpha * avg_nxt_obfuscation +
                                                     (1 - alpha) * avg_stl_obfuscation)
                            pt_optimal = [x_try, y_try]
        else:
            obfuscation_potential = 100.0
            for x in range(3):
                for y in range(3):
                    next_x = x_ch + math.floor((x - 1) * speed)
                    next_y = y_ch + math.floor((y - 1) * speed)
                    if math.sqrt((x_c - next_x) ** 2 + (y_c - next_y) ** 2) < r_max:
                        curr_obfuscation = env.get_obfuscation(x_ch, y_ch, step)
                        avg_nxt_obfuscation = 0
                        for w in range(3):
                            for v in range(3):
                                avg_nxt_obfuscation += env.get_obfuscation(next_x + math.floor((w - 1) * speed),
                                                                           next_y + math.floor((v - 1) * speed),
                                                                           step + 1)
                        avg_nxt_obfuscation = 1 / 9 * avg_nxt_obfuscation
                        avg_stl_obfuscation = 0
                        for timestep in range(env.max_num_steps - step):
                            avg_stl_obfuscation += env.get_obfuscation(next_x, next_y, timestep)
                        avg_stl_obfuscation = avg_stl_obfuscation / (env.max_num_steps - step)
                        if (curr_obfuscation + alpha * avg_nxt_obfuscation + (1 - alpha) * avg_stl_obfuscation
                                < obfuscation_potential):
                            obfuscation_potential = (curr_obfuscation + alpha * avg_nxt_obfuscation +
                                                     (1 - alpha) * avg_stl_obfuscation)
                            pt_optimal = [next_x, next_y]

        return pt_optimal


    def find_power(self, env, x: int, y: int, step: int,
                   sol_area: int, tilt: int, azimuth: int):

        # Sunlight reaching the panel: 1 - obfuscation (previously multiplied by the obfuscation itself,
        # so open ground produced no power and only shaded spots did)
        sunlight = 1.0 - env.get_obfuscation(x, y, step)
        poa = env.poa_irradiance(self.lat, self.long, tilt, azimuth, step)
        # Panel output in watts (UGV-style model, see SYSTEM_EFFICIENCY). Follows the sun through pvlib
        # and the shade at this spot; the old time-of-day polynomial factor has been removed.
        return poa * sol_area * SYSTEM_EFFICIENCY * sunlight

    def path(self, env, num_legs: int, r_max: int, max_speed: int, acceleration: int, u_rr: float, w: int,
             sol_area: int, sol_current: int, sol_voltage: int, tilt: int, azimuth: int):
        """
        Where this mobile device should park through the day, as (legs, times): one spot per time
        window, each the sunniest reachable spot for that window as the shadows move (see
        sim_env.plan_legs). The remaining parameters are kept so callers don't change.

        This replaces the old per-point, per-minute search, which crashed on a tuple index
        (centroid = [self.get_indicies()]) and could loop forever: its grace-period and final
        loops only advanced time when a point had power.
        """
        return env.plan_legs(self.id_x, self.id_y, r_max, num_legs)


class Sensor(EdgeDevice):
    """
    Child class of EdgeDevice that simulates a sensor
    """
    def __init__(self, x: int, y: int, long: float, lat: float, data_type: int):
        super().__init__(x, y, long, lat, 256_000)
        self.type = 1
        self.data_type = data_type
        self.stored_data = random.randint(self.reset_min, self.reset_max)  # Initialization of data storage
        self.curr_pt = [self.id_x, self.id_y]

        # Sensor power requirements
        self.sens_pow = 2.2  # 2.2 W power consumption
        self.sens_amp = 1_000 * self.sens_pow / self._comms["voltage_ambc"]  # mA (was A)

        # Solar panel specifics
        self.azimuth = 0  # 0 for North pointing
        self.tilt = 0  # 0 for no horizontal tilt
        self.solar_voltage = 5  # V
        self.solar_current = 2  # A
        self.solar_power = 10  # W
        self.solar_area = .137 * .222   # 137 mm x 222 mm

        # Sensor data collection
        self.max_col_rate = 16  # Sensor max collection rate
        self.sample_freq = 25  # Percent of a minute spent sampling
        self.sample_len = 1  # Number of seconds to sample over
        self.sample_rate = 300  # Samples taken per period
        self.bits_per_min = round(self.max_col_rate * self.sample_len *
                                  self.sample_rate / (self.sample_freq / 100))

        # Battery Specifics
        self.C = 1  # Capacitance of the energy storage (F)
        self.max_energy = 1_280  # Maximum battery capacity (mAh)
        self.battery_voltage = 3.3  # V (sensor loads are specified at 3.3 V)
        self.max_charge_current = 1.28  # A (1C)
        self.charge_rate = 1/3  # 20 mins to charge
        self.discharge_rate = 2  # 2 h to discharge
        self.stored_energy = 100.0  # Percent

        self.total_amp_spent = 0.0
        self.step_charge = 100 / (self.charge_rate * 60)

    def reset(self):
        self.stored_energy = 100.0 # Percent
        self.stored_data = random.randint(0, self.reset_max)
        self.total_amp_spent = 0.0

    def step(self, env, step):
        self.total_amp_spent = 0.0
        self.harvest_energy(env, step)
        self.harvest_data()
        self.step_battery()

    def harvest_data(self):
        if self.is_solar:
            self.stored_data += min(self.bits_per_min, self.max_data)
            return True

        elif self.stored_energy > round(10):
            self.stored_data += min(self.bits_per_min, self.max_data)
            self.total_amp_spent += self.sens_amp * (self.sample_freq / 100)
            return True

        else:
            return False

    def harvest_energy(self, env, step):
        """
        Panel power this minute. If it covers the sampling load, the sensor samples on solar
        (is_solar) and the surplus charges the battery; otherwise the battery runs the sampling
        (see harvest_data) and the panel output still charges it.
        """
        power = self.find_power(env, self.id_x, self.id_y, step, self.solar_area, self.tilt, self.azimuth)
        sampling_w = self.sens_pow * (self.sample_freq / 100)
        self.is_solar = power >= sampling_w
        self.harvested_mAh = self.solar_charge_mAh(power - sampling_w if self.is_solar else power)

        amp_upkeep = (self.current_cpu + self._comms.get("current_ambc")) / 1_000  # to mA
        self.total_amp_spent += amp_upkeep

    def upload(self, X, Y, los: bool = True):
        """
        Backscatters stored data to a reader at (X, Y) if it is in range: 500 m with line of sight,
        1/5 of that (100 m) without. The airborne UAV always has line of sight; ground links to a
        cluster head pass the los flag the environment worked out from the landscape.
        """
        reach = self._comms.get("max_dist_ambc") * (1.0 if los else self._comms.get("nlos_factor_ambc"))
        if math.sqrt(pow((self.id_x - X), 2) + pow((self.id_y - Y), 2)) <= reach:
            temp = min(self._comms.get("max_bitrate_ambc") * 26, self.stored_data)
            self.stored_data -= temp
            return temp
        else:
            return -1


class ClusterHead(EdgeDevice):
    """
    Child class of EdgeDevice that simulates a cluster head
    """
    is_base = False   # True for the grid-powered BaseStation subclass
    perceived_soc = 50.0   # fleet-shared charge reading from the last UAV contact (prior 50 %)
    charger_m = 0.0        # distance to the nearest guaranteed charger (base station), set by sim_env
    plan_short = False     # the last cached sortie was cut short by the UAV's battery

    def __init__(self, env, x: int, y: int, long: float, lat: float, sensor_list: list, r_move: float, cluster_num: int):
        super().__init__(x, y, long, lat, 12_500_000)
        self.type = 2
        self.typeStr = "Clusterhead"
        self.cluster_num = cluster_num
        self.stored_data = self.reset_max  # Storage initialization

        # Solar panel specifics
        self.azimuth = 180
        self.tilt = 45
        self.h = 0
        self.solar_area = (1.020 * 0.520) * 0.45  # UGV panel: 45 % of a 1020 mm x 520 mm module (0.24 m^2)
        self.solar_voltage = 18  # V
        self.solar_current = 6  # A
        self.solar_power = 100  # W

        self.r_max = r_move
        self.speed = 20  # m / min
        self.acceleration = 300  # m / min**2
        self.u_rr = 0.1
        self.num_legs = 4
        self.w = 21  # N
        self.legs, self.times = self.path(env, self.num_legs, self.r_max, self.speed, self.acceleration,
                                          self.u_rr, self.w, self.solar_area, self.solar_current,
                                          self.solar_voltage, self.tilt, self.azimuth)

        self.leg_num = 0
        self.leg = self.legs[self.leg_num]
        self.est_point = self.leg
        self.curr_pt = self.leg
        self.curr_spd = 0

        # State tracking of sensors
        self.max_age = 0
        self.avg_age = 0
        self.num_sensors = 0
        self.contribution = 0
        self.action_p = 0
        self.sens_dict = None
        self.sens_table = None
        self.active_table = None
        self.age_table = None
        self.data_table = None
        self.set_sensor_data(sensor_list)

        # Target tracking for UAV
        """
        Note: Too many lists for full use. Find method to store various lists
        at different sizes to reduce numbers of variables.
        """
        self.last_target = 0
        self.target_time = 0

        self.tour = []
        self.next_tour = []
        self.next_dist = 0.0
        self.next1_tour = []
        self.next1_dist = 0.0
        self.next2_tour = []
        self.next2_dist = 0.0

        # Battery Specifics
        self.C = 3200  # Capacitance of the energy storage (F)
        self.max_energy = 5_000  # Maximum battery capacity (mAh), UGV battery: 5 Ah
        self.battery_voltage = 14.8  # V, UGV nominal (4S Li-ion)
        self.max_charge_current = 8.0  # A, UGV peak charging current
        self.charge_reserve = 10.0  # % of battery the CH keeps for itself (won't charge the UAV below it)
        self.charge_rate = 1 / 3  # Charge rate of battery (A/h)
        self.discharge_rate = 0.655  # Discharge rate of batter (A/h)
        self.outage_minutes = 0  # minutes the CH could not supply all the charge the UAV asked for

        self.stored_energy = 100  # Battery initialization
        self.total_amp_spent = 0.0
        self.step_charge = 100 / (self.charge_rate * 60)

    def reset(self):
        self.perceived_soc = 50.0 if not self.is_base else 100.0   # fleet's reading resets each day
        self.tour_idx = []
        # Reset State
        self.max_age = 0
        self.avg_age = 0
        self.data_table[0] = self.stored_data
        for sens in range(self.num_sensors):
            self.active_table[sens] = True
            self.age_table[sens] = 0
            self.data_table[sens + 1] = 0

        # Reset data and battery
        self.stored_data = random.randint(0, self.reset_max)
        self.stored_energy = 100
        self.total_amp_spent = 0.0
        self.harvested_mAh = 0.0
        self.outage_minutes = 0
        self.step_charge = 100 / (self.charge_rate * 60)

        self.leg_num = 0
        self.leg = self.legs[self.leg_num]
        self.est_point = self.leg
        self.curr_pt = self.leg
        self.curr_spd = 0

        # Reset target
        self.contribution = 0
        self.action_p = 0
        self.last_target = 0
        self.target_time = 0

        self.tour = []
        self.next_tour = []
        self.next_dist = 0.0
        self.next1_tour = []
        self.next1_dist = 0.0
        self.next2_tour = []
        self.next2_dist = 0.0

    def step(self, env, step):
        self.total_amp_spent = 0.0
        self.harvest_energy(env, step)
        self.download(step)

    def harvest_energy(self, env, step):
        power = self.find_power(env, self.curr_pt[0], self.curr_pt[1], step, self.solar_area, self.tilt, self.azimuth)

        # Powered by the panel if it covers the CH's own load (processor + LoRa); all of the panel's
        # output goes to the battery, which supplies every load (including charging the UAV)
        own_load_w = (self.current_cpu + self._comms.get("current_active_lora")) / 1e6 * self.battery_voltage
        self.is_solar = power >= own_load_w
        self.harvested_mAh = self.solar_charge_mAh(power)

        # Advance to the next leg once its start time is reached (was `step == times[...]`, which
        # never matched the float times the old path() produced)
        while self.leg_num + 1 < len(self.legs) and step >= self.times[self.leg_num]:
            self.leg_num += 1
            self.leg = self.legs[self.leg_num]

        energy_move = 0
        if self.curr_pt != self.leg:
            travel_dist = math.sqrt((self.leg[0] - self.est_point[0]) ** 2 + (self.leg[1] - self.est_point[1]) ** 2)
            accel_dist = ((0.5 * self.acceleration * (self.speed / self.acceleration) ** 2) +
                          (self.speed * (self.speed / self.acceleration) - 0.5 * self.acceleration *
                           (self.speed / self.acceleration) ** 2))
            if accel_dist < travel_dist:
                t_accel = math.sqrt(travel_dist / self.acceleration)
                v_final = self.acceleration * t_accel
                energy_move = self.w / 9.8 * v_final ** 2
                self.curr_pt = self.leg
                self.est_point = self.leg
            elif (accel_dist + (1 - 2 * self.speed / self.acceleration) * self.speed) >= travel_dist:
                t_remain = (travel_dist - accel_dist) / self.speed
                power_velocity = self.w * 3600 * self.speed * self.u_rr
                energy_move = self.w / 9.8 * self.speed ** 2 + power_velocity * t_remain
                self.curr_pt = self.leg
                self.est_point = self.leg
            elif self.curr_spd < self.speed:
                t_remain = (travel_dist - 0.5 * accel_dist) / self.speed
                power_velocity = self.w * 3600 * self.speed * self.u_rr
                energy_move = self.w / 9.8 * self.speed ** 2 + power_velocity * t_remain
                travelled = self.speed * t_remain + 0.5 * self.acceleration * (self.speed / self.acceleration)**2
                self.est_point = [self.est_point[0] + (travelled / travel_dist) * (self.leg[0] - self.est_point[0]),
                                  self.est_point[1] + (travelled / travel_dist) * (self.leg[1] - self.est_point[1])]
                self.curr_pt = [round(self.est_point[0]), round(self.est_point[1])]
            elif (0.5 * accel_dist + (1 - self.speed / self.acceleration) * self.speed) > travel_dist:
                t_remain = (travel_dist - 0.5 * accel_dist) / self.speed
                power_velocity = self.w * 3600 * self.speed * self.u_rr
                energy_move = self.w / 9.8 * self.speed ** 2 + power_velocity * t_remain
                self.curr_pt = self.leg
                self.est_point = self.leg
            else:
                power_velocity = self.w * 3600 * self.speed * self.u_rr
                energy_move = power_velocity
                travelled = self.speed
                self.est_point = [self.est_point[0] + (travelled / travel_dist) * (self.leg[0] - self.est_point[0]),
                                  self.est_point[1] + (travelled / travel_dist) * (self.leg[1] - self.est_point[1])]
                self.curr_pt = [round(self.est_point[0]), round(self.est_point[1])]

            self.total_amp_spent += energy_move / self.battery_voltage

        amp_upkeep = (self.current_cpu + self._comms.get("current_active_lora")) / 1_000  # to mA
        self.total_amp_spent += amp_upkeep

    def set_sensor_data(self, sens_list: list):
        self.num_sensors = len(sens_list)

        self.active_table = np.array([0] * self.num_sensors)
        self.age_table = np.array([0] * self.num_sensors)
        self.sens_table = pd.DataFrame(np.array(sens_list))
        # Line of sight from each sensor to this CH; set by the environment once the landscape exists
        self.sensor_los = [True] * self.num_sensors

        data_list = [self.stored_data]
        for sensor in range(self.num_sensors):
            data_list.append(0)
        self.data_table = np.array(data_list)

    def download(self, step):
        rotations = math.ceil(len(self.sens_table.index) / 2)
        rotation = step % rotations
        sensor = rotation * 2
        activeChannels = []
        sensor1 = self.sens_table.iat[sensor, 0]
        activeChannels.append(sensor1.upload(self.id_x, self.id_y, self.sensor_los[sensor]))

        if rotation < (rotations - 1) or len(self.sens_table.index) % 2 == 0:
            sensor2 = self.sens_table.iat[sensor + 1, 0]
            activeChannels.append(sensor2.upload(self.id_x, self.id_y, self.sensor_los[sensor + 1]))

        totalChannels = 0
        for channel in range(len(activeChannels)):
            if activeChannels[channel] >= 0:
                self.active_table[sensor + channel] = True
                self.age_table[sensor + channel] = step
                self.data_table[sensor + channel + 1] += activeChannels[channel]
                self.stored_data += activeChannels[channel]
                totalChannels += 1
            else:
                self.active_table[sensor + channel] = False

        # ADF 2.0
        self.max_age = self.age_table[0]
        self.avg_age = self.age_table[0]
        for sens in range(len(self.sens_table.index) - 1):
            self.avg_age += self.age_table[sens + 1]
            if self.age_table[sens + 1] < self.max_age:
                self.max_age = self.age_table[sens + 1]
        self.avg_age = math.ceil(self.avg_age / len(self.age_table))

    def upload(self, X: int, Y: int):
        if self.is_solar:
            if math.sqrt(pow((self.id_x - X), 2) + pow((self.id_y - Y), 2)) <= self._comms.get("max_dist_lora"):

                if self.stored_data > 0:
                    sent_data = min(self._comms.get("max_bitrate_lora") * 56, self.stored_data)
                    self.stored_data -= sent_data
                    self.contribution += sent_data

                    return sent_data, self.max_age, self.avg_age

                else:
                    return 0, self.max_age, self.avg_age

            else:
                return -1, self.max_age, self.avg_age

        # Extra drain for another comms channel
        elif self.stored_energy > ( ( self._comms.get("current_active_lora") / 1_000 ) / ( self.max_energy * 60 ) ):
            if math.sqrt(pow((self.id_x - X), 2) + pow((self.id_y - Y), 2)) <= self._comms.get("max_dist_lora"):

                if self.stored_data > 0:
                    sent_data = min(self._comms.get("max_bitrate_lora") * 56, self.stored_data)
                    self.stored_data -= sent_data
                    self.contribution += sent_data
                    time = sent_data / self._comms.get("max_bitrate_lora")  # seconds (was * 56: up to 52 min)

                    self.total_amp_spent += (self._comms.get("current_active_lora") / 1_000) * (time / 60)
                    return sent_data, self.max_age, self.avg_age

                else:
                    self.total_amp_spent += self._comms.get("current_active_lora") / 1_000
                    self.total_amp_spent += (self._comms.get("current_active_lora") / 1_000) * (4 / 60)
                    return 0, self.max_age, self.avg_age

            else:
                return -1, self.max_age, self.avg_age
        else:
            return -1, self.max_age, self.avg_age

    def charge_time(self, X: int, Y: int, charge, needed_wh: float = 0.0):
        """
        Seconds of full-rate charging this CH gives a docked UAV this minute (0-60).

        The UAV is charged only when it is charging (charge=True) and asks for needed_wh (the energy
        it would take in a full minute). The CH pays for it from its battery, topped up by this
        minute's solar harvest, and keeps charge_reserve % for itself. If it can't cover the whole
        minute it gives a proportional part (an outage minute). Previously the UAV was charged for
        free while the CH counted as solar-powered, and got half-rate charge whenever it was docked,
        even when not charging.
        """
        docked = abs(self.curr_pt[0] - X) < 1.0 and abs(self.curr_pt[1] - Y) < 1.0
        if not docked or not charge or needed_wh <= 0.0:
            return 0.0
        need_mAh = needed_wh * 1_000.0 / self.battery_voltage / CHARGE_EFFICIENCY
        stored_mAh = self.stored_energy / 100.0 * self.max_energy
        available = (stored_mAh - self.charge_reserve / 100.0 * self.max_energy
                     + self.harvested_mAh - self.total_amp_spent / 60.0)
        fraction = float(min(1.0, max(0.0, available / need_mAh)))
        if fraction < 1.0:
            self.outage_minutes += 1
        self.total_amp_spent += fraction * need_mAh * 60.0   # as an average current over the minute
        return 60.0 * fraction

    def _sensor_order(self):
        """
        Cached visiting order of this CH's sensors: nearest-neighbour from the CH, then 2-opt on the
        closed tour CH -> sensors -> CH. Sensors don't move, so it is computed once per layout.
        """
        if getattr(self, "_order", None) is not None:
            return self._order
        pts = np.array([[self.id_x, self.id_y]] +
                       [[self.sens_table.iat[i, 0].id_x, self.sens_table.iat[i, 0].id_y]
                        for i in range(len(self.sens_table))], float)
        D = np.hypot(*(pts[:, None, :] - pts[None, :, :]).transpose(2, 0, 1))
        n = len(pts)
        tour, left = [0], set(range(1, n))
        while left:
            j = min(left, key=lambda k: D[tour[-1], k])
            tour.append(j)
            left.remove(j)
        t = np.array(tour + [0])                    # closed tour, CH at both ends
        improved = True
        while improved and n > 3:
            improved = False
            for i in range(1, n - 1):
                a, b = t[i - 1], t[i]
                c, d = t[i + 1:n], t[i + 2:n + 1]
                delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]
                j = int(np.argmin(delta))
                if delta[j] < -1e-9:
                    j = i + 1 + j
                    t[i:j + 1] = t[i:j + 1][::-1]
                    improved = True
        self._order = [int(k) - 1 for k in t[1:n]]  # sensor indices into sens_table
        return self._order

    def get_dest(self, state, full_sensor_list, model, model_p,
                     step, p_count, targetType, _=None):
        """
        Returns the UAV's next destination while it is sitting at this cluster head.

        Ablation switches read from sim_mods.ablation.CFG:
            sensor_sweep    christofides | legacy | none
            decision_state  projected | raw
        The model's choice of cluster head is always used as-is (no forced change).
        """
        my_contribution = state[self.cluster_num + 1][1]
        if my_contribution > self.contribution:
            self.contribution = my_contribution

            if CFG.sensor_sweep == "legacy":
                # Old contribution rule: only the single sensor the UAV was sent to is refreshed
                if 0 <= self.last_target < len(self.age_table):
                    self.age_table[self.last_target] = self.target_time
            elif CFG.sensor_sweep == "cached":
                # Only the sensors the finished sortie actually flew to are refreshed
                for sens in getattr(self, "tour_idx", []):
                    self.age_table[sens] = self.target_time
                self.tour_idx = []
            else:
                # New contribution rule: every sensor that was out of the CH's range is refreshed
                for sens in range(len(self.active_table)):
                    if not self.active_table[sens]:
                        self.age_table[sens] = self.target_time     # Sensors on range 1 to num_sens

            self.max_age = self.age_table[0]
            self.avg_age = self.age_table[0]
            for sens in range(len(self.sens_table.index) - 1):
                self.avg_age += self.age_table[sens + 1]
                if self.age_table[sens + 1] < self.max_age:
                    self.max_age = self.age_table[sens + 1]
            self.avg_age = math.ceil(self.avg_age / len(self.age_table))

            state[self.cluster_num + 1][2] = step - self.max_age
            state[self.cluster_num + 1][3] = step - self.avg_age

        decision_state = copy.deepcopy(state)
        if CFG.decision_state == "projected":
            # Project the state forward past any charging the UAV has committed to (p_count minutes)
            decision_state[0][2] = state[0][2] + ablation.charge_projection(p_count, state[0][2])
            for CH in range(len(state) - 1):
                decision_state[CH+1][2] = state[CH+1][2] + p_count
                decision_state[CH+1][3] = state[CH+1][3] + p_count
        # "raw": the model sees the state exactly as the UAV currently holds it

        if CFG.sensor_sweep == "legacy":
            return self._legacy_targeting(state, decision_state, full_sensor_list, model, model_p,
                                          step, p_count)

        action = -1
        target = self
        model_help = True
        change_transit = False
        dist = 0

        """Christofides' Tour"""
        if CFG.sensor_sweep == "none":
            # Ablation: never sweep out-of-range sensors; the UAV only hops between cluster heads
            targetType = True
            model_help = True

        elif not targetType and CFG.sensor_sweep == "cached":
            order = self._sensor_order()
            need = [i for i in order if not self.active_table[i]
                    and step - self.age_table[i] >= CFG.tour_fresh_min]
            # The cached order is a loop: start at the stalest sensor that needs a visit and walk the
            # loop, taking sensors while the sortie (out, along, back to this CH) still leaves the 20 %
            # reserve -- plus the trip to the base station if this CH cannot recharge the UAV. No split:
            # sensors that don't fit wait for the next visit, and starting at the stalest one means the
            # same sensors are never always left out.
            supplies = self.is_base or self.stored_energy >= CFG.divert_min_soc
            budget = state[0][2] - 0.2 * ablation.batt_full() - (
                0.0 if supplies or not self.charger_m
                else CFG.tour_margin * ablation.minutes_pct(ablation.leg_minutes(self.charger_m)))
            prefix, length = [], 0.0
            if need:
                start = min(need, key=lambda i: self.age_table[i])          # oldest collection
                pos = {s_: k for k, s_ in enumerate(order)}
                wanted = set(need)
                loop = [order[(pos[start] + k) % len(order)] for k in range(len(order))]
                px, py, minutes = self.id_x, self.id_y, 0
                for i in loop:
                    if i not in wanted:
                        continue
                    sx, sy = self.sens_table.iat[i, 0].id_x, self.sens_table.iat[i, 0].id_y
                    leg = math.hypot(sx - px, sy - py)
                    m = minutes + ablation.leg_minutes(leg)
                    back = ablation.leg_minutes(math.hypot(sx - self.id_x, sy - self.id_y))
                    if CFG.tour_margin * ablation.minutes_pct(m + back) > budget:
                        continue                     # skip this one; a nearer one later may still fit
                    prefix.append(i)
                    minutes, length, px, py = m, length + leg, sx, sy
            self.plan_short = len(prefix) < len(need)   # the battery, not the need, ended the sortie
            if prefix:
                length += math.hypot(px - self.id_x, py - self.id_y)
                self.tour = [self.sens_table.iat[i, 0] for i in prefix]
                self.tour_idx = list(prefix)
                dist = length
                target = self.tour[0]
                self.last_target = self.cluster_num
                self.target_time = step
                action = self.cluster_num
                model_help = False
            else:
                targetType = True
                model_help = True

        elif not targetType:
            inactive = []
            for sens in range(len(self.active_table)):
                if not self.active_table[sens]:
                    inactive.append(sens)     # Sensors on range 1 to num_sens

            if len(inactive) == 1:
                target = self.sens_table.iat[inactive[0], 0]
                dist = 2 * math.sqrt(pow((target.id_x - self.id_x), 2) + pow((target.id_y - self.id_y), 2))

                self.last_target = self.cluster_num
                self.target_time = step
                action = self.cluster_num
                model_help = False

            elif len(inactive) > 1:
                G = nx.Graph()
                for i in range(len(inactive)+1):
                    for j in range(i+1, len(inactive)+1):
                        if i == 0:
                            G.add_edge(
                                i,
                                j,
                                weight = math.sqrt(pow((self.sens_table.iat[inactive[j-1], 0].id_x - self.id_x), 2)
                                                   + pow((self.sens_table.iat[inactive[j-1], 0].id_y - self.id_y), 2))
                            )  # Use Euclidean distance (2-norm) as graph weight
                        elif j == 0:
                            G.add_edge(
                                i,
                                j,
                                weight=math.sqrt(pow((self.sens_table.iat[inactive[i-1], 0].id_x - self.id_x), 2)
                                                 + pow((self.sens_table.iat[inactive[i-1], 0].id_y - self.id_y), 2))
                            )  # Use Euclidean distance (2-norm) as graph weight
                        else:
                            G.add_edge(
                                i,
                                j,
                                weight = math.sqrt(pow((self.sens_table.iat[inactive[j-1], 0].id_x
                                                        - self.sens_table.iat[inactive[i-1], 0].id_x), 2)
                                                   + pow((self.sens_table.iat[inactive[j-1], 0].id_y
                                                          - self.sens_table.iat[inactive[i-1], 0].id_y), 2))
                            )

                tour = nx.algorithms.approximation.christofides(G)
                dists = []
                for i in range(len(tour)):
                    if i == 0:
                        dists.append(math.sqrt(pow((self.sens_table.iat[inactive[tour[i+1]-1], 0].id_x - self.id_x), 2)
                                              + pow((self.sens_table.iat[inactive[tour[i+1]-1], 0].id_y - self.id_y), 2)))
                    elif i+1 == len(tour):
                        dists.append(math.sqrt(pow((self.sens_table.iat[inactive[tour[i]-1], 0].id_x - self.id_x), 2)
                                              + pow((self.sens_table.iat[inactive[tour[i]-1], 0].id_y - self.id_y), 2)))
                    else:
                        dists.append(math.sqrt(pow((self.sens_table.iat[inactive[tour[i+1]-1], 0].id_x -
                                                   self.sens_table.iat[inactive[tour[i]-1], 0].id_x), 2)
                                              + pow((self.sens_table.iat[inactive[tour[i+1]-1], 0].id_y -
                                                     self.sens_table.iat[inactive[tour[i]-1], 0].id_y), 2)))

                # split into two function if too much distance\
                tour_discharge = ablation.tour_discharge(sum(dists))
                if (state[0][2] - tour_discharge) >= (0.2 * ablation.batt_full()):
                    self.tour = [self.sens_table.iat[inactive[tour[i]-1], 0] for i in range(len(tour))]
                    dist = sum(dists)
                    target = self.tour[0]
                elif len(tour) > 1:
                    tour1 = tour[0:math.ceil(len(tour)/2)]
                    tour2 = tour[math.ceil(len(tour)/2):]
                    dists1= (sum(dists[0:math.ceil(len(dists)/2)]) +
                             math.sqrt(pow((self.sens_table.iat[inactive[tour1[-1] - 1], 0].id_x - self.id_x), 2)
                                       + pow((self.sens_table.iat[inactive[tour1[-1] - 1], 0].id_y - self.id_y), 2))
                             )
                    dists2= (sum(dists[math.ceil(len(dists)/2):]) +
                             math.sqrt(pow((self.sens_table.iat[inactive[tour2[0] - 1], 0].id_x - self.id_x), 2)
                                       + pow((self.sens_table.iat[inactive[tour2[0] - 1], 0].id_y - self.id_y), 2))
                             )

                    tour1_discharge = ablation.tour_discharge(dists1)
                    if (state[0][2] - tour1_discharge) >= (0.2 * ablation.batt_full()):
                        self.tour = [self.sens_table.iat[inactive[tour1[i] - 1], 0] for i in range(len(tour1))]
                        dist = dists1
                        target = self.tour[0]
                    elif len(tour1) > 1:
                        distsA = dists[0:math.ceil(len(dists)/2)]
                        tour11 = tour[0:math.ceil(len(tour1) / 2)]
                        tour21 = tour[math.ceil(len(tour1) / 2):]
                        dists11 = (sum(distsA[0:math.ceil(len(distsA) / 2)]) +
                                  math.sqrt(pow((self.sens_table.iat[inactive[tour11[-1] - 1], 0].id_x - self.id_x), 2)
                                            + pow((self.sens_table.iat[inactive[tour11[-1] - 1], 0].id_y - self.id_y),
                                                  2))
                                  )
                        dists21 = (sum(distsA[math.ceil(len(distsA) / 2):]) +
                                  math.sqrt(pow((self.sens_table.iat[inactive[tour21[0] - 1], 0].id_x - self.id_x), 2)
                                            + pow((self.sens_table.iat[inactive[tour21[0] - 1], 0].id_y - self.id_y), 2))
                                  + math.sqrt(pow((self.sens_table.iat[inactive[tour21[-1] - 1], 0].id_x - self.id_x), 2)
                                            + pow((self.sens_table.iat[inactive[tour21[-1] - 1], 0].id_y - self.id_y), 2))
                                  )

                        self.tour = [self.sens_table.iat[inactive[tour11[i] - 1], 0] for i in
                                     range(len(tour11))]
                        dist = dists11
                        self.next1_tour = [self.sens_table.iat[inactive[tour21[i] - 1], 0] for i in
                                          range(len(tour21))]
                        self.next1_dist = dists21
                        target = self.tour[0]

                    tour2_discharge = ablation.tour_discharge(dists2)
                    if (state[0][2] - tour2_discharge) >= (0.2 * ablation.batt_full()):
                        self.next_tour = [self.sens_table.iat[inactive[tour2[i] - 1], 0] for i in
                                          range(len(tour2))]
                        self.next_dist = dists2
                    elif len(tour2) > 1:
                        distsB = dists[math.ceil(len(dists)/2):]
                        tour12 = tour[0:math.ceil(len(tour2) / 2)]
                        tour22 = tour[math.ceil(len(tour2) / 2):]
                        dists12 = (sum(distsB[0:math.ceil(len(distsB) / 2)]) +
                                   math.sqrt(pow((self.sens_table.iat[inactive[tour12[-1] - 1], 0].id_x - self.id_x), 2)
                                             + pow((self.sens_table.iat[inactive[tour12[-1] - 1], 0].id_y - self.id_y),2))
                                   + math.sqrt(pow((self.sens_table.iat[inactive[tour12[0] - 1], 0].id_x - self.id_x), 2)
                                             + pow((self.sens_table.iat[inactive[tour12[0] - 1], 0].id_y - self.id_y),2))
                                   )
                        dists22 = (sum(distsB[math.ceil(len(distsB) / 2):]) +
                                   math.sqrt(pow((self.sens_table.iat[inactive[tour22[0] - 1], 0].id_x - self.id_x), 2)
                                             + pow((self.sens_table.iat[inactive[tour22[0] - 1], 0].id_y - self.id_y),
                                                   2))
                                   )

                        self.next_tour = [self.sens_table.iat[inactive[tour12[i] - 1], 0] for i in
                                     range(len(tour12))]
                        self.next_dist = dists12
                        self.next2_tour = [self.sens_table.iat[inactive[tour22[i] - 1], 0] for i in
                                           range(len(tour22))]
                        self.next2_dist = dists22

                self.last_target = self.cluster_num
                self.target_time = step
                action = self.cluster_num
                model_help = False

            else:
                targetType = True
                model_help = True

        elif len(self.next_tour) > 0:
            self.tour = self.next_tour
            dist = self.next_dist
            target = self.tour[0]
            self.target_time = step
            action = self.cluster_num
            model_help = False
            targetType = False

            self.next_tour = []
            self.next_dist = 0.0

        elif len(self.next1_tour) > 0:
            self.tour = self.next1_tour
            dist = self.next1_dist
            target = self.tour[0]
            self.target_time = step
            action = self.cluster_num
            model_help = False
            targetType = False

            self.next1_tour = []
            self.next1_dist = 0.0

        elif len(self.next2_tour) > 0:
            self.tour = self.next2_tour
            dist = self.next2_dist
            target = self.tour[0]
            self.target_time = step
            action = self.cluster_num
            model_help = False
            targetType = False

            self.next2_tour = []
            self.next2_dist = 0.0

        # Next CH
        """Cluster Targeting"""
        if model_help:
            action = model.act(decision_state)

            target = full_sensor_list.iat[action + 1, 0]
            change_transit = True

            dist = math.sqrt(pow((target.id_x - self.id_x), 2) + pow((target.id_y - self.id_y), 2))

        AoI_peak = decision_state[1][2]
        AoI_avg = decision_state[1][3]
        for entry in range(len(full_sensor_list) - 2):
            AoI_avg += decision_state[entry + 2][3]
            AoI = decision_state[entry + 2][2]
            if AoI > AoI_peak:
                AoI_peak = AoI
        AoI_avg = math.ceil(AoI_avg / len(full_sensor_list))

        return self._finish_dest(state, model_p, p_count, model_help, change_transit, target,
                                 decision_state, action, dist, AoI_peak, AoI_avg, targetType)

    # ---- Helpers shared by the targeting strategies ----------------------
    def _finish_dest(self, state, model_p, p_count, model_help, change_transit, target,
                     decision_state, action, dist, AoI_peak, AoI_avg, targetType):
        """Builds the power-agent state, queries it, and packages get_dest's return tuple."""
        p_state = [dist, state[0][2] + ablation.charge_projection(p_count, state[0][2]),
                   AoI_peak + p_count, AoI_avg + p_count]
        # Charging duration in minutes (0 to nA-1 of the power agent). Only used when charging == "model";
        # the other charging policies are handed a NullPowerAgent that always returns 0.
        self.action_p = model_p.act(p_state)

        return (model_help, change_transit, target, decision_state, action, self.action_p, p_state,
                dist, AoI_peak, AoI_avg, self.tour, targetType)

    def _legacy_targeting(self, state, decision_state, full_sensor_list, model, model_p,
                          step, p_count):
        """
        Old targeting (reconstructed from IoT_Device.get_dest, "Old Targeting" block).

        The targeting model has num_ch + 1 actions. Actions 0..num_ch-1 pick a cluster head;
        action num_ch sends the UAV to this CH's single stalest out-of-range sensor, with no tour.

        Differences from the original commented code, all needed for it to run:
          * The stalest sensor is described in an extra appended state row instead of overwriting
            the last CH's row, so the state has num_ch + 2 rows. The original wrote
            data_table[max_idx] (off by one, since data_table[0] is the CH's own store) and then
            overwrote the data column with the age; here the row is
            [-2, data_table[max_idx + 1], step - age, step - age].
          * If no sensor is out of range, action num_ch falls back to staying at this CH.
        """
        max_idx = -1
        max_age = float(step + 1)
        max_sens = None
        for sens in range(len(self.active_table)):
            if not self.active_table[sens]:
                if self.age_table[sens] < max_age:
                    max_idx = sens
                    max_age = self.age_table[sens]
                    max_sens = self.sens_table.iat[sens, 0]

        if max_sens is not None:
            sens_age = step - self.age_table[max_idx]
            decision_state.append([-2, self.data_table[max_idx + 1], sens_age, sens_age])
        else:
            decision_state.append([-2, 0, 0, 0])

        num_ch = len(full_sensor_list) - 1
        action = model.act(decision_state)

        if action == num_ch and max_sens is not None:
            target = max_sens
            self.last_target = max_idx
            targetType = False
        else:
            if action == num_ch:
                # Nothing to sweep: treat as "stay at this CH"
                action = self.cluster_num
            target = full_sensor_list.iat[action + 1, 0]
            targetType = True

        self.target_time = step
        model_help = True
        change_transit = True
        dist = math.sqrt(pow((target.id_x - self.id_x), 2) + pow((target.id_y - self.id_y), 2))
        self.tour = []

        AoI_peak = decision_state[1][2]
        AoI_avg = decision_state[1][3]
        for entry in range(num_ch - 1):
            AoI_avg += decision_state[entry + 2][3]
            AoI = decision_state[entry + 2][2]
            if AoI > AoI_peak:
                AoI_peak = AoI
        AoI_avg = math.ceil(AoI_avg / len(full_sensor_list))

        return self._finish_dest(state, model_p, p_count, model_help, change_transit, target,
                                 decision_state, action, dist, AoI_peak, AoI_avg, targetType)

class BaseStation(ClusterHead):
    """
    Grid-powered base station at the farmstead. It is a cluster head for the sensors within its
    backscatter range, but its battery never drains: it does not depend on the sun and always gives a
    docked UAV the full charge it asks for (no outage minutes). It does not move.
    """
    is_base = True

    def __init__(self, env, x, y, long, lat, sensor_list, cluster_num):
        super().__init__(env, x, y, long, lat, sensor_list, 0.0, cluster_num)
        self.typeStr = "BaseStation"
        self.stored_energy = 100.0
        self.perceived_soc = 100.0

    def reset(self):
        super().reset()
        self.stored_energy = 100.0

    def harvest_energy(self, env, step):
        self.is_solar = True          # mains power: always able to upload and run
        self.harvested_mAh = 0.0
        self.total_amp_spent = 0.0

    def step_battery(self):
        self.stored_energy = 100.0
        self.total_amp_spent = 0.0
        self.harvested_mAh = 0.0

    def charge_time(self, X: int, Y: int, charge, needed_wh: float = 0.0):
        docked = abs(self.curr_pt[0] - X) < 1.0 and abs(self.curr_pt[1] - Y) < 1.0
        return 60.0 if docked and charge and needed_wh > 0.0 else 0.0
