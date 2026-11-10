"""
File: devices_iot.py
Author: Mason Conkel
Creation Date: 2025-10-06
Description: This script simulates the uav devices to be used in the simulation of a remote IoT network.
"""
import math
import pandas as pd
import numpy as np

from sim_mods import ablation
from sim_mods.ablation import CFG


class QuadUAV:
    def __init__(self, X: int, Y: int, long: float, lat: float, uavNum: int, CHList: list):
        self.action = None
        self.type = 3
        self.serial = uavNum
        self.typeStr = "UAV"

        # Communication Specifications
        self._comms = {
            "max_dist_lora": 5_000,  # LoRa maximum distance (m)
            "max_bitrate_lora": 24_975,  # LoRa maximum bitrate (bps)
            "current_active_lora": 4_200,  # LoRa active current (muA)
            "current_sleep_lora": 1_200,  # LoRa dormant current (muA)
            "voltage_lora": 3.7,  # LoRa voltage requirements (V)
            "pow_active_lora": 20,  # LoRa active power (muW)
            "pow_sleep_lora": 4,  # LoRa dormant power (muW)

            "max_dist_ambc": 500,  # AmBC maximum distance (m); the airborne UAV has line of sight
            "max_bitrate_ambc": 1_592,  # AmBC maximum bitrate (bps)
            "pow_ambc": 3,  # AmBC upkeep power (muW)
            "voltage_ambc": 3.3,  # AmBC voltage requirements (V)
            "current_ambc": 785  # AmBC upkeep current (muA)
        }

        # Positioning
        self.id_x = int(X)
        self.id_y = int(Y)
        self.maxH = 20
        self.h = 1

        # Trajectory
        self.target = None
        self.targetHead = None
        self.last_Head = None
        self.targetSerial = 0
        self.tour = []
        self.tour_iter = 0

        self.target_x = int(X)
        self.target_y = int(Y)
        self.lat = lat
        self.long = long
        self.last_AoI = 0
        self.bad_target = 0

        # Movement
        self.maxSpd = 15  # 15 m/s max speed cap
        self.maxTurn = 200  # 200 degree per sec max yaw
        self.maxClimb = 3  # 3 m/s ascent and descent
        self.inRange = True

        self.p_cycle = 0
        self.p_count = 0

        # State
        self.crash = False
        self.model_transit = False
        self.no_hold = True
        self.force_count = 0
        self.targetType = True

        self.origin_state = None
        self.origin_action = None

        # Pandas version of state used for environment comparisons
        ch_list = [len(CHList)]

        for ch in range(len(CHList)):
            ch_list.append(CHList[ch])

        self.full_sensor_list = pd.DataFrame(np.array(ch_list))
        self.full_sensor_list.rename(
            columns={0: "Device"},
            inplace=True
        )

        # Battery Usage
        self.max_energy = 6_800  # 6800 mAh
        self.battery_voltage = 3.7  # V (single-cell Li-ion); used to convert charge to energy
        self.cpu_pow = 3.7  # milli-watts/ 2 micro Joule
        self.cpu_amps = 1_000  # micro-amps

        self.charge_rate = 1/3  # 20 min charging time
        self.flight_discharge = 0.75  # 45 min flight time
        self.flight_amps = self.max_energy / self.flight_discharge # A in 1 min
        self.launch_time = self.maxH / self.maxClimb # s
        self.launch_cost = 18.889 * self.launch_time / 60 # mA

        self.is_charging = False

        self.stored_energy = 100  # Initialize at full battery
        self.total_amp_spent = 0.0
        self.percent_gain = 0.0
        self.step_charge = 100 / (self.charge_rate * 60)
        self.step_discharge = 100 / (self.flight_discharge * 60)

        # State used for model
        """
        Old state +1, Date +2, New +1
        """
        self.state = [[0, 0, 0, 0] for _ in range(len(CHList) + 1)]

        self.state[0][0], self.state[0][1], self.state[0][2] = -1, 0, 100
        count = 0
        for row in range(len(self.state) - 1):
            self.state[row + 1][0] = count
            count += 1

        self.step_comms_cost = 0
        self.step_move_cost = 0
        self.energy_harvested = 0
        self.step_move_mwh = self.step_comms_mwh = self.step_cpu_mwh = self.step_charge_mwh = 0.0

    short_minutes = 0      # consecutive docked minutes the CH supplied no charge
    seek_charge = False    # flying to a charger after a diversion; charge on arrival

    def reset(self):
        self.short_minutes = 0
        self.seek_charge = False
        # Reset Flags
        self.stored_energy = 100
        self.total_amp_spent = 0.0
        self.percent_gain = 0.0

        self.crash = False
        self.model_transit = False
        self.is_charging = False
        self.no_hold = True
        self.force_count = 0
        self.h = 1

        self.target = None
        self.targetHead = None
        self.last_Head = None

        self.tour = []
        self.tour_iter = 0

        self.bad_target = 0
        self.targetType = True

        self.step_move_cost = 0
        self.step_comms_cost = 0
        self.energy_harvested = 0
        self.last_AoI = 0
        self.p_cycle = 0
        self.p_count = 0

        # Reset State
        self.state[0][0], self.state[0][1], self.state[0][2] = -1, 0, 100

        for row in range(len(self.state) - 1):
            self.state[row + 1][1] = 0
            self.state[row + 1][2] = 0
            self.state[row + 1][3] = 0

    def battery_calc(self, stored_percentage, max_storage, amp_spent, step_charge):
        """
        One-minute battery update, in percent.
            amp_spent   average current drawn over this minute (mA)
            max_storage battery capacity (mAh)
            step_charge percentage points gained this minute
        I mA for one minute uses I / 60 mAh, i.e. 100 * I / (60 * capacity) percent.
        (Previously the factor of 100 was missing.)
        """
        stored_percentage -= 100.0 * amp_spent / (max_storage * 60)
        stored_percentage = min(100.0, (stored_percentage + step_charge))

        return max(0.0, stored_percentage)

    def step_battery(self):
        self.stored_energy = self.battery_calc(self.stored_energy, self.max_energy, self.total_amp_spent, self.percent_gain)
        # Clear the per-minute accumulator (previously never cleared, so drain compounded every step)
        self.total_amp_spent = 0.0

        self.state[0][2] = self.stored_energy
        return 0

    # Internal UAV Mechanics
    def _mwh(self, ma_over_minute: float) -> float:
        """Energy (mWh) of an average current (mA) drawn for one minute at the battery voltage."""
        return ma_over_minute / 60.0 * self.battery_voltage

    def navigate_step(self, env: object):
        # Per-minute energy log (mWh), filled by navigate_step, receive_data and receive_energy
        self.step_move_mwh = 0.0      # flight + launch/landing
        self.step_comms_mwh = 0.0     # radio (LoRa / AmBC reader)
        self.step_cpu_mwh = 0.0       # processor
        self.step_charge_mwh = 0.0    # energy taken in while docked
        self.energy_harvested = 0
        move_before = self.total_amp_spent

        self.total_amp_spent += self.cpu_amps / 1_000
        self.step_cpu_mwh = self._mwh(self.cpu_amps / 1_000)
        move_before = self.total_amp_spent
        if (math.sqrt(pow(self.id_x - self.target_x, 2) + pow(self.id_y - self.target_y, 2))
                < self._comms["max_dist_ambc"]):
            self.target_x, self.target_y = self.target.curr_pt

        maxDist = math.sqrt(pow(self.id_x - self.target_x, 2) + pow(self.id_y - self.target_y, 2))

        if maxDist < 1.0:
            if self.h == 1 and self.target.type == 2:
                self.h = 0
                self.total_amp_spent += self.launch_cost

        elif self.stored_energy > self.step_discharge:
            time = 0.0
            if self.h == 0:
                self.h = 1
                self.total_amp_spent += self.launch_cost
                time += self.launch_time

            if maxDist < self.maxSpd * ((60.0 - time) - self.launch_time):
                self.id_x = self.target_x
                self.id_y = self.target_y
                time = maxDist / self.maxSpd
                if self.target.type == 2:
                    self.h = 0
                    self.total_amp_spent += self.launch_cost

            else:
                time = (60.0 - time)
                vectAngle = math.atan(abs(self.target_y - self.id_y) / max(abs(self.target_x - self.id_x), 1))  # Returns radians
                directionX = (self.target_x - self.id_x) / max(abs(self.target_x - self.id_x), 1)
                directionY = (self.target_y - self.id_y) / max(abs(self.target_y - self.id_y), 1)
                self.id_x += math.floor(directionX * self.maxSpd * time * math.cos(vectAngle))
                self.id_y += math.floor(directionY * self.maxSpd * time * math.sin(vectAngle))

            self.total_amp_spent += self.flight_amps * time / 60

        else:
            self.crash = True

        self.step_move_mwh = self._mwh(self.total_amp_spent - move_before)
        self.step_move_cost = self.step_move_mwh

    # Finish with battery drain
    # UAV-IoT Communication
    def receive_data(self, step):
        comms_before = self.total_amp_spent
        totalData = 0
        device = self.target
        train_model = False
        change_archives = False

        if self.target.type == 1:
            dataReturn = max(0, device.upload(int(self.id_x), int(self.id_y)))

            if math.sqrt(pow((self.id_x - self.target.id_x), 2) + pow((self.id_y - self.target.id_y), 2)) < \
                    self._comms.get("max_dist_ambc"):

                totalData += dataReturn

                if self.tour_iter < len(self.tour):
                    self.target = self.tour[self.tour_iter]
                    self.tour_iter += 1
                    train_model, change_archives = False, False
                else:
                    # Assess Reward at this point
                    self.target = self.targetHead
                    self.targetType = True
                    self.p_cycle = 0
                    train_model, change_archives = False, False

                self.target_x = self.target.id_x
                self.target_y = self.target.id_y
                self.inRange = True

            else:
                self.inRange = False

            totalTime = totalData / self._comms.get("max_bitrate_ambc")
            self.total_amp_spent += (totalTime * self._comms.get("current_active_lora")
                                     + (60 - totalTime) * self._comms.get("current_sleep_lora")) / 60 / 1_000  # muA -> mA

            self.state[self.last_Head + 1][2] = step - self.last_AoI
            self.state[self.last_Head + 1][1] += totalData
            self.state[0][1] += totalData
            self.targetSerial = self.targetHead.cluster_num


        else:
            if math.sqrt(pow((self.id_x - self.target.id_x), 2) + pow((self.id_y - self.target.id_y), 2)) < \
                    self._comms.get("max_dist_lora"):

                self.inRange = True

                dataReturn, self.last_AoI, avg_AoI = device.upload(int(self.id_x), int(self.id_y))
                dataReturn = max(0, dataReturn)
                totalData += dataReturn

                totalTime = totalData / self._comms.get("max_bitrate_lora")
                self.total_amp_spent += (totalTime * self._comms.get("current_active_lora")
                                         + (60 - totalTime) * self._comms.get("current_sleep_lora")) / 60 / 1_000  # muA -> mA

                # ADF 2
                self.state[self.targetSerial + 1][2] = step - self.last_AoI
                self.state[self.targetSerial + 1][3] = step - avg_AoI
                self.state[self.targetSerial + 1][1] += totalData

                self.state[0][1] += totalData
            else:
                totalTime = 4
                self.total_amp_spent += (totalTime * self._comms.get("current_active_lora")
                                         + (60 - totalTime) * self._comms.get("current_sleep_lora")) / 60 / 1_000  # muA -> mA
                self.inRange = False

        for CH in range(len(self.full_sensor_list) - 1):
            if device.type == 1:
                self.state[CH + 1][2] += 1
                self.state[CH + 1][3] += 1
            elif device.cluster_num != self.full_sensor_list.iat[CH + 1, 0].cluster_num:
                self.state[CH + 1][2] += 1
                self.state[CH + 1][3] += 1
            elif not self.inRange:
                self.state[CH + 1][2] += 1
                self.state[CH + 1][3] += 1

        self.step_comms_mwh = self._mwh(self.total_amp_spent - comms_before)
        self.step_comms_cost = self.step_comms_mwh
        return train_model, change_archives

    def receive_energy(self):
        """
        Charges the UAV if it is docked at a cluster head and returns the charging reward term
        (--energy-reward):
            paper (default)  Eq. (20): R_chg = B_uav if B_uav < 1, else 0. B_uav is the battery fraction
                             after this minute (Eq. 14: stored + charge gained - energy spent), before
                             capping at full, so an overcharging minute scores 0.
            overflow         share of this minute's charge that overflowed a full battery (experimental).
            level            battery level after this minute's charge, capped at 1 (older UAV.py).
        """
        excess_percent = 0.0
        self.percent_gain = 0.0
        if self.target.type == 2 and self.h == 0:
            rate = 2.0 if self.stored_energy < 80 else 0.5          # taper above 80 %
            needed_wh = rate * self.step_charge / 100.0 * self.max_energy * self.battery_voltage / 1_000.0
            t = self.target.charge_time(int(self.id_x), int(self.id_y), self.is_charging, needed_wh)

            if (self.is_charging and abs(self.id_x-self.target_x) < 1.0 and
                    abs(self.id_y-self.target_y) < 1.0 and t < 1.0):
                self.no_hold = False
            if self.is_charging and abs(self.id_x - self.target_x) < 1.0 and abs(self.id_y - self.target_y) < 1.0:
                self.short_minutes = self.short_minutes + 1 if t < 1.0 else 0

            if self.stored_energy < 80:
                self.percent_gain = 2 * self.step_charge * (t / 60)
            else:
                self.percent_gain = 0.5 * self.step_charge * (t / 60)

            excess_percent = self.stored_energy + self.percent_gain - 100

            self.energy_harvested = self.percent_gain
            self.step_charge_mwh = self.percent_gain / 100.0 * self.max_energy * self.battery_voltage

        if CFG.energy_reward == "paper":
            spent = 100.0 * self.total_amp_spent / (self.max_energy * 60)   # this minute's drain, %
            b_uav = (self.stored_energy + self.percent_gain - spent) / 100.0
            return float(b_uav) if 0.0 < b_uav < 1.0 else 0.0
        if CFG.energy_reward == "level":
            return min(100.0, self.stored_energy + self.percent_gain) / 100.0
        if self.percent_gain <= 0.0:
            return 0.0
        return float(np.clip(excess_percent / self.percent_gain, 0.0, 1.0))

    def set_dest(self, model, model_p, step, _=None):
        train_model = False
        DCH = 0
        used_model = False
        train_p = False
        action_p = 0
        p_state = [0, 0, 0, 0]

        if self.targetHead is not None:
            self.last_Head = self.targetHead.cluster_num

        if self.target is None:
            minDist = 10_000.0
            minCH = self.full_sensor_list.iat[1, 0]
            for CH in range(len(self.full_sensor_list.index) - 1):
                dist = math.sqrt(pow((self.id_x - self.full_sensor_list.iat[CH + 1, 0].id_x), 2)
                                 + pow((self.id_y - self.full_sensor_list.iat[CH + 1, 0].id_y), 2))
                if dist < minDist:
                    minDist = dist
                    minCH = self.full_sensor_list.iat[CH + 1, 0]

            self.target = minCH
            self.targetHead = minCH
            self.target_x = minCH.id_x
            self.target_y = minCH.id_y
            self.targetSerial = self.targetHead.cluster_num
            self.targetType = False
            DCH = 0

        elif not self.inRange:
            self.target = self.target
            self.p_cycle -= 1
            self.force_count += 1

        elif self.target.type == 1:
            self.target = self.target
            self.p_cycle -= 1
            self.force_count += 1

        else:
            ### TargetType > sensor

            # The fleet's reading of this CH's battery (shared through the fleet's comms)
            self.target.perceived_soc = 100.0 if getattr(self.target, "is_base", False) else self.target.stored_energy

            # True, True, sensor, CHstate, action, action_p
            (used_model, changed_transit, dest, state, action, action_p, p_state,
             dist, peak, avg, tour, self.targetType) =\
                self.target.get_dest(self.state, self.full_sensor_list, model, model_p,
                                     step, self.p_count, self.targetType)

            self.p_cycle -= 1

            if self.targetType:
                if dest.cluster_num == self.targetSerial:
                    self.force_count += 1
                else:
                    self.force_count = 1
            else:
                self.force_count += 1

            DCH = self.targetSerial
            self.action = action


            if self.model_transit and changed_transit:
                self.model_transit = False

            if self.h == 0:
                # Committed charging minutes count down while docked, but never below 0. Static charging
                # never sets p_count, so without the floor it fell by 1 every decision (to about -140 per
                # episode) and the projected decision state subtracted that from every cluster's age.
                self.p_count = max(self.p_count - 1, 0)


            """
            Charging policy (ablation axis: --charging)
            """
            if CFG.charging == "static":
                train_p = self._charge_static()
            elif CFG.charging == "dynamic":
                train_p = self._charge_dynamic(dist, peak)
            else:
                train_p = self._charge_model(action_p)

            if CFG.charger_divert and self.seek_charge:
                # Arrived at the charger chosen by a diversion: charge before anything else
                self.seek_charge = False
                self.is_charging = True
                self.p_count = max(self.p_count, 20)
                self.p_cycle = 30
                self.short_minutes = 0

            if CFG.charging != "static":
                # Shared exit conditions for the p_count-based policies (dynamic and model)
                if changed_transit and self.p_count < 1.0:
                    self.p_cycle = 0

                elif self.is_charging and (self.p_count < 1.0 or self.state[0][2] > 0.8 * ablation.batt_full()):
                    self.p_count = 0
                    self.is_charging = False

            if CFG.charger_divert:
                dest = self._divert_if_needed(dest)

            if self.is_charging:
                used_model = False
                self.target = self.target
                return (train_model, DCH, used_model, train_p, state, action, action_p, p_state,
                        self.step_comms_cost, self.step_move_cost, self.energy_harvested)

            else:
                self.no_hold = True
                self.is_charging = False
                if used_model:
                    self.model_transit = True


                if dest.type == 1:
                    """
                    Old targeting
                    """
                    # self.tour = []
                    # self.tour_iter = 1
                    """
                    New targeting
                    """
                    self.tour = tour
                    self.tour_iter = 1

                    self.target = dest
                    self.target_x = dest.id_x
                    self.target_y = dest.id_y

                    return (train_model, DCH, used_model, train_p, state, action, action_p, p_state,
                            self.step_comms_cost, self.step_move_cost, self.energy_harvested)

                else:
                    if dest.cluster_num == self.targetSerial:
                        self.targetType = True
                        self.bad_target = 1
                    else:
                        self.targetType = False
                        self.bad_target = 0

                    if changed_transit:
                        train_model = True

                    self.target = dest
                    self.targetHead = dest
                    self.targetSerial = self.targetHead.cluster_num
                    self.target_x = dest.id_x
                    self.target_y = dest.id_y

                    return (train_model, DCH, used_model, train_p, state, action, action_p, p_state,
                            self.step_comms_cost, self.step_move_cost, self.energy_harvested)

        return (train_model, DCH, used_model, train_p, self.state, self.targetHead.cluster_num, action_p, p_state,
                self.step_comms_cost, self.step_move_cost, self.energy_harvested)

    # ---- Charger diversion (CFG.charger_divert) ------------------------------
    def _leg_pct(self, dist):
        return CFG.tour_margin * ablation.minutes_pct(ablation.leg_minutes(dist))

    def _pick_charger(self):
        """
        Best reachable charger other than the current one: score = distance / rating, rating = the
        fleet's perceived usable charge (base station always 1). A UGV qualifies only if its perceived
        charge is at least divert_min_soc. Reachable = arrives with at least 2 % left.
        """
        best, best_score = None, float("inf")
        for row in range(1, len(self.full_sensor_list.index)):
            ch = self.full_sensor_list.iat[row, 0]
            if ch is self.target:
                continue
            base = getattr(ch, "is_base", False)
            if not base and ch.perceived_soc < CFG.divert_min_soc:
                continue
            d = math.hypot(ch.id_x - self.id_x, ch.id_y - self.id_y)
            if self.state[0][2] - self._leg_pct(d) < 2.0:
                continue
            score = d / max(ablation.charger_rating(ch), 1e-3)
            if score < best_score:
                best, best_score = ch, score
        return best

    def _divert_if_needed(self, dest):
        """
        Returns the destination to fly to. Diverts to _pick_charger() when
          * the docked CH has supplied no charge for divert_after minutes (it is at its reserve), or
          * the next CH leg would leave less than the 20 % reserve plus the trip from there to a
            guaranteed charger, and the CH here cannot supply. If it can, the UAV charges here instead.
        """
        here = self.target
        here_supplies = getattr(here, "is_base", False) or (here.stored_energy > 10.0 + 1e-6
                                                           and self.short_minutes < CFG.divert_after)
        if self.is_charging:
            if self.short_minutes < CFG.divert_after:
                return dest
        elif dest is not None and dest.type == 1 and getattr(here, "plan_short", False) and here_supplies \
                and self.state[0][2] < 90.0:
            # The battery, not the need, cut the sortie short and this CH can supply: top up first
            # (towards 90 %), then the next decision re-plans the sortie with the larger budget
            self.is_charging = True
            self.p_count = max(self.p_count, min(20, math.ceil((90.0 - self.state[0][2]) / (2 * self.step_charge))))
            return dest
        elif dest is not None and dest.type == 2:
            need = (self._leg_pct(math.hypot(dest.id_x - self.id_x, dest.id_y - self.id_y)) + 20.0
                    + (self._leg_pct(dest.charger_m) if dest.charger_m else 0.0))
            if self.state[0][2] >= need:
                return dest
            if here_supplies:
                self.is_charging = True
                self.p_count = max(self.p_count, 10)
                return dest
        else:
            return dest
        ch = self._pick_charger()
        if ch is None:
            return dest                     # nowhere better reachable: keep the current plan
        self.is_charging = False
        self.p_count = 0
        self.short_minutes = 0
        self.seek_charge = True
        return ch

    # ---- Charging policies ------------------------------------------------
    # Each runs once per decision while the UAV sits at a cluster head, sets self.is_charging
    # (and for the p_count policies self.p_count, the minutes of charging still committed),
    # and returns train_p: whether the power agent should store a transition this step.

    def _charge_static(self):
        """
        Old static charging: charge when the battery falls below 20% and keep charging until full.
        No power agent is consulted.
        """
        if self.state[0][2] < 0.2 * ablation.batt_full():
            self.is_charging = True
        elif self.is_charging and self.state[0][2] >= ablation.batt_full():
            self.is_charging = False
        return False

    def _charge_dynamic(self, dist, peak):
        """
        Old dynamic charging: every 30 decisions, pick a charging duration from a heuristic.
          1) Emergency: below 20%, charge for 1.5x the time the next leg needs.
          2) Look-ahead: if flying the next leg (with 25% margin) would drop below 20%, same as 1.
          3) Opportunistic: if peak AoI <= 240 and battery < 80%, top up for 0.5x the leg time.
        Capped at 20 minutes. No power agent is consulted.
        """
        if self.p_cycle < 1.0:
            self.p_cycle = 30
            leg_charge = (dist / self.maxSpd) * (self.flight_discharge / self.charge_rate)
            per_s = ablation.flight_cost_per_s(self.flight_discharge)  # percentage points per second
            full = ablation.batt_full()

            if self.state[0][2] < 0.2 * full:
                self.p_count = min(20, round(1.5 * leg_charge))
            elif self.state[0][2] - 1.25 * (dist / self.maxSpd) * per_s <= 0.2 * full:
                self.p_count = min(20, round(1.5 * leg_charge))
            elif peak <= 240 and self.state[0][2] < 0.8 * full:
                self.p_count = min(20, round(0.5 * leg_charge))
            else:
                self.p_count = 0

            self.is_charging = self.p_count > 0
        return False

    def _charge_model(self, action_p):
        """
        Model charging (current default): every 30 decisions the power agent (get_ddqn_agentp)
        chooses a charging duration action_p in minutes. Between those decisions, a lower
        suggestion from the agent pulls the committed time down to the average of the two.
        """
        train_p = False
        if action_p < self.p_count:
            self.p_count = math.ceil((self.p_count + action_p) / 2)

        if self.p_cycle < 1.0:
            self.p_cycle = 30
            self.p_count = action_p
            train_p = True
            self.is_charging = self.p_count > 0
        return train_p