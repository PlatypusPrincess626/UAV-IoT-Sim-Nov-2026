"""
File: uav_iot_sim.py
Author: Mason Conkel
Creation Date: 2026-02-10
Description: This script manages the uav-iot simulation
"""
# Dependecies to Import
import pandas as pd
import numpy as np

# Simulation Files to Import
from sim_mods import environments, devices_iot, devices_uav


class make_env:
    def __init__(
        self,
        scene: str = "test",
        num_sensors: int = 50,
        num_ch: int = 5,
        num_uav: int = 1,
        max_num_steps: int = 720,
        map_seed: int = 0,
        map_res: int = 5,
        tree_cover: float = environments.TREE_COVER,
        patch_sigma_m: float = environments.FOLIAGE_PATCH_SIGMA_M,
        area_km2: float = 25.0,
        auto_clusterheads: bool = False,
        ambc_los: str = "map"
    ):
        self.scene = scene
        self._env = environments.sim_env(scene, num_sensors, num_uav, num_ch, max_num_steps,
                                         map_seed=map_seed, map_res=map_res,
                                         tree_cover=tree_cover, patch_sigma_m=patch_sigma_m,
                                         area_km2=area_km2, auto_clusterheads=auto_clusterheads,
                                         ambc_los=ambc_los)
        self.dim = self._env.dim

        self.uav_x = 0
        self.uav_y = 0
        self.sensor_pts = self._env.sensor_pts
        self.ch_pts = self._env.ch_pts

        self._num_sensors = num_sensors
        self.num_ch = self._env.total_clusterheads  # equals num_ch unless auto_clusterheads
        self._num_uav = num_uav
        self.max_num_steps = max_num_steps
        
        self.curr_step = 0
        self.last_action = 0
        self.archived_action = 0
        self.archived_paction = 0
        self.deciding_CH = 0

        self.full_reward = 0.0
        self.accum_reward = 0.0
        self.accum_steps = 0.0
        self.track_reward = False
        self.total_average = 0

        self.curr_state = [[0, 0, 0, 0] for _ in range(self.num_ch + 1)]
        self.archived_state = [[0, 0, 0, 0] for _ in range(self.num_ch + 1)]
        self.rewards = [0.0, 0.0, 0.0]
        self.accum_rewards = [0.0, 0.0, 0.0]
        self.archived_rewards = [0.0, 0.0, 0.0]

        self.full_reward2 = 0.0
        self.accum_reward_p = 0.0
        self.accum_steps_p = 0.0
        self.track_reward_p = False

        self.curr_pstate = [0, 0, 0, 0]
        self.archived_pstate = [0, 0, 0, 0]
        self.accum_rewardsp = [0.0, 0.0, 0.0]
        self.archived_rewardsp = [0.0, 0.0, 0.0]

        self.curr_reward = 0
        self.reward2 = 0

        self.curr_info = {
            "Last_Action": None,
            "Reward_Change": 0.0,          # -> Change in reward at step
            "Avg_Age": 0.0,               # -> avgAoI
            "Peak_Age": 0.0,              # -> peakAoI
            "Data_Distribution": 0.0,     # -> Distribution of Data
            "Total_Data_Change": 0.0,      # -> Change in Total Data
            "Total_Data": 0.0,            # -> Total Data
            "Crashed": False,
            "Truncated": False
        }

        self._aoi_threshold = 120
        self.step_cpu_mwh = 0.0
        self.truncated = False
        self.terminated = False
        self._curr_total_data = 0
    
    def reset(self):
        if self.scene == "test":
            for sensor in range(self._num_sensors):
                self._env.sensor_table.iat[sensor, 0].reset()
            for CH in range(self.num_ch):
                self._env.ch_table.iat[CH, 0].reset()
            for uav in range(self._num_uav):
                self._env.uav_table.iat[uav, 0].reset()

            self.curr_step = 0
            self.last_action = 0
            self.archived_action = 0
            self.archived_paction = 0
            self.deciding_CH = 0

            self.full_reward = 0.0
            self.accum_reward = 0.0
            self.accum_steps = 0.0
            self.track_reward = False
            self.total_average = 0

            self.curr_state = [[0, 0, 0, 0] for _ in range(self.num_ch + 1)]
            self.archived_state = [[0, 0, 0, 0] for _ in range(self.num_ch + 1)]
            self.rewards = [0.0, 0.0, 0.0]
            self.accum_rewards = [0.0, 0.0, 0.0]
            self.archived_rewards = [0.0, 0.0, 0.0]

            self.full_reward2 = 0.0
            self.accum_reward_p = 0.0
            self.accum_steps_p = 0.0
            self.track_reward_p = False

            self.curr_pstate = [0, 0, 0, 0]
            self.archived_pstate = [0, 0, 0, 0]
            self.accum_rewardsp = [0.0, 0.0, 0.0]
            self.archived_rewardsp = [0.0, 0.0, 0.0]

            self.curr_reward = 0
            self.reward2 = 0

            self.curr_info = {
                "Last_Action": None,
                "Reward_Change": 0.0,          # -> Change in reward at step
                "Avg_Age": 0.0,               # -> avgAoI
                "Peak_Age": 0.0,              # -> peakAoI
                "Data_Distribution": 0.0,     # -> Distribution of Data
                "Total_Data_Change": 0.0,      # -> Change in Total Data
                "Total_Data": 0.0,            # -> Total Data
                "Crashed": False, 
                "Truncated": False
            }

            self.truncated = False
            self.terminated = False
            self._curr_total_data = 0
            return self._env
        return None

    def step(self, model, model_p):
        train_model = False
        train_CH = 0
        old_action = 0
        comms, move, harvest = 0, 0, 0
        self.total_average = 0
        last_CH = None
        curr_CH = None

        train_p = False
        old_paction = 0
        old_pstate = [0, 0, 0, 0]
        self.reward2 = 0

        old_state = [[0, 0, 0, 0] for _ in range(self.num_ch + 1)]

        if not self.terminated:
            self.truncated = False
            if self.curr_step < self.max_num_steps:
                for sens in range(self._num_sensors):
                    self._env.sensor_table.iat[sens, 0].harvest_energy(self._env, self.curr_step)
                    self._env.sensor_table.iat[sens, 0].harvest_data()

                for CH in range(self.num_ch):
                    self._env.ch_table.iat[CH, 0].harvest_energy(self._env, self.curr_step)
                    self._env.ch_table.iat[CH, 0].download(self.curr_step)

                for UAV in range(self._num_uav):
                    uav = self._env.uav_table.iat[UAV, 0]
                    train_model, DCH, used_model, train_p, state, action, action_p, p_state, comms, move, harvest = \
                        (uav.set_dest(model, model_p, self.curr_step))
                    uav.navigate_step(self._env)
                    self.uav_x = uav.id_x
                    self.uav_y = uav.id_y

                    train_model2, change_archives = uav.receive_data(self.curr_step)
                    charge_reward = uav.receive_energy()

                    self._env.uav_table.iat[UAV, 0].step_battery()
                    # This minute's UAV energy use (mWh), returned to run.py for logging
                    comms, move, harvest = uav.step_comms_mwh, uav.step_move_mwh, uav.step_charge_mwh
                    self.step_cpu_mwh = uav.step_cpu_mwh

                    crashed = bool(uav.crash) or uav.stored_energy <= 0
                    self.reward(charge_reward, crashed)

                    # train_model = True
                    if train_model or train_model2:
                        train_model = True
                        train_CH = self.deciding_CH
                        self.full_reward = self.accum_reward / max(self.accum_steps, 1)
                        # Reward vector [R_peak, R_avg, R_chg]; each agent weights it (Eqs. 22, 23)
                        self.archived_rewards = np.array(self.rewards, dtype=float)
                    if crashed:
                        # Eq. (21): mission failure gives both models a reward of 0
                        self.archived_rewards = np.zeros(3)

                    self.curr_state = uav.state
                    self.last_action = action
                    self.terminated = crashed
                    if crashed:
                        self.archived_rewardsp = np.zeros(3)   # Eq. (21) for the power model too
                    self.curr_pstate = p_state

                    old_state = self.archived_state
                    old_action = self.archived_action
                    old_pstate = self.archived_pstate
                    old_paction = self.archived_paction

                    if used_model:
                        self.archived_state = state
                        self.archived_action = action
                        self.deciding_CH = DCH
                        self.track_reward = True
                        self.accum_rewards = self.rewards

                    """Train Dual Model"""
                    if train_p:
                        if self.curr_step < 15:
                            train_p = False
                        else:
                            self.full_reward2 = np.array(self.accum_reward_p / max(self.accum_steps_p, 1))
                            self.archived_rewardsp = np.array(self.rewards, dtype=float)
                        self.archived_pstate = p_state
                        self.archived_paction = action_p
                        self.track_reward_p = True
                        self.accum_rewardsp = self.rewards

                # Sensor and CH batteries are stepped once per minute, after every UAV has drawn on them
                # (previously inside the UAV loop, so with >1 UAV they would be stepped more than once)
                for sens in range(self._num_sensors):
                    self._env.sensor_table.iat[sens, 0].step_battery()
                for CH in range(self.num_ch):
                    self._env.ch_table.iat[CH, 0].step_battery()

                if self.track_reward:
                    self.accum_steps += 1
                    self.accum_reward += self.curr_reward

                if self.track_reward_p:
                    if self.terminated:
                        self.accum_reward_p = 0
                    else:
                        self.accum_steps_p += 1
                        self.accum_reward_p += self.reward2

                self.curr_step += 1

            else:
                self.truncated = True
                self.curr_info = {
                    "Last_Action": self.last_action,
                    "Reward_Change": 0,        # -> Change in reward at step
                    "Avg_Age": 0,                   # -> avgAoI
                    "Peak_Age": 0,                 # -> peakAoI
                    "Data_Distribution": 0,     # -> Distribution of Data
                    "Total_Data_Change": 0,      # -> Change in Total Data
                    "Total_Data": self._curr_total_data, # -> Total Data
                    "Crashed": self.terminated,         # -> True if UAV is crashed
                    "Truncated": self.truncated         # -> Max episode steps reached
                }
        else:
            self.curr_reward = 0
            self.curr_info = {
                "Last_Action": self.last_action,
                "Reward_Change": 0,        # -> Change in reward at step
                "Avg_Age": 0,                   # -> avgAoI
                "Peak_Age": 0,                 # -> peakAoI
                "Data_Distribution": 0,     # -> Distribution of Data
                "Total_Data_Change": 0,      # -> Change in Total Data
                "Total_Data": self._curr_total_data, # -> Total Data
                "Crashed": self.terminated,         # -> True if UAV is crashed
                "Truncated": self.truncated         # -> Max episode steps reached
            }

        return train_model, train_p, old_state, old_action, old_paction, old_pstate, comms, move, harvest

    def reward(self, charge_reward, crashed=False):
        '''
        Rewards from the paper (Sec. 5.1):
            R_avg  = 1 - AoI_avg / t_step                                   Eq. (18)
            R_peak = 1 - AoI_max / AoI_sat, or 0 once AoI_max >= AoI_sat     Eq. (19), AoI_sat = 120 min
            R_chg  = B_uav if B_uav < 1, else 0                             Eq. (20), from receive_energy
            R      = 0 if the UAV's battery is depleted, else the weighted sum  Eq. (21)
            targeting (DDAQN): 0.7 R_peak + 0.3 R_avg + 0 R_chg              Eq. (23)
            charging (DDQN):   1/4 R_peak + 1/4 R_avg + 1/2 R_chg            Eq. (22)
        Data distribution and total data are tracked for logging only.
        '''
        totalAge = 0
        peakAge = 0
        minAge = self.curr_step

        for index in range(len(self.curr_state) - 1):
            peak_age = self.curr_state[index + 1][2]
            average_age = self.curr_state[index + 1][3]

            totalAge += average_age
            if peak_age > peakAge:
                peakAge = peak_age
            if peak_age < minAge:
                minAge = peak_age
        avgAge = float(totalAge) / float(self.num_ch)
       
        dataChange = 0
        maxColl = 0.0
        minColl = 1.0
        index: int

        for index in range(len(self.curr_state)):
            if index > 0:
                val = self.curr_state[index][1] / (max(self._curr_total_data, 1))
                if val > maxColl:
                    maxColl = val
                if val < minColl:
                    minColl = val
            else:
                dataChange = max(0, self.curr_state[index][1] - self._curr_total_data)
                self._curr_total_data += abs(round(dataChange))
        
        distOffset = maxColl - minColl

        rewardDist = 1 - distOffset

        rewardPeak = (1-peakAge/self._aoi_threshold) if (1-peakAge/self._aoi_threshold) > 0 else 0

        # Eq. (18); clipped at 0 because a CH's average age can briefly exceed t_step at the very start
        # of an episode. The agents always received the clipped value; the logged rewards now match it.
        rewardAvgAge = max(1 - avgAge/max(self.curr_step, 1), 0.0)


        rewardDataChange = dataChange / 1_498_500

        rewardChange = 0.7 * rewardPeak + 0.3 * rewardAvgAge

        reward_energy = min(max(charge_reward, 0), 1)  # QuadUAV.receive_energy returns a value in [0, 1]
        reward2Change = 0.25 * rewardPeak + 0.25 * rewardAvgAge + 0.5 * reward_energy

        if crashed:
            print("Died")
            rewardChange = 0
            reward2Change = 0
        
        self.curr_info = {
            "Last_Action": max(self.last_action, -1),
            "Reward_Change": rewardChange,        # -> Change in reward at step
            "Avg_Age": avgAge,                   # -> avgAoI
            "Peak_Age": peakAge,                 # -> peakAoI
            "Data_Distribution": distOffset,     # -> Distribution of Data
            "Total_Data_Change": dataChange,      # -> Change in Total Data
            "Total_Data": self._curr_total_data, # -> Total Data
            "Crashed": self.terminated,         # -> True if UAV is crashed
            "Truncated": self.truncated         # -> Max episode steps reached
        }
        
        self.curr_reward = rewardChange
        self.reward2 = reward2Change

        self.rewards = [max(rewardPeak, 0), max(rewardAvgAge, 0), max(reward_energy, 0)]
        if crashed:
            self.rewards = [0.0, 0.0, 0.0]