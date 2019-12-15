# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# ++++++++++++++++++++++SUMO Environment Interface++++++++++++++++++++++ #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
import cityflow
from sys import platform
import sys
import os
import numpy as np
import shutil
import json

import random
from queue import Queue  # LILO队列
import re

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# +++++++++++++systems identification for sumo environment++++++++++++++ #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
if platform == "linux" or platform == "linux2":  # this is linux
    os.environ['SUMO_HOME'] = '/usr/share/sumo'
    try:
        import traci
        import traci.constants as tc
    except ImportError:
        if "SUMO_HOME" in os.environ:
            print(os.path.join(os.environ["SUMO_HOME"], "tools"))
            sys.path.append(
                os.path.join(os.environ["SUMO_HOME"], "tools")
            )
            try:
                import traci
                import traci.constants as tc
            except ImportError:
                raise EnvironmentError("Please set SUMO_HOME environment variable or install traci as python module!")
        else:
            raise EnvironmentError("Please set SUMO_HOME environment variable or install traci as python module!")
elif platform == "win32":
    os.environ['SUMO_HOME'] = 'C:\\Program Files (x86)\\DLR\\Sumo'

    try:
        import traci
        import traci.constants as tc
    except ImportError:
        if "SUMO_HOME" in os.environ:
            print(os.path.join(os.environ["SUMO_HOME"], "tools"))
            sys.path.append(
                os.path.join(os.environ["SUMO_HOME"], "tools")
            )
            try:
                import traci
                import traci.constants as tc
            except ImportError:
                raise EnvironmentError("Please set SUMO_HOME environment variable or install traci as python module!")
        else:
            raise EnvironmentError("Please set SUMO_HOME environment variable or install traci as python module!")
elif platform == 'darwin':
    os.environ['SUMO_HOME'] = "/Users/{0}/sumo/sumo-git".format(os.getlogin())

    try:
        import traci
        import traci.constants as tc
    except ImportError:
        if "SUMO_HOME" in os.environ:
            print(os.path.join(os.environ["SUMO_HOME"], "tools"))
            sys.path.append(
                os.path.join(os.environ["SUMO_HOME"], "tools")
            )
            try:
                import traci
                import traci.constants as tc
            except ImportError:
                raise EnvironmentError("Please set SUMO_HOME environment variable or install traci as python module!")
        else:
            raise EnvironmentError("Please set SUMO_HOME environment variable or install traci as python module!")
else:
    sys.exit("platform error")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# +++++++++++++++++++++++Parameter Definition+++++++++++++++++++++++++++ #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
step_length = 15
yellow_len = 3
phase_num = 4
max_veh_num = 70

carWidth = 3
current_folder = os.getcwd()
config_file = os.path.join(current_folder, "setting/config.json")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# ++++++++++++++++++++++++Function Construct++++++++++++++++++++++++++++ #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

config_path = "./setting/config.json"
with open(config_path, 'r') as conf:
    conf_dict = json.loads(conf.read())
road_net_path = "./setting/"+conf_dict['roadnetFile']
with open(road_net_path, 'r') as RoadNet:
    RoadNet_dict = json.loads(RoadNet.read())


class Traffic_Env(object):

    # +++++++++++++++++++++Main Part for Interaction+++++++++++++++++++ #

    def __init__(self, obs_dim, action_dim, state_dim, INT_id_list):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.num_agent = len(INT_id_list)
        self.INT_id_list = INT_id_list
        self.step_length = step_length
        self.yellow_len = yellow_len
        self.phase_num = phase_num
        self.max_veh_num = max_veh_num

        self.CityFlow = cityflow.Engine(config_path, thread_num=16)
        self.state = {}
        self.obs = np.zeros((self.obs_dim, self.num_agent))
        self.RoadNet_dict = RoadNet_dict
        self.junction_num = len(self.RoadNet_dict['intersections'])
        self.ini_steps()
        self.update_state()
        self.update_obs()

    def ini_steps(self):
        for I_id in self.INT_id_list:
            self.CityFlow.set_tl_phase(I_id, 0)
        for ini_step in range(30):
            self.CityFlow.next_step()

    def reset(self):
        self.CityFlow.reset(seed=True)

    def get_state(self):
        self.update_state()

        return np.array(list(self.state.values())).reshape((self.state_dim))
        # return np.zeros(self.state_dim)

    def get_obs(self):
        self.update_obs()
        return self.obs
        # return np.zeros((self.obs_dim, self.num_agent))

    def get_agent_obs(self, agent_idx):

        return self.obs[:, agent_idx]
        # return np.zeros(self.obs_dim)

    def step(self, action_list):
        change_phase_INT_id_list = []
        for INT_idx in range(self.num_agent):
            action = action_list[INT_idx]
            # print('INT_idx:',INT_idx)
            # print('action:',action)
            if action == 1:
                I_id = self.INT_id_list[INT_idx]
                cur_phase = int(self.state[I_id][-1]*2)
                # print('id:', I_id)
                # print('cur_phase:',cur_phase)
                if cur_phase % 2 != 0:
                    print("Traffic light phase error!")
                self.CityFlow.set_tl_phase(I_id, int(cur_phase + 1))
                change_phase_INT_id_list.append(I_id)
        for step_idx in range(self.yellow_len):
            self.CityFlow.next_step()

        for change_INT_id in change_phase_INT_id_list:
            cur_phase = int(self.state[change_INT_id][-1]*2)
            self.CityFlow.set_tl_phase(change_INT_id, int((cur_phase+2) % self.phase_num))
        for step_idx in range(self.step_length - self.yellow_len):
            self.CityFlow.next_step()
        return False  # done

    def get_reward(self):
        reward = 0
        for I_id in self.INT_id_list:
            reward += -1*abs(sum(self.state[I_id][0:4]))

        return np.array(reward)
        # return np.zeros(1)


    # ++++++++++++++++++++++++Concrete function++++++++++++++++++++++++++++ #
    def update_state(self):
        self.lane_vehicle_count = self.CityFlow.get_lane_vehicle_count()
        # self.lane_waiting_vehicle_count = self.CityFlow.get_lane_waiting_vehicle_count()
        self.current_phase_idx = self.CityFlow.get_current_traffic_phase()
        # lane_veh = self.CityFlow.get_lane_vehicles()
        for Junc_idx in range(self.junction_num):
            junc_id = self.RoadNet_dict['intersections'][Junc_idx]['id']
            if junc_id in self.INT_id_list:
                # print(junc_id)
                self.state[junc_id] = []
                for road_idx in range(4):
                    road_veh_num = [0, 0]
                    for in_out_idx in range(2):
                        road_id = self.RoadNet_dict['intersections'][Junc_idx]['roads'][road_idx+in_out_idx*(7-road_idx)]
                        for lane_idx in range(3):
                            lane_id = road_id + '_' + str(lane_idx)
                            lane_veh_num = self.lane_vehicle_count[lane_id]
                            road_veh_num[in_out_idx] += lane_veh_num
                            # print(road_veh_num)
                    self.state[junc_id].append((road_veh_num[0] - road_veh_num[1])/self.max_veh_num)
                self.state[junc_id].append(self.current_phase_idx[junc_id]/2)

                # junc_roads_list = self.RoadNet_dict['intersections'][Junc_idx]['roads'][0:4]  #前半部分是 incoming road
                # for road_id in junc_roads_list:
                #     road_veh_num = 0
                #     for lane_idx in range(3):
                #         lane_id = road_id + '_' + str(lane_idx)
                #         lane_veh_num = self.lane_vehicle_count[lane_id] + self.lane_waiting_vehicle_count[lane_id]
                #         road_veh_num += lane_veh_num
                #     self.state[junc_id].append(road_veh_num)
                # self.state[junc_id].append(self.current_phase_idx[junc_id])

    def update_obs(self):
        for INT_idx in range(self.num_agent):
            self.obs[:, INT_idx] = np.array(self.state[self.INT_id_list[INT_idx]])




