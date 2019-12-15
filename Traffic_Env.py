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

carWidth = 3
current_folder = os.getcwd()
config_file = os.path.join(current_folder, "setting/config.json")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# ++++++++++++++++++++++++Function Construct++++++++++++++++++++++++++++ #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class Traffic_Env(object):

    # +++++++++++++++++++++Main Part for Interaction+++++++++++++++++++ #

    def __init__(self, obs_dim, action_dim, state_dim, num_agent):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.num_agent = num_agent

        self.CityFlow = cityflow.Engine(config_file, thread_num=16)
    def step(self, action_list):
        pass
        return False

    def reset(self):
        self.CityFlow.reset(seed=True)

    def get_obs(self):
        pass
        return np.zeros((self.obs_dim, self.num_agent))

    def get_agent_obs(self, agent):
        pass
        return np.zeros(self.obs_dim)

    def get_state(self):
        pass
        return np.zeros(self.state_dim)

    def get_reward(self, pre_state):
        pass
        return np.zeros(1)

    def close(self):
        pass

    # ++++++++++++++++++++++++Concrete function++++++++++++++++++++++++++++ #
