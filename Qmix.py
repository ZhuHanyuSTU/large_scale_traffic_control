from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from Traffic_Env import Traffic_Env
import os
import random
import copy
from tkinter import _flatten
import scipy.io as scio
import time
import shutil


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# ++++++++++++++++++++++++++DNN network definition++++++++++++++++++++++ #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class DNNAgent(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DNNAgent, self).__init__()
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_actions)
        self.init_weight()

    def init_weight(self):
        layers = [self.fc1, self.fc2, self.fc3]
        [nn.init.xavier_normal_(layer.weight) for layer in layers]

    # +++++++++++++++++++++++DNN forward function++++++++++++++++++++++ #
    def forward(self, inputs):
        x1 = F.relu((self.fc1(inputs)))
        x2 = F.relu((self.fc2(x1)))
        q = self.fc3(x2)  # output unity values of all actions for an agent
        return q


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# +++++++++++++++++++++++Mixing network definition++++++++++++++++++++++ #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class QMixer(nn.Module):
    def __init__(self, num_agents, state_dim, mixing_embed_dim, obs_dim, action_dim, greedy=False):
        super(QMixer, self).__init__()
        self.greedy = greedy
        self.num_agents = num_agents
        self.state_dim = int(np.prod(state_dim))
        self.embed_dim = mixing_embed_dim
        for i in range(self.num_agents):
            exec("self.Agents_"+str(Agent_id_list[i])+"=DNNAgent(obs_dim, action_dim)", {"DNNAgent": DNNAgent}, {"self": self,"obs_dim":obs_dim,"action_dim":action_dim})
        # self.Agents = DNNAgent(obs_dim, action_dim)  # Agent Net in cloud node
        self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.num_agents)
        self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        self.init_weight()

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def init_weight(self):
        layers = [self.hyper_w_1, self.hyper_w_final]
        [th.nn.init.xavier_normal_(layer.weight) for layer in layers]

    # +++++++++++++++++++++++Mixing forward function++++++++++++++++++++++ #
    def forward(self, agents_obs, agents_action, states):
        # input definition:
        # agents_obs: the observation of all agents, batch_size*O_dim*NumAgent
        # agents_action: the action of all agents, batch_size*A_dim*NumAgent
        # states: the state of whole system, batch_size*S_dim
        bs = agents_obs.size()[0]
        for agent_idx in range(self.num_agents):
            # make decision for each agent
            loc = locals()
            exec('agent_q_values = self.Agents_' + str(Agent_id_list[agent_idx]) + '.forward(agents_obs[:, :, agent_idx])')
            agent_q_values = loc['agent_q_values']
            if self.greedy is True:
                agent_q = th.max(agent_q_values, dim=1)[0]  # for target network to eval target Q_total
            else:
                agent_q = agent_q_values[th.LongTensor(range(bs)), agents_action[:, agent_idx].type(th.LongTensor)]  # for normal network to eval current Q_total
            if agent_idx == 0:
                agent_qs = agent_q
            else:
                agent_qs = th.cat((agent_qs.view(bs, -1), agent_q.view(-1, 1)), dim=1)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.num_agents)
        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.num_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, 1)  # output total Q value
        return q_tot

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# ++++++++++++++++++++++++++++replay buffer+++++++++++++++++++++++++++++ #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        # self.preload()

    def preload(self):
        for buffer_data_file in os.listdir(os.path.join("buffer_data")):
            buffer_data_array = np.load(os.path.join("buffer_data", buffer_data_file), allow_pickle=True)
            buffer_data_list = buffer_data_array.tolist()
            for one_step_tuple in buffer_data_list:
                self.push(one_step_tuple[0], one_step_tuple[1], one_step_tuple[2], one_step_tuple[3])

    def push(self, state, obs, action, reward, next_state, next_obs):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, obs, action, reward, next_state, next_obs)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return batch
        # state, action, reward, next_state = map(np.stack, zip(*batch))
        # return state, action, reward, next_state

    def __len__(self):
        return len(self.buffer)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# +++++++++++++++++++++++epsilon greedy policy++++++++++++++++++++++++++ #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
def select_action(q_input, epsilon, test_mode=False):

    # Assuming agent_inputs is a batch of Q-Values for each agent bav
    if test_mode:
        # Greedy action selection only
        adopt_eps = 0.0
    else:
        adopt_eps = epsilon

    random_number = np.random.rand(1)
    pick_random = (random_number > adopt_eps)
    random_action = random.randint(0, q_input.size-1)

    picked_action = pick_random * random_action + (1 - pick_random) * q_input.argmax()
    return picked_action


def update_qmix(train_batch, update_tar=False):
    state_batch, obs_batch, action_batch, reward_batch, next_state_batch, next_obs_batch = map(np.stack, zip(*train_batch))
    state_batch = th.FloatTensor(state_batch).to(device)
    obs_batch = th.FloatTensor(obs_batch).to(device)
    action_batch = th.FloatTensor(action_batch).to(device)
    reward_batch = th.FloatTensor(reward_batch).unsqueeze(1).to(device)
    next_state_batch = th.FloatTensor(next_state_batch).to(device)
    next_obs_batch = th.FloatTensor(next_obs_batch).to(device)


    q_total = qmixer.forward(obs_batch, action_batch, state_batch)
    target_q_total = target_qmixer.forward(next_obs_batch, action_batch, next_state_batch)
    y_value = reward_batch + gamma * target_q_total

    q_loss = Loss_Func(q_total, y_value.detach())
    q_optimizer.zero_grad()
    q_loss.backward()
    th.nn.utils.clip_grad_norm_(qmixer.parameters(), 0.05)
    # q_grad_norm = 0
    # for sub_para in q_optimizer.param_groups[0]["params"]:
    #     q_grad_norm += np.linalg.norm(sub_para.grad)
    q_optimizer.step()
    if update_tar is True:
        for target_param, param in zip(target_qmixer.parameters(), qmixer.parameters()):
            target_param.data.copy_(param.data)


def log_record(log_str, log_file):
    f_log = open(log_file, "a")
    f_log.write(log_str)
    f_log.close()

if __name__ == '__main__':
    SEED = 32013
    random.seed(SEED)
    np.random.seed(SEED)

    test_flag = False  # test mode flag, adopted epsilon = 1 is test_flag = True
    Agent_id_list = ['INT_01', 'INT_11', 'INT_21']  # id list of intersections
    # Agent_id_list = ['INT_01']  # id list of intersections
    n_agents = len(Agent_id_list)  # number of agent

    use_cuda = th.cuda.is_available()
    device = th.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        th.cuda.set_device(1)
    print('Use CUDA:'.format(use_cuda))
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"  # 使用 GPU 0

    total_epi = 1  # set episode number
    step_p_epi = 100  # set step number per episode
    replay_buffer_size = 5000  # replay buffer size
    train_start_buffer = 50  # start training threshold
    target_interval = 10
    start_epsilon = 0.05
    end_epsilon = 0.95  # epsilon parameter for selection policy
    gamma = 0.9
    learning_rate = 0.001
    batch_size = 32  # batch size for training
    obs_dim = 5  # observation dimension of each agent
    action_dim = 2  # action dimension of each agent
    mixing_embed_dim = 32
    state_dim = 5*n_agents  # shape of state

    record_path = "record/test/"
    folder_for_this_exp ="max_episodes{0}_max_steps{1}_gamma{2}_time{3}_seed{4}".format(str(total_epi), str(step_p_epi), str(gamma),time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time())),str(SEED))
    current_exp_path = os.path.join(record_path, folder_for_this_exp)
    model_path = os.path.join(current_exp_path, "model")
    traffic_data_path = os.path.join(current_exp_path, "traffic_data")
    episode_store_path = os.path.join(current_exp_path, "log_reward.txt")
    step_store_path = os.path.join(current_exp_path, "log_step.txt")

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(traffic_data_path):
        os.makedirs(traffic_data_path)
    shutil.copy(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Qmix.py'),
        os.path.join(current_exp_path, 'Qmix_back.py'))
    shutil.copy(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Traffic_Env.py'),
        os.path.join(current_exp_path, 'Traffic_Env_back.py'))

    print("Episode Num: %d\r\n Step_Num per epi: %d \r\n buffer size: %d\r\n epsilon: %f\r\n" %
          (total_epi, step_p_epi, replay_buffer_size, end_epsilon))

    # initialize the replay buffer
    print("initialize the replay buffer")
    replay_buffer = ReplayBuffer(replay_buffer_size)

    # initialize the network in cloud
    print("initialize the network in cloud")
    qmixer = QMixer(n_agents, state_dim, mixing_embed_dim, obs_dim, action_dim).to(device)

    # initialize the target network in cloud
    print("initialize the target network in cloud")
    target_qmixer = QMixer(n_agents, state_dim, mixing_embed_dim, obs_dim, action_dim, greedy=True).to(device)
    for target_param, param in zip(target_qmixer.parameters(), qmixer.parameters()):
        target_param.data.copy_(param.data)

    # define optimizer in cloud
    q_optimizer = Adam(qmixer.parameters(), lr=learning_rate)  # need to check the parameters, do they include all para.
    Loss_Func = nn.MSELoss()
    # initialize the DNN network in fog node
    print("initialize the DNN network in fog node")
    for i in range(n_agents):
        exec('Fog_Agents_' + str(Agent_id_list[i]) + '=copy.deepcopy(qmixer.Agents_' + str(Agent_id_list[i]) + ')')  # Agent Net in fog node, copy the network from cloud

    print("Start Experiment")
    rewards = []
    for episode in range(total_epi):
        episode_reward = 0
        pre_update_step = 0
        update_tar_flag = False
        # create traffic environment
        Env = Traffic_Env(obs_dim, action_dim, state_dim, Agent_id_list)
        # reset traffic environment
        Env.reset()
        # get state of whole environment
        env_state = Env.get_state()
        # get observation of whole environment
        env_obs = Env.get_obs()  # observation of whole environment
        for step in range(step_p_epi):
            epsilon = start_epsilon + ((episode*step_p_epi+step)/(total_epi*step_p_epi))*(end_epsilon-start_epsilon)
            # +++++++++++++++++++++++Done in fog node++++++++++++++++++++++ #
            action_list = []  # list of storing action for each agent
            for agent_idx in range(n_agents):
                # get observation of each agent
                agent_obs = Env.get_agent_obs(agent_idx)  # (format is not determined)
                # make decision for each agent
                exec('agent_q_value = Fog_Agents_' + str(Agent_id_list[agent_idx]) + '.forward(th.FloatTensor(agent_obs).unsqueeze(0).to(device)).detach().cpu().numpy()[0]')
                sel_act = select_action(agent_q_value, epsilon, test_mode=test_flag)
                action_list.append(sel_act.tolist())
            action_list = list(_flatten(action_list))
            # ++++++++++++execution in fog node, receive++++++++++++++++++++++ #
            # decentralized execution of traffic system, receive reward
            done = Env.step(action_list)  # step execution in traffic environment
            env_next_state = Env.get_state()  # get new state after taking new action
            env_next_obs = Env.get_obs()  # get new observation after taking new action
            env_reward = Env.get_reward()  # get reward
            # print("Episode: %d, Step: %d, epsilon: %.3f, Action: %d, Reward: %.3f, " %(episode, step, epsilon, action_list, env_reward))
            step_str = "Episode:{:}, Step:{:}, epsilon:{:.3}, Action:{:}, Reward:{:.4}".format(episode, step, epsilon, action_list, env_reward) + '\r\n'
            print(step_str)
            log_record(step_str, step_store_path)
            replay_buffer.push(env_state, env_obs, np.array(action_list), env_reward, env_next_state, env_next_obs)

            if (test_flag is False) and (len(replay_buffer) > train_start_buffer):
                transition_batch = replay_buffer.sample(batch_size)
                # centralized training in cloud node
                if step - pre_update_step >= target_interval:
                    update_tar_flag = True
                    pre_update_step = step
                update_qmix(transition_batch, update_tar=update_tar_flag)
                update_tar_flag = False
                for i in range(n_agents):
                    exec('for Fog_param, cloud_param in zip(Fog_Agents_' + str(Agent_id_list[i]) + '.parameters(), qmixer.Agents_' + str(Agent_id_list[i]) + '.parameters()): Fog_param.data.copy_(cloud_param)')


            env_state = env_next_state
            env_obs = env_next_obs
            episode_reward += env_reward
            rewards.append(env_reward)

        reward_str = "episode:{0}, episode_reward:{1}" \
                         .format(episode, episode_reward) + '\r\n'
        log_record(reward_str, episode_store_path)
        if episode % 10 == 0:
            th.save(qmixer.state_dict(),
                       os.path.join(model_path, 'qmixer_net' + str(episode) + '.pth'))
    scio.savemat(os.path.join(current_exp_path, 'epi_data_list.mat'),{'reward_list': rewards})
        # for i in qmixer.state_dict():
        #     print(i)
        # for i in qmixer.named_parameters():
        #     print(i)
