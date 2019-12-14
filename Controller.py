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
# # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# # ++++++++++++++++++++++++++RNN class definition++++++++++++++++++++++++ #
# # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# class RNNAgent(nn.Module):
#     def __init__(self, input_shape, rnn_hidden_dim, n_actions):
#         super(RNNAgent, self).__init__()
#         self.rnn_hidden_dim = rnn_hidden_dim
#         self.fc1 = nn.Linear(input_shape, rnn_hidden_dim)
#         self.rnn = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)
#         self.fc2 = nn.Linear(rnn_hidden_dim, n_actions)
#
#     def init_hidden(self):
#         # make hidden states on same device as model
#         return self.fc1.weight.new(1, self.rnn_hidden_dim).zero_()
#
#     # +++++++++++++++++++++++RNN forward function++++++++++++++++++++++ #
#     def forward(self, inputs, hidden_state):
#         x = F.relu(self.fc1(inputs))
#         h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
#         h = self.rnn(x, h_in)  # output hidden state for next iteration
#         q = self.fc2(h)  # output unity values of all actions for an agent
#         return q, h


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# ++++++++++++++++++++++++++DNN class definition++++++++++++++++++++++++ #
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
        if use_cuda:
            for layer in layers:
                layer = layer

    # +++++++++++++++++++++++RNN forward function++++++++++++++++++++++ #
    def forward(self, inputs):
        x1 = F.relu((self.fc1(inputs)))
        x2 = F.relu((self.fc2(x1)))
        q = self.fc3(x2)  # output unity values of all actions for an agent
        return q


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# +++++++++++++++++++++++Mixing network definition++++++++++++++++++++++ #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class QMixer(nn.Module):
    def __init__(self, n_agents, state_shape, mixing_embed_dim, obs_dim, action_dim):
        super(QMixer, self).__init__()
        self.Agents = [DNNAgent(obs_dim, action_dim) for i in range(n_agents)]
        self.n_agents = n_agents
        self.state_dim = int(np.prod(state_shape))

        self.embed_dim = mixing_embed_dim

        self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
        self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        self.init_weight()
        # hypernet_embed = self.hypernet_embed
        # self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
        #                                nn.ReLU(),
        #                                nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
        # self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
        #                                nn.ReLU(),
        #                                nn.Linear(hypernet_embed, self.embed_dim))

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def init_weight(self):
        layers = [self.hyper_w_1, self.hyper_w_final]
        [th.nn.init.xavier_normal_(layer.weight) for layer in layers]
        if use_cuda:
            for layer in layers:
                layer = layer

    # +++++++++++++++++++++++Mixing forward function++++++++++++++++++++++ #
    def forward(self, agents_obs, states):
        agent_qs = th.zeros(n_agents)
        actions = th.zeros(n_agents)
        for agent_idx in range(self.n_agents):
            # make decision for each agent
            agent_q_values = self.Agents[agent_idx].forward(agents_obs[agent_idx])  # (format is not determined)
            agent_qs[agent_idx], actions[agent_idx] = th.max(agent_q_values)
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
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
        q_tot = y.view(bs, -1, 1)  # output total Q value
        return q_tot, actions




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

    random_number = th.rand(1)
    pick_random = (random_number > adopt_eps).long()
    random_action = th.floor(th.rand(1) * q_input.size())

    picked_action = pick_random * random_action + (1 - pick_random) * q_input.argmax()
    return picked_action

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


def update_qmix(train_batch):
    state_batch, obs_batch, action_batch, reward_batch, next_state_batch, next_obs_batch = map(np.stack, zip(*train_batch))
    state_batch = th.FloatTensor(state_batch).to(device)
    obs_batch = th.FloatTensor(obs_batch).to(device)
    action_batch = th.FloatTensor(action_batch).to(device)
    reward_batch = th.FloatTensor(reward_batch).unsqueeze(1).to(device)
    next_state_batch = th.FloatTensor(next_state_batch).to(device)
    next_obs_batch = th.FloatTensor(next_obs_batch).to(device)

    agent_idx = 0
    q_value_array = np.zeros(batch_size, n_agents)
    next_q_value_array = np.zeros(batch_size, n_agents)
    for agt_id in Agent_id_list:
        q_value, _ = DNNAgent_dict[agt_id].forward(obs_batch[:, :, agt_id])
        q_value_array[:, agt_id] = q_value[:, action_batch]

        next_q_value, _ = target_DNNAgent_dict[agent_id].forward(next_obs_batch[:, :, agt_id])
        next_q_value_array[agt_id] = next_q_value.max(dim=2)

        agent_idx += 1
    q_value_array = th.FloatTensor(q_value_array).to(device)
    next_q_value_array = th.FloatTensor(next_q_value_array).to(device)

    q_total = qmixer.forward(q_value_array, state_batch)
    target_q_total = target_qmixer.forward(next_q_value_array, next_state_batch)
    y_value = reward_batch + gamma * target_q_total

    q_loss = nn.MSELoss(q_total, y_value.detach())
    q_optimizer.zero_grad()
    q_loss.backward()
    th.nn.utils.clip_grad_norm_(value_net.parameters(), 0.05)
    # q_grad_norm = 0
    # for sub_para in q_optimizer.param_groups[0]["params"]:
    #     q_grad_norm += np.linalg.norm(sub_para.grad)
    q_optimizer.step()
    pass

if __name__ == '__main__':
    use_cuda = th.cuda.is_available()
    device = th.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        th.cuda.set_device(1)
    print('Use CUDA:'.format(use_cuda))
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"  # 使用 GPU 0

    total_epi = 1000  # set episode number
    step_p_epi = 1000  # set step number per episode
    replay_buffer_size = 5000  # replay buffer size
    train_start_buffer = 500  # start training threshold
    epsilon = 0.95  # epsilon parameter for selection policy
    gamma = 0.9
    learning_rate = 0.001
    batch_size = 32  # batch size for training
    obs_dim = 1  # observation dimension of each agent
    hidden_dim = 1  # RNN hidden layer dimension
    action_dim = 1  # action dimension of each agent
    mixing_embed_dim = 1
    state_shape = (4, 1)  # shape of state

    test_flag = False  # test mode flag, adopted epsilon = 1 is test_flag = True
    Agent_id_list = ['INT_01', 'INT_11', 'INT_21']  # id list of intersections
    n_agents = len(Agent_id_list)  # number of agent

    print("Episode Num: %d\r\n Step_Num per epi: %d \r\n buffer size: %d\r\n epsilon: %f\r\n" %
          (total_epi, step_p_epi, replay_buffer_size, epsilon))

    # initialize the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size)

    # initialize the Mixing network
    qmixer = QMixer(n_agents, state_shape, mixing_embed_dim, obs_dim, action_dim).to(device)
    q_optimizer = Adam(qmixer.parameters(), lr=learning_rate)
    target_qmixer = copy.deepcopy(qmixer)

    action_dict = {}  # dictionary of storing action for each agent
    print("Start Experiment")
    for episode in range(total_epi):
        episode_reward = 0
        Env = Traffic_Env()  #lack of input
        Env.reset()
        env_state = Env.get_state()  # state of whole environment
        env_obs = Env.get_obs()  # observation of whole environment
        for step in range(step_p_epi):
            # +++++++++++++++++++++++Done in fog node++++++++++++++++++++++ #
            for agent_id in Agent_id_list:
                # get observation of each agent
                agent_obs = Env.get_agent_obs(agent_id)  # (format is not determined)
                # make decision for each agent
                agent_q_value = DNNAgent_dict[agent_id].forward(agent_obs).detach().cpu().numpy()  # (format is not determined)
                action_dict[agent_id] = select_action(agent_q_value, epsilon, test_mode=test_flag)
            # ++++++++++++execution in fog node, receive++++++++++++++++++++++ #
            # decentralized execution of traffic system, receive reward
            done = Env.step(action_dict)  # step execution in traffic environment
            env_next_state = Env.get_state()  # get new state after taking new action
            env_next_obs = Env.get_obs()  # get new observation after taking new action
            env_reward = Env.get_reward(env_state)  # get reward

            print("Episode", episode, "Step", step, "Action", np.array(action_dict.values()), "Reward", env_reward)
            replay_buffer.push(env_state, env_obs, np.array(action_dict.values()), env_reward, env_next_state, env_next_obs)

            if (test_flag is False) and (len(replay_buffer) > train_start_buffer):
                transition_batch = replay_buffer.sample(batch_size)
                # centralized training in cloud node
                update_qmix(transition_batch)
            env_state = env_next_state
            env_obs = env_next_obs
            episode_reward += env_reward
