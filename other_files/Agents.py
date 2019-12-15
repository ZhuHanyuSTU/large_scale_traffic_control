import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self, N_STATE, N_ACTIONS):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATE, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, N_ACTIONS)
        self.init_weight()

    def init_weight(self, using_cuda=False):
        layers = [self.fc1, self.fc2, self.fc3]
        [torch.nn.init.xavier_normal_(layer.weight) for layer in layers]
        if using_cuda:
            for layer in layers:
                layer = layer

    def forward(self, local_observation):
        x1 = torch.sigmoid(self.fc1(local_observation))
        x2 = torch.sigmoid(self.fc2(x1))
        x3 = F.relu(self.fc3(x2))
        return x3


class Qmix(nn.Module):
    def __init__(self, GLOBAL_STATE_SIZE, HIDDEN_LAYER_SIZE, N_AGENTS, N_STATE, N_ACTIONS):
        super(Qmix, self).__init__()
        self.Agents = [Net(N_STATE, N_ACTIONS) for i in range(N_AGENTS)]
        self.hyper_1 = nn.Linear(GLOBAL_STATE_SIZE, HIDDEN_LAYER_SIZE * N_AGENTS)
        self.hyper_2 = nn.Linear(GLOBAL_STATE_SIZE, HIDDEN_LAYER_SIZE)
        self.init_weight()
        self.N_STATE = N_STATE
        self.N_AGENTS = N_AGENTS
        self.N_ACTIONS = N_ACTIONS
        self.GLOBAL_STATE_SIZE = GLOBAL_STATE_SIZE
        self.HIDDEN_LAYER_SIZE = HIDDEN_LAYER_SIZE

    def init_weight(self, using_cuda=False):
        layers = [self.hyper_1, self.hyper_2]
        [torch.nn.init.xavier_normal_(layer.weight) for layer in layers]
        if using_cuda:
            for layer in layers:
                layer = layer

    def forward(self, N_OBSERVATIONS, GLOBAL_STATE):

        # PARAM N_OBSERVATIONS -> N_AGENTS*N_STATE
        # N_Q_VALUES = torch.sigmoid(self.fc1(N_OBSERVATIONS))
        # #print(N_Q_VALUES.shape)
        # N_Q_VALUES = torch.sigmoid(self.fc2(N_Q_VALUES))
        # #print(N_Q_VALUES.shape)
        # N_Q_VALUES = F.relu(self.fc3(N_Q_VALUES))
        N_OBSERVATIONS = N_OBSERVATIONS.view(np.int(N_OBSERVATIONS.numel() / (self.N_AGENTS * self.N_STATE)),
                                             self.N_AGENTS, self.N_STATE)
        N_Q_VALUES = torch.zeros(
            (np.int(N_OBSERVATIONS.numel() / (self.N_AGENTS * self.N_STATE)), self.N_AGENTS, self.N_ACTIONS))
        for i in range(self.N_AGENTS):
            N_Q_VALUES[:, i, :] = self.Agents[i](N_OBSERVATIONS[:, i, :])
        # N_Q_VALUES = [self.Agents[i](N_OBSERVATIONS[:,i,:]) for i in range(self.N_AGENTS)]
        # N_Q_VALUES = torch.tensor([[N_Q_VALUES[i][:][0],N_Q_VALUES[i][:][1]] for i in range(self.N_AGENTS)])
        Middle_Result = N_Q_VALUES
        # print(N_Q_VALUES.shape)
        actions = N_Q_VALUES.max(dim=-1)[1]
        # print(actions)
        # print(N_Q_VALUES.max(dim=-1)[0].shape)
        N_Q_VALUES = N_Q_VALUES.max(dim=-1)[0].view(np.int(N_Q_VALUES.numel() / (self.N_ACTIONS * self.N_AGENTS)), 1,
                                                    self.N_AGENTS)
        w1 = self.hyper_1(GLOBAL_STATE).abs().view(np.int(GLOBAL_STATE.numel() / (self.N_AGENTS)), self.N_AGENTS,
                                                   self.HIDDEN_LAYER_SIZE)
        w2 = self.hyper_2(GLOBAL_STATE).abs().view(np.int(GLOBAL_STATE.numel() / (self.N_AGENTS)),
                                                   self.HIDDEN_LAYER_SIZE, 1)
        q_tot = F.elu(torch.matmul(N_Q_VALUES, w1))
        q_tot = F.relu(torch.matmul(q_tot, w2))
        return q_tot, actions, Middle_Result


class DQN():
    def __init__(self,
                 memory_size,
                 GLOBAL_STATE_SIZE,
                 HIDDEN_LAYER_SIZE,
                 N_AGENTS,
                 N_STATE,
                 N_ACTIONS,
                 epsilon=0.9,
                 target_update_gap=50,
                 batch_size=5,
                 gamma=0.8):
        self.estimator = Qmix(GLOBAL_STATE_SIZE, HIDDEN_LAYER_SIZE, N_AGENTS, N_STATE, N_ACTIONS)
        self.target = Qmix(GLOBAL_STATE_SIZE, HIDDEN_LAYER_SIZE, N_AGENTS, N_STATE, N_ACTIONS)
        self.memory = np.zeros((
            memory_size, (1 + 2 * (GLOBAL_STATE_SIZE + (N_STATE * N_AGENTS)) + N_AGENTS)
        ))  # 新建一个空的memory pool
        self.memory_counter = 0  # 记录有多少个tuple被存储
        self.optimizer = torch.optim.Adam(self.estimator.parameters())  # Adam优化器
        self.LossF = nn.MSELoss()
        self.epsilon = epsilon
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.target_update_gap = target_update_gap
        self.N_ACTIONS = N_ACTIONS
        self.N_STATE = N_STATE
        self.gamma = gamma
        self.loss_records = []
        self.waitingtime = []
        self.action = dict()
        self.HIDDEN_LAYER_SIZE = HIDDEN_LAYER_SIZE
        self.N_AGENTS = N_AGENTS
        self.GLOBAL_STATE_SIZE = GLOBAL_STATE_SIZE

    def choose_action(self, N_OBSERVATIONS, GLOBAL_STATE):
        # 输出action list, np.array
        if np.random.random() <= self.epsilon:
            action = self.estimator.forward(N_OBSERVATIONS, GLOBAL_STATE)[1].cpu().numpy()  # 根据网络选
        else:
            action = np.random.choice(self.N_ACTIONS, self.N_AGENTS, replace=True)  # 随机选
        return action

    def store_memory(self, s_l, s_g, a, r, s_l_, s_g_):
        # 将 state_t,a_t,state_{t+1},reward_t存进一个tuple,位置的放置可以选择
        experience = np.hstack((s_l, s_g, a, r, s_l_, s_g_))
        index = self.memory_counter % self.memory_size
        index = np.random.randint(0, max(self.memory_size // 2, self.memory_counter % self.memory_size))
        self.memory[index, :] = experience
        # self.memory = self.memory[self.memory[:,self.N_STATE+1].argsort()]
        self.memory_counter += 1

    def update_network(self, drop_off=False, soft_update=False):
        # parameters transimitation
        if soft_update:
            return True
        if not drop_off:
            if np.mod(self.memory_counter, self.target_update_gap) == 0:
                self.target.load_state_dict(self.estimator.state_dict())

        # draw the samples from the memory pool
        prob = list(np.ones([1, self.memory_size])[0] / self.memory_size)  # uniformly draw samples
        prob = [(i + 1) * 2 / ((self.memory_size + 1) * self.memory_size) for i in
                range(self.memory_size)]  # higher reward, higher prob
        sample_index = np.random.choice(self.memory_size, self.batch_size, p=prob, replace=False)
        b_memory = self.memory[sample_index, :]
        b_s_l = torch.FloatTensor(b_memory[:, :self.N_STATE * self.N_AGENTS]).view(self.batch_size, self.N_AGENTS,
                                                                                   self.N_STATE)
        b_s_g = torch.FloatTensor(
            b_memory[:, self.N_STATE * self.N_AGENTS:self.N_STATE * self.N_AGENTS + self.GLOBAL_STATE_SIZE]).view(
            self.batch_size, self.GLOBAL_STATE_SIZE)
        b_a = torch.LongTensor(b_memory[:,
                               self.N_STATE * self.N_AGENTS + self.GLOBAL_STATE_SIZE:self.N_STATE * self.N_AGENTS + self.GLOBAL_STATE_SIZE + self.N_AGENTS]).view(
            self.batch_size, self.N_AGENTS, 1)
        b_r = torch.FloatTensor(b_memory[:,
                                self.N_STATE * self.N_AGENTS + self.GLOBAL_STATE_SIZE + self.N_AGENTS:self.N_STATE * self.N_AGENTS + self.GLOBAL_STATE_SIZE + self.N_AGENTS + 1])

        b_s_l_ = torch.FloatTensor(b_memory[:,
                                   -(
                                               self.N_STATE * self.N_AGENTS + self.GLOBAL_STATE_SIZE):-self.GLOBAL_STATE_SIZE]).view(
            self.batch_size, self.N_AGENTS, self.N_STATE)
        b_s_g_ = torch.FloatTensor(b_memory[:, -self.GLOBAL_STATE_SIZE:]).view(self.batch_size, self.GLOBAL_STATE_SIZE)

        # optimize the estimator network
        q_eval = self.estimator(b_s_l, b_s_g)[2].gather(2, b_a).view(self.batch_size, 1,
                                                                     self.N_AGENTS)  # shape (batch, 1)
        w1_eval = self.estimator.hyper_1(b_s_g).abs().view(self.batch_size, self.N_AGENTS, self.HIDDEN_LAYER_SIZE)
        w2_eval = self.estimator.hyper_2(b_s_g).abs().view(self.batch_size, self.HIDDEN_LAYER_SIZE, 1)
        q_eval = F.elu(torch.matmul(q_eval, w1_eval))
        q_eval = F.elu(torch.matmul(q_eval, w2_eval)).view(self.batch_size, 1)
        q_next = self.target(b_s_l_, b_s_g_)[0].detach()  # detach from graph, don't backpropagate
        q_target = b_r + self.gamma * q_next.view(self.batch_size, 1)  # shape (batch, 1)
        loss = self.LossF(q_eval, q_target)
        self.loss_records.append(loss.cpu().detach().numpy())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


'''
TEST PART
----------------------------------------------------------------
'''
GLOBAL_STATE_SIZE = 10
HIDDEN_LAYER_SIZE = 32
N_AGENTS = 10
N_STATE = 2
MEMORY_SIZE = 10
N_ACTIONS = 2
QNET = DQN(MEMORY_SIZE, GLOBAL_STATE_SIZE, HIDDEN_LAYER_SIZE, N_AGENTS, N_STATE, N_ACTIONS)


for i in range(100):
    Observation = torch.rand(N_AGENTS, N_STATE) * 5
    global_state = torch.rand(GLOBAL_STATE_SIZE)
    action = QNET.choose_action(Observation, global_state)

    Observation_n = torch.rand(N_AGENTS, N_STATE) * 5
    global_state_n = torch.rand(GLOBAL_STATE_SIZE)
    reward = np.random.rand() * 10
    QNET.store_memory(Observation.view(1, N_AGENTS * N_STATE), global_state.view(1, GLOBAL_STATE_SIZE),
                      action.reshape((1, N_AGENTS)), np.array(reward).reshape((1, 1)),
                      Observation_n.view(1, N_AGENTS * N_STATE), global_state_n.view(1, GLOBAL_STATE_SIZE))
    if QNET.memory_counter >= MEMORY_SIZE:
        QNET.update_network()


# for i in QNET.estimator.state_dict():
#     print(i)
for i in QNET.estimator.named_parameters():
    print(i)
# plt.plot(QNET.loss_records)
# plt.show()
Agents = [Net(N_STATE, N_ACTIONS) for i in range(N_AGENTS)]
