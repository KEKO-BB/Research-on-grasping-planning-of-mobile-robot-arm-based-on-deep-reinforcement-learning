import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ReplayBuffer:
    def __init__(self, capacity) -> None:
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class LSTMNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class Actor(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim, lstm_dim, lstm_layers, time_step_features, init_w=3e-3) -> None:
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(n_states, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim + lstm_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, n_actions)

        self.lstm_net = LSTMNetwork(time_step_features, lstm_dim, lstm_layers, lstm_dim)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, x, time_seq):
        x = F.relu(self.linear1(x))
        lstm_out = self.lstm_net(time_seq)
        x = torch.cat([x, lstm_out], 1)
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x


class Critic(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim, lstm_dim, lstm_layers, time_step_features, init_w=3e-3) -> None:
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(n_states + n_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim + lstm_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.lstm_net = LSTMNetwork(time_step_features, lstm_dim, lstm_layers, lstm_dim)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action, time_seq):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        lstm_out = self.lstm_net(time_seq)
        x = torch.cat([x, lstm_out], 1)
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class DDPG:
    def __init__(self, n_states, n_actions, cfg) -> None:
        self.device = torch.device(cfg.device)
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.critic = Critic(n_states, n_actions, cfg.hidden_dim, cfg.lstm_dim, cfg.lstm_layers,
                             cfg.time_step_features).to(self.device)
        self.actor = Actor(n_states, n_actions, cfg.hidden_dim, cfg.lstm_dim, cfg.lstm_layers,
                           cfg.time_step_features).to(self.device)
        self.target_critic = Critic(n_states, n_actions, cfg.hidden_dim, cfg.lstm_dim, cfg.lstm_layers,
                                    cfg.time_step_features).to(self.device)
        self.target_actor = Actor(n_states, n_actions, cfg.hidden_dim, cfg.lstm_dim, cfg.lstm_layers,
                                  cfg.time_step_features).to(self.device)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.memory = ReplayBuffer(cfg.memory_capacity)
        self.batch_size = cfg.batch_size
        self.soft_tau = cfg.soft_tau
        self.gamma = cfg.gamma
        self.actor_policy_loss = 0
        self.critic_value_loss = 0

    def choose_action(self, state):
        # 将整个状态向量分割为基础状态和时间步长状态
        basic_state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # 使用完整的状态
        time_step_state = torch.FloatTensor(state[-220:].reshape(1, -1, self.cfg.time_step_features)).to(self.device)
        action = self.actor(basic_state, time_step_state)
        return action.detach().cpu().numpy()[0]

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        state, action, reward, next_state, done = self.memory.sample(self.batch_size)

        # 使用完整的状态向量
        basic_state = torch.FloatTensor(np.array([s for s in state])).to(self.device)
        time_step_state = torch.FloatTensor(
            np.array([s[-220:].reshape(-1, self.cfg.time_step_features) for s in state])).to(self.device)

        next_basic_state = torch.FloatTensor(np.array([s for s in next_state])).to(self.device)
        next_time_step_state = torch.FloatTensor(
            np.array([s[-220:].reshape(-1, self.cfg.time_step_features) for s in next_state])).to(self.device)

        action = torch.FloatTensor(np.array(action)).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        policy_action = self.actor(basic_state, time_step_state)
        policy_loss = self.critic(basic_state, policy_action, time_step_state)
        policy_loss = -policy_loss.mean()
        self.actor_policy_loss = policy_loss.item()

        next_action = self.target_actor(next_basic_state, next_time_step_state)
        target_value = self.target_critic(next_basic_state, next_action.detach(), next_time_step_state)
        expected_value = reward + (1.0 - done) * self.gamma * target_value
        expected_value = torch.clamp(expected_value, -np.inf, np.inf)

        value = self.critic(basic_state, action, time_step_state)
        value_loss = nn.MSELoss()(value, expected_value.detach())
        self.critic_value_loss = value_loss.item()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)

    def save(self, path):
        torch.save(self.actor.state_dict(), path + "checkpoint.pt")

    def load(self, path):
        self.actor.load_state_dict(torch.load(path + "checkpoint.pt"))
