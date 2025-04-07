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


class GRUNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRUNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
        out = out[:, -1, :]  # Get the last output for sequence input
        out = self.fc(out)
        return out



class Actor(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim, gru_dim, gru_layers, time_step_features, init_w=3e-3) -> None:
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(n_states, hidden_dim)  # 确保n_states是121
        self.linear2 = nn.Linear(hidden_dim + gru_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, n_actions)

        self.gru_net = GRUNetwork(time_step_features, gru_dim, gru_layers, gru_dim)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, x, time_seq):
        x = F.relu(self.linear1(x))  # 输入的x现在是(1, 121)
        gru_out = self.gru_net(time_seq)
        x = torch.cat([x, gru_out], 1)
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x



class Critic(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim, gru_dim, gru_layers, time_step_features, init_w=3e-3) -> None:
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(n_states + n_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim + gru_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.gru_net = GRUNetwork(time_step_features, gru_dim, gru_layers, gru_dim)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action, time_seq):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        gru_out = self.gru_net(time_seq)
        x = torch.cat([x, gru_out], 1)
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class TD3:
    def __init__(self, n_states, n_actions, cfg) -> None:
        self.device = torch.device(cfg.device)
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.critic1 = Critic(n_states, n_actions, cfg.hidden_dim, cfg.gru_dim, cfg.gru_layers,
                             cfg.time_step_features).to(self.device)
        self.critic2 = Critic(n_states, n_actions, cfg.hidden_dim, cfg.gru_dim, cfg.gru_layers,
                             cfg.time_step_features).to(self.device)
        self.actor = Actor(n_states, n_actions, cfg.hidden_dim, cfg.gru_dim, cfg.gru_layers,
                           cfg.time_step_features).to(self.device)
        self.target_critic1 = Critic(n_states, n_actions, cfg.hidden_dim, cfg.gru_dim, cfg.gru_layers,
                                     cfg.time_step_features).to(self.device)
        self.target_critic2 = Critic(n_states, n_actions, cfg.hidden_dim, cfg.gru_dim, cfg.gru_layers,
                                     cfg.time_step_features).to(self.device)
        self.target_actor = Actor(n_states, n_actions, cfg.hidden_dim, cfg.gru_dim, cfg.gru_layers,
                                  cfg.time_step_features).to(self.device)

        # Initialize target networks with same weights as the original networks
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=cfg.critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=cfg.critic_lr)
        self.memory = ReplayBuffer(cfg.memory_capacity)
        self.batch_size = cfg.batch_size
        self.soft_tau = cfg.soft_tau
        self.gamma = cfg.gamma
        self.policy_noise = cfg.policy_noise
        self.noise_clip = cfg.noise_clip
        self.policy_update_freq = cfg.policy_update_freq

    def choose_action(self, state):
        # 提取完整的状态空间 (121维)
        basic_state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # 维度为(1, 121)

        # 提取过去5个时间步的状态
        time_step_state = torch.FloatTensor(state[46:]).reshape(1, 5, 15).to(self.device)

        # 使用 actor 网络来选择动作
        action = self.actor(basic_state, time_step_state)
        return action.detach().cpu().numpy()[0]

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        state, action, reward, next_state, done = self.memory.sample(self.batch_size)

        # 更新 basic_state 和 time_step_state 的提取方式
        basic_state = torch.FloatTensor(np.array([s[:46] for s in state])).to(self.device)
        time_step_state = torch.FloatTensor(np.array([s[46:].reshape(-1, 5, 15) for s in state])).to(self.device)
        next_basic_state = torch.FloatTensor(np.array([s[:46] for s in next_state])).to(self.device)
        next_time_step_state = torch.FloatTensor(np.array([s[46:].reshape(-1, 5, 15) for s in next_state])).to(
            self.device)

        action = torch.FloatTensor(np.array(action)).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        # Critic loss
        noise = torch.randn_like(action) * self.policy_noise
        noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
        next_action = self.target_actor(next_basic_state, next_time_step_state) + noise
        next_action = torch.clamp(next_action, -1, 1)

        target_value1 = self.target_critic1(next_basic_state, next_action.detach(), next_time_step_state)
        target_value2 = self.target_critic2(next_basic_state, next_action.detach(), next_time_step_state)
        target_value = torch.min(target_value1, target_value2)
        expected_value = reward + (1.0 - done) * self.gamma * target_value
        expected_value = torch.clamp(expected_value, -np.inf, np.inf)

        value1 = self.critic1(basic_state, action, time_step_state)
        value2 = self.critic2(basic_state, action, time_step_state)
        critic1_loss = F.mse_loss(value1, expected_value.detach())
        critic2_loss = F.mse_loss(value2, expected_value.detach())

        # Actor loss
        policy_loss = -self.critic1(basic_state, self.actor(basic_state, time_step_state), time_step_state).mean()

        # Update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Soft update target networks
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)

        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)

        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)

    def save(self, path):
        torch.save(self.actor.state_dict(), path + "checkpoint_actor.pt")
        torch.save(self.critic1.state_dict(), path + "checkpoint_critic1.pt")
        torch.save(self.critic2.state_dict(), path + "checkpoint_critic2.pt")

    def load(self, path):
        self.actor.load_state_dict(torch.load(path + "checkpoint_actor.pt"))
        self.critic1.load_state_dict(torch.load(path + "checkpoint_critic1.pt"))
        self.critic2.load_state_dict(torch.load(path + "checkpoint_critic2.pt"))
