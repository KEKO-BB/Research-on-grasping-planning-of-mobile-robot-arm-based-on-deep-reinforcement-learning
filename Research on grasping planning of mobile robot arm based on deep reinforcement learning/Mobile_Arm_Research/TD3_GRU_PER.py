import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ReplayBuffer:  # 经验池
    def __init__(self, capacity) -> None:
        self.capacity = capacity  # 经验池的容量
        self.buffer = []  # 缓冲区
        self.position = 0

    def push(self, state, action, reward, next_state, done):  # 将经验压入缓冲区
        # 缓冲区是一个队列， 容量超出时去掉开始存入的转移
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)  # 随机从经验池中采出小批量经验
        state, action, reward, next_state, done = zip(*batch)  # 解压成状态，动作等
        return state, action, reward, next_state, done

    def __len__(self):
        # 返回当前存储的量
        return len(self.buffer)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, alpha=0.6):
        super(PrioritizedReplayBuffer, self).__init__(capacity)
        self.alpha = alpha
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        super().push(state, action, reward, next_state, done)
        max_prio = np.max(self.priorities) if len(self.buffer) > 1 else 1.0
        self.priorities[self.position - 1] = max_prio

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.position]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        state, action, reward, next_state, done = zip(*samples)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done), indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio


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
        self.linear1 = nn.Linear(n_states, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim + gru_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, n_actions)

        self.gru_net = GRUNetwork(time_step_features, gru_dim, gru_layers, gru_dim)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, x, time_seq):
        x = F.relu(self.linear1(x))
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
        self.memory = PrioritizedReplayBuffer(cfg.memory_capacity)
        self.batch_size = cfg.batch_size
        self.soft_tau = cfg.soft_tau
        self.gamma = cfg.gamma
        self.policy_noise = cfg.policy_noise
        self.noise_clip = cfg.noise_clip
        self.policy_update_freq = cfg.policy_update_freq

    def choose_action(self, state):
        basic_state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # 维度为(1, 121)
        time_step_state = torch.FloatTensor(state[46:]).reshape(1, 5, 15).to(self.device)
        action = self.actor(basic_state, time_step_state)
        return action.detach().cpu().numpy()[0]

    def update(self, beta=0.4):  # 在这里添加beta参数
        if len(self.memory) < self.batch_size:
            return

        state, action, reward, next_state, done, indices, weights = self.memory.sample(self.batch_size, beta)

        basic_state = torch.FloatTensor(np.array([s[:46] for s in state])).to(self.device)
        time_step_state = torch.FloatTensor(np.array([s[46:].reshape(-1, 5, 15) for s in state])).to(self.device)
        next_basic_state = torch.FloatTensor(np.array([s[:46] for s in next_state])).to(self.device)
        next_time_step_state = torch.FloatTensor(np.array([s[46:].reshape(-1, 5, 15) for s in next_state])).to(self.device)

        action = torch.FloatTensor(np.array(action)).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

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

        policy_loss = -self.critic1(basic_state, self.actor(basic_state, time_step_state), time_step_state).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)

        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)

        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)

        new_priorities = torch.abs(value1 - expected_value).detach().cpu().numpy() + 1e-5
        self.memory.update_priorities(indices, new_priorities)

    def save(self, path):
        torch.save(self.actor.state_dict(), path + "checkpoint_actor.pt")
        torch.save(self.critic1.state_dict(), path + "checkpoint_critic1.pt")
        torch.save(self.critic2.state_dict(), path + "checkpoint_critic2.pt")

    def load(self, path):
        self.actor.load_state_dict(torch.load(path + "checkpoint_actor.pt"))
        self.critic1.load_state_dict(torch.load(path + "checkpoint_critic1.pt"))
        self.critic2.load_state_dict(torch.load(path + "checkpoint_critic2.pt"))

