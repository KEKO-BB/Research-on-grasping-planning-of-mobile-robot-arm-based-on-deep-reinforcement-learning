#-*- utf-8 -*-
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



class Actor(nn.Module):  # actor 网络
    def __init__(self, n_states, n_actions, hidden_dim, init_w=3e-3) -> None:
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(n_states, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, n_actions)  # y=wx+bias

        self.linear3.weight.data.uniform_(-init_w, init_w)  # w权重服从均匀分布
        self.linear3.bias.data.uniform_(-init_w, init_w)  # bias偏置服从均匀分布

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x


class Critic(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim, init_w=3e-3, seed=None) -> None:
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(n_states + n_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)  # y=wx+bias
        if seed is not None:
            torch.manual_seed(seed)
        # 随机初始化为较小的值
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class TD3:
    def __init__(self, n_states, n_actions, cfg) -> None:
        # 初始化过程和DDPG类似，但有两个Critic网络
        self.device = torch.device(cfg.device)
        self.cfg = cfg
        self.actor = Actor(n_states, n_actions, cfg.hidden_dim).to(self.device)
        self.target_actor = Actor(n_states, n_actions, cfg.hidden_dim).to(self.device)

        # TD3 使用两个Critic网络
        self.critic1 = Critic(n_states, n_actions, cfg.hidden_dim, init_w=3e-3, seed=1).to(self.device)
        self.target_critic1 = Critic(n_states, n_actions, cfg.hidden_dim).to(self.device)
        self.critic2 = Critic(n_states, n_actions, cfg.hidden_dim, init_w=3e-3, seed=2).to(self.device)
        self.target_critic2 = Critic(n_states, n_actions, cfg.hidden_dim).to(self.device)

        # 将预测网络的参数复制给目标网络
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(param.data)

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=cfg.critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=cfg.critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)

        self.memory = PrioritizedReplayBuffer(cfg.memory_capacity)
        self.batch_size = cfg.batch_size
        self.soft_tau = cfg.soft_tau
        self.gamma = cfg.gamma
        self.policy_update = 0  # 用于跟踪策略更新的计数器
        self.policy_update_freq = cfg.policy_update_freq  # 策略更新频率

        # 探索噪声的标准差
        self.exploration_noise_std = 0.5  # 这个值可以根据您的环境进行调整
        # 损失记录
        self.actor_policy_loss = 0
        self.critic1_value_loss = 0
        self.critic2_value_loss = 0
        self.td_error_mean = 0

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state)
        return action.detach().cpu().numpy()[0]

    # def choose_action(self, state):
    #     state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
    #     action = self.actor(state).detach().cpu().numpy()[0]
    #
    #     # 在动作上添加探索噪声
    #     noise = np.random.normal(0, self.exploration_noise_std, size=action.shape)
    #     action = action + noise
    #
    #     return action

    def update(self, beta=0.4):
        if len(self.memory) < self.batch_size:
            return
        state, action, reward, next_state, done, indices, weights = self.memory.sample(self.batch_size, beta)

        # 将经验转成张量
        state = torch.FloatTensor(np.array(state)).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        action = torch.FloatTensor(np.array(action)).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(
            self.device)  # torch.unsqueeze(x, 1/0) x对象，对其维度进行扩充，后面的参数为0表示横向扩， 1为纵向扩，再大则错误
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)
        # 确保weights是一个列向量，以便它可以与损失逐元素相乘
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        # 更新Critic网络
        with torch.no_grad():
            # 目标策略平滑
            noise = (torch.randn_like(action) * self.cfg.policy_noise).clamp(self.cfg.noise_clip, self.cfg.noise_clip)
            next_action = (self.target_actor(next_state) + noise).clamp(-1, 1)

            # 使用两个目标Critic网络的最小值
            target_q1 = self.target_critic1(next_state, next_action)
            target_q2 = self.target_critic2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            expected_value = reward + (1.0 - done) * self.gamma * target_q

        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)
        # 加权损失计算，考虑重要性采样权重
        critic1_loss = (F.mse_loss(current_q1, expected_value) * weights).mean()
        critic2_loss = (F.mse_loss(current_q2, expected_value) * weights).mean()

        # 使用两个Critic的最小值计算TD误差
        td_errors = torch.min(torch.abs(current_q1 - expected_value), torch.abs(current_q2 - expected_value))
        self.td_error_mean = td_errors.mean().item()  # 计算TD误差的平均值
        # 更新优先级
        new_priorities = td_errors.detach().cpu().numpy() + 1e-5
        self.memory.update_priorities(indices, new_priorities)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        self.critic1_value_loss = critic1_loss.item()  # 更新Critic1损失

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        self.critic2_value_loss = critic2_loss.item()  # 更新Critic2损失

        # 延迟策略更新
        if self.policy_update % self.policy_update_freq == 0:
            # 更新策略网络（Actor）
            policy_loss = -self.critic1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()
            self.actor_policy_loss = policy_loss.item()  # 更新策略损失

            # 软更新目标网络
            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)
            for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)
            for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)

        self.policy_update += 1  # 更新策略更新计数器

    def save(self, path):  # 保存actor的网络模型
        torch.save(self.actor.state_dict(), path + "checkpoint.pt")
        # torch.save(self.critic.state_dict(),path+"critic_checkpoint.pt")

    def load(self, path):  # 加载 actor的网络模型
        self.actor.load_state_dict(torch.load(path + "checkpoint.pt"))


