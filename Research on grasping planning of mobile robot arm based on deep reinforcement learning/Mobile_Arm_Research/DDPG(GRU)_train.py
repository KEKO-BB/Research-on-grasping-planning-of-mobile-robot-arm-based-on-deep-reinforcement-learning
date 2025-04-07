import os
import sys
import datetime
import argparse
from DDPG_GRU import DDPG
from Dynamic_env_6_double_obstacle_预测 import Mobile_Arm_Env
from torch.utils.tensorboard import SummaryWriter

# 加入 TensorBoard的显示
writer = SummaryWriter("log")

def get_args():
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(description="hyperparameters")
    parser.add_argument('--algo_name', default='DDPG', type=str, help="name of algorithm")
    parser.add_argument('--env_name', default='Dynamic_Arm', type=str, help="name of environment")
    parser.add_argument('--train_eps', default=5000, type=int, help="episodes of training")
    parser.add_argument('--test_eps', default=500, type=int, help="episodes of testing")
    parser.add_argument('--gamma', default=0.99, type=float, help="discounted factor")
    parser.add_argument('--critic_lr', default=1e-4, type=float, help="learning rate of critic")
    parser.add_argument('--actor_lr', default=1e-5, type=float, help="learning rate of actor")
    parser.add_argument('--memory_capacity', default=12000, type=int, help="memory capacity")
    parser.add_argument('--batch_size', default=138, type=int)
    parser.add_argument('--target_update', default=2, type=int)
    parser.add_argument('--soft_tau', default=1e-2, type=float)
    parser.add_argument('--hidden_dim', default=400, type=int)
    parser.add_argument('--device', default='cuda', type=str, help='cpu or cuda')
    parser.add_argument('--result_path', default=f"{curr_time}/results/")
    parser.add_argument('--model_path', default=f"{curr_time}/models/")
    parser.add_argument('--save_fig', default=True, type=bool)
    parser.add_argument('--gru_dim', default=128, type=int, help="dimension of GRU hidden state")
    parser.add_argument('--gru_layers', default=1, type=int, help="number of layers in GRU")
    parser.add_argument('--time_step_features', default=44, type=int, help="number of features per time step in GRU")
    args = parser.parse_args()
    return args

def env_agent_config(cfg, seed=1):
    env = Mobile_Arm_Env()
    env.seed(seed=1)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    agent = DDPG(n_states=n_states, n_actions=n_actions, cfg=cfg)
    return env, agent

def train(cfg, env, agent):
    print("Start training !")
    rewards = []
    ma_rewards = []
    for i_ep in range(cfg.train_eps):
        state = env.reset()
        done = False
        ep_reward = 0
        i_step = 0
        while not done:
            i_step += 1
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            agent.memory.push(state, action, reward, next_state, done)
            agent.update()
            state = next_state

        if i_ep % 1 == 0:
            writer.add_scalar('train_ep_rewards', ep_reward, i_ep)
            print(f'Env:{i_ep + 1}/{cfg.train_eps}, Reward:{ep_reward:.2f}')
        rewards.append(ep_reward)
        ma_rewards.append(ep_reward)
    print('Finish training !')
    return {'rewards': rewards, 'ma_rewards': ma_rewards}

if __name__ == "__main__":
    cfg = get_args()
    env, agent = env_agent_config(cfg)
    res_dic = train(cfg, env, agent)
    writer.close()
    env.close()
    agent.save(cfg.model_path)
