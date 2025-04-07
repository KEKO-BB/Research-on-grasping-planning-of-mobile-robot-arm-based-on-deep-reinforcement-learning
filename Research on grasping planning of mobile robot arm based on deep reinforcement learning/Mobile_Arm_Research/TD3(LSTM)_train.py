import os
import sys
import datetime
import argparse
from TD3_LSTM import TD3  # 使用LSTM版本的TD3
from Dynamic_env_6_double_obstacle_关键预测 import Mobile_Arm_Env
from torch.utils.tensorboard import SummaryWriter
from Normal_Ounoise import OUNoise, NormalizedActions, GaussianExploration  # 导入高斯噪声类

# 加入 TensorBoard的显示
writer = SummaryWriter("log")


def get_args():
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(description="hyperparameters")
    parser.add_argument('--algo_name', default='TD3', type=str, help="name of algorithm")
    parser.add_argument('--env_name', default='Dynamic_Arm', type=str, help="name of environment")
    parser.add_argument('--train_eps', default=5000, type=int, help="episodes of training")
    parser.add_argument('--test_eps', default=500, type=int, help="episodes of testing")
    parser.add_argument('--gamma', default=0.99, type=float, help="discounted factor")
    parser.add_argument('--critic_lr', default=1e-4, type=float, help="learning rate of critic")
    parser.add_argument('--actor_lr', default=1e-5, type=float, help="learning rate of actor")
    parser.add_argument('--memory_capacity', default=10000, type=int, help="memory capacity")
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--target_update', default=2, type=int)
    parser.add_argument('--soft_tau', default=1e-2, type=float)
    parser.add_argument('--hidden_dim', default=400, type=int)

    # TD3特有的参数
    parser.add_argument('--policy_noise', default=0.05, type=float, help="standard deviation of policy noise")
    parser.add_argument('--noise_clip', default=0.1, type=float, help="maximum absolute value of policy noise")
    parser.add_argument('--policy_update_freq', default=2, type=int, help="frequency of policy update")

    parser.add_argument('--device', default='cuda', type=str, help='cpu or cuda')
    parser.add_argument('--result_path', default=f"{curr_time}/results/")
    parser.add_argument('--model_path', default=f"{curr_time}/models/")
    parser.add_argument('--save_fig', default=True, type=bool)

    # LSTM相关参数
    parser.add_argument('--lstm_dim', default=256, type=int, help="dimension of LSTM hidden state")
    parser.add_argument('--lstm_layers', default=1, type=int, help="number of layers in LSTM")
    parser.add_argument('--time_step_features', default=15, type=int, help="number of features per time step in LSTM")

    args = parser.parse_args()
    return args


def env_agent_config(cfg, seed=1):
    env = Mobile_Arm_Env()
    env.seed(seed=1)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    agent = TD3(n_states=n_states, n_actions=n_actions, cfg=cfg)  # 将LSTM参数传递给TD3类
    return env, agent


def train(cfg, env, agent):
    print("Start training !")

    # 加入高斯噪声
    gess_noise = GaussianExploration(env.action_space)  # 高斯噪声

    rewards = []
    ma_rewards = []  # 每轮的滑动平均奖励
    for i_ep in range(cfg.train_eps):
        state = env.reset()
        done = False
        ep_reward = 0
        i_step = 0
        while not done:
            i_step += 1
            action = agent.choose_action(state)
            action = gess_noise.get_action(action, i_step)  # 给动作添加噪声
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            agent.memory.push(state, action, reward, next_state, done)  # 压入经验池
            agent.update()  # 更新神经网络参数
            state = next_state

        if (i_ep) % 1 == 0:
            writer.add_scalar('train_ep_rewards', ep_reward, i_ep)  # 显示数据图像
            writer.add_scalar('policy_loss', agent.actor_policy_loss, i_ep)
            writer.add_scalar('critic1_loss', agent.critic1_value_loss, i_ep)
            writer.add_scalar('critic2_loss', agent.critic2_value_loss, i_ep)
            writer.add_scalar('every_ep_steps', i_step, i_ep)
            print(f'Env:{i_ep + 1}/{cfg.train_eps}, Reward:{ep_reward:.2f}')
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
    print('Finish training !')

    return {'rewards': rewards, 'ma_rewards': ma_rewards}


def test(cfg, env, agent):
    print("Start testing!")
    print(f'Env:{cfg.env_name}, Algorithm:{cfg.algo_name}, Device:{cfg.device}')

    # 初始化统计变量
    failed_both = 0
    success_avoid_obstacle_but_failed_grasp = 0
    success_grasp_but_failed_avoid = 0
    success_both = 0

    rewards = []
    ma_rewards = []
    for i_ep in range(cfg.test_eps):
        state = env.reset()
        done = False
        ep_reward = 0
        i_step = 0

        while not done:
            i_step += 1
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            state = next_state

        # 根据ep_reward的值判断情况
        if ep_reward < -20:
            failed_both += 1
        elif -20 < ep_reward < 0:
            success_avoid_obstacle_but_failed_grasp += 1
        elif 0 < ep_reward < 20:
            success_grasp_but_failed_avoid += 1
        elif ep_reward > 20:
            success_both += 1
        writer.add_scalar('test_ep_rewards1', ep_reward, i_ep)  # 显示数据图像
        writer.add_scalar('reach_target_steps', i_step, i_ep)  # 到达目标点的步数
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        print(f"Epside:{i_ep + 1}/{cfg.test_eps}, Reward:{ep_reward:.1f}")
    # 在TensorBoard中记录统计数据
    writer.add_scalar('Failed_Both', failed_both, cfg.test_eps)
    writer.add_scalar('Success_Avoid_Obstacle_But_Failed_Grasp', success_avoid_obstacle_but_failed_grasp, cfg.test_eps)
    writer.add_scalar('Success_Grasp_But_Failed_Avoid', success_grasp_but_failed_avoid, cfg.test_eps)
    writer.add_scalar('Success_Both', success_both, cfg.test_eps)

    print('Finish testing!')
    return {'rewards': rewards, 'ma_rewards': ma_rewards}


if __name__ == "__main__":
    cfg = get_args()
    # training
    env, agent = env_agent_config(cfg)
    res_dic = train(cfg, env, agent)
    writer.close()
    env.close()
    agent.save(cfg.model_path)

    #################################模型#######################################
    paths = ''  # 5000轮 结束时间：

    # test
    env, agent = env_agent_config(cfg)
    agent.load(path=paths)
    test(cfg, env, agent)
    env.close()
    writer.close()

    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # get current time
    print("stop time at :{}".format(curr_time))
