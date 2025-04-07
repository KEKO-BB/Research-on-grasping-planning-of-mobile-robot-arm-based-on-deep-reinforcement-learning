# -*- utf-8 -*-
import sys, os

curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前路径
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # 加入到环境变量中

from Dynamic_env_6_double_obstacle_关键预测 import Mobile_Arm_Env  # 目前env7，半成功， env8 成功（成功率觉低）
import datetime

import argparse  # argparse 模块 用于命令行选项、参数和子命令解析器
# argparse 可让人轻松编写用户友好的命令行接口。程序定义它的需要的参数， 然后 argparse 将弄清 如何从 sys.argv 解析出哪些参数
# argparse 会自动生成帮助和使用手册，并在用户给程序传入无效参数时报出错误信息
import save_path

from TD3_PER import TD3
from common.utils import save_results, make_dir
from common.utils import plot_rewards, save_args
from Normal_Ounoise import OUNoise, NormalizedActions, GaussianExploration
from torch.utils.tensorboard import SummaryWriter

# 加入 tensorboad的显示
writer = SummaryWriter("log")


def get_args():
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # get current time
    # step1 创建解析器， ArgumentParser 对象，该对象 包含将命令行解析成 Python 数据类型所需的全部信息\academic\search
    parser = argparse.ArgumentParser(description="hyperparameters")
    # step2 添加参数
    parser.add_argument('--algo_name', default='TD3', type=str, help="name of algorithm")
    parser.add_argument('--env_name', default='Dynamic_Arm', type=str, help="name of environment")
    parser.add_argument('--train_eps', default=5000, type=int, help="episodes of training")  # 训练轮数
    parser.add_argument('--test_eps', default=500, type=int, help="episodes of testing")  # 测试轮数
    parser.add_argument('--gamma', default=0.99, type=float, help="discounted factor")  # 衰减因子
    parser.add_argument('--critic_lr', default=1e-4, type=float, help="learning rate of critic")  # 评价网络的学习率
    parser.add_argument('--actor_lr', default=1e-5, type=float, help="learning rate of actor")  # y原学习率为1e-5
    parser.add_argument('--memory_capacity', default=10000, type=int, help="memory capacity")  # 经验池容量
    parser.add_argument('--batch_size', default=100, type=int)  # 批量取出经验的大小
    parser.add_argument('--target_update', default=2, type=int)  # ?
    parser.add_argument('--soft_tau', default=1e-2, type=float)  # 软更新的tau，目标网络与测试网络参数之间
    parser.add_argument('--hidden_dim', default=400, type=int)  # 线性化维度（神经元更新中使用）

    # TD3特有的参数
    parser.add_argument('--policy_noise', default=0.05, type=float, help="standard deviation of policy noise")
    parser.add_argument('--noise_clip', default=0.1, type=float, help="maximum absolute value of policy noise")
    parser.add_argument('--policy_update_freq', default=2, type=int, help="frequency of policy update")

    parser.add_argument('--device', default='cuda', type=str, help='cpu or cuda')
    parser.add_argument('--result_path', default=curr_path + "/outputs/" + parser.parse_args().env_name + \
                                                 '/' + curr_time + '/results/')
    parser.add_argument('--model_path', default=curr_path + "/outputs/" + parser.parse_args().env_name + \
                                                '/' + curr_time + '/models/')  # path save models
    parser.add_argument('--save_fig', default=True, type=bool, help="if save figure or not")
    # step3 解析参数
    args = parser.parse_args()  # 解析参数
    return args


def env_agent_config(cfg, seed=1):
    env = Mobile_Arm_Env()
    env = NormalizedActions(env)
    env.seed(seed=1)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    agent = TD3(n_states=n_states, n_actions=n_actions, cfg=cfg)
    return env, agent


def beta_by_frame(frame_idx, beta_start=0.4, beta_frames=10000):
    return min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)


def train(cfg, env, agent):
    print("Start training !")
    print(f'Env:{cfg.env_name}, Algorithm:{cfg.algo_name}, Device:{cfg.device}')
    gess_noise = GaussianExploration(env.action_space)  # 高斯噪声

    rewards = []
    ma_rewards = []  # 每轮的滑动平均奖励
    total_steps = 0  # 在训练开始前初始化总步数
    for i_ep in range(cfg.train_eps):
        state = env.reset()
        done = False
        ep_reward = 0
        i_step = 0
        while not done:
            total_steps += 1  # 在每个步骤中增加总步数
            i_step += 1
            action = agent.choose_action(state)
            action = gess_noise.get_action(action, i_step)  # 加入噪声后
            next_state, reward, done, _ = env.step(action)
            env.current_action = action  ##
            ep_reward += reward
            agent.memory.push(state, action, reward, next_state, done)  # 压入经验池
            beta = beta_by_frame(total_steps)  # 使用总步数来计算beta值
            agent.update(beta=beta)  # 使用计算出的beta值调用update方法
            # agent.update(beta=cfg.beta_start + i_ep * (1.0 - cfg.beta_start) / cfg.train_eps)
            state = next_state

        if (i_ep) % 1 == 0:
            writer.add_scalar('train_ep_rewards', ep_reward, i_ep)  # 显示数据图像
            writer.add_scalar('policy_loss', agent.actor_policy_loss, i_ep)
            writer.add_scalar('critic1_loss', agent.critic1_value_loss, i_ep)
            writer.add_scalar('critic2_loss', agent.critic2_value_loss, i_ep)
            writer.add_scalar('td_error_mean', agent.td_error_mean, i_ep)
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


# 单轮测试
def test_simple(cfg, env, agent):
    print("Start testing!")
    print(f'Env:{cfg.env_name}, Algorithm:{cfg.algo_name}, Device:{cfg.device}')
    rewards = []
    ma_rewards = []
    for i_ep in range(1):
        # state = env.reset()
        state = env.reset_simple(i_ep)
        done = False
        ep_reward = 0
        i_step = 0
        while not done:
            i_step += 1
            action = agent.choose_action(state)  # 经过actor神经网络
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            state = next_state
        writer.add_scalar('test_ep_rewards1', ep_reward, i_ep)  # 显示数据图像
        writer.add_scalar('reach_target_steps', i_step, i_ep)  # 到达目标点的步数
        rewards.append(ep_reward)
        # save_path.save_csv(env.path,['speed','joint1','joint2','joint3','joint4','joint5'],2) # 保存路径为csv
        # save_path.save_csv(env.distance_and_speed, ['distance','speed'],'distance-speed-DDPG')
        save_path.save_csv(env.end_pose_world, ['x', 'y', 'z'], 'end_pose_world-3(ddpg)')  # 保存轨迹
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        print(f"Epside:{i_ep + 1}/{cfg.test_eps}, Reward:{ep_reward:.1f}")
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
    # env, agent = env_agent_config(cfg)
    # agent.load(path=paths)
    # test(cfg, env, agent)
    # env.close()
    # writer.close()

    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # get current time
    print("stop time at :{}".format(curr_time))
