import numpy as np
from Dynamic_env_4_double_obstacle import Mobile_Arm_Env
from Normal_Ounoise import OUNoise, NormalizedActions, GaussianExploration

env = Mobile_Arm_Env()  # 创建您的环境实例
num_episodes = 400  # 要运行的初始探索轮数
state_min = np.inf * np.ones(env.observation_space.shape)
state_max = -np.inf * np.ones(env.observation_space.shape)
gess_noise = GaussianExploration(env.action_space)  # 高斯噪声

for episode in range(num_episodes):
    state = env.reset()
    done = False
    i_step = 0
    while not done:
        action = env.action_space.sample()  # 随机选择动作
        action = gess_noise.get_action(action, i_step)  # 加入噪声后
        next_state, _, done, _ = env.step(action)

        # 更新状态变量的最小和最大值
        state_min = np.minimum(state_min, next_state)
        state_max = np.maximum(state_max, next_state)

        state = next_state

    # 打印当前轮数
    print(f"Episode {episode + 1}/{num_episodes} completed")

# 打印结果
print("State minimum values:", state_min)
print("State maximum values:", state_max)
