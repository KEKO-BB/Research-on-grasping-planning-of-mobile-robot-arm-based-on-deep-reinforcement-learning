# -*- utf-8 -*-
from cmath import pi
import math
import time
import gym
from gym.utils import seeding
from gym import spaces, logger
import numpy as np
import sys

sys.path.append('../VREP_RemoteAPIs')
sys.path.append('../')
from connect_collpeliasim import Connection
import VREP_RemoteAPIs.sim as vrep_sim
from vrep_pre_joints import Pro_Pose
from vrep_pre_joints import Range_eare


class Mobile_Arm_Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, action_type='continuous') -> None:
        super(Mobile_Arm_Env, self).__init__()
        self.action_type = action_type
        # 米为单位状态空间的范围

        # 修改为0.5到0.5之间（缩小步长）
        ############### action的变化范围 ########################
        action_base_down = -1
        action_base_up = 1
        self.arm_joint_1_range = [action_base_down, action_base_up]
        self.arm_joint_2_range = [action_base_down, action_base_up]
        self.arm_joint_3_range = [action_base_down, action_base_up]
        self.arm_joint_4_range = [action_base_down, action_base_up]
        self.arm_joint_5_range = [action_base_down, action_base_up]
        ########################################
        self.last_robot_state = np.zeros(24)  #
        self.end_to_target_shortdis = 0.01  # 判断可抓取的最小距离
        self.arm_base_handle = None  # 机械臂底座

        self.low = np.array(
            [
                0,  # end_x
                0,  # end_y
                0,  # end_z
                0,  # end_target_dis
                0,  # end_target_dx
                0,  # end_target_dy
                0,  # end_target_dz
                0,  # arm_joint4_x
                0,  # arm_joint4_y
                0,  # arm_joint4_z
                0,  # arm_joint4_target_dis
                0,  # arm_joint4_dx
                0,  # arm_joint4_dy
                0,  # arm_joint4_dz
                0,  # arm_joint3_x
                0,  # arm_joint3_y
                0,  # arm_joint3_z
                0,  # arm_joint3_target_dis
                0,  # arm_joint3_dx
                0,  # arm_joint3_dy
                0,  # arm_joint3_dz
                0,  # reach_flag
                0,  # collision_flag
                0  # car_body_collision_flag
            ],
            dtype=np.float32,
        )

        self.high = np.array(
            [
                0,  # end_x
                0,  # end_y
                0,  # end_z
                0,  # end_target_dis
                0,  # end_target_dx
                0,  # end_target_dy
                0,  # end_target_dz
                0,  # arm_joint4_x
                0,  # arm_joint4_y
                0,  # arm_joint4_z
                0,  # arm_joint4_target_dis
                0,  # arm_joint4_dx
                0,  # arm_joint4_dy
                0,  # arm_joint4_dz
                0,  # arm_joint3_x
                0,  # arm_joint3_y
                0,  # arm_joint3_z
                0,  # arm_joint3_target_dis
                0,  # arm_joint3_dx
                0,  # arm_joint3_dy
                0,  # arm_joint3_dz
                0,  # reach_flag
                0,  # collision_flag
                0  # car_body_collision_flag
            ],
            dtype=np.float32,
        )

        self.action_low = np.array(
            [
                self.arm_joint_1_range[0],
                self.arm_joint_2_range[0],
                self.arm_joint_3_range[0],
                self.arm_joint_4_range[0],
                # self.arm_joint_5_range[0]
            ]
        )

        self.action_high = np.array(
            [
                self.arm_joint_1_range[1],
                self.arm_joint_2_range[1],
                self.arm_joint_3_range[1],
                self.arm_joint_4_range[1],
                # self.arm_joint_5_range[1]
            ]
        )

        self.observation_space = spaces.Box(low=self.low, high=self.high, shape=(24,), dtype=np.float32)
        self.action_space = spaces.Box(low=self.action_low, high=self.action_high, shape=(4,), dtype=np.float32)

        self.seed()
        self.state = None
        self.count = 0
        self.steps_beyond = 200
        self.step_count = 0
        self.dt = 0.1  # 增量时间

        # connect vrep
        self.Connectioner = Connection()  # 实例化一个连接对象()
        self.Connectioner.Connect_verp()  # 连接coppeliasim并初始化机器人模型(机器模型已经创建)
        ########################### 区域图 ################################
        self.graph_eare = Pro_Pose(0.14,  0.3, -30, -10, 10, 30, 30, 60,90)  # 区域图
        ## 只有初始状态下的
        self.arm_base_handle = self.Connectioner.robot_model.base_frame  # 机械臂底座
        ## 记录当前目标区域
        self.current_eare = 0
        self.current_eare_range = Range_eare()  # 区域结构体
        self.current_target_angle = None  # 目标夹角
        self.current_target_dis = None  # 目标极坐标长度
        self.current_target_phi = None  # 目标方位角
        self.cha_target_angle = None  # 夹角补偿值
        self.base_joints_pose = None  # 基准姿态
        ##############################################################
        # 测试#
        self.end_pose_world = []



    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def normal_action(self, action):  # 线性归一化处理到[-1,1], 2*(x-x_min)/(x_max-x_min)-1
        low_bound = self.action_space.low
        upper_bound = self.action_space.high
        action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
        # 参数1位输入数组， 参数2限定的最小值（数组也行但形状必须一样），参数3限定的最大值，超出最大则变最大，超出最小则变最小
        action = np.clip(action, -1, 1)
        return action

    def reverse_action(self, action):  # 将action从归一化状态返回到正常状态值
        low_bound = self.action_space.low
        upper_bound = self.action_space.high
        action = (action + 1) * 0.5 * (upper_bound - low_bound) + low_bound
        action = np.clip(action, low_bound, upper_bound)
        return action

    def normal_state(self, state):  # 将状态数据归一化处理[0,1] 最大最小标准化
        low_state = self.observation_space.low
        high_state = self.observation_space.high
        state = (state - low_state) / (high_state - low_state)
        # state = np.clip(state, -1, 1)
        return state

    def check_collision(self):
        _, flag_collision_0 = vrep_sim.simxGetIntegerSignal(
            self.Connectioner.client_ID,
            'collision_flag0',
            vrep_sim.simx_opmode_blocking
        )
        _, flag_collision_1 = vrep_sim.simxGetIntegerSignal(
            self.Connectioner.client_ID,
            'collision_flag1',
            vrep_sim.simx_opmode_blocking
        )
        _, flag_collision_2 = vrep_sim.simxGetIntegerSignal(
            self.Connectioner.client_ID,
            'collision_flag2',
            vrep_sim.simx_opmode_blocking
        )
        _, flag_collision_3 = vrep_sim.simxGetIntegerSignal(
            self.Connectioner.client_ID,
            'collision_flag3',
            vrep_sim.simx_opmode_blocking
        )
        _, flag_collision_4 = vrep_sim.simxGetIntegerSignal(
            self.Connectioner.client_ID,
            'collision_flag4',
            vrep_sim.simx_opmode_blocking
        )
        _, flag_collision_5 = vrep_sim.simxGetIntegerSignal(
            self.Connectioner.client_ID,
            'collision_flag5',
            vrep_sim.simx_opmode_blocking
        )

        if flag_collision_0 > 0 or flag_collision_1 > 0 or \
                flag_collision_2 > 0 or flag_collision_3 > 0 or \
                flag_collision_4 > 0 or flag_collision_5 > 0:
            return 1
            # print('发生碰撞')
        return 0
    def get_current_state(self):
        '''获取状态,状态由8个维度组成
        '''
        _, end_pose = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.arm_end_handle,
            # -1,
            self.arm_base_handle,
            vrep_sim.simx_opmode_blocking
        )

        _, target_pose = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.target,
            # -1,
            self.arm_base_handle,
            vrep_sim.simx_opmode_blocking
        )

        _, arm_joint4_pose = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.arm_joint_4,
            # -1,
            self.arm_base_handle,
            vrep_sim.simx_opmode_blocking
        )

        _, arm_joint3_pose = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.arm_joint_3,
            # -1,
            self.arm_base_handle,
            vrep_sim.simx_opmode_blocking
        )

        #### 测试 #############################################
        self.end_pose_world.append(end_pose)
        #######################################################

        reach_flag = 0
        collision_flag = 0  # 包括两个部分，1是超出范围，2是发生碰撞
        car_body_collision_flag = 0  # 车身碰撞

        # 地面碰撞
        if end_pose[2] <= 0:
            collision_flag = 1
        # 机械臂与底盘碰撞
        _, flag_collision_gripper = vrep_sim.simxGetIntegerSignal(
            self.Connectioner.client_ID,
            'collision_flag_base',
            vrep_sim.simx_opmode_blocking
        )
        if flag_collision_gripper > 0:
            car_body_collision_flag = 1

        # 相关的距离的计算
        end_cha_x = end_pose[0] - target_pose[0]
        end_cha_y = end_pose[1] - target_pose[1]
        end_cha_z = end_pose[2] - target_pose[2]
        end_target_dis = abs(math.sqrt(math.pow(end_cha_x, 2) + math.pow(end_cha_y, 2) + math.pow(end_cha_z, 2)))

        arm_joint4_cha_x = arm_joint4_pose[0] - target_pose[0]
        arm_joint4_cha_y = arm_joint4_pose[1] - target_pose[1]
        arm_joint4_cha_z = arm_joint4_pose[2] - target_pose[2]
        arm_joint4_target_dis = abs(
            math.sqrt(math.pow(arm_joint4_cha_x, 2) + math.pow(arm_joint4_cha_y, 2) + math.pow(arm_joint4_cha_z, 2)))

        arm_joint3_cha_x = arm_joint3_pose[0] - target_pose[0]
        arm_joint3_cha_y = arm_joint3_pose[1] - target_pose[1]
        arm_joint3_cha_z = arm_joint3_pose[2] - target_pose[2]
        arm_joint3_target_dis = abs(
            math.sqrt(math.pow(arm_joint3_cha_x, 2) + math.pow(arm_joint3_cha_y, 2) + math.pow(arm_joint3_cha_z, 2)))

        # 到达目标点的标志
        if end_target_dis <= self.end_to_target_shortdis:
            reach_flag = 1

        robot_state = np.zeros(24)

        robot_state[0] = end_pose[0]
        robot_state[1] = end_pose[1]
        robot_state[2] = end_pose[2]
        robot_state[3] = end_target_dis
        robot_state[4] = abs(end_cha_x)
        robot_state[5] = abs(end_cha_y)
        robot_state[6] = abs(end_cha_z)
        robot_state[7] = arm_joint4_pose[0]
        robot_state[8] = arm_joint4_pose[1]
        robot_state[9] = arm_joint4_pose[2]
        robot_state[10] = arm_joint4_target_dis
        robot_state[11] = abs(arm_joint4_cha_x)
        robot_state[12] = abs(arm_joint4_cha_y)
        robot_state[13] = abs(arm_joint4_cha_z)
        robot_state[14] = arm_joint3_pose[0]
        robot_state[15] = arm_joint3_pose[1]
        robot_state[16] = arm_joint3_pose[2]
        robot_state[17] = arm_joint3_target_dis
        robot_state[18] = abs(arm_joint3_cha_x)
        robot_state[19] = abs(arm_joint3_cha_y)
        robot_state[20] = abs(arm_joint3_cha_z)
        robot_state[21] = reach_flag
        robot_state[22] = collision_flag
        robot_state[23] = car_body_collision_flag

        return robot_state

    def step(self, action):
        # '''reverse_action'''
        # 合成动作
        cha_joint1 = self.base_joints_pose[0] - self.Connectioner.robot_model.arm_current_joints_red[0]
        cha_joint2 = self.base_joints_pose[1] - self.Connectioner.robot_model.arm_current_joints_red[1]
        cha_joint3 = self.base_joints_pose[2] - self.Connectioner.robot_model.arm_current_joints_red[2]
        cha_joint4 = self.base_joints_pose[3] - self.Connectioner.robot_model.arm_current_joints_red[3]

        w1 =0.6*np.log(self.step_count+1) #引导
        # w1 = 0
        w2 = 1
        # Gt 的设计
        ''' set action '''
        arm_joint1_action = self.Connectioner.robot_model.arm_current_joints_red[0] + (
                    w1 * cha_joint1 + w2 * action[0]) * self.dt
        arm_joint2_action = self.Connectioner.robot_model.arm_current_joints_red[1] + (
                    w1 * cha_joint2 + w2 * action[1]) * self.dt
        arm_joint3_action = self.Connectioner.robot_model.arm_current_joints_red[2] + (
                    w1 * cha_joint3 + w2 * action[2]) * self.dt
        arm_joint4_action = self.Connectioner.robot_model.arm_current_joints_red[3] + (
                    w1 * cha_joint4 + w2 * action[3]) * self.dt
        # arm_joint5_action = self.Connectioner.robot_model.arm_current_joints_red[4] + action[4]*self.dt

        if arm_joint1_action > (7 / 36) * np.pi:
            arm_joint1_action = (7 / 36) * np.pi
        elif arm_joint1_action < -(7 / 36) * np.pi:
            arm_joint1_action = -(7 / 36) * np.pi

        if arm_joint2_action > 0.25 * np.pi:
            arm_joint2_action = 0.25 * np.pi
        elif arm_joint2_action < -0.5 * np.pi:
            arm_joint2_action = -0.5 * np.pi

        if arm_joint3_action < 0:
            arm_joint3_action = 0
        elif arm_joint3_action > 0.75 * np.pi:
            arm_joint3_action = 0.75 * np.pi

        if arm_joint4_action < 0:
            arm_joint4_action = 0
        elif arm_joint4_action > 0.5 * np.pi:
            arm_joint4_action = 0.5 * np.pi

            # if arm_joint5_action < -0.5*np.pi:
        #     arm_joint5_action = -0.5*np.pi
        # elif arm_joint5_action > 0.5*np.pi:
        #     arm_joint5_action = 0.5*np.pi

        arm_joints = [arm_joint1_action, arm_joint2_action, arm_joint3_action, arm_joint4_action, 0]
        # print("step return is {}".format(arm_joints))
        self.Connectioner.robot_model.rotateAllAngle_2(arm_joints)  # 运行1个设置角度
        time.sleep(0.001)
        robot_state = self.get_current_state()  # 获取当前状态

        ##保存路径###
        # ls = [speed_action,0.5*np.pi, arm_joint2_action, arm_joint3_action, arm_joint4_action, 0]
        # self.path.append(ls)
        #####
        self.step_count += 1  # 步数加1

        ''' get reward and judge if done '''
        ######################## 奖励函数设置 ##############################
        K_dis = 30  # 距离惩罚的系数0-9
        k_near = 0.5

        R_touch = 0  # 主线任务
        R_end_to_target = 0
        R_collision = 0  # 碰撞惩罚
        R_collision_map = 0
        R_collision_car = 0
        done = False

        # 主线目标
        if robot_state[21] > 0:
            R_touch = 100
            done = True
        # 与障碍物发生碰撞
        if robot_state[22] > 0:  # 地面碰撞
            R_collision_map = -20
        if robot_state[23] > 0:  # 机身碰撞
            R_collision_car = -10
        R_collision = R_collision_car + R_collision_map

        # 欧式距离惩罚
        # R_end_to_target = -robot_state[3] * 15
        if robot_state[3] > 0.6:
            R_end_to_target = -np.arctan(50 * robot_state[3])
        elif robot_state[3] <= 0.6:
            R_end_to_target = -np.arctan(30 * robot_state[3])


        # 靠近奖励
        if robot_state[3] < 0.05:
            t = robot_state[3]/0.05
            R_near = (1.0 - t) * k_near
        else:
            R_near = 0.0

        # 时间步数惩罚
        R_time_step = -0.1

        # print("{}".format(R_end_to_target))
        # 不能超过100步
        if self.step_count > 100:
            done = True

        # 该step的奖励
        reward = R_touch + R_end_to_target + R_collision + R_near + R_time_step

        # 更新状态
        self.last_robot_state = robot_state

        return np.array(robot_state), reward, done, {}

    def reset(self):
        # print("上一轮轮步数：{}".format(self.step_count))
        self.step_count = 0  # 步数置0
        # 初始姿态
        arm_join_1 = 0
        arm_join_2 = 0
        arm_join_3 = 0.5 * np.pi
        arm_join_4 = 0.5 * np.pi
        arm_join_5 = 0

        self.Connectioner.robot_model.rotateAllAngle_2(
            [arm_join_1,
             arm_join_2,
             arm_join_3,
             arm_join_4,
             arm_join_5])

        # 先重置目标位置
        # self.Connectioner.robot_model.Target_random_vrep(0.165, 0.29,-0.5,0.5,0.5759,1.53)  # 目标位置
        self.Connectioner.robot_model.Target_random_vrep(0.175, 0.29, -0.5, 0.5, 0.57, 1.53)  # 目标位置

        # 先获取目标位置
        _, self.target_current_position = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.target,
            self.arm_base_handle,  # 机械臂底座
            vrep_sim.simx_opmode_blocking
        )
        # 获取区域编号
        self.current_eare = self.graph_eare.check_eare(self.target_current_position)
        # print(self.current_eare)
        # 获取目标夹角 和 水平距离
        self.current_target_dis, self.current_target_angle, self.current_target_phi = self.graph_eare.to_polar(
            self.target_current_position)
        # 根据区域编号获得基准姿态
        self.base_joints_pose = self.graph_eare.find_base_pose(self.current_eare)
        # 获取当前状态
        reset_state = self.get_current_state()  # 得到初始状态
        # 获取范围边界
        self.current_eare_range = self.graph_eare.find_range(0)  # 大范围
        # 记录状态
        self.last_robot_state = reset_state

        return np.array(reset_state)

    # 用于自定义目标位置和障碍物位置
    def reset_simple(self, index):
        # print("上一轮轮步数：{}".format(self.step_count))
        self.step_count = 0  # 步数置0
        # 初始姿态
        arm_join_1 = 0
        arm_join_2 = 0
        arm_join_3 = 0.5 * np.pi
        arm_join_4 = 0.5 * np.pi
        arm_join_5 = 0

        self.Connectioner.robot_model.rotateAllAngle_2(
            [arm_join_1,
             arm_join_2,
             arm_join_3,
             arm_join_4,
             arm_join_5])

        # 先重置目标位置
        # self.Connectioner.robot_model.Set_Target(index) # 目标位置
        self.Connectioner.robot_model.Target_random_reset_vrep(3)
        # 先获取目标位置
        _, self.target_current_position = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.target,
            self.arm_base_handle,  # 机械臂底座
            vrep_sim.simx_opmode_blocking
        )
        # 获取区域编号
        self.current_eare = self.graph_eare.check_eare(self.target_current_position)
        print(self.current_eare)
        # 获取目标夹角 和 水平距离
        self.current_target_dis, self.current_target_angle ,self.current_target_phi= self.graph_eare.to_polar(self.target_current_position)
        # 根据区域编号获得基准姿态
        self.base_joints_pose = self.graph_eare.find_base_pose(self.current_eare)
        # 获取当前状态
        reset_state = self.get_current_state()  # 得到初始状态
        # 获取范围边界
        self.current_eare_range = self.graph_eare.find_range(0)  # 大范围
        # 记录状态
        self.last_robot_state = reset_state

        return np.array(reset_state)

    def render(self):
        return None

    def close(self):
        vrep_sim.simxStopSimulation(self.Connectioner.robot_model.client_ID, vrep_sim.simx_opmode_blocking)
        vrep_sim.simxFinish(-1)
        print('Close the env !!!')
        return None


if __name__ == "__main__":
    env = Mobile_Arm_Env()
    env.reset()
    n_states = env.observation_space.shape[0]
    print(n_states)
    n_actions = env.action_space.shape[0]
    print(n_actions)

    for i in range(40):
        print("**************************************")
        action = env.action_space.sample()
        # action = env.normal_action(action)
        # print("action: {} and the shape is {}".format(action, action.shape))
        # action = env.reverse_action(action)
        # print("reverse_action: {}".format(action))
        state, reward, done, _ = env.step(action)
        # print("state: {} and the shape is {}".format(state, state.shape))
        print("reward: {} and step :{}".format(reward, i))

        # env.reset_simple(i)
        if done:
            env.reset()
    env.close()
# 测试
#