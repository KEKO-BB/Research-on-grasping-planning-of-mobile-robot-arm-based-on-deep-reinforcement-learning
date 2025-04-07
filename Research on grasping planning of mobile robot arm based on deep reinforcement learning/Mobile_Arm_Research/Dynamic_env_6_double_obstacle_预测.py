# -*- utf-8 -*-
import math
import time
import gym
from gym.utils import seeding
from gym import spaces
import numpy as np
import sys

sys.path.append('../VREP_RemoteAPIs')
sys.path.append('../')
from connect_collpeliasim import Connection
import VREP_RemoteAPIs.sim as vrep_sim
import InverseKinematics02


class Mobile_Arm_Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, action_type='continuous') -> None:
        super(Mobile_Arm_Env, self).__init__()
        self.action_type = action_type
        # 米为单位状态空间的范围

        # 修改为0.5到0.5之间（缩小步长）
        ############### action的变化范围 ########################
        self.arm_joint_1_range = 0
        self.arm_joint_2_range = [-1, 1]
        self.arm_joint_3_range = [-1, 1]
        self.arm_joint_4_range = [-1, 1]
        self.arm_joint_5_range = 0
        self.speed_range = [-0.5, 0.5]  # 待修改
        ########################################
        ################# 车的速度必须正在此范围内，不能让车停下来 #######################
        self.base_speed_max = 1.5
        self.base_speed_min = 0.2
        self.set_base_speed = 0
        self.speed_current = 0
        ##########################################
        self.path = []  # 保存姿态路径
        ######################################
        self.last_robot_state = np.zeros(264)  #

        self.end_to_target_shortdis = 0.02  # 判断可抓取的最小距离

        self.target_current_position = None  # 记录每一轮的目标点位置
        self.obstacle_current_position = None  # 记录每一轮障碍物的位置

        # 定义状态空间
        self.low = np.array([0] * 264, dtype=np.float32)
        self.high = np.array([0] * 264, dtype=np.float32)

        # 定义动作空间
        self.action_low = np.array(
            [
                # self.arm_joint_1_range[0],
                self.arm_joint_4_range[0],
                self.arm_joint_3_range[0],
                self.arm_joint_2_range[0],
                self.speed_range[0]
                # self.arm_joint_5_range
            ]
        )

        self.action_high = np.array(
            [
                # self.arm_joint_1_range[1],
                self.arm_joint_4_range[1],
                self.arm_joint_3_range[1],
                self.arm_joint_2_range[1],
                self.speed_range[1]
                # self.arm_joint_5_range
            ]
        )
        # 添加用于存储相对位置的列表
        self.history_positions = {
            'end': [], 
            'dis1':[], 
            'arm_joint4':[],
            'dis2':[],  
            'arm_joint3':[],
            'dis3':[],  
            'base_link': [],
            'obstacle':[],
            'mark': []
        }
        self.history_length = 5  # 历史长度，存储过去5个时间步的位置
        # 假设每个历史位置增加了2个浮点数，总共6个历史位置
        extra_state_size = 44 * self.history_length
        state_size = 44 + extra_state_size  # 原始状态大小加上额外的历史信息大小
        #############################################################
        self.observation_space = spaces.Box(low=self.low, high=self.high, shape=(state_size,), dtype=np.float32)
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
        ################# 人工势能场法 ######################
        self.InverseKinematics_function = InverseKinematics02.InverseKinematics_planning(L1=0.01351793110370636,
                                                                                         L2=0.10496972501277924,
                                                                                         L3=0.09500536322593689,
                                                                                         L4=0.175,
                                                                                         base_height=0.15455463528633118,
                                                                                         offset_z=0.027126595377922058,
                                                                                         joint2_range=(
                                                                                             -np.pi / 2, np.pi / 2),
                                                                                         joint3_range=(
                                                                                             -0.25 * np.pi,
                                                                                             0.75 * np.pi),
                                                                                         joint4_range=(
                                                                                             -0.25 * np.pi,
                                                                                             0.5 * np.pi),
                                                                                         Connectioner=self.Connectioner)
        self.InverseKinematics_path = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_current_state(self):
        '''获取状态,状态由8个维度组成
        '''
        # 关节相对位置
        _, base_link_target = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.base_link_handle,
            # -1,
            self.Connectioner.robot_model.target,
            vrep_sim.simx_opmode_blocking
        )

        _, base_link_collision_1 = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.base_link_handle,
            # -1,
            self.Connectioner.robot_model.obstacle_handle_1,
            vrep_sim.simx_opmode_blocking
        )
        _, base_link_collision_2 = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.base_link_handle,
            # -1,
            self.Connectioner.robot_model.obstacle_handle_2,
            vrep_sim.simx_opmode_blocking
        )

        _, end_target = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.arm_end_handle,
            # -1,
            self.Connectioner.robot_model.target,
            vrep_sim.simx_opmode_blocking
        )

        _, end_collision_1 = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.arm_end_handle,
            # -1,
            self.Connectioner.robot_model.obstacle_handle_1,
            vrep_sim.simx_opmode_blocking
        )
        _, end_collision_2 = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.arm_end_handle,
            # -1,
            self.Connectioner.robot_model.obstacle_handle_2,
            vrep_sim.simx_opmode_blocking
        )

        _, arm_joint4_target = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.arm_joint_4,
            # -1,
            self.Connectioner.robot_model.target,
            vrep_sim.simx_opmode_blocking
        )

        _, arm_joint4_collision_1 = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.arm_joint_4,
            # -1,
            self.Connectioner.robot_model.obstacle_handle_1,
            vrep_sim.simx_opmode_blocking
        )
        _, arm_joint4_collision_2 = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.arm_joint_4,
            # -1,
            self.Connectioner.robot_model.obstacle_handle_2,
            vrep_sim.simx_opmode_blocking
        )

        _, arm_joint3_target = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.arm_joint_3,
            # -1,
            self.Connectioner.robot_model.target,
            vrep_sim.simx_opmode_blocking
        )

        _, arm_joint3_collision_1 = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.arm_joint_3,
            # -1,
            self.Connectioner.robot_model.obstacle_handle_1,
            vrep_sim.simx_opmode_blocking
        )
        _, arm_joint3_collision_2 = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.arm_joint_3,
            # -1,
            self.Connectioner.robot_model.obstacle_handle_2,
            vrep_sim.simx_opmode_blocking
        )
        # 障碍物速度
        _, obstacle1_linear_velocity, _ = vrep_sim.simxGetObjectVelocity(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.obstacle_handle_1,
            vrep_sim.simx_opmode_blocking
        )
        _, obstacle2_linear_velocity, _ = vrep_sim.simxGetObjectVelocity(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.obstacle_handle_2,
            vrep_sim.simx_opmode_blocking
        )

        reach_flag = 0
        collision_flag = 0

        _, arm_end = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.arm_end_handle,
            -1,
            vrep_sim.simx_opmode_blocking
        )

        # 超出范围的碰撞
        if arm_end[1] < -0.25 or arm_end[1] > 0.3 or arm_end[2] < 0.15:
            collision_flag = 1

        # 与障碍物的碰撞检测
        _, flag_collision = vrep_sim.simxGetIntegerSignal(
            self.Connectioner.client_ID,
            'collision_flag',
            vrep_sim.simx_opmode_blocking
        )
        _, flag_collision_2 = vrep_sim.simxGetIntegerSignal(
            self.Connectioner.client_ID,
            'collision_flag_2',
            vrep_sim.simx_opmode_blocking
        )
        _, flag_collision_wall = vrep_sim.simxGetIntegerSignal(
            self.Connectioner.client_ID,
            'collision_flag_wall',
            vrep_sim.simx_opmode_blocking
        )
        if flag_collision > 0 or flag_collision_2 > 0:
            collision_flag = 1
            # print('发生碰撞')

        end_target_dis = abs(
            math.sqrt(math.pow(end_target[0], 2) + math.pow(end_target[1], 2) + math.pow(end_target[2], 2)))
        end_target_horizontal_dis = abs(math.sqrt(math.pow(end_target[1], 2) + math.pow(end_target[2], 2)))
        end_collision1_horizontal_dis = abs(
            math.sqrt(math.pow(end_collision_1[1], 2) + math.pow(end_collision_1[2], 2)))
        end_collision2_horizontal_dis = abs(
            math.sqrt(math.pow(end_collision_2[1], 2) + math.pow(end_collision_2[2], 2)))
        arm_joint4_horizontal_target_dis = abs(
            math.sqrt(math.pow(arm_joint4_target[1], 2) + math.pow(arm_joint4_target[2], 2)))
        arm_joint4_horizontal_collision1_dis = abs(
            math.sqrt(math.pow(arm_joint4_collision_1[1], 2) + math.pow(arm_joint4_collision_1[2], 2)))
        arm_joint4_horizontal_collision2_dis = abs(
            math.sqrt(math.pow(arm_joint4_collision_2[1], 2) + math.pow(arm_joint4_collision_2[2], 2)))
        arm_joint3_horizontal_target_dis = abs(
            math.sqrt(math.pow(arm_joint3_target[1], 2) + math.pow(arm_joint3_target[2], 2)))
        arm_joint3_horizontal_collision1_dis = abs(
            math.sqrt(math.pow(arm_joint3_collision_1[1], 2) + math.pow(arm_joint3_collision_1[2], 2)))
        arm_joint3_horizontal_collision2_dis = abs(
            math.sqrt(math.pow(arm_joint3_collision_2[1], 2) + math.pow(arm_joint3_collision_2[2], 2)))
        base_speed = self.speed_current

        # 更新历史位置信息
        self.update_history_positions(end_target, end_collision_1, end_collision_2,
                                      end_target_horizontal_dis,end_collision1_horizontal_dis,end_collision2_horizontal_dis,
                                      arm_joint4_target,arm_joint4_collision_1,arm_joint4_collision_2,
                                      arm_joint4_horizontal_target_dis,arm_joint4_horizontal_collision1_dis,arm_joint4_horizontal_collision2_dis,
                                      arm_joint3_target,arm_joint3_collision_1, arm_joint3_collision_2,
                                      arm_joint3_horizontal_target_dis,arm_joint3_horizontal_collision1_dis,arm_joint3_horizontal_collision2_dis,
                                      base_link_target, base_link_collision_1, 
                                      base_link_collision_2,obstacle1_linear_velocity, obstacle2_linear_velocity,
                                      base_speed, reach_flag, collision_flag)
        # 到达目标点的标志
        if end_target_dis <= self.end_to_target_shortdis:
            reach_flag = 1

        # 创建固定长度的状态向量
        state_size = 44 + 44 * self.history_length
        robot_state = np.zeros(state_size)  # 调整状态向量的长度

        robot_state[0] = end_target[0]
        robot_state[1] = end_target[1]
        robot_state[2] = end_target[2]
        robot_state[3] = end_collision_1[0]
        robot_state[4] = end_collision_1[1]
        robot_state[5] = end_collision_1[2]
        robot_state[6] = end_collision_2[0]
        robot_state[7] = end_collision_2[1]
        robot_state[8] = end_collision_2[2]
        robot_state[9] = end_target_horizontal_dis
        robot_state[10] = end_collision1_horizontal_dis
        robot_state[11] = end_collision2_horizontal_dis
        robot_state[12] = arm_joint4_target[0]
        robot_state[13] = arm_joint4_target[1]
        robot_state[14] = arm_joint4_target[2]
        robot_state[15] = arm_joint4_collision_1[0]
        robot_state[16] = arm_joint4_collision_1[1]
        robot_state[17] = arm_joint4_collision_1[2]
        robot_state[18] = arm_joint4_collision_2[0]
        robot_state[19] = arm_joint4_collision_2[1]
        robot_state[20] = arm_joint4_collision_2[2]
        robot_state[21] = arm_joint4_horizontal_target_dis
        robot_state[22] = arm_joint4_horizontal_collision1_dis
        robot_state[23] = arm_joint4_horizontal_collision2_dis
        robot_state[24] = arm_joint3_target[0]
        robot_state[25] = arm_joint3_target[1]
        robot_state[26] = arm_joint3_target[2]
        robot_state[27] = arm_joint3_collision_1[0]
        robot_state[28] = arm_joint3_collision_1[1]
        robot_state[29] = arm_joint3_collision_1[2]
        robot_state[30] = arm_joint3_collision_2[0]
        robot_state[31] = arm_joint3_collision_2[1]
        robot_state[32] = arm_joint3_collision_2[2]
        robot_state[33] = arm_joint3_horizontal_target_dis
        robot_state[34] = arm_joint3_horizontal_collision1_dis
        robot_state[35] = arm_joint3_horizontal_collision2_dis
        robot_state[36] = base_link_target[0]
        robot_state[37] = base_link_collision_1[0]
        robot_state[38] = base_link_collision_2[0]
        robot_state[39] = obstacle1_linear_velocity[1]
        robot_state[40] = obstacle2_linear_velocity[1]
        robot_state[41] = base_speed
        robot_state[42] = reach_flag
        robot_state[43] = collision_flag

        # 填充历史位置信息到状态向量
        history_start_index = 44
        for key in self.history_positions:
            for i, pos in enumerate(self.history_positions[key]):
                # 检查 pos 的长度，并根据长度分配到 robot_state 中
                if len(pos) == 1:
                    robot_state[history_start_index + i] = pos[0]
                elif len(pos) == 2:
                    robot_state[history_start_index + i * 2: history_start_index + i * 2 + 2] = pos
                # 如果需要，为其他长度的 pos 添加更多的条件分支
            # 更新历史状态的起始索引
            history_start_index += len(self.history_positions[key][0]) * self.history_length

        # 打印状态向量的形状来确认它的长度是否正确
        # print("State vector shape:", robot_state.shape)
        return robot_state, arm_end

    def update_history_positions(self, end_target, end_collision_1, end_collision_2,
                                      end_target_horizontal_dis,end_collision1_horizontal_dis,end_collision2_horizontal_dis,
                                      arm_joint4_target,arm_joint4_collision_1,arm_joint4_collision_2,
                                      arm_joint4_horizontal_target_dis,arm_joint4_horizontal_collision1_dis,arm_joint4_horizontal_collision2_dis,
                                      arm_joint3_target,arm_joint3_collision_1, arm_joint3_collision_2,
                                      arm_joint3_horizontal_target_dis,arm_joint3_horizontal_collision1_dis,arm_joint3_horizontal_collision2_dis,
                                      base_link_target, base_link_collision_1, 
                                      base_link_collision_2,obstacle1_linear_velocity, obstacle2_linear_velocity,
                                      base_speed, reach_flag, collision_flag):
        # 更新历史相对位置
        self.history_positions['end'].append([end_target[0],end_target[1],end_target[2],end_collision_1[0],
                                              end_collision_1[1],end_collision_1[2],end_collision_2[0],end_collision_2[1],
                                              end_collision_2[2]])
        self.history_positions['dis1'].append([end_target_horizontal_dis,end_collision1_horizontal_dis,end_collision2_horizontal_dis])
        self.history_positions['arm_joint4'].append([arm_joint4_target[0],arm_joint4_target[1],arm_joint4_target[2],
                                                     arm_joint4_collision_1[0],arm_joint4_collision_1[1],arm_joint4_collision_1[2],
                                                     arm_joint4_collision_2[0],arm_joint4_collision_2[1],arm_joint4_collision_2[2]])
        self.history_positions['dis2'].append([arm_joint4_horizontal_target_dis,arm_joint4_horizontal_collision1_dis,arm_joint4_horizontal_collision2_dis])
        self.history_positions['arm_joint3'].append([arm_joint3_target[0], arm_joint3_target[1], arm_joint3_target[2],
                                            arm_joint3_collision_1[0], arm_joint3_collision_1[1], arm_joint3_collision_1[2],
                                            arm_joint3_collision_2[0], arm_joint3_collision_2[1], arm_joint3_collision_2[2]])
        self.history_positions['dis3'].append([arm_joint3_horizontal_target_dis,arm_joint3_horizontal_collision1_dis,arm_joint3_horizontal_collision2_dis])
        self.history_positions['base_link'].append([base_link_target[0],base_link_collision_1[0],base_link_collision_2[0]])
        self.history_positions['obstacle'].append([obstacle1_linear_velocity[1],obstacle2_linear_velocity[1]])
        self.history_positions['mark'].append([base_speed,reach_flag,collision_flag])
        
        # 保持历史位置的数量不超过设定的长度
        for key in self.history_positions:
            if len(self.history_positions[key]) > self.history_length:
                self.history_positions[key].pop(0)

    def step(self, action):
        # '''reverse_action'''
        # 合成动作
        # Gt 的设计
        Wt1 = 0
        if self.last_robot_state[38] < 0:
            Wt1 = 0
        else:
            Wt1 = 3.0 / (3 * abs(self.last_robot_state[36]) + 1)
        Wt2 = 1
        Wt3 = 1

        derta_q = np.zeros(4)
        derta_q[1] = self.InverseKinematics_path[0] - self.Connectioner.robot_model.arm_current_joints_red[1]
        derta_q[2] = self.InverseKinematics_path[1] - self.Connectioner.robot_model.arm_current_joints_red[2]
        derta_q[3] = self.InverseKinematics_path[2] - self.Connectioner.robot_model.arm_current_joints_red[3]

        ''' set action '''
        speed_action = self.speed_current + action[3]
        arm_joint2_action = self.Connectioner.robot_model.arm_current_joints_red[1] + (
                Wt1 * derta_q[1] + Wt2 * action[2]) * self.dt
        arm_joint3_action = self.Connectioner.robot_model.arm_current_joints_red[2] + (
                Wt1 * derta_q[2] + Wt2 * action[1]) * self.dt
        arm_joint4_action = self.Connectioner.robot_model.arm_current_joints_red[3] + (
                Wt1 * derta_q[3] + Wt2 * action[0]) * self.dt

        if arm_joint2_action < -0.5 * np.pi:
            arm_joint2_action = -0.5 * np.pi
        elif arm_joint2_action > 0.5 * np.pi:
            arm_joint2_action = 0.5 * np.pi

        # 有所改动
        if arm_joint3_action < -0.25 * np.pi:
            arm_joint3_action = -0.25 * np.pi
        elif arm_joint3_action > 0.75 * np.pi:
            arm_joint3_action = 0.75 * np.pi

        if arm_joint4_action < -0.25 * np.pi:
            arm_joint4_action = -0.25 * np.pi
        elif arm_joint4_action > 0.5 * np.pi:
            arm_joint4_action = 0.5 * np.pi

        if speed_action > self.base_speed_max:
            speed_action = self.base_speed_max
        elif speed_action < self.base_speed_min:
            speed_action = self.base_speed_min

        arm_joints = [0.5 * np.pi, arm_joint2_action, arm_joint3_action, arm_joint4_action, 0]
        self.Connectioner.robot_model.rotateAllAngle_2(arm_joints)  # 运行1个设置角度
        self.Connectioner.robot_model.set_wheels_sppeds(Wt3 * speed_action, 0)  # 运行1个step
        time.sleep(0.1)  # 延时以让底盘运动step
        self.speed_current = speed_action
        self.Connectioner.robot_model.base_stop()  # 停止以获取当前状态
        time.sleep(0.001)
        robot_state, arm_end = self.get_current_state()  # 获取当前状态
        self.step_count += 1  # 步数加1

        ''' get reward and judge if done '''
        # 奖励函数设置
        done = False

        # 障碍物惩罚
        # 与障碍物距离惩罚
        obstacle_distance_penalty1 = 0
        if -0.12 < robot_state[3] <= 0.04:
            # 定义安全距离阈值
            safe_distance = 0.14  # 假设关节与障碍物之间的安全距离为0.2米
            # 定义每个关节与障碍物距离的惩罚系数
            distance_penalty_coefficient = -20  # 当关节与障碍物的距离小于安全距离时的惩罚系数
            # 关节索引与其在状态向量中的对应位置
            joint_distance_indices = {
                5: 10,  # 机械臂末端与障碍物1的水平距离
                4: 22,  # 关节4与障碍物1的水平距离在状态向量中的索引
                3: 34,  # 关节3与障碍物1的水平距离在状态向量中的索引
            }
            # 遍历每个关键关节，计算与障碍物的距离惩罚
            for joint, index in joint_distance_indices.items():
                # 获取当前关节与障碍物的水平距离
                distance_to_obstacle = robot_state[index]
                # 如果距离小于安全距离阈值，则施加惩罚
                if distance_to_obstacle < safe_distance:
                    # 惩罚随着距离的减小而增大
                    penalty = (safe_distance - distance_to_obstacle) * distance_penalty_coefficient
                    # print(f"Joint {joint}, Distance: {distance_to_obstacle}, Penalty: {penalty}")
                    obstacle_distance_penalty1 += penalty

        obstacle_distance_penalty2 = 0
        if -0.12 < robot_state[6] <= 0.04:
            # 定义安全距离阈值
            safe_distance = 0.14  # 假设关节与障碍物之间的安全距离为0.2米
            # 定义每个关节与障碍物距离的惩罚系数
            distance_penalty_coefficient = -20  # 当关节与障碍物的距离小于安全距离时的惩罚系数
            # 关节索引与其在状态向量中的对应位置
            joint_distance_indices = {
                5: 11,  # 机械臂末端与障碍物2的水平距离
                4: 23,  # 关节4与障碍物2的水平距离在状态向量中的索引
                3: 35,  # 关节3与障碍物2的水平距离在状态向量中的索引
            }
            # 遍历每个关键关节，计算与障碍物的距离惩罚
            for joint, index in joint_distance_indices.items():
                # 获取当前关节与障碍物的水平距离
                distance_to_obstacle = robot_state[index]
                # 如果距离小于安全距离阈值，则施加惩罚
                if distance_to_obstacle < safe_distance:
                    # 惩罚随着距离的减小而增大
                    penalty = (safe_distance - distance_to_obstacle) * distance_penalty_coefficient
                    # print(f"Joint {joint}, Distance: {distance_to_obstacle}, Penalty: {penalty}")
                    obstacle_distance_penalty2 += penalty
        obstacle_distance_penalty = obstacle_distance_penalty1 + obstacle_distance_penalty2
        # 与障碍物碰撞惩罚
        R_collision = 0  # 碰撞惩罚
        if robot_state[43] > 0:
            R_collision = -40

        # 目标点奖励
        # 与目标物距离惩罚
        if robot_state[6]<0:
            Katt = 1
        else:
          Katt = 2
        R_end_to_target = -robot_state[9] * Katt

        # 到达目标点奖励
        if robot_state[42] > 0:
            R_touch = 100
            done = True
        else:
            R_touch = 0

        # 时间步数惩罚
        R_time_step = -0.1  # 时间步数惩罚

        # 停止条件
        # 车身已经超过目标点
        if arm_end[0] - self.target_current_position[0] > 0.05:
            done = True
        if self.step_count > 120:  # 不能超过120步
            done = True
        # print(R_end_to_target)
        reward = R_touch + R_time_step + R_end_to_target + R_collision + obstacle_distance_penalty
        # reward = R_end_to_target

        # 更新状态
        self.last_robot_state = robot_state

        return np.array(robot_state), reward, done, {}

    def reset(self):
        # print("上一轮轮步数：{}".format(self.step_count))
        self.step_count = 0  # 步数置0
        # 重置历史信息
        for key in self.history_positions:
            self.history_positions[key] = []

        arm_join_1 = 0.5 * np.pi
        arm_join_2 = 0
        arm_join_3 = 0.5 * np.pi
        arm_join_4 = 0
        arm_join_5 = 0

        self.Connectioner.robot_model.rotateAllAngle_2(
            [arm_join_1,
             arm_join_2,
             arm_join_3,
             arm_join_4,
             arm_join_5])
        # 先重置位置
        # 增加起始位置随机
        pos_new = [0, 0, 0]
        pos_new[0] = self.Connectioner.robot_model.base_link_position[0] + np.random.uniform(-0.02, 0.02)
        pos_new[1] = self.Connectioner.robot_model.base_link_position[1]
        pos_new[2] = self.Connectioner.robot_model.base_link_position[2]
        vrep_sim.simxSetObjectPosition(
            self.Connectioner.robot_model.client_ID,
            self.Connectioner.robot_model.base_link_handle,
            -1,
            pos_new,
            vrep_sim.simx_opmode_oneshot
        )
        vrep_sim.simxSetObjectOrientation(
            self.Connectioner.robot_model.client_ID,
            self.Connectioner.robot_model.base_link_handle,
            -1,
            self.Connectioner.robot_model.base_link_orientation,
            vrep_sim.simx_opmode_oneshot
        )

        self.Connectioner.robot_model.Target_random_double()  # 目标位置随机产生一次
        # self.Connectioner.robot_model.Obstacle_random_3()  # 障碍物随机位置一次

        _, self.target_current_position = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.target,
            -1,
            # self.Connectioner.robot_model.arm_end_handle,
            vrep_sim.simx_opmode_blocking
        )
        self.speed_current = 0
        # 运动学逆解法先确定，初始训练姿态
        self.InverseKinematics_path = self.InverseKinematics_function.path_planning()  # 这里就已经确定一个指导路径

        self.Connectioner.robot_model.rotateAllAngle_2(
            [arm_join_1,
             arm_join_2,
             0,
             arm_join_4,
             arm_join_5])

        time.sleep(0.001)

        reset_state, _ = self.get_current_state()  # 得到初始状态

        self.last_robot_state = reset_state

        return np.array(reset_state)

    # 用于自定义目标位置和障碍物位置
    def reset_simple(self):
        # print("上一轮轮步数：{}".format(self.step_count))
        self.step_count = 0  # 步数置0

        arm_join_1 = 0.5 * np.pi
        arm_join_2 = 0
        arm_join_3 = 0.5 * np.pi
        arm_join_4 = 0
        arm_join_5 = 0

        self.Connectioner.robot_model.rotateAllAngle_2(
            [arm_join_1,
             arm_join_2,
             arm_join_3,
             arm_join_4,
             arm_join_5])
        # 先重置位置
        # 增加起始位置随机
        pos_new = [0, 0, 0]
        pos_new[0] = self.Connectioner.robot_model.base_link_position[0] + np.random.uniform(-0.1, 0.1)
        pos_new[1] = self.Connectioner.robot_model.base_link_position[1]
        pos_new[2] = self.Connectioner.robot_model.base_link_position[2]
        vrep_sim.simxSetObjectPosition(
            self.Connectioner.robot_model.client_ID,
            self.Connectioner.robot_model.base_link_handle,
            -1,
            pos_new,
            vrep_sim.simx_opmode_oneshot
        )
        vrep_sim.simxSetObjectOrientation(
            self.Connectioner.robot_model.client_ID,
            self.Connectioner.robot_model.base_link_handle,
            -1,
            self.Connectioner.robot_model.base_link_orientation,
            vrep_sim.simx_opmode_oneshot
        )

        self.Connectioner.robot_model.Target_random_double()  # 目标位置随机产生一次
        # self.Connectioner.robot_model.Obstacle_random_3()  # 障碍物随机位置一次
        # self.Connectioner.robot_model.Set_Target_and_Obstacl() # 设置特定位置的目标为障碍物

        _, self.target_current_position = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.target,
            -1,
            # self.Connectioner.robot_model.arm_end_handle,
            vrep_sim.simx_opmode_blocking
        )
        _, self.obstacle_current_position = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.obstacle_handle,
            -1,
            vrep_sim.simx_opmode_blocking
        )
        self.Connectioner.robot_model.base_stop()
        self.speed_current = 0
        # 人工势能场法先确定，初始训练姿态
        self.Artifial_path = self.InverseKinematics_function.path_planning()  # 这里就已经确定一个指导路径

        self.Connectioner.robot_model.rotateAllAngle_2(
            [arm_join_1,
             arm_join_2,
             0,
             arm_join_4,
             arm_join_5])

        time.sleep(0.001)

        reset_state, _ = self.get_current_state()  # 得到初始状态

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

    for i in range(300):
        print("**************************************")
        action = env.action_space.sample()
        # print("action: {} and the shape is {}".format(action, action.shape))
        # action = env.reverse_action(action)
        # print("reverse_action: {}".format(action))
        state, reward, done, _ = env.step(action)
        # print("state: {} and the shape is {}".format(state, state.shape))
        print("reward: {} and step :{}".format(reward, i))

        # env.reset()
        if done:
            env.reset()
    env.close()

# 测试
#
