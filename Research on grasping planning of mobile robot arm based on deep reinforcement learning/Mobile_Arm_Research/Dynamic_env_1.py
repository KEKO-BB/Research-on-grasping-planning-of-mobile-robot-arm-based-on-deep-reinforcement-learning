# -*- utf-8 -*-
from cmath import pi
import math
import time
import gym
from gym.utils import seeding
from gym  import spaces, logger
import numpy as np
import sys
sys.path.append('../VREP_RemoteAPIs')
sys.path.append('../')
from connect_collpeliasim import Connection
import VREP_RemoteAPIs.sim as vrep_sim
import Artifial_potential_control


class Mobile_Arm_Env(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, action_type = 'continuous') -> None:
        super(Mobile_Arm_Env,self).__init__()
        self.action_type = action_type
        # 米为单位状态空间的范围
        
        # 修改为0.5到0.5之间（缩小步长）
        ############### action的变化范围 ########################
        self.arm_joint_1_range = 0
        self.arm_joint_2_range = [-1, 1]
        self.arm_joint_3_range = [-1, 1]
        self.arm_joint_4_range = [-1, 1]
        self.arm_joint_5_range = 0
        self.speed_range = [-0.5, 0.5] # 待修改
        ########################################   
        ################# 车的速度必须正在此范围内，不能让车停下来 #######################
        self.base_speed_max = 0.8
        self.base_speed_min = 0.2
        self.set_base_speed = 0
        self.speed_current = 0
        ##########################################
        self.path = [] # 保存姿态路径
        ######################################
        self.last_robot_state = np.zeros(29) # 

        self.end_to_target_shortdis = 0.02 # 判断可抓取的最小距离

        self.target_current_position = None # 记录每一轮的目标点位置
        self.obstacle_current_position = None # 记录每一轮障碍物的位置

        self.low = np.array(
            [
              0,        # end_target_x
              0,        # end_target_y
              0,        # end_target_z
              0,        # end_collision_x
              0,        # end_collision_y
              0,        # end_collision_z
              0,        # end_target_dis
              0,        # end_collision_dis
              0,        # arm_joint4_target_x
              0,        # arm_joint4_target_y
              0,        # arm_joint4_target_z
              0,        # arm_joint4_collision_x
              0,        # arm_joint4_collision_y
              0,        # arm_joint4_collision_z
              0,        # arm_joint4_target_dis
              0,        # arm_joint4_collision_dis
              0,        # arm_joint3_target_x
              0,        # arm_joint3_target_y
              0,        # arm_joint3_target_z
              0,        # arm_joint3_collision_x
              0,        # arm_joint3_collision_y
              0,        # arm_joint3_collision_z
              0,        # arm_joint3_target_dis
              0,        # arm_joint3_collision_dis
              0,        # base_link_target_x_dis
              0,        # base_link_collision_x_dis
              0,        # reach_flag
              0,         # collision_flag
              0
            ],
            dtype=np.float32,
        )

        self.high = np.array(
            [
              0,        # end_target_x
              0,        # end_target_y
              0,        # end_target_z
              0,        # end_collision_x
              0,        # end_collision_y
              0,        # end_collision_z
              0,        # end_target_dis
              0,        # end_collision_dis
              0,        # arm_joint4_target_x
              0,        # arm_joint4_target_y
              0,        # arm_joint4_target_z
              0,        # arm_joint4_collision_x
              0,        # arm_joint4_collision_y
              0,        # arm_joint4_collision_z
              0,        # arm_joint4_target_dis
              0,        # arm_joint4_collision_dis
              0,        # arm_joint3_target_x
              0,        # arm_joint3_target_y
              0,        # arm_joint3_target_z
              0,        # arm_joint3_collision_x
              0,        # arm_joint3_collision_y
              0,        # arm_joint3_collision_z
              0,        # arm_joint3_target_dis
              0,        # arm_joint3_collision_dis
              0,        # base_link_target_x_dis
              0,        # base_link_collision_x_dis
              0,        # reach_flag
              0,         # collision_flag
              0
            ], 
            dtype= np.float32,
        )

        self.action_low = np.array(
            [
                #self.arm_joint_1_range[0],
                self.arm_joint_4_range[0],
                self.arm_joint_3_range[0],
                self.arm_joint_2_range[0],
                self.speed_range[0]
                #self.arm_joint_5_range
            ]
        )

        self.action_high = np.array(
            [
                #self.arm_joint_1_range[1],
                self.arm_joint_4_range[1],
                self.arm_joint_3_range[1],
                self.arm_joint_2_range[1],
                self.speed_range[1]
                #self.arm_joint_5_range
            ]
        )

        self.observation_space = spaces.Box(low=self.low,high=self.high,shape=(29,), dtype=np.float32)
        self.action_space = spaces.Box(low=self.action_low, high=self.action_high, shape=(4,), dtype=np.float32)

        self.seed()
        self.state = None
        self.count = 0
        self.steps_beyond = 200
        self.step_count = 0
        self.dt = 0.1 # 增量时间

        # connect vrep
        self.Connectioner = Connection() # 实例化一个连接对象()
        self.Connectioner.Connect_verp() # 连接coppeliasim并初始化机器人模型(机器模型已经创建)
         ################# 人工势能场法 ######################
        self.Artifial_function = Artifial_potential_control.Artifial_planning(end_position=1, 
                                        target_position=1, # 无效
                                        Katt=1, #  
                                        Krep=0.1, # 
                                        d_o_y=0.01, # 无效
                                        d_o_z=0.02, # 无效
                                        Art_step=0.15,  # 步长
                                        Connectioner=self.Connectioner)
        self.Artifial_path = None
        #############################################################
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def normal_action(self, action): # 线性归一化处理到[-1,1], 2*(x-x_min)/(x_max-x_min)-1
        low_bound = self.action_space.low
        upper_bound = self.action_space.high
        action = 2*(action - low_bound)/(upper_bound - low_bound) -1
         # 参数1位输入数组， 参数2限定的最小值（数组也行但形状必须一样），参数3限定的最大值，超出最大则变最大，超出最小则变最小
        action = np.clip(action, -1, 1) 
        return action
    
    def reverse_action(self, action): # 将action从归一化状态返回到正常状态值
        low_bound = self.action_space.low
        upper_bound = self.action_space.high
        action = (action+1) *0.5* (upper_bound -low_bound) + low_bound
        action = np.clip(action, low_bound, upper_bound)
        return action

    def normal_state(self, state): # 将状态数据归一化处理[0,1] 最大最小标准化
        low_state = self.observation_space.low
        high_state = self.observation_space.high
        state = (state - low_state)/(high_state - low_state)
        # state = np.clip(state, -1, 1)
        return state
    

    def get_current_state(self):
        '''获取状态,状态由8个维度组成
        '''
        _,base_link_target = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID, 
            self.Connectioner.robot_model.base_link_handle, 
            #-1,
            self.Connectioner.robot_model.target,
            vrep_sim.simx_opmode_blocking
        )

        _,base_link_collision = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID, 
            self.Connectioner.robot_model.base_link_handle, 
            #-1,
            self.Connectioner.robot_model.obstacle_handle,
            vrep_sim.simx_opmode_blocking
        )
        
        _,end_target = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.arm_end_handle,
            #-1,
            self.Connectioner.robot_model.target,
            vrep_sim.simx_opmode_blocking
        )

        _,end_collison = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.arm_end_handle,
            #-1,
            self.Connectioner.robot_model.obstacle_handle,
            vrep_sim.simx_opmode_blocking
        )

        _,arm_joint4_target = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.arm_joint_4,
            #-1,
            self.Connectioner.robot_model.target,
            vrep_sim.simx_opmode_blocking
        )

        _,arm_joint4_collison = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.arm_joint_4,
            #-1,
            self.Connectioner.robot_model.obstacle_handle,
            vrep_sim.simx_opmode_blocking
        )

        _,arm_joint3_target = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.arm_joint_3,
            #-1,
            self.Connectioner.robot_model.target,
            vrep_sim.simx_opmode_blocking
        )

        _,arm_joint3_collison = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.arm_joint_3,
            #-1,
            self.Connectioner.robot_model.obstacle_handle,
            vrep_sim.simx_opmode_blocking
        )
         
        reach_flag = 0
        collision_flag = 0
        obstacle_flag = 0

        _,arm_end = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.arm_end_handle,
            -1,
            vrep_sim.simx_opmode_blocking
        )

        # 确定障碍物所指区域时左是右
        if self.obstacle_current_position[1] > 0 : 
            obstacle_flag = 1
        
        # 超出范围的碰撞
        if arm_end[1] < -0.25 or arm_end[1] > 0.3 or arm_end[2] < 0.15:
            collision_flag = 1

        # 与障碍物的碰撞检测
        _, flag_collision = vrep_sim.simxGetIntegerSignal(
            self.Connectioner.client_ID,
            'collision_flag',
            vrep_sim.simx_opmode_blocking
            )
        _, flag_collision_wall = vrep_sim.simxGetIntegerSignal(
            self.Connectioner.client_ID,
            'collision_flag_wall',
            vrep_sim.simx_opmode_blocking
            )
        if flag_collision > 0 or flag_collision_wall > 0:
            collision_flag = 1
            #print('发生碰撞')
        
        end_target_dis = abs(math.sqrt(math.pow(end_target[0], 2)+ math.pow(end_target[1], 2) + math.pow(end_target[2], 2)))

        end_target_horizontal_dis = abs(math.sqrt(math.pow(end_target[1], 2) + math.pow(end_target[2], 2)))
        end_collison_horizontal_dis = abs(math.sqrt(math.pow(end_collison[1], 2) + math.pow(end_collison[2], 2)))
        arm_joint4_horizontal_target_dis = abs(math.sqrt(math.pow(arm_joint4_target[1], 2) + math.pow(arm_joint4_target[2] , 2)))
        arm_joint4_horizontal_collison_dis = abs(math.sqrt(math.pow(arm_joint4_collison[1], 2) + math.pow(arm_joint4_collison[2], 2)))
        arm_joint3_horizontal_target_dis = abs(math.sqrt(math.pow(arm_joint3_target[1], 2) + math.pow(arm_joint3_target[2], 2)))
        arm_joint3_horizontal_collison_dis = abs(math.sqrt(math.pow(arm_joint3_collison[1], 2) + math.pow(arm_joint3_collison[2], 2)))

        # 到达目标点的标志
        if end_target_dis <= self.end_to_target_shortdis:
            reach_flag = 1

        robot_state = np.zeros(29)
        
        robot_state[0] = end_target[0]
        robot_state[1] = end_target[1]
        robot_state[2] = end_target[2]
        robot_state[3] = end_collison[0]
        robot_state[4] = end_collison[1]
        robot_state[5] = end_collison[2]
        robot_state[6] = end_target_horizontal_dis
        robot_state[7] = end_collison_horizontal_dis
        robot_state[8] = arm_joint4_target[0]
        robot_state[9] = arm_joint4_target[1]
        robot_state[10] = arm_joint4_target[2]
        robot_state[11] = arm_joint4_collison[0]
        robot_state[12] = arm_joint4_collison[1]
        robot_state[13] = arm_joint4_collison[2]
        robot_state[14] = arm_joint4_horizontal_target_dis
        robot_state[15] = arm_joint4_horizontal_collison_dis
        robot_state[16] = arm_joint3_target[0]
        robot_state[17] = arm_joint3_target[1]
        robot_state[18] = arm_joint3_target[2]
        robot_state[19] = arm_joint3_collison[0]
        robot_state[20] = arm_joint3_collison[1]
        robot_state[21] = arm_joint3_collison[2]
        robot_state[22] = arm_joint3_horizontal_target_dis
        robot_state[23] = arm_joint3_horizontal_collison_dis
        robot_state[24] = base_link_target[0]
        robot_state[25] = base_link_collision[0]
        robot_state[26] = reach_flag
        robot_state[27] = collision_flag
        robot_state[28] = obstacle_flag

        return robot_state, arm_end

    def step(self, action):
        #'''reverse_action'''
        # 合成动作
        #Gt 的设计
        Wt1 = 0
        if self.last_robot_state[25]<0:
            #Wt1=np.log(0.12*self.step_count+1)
            Wt1=0
        else:
            Wt1 =3.0/(3*abs(self.last_robot_state[24])+1)

        Wt2 = 1
        Wt3 = 1
        
        derta_q = np.zeros(4)
        derta_q[1] = self.Artifial_path[1] - self.Connectioner.robot_model.arm_current_joints_red[1]
        derta_q[2] = self.Artifial_path[2] - self.Connectioner.robot_model.arm_current_joints_red[2]
        derta_q[3] = self.Artifial_path[3] - self.Connectioner.robot_model.arm_current_joints_red[3]

        ''' set action '''
        speed_action = self.speed_current + action[3]
        arm_joint2_action = self.Connectioner.robot_model.arm_current_joints_red[1] + (Wt1*derta_q[1] + Wt2*action[2])*self.dt 
        arm_joint3_action = self.Connectioner.robot_model.arm_current_joints_red[2] + (Wt1*derta_q[2] + Wt2*action[1])*self.dt
        arm_joint4_action = self.Connectioner.robot_model.arm_current_joints_red[3] + (Wt1*derta_q[3] + Wt2*action[0])*self.dt

        if arm_joint2_action < -0.5*np.pi :
            arm_joint2_action = -0.5*np.pi
        elif arm_joint2_action > 0.5*np.pi:
            arm_joint2_action = 0.5*np.pi
        
        # 有所改动
        if arm_joint3_action < -0.25*np.pi:
            arm_joint3_action = -0.25*np.pi
        elif arm_joint3_action > 0.75*np.pi:
            arm_joint3_action = 0.75*np.pi

        if arm_joint4_action < -0.25*np.pi:
            arm_joint4_action = -0.25*np.pi
        elif arm_joint4_action > 0.5*np.pi:
            arm_joint4_action = 0.5*np.pi 
        
        if speed_action > self.base_speed_max:
            speed_action = self.base_speed_max
        elif speed_action < self.base_speed_min:
            speed_action = self.base_speed_min

        arm_joints = [0.5*np.pi, arm_joint2_action, arm_joint3_action, arm_joint4_action, 0]
        #print("step return is {}".format(arm_joints))
        self.Connectioner.robot_model.rotateAllAngle_2(arm_joints) # 运行1个设置角度
        self.Connectioner.robot_model.set_wheels_sppeds(Wt3*speed_action,0) # 运行1个step
        time.sleep(0.1) # 延时以让底盘运动step
        self.speed_current = speed_action
        self.Connectioner.robot_model.base_stop() # 停止以获取当前状态
        time.sleep(0.001)
        robot_state, arm_end= self.get_current_state() # 获取当前状态
        
        ##保存路径###
        # ls = [speed_action,0.5*np.pi, arm_joint2_action, arm_joint3_action, arm_joint4_action, 0]
        # self.path.append(ls)
        #####
        self.step_count += 1 # 步数加1

        ''' get reward and judge if done '''
        # 奖励函数设置
        Katt = 2
        R_touch = 0 # 主线任务
        R_collision = 0 # 碰撞惩罚
        R_time_step = -0.1   # 时间步数惩罚
        done = False
        
        # 距离奖励
        R_end_to_target = -robot_state[6]*Katt

        # 主线目标
        if robot_state[26] > 0:
            R_touch = 100
            done = True
        else:
            R_touch = 0

        # 与障碍物发生碰撞
        if robot_state[27] > 0 : 
             R_collision = -40
        # 车身已经
        if arm_end[0]-self.target_current_position[0] > 0.05:
             done = True

        if self.step_count > 120: # 不能超过100步
            done = True
        
        # 该step的奖励
        reward = R_touch + R_time_step + R_collision + \
                 R_end_to_target
        #R_end_to_target_change

        # 更新状态
        self.last_robot_state = robot_state

        return np.array(robot_state), reward, done, {}

        
    def reset(self):
        #print("上一轮轮步数：{}".format(self.step_count))
        self.step_count = 0 # 步数置0

        arm_join_1 = 0.5*np.pi
        arm_join_2 = 0
        arm_join_3 = 0.5*np.pi
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
        pos_new = [0,0,0]
        pos_new[0] = self.Connectioner.robot_model.base_link_position[0] + np.random.uniform(-0.1,0.1)
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

        self.Connectioner.robot_model.Target_random_double() # 目标位置随机产生一次
        self.Connectioner.robot_model.Obstacle_random_3() # 障碍物随机位置一次
        
        _,self.target_current_position = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.target,
            -1,
            #self.Connectioner.robot_model.arm_end_handle,
            vrep_sim.simx_opmode_blocking
        )
        _,self.obstacle_current_position = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.obstacle_handle,
            -1,
            vrep_sim.simx_opmode_blocking
        )
        self.Connectioner.robot_model.base_stop()
        self.speed_current = 0
        # 人工势能场法先确定，初始训练姿态
        self.Artifial_path = self.Artifial_function.path_planning() # 这里就已经确定一个指导路径

        self.Connectioner.robot_model.rotateAllAngle_2(
            [arm_join_1,
            arm_join_2,
            0,
            arm_join_4,
            arm_join_5])

        time.sleep(0.001)

        reset_state, _  = self.get_current_state() # 得到初始状态
        
        self.last_robot_state = reset_state
        
        return np.array(reset_state)
    
    # 用于自定义目标位置和障碍物位置
    def reset_simple(self):
        #print("上一轮轮步数：{}".format(self.step_count))
        self.step_count = 0 # 步数置0

        arm_join_1 = 0.5*np.pi
        arm_join_2 = 0
        arm_join_3 = 0.5*np.pi
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
        pos_new = [0,0,0]
        pos_new[0] = self.Connectioner.robot_model.base_link_position[0] + np.random.uniform(-0.1,0.1)
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

        self.Connectioner.robot_model.Target_random_double() # 目标位置随机产生一次
        self.Connectioner.robot_model.Obstacle_random_3() # 障碍物随机位置一次
        #self.Connectioner.robot_model.Set_Target_and_Obstacl() # 设置特定位置的目标为障碍物
        
        _,self.target_current_position = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.target,
            -1,
            #self.Connectioner.robot_model.arm_end_handle,
            vrep_sim.simx_opmode_blocking
        )
        _,self.obstacle_current_position = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.obstacle_handle,
            -1,
            vrep_sim.simx_opmode_blocking
        )
        self.Connectioner.robot_model.base_stop()
        self.speed_current = 0
        # 人工势能场法先确定，初始训练姿态
        self.Artifial_path = self.Artifial_function.path_planning() # 这里就已经确定一个指导路径

        self.Connectioner.robot_model.rotateAllAngle_2(
            [arm_join_1,
            arm_join_2,
            0,
            arm_join_4,
            arm_join_5])

        time.sleep(0.001)

        reset_state, _  = self.get_current_state() # 得到初始状态
        
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

    for i in range(10):
        print("**************************************")
        action = env.action_space.sample()
        action = env.normal_action(action)
        #print("action: {} and the shape is {}".format(action, action.shape))
        #action = env.reverse_action(action)
        #print("reverse_action: {}".format(action))
        state, reward, done, _ =env.step(action)
        #print("state: {} and the shape is {}".format(state, state.shape))
        print("reward: {} and step :{}".format(reward,i))

        #env.reset()
        if done:
            env.reset()
    env.close()
        

# 测试 
#                 