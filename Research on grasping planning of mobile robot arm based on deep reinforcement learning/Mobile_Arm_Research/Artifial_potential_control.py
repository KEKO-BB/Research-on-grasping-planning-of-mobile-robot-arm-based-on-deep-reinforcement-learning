# -*- utf-8 -*-

from typing import List
import sys
sys.path.append('../VREP_RemoteAPIs')
sys.path.append('../')
from connect_collpeliasim import Connection
import VREP_RemoteAPIs.sim as vrep_sim
import math
import time
import numpy as np
from operator import *

class Artifial_planning():
    """
        人工势能场法测试
    """
    def __init__(self, end_position, target_position, Katt, Krep, d_o_y, d_o_z, Art_step, Connectioner) -> None:
        self.end_position = end_position
        self.target_position = target_position

        # 引力增益，斥力增益
        self.Katt = Katt
        self.Krep = Krep
        # 障碍最小警报距离
        self.d_o_y = d_o_y
        self.d_o_z = d_o_z
        # 人工势能长的步长
        self.Art_step = Art_step
        self.Connectioner = Connectioner
        
    def get_end_position(self):
        _, arm_end = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.arm_end_handle,
            -1,
            #self.Connectioner.robot_model.mobile_frame,
            vrep_sim.simx_opmode_blocking
        )
        return arm_end
    
    def random_target(self):
        self.Connectioner.robot_model.rotateAllAngle_2([0.5*np.pi, 0, 0.5*np.pi,0, 0.5*np.pi])
        self.Connectioner.robot_model.Target_random_wall()

    def get_attractive_energy(self):
        """
            获取目标产生引力势能
        """
        _,Target_position = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.target,
            -1,
            #self.Connectioner.robot_model.mobile_frame,
            vrep_sim.simx_opmode_blocking
        )
        arm_end = self.get_end_position() # 获取末端位置

        # end_to_target_x_dis = abs(arm_end[0] - Target_position[0])
        end_to_target_y_dis = abs(arm_end[1] - Target_position[1])
        end_to_target_z_dis = abs(arm_end[2] - Target_position[2])

        end_to_target_dis_yz_2 = math.pow(end_to_target_y_dis,2)+ math.pow(end_to_target_z_dis,2)

        power_att = 0.5*self.Katt*end_to_target_dis_yz_2 # 引力势能

        return power_att

    def get_repulsive_energy(self):
        """
            获取斥力势能，主要是车身为障碍
        """
        # 不可触碰区域
        car_z_1 = 0.15
        car_z_2 = 0.42
        car_y_1 = 0.1
        car_y_2 = 0.3
        # 不可触碰区域开始产生斥力范围
        dmz = 0.02
        dmy = 0.02

        arm_end = self.get_end_position()

        dy1 = arm_end[1] - car_y_1
        dy2 = car_y_2 - arm_end[1]
        dz1 = arm_end[2] - car_z_1
        dz2 = car_z_2 - arm_end[2]

        U_i = list() # 存储所有的斥力势能,y轴，机器人底座一边宽0.14，z轴机器人底座高0.14
        if abs(dy1) < dmy and dy1 != 0: # y轴方向上的斥力
            temp_rep = 0.5*self.Krep*math.pow((1/abs(dy1) - 1/dmy),2)
            U_i.append(temp_rep)
        else:
            U_i.append(0)

        if abs(dy2) < dmy and dy2 != 0: # z轴方向上的斥力
            temp_rep = 0.5*self.Krep*math.pow((1/abs(dy2) - 1/dmy),2)
            U_i.append(temp_rep)
        else:
            U_i.append(0)

        if abs(dz1) < dmz and dz1 != 0: # z轴方向上的斥力
            temp_rep = 0.5*self.Krep*math.pow((1/abs(dz1) - 1/dmz),2)
            U_i.append(temp_rep)
        else:
            U_i.append(0)

        if abs(dz2) < dmz and dz2 != 0: # z轴方向上的斥力
            temp_rep = 0.5*self.Krep*math.pow((1/abs(dz2) - 1/dmz),2)
            U_i.append(temp_rep)
        else:
            U_i.append(0)

        if dy1 <=0 or dy2 <=0 or dz1 <=0 or dz2 <0:
            U_i.append(1000000)

        return sum(U_i)

    def path_planning(self):
        """
        人工势能场构建轨迹
        """
        self.Connectioner.robot_model.get_current_joint_red()

        theta = self.Connectioner.robot_model.arm_current_joints_red # 获取当前状态

        path = [theta]

        while True:
            """
            初始化合力势能，和关节角度列表
            """
            U = []
            joint_list = []
            theta1 = 0.5*np.pi
            #for theta1 in [theta[0] - self.Art_step, theta[0], theta[0] + self.Art_step]:
            for theta2 in [theta[1] - self.Art_step, theta[1], theta[1] + self.Art_step]:
                for theta3 in [theta[2] - self.Art_step, theta[2], theta[2] + self.Art_step]:
                    for theta4 in [theta[3] - self.Art_step, theta[3], theta[3] + self.Art_step]:
                        theta5 = theta[4]
                            
                        # if theta1 < -0.5*np.pi:
                        #     theta1 = -0.5*np.pi
                        # elif theta1 > 0.5*np.pi:
                        #     theta1 = 0.5*np.pi
                        
                        if theta2 < -0.5*np.pi:
                            theta2 = -0.5*np.pi
                        elif theta2 > 0.5*np.pi:
                            theta2 = 0.5*np.pi
                            
                        if theta3 < 0:
                            theta3 = 0
                        elif theta3 > 0.75*np.pi:
                            theta3 = 0.75*np.pi
                            
                        if theta4 < -0.25*np.pi :
                            theta4 = -0.25*np.pi
                        elif theta4 > 0.5*np.pi:
                            theta4 = 0.5*np.pi

                        joint_point = [theta1, theta2, theta3, theta4, theta5]
                        self.Connectioner.robot_model.rotateAllAngle_2(joint_point) # 做动作
                        time.sleep(0.001)
                        Uatt = self.get_attractive_energy() # 获取引力势能
                        Urep = self.get_repulsive_energy() # 获取斥力势能
                        flag_u = Uatt
                        U.append(Uatt+Urep)
                        joint_list.append(joint_point)

            # 找到最小的合力势能
            index = U.index(min(U))
            # 获取最小合力势能对应的角度
            path.append(joint_list[index])
            #print(path[-1])
            # 更新
            theta = joint_list[index]
            # 如果路径中最后两步一样了则就到了梯度为0 的地方了
            if round(path[-1][1],4) == round(path[-2][1], 4) and \
                round(path[-1][2], 4) == round(path[-2][2], 4) and \
                 round(path[-1][3], 4) == round(path[-2][3], 4):
                    break
        #print('势能场初始姿态确定')
        
        return path[-1]

if __name__ == "__main__":
    Connectioner = Connection() # 实例化一个连接对象()
    Connectioner.Connect_verp() # 连接coppeliasim并初始化机器人模型(机器模型已经创建)
    Artifial_planning = Artifial_planning(end_position=1, # 无效
                                        target_position=1, # 无效
                                        Katt=1,
                                        Krep=0.1,
                                        d_o_y=0.01,# 无效
                                        d_o_z=0.02, # 无效
                                        Art_step=0.1,
                                        Connectioner=Connectioner)
    Artifial_planning.random_target()
    Artifial_planning.path_planning()
    

        