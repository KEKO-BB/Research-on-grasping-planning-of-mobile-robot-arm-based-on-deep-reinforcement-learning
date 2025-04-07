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


class InverseKinematics_planning():

    def __init__(self, L2, L3, L4, base_height, offset_y, offset_z, Connectioner) -> None:
        self.L2 = L2
        self.L3 = L3
        self.L4 = L4
        self.base_height = base_height
        self.offset_y = offset_y
        self.offset_z = offset_z
        # 考虑到L4与关节4有偏移，计算虚拟连杆长度
        self.virtual_L4 = np.sqrt((L4 + offset_z) ** 2 + offset_y ** 2)
        self.Connectioner = Connectioner

    def random_target(self):
        self.Connectioner.robot_model.rotateAllAngle_2([0.5 * np.pi, 0, 0.5 * np.pi, 0, 0.5 * np.pi])
        self.Connectioner.robot_model.Target_random_wall()

    def path_planning(self):

        _, Target_position = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.target,
            -1,
            vrep_sim.simx_opmode_blocking
        )
        # 考虑到机械臂末端相对于关节4的偏移
        target_y = Target_position[1]
        target_z = Target_position[2]
        target_z -= self.offset_z

        # 使用几何方法计算关节角度
        # 首先，计算关节2和目标点之间的距离
        distance = np.sqrt(target_y ** 2 + target_z ** 2)

        # 余弦定理计算关节3的角度
        cos_angle3 = (self.L2 ** 2 + self.L3 ** 2 - distance ** 2) / (2 * self.L2 * self.L3)
        angle3 = np.arccos(np.clip(cos_angle3, -1.0, 1.0))

        # 正弦定理计算关节2的角度
        sin_angle3 = np.sqrt(1 - cos_angle3 ** 2)
        angle2 = np.arctan2(target_z, target_y) - np.arctan2(self.L3 * sin_angle3, self.L2 + self.L3 * cos_angle3)

        # 根据几何关系计算关节4的角度
        angle4 = np.arctan2(target_z - self.L3 * np.sin(angle3), target_y - self.L3 * np.cos(angle3)) - angle3

        # 将角度限制在指定的范围内
        if angle2 < -0.5 * np.pi:
            angle2 = -0.5 * np.pi
        elif angle2 > 0.5 * np.pi:
            angle2 = 0.5 * np.pi

        if angle3 < 0:
            angle3 = 0
        elif angle3 > 0.75 * np.pi:
            angle2 = 0.75 * np.pi

        if angle4 < -0.25 * np.pi:
            angle4 = -0.25 * np.pi
        elif angle4 > 0.5 * np.pi:
            angle4 = 0.5 * np.pi

        theta1 = 0.5 * np.pi
        theta2 = angle2
        theta3 = angle3
        theta4 = angle4
        theta5 = 0
        joint_point = [theta1, theta2, theta3, theta4, theta5]
        self.Connectioner.robot_model.rotateAllAngle_2(joint_point)  # 做动作
        return np.degrees(angle2), np.degrees(angle3), np.degrees(angle4)


if __name__ == "__main__":
    Connectioner = Connection()  # 实例化一个连接对象()
    Connectioner.Connect_verp()  # 连接coppeliasim并初始化机器人模型(机器模型已经创建)
    InverseKinematics_planning = InverseKinematics_planning(L2=0.11671250143275243, L3=0.09498204484085858,
                                                            L4=0.1662244183562812,
                                                            base_height=0.16807551681995392,
                                                            offset_y=0.008192494308877892, offset_z=0.0280763506889343,
                                                            Connectioner=Connectioner)
    InverseKinematics_planning.random_target()
    InverseKinematics_planning.path_planning()


