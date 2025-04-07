# -*- utf-8 -*-

import sys
sys.path.append('../VREP_RemoteAPIs')
sys.path.append('../')
from connect_collpeliasim import Connection
import VREP_RemoteAPIs.sim as vrep_sim
import numpy as np


class InverseKinematics_planning():
    def __init__(self, Connectioner) -> None:
        self.Connectioner = Connectioner

    def path_planning(self):
        theta1 = 0.5*np.pi
        theta2 = 0
        theta3 = 0
        theta4 = 0
        theta5 = 0
        joint_point = [theta1, theta2, theta3, theta4, theta5]
        self.Connectioner.robot_model.rotateAllAngle_2(joint_point)  # 做动作
        _, arm1 = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.arm_joint_1,
            -1,
            vrep_sim.simx_opmode_blocking
        )
        _, arm2 = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.arm_joint_2,
            -1,
            vrep_sim.simx_opmode_blocking
        )
        _, arm3 = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.arm_joint_3,
            -1,
            vrep_sim.simx_opmode_blocking
        )
        _, arm4 = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.arm_joint_4,
            -1,
            vrep_sim.simx_opmode_blocking
        )
        _, end = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.arm_end_handle,
            -1,
            vrep_sim.simx_opmode_blocking
        )
        base_height=arm1[2]
        L1=arm2[2]-arm1[2]
        L2=arm3[2]-arm2[2]
        L3=arm4[2]-arm3[2]
        L4=end[2]-arm4[2]
        Ly=arm1[0]-end[0]
        Lz=arm1[1]-end[1]

        print(f'base_height:{base_height},L1:{L1},L2:{L2},L3:{L3},L4:{L4},Ly:{Ly},Lz:{Lz}')

if __name__ == "__main__":
    Connectioner = Connection() # 实例化一个连接对象()
    Connectioner.Connect_verp() # 连接coppeliasim并初始化机器人模型(机器模型已经创建)
    InverseKinematics_planning = InverseKinematics_planning(Connectioner=Connectioner)
    InverseKinematics_planning.path_planning()

        