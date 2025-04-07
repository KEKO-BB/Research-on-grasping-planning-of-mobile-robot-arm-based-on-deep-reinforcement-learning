# -*- utf-8 -*-
from scipy.optimize import minimize
import sys

sys.path.append('../VREP_RemoteAPIs')
sys.path.append('../')
from connect_collpeliasim import Connection
import VREP_RemoteAPIs.sim as vrep_sim
import numpy as np


class InverseKinematics_planning():
    def __init__(self, L1, L2, L3, L4, base_height, offset_z, joint2_range, joint3_range, joint4_range,
                 Connectioner) -> None:
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.L4 = L4
        self.base_height = base_height  # 底盘高度
        self.offset_z = offset_z  # 连杆4相对于关节4在z方向上的偏移量
        self.joint2_range = joint2_range
        self.joint3_range = joint3_range
        self.joint4_range = joint4_range
        self.Connectioner = Connectioner

    def random_target(self):
        self.Connectioner.robot_model.Target_random_double()   # 设置目标位置的移动范围

    def forward_kinematics(self, angles):  # 运动学正解解出机械臂末端坐标
        theta2, theta3, theta4 = angles
        y = (self.L2 * np.sin(theta2) +
             self.L3 * np.sin(theta2 + theta3) +
             self.L4 * np.sin(theta2 + theta3 + theta4))
        z = self.base_height + self.L1 + self.offset_z + (self.L2 * np.cos(theta2) +
                                                          self.L3 * np.cos(theta2 + theta3) +
                                                          self.L4 * np.cos(theta2 + theta3 + theta4))
        return np.array([y, z])

    def objective_function(self, angles, target_position):  # 计算当前机械臂末端与目标位置之间的位置误差

        y, z = self.forward_kinematics(angles)
        return np.linalg.norm([y - target_position[0], z - target_position[1]])

    def calculate_inverse_kinematics(self):   # 通过优化算法SLSQP迭代优化位置误差并返回各个关节角度
        # 获取目标位置
        _, target_position = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.target,
            -1,
            vrep_sim.simx_opmode_blocking)
        # 只取y和z坐标
        target_position = np.array([target_position[1], target_position[2]])

        # # 定义初始猜测和界限，此处添加随机性(同一个目标点，可以逆解出不同的姿态)
        # initial_guess = np.array(
        #     [np.random.uniform(bound[0], bound[1]) for bound in
        #      [self.joint2_range, self.joint3_range, self.joint4_range]])
        # bounds = [self.joint2_range, self.joint3_range, self.joint4_range]

        # 定义初始猜测和界限
        initial_guess = np.array(
            [(bound[0] + bound[1]) / 2 for bound in [self.joint2_range, self.joint3_range, self.joint4_range]])
        bounds = [self.joint2_range, self.joint3_range, self.joint4_range]

        # 优化选项
        options = {
            'maxiter': 100,  # 最大迭代次数
            'ftol': 1e-6,  # 目标函数的容忍度
            'disp': False  # 显示优化过程
        }
        # 使用minimize函数进行优化
        result = minimize(self.objective_function, initial_guess, args=(target_position,), bounds=bounds,
                          method='SLSQP', options=options)

        if result.success:
            return result.x
        else:
            raise ValueError("Inverse kinematics calculation did not converge")

    def path_planning(self):

        joint_angles = self.calculate_inverse_kinematics()
        theta1 = 0.5 * np.pi
        theta2 = -joint_angles[0]
        theta3 = joint_angles[1]
        theta4 = joint_angles[2]
        theta5 = 0
        joint_point = [theta1, theta2, theta3, theta4, theta5]
        self.Connectioner.robot_model.rotateAllAngle_2(joint_point)  # 控制机械臂旋转
        return theta2, theta3, theta4


if __name__ == "__main__":
    Connectioner = Connection()  # 实例化一个连接对象()
    Connectioner.Connect_verp()  # 连接coppeliasim并初始化机器人模型(机器模型已经创建)
    InverseKinematics_planning = InverseKinematics_planning(L1=0.01351793110370636, L2=0.10496972501277924,
                                                            L3=0.09500536322593689, L4=0.172,
                                                            base_height=0.15455463528633118,
                                                            offset_z=0.027126595377922058,
                                                            joint2_range=(-np.pi / 2, np.pi / 2),
                                                            joint3_range=(-0.25 * np.pi, 0.75 * np.pi),
                                                            joint4_range=(-0.25 * np.pi, 0.5 * np.pi),
                                                            Connectioner=Connectioner)
    InverseKinematics_planning.random_target()
    InverseKinematics_planning.path_planning()
