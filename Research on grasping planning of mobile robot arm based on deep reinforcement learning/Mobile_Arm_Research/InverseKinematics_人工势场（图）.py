# -*- utf-8 -*-

from typing import List
import sys
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from scipy.signal import savgol_filter

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

        # connect vrep
        self.Connectioner = Connectioner
        # self.Connectioner = Connection() # 实例化一个连接对象()
        # self.Connectioner.Connect_verp() # 连接coppeliasim并初始化机器人模型(机器模型已经创建)

    def get_end_position(self):
        _, arm_end = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.arm_end_handle,
            -1,
            # self.Connectioner.robot_model.mobile_frame,
            vrep_sim.simx_opmode_blocking
        )
        return arm_end

    def random_target(self):
        self.Connectioner.robot_model.rotateAllAngle_2([0.5 * np.pi, 0, 0.5 * np.pi, 0, 0.5 * np.pi])
        self.Connectioner.robot_model.Target_random_wall()

    def get_attractive_energy(self):
        """
            获取目标产生引力势能
        """
        _, Target_position = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.target,
            -1,
            # self.Connectioner.robot_model.mobile_frame,
            vrep_sim.simx_opmode_blocking
        )
        arm_end = self.get_end_position()  # 获取末端位置

        # end_to_target_x_dis = abs(arm_end[0] - Target_position[0])
        end_to_target_y_dis = abs(arm_end[1] - Target_position[1])
        end_to_target_z_dis = abs(arm_end[2] - Target_position[2])

        end_to_target_dis_yz_2 = math.pow(end_to_target_y_dis, 2) + math.pow(end_to_target_z_dis, 2)

        power_att = 0.5 * self.Katt * end_to_target_dis_yz_2  # 引力势能

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

        U_i = list()  # 存储所有的斥力势能,y轴，机器人底座一边宽0.14，z轴机器人底座高0.14
        if abs(dy1) < dmy and dy1 != 0:  # y轴方向上的斥力
            temp_rep = 0.5 * self.Krep * math.pow((1 / abs(dy1) - 1 / dmy), 2)
            U_i.append(temp_rep)
        else:
            U_i.append(0)

        if abs(dy2) < dmy and dy2 != 0:  # z轴方向上的斥力
            temp_rep = 0.5 * self.Krep * math.pow((1 / abs(dy2) - 1 / dmy), 2)
            U_i.append(temp_rep)
        else:
            U_i.append(0)

        if abs(dz1) < dmz and dz1 != 0:  # z轴方向上的斥力
            temp_rep = 0.5 * self.Krep * math.pow((1 / abs(dz1) - 1 / dmz), 2)
            U_i.append(temp_rep)
        else:
            U_i.append(0)

        if abs(dz2) < dmz and dz2 != 0:  # z轴方向上的斥力
            temp_rep = 0.5 * self.Krep * math.pow((1 / abs(dz2) - 1 / dmz), 2)
            U_i.append(temp_rep)
        else:
            U_i.append(0)

        if dy1 <= 0 or dy2 <= 0 or dz1 <= 0 or dz2 < 0:
            U_i.append(1000000)

        return sum(U_i)

    def path_planning(self):
        """
        人工势能场构建轨迹
        """
        # 初始化数据收集列表
        iterations = []  # 存储迭代次数
        times = []  # 存储迭代时间
        errors = []  # 存储误差值

        start_time = datetime.now()  # 开始计时

        # 获取当前状态
        self.Connectioner.robot_model.get_current_joint_red()
        theta = self.Connectioner.robot_model.arm_current_joints_red
        path = [theta]  # 存储路径
        iteration_count = 0  # 初始化迭代计数器
        while True:
            """
            初始化合力势能，和关节角度列表
            """
            U = []
            joint_list = []
            theta1 = 0.5 * np.pi
            # for theta1 in [theta[0] - self.Art_step, theta[0], theta[0] + self.Art_step]:
            for theta2 in [theta[1] - self.Art_step, theta[1], theta[1] + self.Art_step]:
                for theta3 in [theta[2] - self.Art_step, theta[2], theta[2] + self.Art_step]:
                    for theta4 in [theta[3] - self.Art_step, theta[3], theta[3] + self.Art_step]:
                        theta5 = theta[4]

                        # if theta1 < -0.5*np.pi:
                        #     theta1 = -0.5*np.pi
                        # elif theta1 > 0.5*np.pi:
                        #     theta1 = 0.5*np.pi

                        if theta2 < -0.5 * np.pi:
                            theta2 = -0.5 * np.pi
                        elif theta2 > 0.5 * np.pi:
                            theta2 = 0.5 * np.pi

                        if theta3 < 0:
                            theta3 = 0
                        elif theta3 > 0.75 * np.pi:
                            theta3 = 0.75 * np.pi

                        if theta4 < -0.25 * np.pi:
                            theta4 = -0.25 * np.pi
                        elif theta4 > 0.5 * np.pi:
                            theta4 = 0.5 * np.pi

                        joint_point = [theta1, theta2, theta3, theta4, theta5]
                        self.Connectioner.robot_model.rotateAllAngle_2(joint_point)  # 做动作
                        time.sleep(0.001)
                        Uatt = self.get_attractive_energy()  # 获取引力势能
                        Urep = self.get_repulsive_energy()  # 获取斥力势能
                        flag_u = Uatt
                        U.append(Uatt + Urep)
                        joint_list.append(joint_point)
                        iteration_count += 1  # 增加迭代计数器

                        # 每次迭代后，收集当前时间和当前误差
                        current_time = datetime.now()
                        elapsed_time = (current_time - start_time).total_seconds()
                        times.append(elapsed_time)

                        current_error = self.calculate_current_error()  # 计算当前误差
                        errors.append(current_error)

                        iterations.append(iteration_count)  # 记录迭代次数
            # 检查终止条件，例如误差小于一定阈值
            # if current_error < 1e-1:
            #     break
            # 找到最小的合力势能
            index = U.index(min(U))
            # 获取最小合力势能对应的角度
            path.append(joint_list[index])
            # print(path[-1])
            # 更新
            theta = joint_list[index]
            # 如果路径中最后两步一样了则就到了梯度为0 的地方了
            if round(path[-1][1], 4) == round(path[-2][1], 4) and \
                    round(path[-1][2], 4) == round(path[-2][2], 4) and \
                    round(path[-1][3], 4) == round(path[-2][3], 4):
                break

        # 确保 window_length 是奇数，并且小于 errors 列表的长度
        if len(errors) >= 3:  # 至少需要3个数据点才能进行滤波
            if len(errors) % 2 == 0:
                window_length = len(errors) - 1  # 如果是偶数，减去1使其变为奇数
            else:
                window_length = len(errors)

            # 选择一个合适的 poly_order，通常为3或更小的数，但要小于window_length
            poly_order = min(3, window_length - 1)

            smoothed_errors = savgol_filter(errors, window_length, poly_order)
        else:
            smoothed_errors = errors  # 如果数据点不足以滤波，则不进行滤波

        # 绘图部分
        plt.figure(figsize=(12, 6))

        # 绘制迭代次数与时间的关系图
        plt.subplot(1, 2, 1)
        plt.plot(times, iterations, marker='o', color='blue', label='Iterations')
        plt.xlabel('Time (s)')
        plt.ylabel('Iteration')
        plt.title('Iterations over Time')
        plt.legend()

        # 绘制误差随迭代次数变化的关系图
        plt.subplot(1, 2, 2)
        plt.plot(iterations, smoothed_errors, marker='o', color='orange', label='Error')
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.title('Error over Iterations')
        plt.legend()

        plt.tight_layout()
        plt.show()

        return path[-1]

    def calculate_current_error(self):
        # 这个方法应该计算和返回当前误差
        # 这里是一个示例，您需要根据自己的逻辑来实现这个方法
        _, target_position = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.target,
            -1,
            vrep_sim.simx_opmode_blocking
        )
        arm_end = self.get_end_position()
        return np.linalg.norm(np.array(arm_end) - np.array(target_position))


if __name__ == "__main__":
    Connectioner = Connection()  # 实例化一个连接对象()
    Connectioner.Connect_verp()  # 连接coppeliasim并初始化机器人模型(机器模型已经创建)
    Artifial_planning = Artifial_planning(end_position=1,  # 无效
                                          target_position=1,  # 无效
                                          Katt=1,
                                          Krep=0.1,
                                          d_o_y=0.01,  # 无效
                                          d_o_z=0.02,  # 无效
                                          Art_step=0.1,
                                          Connectioner=Connectioner)
    Artifial_planning.random_target()
    Artifial_planning.path_planning()


