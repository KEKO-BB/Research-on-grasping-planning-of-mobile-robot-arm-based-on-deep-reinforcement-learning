# -*- utf-8 -*-
import csv

from scipy.optimize import minimize
import sys
import time
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

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
        self.base_height = base_height
        self.offset_z = offset_z
        self.joint2_range = joint2_range
        self.joint3_range = joint3_range
        self.joint4_range = joint4_range
        self.Connectioner = Connectioner
        self.iter_times = []
        self.times = []
        self.function_values = []
        self.start_time = None

    def callback_function(self, xk):
        # Called after each iteration
        self.iter_times.append(len(self.iter_times) + 1)
        self.times.append(time.time() - self.start_time)

    def random_target(self):
        self.Connectioner.robot_model.Target_random_double()

    def forward_kinematics(self, angles):
        theta2, theta3, theta4 = angles
        y = (self.L2 * np.sin(theta2) +
             self.L3 * np.sin(theta2 + theta3) +
             self.L4 * np.sin(theta2 + theta3 + theta4))
        z = self.base_height + self.L1 + self.offset_z + (self.L2 * np.cos(theta2) +
                                                          self.L3 * np.cos(theta2 + theta3) +
                                                          self.L4 * np.cos(theta2 + theta3 + theta4))
        return np.array([y, z])

    def objective_function(self, angles, target_position):
        y, z = self.forward_kinematics(angles)
        func_val = np.linalg.norm([y - target_position[0], z - target_position[1]])
        # Update the function values only if the callback_function has been called
        if len(self.function_values) < len(self.iter_times):
            self.function_values.append(func_val)
        return func_val

    def calculate_inverse_kinematics(self):
        # 获取目标位置
        _, target_position = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.target,
            -1,
            vrep_sim.simx_opmode_blocking)
        # 只取y和z坐标
        self.target_position = np.array([target_position[1], target_position[2]])
        # 定义初始猜测和界限
        initial_guess = np.array(
            [(bound[0] + bound[1]) / 2 for bound in [self.joint2_range, self.joint3_range, self.joint4_range]])
        bounds = [self.joint2_range, self.joint3_range, self.joint4_range]

        # 优化选项
        options = {
            'maxiter': 500,  # 最大迭代次数
            'ftol': 1e-7,  # 目标函数的容忍度
            'disp': True  # 显示优化过程
        }
        self.iter_times = []
        self.times = []
        self.function_values = []
        self.start_time = time.time()
        # 使用minimize函数进行优化，添加callback
        result = minimize(self.objective_function, initial_guess,
                          args=(target_position,),
                          callback=self.callback_function,
                          bounds=bounds, method='Nelder-Mead', options=options)

        if result.success:
            return result.x, self.iter_times, self.times, self.function_values
        else:
            raise ValueError("Inverse kinematics calculation did not converge")

    def path_planning(self):

        joint_angles, iter_times, times, function_values = self.calculate_inverse_kinematics()
        theta1 = 0.5 * np.pi
        theta2 = -joint_angles[0]
        theta3 = joint_angles[1]
        theta4 = joint_angles[2]
        theta5 = 0
        joint_point = [theta1, theta2, theta3, theta4, theta5]
        self.Connectioner.robot_model.rotateAllAngle_2(joint_point)  # 做动作

        # 确保有足够的数据点进行滤波
        window_length = 11  # 设定的窗口长度
        if len(self.times) >= window_length and len(self.function_values) >= window_length:
            # 数据点足够，使用滤波
            smooth_times = savgol_filter(self.times, window_length, polyorder=1)
            smooth_function_values = savgol_filter(self.function_values, window_length, polyorder=1)
        else:
            # 数据点不足，不使用滤波
            smooth_times = self.times
            smooth_function_values = self.function_values

        # 确保有足够的数据点进行滤波
        # 增加窗口长度和多项式阶数以获得更平滑的曲线
        window_length = 51 if len(times) > 50 else len(times) // 2 * 2 + 1  # 选择一个较大的奇数窗口长度，但不超过数据点数量
        polyorder = 3  # 提高多项式阶数以更好地拟合数据

        # 使用滤波
        smooth_times = savgol_filter(times, window_length, polyorder) if len(times) >= window_length else times
        # 迭代次数本身是等差数列，不需要滤波
        smooth_iter_times = range(1, len(smooth_times)+1)

        #保存数据
        with open(r'D:\Dynamic_Arm_project\DDPG\逆解图\ik_results.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Time', 'Iteration', 'Function Value'])  # 修改列标题
            for i in range(len(times)):
                writer.writerow([
                    times[i],
                    iter_times[i] if i < len(iter_times) else '',
                    function_values[i] if i < len(function_values) else ''
                ])

        # 绘制图表
        plt.figure(figsize=(12, 6))

        # 绘制迭代次数与时间的关系
        plt.subplot(1, 2, 1)
        plt.plot(smooth_times, smooth_iter_times, marker='o', color='blue', label='Iteration')
        # 手动设置刻度的位置和标签
        tick_positions = np.linspace(min(smooth_times), max(smooth_times), 6)  # 假设你希望有6个刻度
        tick_labels = ['{:.1f}'.format(label) for label in np.linspace(0, 1.2, 6)]  # 从0.2到1.2的刻度标签
        plt.xticks(tick_positions, tick_labels)
        plt.xlabel('time (s)', fontsize=23, labelpad=6)
        plt.ylabel('Iteration Count', fontsize=23,  labelpad=6)
        # plt.title('SLSQP', fontsize=18)
        plt.grid(True)
        plt.legend(loc='upper left', fontsize=22)
        plt.tick_params(axis='both', labelsize=20, pad=8)

        # 绘制迭代次数与目标函数值的关系
        plt.subplot(1, 2, 2)
        min_length = min(len(self.iter_times), len(smooth_function_values))
        plt.plot(self.iter_times[:min_length], smooth_function_values[:min_length], marker='x', color='orange',
                 label='error')
        plt.xlabel('Iteration Count', fontsize=23, labelpad=6)
        plt.ylabel('objective function value', fontsize=23, labelpad=6)
        # plt.title('SLSQP', fontsize=18)
        plt.grid(True)
        plt.legend(loc='upper left', fontsize=22)
        plt.tick_params(axis='both', labelsize=20, pad=8)

        # # 添加整体图表的标题
        # # 在plt.figure()和plt.show()之间的适当位置添加此行
        # plt.suptitle('SLSQP', fontsize=25, x=0.535, y=0.93)  # 调整x和y值来移动标题位置

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整子图布局，为整体标题留出空间
        plt.show()

        return joint_angles


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
