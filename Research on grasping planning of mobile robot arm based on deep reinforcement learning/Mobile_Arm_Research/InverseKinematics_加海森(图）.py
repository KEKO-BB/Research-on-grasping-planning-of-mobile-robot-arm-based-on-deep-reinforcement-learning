# -*- utf-8 -*-
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

import matplotlib
# 设置 matplotlib 使用 Times New Roman 字体
matplotlib.rc('font', family='Times New Roman')

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
        if len(self.function_values) < len(self.iter_times):
            self.function_values.append(func_val)
        return func_val

    def calculate_inverse_kinematics(self):
        _, target_position = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.target,
            -1,
            vrep_sim.simx_opmode_blocking)
        self.target_position = np.array([target_position[1], target_position[2]])
        initial_guess = np.array(
            [(bound[0] + bound[1]) / 2 for bound in [self.joint2_range, self.joint3_range, self.joint4_range]])
        bounds = [self.joint2_range, self.joint3_range, self.joint4_range]
        options = {'maxiter': 100, 'ftol': 1e-6, 'disp': False}
        self.start_time = time.time()
        result = minimize(self.objective_function, initial_guess,
                          args=(self.target_position,), bounds=bounds,
                          method='SLSQP', options=options, callback=self.callback_function)

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
        self.Connectioner.robot_model.rotateAllAngle_2(joint_point)

        window_length = 11
        if len(times) >= window_length and len(function_values) >= window_length:
            smooth_times = savgol_filter(times, window_length, polyorder=1)
            smooth_function_values = savgol_filter(function_values, window_length, polyorder=1)
        else:
            smooth_times = times
            smooth_function_values = function_values

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(smooth_times, iter_times, marker='o', color='blue', label='Iteration')
        plt.ylabel('Iteration Count', fontsize=20, labelpad=6)
        plt.xlabel('time (s)', fontsize=20, labelpad=6)
        plt.legend(loc='upper left', fontsize=16)
        plt.grid(True)
        plt.tick_params(axis='both', labelsize=16, pad=8)

        plt.subplot(1, 2, 2)
        min_length = min(len(iter_times), len(smooth_function_values))
        plt.plot(iter_times[:min_length], smooth_function_values[:min_length], marker='x', color='orange',
                 label='Error')
        plt.xlabel('Iteration Count', fontsize=20, labelpad=6)
        plt.ylabel('Objective Function Value', fontsize=20, labelpad=6)
        plt.legend(loc='upper left', fontsize=16)
        plt.grid(True)
        plt.tick_params(axis='both', labelsize=16, pad=8)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        return theta2, theta3, theta4

if __name__ == "__main__":
    Connectioner = Connection()
    Connectioner.Connect_verp()
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
