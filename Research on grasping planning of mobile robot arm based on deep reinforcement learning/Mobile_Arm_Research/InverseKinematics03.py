# -*- utf-8 -*-
import numpy as np
from scipy.optimize import minimize
import sys
sys.path.append('../VREP_RemoteAPIs')
sys.path.append('../')
from connect_collpeliasim import Connection
import VREP_RemoteAPIs.sim as vrep_sim

class InverseKinematics_planning():
    def __init__(self, L1, L2, L3, L4, base_height, offset_z, joint2_range, joint3_range, joint4_range, Connectioner):
        # 初始化参数
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

        # 增量逼近法的参数
        self.max_iterations = 100
        self.tolerance = 1e-6

    def random_target(self):
        self.Connectioner.robot_model.Target_random_wall()

    def forward_kinematics(self, angles):
        theta2, theta3, theta4 = angles
        y = (self.L2 * np.sin(theta2) + self.L3 * np.sin(theta2 + theta3) + self.L4 * np.sin(theta2 + theta3 + theta4))
        z = self.base_height + self.L1 + self.offset_z + (self.L2 * np.cos(theta2) + self.L3 * np.cos(theta2 + theta3) + self.L4 * np.cos(theta2 + theta3 + theta4))
        return np.array([y, z])

    def objective_function(self, angles, target_position):
        y, z = self.forward_kinematics(angles)
        return np.linalg.norm([y - target_position[0], z - target_position[1]])

    def calculate_inverse_kinematics(self):
        # 获取目标位置
        _, target_position_3d = vrep_sim.simxGetObjectPosition(
            self.Connectioner.client_ID,
            self.Connectioner.robot_model.target,
            -1,
            vrep_sim.simx_opmode_blocking)
        # 只取y和z坐标
        target_position = np.array([target_position_3d[1], target_position_3d[2]])
        # 初始猜测和界限
        initial_guess = np.array([(bound[0] + bound[1]) / 2 for bound in [self.joint2_range, self.joint3_range, self.joint4_range]])
        bounds = [self.joint2_range, self.joint3_range, self.joint4_range]

        # 使用优化算法获得初步解
        result = minimize(self.objective_function, initial_guess, args=(target_position,), bounds=bounds, method='SLSQP', options={'maxiter': 100, 'ftol': 1e-6, 'disp': False})
        if result.success:
            return self.incremental_refinement(result.x, target_position)
        else:
            raise ValueError("Inverse kinematics calculation did not converge")

    def incremental_refinement(self, initial_angles, target_position):
        joint_angles = np.copy(initial_angles)
        for _ in range(self.max_iterations):
            current_position = self.forward_kinematics(joint_angles)
            error = target_position - current_position

            if np.linalg.norm(error) < self.tolerance:
                break

            adjustment = self.calculate_adjustment(error, joint_angles)
            joint_angles = self.apply_joint_limits(joint_angles + adjustment)

        return joint_angles

    def calculate_adjustment(self, error, joint_angles):
        adjustment = np.zeros_like(joint_angles)

        # 调整第二关节 (theta2)
        # 假设y轴误差较小时，关节2需要较小的调整
        adjustment[0] = error[0] * 0.01

        # 联合调整第三关节 (theta3) 和第四关节 (theta4)
        # 假设z轴误差较小时，这两个关节需要较小的联合调整
        z_adjustment = error[1] * 0.01
        adjustment[1] = z_adjustment
        adjustment[2] = -z_adjustment  # 可以选择不同的策略进行调整
        return adjustment

    def apply_joint_limits(self, joint_angles):
        # 确保关节角度保持在旋转范围内
        joint_angles[0] = np.clip(joint_angles[0], self.joint2_range[0], self.joint2_range[1])
        joint_angles[1] = np.clip(joint_angles[1], self.joint3_range[0], self.joint3_range[1])
        joint_angles[2] = np.clip(joint_angles[2], self.joint4_range[0], self.joint4_range[1])
        return joint_angles
    def path_planning(self):
        joint_angles = self.calculate_inverse_kinematics()
        theta1 = 0.5 * np.pi
        joint_point = [theta1, -joint_angles[0], joint_angles[1], joint_angles[2], 0]
        self.Connectioner.robot_model.rotateAllAngle_2(joint_point)
        return joint_angles

if __name__ == "__main__":
    Connectioner = Connection()
    Connectioner.Connect_verp()
    ik_planning = InverseKinematics_planning(L1=0.01351793110370636, L2=0.10496972501277924, L3=0.09500536322593689, L4=0.17,
                                                            base_height=0.15455463528633118,
                                                            offset_z=0.027126595377922058, joint2_range=(-np.pi/2, np.pi/2),
                                                            joint3_range=(-0.25 * np.pi, 0.75 * np.pi),joint4_range=(-0.25 * np.pi,  0.5*np.pi),
                                                            Connectioner=Connectioner)
    ik_planning.random_target()
    joint_angles = ik_planning.path_planning()
    print("Calculated joint angles:", joint_angles)
