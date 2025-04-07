# -*- utf-8 -*-
import os
from turtle import right
import cv2
import math
import time
import random
import string
import pygame  # 用于使用键盘控制机械臂运动
import numpy as np
import sys

sys.path.append('./VREP_RemoteAPIs')
import VREP_RemoteAPIs.sim as vrep_sim
from load_data import Sample_data


class Mobile_Arm_SimModel():
    def __init__(self, name="Xtark") -> None:
        super(self.__class__, self).__init__()
        self.target_pose = None
        self.name = name
        self.client_ID = None

        self.resolutionX = 640  # 摄像头画面长
        self.resolutionY = 480  # 摄像头画面宽
        self.rad2deg = 180 / math.pi  # 弧度与角度转换
        self.jointNum = 5

        self.model_arm_joints = [0, 0, 0, 0, 0]  # 机械臂的轴（适用于设置）
        self.arm_current_joints = np.zeros(self.jointNum)  # 存储机械臂当前的角度
        self.arm_current_joints_red = np.zeros(self.jointNum, dtype=float)  # 保存弧度值
        self.jointHandle = np.zeros(self.jointNum, dtype=int)  # 机械臂的话柄
        self.arm_joint_name = 'arm_joint_'

        self.base_name = 'base_link_respondable'
        self.base_handle = None
        self.camera_rgb_name = 'kinect_rgb'
        self.camera_rgb_handle = None
        self.camera_depth_name = 'kinect_depth'
        self.camera_depth_handle = None

        self.base_wheel_speed = np.zeros(4)  # 四个轮子的速度，固定前两个为left前后， 后两个为right前后
        self.base_wheel_handle = np.zeros(4, dtype=int)  # 保存轮子的话柄

        self.base_left_front_joint = "left_front_joint"  #
        self.base_left_back_joint = "left_back_joint"
        self.base_right_front_joint = "right_front_joint"
        self.base_right_back_joint = "right_back_joint"

        self.target = None  # 目标话柄
        self.init_target_position = None

        self.linear_max_speed = 5.0
        self.angle_max_speed = 4.0
        self.base_l = 0.35  # 机器人车身距离
        self.linear_speed = 1.0
        self.angle_speed = 1.0

        self.base_link_handle = None  # 机器人底座
        self.arm_end_handle = None  # 机械臂末端执行机构
        self.base_link_position = None
        self.base_link_orientation = None
        self.arm_link_1_handle = None  # 机械臂底座
        self.arm_joint_3 = None
        self.arm_joint_4 = None
        self.arm_joint_1 = None
        self.base_frame = None  # 机械臂基座

        self.mobile_frame = None  # 移动参考系
        self.Cuboid_left_center = None  # 目标在左边随机位置的中心位置
        self.Cuboid_right_center = None  # 目标在右边随机位置的中心位置
        self.Cuboid_left_center_position = None
        self.Cuboid_right_center_position = None

        self.robot_current_speed = None  # 记录机器人当前的移动速度
        self.obstacle_handle_1 = None  # 障碍物话柄
        self.obstacle_handle_2 = None  # 障碍物话柄

        # self.Sample_test = Sample_data() # 测试

        self.arm_link_2_visual = None
        self.arm_link_3_visual = None
        self.arm_link_4_visual = None

    # 初始化仿真模型
    def init_SimMode(self, client_ID):
        self.client_ID = client_ID
        try:
            print('connectde to remote API server')
            client_ID != -1
        except:
            print('Failed connecting to remote API server')

        # print("clientID :{}".format(self.client_ID))

        # get arm joint handle
        for i in range(self.jointNum):
            return_code, self.jointHandle[i] = vrep_sim.simxGetObjectHandle(client_ID, self.arm_joint_name + str(i + 1),
                                                                            vrep_sim.simx_opmode_blocking)
            if (return_code == vrep_sim.simx_return_ok):
                print("{} is ok".format(self.arm_joint_name + str(i + 1)))

        return_code, self.base_handle = vrep_sim.simxGetObjectHandle(client_ID, self.base_name,
                                                                     vrep_sim.simx_opmode_blocking)
        if (return_code == vrep_sim.simx_return_ok):
            print("{} is ok".format(self.base_name))

        return_code, self.camera_rgb_handle = vrep_sim.simxGetObjectHandle(client_ID, self.camera_rgb_name,
                                                                           vrep_sim.simx_opmode_blocking)
        if (return_code == vrep_sim.simx_return_ok):
            print("{} is ok".format(self.camera_rgb_name))
        else:
            print("{} is failed".format(self.camera_rgb_name))

        return_code, self.camera_depth_handle = vrep_sim.simxGetObjectHandle(client_ID, self.camera_depth_name,
                                                                             vrep_sim.simx_opmode_blocking)
        if (return_code == vrep_sim.simx_return_ok):
            print("{} is ok".format(self.camera_depth_name))
        else:
            print("{} is failed".format(self.camera_depth_name))

        # get base wheel handle
        return_code, self.base_wheel_handle[0] = vrep_sim.simxGetObjectHandle(client_ID, self.base_left_front_joint,
                                                                              vrep_sim.simx_opmode_blocking)
        if (return_code == vrep_sim.simx_return_ok):
            print("{} is ok".format(self.base_left_front_joint))
        else:
            print("{} is failed".format(self.base_left_front_joint))

        return_code, self.base_wheel_handle[1] = vrep_sim.simxGetObjectHandle(client_ID, self.base_left_back_joint,
                                                                              vrep_sim.simx_opmode_blocking)
        if (return_code == vrep_sim.simx_return_ok):
            print("{} is ok".format(self.base_left_back_joint))
        else:
            print("{} is failed".format(self.base_left_back_joint))

        return_code, self.base_wheel_handle[2] = vrep_sim.simxGetObjectHandle(client_ID, self.base_right_front_joint,
                                                                              vrep_sim.simx_opmode_blocking)
        if (return_code == vrep_sim.simx_return_ok):
            print("{} is ok".format(self.base_right_front_joint))
        else:
            print("{} is failed".format(self.base_right_front_joint))

        return_code, self.base_wheel_handle[3] = vrep_sim.simxGetObjectHandle(client_ID, self.base_right_back_joint,
                                                                              vrep_sim.simx_opmode_blocking)
        if (return_code == vrep_sim.simx_return_ok):
            print("{} is ok".format(self.base_right_back_joint))
        else:
            print("{} is failed".format(self.base_right_back_joint))

        return_code, self.target = vrep_sim.simxGetObjectHandle(client_ID, "Target_Sphere",
                                                                vrep_sim.simx_opmode_blocking)
        if (return_code == vrep_sim.simx_return_ok):
            print("Target_Sphere is ok")
        else:
            print("Target_Sphere is failed")

        # base_link_handle
        return_code, self.base_link_handle = vrep_sim.simxGetObjectHandle(client_ID, "base_link_respondable",
                                                                          vrep_sim.simx_opmode_blocking)
        if (return_code == vrep_sim.simx_return_ok):
            print("base_link_respondable is ok")
        else:
            print("base_link_respondable is failed")
        # arm_end_handle
        return_code, self.arm_end_handle = vrep_sim.simxGetObjectHandle(client_ID, "goal_visual",
                                                                        vrep_sim.simx_opmode_blocking)
        if (return_code == vrep_sim.simx_return_ok):
            print("goal_visual is ok")
        else:
            print("goal_visual is failed")

        return_code, self.arm_link_1_handle = vrep_sim.simxGetObjectHandle(client_ID, "arm_link_1_respondable",
                                                                           vrep_sim.simx_opmode_blocking)
        if (return_code == vrep_sim.simx_return_ok):
            print("arm_link_1_respondable is ok")
        else:
            print("arm_link_1_respondable is failed")

        return_code, self.arm_joint_1 = vrep_sim.simxGetObjectHandle(client_ID, "arm_joint_1",
                                                                     vrep_sim.simx_opmode_blocking)
        if (return_code == vrep_sim.simx_return_ok):
            print("arm_joint_1 is ok")
        else:
            print("arm_joint_1 is failed")

        return_code, self.arm_joint_2 = vrep_sim.simxGetObjectHandle(client_ID, "arm_joint_2",
                                                                     vrep_sim.simx_opmode_blocking)
        if (return_code == vrep_sim.simx_return_ok):
            print("arm_joint_2 is ok")
        else:
            print("arm_joint_2 is failed")

        return_code, self.arm_joint_3 = vrep_sim.simxGetObjectHandle(client_ID, "arm_joint_3",
                                                                     vrep_sim.simx_opmode_blocking)
        if (return_code == vrep_sim.simx_return_ok):
            print("arm_joint_3 is ok")
        else:
            print("arm_joint_3 is failed")

        return_code, self.arm_joint_4 = vrep_sim.simxGetObjectHandle(client_ID, "arm_joint_4",
                                                                     vrep_sim.simx_opmode_blocking)
        if (return_code == vrep_sim.simx_return_ok):
            print("arm_joint_4 is ok")
        else:
            print("arm_joint_4 is failed")

            # 移动参考系
        return_code, self.mobile_frame = vrep_sim.simxGetObjectHandle(client_ID, "mobile_frame",
                                                                      vrep_sim.simx_opmode_blocking)
        if (return_code == vrep_sim.simx_return_ok):
            print("mobile_frame is ok")
        else:
            print("mobile_frame is failed")

        # 机械臂坐标中心
        return_code, self.base_frame = vrep_sim.simxGetObjectHandle(client_ID, "base_frame",
                                                                    vrep_sim.simx_opmode_blocking)
        if (return_code == vrep_sim.simx_return_ok):
            print("base_frame is ok")
        else:
            print("base_frame is failed")

        # 目标左边随机位置范围中心
        return_code, self.Cuboid_left_center = vrep_sim.simxGetObjectHandle(client_ID, "Cuboid_left",
                                                                            vrep_sim.simx_opmode_blocking)
        if (return_code == vrep_sim.simx_return_ok):
            print('Cuboid_left is ok')
        else:
            print('Cuboid_left is failed')

            # 目标右边随机位置范围中心
        return_code, self.Cuboid_right_center = vrep_sim.simxGetObjectHandle(client_ID, "Cuboid_right",
                                                                             vrep_sim.simx_opmode_blocking)
        if (return_code == vrep_sim.simx_return_ok):
            print('Cuboid_right is ok')
        else:
            print('Cuboid_right is failed')

        return_code, self.obstacle_handle_1 = vrep_sim.simxGetObjectHandle(client_ID, "Sphere_1",
                                                                         vrep_sim.simx_opmode_blocking)
        if (return_code == vrep_sim.simx_return_ok):
            print('Sphere_1 obstacle is ok')
        else:
            print('Sphere_1 obstacle is failed')

        return_code, self.obstacle_handle_2 = vrep_sim.simxGetObjectHandle(client_ID, "Sphere_2",
                                                                         vrep_sim.simx_opmode_blocking)
        if (return_code == vrep_sim.simx_return_ok):
            print('Sphere_2 obstacle is ok')
        else:
            print('Sphere_2 obstacle is failed')

        # 新增visual
        # return_code, self.arm_link_2_visual = vrep_sim.simxGetObjectHandle(client_ID, "arm_link_2_visual",
        #                                                                      vrep_sim.simx_opmode_blocking)
        # if (return_code == vrep_sim.simx_return_ok):
        #     print('arm_link_2_visual is ok')
        # else:
        #     print('arm_link_2_visual is failed')
        #
        # return_code, self.arm_link_3_visual = vrep_sim.simxGetObjectHandle(client_ID, "arm_link_3_visual",
        #                                                                           vrep_sim.simx_opmode_blocking)
        # if (return_code == vrep_sim.simx_return_ok):
        #     print('arm_link_3_visual is ok')
        # else:
        #     print('arm_link_3_visual is failed')
        #
        # return_code, self.arm_link_4_visual = vrep_sim.simxGetObjectHandle(client_ID, "arm_link_4_visual",
        #                                                                           vrep_sim.simx_opmode_blocking)
        # if (return_code == vrep_sim.simx_return_ok):
        #     print('arm_link_4_visual is ok')
        # else:
        #     print('arm_link_4_visual is failed')

        _, self.base_link_position = vrep_sim.simxGetObjectPosition(client_ID, self.base_link_handle, -1,
                                                                    vrep_sim.simx_opmode_blocking)
        _, self.base_link_orientation = vrep_sim.simxGetObjectOrientation(client_ID, self.base_link_handle, -1,
                                                                          vrep_sim.simx_opmode_blocking)

        self.init_target_position = vrep_sim.simxGetObjectPosition(client_ID, self.target, -1,
                                                                   vrep_sim.simx_opmode_blocking)
        # print(self.init_target_position)

        _, self.Cuboid_left_center_position = vrep_sim.simxGetObjectPosition(client_ID, self.Cuboid_left_center, -1,
                                                                             vrep_sim.simx_opmode_blocking)
        _, self.Cuboid_right_center_position = vrep_sim.simxGetObjectPosition(client_ID, self.Cuboid_right_center, -1,
                                                                              vrep_sim.simx_opmode_blocking)
        _, arm_joint_1_orientation = vrep_sim.simxGetObjectOrientation(self.client_ID,
                                                             self.arm_joint_1,
                                                             self.base_frame,
                                                             vrep_sim.simx_opmode_blocking)
        _, self.target_pose = vrep_sim.simxGetObjectPosition(
            self.client_ID,
            self.target,
            self.base_frame,
            vrep_sim.simx_opmode_blocking
        )


        # # get arm joint position
        for i in range(self.jointNum):
            _, joins_pos = vrep_sim.simxGetJointPosition(client_ID, self.jointHandle[i], vrep_sim.simx_opmode_blocking)
            self.arm_current_joints[i] = round(float(joins_pos) * self.rad2deg, 2)
            # self.model_arm_joints[i] = joins_pos
        # get arm joint position red
        for i in range(self.jointNum):
            _, joins_pos = vrep_sim.simxGetJointPosition(client_ID, self.jointHandle[i], vrep_sim.simx_opmode_blocking)
            self.arm_current_joints_red[i] = joins_pos


    def get_current_joint(self):
        # get arm joint position
        for i in range(self.jointNum):
            _, joins_pos = vrep_sim.simxGetJointPosition(self.client_ID, self.jointHandle[i],
                                                         vrep_sim.simx_opmode_blocking)
            self.arm_current_joints[i] = round(float(joins_pos) * self.rad2deg, 2)
            # self.model_arm_joints[i] = joins_pos

    def get_current_joint_red(self):
        # get arm joint position red
        for i in range(self.jointNum):
            _, joins_pos = vrep_sim.simxGetJointPosition(self.client_ID, self.jointHandle[i],
                                                         vrep_sim.simx_opmode_blocking)
            self.arm_current_joints_red[i] = joins_pos

    # show current joints' value
    def showJointArmAngles(self):
        for i in range(self.jointNum):
            _, joins_pos = vrep_sim.simxGetJointPosition(self.client_ID, self.jointHandle[i],
                                                         vrep_sim.simx_opmode_blocking)
            print(round(float(joins_pos) * self.rad2deg, 2), end=" ")
        print('\n')

    # get RGB images
    def getImageRGB(self):
        clientID = self.client_ID
        cameraRGBHandle = self.camera_rgb_handle
        resolutionX = self.resolutionX
        resolutionY = self.resolutionY

        res1, resolution1, image_rgb = vrep_sim.simxGetVisionSensorImage(clientID, cameraRGBHandle, 0,
                                                                         vrep_sim.simx_opmode_blocking)
        # print(image_rgb)

        image_rgb_r = [image_rgb[i] for i in range(0, len(image_rgb), 3)]
        image_rgb_r = np.array(image_rgb_r)
        image_rgb_r = image_rgb_r.reshape(resolutionY, resolutionX)
        image_rgb_r = image_rgb_r.astype(np.uint8)

        image_rgb_g = [image_rgb[i] for i in range(1, len(image_rgb), 3)]
        image_rgb_g = np.array(image_rgb_g)
        image_rgb_g = image_rgb_g.reshape(resolutionY, resolutionX)
        image_rgb_g = image_rgb_g.astype(np.uint8)

        image_rgb_b = [image_rgb[i] for i in range(2, len(image_rgb), 3)]
        image_rgb_b = np.array(image_rgb_b)
        image_rgb_b = image_rgb_b.reshape(resolutionY, resolutionX)
        image_rgb_b = image_rgb_b.astype(np.uint8)

        result_rgb = cv2.merge([image_rgb_b, image_rgb_g, image_rgb_r])
        # 镜像翻转， opencv 这里返回的是一张翻转的图
        result_rgb = cv2.flip(result_rgb, 0)
        return result_rgb

    # get depth images
    def getImageDepth(self):
        clientID = self.client_ID
        cameraDepthHandle = self.camera_depth_handle
        resolutionX = self.resolutionX
        resolutionY = self.resolutionY

        res2, resolution2, image_depth = vrep_sim.simxGetVisionSensorImage(clientID, cameraDepthHandle, 0,
                                                                           vrep_sim.simx_opmode_blocking)

        image_depth_r = [image_depth[i] for i in range(0, len(image_depth), 3)]
        image_depth_r = np.array(image_depth_r)
        image_depth_r = image_depth_r.reshape(resolutionY, resolutionX)
        image_depth_r = image_depth_r.astype(np.uint8)

        image_depth_g = [image_depth[i] for i in range(1, len(image_depth), 3)]
        image_depth_g = np.array(image_depth_g)
        image_depth_g = image_depth_g.reshape(resolutionY, resolutionX)
        image_depth_g = image_depth_g.astype(np.uint8)

        image_depth_b = [image_depth[i] for i in range(2, len(image_depth), 3)]
        image_depth_b = np.array(image_depth_b)
        image_depth_b = image_depth_b.reshape(resolutionY, resolutionX)
        image_depth_b = image_depth_b.astype(np.uint8)

        result_depth = cv2.merge([image_depth_b, image_depth_g, image_depth_r])
        # 镜像翻转， opencv 在这里返回的是一张的翻转的图
        result_depth = cv2.flip(result_depth, 0)

        # 黑白取反
        height, width, channels = result_depth.shape
        for row in range(height):
            for list in range(width):
                for c in range(channels):
                    pv = result_depth[row, list, c]
                    result_depth[row, list, c] = 255 - pv

        return result_depth

    # open gripper
    def open_gripper(self):
        pass

    # close gripper
    def close_gripper(self):
        pass

    # 设置目标的随机位置
    def Target_random(self):
        y_direction = np.random.uniform(-0.08, 0.08)
        # y_direction = np.random.uniform(-0.1, 0.1)
        x_direction = np.random.uniform(0.22, 0.30)
        # x_direction = np.random.uniform(0.22,0.32)

        new_target_position = [0.0, 0.0, 0.0]
        new_target_position[0] = x_direction
        new_target_position[1] = y_direction
        new_target_position[2] = self.init_target_position[1][2]
        # print(new_target_position)
        # vrep_sim.simxSetObjectPosition(self.client_ID, self.target, -1, new_target_position, vrep_sim.simx_opmode_oneshot)
        vrep_sim.simxSetObjectPosition(self.client_ID, self.target, self.mobile_frame, new_target_position,
                                       vrep_sim.simx_opmode_oneshot)  # 移动参考系

    def Target_random_advance(self):
        theta = np.random.uniform(-0.25 * np.pi, 1.25 * np.pi)
        l = np.random.uniform(0, 0.1) + 0.16
        x_direction = l * np.sin(theta) + 0.054
        y_direction = -l * np.cos(theta)
        if theta > 0.25 * np.pi and theta < 0.75 * np.pi:
            z_direction = np.random.uniform(0.015, 0.25)
        else:
            z_direction = np.random.uniform(0.165, 0.25)

        new_target_position = [0.0, 0.0, 0.0]
        new_target_position[0] = x_direction
        new_target_position[1] = y_direction
        new_target_position[2] = z_direction

        vrep_sim.simxSetObjectPosition(self.client_ID, self.target, self.mobile_frame, new_target_position,
                                       vrep_sim.simx_opmode_oneshot)

    # 设置目标物在可抓取范围内的随机位置
    def Target_random_wall(self):
        x_direction = self.Cuboid_left_center_position[0] + np.random.uniform(-0.15, 0.15)
        y_direction = self.Cuboid_left_center_position[1] + np.random.uniform(-0.02, 0.03)
        z_direction = np.random.uniform(0.22, 0.42)
        # if z_direction > 0.175 and z_direction < 0.24:
        #     z_direction = 0.175
        # if z_direction >= 0.24 and z_direction < 0.265:
        #     z_direction = 0.265
        new_target_position = [0.0, 0.0, 0.0]
        new_target_position[0] = x_direction
        new_target_position[1] = y_direction
        new_target_position[2] = z_direction
        vrep_sim.simxSetObjectPosition(self.client_ID, self.target, -1, new_target_position,
                                       vrep_sim.simx_opmode_oneshot)

    # 障碍物的随机位置
    def Obstacle_random(self):
        x_direction = 0.5
        y_direction = np.random.uniform(0.125, 0)
        z_direction = np.random.uniform(0.27, 0.37)
        new_target_position = [0.0, 0.0, 0.0]
        new_target_position[0] = x_direction
        new_target_position[1] = y_direction
        new_target_position[2] = z_direction
        vrep_sim.simxSetObjectPosition(self.client_ID, self.obstacle_handle, -1, new_target_position,
                                       vrep_sim.simx_opmode_oneshot)

    def Obstacle_random_2(self):
        x_direction = 0.5
        y_direction = np.random.uniform(0.12, 0)  # 0
        z_direction = np.random.uniform(0.25, 0.31)
        new_target_position = [0.0, 0.0, 0.0]
        new_target_position[0] = x_direction
        new_target_position[1] = y_direction
        new_target_position[2] = z_direction
        vrep_sim.simxSetObjectPosition(self.client_ID, self.obstacle_handle, -1, new_target_position,
                                       vrep_sim.simx_opmode_oneshot)

    def Obstacle_random_3(self):
        x_direction = 0.5
        y_direction = np.random.uniform(0.12, -0.12)  # 0
        z_direction = np.random.uniform(0.25, 0.31)
        new_target_position = [0.0, 0.0, 0.0]
        new_target_position[0] = x_direction
        new_target_position[1] = y_direction
        new_target_position[2] = z_direction
        vrep_sim.simxSetObjectPosition(self.client_ID, self.obstacle_handle, -1, new_target_position,
                                       vrep_sim.simx_opmode_oneshot)

    def Target_random_double(self):
        x_direction = self.Cuboid_left_center_position[0] + np.random.uniform(-0.2, 0.2)
        y_direction = self.Cuboid_left_center_position[1] + np.random.uniform(-0.02, 0.03)
        z_direction = np.random.uniform(0.22, 0.42)
        new_target_position = [0.0, 0.0, 0.0]
        new_target_position[0] = x_direction
        new_target_position[1] = y_direction
        new_target_position[2] = z_direction
        vrep_sim.simxSetObjectPosition(self.client_ID, self.target, -1, new_target_position,
                                       vrep_sim.simx_opmode_oneshot)

    def Target_random_reset_vrep(self, index):
        target_pos = []
        target_pos.append([0.25, 0.055, 0.01])
        target_pos.append([0.225, 0.0, 0.01])
        target_pos.append([0.275, -0.06, 0.01])
        target_pos.append([0.265, -0.07, 0.01])
        vrep_sim.simxSetObjectPosition(self.client_ID, self.target, -1, target_pos[index], vrep_sim.simx_opmode_oneshot)

    # def Target_random_vrep(self,dis_min, dis_max):
    #     np.random.seed()
    #     target_dis = np.random.uniform(dis_min, dis_max)
    #     target_tha = np.random.uniform(-0.166*np.pi,0.166*np.pi)
    #     x = round(target_dis*np.cos(target_tha),4)
    #     y = round(target_dis*np.sin(target_tha),4)
    #     z = 0.01
    #     target_pose = [x, y, z]
    #     #print(target_pose)
    #     vrep_sim.simxSetObjectPosition(self.client_ID, self.target, self.base_frame, target_pose, vrep_sim.simx_opmode_oneshot)

    def Target_random_vrep(self, dis_min, dis_max, theta_min, theta_max, phi_min, phi_max):
        np.random.seed()
        # Generate random spherical coordinates
        r = np.random.uniform(dis_min, dis_max)
        theta = np.random.uniform(theta_min, theta_max)
        phi = np.random.uniform(phi_min, phi_max)
        # Convert spherical coordinates to Cartesian coordinates
        x = round(r * math.sin(phi) * math.cos(theta), 4)
        y = round(r * math.sin(phi) * math.sin(theta), 4)
        z = round(r * math.cos(phi), 4)
        target_pose = [x, y, z]
        vrep_sim.simxSetObjectPosition(self.client_ID, self.target, self.base_frame, target_pose,
                                       vrep_sim.simx_opmode_oneshot)

    def update_target_position(self, velocity_min, velocity_max, time_step):
        np.random.seed()
        # Generate random velocity components within the specified range
        vx = np.random.uniform(velocity_min, velocity_max)
        vy = np.random.uniform(velocity_min, velocity_max)
        vz = np.random.uniform(velocity_min, velocity_max)
        # Calculate displacement increments based on velocity and time step
        dx = vx * time_step
        dy = vy * time_step
        dz = vz * time_step
        # Update target position
        self.target_pose[0] += dx
        self.target_pose[1] += dy
        self.target_pose[2] += dz
        # Set new target position in the simulation
        vrep_sim.simxSetObjectPosition(self.client_ID, self.target, self.base_frame, self.target_pose,
                                       vrep_sim.simx_opmode_oneshot)


    def Target_random_vrep_2(self, dis_min, dis_max):
        np.random.seed()
        target_dis = np.random.uniform(dis_min, dis_max)
        target_tha = np.random.uniform(-0.166 * np.pi, 0.166 * np.pi)
        x = round(target_dis * np.cos(target_tha), 4)
        y = round(target_dis * np.sin(target_tha), 4)
        z = round(np.random.uniform(0.01, 0.3), 4)
        target_pose = [x, y, z]
        # print(target_pose)
        vrep_sim.simxSetObjectPosition(self.client_ID, self.target, self.base_frame, target_pose,
                                       vrep_sim.simx_opmode_oneshot)

    # 500轮不同场景测试
    def Set_Target(self, index):
        cup = self.Sample_test.Sample(index)
        x = round(cup[0] * np.cos(cup[1]), 4)
        y = round(cup[0] * np.sin(cup[1]), 4)
        z = 0.01
        target_pose = [x, y, z]
        vrep_sim.simxSetObjectPosition(self.client_ID, self.target, self.base_frame, target_pose,
                                       vrep_sim.simx_opmode_oneshot)

    def Set_Target_and_Obstacl(self):
        # new_target_position = [1, 0.23, 0.25]
        # new_obstacl_position = [0.5,0.0, 0.31]
        # 测试1
        # new_target_position = [0.85, 0.23, 0.25]
        # new_obstacl_position = [0.5,-0.05, 0.25]
        # 测试2
        new_target_position = [0.95, 0.18, 0.3]
        new_obstacl_position = [0.5, 0.05, 0.3]
        # # 测试3
        # new_target_position = [1.15, 0.2, 0.21]
        # new_obstacl_position = [0.5,0.01, 0.27]
        vrep_sim.simxSetObjectPosition(self.client_ID, self.target, -1, new_target_position,
                                       vrep_sim.simx_opmode_oneshot)
        vrep_sim.simxSetObjectPosition(self.client_ID, self.obstacle_handle, -1, new_obstacl_position,
                                       vrep_sim.simx_opmode_oneshot)

    # 限制各个关节的旋转角度范围
    def limit_joint_angle(self, num, angle):
        if num == 0 and (angle >= -90 and angle <= 90):
            return True
        elif num == 1 and (angle >= -90 and angle <= 90):
            return True
        elif num == 2 and (angle >= -45 and angle <= 135):
            return True
        elif num == 3 and (angle >= -45 and angle <= 90):
            return True
        # elif num == 4 and (angle >= -180 and angle <= 180):
        #     return True
        return False

        # 从初始化位置进行设置angle

    def rotateAllAngle(self, joint_angle):
        # 暂停通信， 用于存储所有控制命令一起发送
        vrep_sim.simxPauseCommunication(self.client_ID, True)
        for i in range(self.jointNum):
            vrep_sim.simxSetJointTargetPosition(self.client_ID, self.jointHandle[i],
                                                joint_angle[i] / self.rad2deg, vrep_sim.simx_opmode_oneshot)
        vrep_sim.simxPauseCommunication(self.client_ID, False)

        self.arm_current_joints = joint_angle

    # 从初始化位置进行设置angle(弧度值)
    def rotateAllAngle_2(self, joint_angle_red):
        '''
        传入弧度值
        '''
        # 暂停通信， 用于存储所有控制命令一起发送
        vrep_sim.simxPauseCommunication(self.client_ID, True)
        for i in range(self.jointNum):
            vrep_sim.simxSetJointTargetPosition(self.client_ID, self.jointHandle[i],
                                                joint_angle_red[i], vrep_sim.simx_opmode_oneshot)
        vrep_sim.simxPauseCommunication(self.client_ID, False)

        self.arm_current_joints_red = joint_angle_red

    # 设置第 num 个关节正传转 angle 度
    def rotate_Certain_Angle_Positive(self, num, angle):
        self.model_arm_joints[num] = self.arm_current_joints[num]
        # print("current joint{} is {} ".format(num,self.arm_current_joints[num]))
        if self.limit_joint_angle(num, self.model_arm_joints[num] + angle):
            vrep_sim.simxSetJointTargetPosition(self.client_ID, self.jointHandle[num],
                                                (self.model_arm_joints[num] + angle) / self.rad2deg,
                                                vrep_sim.simx_opmode_oneshot)
            self.arm_current_joints[num] = self.model_arm_joints[num] + angle
            print(self.arm_current_joints[num])
        else:
            print("over the angle range of joint_{} ".format(num))

    # 设置第 num 个关节反转 angle 度
    def rotate_Certain_Angle_Negative(self, num, angle):
        self.model_arm_joints[num] = self.arm_current_joints[num]
        if self.limit_joint_angle(num, self.model_arm_joints[num] - angle):
            vrep_sim.simxSetJointTargetPosition(self.client_ID, self.jointHandle[num],
                                                (self.model_arm_joints[num] - angle) / self.rad2deg,
                                                vrep_sim.simx_opmode_oneshot)
            self.arm_current_joints[num] = self.model_arm_joints[num] - angle
            print(self.arm_current_joints[num])
        else:
            print("over the angle range of joint_{} ".format(num))

    # 停止底盘
    def base_stop(self):
        vrep_sim.simxPauseCommunication(self.client_ID, True)
        vrep_sim.simxSetJointTargetVelocity(self.client_ID, self.base_wheel_handle[0], 0, vrep_sim.simx_opmode_oneshot)
        vrep_sim.simxSetJointTargetVelocity(self.client_ID, self.base_wheel_handle[1], 0, vrep_sim.simx_opmode_oneshot)
        vrep_sim.simxSetJointTargetVelocity(self.client_ID, self.base_wheel_handle[2], 0, vrep_sim.simx_opmode_oneshot)
        vrep_sim.simxSetJointTargetVelocity(self.client_ID, self.base_wheel_handle[3], 0, vrep_sim.simx_opmode_oneshot)
        vrep_sim.simxPauseCommunication(self.client_ID, False)
        # self.robot_current_speed = 0

    # 设置 底盘的速度
    def set_wheels_sppeds(self, linear, angle):
        # 暂停通信， 用于存储所有控制命令一起发送
        # linear = self.linear_speed + linear
        # angle = self.angle_speed + angle
        # print("the current linear is {}".format(linear))
        # print("the current angle is {}".format(angle))

        if linear >= self.linear_max_speed:
            linear = self.linear_max_speed
        if angle >= self.angle_max_speed:
            angle = self.angle_max_speed

        if linear != 0 and angle != 0:
            rand = math.fabs(linear / angle)
            left_vel = ((2 * linear - angle * self.base_l) / 2 * rand)
            right_vel = ((2 * linear + angle * self.base_l) / 2 * rand)
        else:
            left_vel = linear
            right_vel = linear

        vrep_sim.simxPauseCommunication(self.client_ID, True)

        vrep_sim.simxSetJointTargetVelocity(self.client_ID, self.base_wheel_handle[0], left_vel,
                                            vrep_sim.simx_opmode_oneshot)
        vrep_sim.simxSetJointTargetVelocity(self.client_ID, self.base_wheel_handle[1], left_vel,
                                            vrep_sim.simx_opmode_oneshot)
        vrep_sim.simxSetJointTargetVelocity(self.client_ID, self.base_wheel_handle[2], -right_vel,
                                            vrep_sim.simx_opmode_oneshot)
        vrep_sim.simxSetJointTargetVelocity(self.client_ID, self.base_wheel_handle[3], -right_vel,
                                            vrep_sim.simx_opmode_oneshot)

        vrep_sim.simxPauseCommunication(self.client_ID, False)
        # update speed
        self.linear_speed = linear
        self.angle_speed = angle
        self.robot_current_speed = linear

    # convert array from vrep to image
    def arrayToImage(self):
        path = "./imgTemp/frame.jpg"
        if os.path.exists(path):
            os.remove(path)
        ig = self.getImageRGB()
        cv2.imwrite(path, ig)

    # convert array from vrep to depth image
    def arrayToDepthImage(self):
        path = "./imgTempDep/frame.jpg"
        if os.path.exists(path):
            os.remove(path)
        ig = self.getImageDepth()
        cv2.imwrite(path, ig)

    # 键盘控制
def keyboard_control(robot_model):
    robot = robot_model

    angle = 1
    linear = 0.01
    angle_vel = 0.1

    pygame.init()
    screen = pygame.display.set_mode((robot.resolutionX, robot.resolutionY))
    screen.fill((255, 255, 255))
    pygame.display.set_caption("Coppeliasim_Mobile_Arm")

    # 循环事件， 按住一个键可持续移动
    pygame.key.set_repeat(200, 50)
    robot.get_current_joint()  # 获取当前关节
    robot.showJointArmAngles()  # 打印当前的关节状态

    while True:
        robot.arrayToImage()
        ig = pygame.image.load("./imgTemp/frame.jpg")
        screen.blit(ig, (0, 0))
        pygame.display.update()
        key_pressed = pygame.key.get_pressed()
        for event in pygame.event.get():
            # 关闭程序
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    sys.exit()
                # joinit 0
                elif event.key == pygame.K_q:
                    robot.rotate_Certain_Angle_Positive(0, angle)
                elif event.key == pygame.K_w:
                    robot.rotate_Certain_Angle_Negative(0, angle)
                # joinit 1
                elif event.key == pygame.K_a:
                    robot.rotate_Certain_Angle_Positive(1, angle)
                elif event.key == pygame.K_s:
                    robot.rotate_Certain_Angle_Negative(1, angle)
                # joinit 2
                elif event.key == pygame.K_z:
                    robot.rotate_Certain_Angle_Positive(2, angle)
                elif event.key == pygame.K_x:
                    robot.rotate_Certain_Angle_Negative(2, angle)
                # joinit 3
                elif event.key == pygame.K_e:
                    robot.rotate_Certain_Angle_Positive(3, angle)
                elif event.key == pygame.K_r:
                    robot.rotate_Certain_Angle_Negative(3, angle)
                # joinit 4
                elif event.key == pygame.K_d:
                    robot.rotate_Certain_Angle_Positive(4, angle)
                elif event.key == pygame.K_f:
                    robot.rotate_Certain_Angle_Negative(4, angle)
                # # joinit 5
                # elif event.key == pygame.K_c:
                #     robot.rotate_Certain_Angle_Positive(5, angle)
                # elif event.key == pygame.K_v:
                #     robot.rotate_Certain_Angle_Negative(5, angle)
                # close gripper
                elif event.key == pygame.K_i:  # 前进
                    robot.set_wheels_sppeds(0.5, 0)
                elif event.key == pygame.K_COMMA:  # 后退
                    robot.set_wheels_sppeds(-1, 0)
                elif event.key == pygame.K_j:  # 左转
                    robot.set_wheels_sppeds(1, 1)
                elif event.key == pygame.K_l:  # 右转
                    robot.set_wheels_sppeds(1, -1)
                elif event.key == pygame.K_k:  # 停止
                    robot.base_stop()

                elif event.key == pygame.K_t:
                    robot.Set_Target(10)
                # # open gripper
                elif event.key == pygame.K_y:
                    pass

                elif event.key == pygame.K_g:
                    robot.Obstacle_random_3()

                # save Images
                elif event.key == pygame.K_SPACE:
                    rgbImg = robot.getImageRGB()
                    depthImg = robot.getImageDepth()
                    # 随机生成8位ascii码和数字作为文件名
                    ran_str = ''.join(random.sample(string.ascii_letters + string.digits, 8))
                    cv2.imwrite("saveImg\\rgbImg\\" + ran_str + "_rgb.jpg", rgbImg)
                    cv2.imwrite("saveImg\\depthImg\\" + ran_str + "_depth.jpg", depthImg)
                    print("save image")
                # reset angle
                elif event.key == pygame.K_g:
                    robot.rotateAllAngle([0, 46, 136, 0, 0, 0])
                    # angle = float(eval(input("please input velocity: ")))
                else:
                    print("Invalid input, no corresponding function for this key!")
