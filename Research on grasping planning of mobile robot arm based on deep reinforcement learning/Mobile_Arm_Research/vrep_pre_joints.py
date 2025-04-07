import numpy as np
import math


class Range_eare:
    def __init__(self, angle_left=None, angle_right=None, dis_down=None, dis_top=None, phi_down=None,
                 phi_top=None) -> None:
        self.angle_left = angle_left
        self.angle_right = angle_right
        self.dis_down = dis_down
        self.dis_top = dis_top
        self.phi_top = phi_top
        self.phi_down = phi_down


class Pro_Pose:
    def __init__(self, dis_1, dis_2, agl_1, agl_2, agl_3, agl_4, phi_1, phi_2,phi_3) -> None:
        self.pro_pose = []
        pos_1 = [-22, 14, 84.02, 82.02, 0]  # 1 区域基准姿态
        pos_2 = [-1.3, 14, 84.02, 82.02, 0]  # 2 区域基准姿态
        pos_3 = [17.5, 14, 84.02, 82.02, 0]  # 3 区域基准姿态
        pos_4 = [-22, -43, 64.02, 57.01, 0]  # 4 区域基准姿态
        pos_5 = [-1.3, -43, 64.02, 57.01, 0]  # 5 区域基准姿态
        pos_6 = [17.5, -43, 64.02, 57.01, 0]  # 6 区域基准姿态

        self.pro_pose.append(self.transform_angle(pos_1))
        self.pro_pose.append(self.transform_angle(pos_2))
        self.pro_pose.append(self.transform_angle(pos_3))
        self.pro_pose.append(self.transform_angle(pos_4))
        self.pro_pose.append(self.transform_angle(pos_5))
        self.pro_pose.append(self.transform_angle(pos_6))
        self.dis_1 = dis_1
        self.dis_2 = dis_2
        self.angle_1 = (agl_1 / 180) * np.pi
        self.angle_2 = (agl_2 / 180) * np.pi
        self.angle_3 = (agl_3 / 180) * np.pi
        self.angle_4 = (agl_4 / 180) * np.pi
        self.phi_1 = (phi_1 / 180) * np.pi
        self.phi_2 = (phi_2 / 180) * np.pi
        self.phi_3 = (phi_3 / 180) * np.pi

    def transform_angle(self, pos):
        new_pos = []
        for angle in pos:
            angle = (angle / 180) * np.pi
            new_pos.append(angle)
        return new_pos


    def to_polar(self, pose):
        x, y, z = pose[0], pose[1], pose[2]
        dis = math.sqrt(x ** 2 + y ** 2 + z ** 2)
        if dis == 0:
            dis = 0.17
            angle = 0
            phi = 0.6
        else:
            angle = math.atan2(y, x)
            phi = math.acos(z / dis)
        return dis, angle, phi
    # 参数1可为target或end， 参数2 为空表示检查目标位置区域，参数2不为空表示检测末端所在区域
    def check_eare(self, pose):
        dis, angle, phi = self.to_polar(pose)
        if dis>=self.dis_1 and dis<=self.dis_2:
            if phi > self.phi_1 and phi <= self.phi_2:
                if angle >= self.angle_1 and angle < self.angle_2:
                    return 1
                elif angle >= self.angle_2 and angle <= self.angle_3:
                    return 2
                elif angle > self.angle_3 and angle <= self.angle_4:
                    return 3
            elif phi > self.phi_2 and phi <= self.phi_3:
                if angle >= self.angle_1 and angle < self.angle_2:
                    return 4
                elif angle >= self.angle_2 and angle <= self.angle_3:
                    return 5
                elif angle > self.angle_3 and angle <= self.angle_4:
                    return 6
        return 0

    # 根据区域编号查询基准姿态
    def find_base_pose(self, eare_id):
        if   eare_id == 1:
            return self.pro_pose[0]
        elif eare_id == 2:
            return self.pro_pose[1]
        elif eare_id == 3:
            return self.pro_pose[2]
        elif eare_id == 4:
            return self.pro_pose[3]
        elif eare_id == 5:
            return self.pro_pose[4]
        elif eare_id == 6:
            return self.pro_pose[5]
        return None

    # 根据区域查询边界
    def find_range(self, eare_id):
        eare_range = Range_eare()
        if eare_id == 1:
            eare_range.angle_left = self.angle_1
            eare_range.angle_right = self.angle_2
            eare_range.dis_down = self.dis_1
            eare_range.dis_top = self.dis_2
            eare_range.phi_down = self.phi_1
            eare_range.phi_top = self.phi_2
        elif eare_id == 2:
            eare_range.angle_left = self.angle_2
            eare_range.angle_right = self.angle_3
            eare_range.dis_down = self.dis_1
            eare_range.dis_top = self.dis_2
            eare_range.phi_down = self.phi_1
            eare_range.phi_top = self.phi_2
        elif eare_id == 3:
            eare_range.angle_left = self.angle_3
            eare_range.angle_right = self.angle_4
            eare_range.dis_down = self.dis_1
            eare_range.dis_top = self.dis_2
            eare_range.phi_down = self.phi_1
            eare_range.phi_top = self.phi_2
        elif eare_id == 4:
            eare_range.angle_left = self.angle_1
            eare_range.angle_right = self.angle_2
            eare_range.dis_down = self.dis_1
            eare_range.dis_top = self.dis_2
            eare_range.phi_down = self.phi_2
            eare_range.phi_top = self.phi_3
        elif eare_id == 5:
            eare_range.angle_left = self.angle_2
            eare_range.angle_right = self.angle_3
            eare_range.dis_down = self.dis_1
            eare_range.dis_top = self.dis_2
            eare_range.phi_down = self.phi_2
            eare_range.phi_top = self.phi_3
        elif eare_id == 6:
            eare_range.angle_left = self.angle_3
            eare_range.angle_right = self.angle_4
            eare_range.dis_down = self.dis_1
            eare_range.dis_top = self.dis_2
            eare_range.phi_down = self.phi_2
            eare_range.phi_top = self.phi_3
        elif eare_id == 0:  # 大范围
            eare_range.angle_left = self.angle_1
            eare_range.angle_right = self.angle_4
            eare_range.dis_down = self.dis_1
            eare_range.dis_top = self.dis_2
            eare_range.phi_down = self.phi_1
            eare_range.phi_top = self.phi_3
        return eare_range


if __name__ == "__main__":
    tmp = Pro_Pose(0.12, 0.3, -30, -10, 10, 30, 30, 60,90)
    target = [0.28 - 0.052569, -0.09]
    id = tmp.check_eare(target)
    print(id)
    print(tmp.find_base_pose(id))
