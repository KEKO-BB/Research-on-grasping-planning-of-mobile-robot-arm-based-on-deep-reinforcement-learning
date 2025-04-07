#-*- utf-8 -*-
import sys
sys.path.append('./VREP_RemoteAPIs')
import VREP_RemoteAPIs.sim as vrep_sim
import CoppeliaSim_model

class Connection:
    def __init__(self) -> None:
        self.robot_model = None
        self.client_ID = None

    def Connect_verp(self):
        vrep_sim.simxFinish(-1) # 关闭所以打开的连接
        while True:
            client_ID = vrep_sim.simxStart('127.0.0.1', 19997, True, False,5000,5) # 连接Coppeliasim
            if client_ID > -1:
                print('connect coppeliaSim successfully !')
                break
            else:
                print('Failed connecting to remote API server! Try it again...')

        self.client_ID = client_ID 
        # 打开同步模式
        vrep_sim.simxSynchronous(clientID=client_ID,enable=False)
        vrep_sim.simxStartSimulation(clientID=client_ID,operationMode=vrep_sim.simx_opmode_oneshot)
        #vrep_sim.simxSynchronousTrigger(clientID=client_ID)
        # 初始化机器人模型
        self.robot_model = CoppeliaSim_model.Mobile_Arm_SimModel()
        self.robot_model.init_SimMode(client_ID=client_ID)
        #vrep_sim.simxSynchronousTrigger(client_ID)

if __name__ =="__main__":
    Connectioner = Connection() # 实例化一个连接对象
    Connectioner.Connect_verp() # 连接coppeliasim并初始化机器人模型
    CoppeliaSim_model.keyboard_control(Connectioner.robot_model) # 键盘控制测试


