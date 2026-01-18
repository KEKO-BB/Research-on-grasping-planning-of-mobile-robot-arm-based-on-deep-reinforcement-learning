# Research-on-grasping-planning-of-mobile-robot-arm-based-on-deep-reinforcement-learning
Project Overview
This project is based on Deep Reinforcement Learning (DRL) methods, combined with the improved TD3 algorithm (including TD3, TD3(GRU), TD3(LSTM), and TD3(GRU)_PER versions), to implement dynamic obstacle avoidance and target grasping planning for a mobile robotic arm in complex environments. By introducing HQP, COQNLS, GRU (Gated Recurrent Unit), LSTM (Long Short-Term Memory), and Prioritized Experience Replay (PER) mechanisms, the performance and convergence efficiency of the model are significantly enhanced.

Project Structure
The project code is organized into the following six major components:
1. Environment Code
These files simulate the robot's movement and task execution in a dynamic environment, handling interactions between the robot, obstacles, and target objects.
Files: Dynamic_env_1.py to Dynamic_env_7.py, with the final training environment being Dynamic_env_6_double_obstacle_关键预测.py.
Description: These environment files define different versions of the environment model, ranging from Dynamic_env_1.py to Dynamic_env_7.py. Each version includes dynamic obstacles and target grasping tasks. The environment used in the final experiments is Dynamic_env_6_double_obstacle_关键预测.py.

2. Algorithm Code
This section contains the deep reinforcement learning algorithms used to train the robot, covering various versions of the TD3 algorithm.
Files: TD3.py, TD3(GRU).py, TD3(LSTM).py, TD3(GRU)_PER.py
Description: These files implement different versions of the TD3 algorithm:
TD3: The classic TD3 algorithm based on the Actor-Critic model, using two Critic networks to improve stability.
TD3(GRU): Integrates GRU (Gated Recurrent Unit) into TD3 to handle sequential decision-making, especially suitable for dynamic environments.
TD3(LSTM): Integrates LSTM (Long Short-Term Memory) into TD3 to compare the dynamic obstacle avoidance ability of TD3(GRU) and enhance the model's capacity to handle long-term dependencies.
TD3(GRU)_PER: Enhances TD3(GRU) by incorporating Prioritized Experience Replay (PER), improving sample efficiency.

3. Training Code
This code is responsible for managing and executing the training process, including different versions of the TD3 algorithm.
Files: TD3_train.py, TD3(GRU)_train.py, TD3(LSTM)_train.py, TD3(GRU)_PER_train.py
Description: These training scripts are used for training different versions of the TD3 algorithm:
TD3_train.py: For training the standard TD3 algorithm.
TD3(GRU)_train.py: For training the TD3(GRU) algorithm.
TD3(LSTM)_train.py: For training the TD3(LSTM) algorithm.
TD3(GRU)_PER_train.py: For training the TD3(GRU)_PER algorithm.

4. Noise Code
This code enhances the exploration process during training, particularly in continuous action spaces, to avoid the policy from getting stuck in local optima.
Files: Normal_Ounoise.py
Description: This file implements OU noise (Ornstein-Uhlenbeck noise) and Gaussian noise, adding noise to the action selection process to promote exploration in the environment.

5. Inverse Kinematics Optimization Code
This code computes the inverse kinematics for the mobile robot arm, optimizing the approximate grasping posture as expert guidance, ensuring that the robot can accurately grasp the target. Key steps include:
Forward Kinematics: Given joint angles, the position of the robot's end effector is calculated.
Optimization: Using optimization algorithms to minimize target errors, calculating suitable joint angles, which serve as expert guidance for the reinforcement learning model.

6. Communication and Interaction with the Simulation Environment
This section implements communication between PyCharm and the CoppeliaSim simulation environment, enabling interaction through the VREP Remote API.
Files:
connect_collpeliasim.py: Used to establish a connection with the CoppeliaSim simulation environment, initialize the simulation, and start the robot model. It uses the VREP Remote API to start the simulation and control the robot model.
CoppeliaSim_model.py: Defines interactions with the CoppeliaSim simulation environment, including controlling robot joints, cameras, and target objects. It provides functionality for controlling the robot's movement, grabbing tasks, and more via keyboard inputs.

Installation
To get started, clone the repository to your local machine:
git clone https://github.com/yourusername/robotic-arm-research.git
cd robotic-arm-research

Install the project dependencies:
pip install -r requirements.txt

Usage
Training the Model
1.Choose the algorithm version: Select the algorithm version you wish to use and run the corresponding training script. For example, to use the TD3 algorithm, run TD3_train.py. Replace with other versions as needed.
python TD3_train.py --train_eps 5000 --test_eps 500
2.Training parameters: You can pass training hyperparameters via the command line to adjust the training process. Common parameters include:
train_eps: Number of training episodes (e.g., 5000)
test_eps: Number of testing episodes (e.g., 500)
gamma: Discount factor for reinforcement learning (default: 0.99)
batch_size: Batch size for training (default: 100)

Testing the Model
Once training is complete, use the following command to test the model:
python TD3_train.py --train_eps 0 --test_eps 500

Hyperparameter Configuration
You can set hyperparameters via the command line. Some common options include:
--train_eps    # Number of training episodes
--test_eps     # Number of testing episodes
--gamma        # Discount factor
--batch_size   # Batch size
--critic_lr    # Critic network learning rate
--actor_lr     # Actor network learning rate
--device       # Computing device to use ('cuda' or 'cpu')

Detailed Implementation
Environment Code (Dynamic_env_6_double_obstacle_关键预测.py)
This code file defines a simulation environment where the robot interacts with obstacles and a target object. The environment includes:
Target and obstacle positions.
The robot arm's control logic and state updates.
Dynamic obstacle avoidance: The robot adjusts its path based on the position and velocity of obstacles to successfully reach the target.

Algorithm Code (TD3.py, TD3(GRU).py, TD3(LSTM).py, TD3(GRU)_PER.py)
These files implement different versions of the TD3 algorithm:
TD3: The classic TD3 algorithm, based on the Actor-Critic model, with two Critic networks for improved stability.
TD3(GRU): Incorporates GRU (Gated Recurrent Unit) into TD3 for handling sequential data, particularly suitable for dynamic environments.
TD3(LSTM): Integrates LSTM (Long Short-Term Memory) into TD3, comparing the dynamic obstacle avoidance ability of TD3(GRU) and enhancing the model's handling of long-term dependencies.
TD3(GRU)_PER: Enhances TD3(GRU) by integrating Prioritized Experience Replay (PER), improving sample efficiency.

Training Code (TD3_train.py, TD3(GRU)_train.py, TD3(LSTM)_train.py, TD3(GRU)_PER_train.py)
These files manage and execute the training process for different versions of the TD3 algorithm:
Training: Includes the number of training episodes, model saving, reward and loss recording, etc.
Testing: Evaluates the model's performance during testing.

Noise Code (Normal_Ounoise.py)
This code enhances the exploration process, preventing the policy from getting stuck in local optima. It includes:
OU noise (Ornstein-Uhlenbeck noise) and Gaussian noise: Adds noise to action selection, promoting exploration in the environment.

Inverse Kinematics Optimization Code (InverseKinematics.py)
This code computes the inverse kinematics for the mobile robot arm, optimizing the approximate grasping posture as expert guidance to ensure the robot can accurately grasp the target. Key steps include:
Forward Kinematics: Given joint angles, the position of the robot's end effector is calculated.
Optimization: Using optimization algorithms to minimize target errors and calculate suitable joint angles for reinforcement learning guidance.

Communication and Interaction with the Simulation Environment (connect_collpeliasim.py, CoppeliaSim_model.py)
This section implements communication between PyCharm and the CoppeliaSim simulation environment using VREP Remote API:
connect_collpeliasim.py: Establishes a connection to the CoppeliaSim simulation environment, initializes the simulation, and starts the robot model. It uses VREP Remote API to start the simulation and control the robot.
CoppeliaSim_model.py: Defines interactions with the CoppeliaSim environment, controlling robot joints, cameras, and target objects. It provides functionality for controlling robot movement, grasping tasks, etc., via keyboard input.

Results and Evaluation
Once training is complete, you can visualize the training process (rewards, losses, etc.) using TensorBoard. Testing results will demonstrate the model's performance in real-world tasks, including obstacle avoidance and grasping success rates.
