Deep Q-Network (DQN) Agent for Blob Environment

Overview

This project implements a Deep Q-Network (DQN) agent to navigate a simple environment called the Blob Environment. The Blob Environment consists of a grid where the agent (a blob) must navigate to find food while avoiding an enemy. The agent learns to make optimal decisions through reinforcement learning using a DQN architecture.

Dependencies

Python 3.x
TensorFlow 2.x
Keras
NumPy
OpenCV
tqdm
PIL (Python Imaging Library)
Installation

Clone the repository:
git clone https://github.com/your-username/blob-dqn-agent.git

Install the dependencies:
pip install -r requirements.txt

Usage
Run the main script to train the DQN agent:
python dqn_agent.py
Adjust the hyperparameters in the script or modify the Blob environment as needed.

Monitor training progress and agent behavior using TensorBoard logs and visualization tools.

Files

dqn_agent.py: Main script containing the DQN agent implementation and training loop.
blob_environment.py: Blob environment implementation with actions, rewards, and state transitions.
modified_tensorboard.py: Custom TensorBoard class for logging training statistics.
models/: Folder to store trained DQN models.
logs/: Folder to store TensorBoard logs.
Customization

Feel free to modify the environment size, reward/penalty values, neural network architecture, hyperparameters, and training settings to suit your needs or experiment with different scenarios.

Credits

This project is inspired by the Deep Q-Network (DQN) algorithm and the Blob Environment concept, and it draws on various resources and tutorials for reinforcement learning with Keras and TensorFlow.

License

This project is licensed under the MIT License.


