# rleasy_tut
A simple tutorial to learn basic reinforcement learning.

The task of pin insertion will be learnt according to the insertion phase described in the paper on [<em>Deep Reinforcement Learning for High Precision Assembly Tasks</em>](https://arxiv.org/abs/1708.04033).

## Installation

The following installations are required:

- [Gym](https://gym.openai.com/docs/)
- [MuJoCo and mujoco_py](https://github.com/openai/mujoco-py)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/guide/install.html)
- clone the repo and install `rleasy_tut`

      git clone https://github.com/qj25/rleasy_tut.git
      cd rleasy_tut
      pip install -e .

## Usage

- To start training:

      python pandainsert_rl_sb.py learn

  A folder named `exp` will be created, containing the saved models and log files

- To evaluate the trained policies:

      python pandainsert_rl_sb.py check 1

  '1' is the experiment id to be evaluated. Experiments can be found with their IDs at *rleasy_tut/exp/sb/Saved_Model/PandaInsert-v0*. If no ID is specified, the latest experiment will be evaluated. Experiment ID 1 has already been trained for evaluation.

- To view logs:
    1. cd to file where training log is in 'cd exp/sb/Logs/PandaInsert-v0/PPO_1
    2. cmd 'tensorboard --logdir=.'
    3. access the tensorboard link (e.g. 'http://localhost:6006/) through a browser