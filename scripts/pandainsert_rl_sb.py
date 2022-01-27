import sys
import os
import gym

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from cur_id import exp_id_up, exp_id_get

arg_input = sys.argv[1:]
if len(arg_input) == 0:
    print("Input either 'learn' or 'check' in command line.")
    sys.exit()

exp_id = 0
if arg_input[0] == 'learn':
    learn_eval = 0
elif arg_input[0] == 'check':
    learn_eval = 1
    if len(arg_input) == 2: # if no id is given, default to curr_id
        exp_id = arg_input[1] # change to value desired id for eval
        print(f'Evaluating experiment_id: {exp_id}')       
    exp_id = str(exp_id)

algo = 'ppo'    # ppo or dqn

env_name = "PandaInsert-v0" # "CartPole-v1"
env_name_full = "rleasy_tut:" + env_name
time_steps_str = "1e5"
time_steps_int = int(float(time_steps_str))

# user inputs
# learn or eval
if learn_eval == 0:
    cur_exp_id = exp_id_up()
elif learn_eval == 1:
    cur_exp_id = exp_id_get()
    if exp_id != '0':
        cur_exp_id = exp_id

# Define save locations
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_PATH = os.path.join(parent_dir, "exp", "sb")
log_path = os.path.join(EXP_PATH, 'Logs', env_name)
model_path = os.path.join(EXP_PATH, 'Saved_Models', env_name)
model_name = algo + "_" + cur_exp_id \
  + "_" + env_name + "_" + time_steps_str
model_path = os.path.join(model_path, model_name)

# Create environment
env = gym.make(env_name_full)

if learn_eval == 0:
    env = DummyVecEnv([lambda: env])
    # Instantiate the agent
    if algo == 'ppo':
        model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
    elif algo == 'dqn':
        model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
    # Train the agent
    model.learn(total_timesteps=int(time_steps_int))
    # Save the agent
    model.save(model_path)
    # del model  # delete trained model to demonstrate loading

elif learn_eval == 1:
    # Load the trained agent
    if algo == 'ppo':
        model = PPO.load(model_path, env=env)
    elif algo == 'dqn':
        model = DQN.load(model_path, env=env)
    # Evaluate the agent
    # NOTE: If you use wrappers with your environment that modify rewards,
    #       this will be reflected here. To evaluate with original rewards,
    #       wrap environment in a "Monitor" wrapper before other wrappers.
    # mean_reward, std_reward = evaluate_policy(
    #     model, model.get_env().envs, n_eval_episodes=10, render=True
    # )
    # print(f'mean_reward={mean_reward} +/- {std_reward}')
    env = env.env
    episodes = 5
    for episode in range(1, episodes+1):
        env.do_render = True
        obs = env.reset()
        done = False
        score = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            print(action)
            score += reward
        print('Episode:{} Score:{}'.format(episode,score))

# To view logs:
# 1. cd to file where training log is in 'cd Training/Logs/PPO_1
# 2. cmd 'tensorboard --logdir=.'
# 3. access the tensorboard link (i.e. 'http://localhost:6006/) through browser