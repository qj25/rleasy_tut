import sys
import os
import gym
import matplotlib.pyplot as plt

from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import plot_results
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from cur_id import exp_id_up, exp_id_get
from rleasy_tut.utils.callback import SaveOnBestTrainingRewardCallback

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

env_name = "PandaInsert-v1" # "CartPole-v1"
env_name_full = "rleasy_tut:" + env_name
time_steps_str = "1e6"  # change training time steps here
time_steps_int = int(float(time_steps_str))

# user inputs
# learn or eval
if learn_eval == 0:
    cur_exp_id = exp_id_up()
elif learn_eval == 1:
    cur_exp_id = exp_id_get()
    check_best = True
    if exp_id != '0':
        cur_exp_id = exp_id
        check_best = False
    else:
        print('Evaluating best model...')

# Define save locations
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_PATH = os.path.join(parent_dir, "exp", "sb")
log_path = os.path.join(EXP_PATH, 'Logs', env_name)
bestmodel_path = os.path.join(log_path, 'best_model.zip')
model_path = os.path.join(EXP_PATH, 'Saved_Models', env_name)
model_name = algo + "_" + cur_exp_id \
  + "_" + env_name + "_" + time_steps_str
model_path = os.path.join(model_path, model_name)

# Create environment
env = gym.make(env_name_full)
env = Monitor(env, log_path)
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_path)

if learn_eval == 0:
    env = DummyVecEnv([lambda: env])
    # Instantiate the agent
    if algo == 'ppo':
        model = PPO(
            'MlpPolicy',
            env,
            learning_rate=8e-4,
            # ent_coef=1e-3,
            verbose=1,
            tensorboard_log=log_path,
        )
    elif algo == 'dqn':
        model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
    # Train the agent
    model.learn(
        total_timesteps=int(time_steps_int),
        callback=callback
    )
    # Save the agent
    model.save(model_path)
    # del model  # delete trained model to demonstrate loading
    # Plot results
    plot_results(
        [log_path],
        time_steps_int,
        results_plotter.X_TIMESTEPS,
        "PPO PandaInsert")
    plt.show()

elif learn_eval == 1:
    # Load the trained agent
    if algo == 'ppo':
        if check_best == False:
            model = PPO.load(model_path, env=env)
        else:
            model = PPO.load(bestmodel_path, env=env)
    elif algo == 'dqn':
        model = DQN.load(model_path, env=env)
    env = env.env
    episodes = 5
    for episode in range(1, episodes+1):
        env.unwrapped.do_render = True
        obs = env.reset()
        done = False
        score = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            # Printing info for each timestep
            print(f"Discrete action taken: {action}")
            print(f"observation: {obs}")
            print(f"reward = {reward}")
            score += reward
        print('Episode:{} Score:{}'.format(episode,score))

# To view logs:
# 1. cd to file where training log is in 'cd Training/Logs/PPO_1
# 2. cmd 'tensorboard --logdir=.'
# 3. access the tensorboard link (i.e. 'http://localhost:6006/) through browser