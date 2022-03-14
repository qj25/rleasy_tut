import mujoco_py
import matplotlib.pyplot as plt
import numpy as np
import gym
import random

from rleasy_tut.envs.panda_insert import PandaInsertEnv

def test_move():
    # env = PandaInsertEnv(do_render=True)
    env = gym.make("rleasy_tut:PandaInsert-v1")
    
    env.env.do_render = True
    phase = 1

    env.reset()

    # settings
    new_action = True

    # plot data
    plot_dat = True

    # move towards goal
    targ_pos = [0.48, 0.01, 0.10] # env.hole_pos.copy()
    rob_pos = env.observations['eef_pos'].copy()

    # intialize variables
    step_counter = 0
    action_counter = 0
    done = False
    eps_time = []
    dist_to_goal = []
    force_z = []
    dtg_z = []
    reward = []
    score = 0
    rand_move = random.randint(0,4)

    print("Moving...")
    while not done:
        step_counter += 1
        if phase == 0:
            if (
                step_counter % 3 == 0 
                or step_counter % 4 == 0
                or step_counter % 20 == 0
            ):
                move_step = 0
                action_counter += 1
            else:
                move_step = 1
        elif phase == 1:
            move_step = rand_move # random.randint(0,4)
        obs, rew, done, info = env.step(move_step)
        rob_pos = info['rob_pos']
        score += rew
        if step_counter == 30:
            pass
        if step_counter % 3 == 0:
            # input()
            if plot_dat:
                eps_time.append(info['eps_time'])
                dist_to_goal.append(info['dist_norm'])
                force_z.append(info['ft_world'][2])
                dtg_z.append(info['dtg_xyz'][2])
                reward.append(rew)
                print(f"---------- EPS_time = {info['eps_time']:7.3f} ----------")
                # print(f"dist_to_goal = {info['dist_norm']}")
                # print(f"reward = {rew}")
                # print(f"dtg_xyz = {info['dtg_xyz']}")
                print(f"move_step = {move_step}")
                print(f"rob_pos = {rob_pos}")
                # print(f"obs.type = {type(env.observation_space)}")
                # print(f"qpos = {env.observations['qpos']}")
                print(f"force-z = {info['ft_world'][2]}")
                # print(f"ft_world = {info['ft_world']}")
                print(f"env_steps = {env.env_steps}")
    
    # last data set
    eps_time.append(info['eps_time'])
    dist_to_goal.append(info['dist_norm'])
    force_z.append(info['ft_world'][2])
    dtg_z.append(info['dtg_xyz'][2])
    reward.append(rew)

    print('Score:{}'.format(score))
    if info['success'][1]:
        print("successfully reached pose")
    else:
        print("Did not reach pose")

    # plotting
    plt.figure("Z displacement against Time")
    plt.plot(eps_time, dtg_z)
    plt.legend(["dtg_z"])
    plt.ylabel('Depth')
    plt.xlabel('Time')
    plt.grid()

    plt.figure("Reward against Time")
    plt.plot(eps_time, reward)
    plt.legend(["rew"])
    plt.ylabel('Reward')
    plt.xlabel('Time')
    plt.grid()

    plt.show()


def scale_action(action):
    """
    Clips @action to be within self.input_min and self.input_max, and then re-scale the values to be within
    the range self.output_min and self.output_max

    Args:
        action (Iterable): Actions to scale

    Returns:
        np.array: Re-scaled action
    """
    input_max = np.array([1] * 3)
    input_min = np.array([-1] * 3)
    output_max = np.array([0.002] * 3)
    output_min = np.array([-0.002] * 3)

    action_scale = abs(output_max - output_min) / abs(
        input_max - input_min
    )
    action_output_transform = (output_max + output_min) / 2.0
    action_input_transform = (input_max + input_min) / 2.0
    action = np.clip(action, input_min, input_max)
    transformed_action = (
        action - action_input_transform
    ) * action_scale + action_output_transform

    return transformed_action

test_move()