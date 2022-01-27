#!/usr/bin/env python3

## Next:
# -change action limit (or change to discrete)
# -change action rot to velocity

# -try learning search phase
# -create new rew func for insertion phase
# -2 learning on 1 model

## Rmb:
# -conducting only search phase currently
# -distance not rounded
# -no hole offset
# -penalize for being in env

## Questions:
# -rounding of c in paper

import numpy as np
import os
import mujoco_py
import gym
from gym import utils, spaces
import rleasy_tut.utils.transform_utils as T
from rleasy_tut.utils.transform_utils import IDENTITY_QUATERNION
from rleasy_tut.utils.mjc_utils import MjSimWrapper
from rleasy_tut.controllers.hpf_controller import HPFController


IDENTITY_QUATERNION = np.array([1.0, 0, 0, 0])

class PandaInsertEnv(gym.Env, utils.EzPickle):

    def __init__(
        self,
        do_render=False,
        xml_string=None,
    ):
        """
        General:

        action space --> [Fdx, Fdy, Fdx, Rdx, Rdy]  
        3 force and 2 rotation (x,y) commands

        obs space    --> [Fx, Fy, Fz, Mx, My, Px, Py]
        """
        utils.EzPickle.__init__(self)

        # simulation-specific attributes
        self.viewer = None
        self.xml_string = xml_string
        # self.model = None
        self.sim = None
        self.do_render = do_render

        # init obs
        self.observations = dict(
            eef_pose_hf=np.concatenate((np.zeros(3), IDENTITY_QUATERNION)),
            qpos=np.zeros(7),
            eef_pos=np.zeros(3),
            eef_quat=np.zeros(4),
            eef_vel=np.zeros(7),
            ft_world=np.zeros(6),
        )

        # load model
        if self.xml_string is None:
            fullpath = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "assets/pandainsert_world.xml"
            )
        # initialize simulation
        self.model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(self.model)
        self.sim = MjSimWrapper(self.sim)   # wrap sim
        self.data = self.sim.data
        self.viewer = None

        # init hole an robot pos
        self.hole_top_z = 0.15
        self.hole_pos_true = np.array([0.53, 0.08, 0.13])
        self.hole_pos = self.hole_pos_true.copy()
        self.hole_quat = IDENTITY_QUATERNION

        # robot's initial position (add error) = [0.48, 0.01, 0.14]
        self.init_qpos = [
            1.02877863e-02,
            9.78997975e-03,
            1.05301861e-02,
            -2.45014158e+00,
            -1.62896285e-04,
            2.45993274e+00,
            8.06834195e-01,
        ]
        # np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])
        self.init_qvel = np.zeros(7)
        self.data.qpos[:] = np.array(self.init_qpos)
        self.data.qvel[:] = np.array(self.init_qvel)
        # self.data.ctrl[:] = np.zeros(self.model.nv)
        self.sim.forward()

        # initialize viewer
        if self.do_render and self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim.sim)
            self.viewer.cam.azimuth = 120
            self.viewer.cam.distance = self.sim.model.stat.extent * 0.75 
            self.viewer.cam.elevation = 0

        # initialize controller
        self.controller = HPFController(
            self.sim,
            actuator_names=self.model.actuator_names,
            eef_name="cylinder1",
            eef_site_name="cyl_top_site",
        )

        # other variables
        self.insert_done = False
        self.max_env_steps = 1000
        self.env_steps = 0
        self.cur_time = 0
        self.dt = self.sim.model.opt.timestep

        self.first_depth = 5e-4
        self.final_depth = 19e-3
        self.init_insertion_phase()


    def _get_observations(self):
        self.observations["eef_vel"] = self.controller.state["eef_vel"]
        self.observations["eef_pos"] = self.controller.state["eef_pos"]
        self.observations["eef_quat"] = self.controller.state["eef_quat"]
        qpos = self.data.qpos
        self.observations["qpos"] = qpos
        self.observations["ft_world"] = self.controller.state["ft_world"]

        h_p_inv, h_q_inv = T.pose_inverse(self.hole_pos, self.hole_quat)
        eef_p_hf, eef_q_hf = T.multiply_pose(
            h_p_inv,
            h_q_inv,
            self.observations["eef_pos"],
            self.observations["eef_quat"],
        )
        self.observations["eef_pose_hf"] = np.concatenate((eef_p_hf, eef_q_hf))
        return np.concatenate(
            (
                [0, 0],
                self.observations["ft_world"][2:5],
                [0, 0],
            )
        )

    def reset(self):
        self.sim.reset()

        if self.do_render and self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim.sim)
            self.viewer.cam.azimuth = 120
            self.viewer.cam.distance = self.sim.model.stat.extent * 0.75 
            self.viewer.cam.elevation = 0
        
        # reset hole
        self.hole_pos_true = np.array([0.53, 0.08, 0.13])
        self.hole_pos = self.hole_pos_true.copy()
        self.hole_quat = IDENTITY_QUATERNION

        # reset robot
        self.data.qpos[:] = np.array(self.init_qpos)
        self.data.qvel[:] = np.array(self.init_qvel)
        # self.data.ctrl[:] = np.zeros(self.model.nv)
        self.sim.forward()

        # reset obs
        self.observations = dict(
            eef_pose_hf=np.concatenate((np.zeros(3),IDENTITY_QUATERNION)),
            qpos=np.zeros(7),
            eef_pos=np.zeros(3),
            eef_quat=np.zeros(4),
            eef_vel=np.zeros(6),
            ft_world=np.zeros(6),
        )
        self.controller.reset()

        # reset time
        self.cur_time = 0   #clock time of episode
        self.env_steps = 0

        self.search_done = False
        self.insert_done = False
        self.init_pos()

        return self._get_observations()

    def render(self):
        self.viewer.render()

    def reward(self, ):
        rew = 0
        # rew for distance and rounding
        self.dist_to_goal = self.observations["eef_pose_hf"][:3].copy()            
        Z = self.final_depth - self.first_depth
        curr_depth = (
            self.hole_top_z 
            - self.observations['eef_pos'][2]
            - self.first_depth
        )
        rew = - (Z - curr_depth) / Z

        rew = np.clip(rew, -1, 0)
        if rew <= -1:
            rew = -100.
        # reward for successful run
        rew_success = 100 * (1.0 - self.env_steps / self.max_env_steps)
        if self.search_done or self.insert_done:
            rew = rew_success
        return rew

    def _check_proximity(self, pos1, pos2):
        # True if close, False if not close enough
        distbtwn = pos1 - pos2
        if(
            np.abs(distbtwn)[2] < 2e-3
            and np.linalg.norm(distbtwn[:2]) < 1e-5
        ):
            return True
        return False

    def step(self, action):
        # action
        decoded_action = self.decode_action(action)
        self.controller.set_goal(decoded_action)
        while not self.controller.is_stepped():
            tau_cmd = self.controller.compute_torque_cmd()
            self.controller.set_torque_cmd(tau_cmd)
            self.sim.step()
            self.sim.forward()
            
            if self.do_render:
                self.render()
            self.cur_time += self.dt
        self.env_steps += 1

        # check done 
        if (
            self.hole_top_z - self.observations['eef_pos'][2] 
            > self.final_depth
        ):
            self.insert_done = True
        done = (self.env_steps >= self.max_env_steps) or self.insert_done

        # reward (comes after check done to check success)
        reward = self.reward()
        if reward <= -1: # exited safe boundary
            done = True

        # additional information
        info = dict(
            success=[self.search_done, self.insert_done],
            eps_time=self.cur_time,
            sim_steps=self.cur_time / self.dt,
            dist_norm=np.linalg.norm(self.dist_to_goal),
            dtg_xyz=self.dist_to_goal,
            rob_pos=self.observations['eef_pos'],
            ft_world=self.observations['ft_world']
        )

        return self._get_observations(), reward, done, info

    def init_pos(self):
        targ_pos = self.hole_pos.copy()
        targ_pos[2] = self.hole_top_z + 1e-3
        # added 1e-3 to ensure no collision 
        # due rotated pin during first move
        self.rob_rotxy_error = np.random.uniform(
            -self.rob_rot_error_limit, self.rob_rot_error_limit, size=2
        )

        # intialize variables
        done = False
        step_counter = 0
        while not done:
            step_counter += 1
            rob_pos = self.observations['eef_pos'].copy()
            move_dir = targ_pos[:3] - rob_pos
            move_dir = move_dir * 10
            move_step = np.concatenate((move_dir, [np.pi, 0]))
            move_step[3:] += self.rob_rotxy_error 
            self.controller.set_goal(move_step, pose_cmd=True)
            tau_cmd = self.controller.compute_torque_cmd(pose_cmd=True)
            self.controller.set_torque_cmd(tau_cmd)
            self.sim.step()
            self.sim.forward()
            if self.do_render:
                self.render()
            done = self._check_proximity(
                targ_pos, self.observations['eef_pos']
            )
            if step_counter > 10000:
                break
            self._get_observations()
        if not done:
            raise RuntimeError('Failed to reach move down position!')
        
        done = False
        while not done:
            step_counter += 1
            move_step = self.decode_action(0)
            move_step[3:] += self.rob_rotxy_error 
            self.controller.set_goal(move_step)
            tau_cmd = self.controller.compute_torque_cmd()
            self.controller.set_torque_cmd(tau_cmd)
            self.sim.step()
            self.sim.forward()
            if self.do_render:
                self.render()
            done = (
                (self.hole_top_z - self.observations['eef_pos'][2])
                > self.first_depth
            )
            if step_counter > 10000:
                break
            self._get_observations()
        if not done:
            raise RuntimeError('Failed to reach init_pos!')

    def init_insertion_phase(self):
        """
        Insertion Phase (1):
        action space --> [0, 0, Fdx, Rdx, Rdy]
        action space will be chosen from one of these 4:
        1) [0,0,−Fdz,0,0]
        2) [0,0,−Fdz,+Rdx,0]
        3) [0,0,−Fdz,−Rdx,0]
        4) [0,0,−Fdz,0,+Rdy]
        5) [0,0,−Fdz,0,−Rdy]
        where all Fd and Rd will be variable.

        obs space    --> [0, 0, Fz, Mx, My, 0, 0]
        """

        # init variables
        self.rob_rot_error_limit = 0.1
        # create observation and action spaces
        obs = self._get_observations()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape
        )
        action_size = 5
        self.action_space = spaces.Discrete(action_size)

    def decode_action(self, action):
        """
        Takes in action depending on phase and returns the general action
        for controller to process.
        """
        Fd = 20.0
        Rd = self.rob_rot_error_limit
        actions_list = np.array([
            [0., 0., -Fd, 0., 0.],
            [0., 0., -Fd, Rd, 0.],
            [0., 0., -Fd, -Rd, 0.],
            [0., 0., -Fd, 0., Rd],
            [0., 0., -Fd, 0., -Rd],
        ])
        decoded_act = actions_list[action]
        decoded_act[3] += np.pi # account for initial rotated position
        decoded_act[:3] *= -1
        return decoded_act