import numpy as np
import rleasy_tut.utils.transform_utils as T
from rleasy_tut.utils.filters import ButterLowPass
from rleasy_tut.utils.mjc_utils import get_contact_force

class HPFController:
    def __init__(
        self,
        sim,
        actuator_names,
        eef_name="capsule1",
        eef_site_name="cap_top_site",
        kp=150,
        kp_f=1.5e-5,
        damping_ratio=1,
        control_freq=40,
        ramp_ratio=0.8,
    ):
        self.sim = sim

        # time
        self.ramp_ratio = 0.8
        self.dynamic_timestep = self.sim.model.opt.timestep
        self.control_freq = control_freq
        self.control_timestep = 1 / self.control_freq
        self.interpolate_steps = np.ceil(
            self.control_timestep / self.dynamic_timestep * ramp_ratio
        )

        # filter
        fs = 1.0 / self.dynamic_timestep
        cutoff = 30
        self.lowpass_filter = ButterLowPass(cutoff, fs, order=5)

        # ref
        self.eef_name = eef_name
        self.eef_site_name = eef_site_name
        self.eef_site_idx = self.sim.model.site_name2id(
            self.eef_site_name
        )
        actuator_names = actuator_names
        self.actuator_ids = [
            self.sim.model.actuator_name2id(n) for n in actuator_names
        ]
        self.joint_ids = self.sim.model.actuator_trnid[self.actuator_ids, 0]
        
        # information
        self._state = dict(
            eef_pos=np.zeros(3),
            eef_quat=np.zeros(4),
            eef_vel=np.zeros(6),
            ft_world=np.zeros(6),
        )

        # gain
        self.damping_ratio = damping_ratio
        self._kp = kp
        self._kd = 2 * np.sqrt(self._kp) * damping_ratio
        self._kp_f = kp_f
        self.dtorque_lim = 1

        self._update_state()
        self.prev_torq = self.sim.data.qfrc_bias[self.joint_ids]
        self._pd = self._state['eef_pos']
        self._qd = self._state['eef_quat']
        self._fd = self._state['ft_world'][:3]
        self.steps = 0

    def set_goal(self, action, pose_cmd=False):
        self.steps = 0
        # pos
        if pose_cmd:
            self._p0 = self._pd
            self.action = self.scale_action(action[:3], out_max=0.002)

        # force
        self._f0 = self._fd
        self.goal_force = action[:3]

        # quat
        self._q0 = self._qd
        eef_quat = self._state['eef_quat']
        concat_axisangle = np.concatenate((
            [action[3]], [0]
        ))
        concat_axisangle = np.concatenate((
            concat_axisangle, [action[4]]
        ))
        action_ori = T.axisangle2quat(concat_axisangle)
        ori_action = T.quat_error(eef_quat, action_ori)
        # ori_action = np.concatenate((action[3:], [0]))
        scaled_ori_a = self.scale_action(ori_action, out_max=0.015)
        scaled_quat_a = T.axisangle2quat(scaled_ori_a)
        self.goal_ori = T.quat_multiply(scaled_quat_a, self._q0)

    def set_torque_cmd(self, torque_cmd):
        # clip change in torque command
        torque = self.prev_torq + np.clip(
            torque_cmd - self.prev_torq, -self.dtorque_lim, self.dtorque_lim
        )
        self.prev_torq = torque
        self.sim.data.ctrl[self.joint_ids] = torque

    def compute_torque_cmd(self, pose_cmd=False):
        self.steps += 1
        # setting desired
        qd = T.quat_slerp(
            self._q0, self.goal_ori, self.steps / self.interpolate_steps
        )
        fd = (
                self._f0
                + (self.goal_force - self._f0)
                * self.steps
                / self.interpolate_steps
            )
        if self.steps > self.interpolate_steps:
            qd = self.goal_ori
            fd = self.goal_force
        
        # get Jacobian
        J_pos = self.sim.data.get_site_jacp(self.eef_site_name).reshape(
            (3, -1)
        )
        J_ori = self.sim.data.get_site_jacr(self.eef_site_name).reshape(
            (3, -1)
        )
        J_full = np.vstack([J_pos, J_ori])[:, self.joint_ids]

        # errors
        # pos
        # e_pos = pd - self._state['eef_pos']
        # force
        eef_force = self._state['ft_world'][:3]
        ef = -(fd - eef_force)
        ep_f = self._kp_f * ef
        pd = self._pd + ep_f
        if pose_cmd:
            pd = (
                self._p0
                + self.action
                * self.steps
                / self.interpolate_steps
            )
            if self.steps > self.interpolate_steps:
                pd = self._p0 + self.action
        e_pos = pd - self._state['eef_pos']
        # quat
        eef_quat = self._state['eef_quat']
        e_ori = T.quat_error(eef_quat, qd)
        # overall
        pose_error = np.concatenate((e_pos, e_ori))
        vd = np.zeros(6)    # desired velocity
        e_vel = vd - self._state['eef_vel']
        # coriolis and gravity compensation
        torque_dynamic = self.sim.data.qfrc_bias[self.joint_ids]
        torque_cmd = (
            np.dot(J_full.T, self._kp * pose_error + self._kd * e_vel) + \
            torque_dynamic
        )
        # update desired p and q
        self._pd = pd
        self._qd = qd
        self._fd = fd
        # update state
        self._update_state()
        
        return torque_cmd

    def _update_state(self):
        # get eef_vel
        ee_pos_vel = self.sim.data.site_xvelp[self.eef_site_idx]
        ee_ori_vel = self.sim.data.site_xvelr[self.eef_site_idx]
        self._state['eef_vel'] = np.concatenate(
            (ee_pos_vel, ee_ori_vel)
        )
        
        # get eef_pos and eef_quat
        eef_pos = np.array(
            self.sim.data.site_xpos[self.eef_site_idx]
        )
        self._state['eef_pos'] = eef_pos
        eefmat = np.array(
            self.sim.data.site_xmat[self.eef_site_idx].reshape((3, 3))
        )
        eef_quat = T.mat2quat(eefmat)
        self._state['eef_quat'] = eef_quat

        # get eef force
        eef_ft = get_contact_force(
            self.sim.model,
            self.sim.data,
            self.eef_name,
            self._state['eef_pos'],
            self._state['eef_quat'],
        )
        eef_ft_filtered = self.lowpass_filter(eef_ft.reshape((-1, 6)))[0, :]
        # force in world frame
        eef_rotmat = T.quat2mat(eef_quat)
        f_world = eef_rotmat.dot(eef_ft_filtered[:3])
        t_world = eef_rotmat.dot(eef_ft_filtered[3:])
        self._state['ft_world'] = np.concatenate((f_world, t_world))

    def reset(self):
        self.steps = 0
        
        self._update_state()
        self.prev_torq = self.sim.data.qfrc_bias[self.joint_ids]
        self._pd = self._state['eef_pos']
        self._qd = self._state['eef_quat']
        self._fd = self._state['ft_world'][:3]
        self.action = np.zeros(3)
        self.goal_ori = self._state['eef_quat']

    def is_stepped(self):
        return self.steps > np.ceil(
            self.control_timestep / self.dynamic_timestep
        )

    def move_to_pose(self, targ_pos):
        # intialize variables
        done = False
        step_counter = 0
        
        while not done:
            step_counter += 1
            rob_pos = self._state['eef_pos']
            move_dir = targ_pos[:3] - rob_pos
            move_dir = move_dir * 10
            move_step = np.concatenate((move_dir, [0, 0]))
            self.set_goal(move_step, pose_cmd=True)
            tau_cmd = self.compute_torque_cmd(pose_cmd=True)
            self.set_torque_cmd(tau_cmd)
            self.sim.forward()
            done = self._check_proximity(
                targ_pos, self._state['eef_pos']
            )
        print(step_counter)

    @property
    def state(self):
        return self._state
    
    def scale_action(self, action, out_max = 0.015):
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
        output_max = np.array([out_max] * 3)
        output_min = np.array([-out_max] * 3)

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
