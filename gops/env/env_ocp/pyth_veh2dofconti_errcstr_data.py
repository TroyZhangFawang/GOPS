#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Vehicle 2DOF data environment with tracking error constraint

import gym
import numpy as np

from gops.env.env_ocp.pyth_base_data import PythBaseEnv
from gops.env.env_ocp.pyth_veh2dofconti_data import VehicleDynamics


class SimuVeh2dofcontiErrCstr(PythBaseEnv):
    def __init__(self, path_para:dict = None,
                 u_para:dict = None, y_error_tol: float = 0.2, **kwargs):
        self.vehicle_dynamics = VehicleDynamics(**kwargs)

        work_space = kwargs.pop("work_space", None)
        if work_space is None:
            # initial range of [delta_y, delta_phi, v, w]
            u = self.vehicle_dynamics.vehicle_params["u"]
            init_high = np.array([1, np.pi / 3, u * 0.25, 0.9], dtype=np.float32)
            init_low = -init_high
            work_space = np.stack((init_low, init_high))
        super().__init__(work_space=work_space, **kwargs)

        self.is_adversary = kwargs.get("is_adversary", False)
        self.is_constraint = kwargs.get("is_constraint", False)
        self.pre_horizon = kwargs["pre_horizon"]
        self.base_frequency = 10
        self.state_dim = 4
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf] * (self.pre_horizon + self.state_dim)),
            high=np.array([np.inf] * (self.pre_horizon + self.state_dim)),
            dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([-np.pi / 6]),
                                           high=np.array([np.pi / 6]),
                                           dtype=np.float32)

        self.max_episode_steps = 200
        self.obs = None
        self.state = None
        self.path_num = None
        self.para = None
        self.u_num = None
        self.u_para = None
        self.t = None
        self.info_dict = {
            "state": {"shape": self.state_dim, "dtype": np.float32},
            "ref": {"shape": (2,), "dtype": np.float32},
            "path_num": {"shape": (), "dtype": np.uint8},
            "u_num": {"shape": (), "dtype": np.uint8},
            "ref_time": {"shape": (), "dtype": np.float32},
            "constraint": {"shape": (1,), "dtype": np.float32},
        }
        self.y_error_tol = y_error_tol
        self.seed()

        path_key = ['A_x',
                    'omega_x',
                    'phi_x',
                    'b_x',
                    'A_y',
                    'omega_y',
                    'phi_y',
                    'b_y',
                    'double_lane_control_point_1',
                    'double_lane_control_point_2',
                    'double_lane_control_point_3',
                    'double_lane_control_point_4',
                    'double_lane_control_y1',
                    'double_lane_control_y3',
                    'double_lane_control_y5',
                    'double_lane_control_y2_a',
                    'double_lane_control_y2_b',
                    'double_lane_control_y4_a',
                    'double_lane_control_y4_b',
                    'square_wave_period',
                    'square_wave_amplitude',
                    'circle_radius',
                    ]
        path_value = [1., 2 * np.pi / 6, 0, 10, 1.5, 2 * np.pi / 10, 0, 0, 5, 9, 14, 18, 0, 3.5, 0, 0.875, -4.375,
                      -0.875, 15.75, 5, 1, 200]
        self.path_para = dict(zip(path_key, path_value))
        if path_para != None:
            for i in path_para.keys(): self.path_para[i] = path_para[i]

        u_key = ['A', 'omega', 'phi', 'b', 'speed']

        u_value = [1, 2 * np.pi / 6, 0, 0.5, 5]

        self.u_para = dict(zip(u_key, u_value))

        if u_para != None:
            for i in u_para.keys(): self.u_para[i] = u_para[i]

    @property
    def additional_info(self):
        return self.info_dict

    def reset(self, init_state=None, ref_time=None, path_num=None, u_num=None, **kwargs):
        init_y = None
        init_phi = None
        init_v = None
        init_w = None
        obs = None
        if (init_state is None) & (ref_time is None) & (path_num is None) & (u_num is None):
            obs = self.sample_initial_state()
            delta_y, delta_phi, v, w = obs
            flag = [0, 1, 2, 3]
            self.path_num = self.np_random.choice(flag)
            u_flag = [1]
            self.u_num = self.np_random.choice(u_flag)
            ref_time = 20. * self.np_random.uniform(low=0., high=1.)
            self.t = ref_time
            init_y = self.vehicle_dynamics.compute_path_y(self.t, self.path_num, self.path_para, self.u_num, self.u_para) + delta_y
            init_phi = self.vehicle_dynamics.compute_path_phi(self.t, self.path_num, self.path_para, self.u_num, self.u_para) + delta_phi
            init_v = v
            init_w = w
        elif (init_state is not None) & (ref_time is not None) & (path_num is not None) & (u_num is not None):
            self.path_num = path_num
            self.u_num = u_num
            self.t = ref_time
            init_y, init_phi, init_v, init_w = init_state[0], init_state[1], init_state[2], init_state[3]
            init_delta_y = self.vehicle_dynamics.compute_path_y(self.t, self.path_num, self.path_para, self.u_num, self.u_para) - init_y
            init_delta_phi = self.vehicle_dynamics.compute_path_phi(self.t, self.path_num, self.path_para, self.u_num, self.u_para) + init_phi
            obs = np.array([init_delta_y, init_delta_phi, init_v, init_w], dtype=np.float32)
        else:
            print("reset error")

        for i in range(self.pre_horizon):
            ref_y = self.vehicle_dynamics.compute_path_y(self.t + (i + 1) / self.base_frequency, self.path_num, self.path_para, self.u_num, self.u_para)
            ref_obs = np.array([init_y - ref_y], dtype=np.float32)
            obs = np.hstack((obs, ref_obs))
        self.obs = obs
        self.state = np.array([init_y, init_phi, init_v, init_w], dtype=np.float32)
        return self.obs, self.info

    def step(self, action: np.ndarray, adv_action=None):  # think of action is in range [-1, 1]
        steer_norm = action
        action = steer_norm
        reward = self.vehicle_dynamics.compute_rewards(self.obs, action)
        self.t = self.t + 1.0 / self.base_frequency
        self.state, self.obs = self.vehicle_dynamics.simulation(
            self.state, action, self.base_frequency, self.path_num, self.path_para, self.u_num, self.u_para, self.t)
        self.done = self.judge_done(self.state, self.t)
        if self.done:
            reward = reward - 100

        return self.obs, reward, self.done, self.info

    def judge_done(self, state, t):
        y, phi, v, w = state[0], state[1], state[2], state[3]
        done = (np.abs(y - self.vehicle_dynamics.compute_path_y(t, self.path_num, self.path_para, self.u_num, self.u_para)) > 2) | \
               (np.abs(phi - self.vehicle_dynamics.compute_path_phi(t, self.path_num, self.path_para, self.u_num, self.u_para)) > np.pi / 4.)
        return done

    def get_constraint(self) -> np.ndarray:
        y = self.state[0]
        y_ref = self.vehicle_dynamics.compute_path_y(self.t, self.path_num, self.path_para, self.u_num, self.u_para)
        constraint = np.array([abs(y - y_ref) - self.y_error_tol], dtype=np.float32)
        return constraint

    @property
    def info(self):
        state = np.array(self.state, dtype=np.float32)
        y_ref = self.vehicle_dynamics.compute_path_y(self.t, self.path_num, self.path_para, self.u_num, self.u_para)
        ref = np.array([y_ref], dtype=np.float32)
        return {
            "state": state,
            "ref": ref,
            "path_num": self.path_num,
            "u_num": self.u_num,
            "ref_time": self.t,
            "constraint": self.get_constraint(),
        }


def env_creator(**kwargs):
    return SimuVeh2dofcontiErrCstr(**kwargs)
