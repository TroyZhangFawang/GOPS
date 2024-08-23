#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: vehicle 3DOF data environment
#  Update Date: 2021-05-55, Jiaxin Gao: create environment
import os
from typing import Dict, Optional, Sequence, Tuple
import pandas as pd
import torch
import gym
import numpy as np
from gops.env.env_ocp.pyth_base_env import PythBaseEnv
from gops.env.env_ocp.resources.ref_traj_data import MultiRefTrajData
from gops.utils.math_utils import angle_normalize


class VehicleDynamicsData:
    def __init__(self):
        self.v_x = 20
        self.m1 = 5760.  # Total mass of the tractor [kg]
        self.m1s = 4455.  # Sprung mass of the tractor [kg]
        self.m2 = 20665  # Total mass of the semitrailer [kg]
        self.m2s = 20000  # Sprung mass of the semitrailer [kg]
        self.gravity = 9.81
        self.a = 1.1  # Distance between the center of gravity (CG) of the tractor and its front axle [m]
        self.b = 2.8  # Distance between the CG of the tractor and its rear axle [m]
        self.c = 1.9  # Distance between the hitch point and the center of gravity (CG) of the tractor [m]
        self.e = 1.24  # Distance between the hitch point and the CG of the semitrailer [m]
        self.d = 6.9  # Distance between the rear axle and the CG of the semitrailer [m]

        self.h1 = 1.175  # Height of the CG of the sprung mass for the tractor [m]
        self.h2 = 2.125  # Height of the CG of the sprung mass for the semitrailer [m]
        self.h1c = 1.1  # Height of hitch point to the roll center of the sprung mass for the tractor [m]
        self.h2c = 1.1  # Height of hitch point to the roll center of the sprung mass for the semitrailer [m]

        self.I1zz = 34802.6  # Yaw moment of inertia of the whole mass of the tractor [kg m^2]
        self.I1xx = 2283  # Roll moment of inertia of the sprung mass of the tractor [kg m^2]
        self.I1xz = 1626  # Roll–yaw product of inertia of the sprung mass of the tractor [kg m^2]
        self.I2zz = 250416  # Yaw moment of inertia of the whole mass of the semitrailer [kg m^2]
        self.I2xx = 22330  # Roll moment of inertia of the sprung mass of the semitrailer [kg m^2]
        self.I2xz = 0.0  # Roll–yaw product of inertia of the sprung mass of the semitrailer [kg m^2]

        self.kf = -4.0889e5  # Tire cornering stiffness of the front axle of the tractor [N/rad]
        self.km = -9.1361e5  # Tire cornering stiffness of the rear axle of the tractor [N/rad]
        self.kr = -6.5922e5  # Tire cornering stiffness of the rear axle of the trailer [N/rad]
        self.kr1 = 9.1731e5  # roll stiffness of tire (front)[N/m]
        self.kr2 = 2.6023e6  # roll stiffness of tire (rear)[N/m]
        self.ka = 3.5503e6  # Roll stiffness of the articulation joint between the tractor and semitrailer [N m/rad]
        self.c1 = 1.2727e6  # Roll damping of the tractor's suspension [N-s/m]
        self.c2 = 4.1745e5  # Roll damping of the semitrailer's suspension [N-s/m]

        self.M11 = self.m1 * self.v_x * self.c
        self.M12 = self.I1zz
        self.M13 = -self.m1s * self.h1c * self.c - self.I1xz

        self.M21 = self.m1 * self.v_x * self.h1c - self.m1s * self.h1 * self.v_x
        self.M22 = -self.I1xz
        self.M24 = self.I1xx + 2 * self.m1s * self.h1 * self.h1 - self.m1s * self.h1 * self.h1c

        self.M31 = self.m1 * self.v_x
        self.M34 = -self.m1s * self.h1
        self.M35 = self.m2 * self.v_x
        self.M38 = -self.m2s * self.h2

        self.M45 = self.m2 * self.v_x * self.e
        self.M46 = -self.I2zz
        self.M48 = self.I2xz - self.m2s * self.h2 * self.e

        self.M55 = self.m2 * self.v_x * self.h2c - self.m2s * self.h2 * self.v_x
        self.M56 = -self.I2xz
        self.M58 = self.I2xx + 2 * self.m2s * self.h2 * self.h2 - self.m2s * self.h2 * self.h2c

        self.M61 = 1
        self.M62 = -self.c / self.v_x
        self.M64 = -self.h1c / self.v_x
        self.M65 = -1
        self.M66 = -self.e / self.v_x
        self.M68 = self.h2c / self.v_x

        self.M73 = 1

        self.M87 = 1

        self.M99 = 1
        self.M1010 = 1
        self.M111 = -self.v_x
        self.M1111 = 1
        self.M1212 = 1
        self.M1313 = 1

        self.A11 = (self.c + self.a) * self.kf + (self.c - self.b) * self.km
        self.A12 = self.a * (self.c + self.a) * self.kf / self.v_x - self.b * (
                self.c - self.b) * self.km / self.v_x - self.m1 * self.v_x * self.c
        self.A21 = (self.kf + self.km) * self.h1c
        self.A22 = (self.a * self.kf - self.b * self.km) * self.h1c / self.v_x + (
                self.m1s * self.h1 - self.m1 * self.h1c) * self.v_x
        self.A23 = self.m1s * self.gravity * self.h1 - self.kr1 - self.ka
        self.A24 = -self.c1
        self.A27 = self.ka

        self.A31 = self.kf + self.km
        self.A32 = (self.a * self.kf - self.b * self.km) / self.v_x - self.m1 * self.v_x
        self.A35 = self.kr
        self.A36 = -self.d * self.kr / self.v_x - self.m2 * self.v_x

        self.A45 = (self.e + self.d) * self.kr
        self.A46 = -self.d * (self.e + self.d) * self.kr / self.v_x - self.m2 * self.v_x * self.e
        self.A53 = self.ka
        self.A55 = self.kr * self.h2c
        self.A56 = (self.m2s * self.h2 - self.m2 * self.h2c) * self.v_x - self.d * self.kr * self.h2c / self.v_x
        self.A57 = self.m2s * self.gravity * self.h2 - self.kr2 - self.ka
        self.A58 = -self.c2
        self.A62 = -1
        self.A66 = 1
        self.A74 = 1
        self.A88, self.A92, self.A106 = 1, 1, 1
        self.A121, self.A129, self.A135, self.A1310 = self.v_x, self.v_x, self.v_x, self.v_x

        self.B11 = - (self.c + self.a) * self.kf
        self.B21 = -self.kf * self.h1c
        self.B31 = -self.kf

    def f_xu(self, states, actions, delta_t):
        state_next = np.empty_like(states)
        self.state_dim = 15

        X_matrix = np.hstack((states[6:14], states[2], states[5], states[14], states[1], states[4]))
        M_matrix = np.zeros((self.state_dim - 2, self.state_dim - 2))
        M_matrix[0, 0] = self.M11
        M_matrix[0, 1] = self.M12
        M_matrix[0, 2] = self.M13

        M_matrix[1, 0], M_matrix[1, 1], M_matrix[1, 3] = self.M21, self.M22, self.M24

        M_matrix[2, 0], M_matrix[2, 3], M_matrix[2, 4], M_matrix[2, 7] = \
            self.M31, self.M34, self.M35, self.M38

        M_matrix[3, 4], M_matrix[3, 5], M_matrix[3, 7], = self.M45, self.M46, self.M48

        M_matrix[4, 4], M_matrix[4, 5], M_matrix[4, 7] = self.M55, self.M56, self.M58

        M_matrix[5, 0], M_matrix[5, 1], M_matrix[5, 3], M_matrix[5, 4], M_matrix[5, 5], M_matrix[5, 7] = \
            self.M61, self.M62, self.M64, self.M65, self.M66, self.M68

        M_matrix[6, 2] = self.M73
        M_matrix[7, 6] = self.M87
        M_matrix[8, 8] = self.M99
        M_matrix[9, 9] = self.M1010
        M_matrix[10, 0], M_matrix[10, 10] = self.M111, self.M1111
        M_matrix[11, 11] = self.M1212
        M_matrix[12, 12] = self.M1313

        A_matrix = np.zeros((self.state_dim - 2, self.state_dim - 2))
        A_matrix[0, 0], A_matrix[0, 1], A_matrix[1, 0], A_matrix[1, 1], A_matrix[1, 2], A_matrix[1, 3], A_matrix[1, 6] = \
            self.A11, self.A12, self.A21, self.A22, self.A23, self.A24, self.A27
        A_matrix[2, 0], A_matrix[2, 1], A_matrix[2, 4], A_matrix[2, 5] \
            = self.A31, self.A32, self.A35, self.A36

        A_matrix[3, 4], A_matrix[3, 5] = self.A45, self.A46
        A_matrix[4, 2], A_matrix[4, 4], A_matrix[4, 5], A_matrix[4, 6], A_matrix[4, 7] = \
            self.A53, self.A55, self.A56, self.A57, self.A58
        A_matrix[5, 1], A_matrix[5, 5], A_matrix[6, 3], A_matrix[7, 7], \
        A_matrix[8, 1], A_matrix[9, 5], A_matrix[11, 0], A_matrix[11, 8], \
        A_matrix[12, 4], A_matrix[12, 9] = self.A62, self.A66, self.A74, self.A88, \
                                           self.A92, self.A106, self.A121, self.A129, \
                                           self.A135, self.A1310

        B_matrix = np.zeros((self.state_dim - 2, 1))
        B_matrix[0, 0] = self.B11
        B_matrix[1, 0] = self.B21
        B_matrix[2, 0] = self.B31
        # beta1dot, phi1_dotdot, psi1dot, psi1_dotdot,
        # beta2dot, phi2_dotdot, psi2dot, psi2_dotdot,
        # psi1dot, psi2dot, vy1dot py1dot,
        X_dot = (np.matmul(np.matmul(np.linalg.inv(M_matrix), A_matrix), X_matrix) + np.matmul(
            np.linalg.inv(M_matrix), np.matmul(B_matrix, actions))).squeeze()

        #px1, py1, psi1, px2, py2, psi2,
        # beta1, psi1_dot, phi1, phi1_dot, beta2, psi2_dot, phi2, phi2_dot, vy1
        state_next[0] = states[0] + delta_t * (self.v_x * np.cos(states[2]) - states[14] * np.sin(states[2]))
        state_next[1] = states[1] + delta_t * (self.v_x * np.sin(states[2]) + states[14] * np.cos(states[2]))
        state_next[2] = states[2] + delta_t * X_dot[8]
        state_next[3] = state_next[0] - self.b * np.cos(states[2]) - self.e * np.cos(
            states[5])  # posx_trailer
        state_next[4] = state_next[1] - self.b * np.sin(states[2]) - self.e * np.sin(
            states[5])  # posy_trailer
        state_next[5] = states[5] + delta_t * X_dot[9]
        state_next[6:14] = states[6:14] + delta_t * X_dot[0:8]
        state_next[14] = states[14] + delta_t * X_dot[10]
        return state_next

class Semitruckpu7dof(PythBaseEnv):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    def __init__(
        self,
        pre_horizon: int = 100,
        path_para: Optional[Dict[str, Dict]] = None,
        u_para: Optional[Dict[str, Dict]] = None,
        max_steer: float = 0.5,
        **kwargs,
    ):
        work_space = kwargs.pop("work_space", None)
        if work_space is None:
            # initial range of delta_px1, delta_py1, delta_phi1, delta_px2, delta_py2, delta_phi2,
            # beta1, phi1_dot, varphi1, varphi1_dot, \
            # beta2, phi2_dot, varphi2, varphi2_dot, \
            # vy1]

            # 用高斯分布去采样
            init_high = np.array([2, 2, 0.1, 2, 2, 0.1,
                                  0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
            init_low = np.array([-2, -2,  -0.1, -2, -2, -0.1,
                                 -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, ], dtype=np.float32)
            work_space = np.stack((init_low, init_high))
        super(Semitruckpu7dof, self).__init__(work_space=work_space, **kwargs)

        self.vehicle_dynamics = VehicleDynamicsData()
        self.target_speed = self.vehicle_dynamics.v_x
        self.ref_traj = MultiRefTrajData(path_para, u_para)

        self.state_dim = 15
        self.pre_horizon = pre_horizon
        ego_obs_dim = 15
        ref_obs_dim = 6
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf] * (ego_obs_dim + ref_obs_dim * pre_horizon)),
            high=np.array([np.inf] * (ego_obs_dim + ref_obs_dim * pre_horizon)),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-max_steer]),
            high=np.array([max_steer]),
            dtype=np.float32,
        )
        obs_scale_default = [1/100, 1/100, 1/10, 1/100, 1/100, 1/10,
                             1, 1, 1, 1,
                             1, 1, 1, 1,
                             1/100]
        self.obs_scale = np.array(kwargs.get('obs_scale', obs_scale_default))

        self.dt = 0.01
        self.max_episode_steps = 3000

        self.state = None
        self.ref_points = None
        self.t = None
        self.t2 = None
        self.path_num = None
        self.u_num = None

        self.info_dict = {
            "state": {"shape": (self.state_dim,), "dtype": np.float32},
            "ref_points": {"shape": (self.pre_horizon + 1, 6), "dtype": np.float32},
            "ref": {"shape": (6,), "dtype": np.float32},
            "path_num": {"shape": (), "dtype": np.uint8},
            "u_num": {"shape": (), "dtype": np.uint8},
            "target_speed": {"shape": (), "dtype": np.float32},
            "ref_time": {"shape": (), "dtype": np.float32},
            "ref_time2": {"shape": (), "dtype": np.float32},
        }

        self.seed()

    @property
    def additional_info(self) -> Dict[str, Dict]:
        return self.info_dict

    def reset(
        self,
        init_state: Optional[Sequence] = None,
        ref_time: Optional[float] = None,
        ref_num: Optional[int] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, dict]:

        if ref_time is not None:
            self.t = ref_time
        else:
            self.t = 20.0 * self.np_random.uniform(0.0, 1.0)

        # Calculate path num and speed num: ref_num = [0, 1, 2,..., 7]
        if ref_num is None:
            path_num = None
            u_num = None
        else:
            path_num = int(ref_num / 2)
            u_num = int(ref_num % 2)

        # If no ref_num, then randomly select path and speed
        if path_num is not None:
            self.path_num = path_num
        else:
            self.path_num = self.np_random.choice([0, 1, 2, 3, 4, 5]) #, 6

        if u_num is not None:
            self.u_num = u_num
        else:
            self.u_num = 0#self.np_random.choice([1])


        ref_points = []
        for i in range(self.pre_horizon + 1):
            ref_x = self.ref_traj.compute_x(
                self.t + i * self.dt, self.path_num, self.u_num
            )
            ref_y = self.ref_traj.compute_y(
                self.t + i * self.dt, self.path_num, self.u_num
            )
            ref_phi = self.ref_traj.compute_phi(
                self.t + i * self.dt, self.path_num, self.u_num
            )

            self.t2 = self.t - (self.vehicle_dynamics.c+self.vehicle_dynamics.e)/self.target_speed
            ref_x2 = self.ref_traj.compute_x(
                self.t2 + i * self.dt, self.path_num, self.u_num
            )
            ref_y2 = self.ref_traj.compute_y(
                self.t2 + i * self.dt, self.path_num, self.u_num
            )
            ref_phi2 = self.ref_traj.compute_phi(
                self.t2 + i * self.dt, self.path_num, self.u_num
            )
            ref_points.append([ref_x, ref_y, ref_phi, ref_x2, ref_y2, ref_phi2])
        self.ref_points = np.array(ref_points, dtype=np.float32)

        if init_state is not None:
            delta_state = np.array(init_state, dtype=np.float32)
        else:
            delta_state = self.sample_initial_state()

        self.state = np.concatenate(
            (self.ref_points[0] + delta_state[:6], delta_state[6:])
        )
        obs = self.get_obs()
        self.action_last = 0
        return obs, self.info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        action = np.clip(action, self.action_space.low, self.action_space.high)

        reward = self.compute_reward(action)

        self.state = self.vehicle_dynamics.f_xu(self.state, action, self.dt)
        self.t = self.t + self.dt
        self.t2 = self.t2 + self.dt

        self.ref_points[:-1] = self.ref_points[1:]
        new_ref_point = np.array(
            [
                self.ref_traj.compute_x(
                    self.t + self.pre_horizon * self.dt, self.path_num, self.u_num
                ),
                self.ref_traj.compute_y(
                    self.t + self.pre_horizon * self.dt, self.path_num, self.u_num
                ),
                self.ref_traj.compute_phi(
                    self.t + self.pre_horizon * self.dt, self.path_num, self.u_num
                ),
            ],
            dtype=np.float32,
        )

        new_ref_point_2 = np.array(
            [
                self.ref_traj.compute_x(
                    self.t2 + self.pre_horizon * self.dt, self.path_num, self.u_num
                ),
                self.ref_traj.compute_y(
                    self.t2 + self.pre_horizon * self.dt, self.path_num, self.u_num
                ),
                self.ref_traj.compute_phi(
                    self.t2 + self.pre_horizon * self.dt, self.path_num, self.u_num
                ),
            ],
            dtype=np.float32,
        )
        self.ref_points[-1] = np.hstack((new_ref_point, new_ref_point_2))

        obs = self.get_obs()
        self.done = self.judge_done()

        self.action_last = action[0]
        return obs, reward, self.done, self.info

    def get_obs(self) -> np.ndarray:
        ref_x_tf, ref_y_tf, ref_phi_tf = \
            ego_vehicle_coordinate_transform(
                self.state[0], self.state[1], self.state[2],
                self.ref_points[:, 0], self.ref_points[:, 1], self.ref_points[:, 2],
            )

        ref_x2_tf, ref_y2_tf, ref_phi2_tf = \
            ego_vehicle_coordinate_transform(
                self.state[3], self.state[4], self.state[5],
                self.ref_points[:, 3], self.ref_points[:, 4], self.ref_points[:, 5],
            )

        # ego_obs: [
        # delta_y, delta_phi, delta_y2, delta_phi2 (of the first reference point)
        # v, w, varphi (of ego vehicle, including tractor and trailer)
        # ]

        ego_obs = np.concatenate(
            ([ref_x_tf[0]*self.obs_scale[0], ref_y_tf[0]*self.obs_scale[1], ref_phi_tf[0]*self.obs_scale[2], ref_x2_tf[0]*self.obs_scale[3], ref_y2_tf[0]*self.obs_scale[4], ref_phi2_tf[0]*self.obs_scale[5]],
             self.state[6:14], self.state[14:]*self.obs_scale[14]))
        # ref_obs: [
        # delta_y, delta_phi (of the second to last reference point)
        # ]
        ref_obs = np.stack((ref_x_tf*self.obs_scale[0], ref_y_tf*self.obs_scale[1], ref_phi_tf*self.obs_scale[2], ref_x2_tf*self.obs_scale[3],ref_y2_tf*self.obs_scale[4], ref_phi2_tf*self.obs_scale[5]), 1)[1:].flatten()
        return np.concatenate((ego_obs, ref_obs))

    def compute_reward(self, action: np.ndarray) -> float:
        px1, py1, phi1, px2, py2, phi2, beta1, phi1_dot, varphi1, varphi1_dot, \
        beta2, phi2_dot, varphi2, varphi2_dot, \
        vy1 = self.state

        ref_x, ref_y, ref_phi = self.ref_points[0][0:3]
        steer = action[0]
        return -(
            1 * ((px1 - ref_x) ** 2 + (py1 - ref_y) ** 2)
            + 0.9 * vy1 ** 2
            + 0.8 * angle_normalize(phi1 - ref_phi) ** 2
            + 0.5 * phi1_dot ** 2
            + 0.5 * beta1 ** 2
            + 0.5 * varphi1 ** 2
            + 0.5 * varphi1_dot ** 2
            + 0.4 * steer ** 2
            + 2.0 * (steer - self.action_last) ** 2
        )

    def judge_done(self) -> bool:
        done = ((np.abs(self.state[1]-self.ref_points[0, 1]) > 3)  # delta_y1
                  + (np.abs(angle_normalize(self.state[2]-self.ref_points[0, 2])) > np.pi)  # delta_phi1
                  + (np.abs(self.state[4]-self.ref_points[0, 4]) > 3) # delta_y2
                  + (np.abs(angle_normalize(self.state[5]-self.ref_points[0, 5])) > np.pi))  # delta_phi2
        #(np.abs(self.state[0]-self.ref_points[0, 0]) > 5)+   # delta_x1
        #+ (np.abs(self.state[3] - self.ref_points[0, 3]) > 5)  # delta_x2
        return done

    @property
    def info(self) -> dict:
        return {
            "state": self.state.copy(),
            "ref_points": self.ref_points.copy(),
            "ref": self.ref_points[0].copy(),
            "target_speed": self.target_speed,
            "path_num": self.path_num,
            "u_num": self.u_num,
            "ref_time": self.t,
            "ref_time2": self.t2,
        }

def state_error_calculate(
    ego_x: np.ndarray,
    ego_y: np.ndarray,
    ego_phi: np.ndarray,
    ref_x: np.ndarray,
    ref_y: np.ndarray,
    ref_phi: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_err = ego_x - ref_x
    y_err = ego_y - ref_y
    phi_err = ego_phi - ref_phi
    return x_err, y_err, phi_err

def ego_vehicle_coordinate_transform(
    ego_x: np.ndarray,
    ego_y: np.ndarray,
    ego_phi: np.ndarray,
    ref_x: np.ndarray,
    ref_y: np.ndarray,
    ref_phi: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform absolute coordinate of ego vehicle and reference points to the ego
    vehicle coordinate. The origin is the position of ego vehicle. The x-axis points
    to heading angle of ego vehicle.

    Args:
        ego_x (np.ndarray): Absolution x-coordinate of ego vehicle, shape ().
        ego_y (np.ndarray): Absolution y-coordinate of ego vehicle, shape ().
        ego_phi (np.ndarray): Absolution heading angle of ego vehicle, shape ().
        ref_x (np.ndarray): Absolution x-coordinate of reference points, shape (N,).
        ref_y (np.ndarray): Absolution y-coordinate of reference points, shape (N,).
        ref_phi (np.ndarray): Absolution tangent angle of reference points, shape (N,).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Transformed x, y, phi of reference
        points.
    """
    cos_tf = np.cos(-ego_phi)
    sin_tf = np.sin(-ego_phi)
    ref_x_tf = (ref_x - ego_x) * cos_tf - (ref_y - ego_y) * sin_tf
    ref_y_tf = (ref_x - ego_x) * sin_tf + (ref_y - ego_y) * cos_tf
    ref_phi_tf = angle_normalize(ref_phi - ego_phi)
    return ref_x_tf, ref_y_tf, ref_phi_tf

def env_creator(**kwargs):
    """
    make env `pyth_semitruckpu7dof`
    """
    return Semitruckpu7dof(**kwargs)
