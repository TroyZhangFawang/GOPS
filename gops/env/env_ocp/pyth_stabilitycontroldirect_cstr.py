#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: 4WD vehicle stability control model environment
#  Update Date: 2024-04-13, Fawang Zhang: create environment
import os
from typing import Dict, Optional, Sequence, Tuple
import pandas as pd
import torch
from scipy.linalg import *
import gym
import numpy as np
from gops.env.env_ocp.pyth_base_env import PythBaseEnv
from gops.env.env_ocp.resources.ref_traj_data import MultiRefTrajData, MultiRoadSlopeData
from gops.utils.math_utils import angle_normalize


class VehicleDynamicsData:
    def __init__(self):
        self.vehicle_params = dict(
            state_dim=8,
            m=2257 + 139.4 + 172,  # Total mass[kg]
            mu=139.4 + 172,
            ms=2257,  # Sprung mass[kg]
            A=3.3,  # Front area
            rho=1.206,  # air mass density
            Cd=0.3,  # coefficient of air force
            g=9.81,
            Rw=0.368,
            Iw=3.1,  # wheel spin inertia [kg m2]
            mu_r=0.015,  # rolling resistence coefficient
            lw=0.8625 * 2,
            lf=1.33,  # Distance between the center of gravity (CG)and its front axle [m]
            lr=3.140 - 1.33,  # Distance between the CGand its rear axle [m]
            hs=0.766731475 - 0.2,  # Height of the CG of the sprung mass for to the ground [m]
            hr=0.2,  # Height of the CG of the roll center to the ground
            hu=0.4,  # Height of the CG of the un-sprung mass to the ground
            Izz=3524.9,  # Yaw moment of inertia of the whole mass[kg m^2]
            Ixx=846.6,  # Roll moment of inertia of the sprung mass[kg m^2]
            Ixz=0,  # Roll–yaw product of inertia of the sprung mass[kg m^2]
            k_alpha1=0.1744 * 1.416 * 1.026e+04 / 3.14 * 180,  # Tire cornering stiffness of the 1st wheel[N/rad]
            k_alpha2=0.1744 * 1.416 * 1.026e+04 / 3.14 * 180,  # Tire cornering stiffness of the 1st wheel[N/rad]
            k_alpha3=0.1744 * 1.416 * 1.026e+04 / 3.14 * 180,  # Tire cornering stiffness of the rear axle[N/rad]
            k_alpha4=0.1744 * 1.416 * 1.026e+04 / 3.14 * 180,  # Tire cornering stiffness of the rear axle[N/rad]
            C_slip1=8.885 * 1.525 * 1.062e+04,  # N
            C_slip2=8.885 * 1.525 * 1.062e+04,  # N
            C_slip3=8.885 * 1.525 * 1.062e+04,  # N
            C_slip4=8.885 * 1.525 * 1.062e+04,  # N
            K_varphi=(569 / 3.14 * 180 + 510 / 3.14 * 180) * 4,  # roll stiffness of suspension [N-m/rad] /3.14*180
            C_varphi=0,  # Roll damping of the suspension [N-m-s/rad]
            mu_road=0.85,  # road Adhesion coefficient
        )

        self.m = self.vehicle_params["m"]  # Total mass[kg]
        self.mu = self.vehicle_params["mu"]
        self.ms = self.vehicle_params["ms"]  # Sprung mass[kg]
        self.g = self.vehicle_params["g"]
        self.mu_r = self.vehicle_params["mu_r"]
        self.Rw = self.vehicle_params["Rw"]
        self.Iw = self.vehicle_params["Iw"]
        self.A = self.vehicle_params["A"]
        self.rho = self.vehicle_params["rho"]
        self.Cd = self.vehicle_params["Cd"]
        self.lw = self.vehicle_params["lw"]
        self.lf = self.vehicle_params["lf"]  # Distance between the center of gravity (CG)and its front axle [m]
        self.lr = self.vehicle_params["lr"]  # Distance between the CGand its rear axle [m]

        self.hs = self.vehicle_params["hs"]  # Height of the CG of the sprung mass for the tractor [m]
        self.hr = self.vehicle_params["hr"]
        self.hu = self.vehicle_params["hu"]
        self.Izz = self.vehicle_params["Izz"]  # Yaw moment of inertia of the whole mass[kg m^2]
        self.Ixx = self.vehicle_params["Ixx"]  # Roll moment of inertia of the sprung mass[kg m^2]
        self.Ixz = self.vehicle_params["Ixz"]  # Roll–yaw product of inertia of the sprung mass[kg m^2]

        self.k_alpha1 = self.vehicle_params["k_alpha1"]  # Tire cornering stiffness of the 1st wheel[N/rad]
        self.k_alpha2 = self.vehicle_params["k_alpha2"]  # Tire cornering stiffness of the 1st wheel[N/rad]
        self.k_alpha3 = self.vehicle_params["k_alpha3"]  # Tire cornering stiffness of the rear axle[N/rad]
        self.k_alpha4 = self.vehicle_params["k_alpha4"]  # Tire cornering stiffness of the rear axle[N/rad]
        self.C_slip1 = self.vehicle_params["C_slip1"]
        self.C_slip2 = self.vehicle_params["C_slip2"]
        self.C_slip3 = self.vehicle_params["C_slip3"]
        self.C_slip4 = self.vehicle_params["C_slip4"]

        self.K_varphi = self.vehicle_params["K_varphi"]  # roll stiffness of tire [N-m/rad] /3.14*180
        self.C_varphi = self.vehicle_params["C_varphi"]  # Roll damping of the suspension [N-m-s/rad]
        self.mu_road = self.vehicle_params["mu_road"]

    def f_xu(self, states, actions, delta_t, road_info):
        theta_road, varphi_road = road_info
        R = np.array([theta_road, varphi_road]).reshape(2, 1)

        x, y, phi, v_x, v_y, phi_dot, varphi, varphi_dot = states[:8]
        X = np.array(states[3:8]).reshape(5, 1)
        U = actions.reshape(5, 1)
        delta = actions[4]
        state_next = np.zeros_like(states)
        dividend = (self.m * self.Ixx * self.Izz - self.Izz * self.ms ** 2 * self.hs ** 2 - self.m * self.Ixz ** 2)
        A_matrix = np.zeros((5, 5))

        A_matrix[1, 2], A_matrix[1, 3], A_matrix[1, 4] = -v_x, -self.ms * self.hs * self.Izz * (
                self.K_varphi - self.ms * self.g * self.hs) / dividend, \
                                                               -self.ms * self.hs * self.Izz * self.C_varphi / dividend

        A_matrix[2, 3], A_matrix[2, 4] = \
            -self.m * self.Ixz * (self.K_varphi - self.ms * self.g * self.hs) / dividend, \
            -self.m * self.Ixz * self.C_varphi / dividend

        A_matrix[3, 4] = 1

        A_matrix[4, 3], A_matrix[4, 4] = \
            -self.m * self.Izz * (self.K_varphi - self.ms * self.g * self.hs) / dividend, \
            -self.m * self.Izz * self.C_varphi / dividend

        B_matrix = np.zeros((5, 3))
        B_matrix[0, 0] = 1 / self.m
        B_matrix[1, 1], B_matrix[1, 2] = \
            (self.Ixx * self.Izz - self.Ixz ** 2) / dividend, \
            self.Ixz * self.ms * self.hs / dividend

        B_matrix[2, 1], B_matrix[2, 2] = \
            self.Ixz * self.ms * self.hs / dividend, \
            (self.m * self.Ixx - self.ms ** 2 * self.hs ** 2) / dividend

        B_matrix[4, 1], B_matrix[4, 2] = \
            self.Izz * self.ms * self.hs / dividend, \
            (self.m * self.Ixz) / dividend

        R_matrix = np.zeros((5, 2))
        R_matrix[0, 0] = -self.g
        R_matrix[1, 1] = (self.Izz * self.ms * self.hs * self.K_varphi -
                          self.g * self.m * (self.Ixx * self.Izz - self.Ixz ** 2)) / dividend

        R_matrix[2, 1] = (self.m * self.Ixz * self.K_varphi -
                          self.m * self.Ixz * self.ms * self.hs * self.g) / dividend

        R_matrix[4, 1] = (self.m * self.Izz * self.K_varphi - self.m * self.Izz * self.ms * self.hs * self.g) / dividend

        Lc_matrix = np.zeros((3, 8))
        Lc_matrix[0, 0], Lc_matrix[0, 2], Lc_matrix[0, 4], Lc_matrix[0, 6] = 1, 1, 1, 1

        Lc_matrix[1, 1], Lc_matrix[1, 3], Lc_matrix[1, 5], Lc_matrix[1, 7] = 1, 1, 1, 1

        Lc_matrix[2, 0], Lc_matrix[2, 1], Lc_matrix[2, 2], Lc_matrix[2, 3], \
            Lc_matrix[2, 4], Lc_matrix[2, 5], Lc_matrix[2, 6], Lc_matrix[2, 7] \
            = -self.lw / 2, self.lf, self.lw / 2, self.lf, \
              -self.lw / 2, -self.lr, self.lw / 2, -self.lr

        Mw1 = np.array([[np.cos(delta), -np.sin(delta)],
                        [np.sin(delta), np.cos(delta)]])
        Mw2 = np.array([[np.cos(delta), -np.sin(delta)],
                        [np.sin(delta), np.cos(delta)]])
        Mw3 = np.array([[1, 0],
                        [0, 1]])
        Mw4 = np.array([[1, 0],
                        [0, 1]])

        Mw_matrix = block_diag(Mw1, Mw2, Mw3, Mw4)

        At_matrix = np.zeros((8, 5))

        At_matrix[1, 1], At_matrix[1, 2] = -self.k_alpha1 / v_x, -self.k_alpha1 * self.lf / v_x

        At_matrix[3, 1], At_matrix[3, 2] = -self.k_alpha2 / v_x, -self.k_alpha2 * self.lf / v_x

        At_matrix[5, 1], At_matrix[5, 2] = -self.k_alpha3 / v_x, -self.k_alpha3 * (-self.lr) / v_x

        At_matrix[7, 1], At_matrix[7, 2] = -self.k_alpha4 / v_x, -self.k_alpha4 * (-self.lr) / v_x

        Bt_matrix = np.zeros((8, 5))
        Bt_matrix[0, 0], Bt_matrix[2, 1], Bt_matrix[4, 2], Bt_matrix[
            6, 3] = 1 / self.Rw, 1 / self.Rw, 1 / self.Rw, 1 / self.Rw
        Bt_matrix[1, 4], Bt_matrix[3, 4] = self.k_alpha1, self.k_alpha2

        temp = np.matmul(At_matrix, X) + np.matmul(Bt_matrix, U)

        X_dot = (np.matmul(A_matrix, X) + np.matmul(np.matmul(np.matmul(
            B_matrix, Lc_matrix), Mw_matrix), temp) + np.matmul(R_matrix, R)).squeeze()

        state_next[0] = x + delta_t * (v_x * np.cos(phi) - v_y * np.sin(phi))
        state_next[1] = y + delta_t * (v_y * np.cos(phi) + v_x * np.sin(phi))
        state_next[2] = phi + delta_t * phi_dot
        state_next[2] = angle_normalize(state_next[2])
        state_next[3:8] = states[3:8] + delta_t * X_dot
        return state_next

class FourwdstabilitycontrolCstr(PythBaseEnv):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    def __init__(
        self,
        pre_horizon: int = 30,
        min_torque: float = 0.0,
        max_torque: float = 298.0,
        max_steer: float = 0.5,
        path_para: Optional[Dict[str, Dict]] = None,
        u_para: Optional[Dict[str, Dict]] = None,
        slope_para: Optional[Dict[str, Dict]] = None,
        **kwargs,
    ):
        work_space = kwargs.pop("work_space", None)
        if work_space is None:
            # initial range of [delta_x, delta_y, delta_yaw, delta_vx, vy, yaw rate, roll, roll rate]
            # 用高斯分布去采样
            init_high = np.array([2, 1, np.pi/6, 2, 2, 0.1, 0.1, 0.1], dtype=np.float32)
            init_low = -init_high
            work_space = np.stack((init_low, init_high))
        super(FourwdstabilitycontrolCstr, self).__init__(work_space=work_space, **kwargs)

        self.vehicle_dynamics = VehicleDynamicsData()
        self.pre_horizon = pre_horizon
        self.ref_traj = MultiRefTrajData(path_para, u_para)
        self.road_slope = MultiRoadSlopeData(slope_para)
        self.state_dim = self.vehicle_dynamics.vehicle_params["state_dim"]
        ego_obs_dim = self.state_dim
        ref_obs_dim = 4
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf] * (ego_obs_dim + ref_obs_dim * pre_horizon+2*pre_horizon)),
            high=np.array([np.inf] * (ego_obs_dim + ref_obs_dim * pre_horizon+2*pre_horizon)),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.array([min_torque] * 4 + [-max_steer]),
            high=np.array([max_torque] * 4 + [max_steer]),
            dtype=np.float32,
        )
        obs_scale_default = [1/100, 1/100, 1/10,
                             1/100, 1/100, 1/10, 1/10, 1/50]
        self.obs_scale = np.array(kwargs.get('obs_scale', obs_scale_default))

        self.dt = 0.01
        self.max_episode_steps = 1000

        self.state = None
        self.ref_x = None
        self.ref_y = None
        self.ref_points = None

        self.info_dict = {
            "state": {"shape": (self.state_dim,), "dtype": np.float32},
            "ref_points": {"shape": (self.pre_horizon + 1, 4), "dtype": np.float32},
            "slope_points": {"shape": (self.pre_horizon + 1, 2), "dtype": np.float32},
            "path_num": {"shape": (), "dtype": np.uint8},
            "u_num": {"shape": (), "dtype": np.uint8},
            "slope_num": {"shape": (), "dtype": np.uint8},
            "ref": {"shape": (4,), "dtype": np.float32},
            "ref_time": {"shape": (), "dtype": np.float32},
            "constraint": {"shape": (2, ), "dtype": np.float32},
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
            self.t = self.np_random.uniform(0.0, 20.0)
        # Calculate path num and speed num: ref_num = [0, 1, 2,..., 7]
        if ref_num is None:
            path_num = None
            u_num = None
            slope_num = None
        else:
            path_num = int(ref_num / 2)
            u_num = int(ref_num % 2)
            slope_num = int(ref_num % 2)

        # If no ref_num, then randomly select path and speed
        if path_num is not None:
            self.path_num = path_num
        else:
            self.path_num = self.np_random.choice([1])

        if u_num is not None:
            self.u_num = u_num
        else:
            self.u_num = self.np_random.choice([0])

        if slope_num is not None:
            self.slope_num = slope_num
        else:
            self.slope_num = self.np_random.choice([1])

        ref_points = []
        slope_points = []
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
            ref_u = self.ref_traj.compute_u(
                self.t + i * self.dt, self.path_num, self.u_num
            )
            ref_points.append([ref_x, ref_y, ref_phi, ref_u])

            road_longi = self.road_slope.compute_longislope(self.t+i*self.dt, self.slope_num)
            road_lat = self.road_slope.compute_latslope(self.t+i*self.dt, self.slope_num)
            slope_points.append([road_longi, road_lat])

        self.ref_points = np.array(ref_points, dtype=np.float32)
        self.slope_points = np.array(slope_points, dtype=np.float32)

        if init_state is not None:
            delta_state = np.array(init_state, dtype=np.float32)
        else:
            delta_state = self.sample_initial_state()

        self.state = np.concatenate(
            (self.ref_points[0] + delta_state[:4], delta_state[4:8]))

        return self.get_obs(), self.info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        action = np.clip(action, self.action_space.low, self.action_space.high)

        reward = self.compute_reward(action)

        self.state = self.vehicle_dynamics.f_xu(self.state, action, self.dt, self.slope_points[1])

        self.t = self.t + self.dt

        self.ref_points[:-1] = self.ref_points[1:]
        self.slope_points[:-1] = self.slope_points[1:]
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
                self.ref_traj.compute_u(
                    self.t + self.pre_horizon * self.dt, self.path_num, self.u_num
                ),
            ],
            dtype=np.float32,
        )
        new_slope_point = np.array([self.road_slope.compute_longislope(self.t+self.pre_horizon*self.dt, self.slope_num),
                                    self.road_slope.compute_latslope(self.t+self.pre_horizon*self.dt, self.slope_num)])

        self.ref_points[-1] = new_ref_point
        self.slope_points[-1] = new_slope_point
        self.done = self.judge_done()
        if self.done:
            reward = reward - 1000
        return self.get_obs(), reward, self.done, self.info

    def get_obs(self) -> np.ndarray:
        ref_x_tf, ref_y_tf, ref_phi_tf = \
            ego_vehicle_coordinate_transform(
                self.state[0], self.state[1], self.state[2],
                self.ref_points[:, 0], self.ref_points[:, 1], self.ref_points[:, 2],
            )
        ref_u_tf = self.ref_points[:, 3] - self.state[3]
        # ego_obs: [
        # delta_x, delta_y, delta_phi, delta_u, (of the first reference point)
        # v, w (of ego vehicle)
        # ]
        ego_obs = np.concatenate(
            ([ref_x_tf[0]*self.obs_scale[0], ref_y_tf[0]*self.obs_scale[1], ref_phi_tf[0]*self.obs_scale[2], ref_u_tf[0]*self.obs_scale[3]],
             [self.state[4]*self.obs_scale[4], self.state[5]*self.obs_scale[5], self.state[6]*self.obs_scale[6], self.state[7]*self.obs_scale[7]]))
        # ref_obs: [
        # delta_x, delta_y, delta_phi, delta_u (of the second to last reference point)
        # ]
        ref_obs = np.stack((ref_x_tf*self.obs_scale[0], ref_y_tf*self.obs_scale[1], ref_phi_tf*self.obs_scale[2], ref_u_tf*self.obs_scale[3], self.slope_points[:, 0], self.slope_points[:, 1]), 1)[1:].flatten()
        return np.concatenate((ego_obs, ref_obs))

    def compute_reward(self, action: np.ndarray) -> float:
        px, py, phi, vx, vy, phi_dot, varphi, varphi_dot = self.state[:8]
        # Q1, Q2,  Q3,  Q4, delta = action
        # delta1, delta2, delta3, delta4 = delta, delta, delta, delta
        # Q1, delta1, Q2, delta2, Q3, delta3, Q4, delta4 = action #, dQ1, ddelta1, dQ2, ddelta2, dQ3, ddelta3, dQ4, ddelta4
        # beta = np.arctan(vy/vx)
        ref_x, ref_y, ref_phi, ref_vx = self.ref_points[0]
        # I_matrix = np.array([[(self.vehicle_dynamics.k_alpha1+self.vehicle_dynamics.k_alpha2+
        #                       self.vehicle_dynamics.k_alpha3+self.vehicle_dynamics.k_alpha4)/(self.vehicle_dynamics.m*vx),
        #                       self.vehicle_dynamics.lf*(self.vehicle_dynamics.k_alpha1+self.vehicle_dynamics.k_alpha2)-
        #                       self.vehicle_dynamics.lr*(self.vehicle_dynamics.k_alpha3+self.vehicle_dynamics.k_alpha4)/(self.vehicle_dynamics.m*vx**2)],
        #                      [self.vehicle_dynamics.lf*(self.vehicle_dynamics.k_alpha1+self.vehicle_dynamics.k_alpha2)-
        #                       self.vehicle_dynamics.lr*(self.vehicle_dynamics.k_alpha3+self.vehicle_dynamics.k_alpha4)/(self.vehicle_dynamics.Izz),
        #                       self.vehicle_dynamics.lf**2*(self.vehicle_dynamics.k_alpha1+self.vehicle_dynamics.k_alpha2)+
        #                       self.vehicle_dynamics.lr**2*(self.vehicle_dynamics.k_alpha3+self.vehicle_dynamics.k_alpha4)/(self.vehicle_dynamics.Izz*vx)]])
        #
        # k_matrix = np.array([[-self.vehicle_dynamics.k_alpha1/(self.vehicle_dynamics.m*vx),
        #                       -self.vehicle_dynamics.k_alpha2/(self.vehicle_dynamics.m*vx),
        #                       -self.vehicle_dynamics.k_alpha3/(self.vehicle_dynamics.m*vx),
        #                       -self.vehicle_dynamics.k_alpha4/(self.vehicle_dynamics.m*vx)],
        #                      [-self.vehicle_dynamics.lf*self.vehicle_dynamics.k_alpha1/self.vehicle_dynamics.Izz,
        #                       -self.vehicle_dynamics.lf*self.vehicle_dynamics.k_alpha2/self.vehicle_dynamics.Izz,
        #                       self.vehicle_dynamics.lr*self.vehicle_dynamics.k_alpha3/self.vehicle_dynamics.Izz,
        #                       self.vehicle_dynamics.lr*self.vehicle_dynamics.k_alpha4/self.vehicle_dynamics.Izz]])
        # delta_matrix = np.array([[delta1], [delta2], [delta3], [delta4]])
        #
        # later_ref = np.matmul(np.matmul(np.linalg.inv(I_matrix), k_matrix), delta_matrix)
        # beta_ref = 0#later_ref[0][0]
        phi_dot_ref = 0#later_ref[1][0]
        C_varphi = 2/(self.vehicle_dynamics.m*self.vehicle_dynamics.g*self.vehicle_dynamics.lw*np.cos(self.slope_points[0, 0])*np.cos(self.slope_points[0, 1]))*\
                   (self.vehicle_dynamics.K_varphi*(1+(self.vehicle_dynamics.ms*self.vehicle_dynamics.hr+
                                                       self.vehicle_dynamics.mu*self.vehicle_dynamics.hu)/
                                                    (self.vehicle_dynamics.ms*self.vehicle_dynamics.hs))-(self.vehicle_dynamics.ms*self.vehicle_dynamics.hr+
                                                       self.vehicle_dynamics.mu*self.vehicle_dynamics.hu)*self.vehicle_dynamics.g*np.cos(self.slope_points[0, 1]))
        C_varphi_dot = 2*C_varphi/(self.vehicle_dynamics.m*self.vehicle_dynamics.g*self.vehicle_dynamics.lw*np.cos(self.slope_points[0, 0])*np.cos(self.slope_points[0, 1]))*\
                       ((1+(self.vehicle_dynamics.ms*self.vehicle_dynamics.hr+self.vehicle_dynamics.mu*self.vehicle_dynamics.hu)/
                                                    (self.vehicle_dynamics.ms*self.vehicle_dynamics.hs)))
        I_rollover = C_varphi*varphi+C_varphi_dot*varphi_dot
        # kappa_constant = 0.15#vx/self.vehicle_dynamics.Rw
        # r_action_Q = np.sum((action[0:4]/298) ** 2)
        # r_action_str = np.sum((action[4:]) ** 2)
        r_action_Qdot = (action[0]/(298*10)) ** 2+(action[1]/(298*10)) ** 2+(action[2]/(298*10)) ** 2+(action[3]/(298*10)) ** 2
        r_action_strdot = (action[4]/(0.1*10)) ** 2
        # r_action_deltaQ = dQ1 ** 2 + dQ2 ** 2 + dQ3 ** 2 + dQ4 ** 2
        # r_action_deltastr = np.sum((action[9:16:2]) ** 2)
        # r_action_deltaQdot = np.sum((action[8:16:2]-self.action_last[8:16:2]) ** 2)
        # r_action_deltastrdot = np.sum((action[9:16:2]-self.action_last[9:16:2]) ** 2)
        return -(
                0.04 * ((px - ref_x) ** 2 + (py - ref_y) ** 2)
                + 0.04 * (vx - ref_vx) ** 2
                + 0.02 * angle_normalize(phi - ref_phi) ** 2
                + 0.01 * (phi_dot - phi_dot_ref) ** 2
                + 0.02 * I_rollover ** 2
                # + 0.01 * r_action_Q
                # + 0.01 * r_action_str
                + 0.01 * r_action_Qdot
                + 0.01 * r_action_strdot
                # + 0.5 * (beta - beta_ref) ** 2
                # + 1e-8 * r_action_deltaQ
                # + 1e-4 * r_action_deltastr
                # + 1e-4 * r_action_deltaQdot
                # + 1e-1 * r_action_deltastrdot
        )

    def judge_done(self) -> bool:
        done = (abs(self.state[0]-self.ref_points[0, 0]) > 5 # delta_x
                +(abs(self.state[1]-self.ref_points[0, 1]) > 3)  # delta_y
                + (abs(angle_normalize(self.state[2] - self.ref_points[0, 2])) > np.pi)  # delta phi
                + (abs(self.state[3]-self.ref_points[0, 3]) > 3))  # delta_vx
                # + (abs(self.state[4]) > 2)  # delta_vy
        return done

    def get_constraint(self) -> np.ndarray:
        side_slip_angle = self.state[4] / self.state[3]
        constraint = np.array([abs(self.state[5]) - abs(self.vehicle_dynamics.mu_road*self.vehicle_dynamics.g/self.state[3]), abs(side_slip_angle) - abs(np.arctan(0.02*self.vehicle_dynamics.mu_road*self.vehicle_dynamics.g))], dtype=np.float32)
        return constraint


    @property
    def info(self) -> dict:
        return {
            "state": self.state.copy(),
            "ref_points": self.ref_points.copy(),
            "slope_points": self.slope_points.copy(),
            "path_num": self.path_num,
            "u_num": self.u_num,
            "slope_num": self.slope_num,
            "ref": self.ref_points[0].copy(),
            "ref_time": self.t,
            "constraint": self.get_constraint(),
        }

def state_error_calculate(
    ego_x: np.ndarray,
    ego_y: np.ndarray,
    ego_phi: np.ndarray,
    ego_vx: np.ndarray,
    ref_x: np.ndarray,
    ref_y: np.ndarray,
    ref_phi: np.ndarray,
    ref_vx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_err = ego_x - ref_x
    y_err = ego_y - ref_y
    phi_err = ego_phi - ref_phi
    vx_err = ego_vx - ref_vx
    return x_err, y_err, phi_err, vx_err

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
    return FourwdstabilitycontrolCstr(**kwargs)


