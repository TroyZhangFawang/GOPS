#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: 4WS 4WD vehicle holistic control model environment
#  Update Date: 2024-04-13, Fawang Zhang: create environment
import os
from typing import Dict, Optional, Sequence, Tuple
import pandas as pd
import torch
from scipy.linalg import *
import gym
import numpy as np
from gops.env.env_ocp.pyth_base_env import PythBaseEnv
from gops.env.env_ocp.resources.ref_traj_data import MultiRefTrajData
from gops.utils.math_utils import angle_normalize

def read_path(root_path):
    data_result = pd.DataFrame(pd.read_csv(root_path, header=None))
    state_1 = np.array(data_result.iloc[1:, 0], dtype='float32') #x
    state_2 = np.array(data_result.iloc[1:, 1], dtype='float32')  #y
    state_traj = np.zeros((len(state_1), 2))
    state_traj[:, 0] = state_1
    state_traj[:, 1] = state_2
    return state_traj

class Ref_Route:
    def __init__(self, ref_vx):
        self.preview_index = 5
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = current_dir+"/resources/cury.csv"

        self.ref_traj = read_path(root_dir)
        self.ref_vx = ref_vx

    def find_nearest_point(self, traj_points):
        # 计算两组点之间的距离
        traj_points_expanded = np.expand_dims(traj_points, axis=1)
        distances = np.linalg.norm(traj_points_expanded - self.ref_traj, axis=2)

        # 找到每个轨迹点在参考轨迹上的最近点的索引
        nearest_point_index = np.argmin(distances, axis=1)+self.preview_index
        ref_x = self.ref_traj[nearest_point_index][:, 0]
        ref_y = self.ref_traj[nearest_point_index][:, 1]
        prev_index = np.maximum(nearest_point_index - 1, 0)
        ref_heading = np.arctan2((ref_y - self.ref_traj[prev_index][:, 1]),
                                 (ref_x - self.ref_traj[prev_index][:, 0]))
        ref_vx = np.empty_like(ref_x)+self.ref_vx
        return np.stack([ref_x, ref_y, ref_heading, ref_vx], axis=1)

class VehicleDynamicsData:
    def __init__(self):
        self.vehicle_params = dict(
            state_dim=8,
            m=2257 + 139.4 + 172,  # Total mass[kg]
            mu=139.4 + 172,
            ms=2257,  # Sprung mass[kg]
            g=9.81,
            Rw=0.368,
            Iw=3.1,  # wheel spin inertia [kg m2]
            mu_r=0.015,  # rolling resistence coefficient
            lw=0.8625 * 2,
            lf=1.33,  # Distance between the center of gravity (CG)and its front axle [m]
            lr=3.140 - 1.33,  # Distance between the CGand its rear axle [m]
            hs=0.766731475-0.2,  # Height of the CG of the sprung mass for to the ground [m]
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
            K_varphi=(569 / 3.14 * 180 + 510 / 3.14 * 180) * 4,  # roll stiffness of tire [N-m/rad] /3.14*180
            C_varphi=0,  # Roll damping of the suspension [N-m-s/rad]
        )
    
        self.m = self.vehicle_params["m"]  # Total mass[kg]
        self.mu = self.vehicle_params["mu"]
        self.ms = self.vehicle_params["ms"]  # Sprung mass[kg]
        self.g = self.vehicle_params["g"]
        self.mu_r = self.vehicle_params["mu_r"]
        self.Rw = self.vehicle_params["Rw"]
        self.Iw = self.vehicle_params["Iw"]

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

    def f_xu(self, states, actions, delta_t):
        x, y, phi, v_x, v_y, gamma, varphi, varphi_dot = states #, kappa1, kappa2, kappa3, kappa4
        Q1, delta1, Q2, delta2, Q3, delta3, Q4, delta4,dQ1, ddelta1, dQ2, ddelta2, dQ3, ddelta3, dQ4, ddelta4 = actions

        D = np.array([Q1, delta1, Q2, delta2, Q3, delta3, Q4, delta4]).reshape(8, 1)
        U = np.array([dQ1, ddelta1, dQ2, ddelta2, dQ3, ddelta3, dQ4, ddelta4]).reshape(8, 1)
        X = np.array(states[3:8]).reshape(5, 1)

        state_next = np.empty_like(states)
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

        Lc_matrix = np.zeros((3, 8))
        Lc_matrix[0, 0], Lc_matrix[0, 2], Lc_matrix[0, 4], Lc_matrix[0, 6] = 1, 1, 1, 1

        Lc_matrix[1, 1], Lc_matrix[1, 3], Lc_matrix[1, 5], Lc_matrix[1, 7]= 1, 1, 1, 1

        Lc_matrix[2, 0], Lc_matrix[2, 1], Lc_matrix[2, 2], Lc_matrix[2, 3], \
        Lc_matrix[2, 4], Lc_matrix[2, 5], Lc_matrix[2, 6], Lc_matrix[2, 7] \
            = -self.lw/2, self.lf, self.lw/2, self.lf, \
          -self.lw/2, -self.lr, self.lw/2, -self.lr

        Ec = [1, 1, 1, 1, 1, 1, 1, 1]
        Ec_matrix = np.diag(Ec)

        Mw1 = np.array([[np.cos(delta1), -np.sin(delta1)],
                        [np.sin(delta1), np.cos(delta1)]])
        Mw2 = np.array([[np.cos(delta2), -np.sin(delta2)],
                        [np.sin(delta2), np.cos(delta2)]])
        Mw3 = np.array([[np.cos(delta3), -np.sin(delta3)],
                        [np.sin(delta3), np.cos(delta3)]])
        Mw4 = np.array([[np.cos(delta4), -np.sin(delta4)],
                        [np.sin(delta4), np.cos(delta4)]])

        Mw_matrix = block_diag(Mw1, Mw2, Mw3, Mw4)

        Ew = [1, 1, 1, 1, 1, 1, 1, 1] # delta action bool matrix[0, 0, 0, 0, 0, 0, 0, 0]
        Ew_matrix = np.diag(Ew)

        A1_matrix = np.zeros((8, 5))

        A1_matrix[1, 1], A1_matrix[1, 2] = -self.k_alpha1/v_x, -self.k_alpha1*self.lf/v_x

        A1_matrix[3, 1], A1_matrix[3, 2] = -self.k_alpha2 / v_x, -self.k_alpha2 * self.lf / v_x

        A1_matrix[5, 1], A1_matrix[5, 2] = -self.k_alpha3 / v_x, -self.k_alpha3 * (-self.lr) / v_x

        A1_matrix[7, 1], A1_matrix[7, 2] = -self.k_alpha4 / v_x, -self.k_alpha4 * (-self.lr) / v_x


        B1 = [1/self.Rw, self.k_alpha1, 1/self.Rw, self.k_alpha2,
              1/self.Rw, self.k_alpha3, 1/self.Rw, self.k_alpha4]

        B1_matrix = np.diag(B1)
        dt_matrix = np.zeros((8, 1))
        dt_matrix[0], dt_matrix[2], dt_matrix[4], dt_matrix[
            6] = -1 / 4 * self.m * self.g * self.mu_r, -1 / 4 * self.m * self.g * self.mu_r, -1 / 4 * self.m * self.g * self.mu_r, -1 / 4 * self.m * self.g * self.mu_r
        temp = np.matmul(A1_matrix, X) + np.matmul(B1_matrix, D) + np.matmul(np.matmul(Ew_matrix, B1_matrix), U)+dt_matrix

        X_dot = (np.matmul(A_matrix, X) + np.matmul(np.matmul(np.matmul(np.matmul(
            B_matrix, Lc_matrix), Ec_matrix), Mw_matrix), temp)).squeeze()

        state_next[0] = x + delta_t*(v_x*np.cos(phi)-v_y*np.sin(phi))
        state_next[1] = y + delta_t*(v_y*np.cos(phi)+v_x*np.sin(phi))
        state_next[2] = phi + delta_t * gamma
        state_next[3:8] = states[3:8] + delta_t * X_dot

        # state_next[8] = kappa1+delta_t*(self.Rw*(Q1-self.Rw*self.C_slip1*kappa1)/(v_x*self.Iw)-(1+kappa1)/(self.m*v_x)*(self.C_slip1*kappa1+self.C_slip2*kappa2+self.C_slip3*kappa3+self.C_slip4*kappa4))
        # state_next[9] = kappa2+delta_t*(self.Rw*(Q2-self.Rw*self.C_slip2*kappa2)/(v_x*self.Iw)-(1+kappa2)/(self.m*v_x)*(self.C_slip1*kappa1+self.C_slip2*kappa2+self.C_slip3*kappa3+self.C_slip4*kappa4))
        # state_next[10] = kappa3+delta_t*(self.Rw*(Q3-self.Rw*self.C_slip3*kappa3)/(v_x*self.Iw)-(1+kappa3)/(self.m*v_x)*(self.C_slip1*kappa1+self.C_slip2*kappa2+self.C_slip3*kappa3+self.C_slip4*kappa4))
        # state_next[11] = kappa4+delta_t*(self.Rw*(Q4-self.Rw*self.C_slip4*kappa4)/(v_x*self.Iw)-(1+kappa4)/(self.m*v_x)*(self.C_slip1*kappa1+self.C_slip2*kappa2+self.C_slip3*kappa3+self.C_slip4*kappa4))
        return state_next

class Fourwsdvehicleholisticcontrol(PythBaseEnv):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    def __init__(
        self,
        ref_vx: float = 10,
        pre_horizon: int = 30,
        max_torque: float = 298,
        max_steer: float = 0.5,
        max_delta_torque=10,
        max_delta_str=0.1,
        **kwargs,
    ):
        work_space = kwargs.pop("work_space", None)
        if work_space is None:
            # initial range of [x, y, yaw, vx, vy, yaw rate, roll, roll rate]
            # 用高斯分布去采样
            init_high = np.array([200, 2, 0.1, 12, 0.1, 0.1, 0.1, 5], dtype=np.float32)
            init_low = np.array([0, -2, -0.1, 8, -0.1, -0.1, -0.1, -5], dtype=np.float32)
            work_space = np.stack((init_low, init_high))
        super(Fourwsdvehicleholisticcontrol, self).__init__(work_space=work_space, **kwargs)

        self.vehicle_dynamics = VehicleDynamicsData()
        self.pre_horizon = pre_horizon
        self.ref_vx = ref_vx
        self.ref_traj = Ref_Route(self.ref_vx)
        self.state_dim = self.vehicle_dynamics.vehicle_params["state_dim"]
        ego_obs_dim = 7
        ref_obs_dim = 3
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf] * (ego_obs_dim + ref_obs_dim * pre_horizon)),
            high=np.array([np.inf] * (ego_obs_dim + ref_obs_dim * pre_horizon)),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-max_torque, -max_steer]*4+[-max_delta_torque, -max_delta_str]*4),
            high=np.array([max_torque, max_steer]*4+[max_delta_torque, max_delta_str]*4),
            dtype=np.float32,
        )
        obs_scale_default = [1/100, 1/100, 1/10,
                             1/100, 1/100, 1/10, 1, 1/50]
        self.obs_scale = np.array(kwargs.get('obs_scale', obs_scale_default))

        self.dt = 0.01
        self.max_episode_steps = 300

        self.state = None
        self.ref_x = None
        self.ref_y = None
        self.ref_points = None

        self.info_dict = {
            "state": {"shape": (self.state_dim,), "dtype": np.float32},
            "ref_points": {"shape": (self.pre_horizon + 1, 4), "dtype": np.float32},
            "ref_x": {"shape": (), "dtype": np.float32},
            "ref_y": {"shape": (), "dtype": np.float32},
            "ref": {"shape": (4,), "dtype": np.float32},
        }

        self.seed()

    @property
    def additional_info(self) -> Dict[str, Dict]:
        return self.info_dict

    def reset(
        self,
        init_state: Optional[Sequence] = None,
        ref_x: Optional[float] = None,
        ref_y: Optional[int] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, dict]:

        if init_state is not None:
            state = np.array(init_state, dtype=np.float32)
        else:
            state = self.sample_initial_state()
        # train mode
        state[0] = self.np_random.uniform(
            low=self.init_space[0][0], high=self.init_space[1][0]
        ) # x
        state[3] = self.np_random.uniform(
            low=self.init_space[0][3], high=self.init_space[1][3]
        ) # vx

        # # # run mode
        state[0] = 0
        state[3] = 8
        self.state = state
        self.ref_x, self.ref_y = state[0], state[1]
        traj_points = [[self.ref_x, self.ref_y]]
        for k in range(self.pre_horizon):
            self.ref_x += state[3] * self.dt
            self.ref_y += state[4] * self.dt
            traj_points.append([self.ref_x, self.ref_y])
        self.ref_points = self.ref_traj.find_nearest_point(np.array(traj_points))  # x, y, phi, u

        obs = self.get_obs()
        self.action_last = np.zeros(16)
        return obs, self.info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        action = np.clip(action, self.action_space.low, self.action_space.high)

        reward = self.compute_reward(action)

        self.state = self.vehicle_dynamics.f_xu(self.state, action, self.dt)

        self.ref_x, self.ref_y = self.state[0], self.state[1]

        self.ref_points[:-1] = self.ref_points[1:]
        self.ref_x += self.state[3] * self.dt
        self.ref_y += self.state[4] * self.dt
        traj_points = [[self.ref_x, self.ref_y]]
        new_ref_point = self.ref_traj.find_nearest_point(np.array(traj_points))  # x, y, phi, u
        self.ref_points[-1] = new_ref_point

        obs = self.get_obs()
        self.done = self.judge_done()

        self.action_last = action
        return obs, reward, self.done, self.info

    def get_obs(self) -> np.ndarray:
        ref_x_tf, ref_y_tf, ref_phi_tf, ref_vx_tf = \
            state_error_calculate(
                self.state[0], self.state[1], self.state[2], self.state[3],
                self.ref_points[:, 0], self.ref_points[:, 1], self.ref_points[:, 2], self.ref_points[:, 3]
            )

        # ego_obs: [
        # y_err, phi_err， vx_err,  (of the first reference point)
        # vy, gamma, varphi, varphi_dot, kappa1-4
        # ]

        ego_obs = np.concatenate(
            ([ref_y_tf[0]*self.obs_scale[1], ref_phi_tf[0]*self.obs_scale[2], ref_vx_tf[0]*self.obs_scale[3]],
             self.state[4:8]*self.obs_scale[4:8]
             )) #, self.state[8:12]*self.obs_scale[8:12]
        # ref_obs: [
        # y_err, phi_err, vx_err (of the second to last reference point)
        # ]
        ref_obs = np.stack((ref_y_tf*self.obs_scale[1], ref_phi_tf*self.obs_scale[2], ref_vx_tf*self.obs_scale[3]), 1)[1:].flatten()
        return np.concatenate((ego_obs, ref_obs))

    def compute_reward(self, action: np.ndarray) -> float:
        px, py, phi, vx, vy, gamma, varphi, varphi_dot = self.state
        Q1, delta1, Q2, delta2, Q3, delta3, Q4, delta4, dQ1, ddelta1, dQ2, ddelta2, dQ3, ddelta3, dQ4, ddelta4 = action
        beta = np.arctan(vy/vx)
        ref_x, ref_y, ref_phi, ref_vx = self.ref_points[0]
        I_matrix = np.array([[(self.vehicle_dynamics.k_alpha1+self.vehicle_dynamics.k_alpha2+
                              self.vehicle_dynamics.k_alpha3+self.vehicle_dynamics.k_alpha4)/(self.vehicle_dynamics.m*vx),
                              self.vehicle_dynamics.lf*(self.vehicle_dynamics.k_alpha1+self.vehicle_dynamics.k_alpha2)-
                              self.vehicle_dynamics.lr*(self.vehicle_dynamics.k_alpha3+self.vehicle_dynamics.k_alpha4)/(self.vehicle_dynamics.m*vx**2)],
                             [self.vehicle_dynamics.lf*(self.vehicle_dynamics.k_alpha1+self.vehicle_dynamics.k_alpha2)-
                              self.vehicle_dynamics.lr*(self.vehicle_dynamics.k_alpha3+self.vehicle_dynamics.k_alpha4)/(self.vehicle_dynamics.Izz),
                              self.vehicle_dynamics.lf**2*(self.vehicle_dynamics.k_alpha1+self.vehicle_dynamics.k_alpha2)+
                              self.vehicle_dynamics.lr**2*(self.vehicle_dynamics.k_alpha3+self.vehicle_dynamics.k_alpha4)/(self.vehicle_dynamics.Izz*vx)]])

        k_matrix = np.array([[-self.vehicle_dynamics.k_alpha1/(self.vehicle_dynamics.m*vx),
                              -self.vehicle_dynamics.k_alpha2/(self.vehicle_dynamics.m*vx),
                              -self.vehicle_dynamics.k_alpha3/(self.vehicle_dynamics.m*vx),
                              -self.vehicle_dynamics.k_alpha4/(self.vehicle_dynamics.m*vx)],
                             [-self.vehicle_dynamics.lf*self.vehicle_dynamics.k_alpha1/self.vehicle_dynamics.Izz,
                              -self.vehicle_dynamics.lf*self.vehicle_dynamics.k_alpha2/self.vehicle_dynamics.Izz,
                              self.vehicle_dynamics.lr*self.vehicle_dynamics.k_alpha3/self.vehicle_dynamics.Izz,
                              self.vehicle_dynamics.lr*self.vehicle_dynamics.k_alpha4/self.vehicle_dynamics.Izz]])
        delta_matrix = np.array([[delta1], [delta2], [delta3], [delta4]])

        later_ref = np.matmul(np.matmul(np.linalg.inv(I_matrix), k_matrix), delta_matrix)
        beta_ref = 0#later_ref[0][0]
        gamma_ref = 0#later_ref[1][0]
        C_varphi = 2/(self.vehicle_dynamics.m*self.vehicle_dynamics.g*self.vehicle_dynamics.lw)*\
                   (self.vehicle_dynamics.K_varphi*(1+(self.vehicle_dynamics.ms*self.vehicle_dynamics.hr+
                                                       self.vehicle_dynamics.mu*self.vehicle_dynamics.hu)/
                                                    (self.vehicle_dynamics.ms*self.vehicle_dynamics.hs))-(self.vehicle_dynamics.ms*self.vehicle_dynamics.hr+
                                                       self.vehicle_dynamics.mu*self.vehicle_dynamics.hu)*self.vehicle_dynamics.g)
        C_varphi_dot = 2*C_varphi/(self.vehicle_dynamics.m*self.vehicle_dynamics.g*self.vehicle_dynamics.lw)*\
                       ((1+(self.vehicle_dynamics.ms*self.vehicle_dynamics.hr+self.vehicle_dynamics.mu*self.vehicle_dynamics.hu)/
                                                    (self.vehicle_dynamics.ms*self.vehicle_dynamics.hs)))
        I_rollover = C_varphi*varphi+C_varphi_dot*varphi_dot
        kappa_constant = 0.15#vx/self.vehicle_dynamics.Rw
        r_action_Q = np.sum((action[0:8:2]) ** 2)
        r_action_str = np.sum((action[1:8:2]) ** 2)
        r_action_Qdot = np.sum((action[0:8:2]-self.action_last[0:8:2]) ** 2)
        r_action_strdot = np.sum((action[1:8:2]-self.action_last[1:8:2]) ** 2)
        r_action_deltaQ = dQ1 ** 2 + dQ2 ** 2 + dQ3 ** 2 + dQ4 ** 2
        r_action_deltastr = np.sum((action[9:16:2]) ** 2)
        r_action_deltaQdot = np.sum((action[8:16:2]-self.action_last[8:16:2]) ** 2)
        r_action_deltastrdot = np.sum((action[9:16:2]-self.action_last[9:16:2]) ** 2)
        return -(
                1.8 * ((px - ref_x) ** 2 + (py - ref_y) ** 2)
                + 3.6 * (vx - ref_vx) ** 2
                + 1.0 * angle_normalize(phi - ref_phi) ** 2
                + 0.3 * (gamma - gamma_ref) ** 2
                + 0.5 * (beta - beta_ref) ** 2
                + 0.5 * I_rollover ** 2
                + 1e-8 * r_action_Q
                + 1e-4 * r_action_str
                + 1e-4 * r_action_Qdot
                + 1e-1 * r_action_strdot
                + 1e-8 * r_action_deltaQ
                + 1e-4 * r_action_deltastr
                + 1e-4 * r_action_deltaQdot
                + 1e-1 * r_action_deltastrdot
        )

    def judge_done(self) -> bool:
        done = ((abs(self.state[1]-self.ref_points[0, 1]) > 3)  # delta_y
                + (abs(self.state[3]-self.ref_points[0, 3]) > 3)  # delta_vx
                # + (abs(self.state[4]) > 2)  # delta_vy
                  + (abs(self.state[2]-self.ref_points[0, 2]) > np.pi/2)) # delta phi
        return done

    @property
    def info(self) -> dict:
        return {
            "state": self.state.copy(),
            "ref_points": self.ref_points.copy(),
            "ref_x": self.ref_x,
            "ref_y": self.ref_y,
            "ref": self.ref_points[0].copy()
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
    return Fourwsdvehicleholisticcontrol(**kwargs)
