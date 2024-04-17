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
    def __init__(self):
        self.preview_index = 5
        root_dir = "C:/Users/Troy.Z/Desktop/GOPS/gops/env/env_ocp/resources/cury.csv"
        print(root_dir)
        self.ref_traj = read_path(root_dir)

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
        return np.stack([ref_x, ref_y, ref_heading], axis=1)

class VehicleDynamicsData_4A:
    def __init__(self):
        self.state_dim = 8
        self.m = 4455+218*2+603*2   # Total mass[kg]
        self.ms = 4455.  # Sprung mass[kg]
        self.g = 9.81
        self.Rw = 0.52

        self.lw = 2.07
        self.l12 = 2.633701099999996  # Distance between the center of gravity (CG)and its front axle [m]
        self.l34 = 0.7837291899999954  # Distance between the CGand its rear axle [m]
        self.l56 = 3.596452280000001  # Distance between the hitch point and the center of gravity (CG)[m]
        self.l78 = 4.796434060000003  # Distance between the hitch point and the CG of the semitrailer [m]

        self.hs = 1.19501249  # Height of the CG of the sprung mass for the tractor [m]

        self.Izz = 34678.2  # Yaw moment of inertia of the whole mass[kg m^2]
        self.Ixx = 2309.5  # Roll moment of inertia of the sprung mass[kg m^2]
        self.Iyy = 35427.9
        self.Ixz = 0  # Roll–yaw product of inertia of the sprung mass[kg m^2]

        self.k_alpha1 = 259752/2  # Tire cornering stiffness of the 1st wheel[N/rad]
        self.k_alpha2 = 259752/2  # Tire cornering stiffness of the 1st wheel[N/rad]
        self.k_alpha3 = 259752/2  # Tire cornering stiffness of the rear axle[N/rad]
        self.k_alpha4 = 259752/2  # Tire cornering stiffness of the rear axle[N/rad]
        self.k_alpha5 = 259752/4  # Tire cornering stiffness of the rear axle of the trailer [N/rad]
        self.k_alpha6 = 259752/4  # Tire cornering stiffness of the rear axle of the trailer [N/rad]
        self.k_alpha7 = 259752/4  # Tire cornering stiffness of the rear axle of the trailer [N/rad]
        self.k_alpha8 = 259752/4  # Tire cornering stiffness of the rear axle of the trailer [N/rad]

        self.K_varphi = 22929.936*4+171974.522*4# roll stiffness of tire [N-m/rad]
        self.C_varphi = 0  #Roll damping of the suspension [N-m-s/rad]

    def f_xu(self, states, actions, delta_t):
        v_x, v_y, gamma, varphi, varphi_dot, x, y, psi = states
        Q1, delta1, Q2, delta2, Q3, delta3, Q4, delta4, Q5, delta5, Q6, delta6, Q7, delta7, Q8, delta8 = actions

        D = np.array(actions).reshape(16, 1)
        U = np.zeros((16, 1))

        X = np.array(states[:5]).reshape(5, 1)
        state_next = np.empty_like(states)
        dividend = (self.m*self.Ixx*self.Izz+self.Izz*self.ms**2*self.hs**2-self.m*self.Ixz**2)
        A_matrix = np.zeros((5, 5))

        A_matrix[1, 2], A_matrix[1, 3], A_matrix[1, 4] = -v_x, -self.ms*self.hs*self.Izz*(self.K_varphi-self.ms*self.g*self.hs)/dividend, \
        -self.ms*self.hs*self.Izz*self.C_varphi/dividend

        A_matrix[2, 3], A_matrix[2, 4]= \
        -self.m*self.Ixz*(self.K_varphi-self.ms*self.g*self.hs)/dividend, \
        -self.m*self.Ixz*self.C_varphi/dividend

        A_matrix[3, 4] = 1

        A_matrix[4, 3], A_matrix[4, 4] = \
        -self.m*self.Izz*(self.K_varphi-self.ms*self.g*self.hs)/dividend, \
        -self.m*self.Izz*self.C_varphi/dividend

        B_matrix = np.zeros((5, 3))
        B_matrix[0, 0] = 1/self.m
        B_matrix[1, 1], B_matrix[1, 2] = \
        (self.Ixx*self.Izz-self.Ixz**2)/dividend, \
        self.Ixz*self.ms*self.hs/dividend

        B_matrix[2, 1], B_matrix[2, 2] = \
        -self.Ixz*self.ms*self.hs/dividend, \
        (self.ms**2*self.hs**2+self.m*self.Ixx)/dividend

        B_matrix[4, 1], B_matrix[4, 2] = \
        -self.Izz*self.ms*self.hs/dividend, \
        (self.m*self.Ixz)/dividend

        Lc_matrix = np.zeros((3, 16))
        Lc_matrix[0, 0], Lc_matrix[0, 2], Lc_matrix[0, 4], Lc_matrix[0, 6],\
        Lc_matrix[0, 8], Lc_matrix[0, 10], Lc_matrix[0, 12], Lc_matrix[0, 14] = 1, 1, 1, 1, 1, 1, 1, 1

        Lc_matrix[1, 1], Lc_matrix[1, 3], Lc_matrix[1, 5], Lc_matrix[1, 7], \
        Lc_matrix[1, 9], Lc_matrix[1, 11], Lc_matrix[1, 13], Lc_matrix[1, 15] = 1, 1, 1, 1, 1, 1, 1, 1

        Lc_matrix[2, 0], Lc_matrix[2, 1], Lc_matrix[2, 2], Lc_matrix[2, 3], \
        Lc_matrix[2, 4], Lc_matrix[2, 5], Lc_matrix[2, 6], Lc_matrix[2, 7], \
        Lc_matrix[2, 8], Lc_matrix[2, 9], Lc_matrix[2, 10], Lc_matrix[2, 11], \
        Lc_matrix[2, 12], Lc_matrix[2, 13], Lc_matrix[2, 14], Lc_matrix[2, 15] \
        = -self.lw/2, self.l12, self.lw/2, self.l12, \
          -self.lw/2, self.l34, self.lw/2, self.l34, \
          -self.lw/2, -self.l56, self.lw/2, -self.l56, \
          -self.lw/2, -self.l78, self.lw/2, -self.l78

        Ec = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        Ec_matrix = np.diag(Ec)

        Mw1 = np.array([[np.cos(delta1), -np.sin(delta1)],
                        [np.sin(delta1), np.cos(delta1)]])
        Mw2 = np.array([[np.cos(delta2), -np.sin(delta2)],
                        [np.sin(delta2), np.cos(delta2)]])
        Mw3 = np.array([[np.cos(delta3), -np.sin(delta3)],
                        [np.sin(delta3), np.cos(delta3)]])
        Mw4 = np.array([[np.cos(delta4), -np.sin(delta4)],
                        [np.sin(delta4), np.cos(delta4)]])
        Mw5 = np.array([[np.cos(delta5), -np.sin(delta5)],
                        [np.sin(delta5), np.cos(delta5)]])
        Mw6 = np.array([[np.cos(delta6), -np.sin(delta6)],
                        [np.sin(delta6), np.cos(delta6)]])
        Mw7 = np.array([[np.cos(delta7), -np.sin(delta7)],
                        [np.sin(delta7), np.cos(delta7)]])
        Mw8 = np.array([[np.cos(delta8), -np.sin(delta8)],
                        [np.sin(delta8), np.cos(delta8)]])

        Mw_matrix = block_diag(Mw1, Mw2, Mw3, Mw4, Mw5, Mw6, Mw7, Mw8)

        Ew = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        Ew_matrix = np.diag(Ew)

        A1_matrix = np.zeros((16, 5))

        A1_matrix[1, 1], A1_matrix[1, 2] = -self.k_alpha1/v_x, -self.k_alpha1*self.l12/v_x

        A1_matrix[3, 1], A1_matrix[3, 2] = -self.k_alpha2 / v_x, -self.k_alpha2 * self.l12 / v_x

        A1_matrix[5, 1], A1_matrix[5, 2] = -self.k_alpha3 / v_x, -self.k_alpha3 * self.l34 / v_x

        A1_matrix[7, 1], A1_matrix[7, 2] = -self.k_alpha4 / v_x, -self.k_alpha4 * self.l34 / v_x

        A1_matrix[9, 1], A1_matrix[9, 2] = -self.k_alpha5 / v_x, -self.k_alpha5 * (-self.l56) / v_x

        A1_matrix[11, 1], A1_matrix[11, 2] = -self.k_alpha6 / v_x, -self.k_alpha6 * (-self.l56) / v_x

        A1_matrix[13, 1], A1_matrix[13, 2] = -self.k_alpha7 / v_x, -self.k_alpha7 * (-self.l78) / v_x

        A1_matrix[15, 1], A1_matrix[15, 2] = -self.k_alpha8 / v_x, -self.k_alpha8 * (-self.l78) / v_x

        B1 = [1/self.Rw, self.k_alpha1, 1/self.Rw, self.k_alpha2,
              1/self.Rw, self.k_alpha3, 1/self.Rw, self.k_alpha4,
              1/self.Rw, self.k_alpha5, 1/self.Rw, self.k_alpha6,
              1/self.Rw, self.k_alpha7, 1/self.Rw, self.k_alpha8]

        B1_matrix = np.diag(B1)

        temp = np.matmul(A1_matrix, X) + np.matmul(B1_matrix, D) + np.matmul(np.matmul(Ew_matrix, B1_matrix), U)

        X_dot = (np.matmul(A_matrix, X) + np.matmul(np.matmul(np.matmul(np.matmul(
            B_matrix, Lc_matrix), Ec_matrix), Mw_matrix), temp)).squeeze()

        state_next[:5] = states[:5] + delta_t * X_dot

        state_next[5] = x + delta_t*(v_x*np.cos(psi)-v_y*np.sin(psi))
        state_next[6] = y + delta_t*(v_y*np.cos(psi)+v_x*np.sin(psi))
        state_next[7] = psi + delta_t * gamma
        return state_next

class VehicleDynamicsData_2A:
    def __init__(self):
        self.state_dim = 8
        self.m = 4455+362+679   # Total mass[kg]
        self.ms = 4455.  # Sprung mass[kg]
        self.g = 9.81
        self.Rw = 0.51

        self.lw = 2.03
        self.l12 = 1.250  # Distance between the center of gravity (CG)and its front axle [m]
        self.l34 = 5.000-1.250 # Distance between the CGand its rear axle [m]

        self.hs = 1.16407072  # Height of the CG of the sprung mass for the tractor [m]

        self.Izz = 34802.6  # Yaw moment of inertia of the whole mass[kg m^2]
        self.Ixx = 2283.9  # Roll moment of inertia of the sprung mass[kg m^2]
        self.Iyy = 35402.8
        self.Ixz = 1626  # Roll–yaw product of inertia of the sprung mass[kg m^2]

        self.k_alpha1 = 259752/2  # Tire cornering stiffness of the 1st wheel[N/rad]
        self.k_alpha2 = 259752/2  # Tire cornering stiffness of the 1st wheel[N/rad]
        self.k_alpha3 = 259752/2  # Tire cornering stiffness of the rear axle[N/rad]
        self.k_alpha4 = 259752/2  # Tire cornering stiffness of the rear axle[N/rad]

        self.K_varphi = (8500/3.14*180+1500/3.14*180)*4# roll stiffness of tire [N-m/rad] /3.14*180
        self.C_varphi = 0  #Roll damping of the suspension [N-m-s/rad]

    def f_xu(self, states, actions, delta_t):
        v_x, v_y, gamma, varphi, varphi_dot, x, y, psi = states
        Q1, delta1, Q2, delta2, Q3, delta3, Q4, delta4 = actions

        D = np.array(actions).reshape(8, 1)
        U = np.zeros((8, 1))

        X = np.array(states[:5]).reshape(5, 1)
        state_next = np.empty_like(states)
        dividend = (self.m * self.Ixx * self.Izz + self.Izz * self.ms ** 2 * self.hs ** 2 - self.m * self.Ixz ** 2)
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
            -self.Ixz * self.ms * self.hs / dividend, \
            (self.ms ** 2 * self.hs ** 2 + self.m * self.Ixx) / dividend

        B_matrix[4, 1], B_matrix[4, 2] = \
            -self.Izz * self.ms * self.hs / dividend, \
            (self.m * self.Ixz) / dividend

        Lc_matrix = np.zeros((3, 8))
        Lc_matrix[0, 0], Lc_matrix[0, 2], Lc_matrix[0, 4], Lc_matrix[0, 6] = 1, 1, 1, 1

        Lc_matrix[1, 1], Lc_matrix[1, 3], Lc_matrix[1, 5], Lc_matrix[1, 7]= 1, 1, 1, 1

        Lc_matrix[2, 0], Lc_matrix[2, 1], Lc_matrix[2, 2], Lc_matrix[2, 3], \
        Lc_matrix[2, 4], Lc_matrix[2, 5], Lc_matrix[2, 6], Lc_matrix[2, 7] \
            = -self.lw/2, self.l12, self.lw/2, self.l12, \
          -self.lw/2, -self.l34, self.lw/2, -self.l34

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

        Ew = [0, 0, 0, 0, 0, 0, 0, 0]
        Ew_matrix = np.diag(Ew)

        A1_matrix = np.zeros((8, 5))

        A1_matrix[1, 1], A1_matrix[1, 2] = -self.k_alpha1/v_x, -self.k_alpha1*self.l12/v_x

        A1_matrix[3, 1], A1_matrix[3, 2] = -self.k_alpha2 / v_x, -self.k_alpha2 * self.l12 / v_x

        A1_matrix[5, 1], A1_matrix[5, 2] = -self.k_alpha3 / v_x, -self.k_alpha3 * (-self.l34) / v_x

        A1_matrix[7, 1], A1_matrix[7, 2] = -self.k_alpha4 / v_x, -self.k_alpha4 * (-self.l34) / v_x


        B1 = [1/self.Rw, self.k_alpha1, 1/self.Rw, self.k_alpha2,
              1/self.Rw, self.k_alpha3, 1/self.Rw, self.k_alpha4]

        B1_matrix = np.diag(B1)

        temp = np.matmul(A1_matrix, X) + np.matmul(B1_matrix, D) + np.matmul(np.matmul(Ew_matrix, B1_matrix), U)

        X_dot = (np.matmul(A_matrix, X) + np.matmul(np.matmul(np.matmul(np.matmul(
            B_matrix, Lc_matrix), Ec_matrix), Mw_matrix), temp)).squeeze()

        state_next[:5] = states[:5] + delta_t * X_dot

        state_next[5] = x + delta_t*(v_x*np.cos(psi)-v_y*np.sin(psi))
        state_next[6] = y + delta_t*(v_y*np.cos(psi)+v_x*np.sin(psi))
        state_next[7] = psi + delta_t * gamma
        return state_next

class ReconfigurableVehicle(PythBaseEnv):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    def __init__(
        self,
        pre_horizon: int = 30,
        max_torque: float= -100,
        max_steer: float = 0.5,
        **kwargs,
    ):
        work_space = kwargs.pop("work_space", None)
        if work_space is None:
            # initial range of [vx, vy, yaw rate, roll, roll rate, x, y, yaw]
            # 用高斯分布去采样
            init_high = np.array([30, 10, 0.5, 0.1, 0.5, 100, 10, 1], dtype=np.float32)
            init_low = np.array([0, -10, -0.5, -0.1, -0.5, 0, -10, -1], dtype=np.float32)
            work_space = np.stack((init_low, init_high))
        super(ReconfigurableVehicle, self).__init__(work_space=work_space, **kwargs)

        self.vehicle_dynamics = VehicleDynamicsData_4A()
        self.ref_traj = Ref_Route()
        self.state_dim = 8
        self.pre_horizon = pre_horizon
        ego_obs_dim = 7
        ref_obs_dim = 2
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf] * (ego_obs_dim + ref_obs_dim * pre_horizon*2)),
            high=np.array([np.inf] * (ego_obs_dim + ref_obs_dim * pre_horizon*2)),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.array([max_torque, -max_steer]*8),
            high=np.array([0, max_steer]*8),
            dtype=np.float32,
        )
        obs_scale_default = [1, 1, 1, 1,
                             1, 1, 1, 1,
                             1/10, 1/10, 1/100, 1/100, 1/100, 1/100, 1/10]
        self.obs_scale = np.array(kwargs.get('obs_scale', obs_scale_default))

        self.dt = 0.0005
        self.max_episode_steps = 200

        self.state = None
        self.ref_x = None
        self.ref_y = None
        self.ref_points = None

        self.info_dict = {
            "state": {"shape": (self.state_dim,), "dtype": np.float32},
            "ref_points": {"shape": (self.pre_horizon + 1, 3), "dtype": np.float32},
            "ref_x": {"shape": (), "dtype": np.float32},
            "ref_y": {"shape": (), "dtype": np.float32},
            "ref": {"shape": (3,), "dtype": np.float32},
            "target_speed": {"shape": (), "dtype": np.float32},
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

        state[12] = state[11] - self.vehicle_dynamics.b * np.sin(state[8]) - self.vehicle_dynamics.e * np.sin(
            state[9])  # posy_trailer

        # 训练用，run的时候注释掉
        state[13] = self.np_random.uniform(
            low=self.init_space[0][13], high=self.init_space[1][13]
        )
        # state[13] = 160
        state[14] = state[13] - self.vehicle_dynamics.b * np.cos(state[8]) - self.vehicle_dynamics.e * np.cos(
            state[9])  # posx_trailer
        self.state = state
        self.ref_x, self.ref_y = state[self.state_dim - 2], state[self.state_dim - 4]
        traj_points = [[self.ref_x, self.ref_y]]
        for k in range(self.pre_horizon):
            self.ref_x += self.target_speed * self.dt
            self.ref_y += state[10] * self.dt
            traj_points.append([self.ref_x, self.ref_y])
        self.ref_points = self.ref_traj.find_nearest_point(np.array(traj_points))  # x, y, phi, u
        
        self.ref_x2, self.ref_y2 = state[self.state_dim - 1], state[self.state_dim - 3]
        traj_points_2 = [[self.ref_x2, self.ref_y2]]
        for k in range(self.pre_horizon):
            self.ref_x2 += self.target_speed * self.dt
            self.ref_y2 += state[10] * self.dt
            traj_points_2.append([self.ref_x2, self.ref_y2])
        self.ref_points_2 = self.ref_traj.find_nearest_point(np.array(traj_points_2))  # x, y, phi, u

        obs = self.get_obs()
        self.action_last = 0
        return obs, self.info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        action = np.clip(action, self.action_space.low, self.action_space.high)

        reward = self.compute_reward(action)

        self.state = self.vehicle_dynamics.f_xu(self.state, action, self.dt)

        self.ref_x, self.ref_y = self.state[self.state_dim - 2], self.state[self.state_dim - 4]

        self.ref_points[:-1] = self.ref_points[1:]
        self.ref_x += self.target_speed * self.dt
        self.ref_y += self.state[10] * self.dt
        traj_points = [[self.ref_x, self.ref_y]]
        new_ref_point = self.ref_traj.find_nearest_point(np.array(traj_points))  # x, y, phi, u
        self.ref_points[-1] = new_ref_point

        self.ref_points_2[:-1] = self.ref_points_2[1:]
        self.ref_x2 += self.target_speed * self.dt
        self.ref_y2 += self.state[10] * self.dt
        traj_points_2 = [[self.ref_x2, self.ref_y2]]
        new_ref_point_2 = self.ref_traj.find_nearest_point(np.array(traj_points_2))  # x, y, phi, u
        self.ref_points[-1] = new_ref_point_2

        obs = self.get_obs()
        self.done = self.judge_done()

        self.action_last = action[0]
        return obs, reward, self.done, self.info

    def get_obs(self) -> np.ndarray:
        ref_x_tf, ref_y_tf, ref_phi_tf = \
            state_error_calculate(
                self.state[13], self.state[11], self.state[8],
                self.ref_points[:, 0], self.ref_points[:, 1], self.ref_points[:, 2],
            )

        ref_x2_tf, ref_y2_tf, ref_phi2_tf = \
            state_error_calculate(
                self.state[14], self.state[12], self.state[9],
                self.ref_points_2[:, 0], self.ref_points_2[:, 1], self.ref_points_2[:, 2],
            )

        # ego_obs: [
        # delta_x, delta_y, delta_psi,delta_x2, delta_y2, delta_psi2 (of the first reference point)
        # v, w, varphi (of ego vehicle, including tractor and trailer)
        # ]

        ego_obs = np.concatenate(
            (self.state[0:8], [ref_phi_tf[0]*self.obs_scale[8], ref_phi2_tf[0]*self.obs_scale[9]], self.state[10:11]*self.obs_scale[10],
             [ref_y_tf[0]*self.obs_scale[11], ref_y2_tf[0]*self.obs_scale[12]]))
        # ref_obs: [
        # delta_x, delta_y, delta_psi (of the second to last reference point)
        # ]
        ref_obs = np.stack((ref_y_tf*self.obs_scale[13], ref_phi_tf*self.obs_scale[14], ref_y2_tf*self.obs_scale[13], ref_phi2_tf*self.obs_scale[14]), 1)[1:].flatten()
        return np.concatenate((ego_obs, ref_obs))

    def compute_reward(self, action: np.ndarray) -> float:
        beta1, psi1_dot, varphi1, varphi1_dot, \
        beta2, psi2_dot, varphi2, varphi2_dot, \
        psi1, psi2, vy1, py1, py2, px1, px2 = self.state

        ref_x, ref_y, ref_psi = self.ref_points[0]
        steer = action[0]
        return -(
            1 * ((px1 - ref_x) ** 2 + 0.04 * (py1 - ref_y) ** 2)
            + 0.9 * vy1 ** 2
            + 0.8 * angle_normalize(psi1 - ref_psi) ** 2
            + 0.5 * psi1_dot ** 2
            + 0.5 * beta1 ** 2
            + 0.5 * varphi1 ** 2
            + 0.5 * varphi1_dot ** 2
            + 0.4 * steer ** 2
            + 2.0 * (steer - self.action_last) ** 2
        )

    def judge_done(self) -> bool:
        done = ((abs(self.state[11]-self.ref_points[0, 1]) > 3)  # delta_y1
                + (abs(self.state[10]) > 2)  # delta_psi1
                  + (abs(self.state[8]-self.ref_points[0, 2]) > np.pi/2)  # delta_psi1
                  + (abs(self.state[12]-self.ref_points_2[0, 1]) > 3) # delta_y2
                  + (abs(self.state[9]-self.ref_points_2[0, 2]) > np.pi / 2))  # delta_psi2
        return done

    @property
    def info(self) -> dict:
        return {
            "state": self.state.copy(),
            "ref_points": self.ref_points.copy(),
            "ref_points_2": self.ref_points_2.copy(),
            "ref_x": self.ref_x,
            "ref_y": self.ref_y,
            "ref_x2": self.ref_x2,
            "ref_y2": self.ref_y2,
            "ref": self.ref_points[0].copy(),
            "ref2": self.ref_points_2[0].copy(),
            "target_speed": self.target_speed,
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
    return ReconfigurableVehicle(**kwargs)
