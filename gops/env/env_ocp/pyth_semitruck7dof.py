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
        self.I1yy = 35402
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

        X_dot = (np.matmul(np.matmul(np.linalg.inv(M_matrix), A_matrix), states[:self.state_dim - 2]) + np.matmul(
            np.linalg.inv(M_matrix), np.matmul(B_matrix, actions))).squeeze()
        state_next[:12] = states[:12] + delta_t * X_dot[:12]
        state_next[12] = state_next[11] - self.b * np.sin(state_next[8]) - self.e * np.sin(
            state_next[9])  # posy_trailer
        state_next[13] = states[13] + delta_t * self.v_x
        state_next[14] = state_next[13] - self.b * np.cos(states[8]) - self.e * np.cos(
            states[9])  # posx_trailer
        return state_next


class PythSemitruck7dof(PythBaseEnv):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    def __init__(
        self,
        pre_horizon: int = 30,
        max_steer: float = 0.5,
        **kwargs,
    ):
        work_space = kwargs.pop("work_space", None)
        if work_space is None:
            # initial range of [delta_x, delta_y, delta_phi, delta_u, v, w]
            init_high = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                  np.pi / 6, np.pi / 6, 0.1, 2, 2, 280, 280], dtype=np.float32)
            init_low = np.array([-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1,
                                  -np.pi / 6, -np.pi / 6, -0.1, -2, -2, 100, 100], dtype=np.float32)
            work_space = np.stack((init_low, init_high))
        super(PythSemitruck7dof, self).__init__(work_space=work_space, **kwargs)

        self.vehicle_dynamics = VehicleDynamicsData()
        self.target_speed = self.vehicle_dynamics.v_x
        self.ref_traj = Ref_Route()
        self.state_dim = 15
        self.pre_horizon = pre_horizon
        ego_obs_dim = 15
        ref_obs_dim = 3
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf] * (ego_obs_dim + ref_obs_dim * pre_horizon*2)),
            high=np.array([np.inf] * (ego_obs_dim + ref_obs_dim * pre_horizon*2)),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-max_steer]),
            high=np.array([max_steer]),
            dtype=np.float32,
        )
        self.dt = 0.01
        self.max_episode_steps = 200

        self.state = None
        self.ref_x = None
        self.ref_y = None
        self.ref_x2 = None
        self.ref_y2 = None
        self.ref_points = None
        self.ref_points_2 = None


        # obs_scale_default = [1, 1, 1, 1,
        #                      1, 1, 1, 1,
        #                      1 / 10, 1 / 10, 1 / 100, 1 / 100, 1 / 100, 1 / (10 * kwargs["pre_horizon"])]
        # self.obs_scale = np.array(kwargs.get('obs_scale', obs_scale_default))
        self.info_dict = {
            "state": {"shape": (self.state_dim,), "dtype": np.float32},
            "ref_points": {"shape": (self.pre_horizon + 1, 3), "dtype": np.float32},
            "ref_points_2": {"shape": (self.pre_horizon + 1, 3), "dtype": np.float32},
            "ref_x": {"shape": (), "dtype": np.float32},
            "ref_y": {"shape": (), "dtype": np.float32},
            "ref_x2": {"shape": (), "dtype": np.float32},
            "ref_y2": {"shape": (), "dtype": np.float32},
            "ref": {"shape": (3,), "dtype": np.float32},
            "ref2": {"shape": (3,), "dtype": np.float32},
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
        self.done = self.judge_done(obs)
        # if self.done:
        #     reward = reward - 100
        # self.action_last = action
        return obs, reward, self.done, self.info

    def get_obs(self) -> np.ndarray:
        ref_x_tf, ref_y_tf, ref_phi_tf = \
            ego_vehicle_coordinate_transform(
                self.state[13], self.state[11], self.state[8],
                self.ref_points[:, 0], self.ref_points[:, 1], self.ref_points[:, 2],
            )

        ref_x2_tf, ref_y2_tf, ref_phi2_tf = \
            ego_vehicle_coordinate_transform(
                self.state[14], self.state[12], self.state[9],
                self.ref_points_2[:, 0], self.ref_points_2[:, 1], self.ref_points_2[:, 2],
            )

        # ego_obs: [
        # delta_x, delta_y, delta_psi,delta_x2, delta_y2, delta_psi2 (of the first reference point)
        # v, w, varphi (of ego vehicle, including tractor and trailer)
        # ]
        ego_obs = np.concatenate(
            (self.state[0:8], [ref_phi_tf[0], ref_phi2_tf[0]], self.state[10:11],
             [ref_y_tf[0], ref_y2_tf[0], ref_x_tf[0], ref_x2_tf[0]]))
        # ref_obs: [
        # delta_x, delta_y, delta_psi (of the second to last reference point)
        # ]
        ref_obs = np.stack((ref_x_tf, ref_y_tf, ref_phi_tf, ref_x2_tf, ref_y2_tf, ref_phi2_tf), 1)[1:].flatten()
        return np.concatenate((ego_obs, ref_obs))

    def compute_reward(self, action: np.ndarray) -> float:
        beta1, psi1_dot, varphi1, varphi1_dot, \
        beta2, psi2_dot, varphi2, varphi2_dot, \
        psi1, psi2, vy1, py1, py2, px1, px2 = self.state

        ref_x, ref_y, ref_psi = self.ref_points[0]
        steer = action
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

    def judge_done(self, obs) -> bool:
        done = ((abs(obs[11]) > 3)  # delta_y1
                  + (abs(obs[8]) > np.pi/2)  # delta_psi1
                  + (abs(obs[12]) > 3) # delta_y2
                  + (abs(obs[9]) > np.pi / 2))  # delta_psi2
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
    make env `pyth_SimuSemiTruck9dof`
    """
    return PythSemitruck7dof(**kwargs)
