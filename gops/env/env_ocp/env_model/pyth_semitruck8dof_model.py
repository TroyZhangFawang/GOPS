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
import pandas as pd
import torch
import gym
from typing import Dict, Optional, Tuple, Union
import numpy as np
from gops.env.env_ocp.pyth_semitruck8dof import VehicleDynamicsData
from gops.env.env_ocp.env_model.pyth_base_model import PythBaseModel
from gops.utils.math_utils import angle_normalize
from gops.utils.gops_typing import InfoDict


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
        root_dir = current_dir + "/../resources/u_turn.csv"
        self.ref_traj = read_path(root_dir)
        # 将NumPy数组转换为PyTorch张量
        self.ref_traj_tensor = torch.tensor(self.ref_traj, dtype=torch.float32)
        self.ref_vx = ref_vx
    def find_nearst_point(self, traj_points):
        # 计算两组点之间的距离
        traj_points_expanded = traj_points.unsqueeze(1)  # 在第二维上扩展以便能够逐元素相减
        distances = torch.norm(traj_points_expanded - self.ref_traj_tensor, dim=2)

        # 找到每个轨迹点在参考轨迹上的最近点的索引
        nearest_point_indices = torch.argmin(distances, dim=1)+self.preview_index
        ref_x = self.ref_traj_tensor[nearest_point_indices][:, 0]
        ref_y = self.ref_traj_tensor[nearest_point_indices][:, 1]
        ref_heading = torch.atan2((ref_y - self.ref_traj_tensor[nearest_point_indices - 1][:, 1]), (ref_x - self.ref_traj_tensor[nearest_point_indices - 1][:, 0]))
        ref_vx = torch.zeros_like(ref_x).add(self.ref_vx)
        return torch.stack([ref_x, ref_y, ref_heading, ref_vx], 1)

class VehicleDynamicsModel(VehicleDynamicsData):
    def __init__(self):
        DynamicsData = VehicleDynamicsData()
        self.state_dim = DynamicsData.vehicle_params["state_dim"]
        self.m1 = DynamicsData.vehicle_params["m1"]  # Total mass of the tractor [kg]
        self.m1s = DynamicsData.vehicle_params["m1s"]  # Sprung mass of the tractor [kg]
        self.m2 = DynamicsData.vehicle_params["m2"] # Total mass of the semitrailer [kg]
        self.m2s = DynamicsData.vehicle_params["m2s"]  # Sprung mass of the semitrailer [kg]
        self.gravity = DynamicsData.vehicle_params["gravity"]
        self.a = DynamicsData.vehicle_params["a"]  # Distance between the center of gravity (CG) of the tractor and its front axle [m]
        self.b = DynamicsData.vehicle_params["b"] # Distance between the CG of the tractor and its rear axle [m]
        self.c = DynamicsData.vehicle_params["c"]  # Distance between the hitch point and the center of gravity (CG) of the tractor [m]
        self.e = DynamicsData.vehicle_params["e"]  # Distance between the hitch point and the CG of the semitrailer [m]
        self.d = DynamicsData.vehicle_params["d"]  # Distance between the rear axle and the CG of the semitrailer [m]

        self.h1s = DynamicsData.vehicle_params["h1s"] # Height of the CG of the sprung mass for the tractor [m]
        self.h2s = DynamicsData.vehicle_params["h2s"]  # Height of the CG of the sprung mass for the semitrailer [m]
        self.hh = DynamicsData.vehicle_params["hh"] # Height of hitch point to the roll center of the sprung mass for the tractor [m]

        self.I1xx = DynamicsData.vehicle_params["I1xx"]# Roll moment of inertia of the sprung mass of the tractor [kg m^2]
        self.I1yy = DynamicsData.vehicle_params["I1yy"]
        self.I1zz = DynamicsData.vehicle_params["I1zz"]  # Yaw moment of inertia of the whole mass of the tractor [kg m^2]
        self.I1xz = DynamicsData.vehicle_params["I1xz"] # Roll–yaw product of inertia of the sprung mass of the tractor [kg m^2]

        self.I2zz = DynamicsData.vehicle_params["I2zz"]  # Yaw moment of inertia of the whole mass of the semitrailer [kg m^2]
        self.I2xx = DynamicsData.vehicle_params["I2xx"]  # Roll moment of inertia of the sprung mass of the semitrailer [kg m^2]
        self.I2xz =DynamicsData.vehicle_params["I2xz"] # Roll–yaw product of inertia of the sprung mass of the semitrailer [kg m^2]
        self.I2yy = DynamicsData.vehicle_params["I2yy"]

        self.k1 = DynamicsData.vehicle_params["k1"] # Tire cornering stiffness of the front axle of the tractor [N/rad]
        self.k2 = DynamicsData.vehicle_params["k2"]  # Tire cornering stiffness of the rear axle of the tractor [N/rad]
        self.k3 = DynamicsData.vehicle_params["k3"] # Tire cornering stiffness of the rear axle of the trailer [N/rad]
        self.k_varphi1 =DynamicsData.vehicle_params["k_varphi1"]  # roll stiffness of tire (front)[N/m]
        self.k_varphi2 = DynamicsData.vehicle_params["k_varphi2"] # roll stiffness of tire (rear)[N/m]
        self.k12 = DynamicsData.vehicle_params["k12"]  # Roll stiffness of the articulation joint between the tractor and semitrailer [N m/rad]
        self.c_varphi1 = DynamicsData.vehicle_params["c_varphi1"] # Roll damping of the tractor's suspension [N-s/m]
        self.c_varphi2 = DynamicsData.vehicle_params["c_varphi2"] # Roll damping of the semitrailer's suspension [N-s/m]

    def f_xu(self, state, action, delta_t):
        self.batch_size = len(state[:, 0])
        x_1, y_1, phi_1, v_x1, x_2, y_2, phi_2, v_x2, \
        v_y1, gamma_1, varphi_1, varphi_1dot, \
        v_y2, gamma_2, varphi_2, varphi_2dot = state[:, 0], state[:, 1],state[:, 2], state[0, 3],\
            state[:, 4], state[:, 5],state[:, 6], state[:, 7], \
            state[:, 8], state[:, 9], state[:, 10], state[:, 11],\
           state[:, 12], state[:, 13],state[:, 14], state[:, 15]

        state_next = torch.zeros_like(state)
        X = torch.hstack([state[:, 3:4], state[:, 8:12], state[:, 7:8], state[:, 12:self.state_dim]])
    
        M_matrix = torch.zeros((self.state_dim - 6, self.state_dim - 6))
        self.M00 = -self.m1
        self.M02 = self.m1s * self.h1s * varphi_1
        self.M05 = -self.m2
        self.M07 = self.m2s * self.h2s * varphi_2

        self.M11 = self.m1
        self.M14 = -self.m1s * self.h1s
        self.M16 = self.m2
        self.M19 = -self.m2s * self.h2s

        self.M20 = self.m1s * self.h1s * varphi_1
        self.M22 = -self.I1zz
        self.M24 = self.I1xz
        self.M26 = self.m2 * self.c
        self.M29 = -self.m2s * self.h2s * self.c

        self.M33 = 1

        self.M41 = self.m1s * self.h1s
        self.M42 = -self.I1xz
        self.M44 = self.I1xx + self.m1s * self.h1s ** 2
        self.M46 = -self.m2 * self.hh
        self.M49 = self.m2s * self.hh * self.h2s

        self.M50 = -torch.cos(phi_2 - phi_1)
        self.M51 = torch.sin(phi_2 - phi_1)
        self.M52 = -torch.sin(phi_2 - phi_1) * self.c
        self.M55 = 1

        self.M60 = -torch.sin(phi_2 - phi_1)
        self.M61 = -torch.cos(phi_2 - phi_1)
        self.M62 = torch.cos(phi_2 - phi_1) * self.c
        self.M66 = 1
        self.M67 = self.c

        self.M75 = -self.m2s * self.h2s * varphi_2
        self.M76 = -self.e * self.m2
        self.M77 = self.I2zz
        self.M79 = (self.m2s * self.h2s - self.I2xz)

        self.M88 = 1

        self.M96 = self.m2s * self.h2s+self.hh*self.m2
        self.M97 = -self.I2xz
        self.M99 = self.I2xx + self.m2s * self.h2s ** 2-self.m2s*self.h2s*self.hh

        M_matrix[0, 0] = self.M00
        M_matrix[0, 2] = self.M02
        M_matrix[0, 5] = self.M05
        M_matrix[0, 7] = self.M07

        M_matrix[1, 1], M_matrix[1, 4], M_matrix[1, 6], M_matrix[1, 9] = self.M11, self.M14, self.M16, self.M19

        M_matrix[2, 0], M_matrix[2, 2], M_matrix[2, 4], M_matrix[2, 6], M_matrix[2, 9] = \
            self.M20, self.M22, self.M24, self.M26, self.M29

        M_matrix[3, 3] = self.M33

        M_matrix[4, 1], M_matrix[4, 2], M_matrix[4, 4], M_matrix[4, 6], M_matrix[4, 9] = \
            self.M41, self.M42, self.M44, self.M46, self.M49

        M_matrix[5, 0], M_matrix[5, 1], M_matrix[5, 2], M_matrix[5, ] = \
            self.M50, self.M51, self.M52, self.M55

        M_matrix[6, 0], M_matrix[6, 1], M_matrix[6, 2], M_matrix[6, 6], M_matrix[6, 7] = \
            self.M60, self.M61, self.M62, self.M66, self.M67

        M_matrix[7, 5], M_matrix[7, 6], M_matrix[7, 7], M_matrix[7, 9] = self.M75, self.M76, self.M77, self.M79

        M_matrix[8, 8] = self.M88

        M_matrix[9, 6], M_matrix[9, 7], M_matrix[9, 9] = self.M96, self.M97, self.M99

        A_matrix = torch.zeros((self.state_dim - 6, self.state_dim - 6))
        self.A02 = -self.m1 * v_y1
        self.A04 = -self.m1s * self.h1s * 2 * gamma_1
        self.A07 = -self.m2 * v_y2
        self.A09 = 2 * self.m2s * self.h2s * gamma_2

        self.A10 = -self.m1 * gamma_1
        self.A11 = (self.k1 + self.k2) / v_x1
        self.A12 = (self.a * self.k1 - self.b * self.k2) / v_x1
        self.A13 = -self.m1s * self.h1s * gamma_1 ** 2
        self.A15 = -self.m2 * gamma_2
        self.A16 = self.k3 / v_x2
        self.A17 = -self.k3 * self.d / v_x2
        self.A18 = -self.m2s * self.h2s * gamma_2 ** 2

        self.A21 = (self.a * self.k1 - self.b * self.k2) / v_x1
        self.A22 = (self.a ** 2 * self.k1 + self.b ** 2 * self.k2) / v_x1
        self.A23 = self.m1s * self.h1s * gamma_1 * v_y1
        self.A25 = -self.c * self.m2 * gamma_2
        self.A26 = self.c * self.k3 / v_x2
        self.A27 = -self.c * self.k3 * self.d / v_x2
        self.A28 = -self.m2s*self.h2s*self.c * gamma_2 ** 2

        self.A34 = 1

        self.A42 = - self.m1s * self.h1s * v_x1
        self.A43 = -(
                    self.k_varphi1 + self.k12 - self.m1s * self.gravity * self.h1s - self.m1s * self.h1s ** 2 * gamma_1 ** 2 - self.I1yy * gamma_1 ** 2 + self.I1zz * gamma_1 ** 2)
        self.A44 = -self.c_varphi1
        self.A45 = (self.m2 * self.hh) * gamma_2
        self.A46 = -self.hh * self.k3 / v_x2
        self.A47 = self.d * self.k3 * self.hh / v_x2
        self.A48 = (self.k12 - self.m2s * self.h2s * self.hh * gamma_2 ** 2)

        self.A50 = -(gamma_2 - gamma_1) * torch.sin(phi_2 - phi_1)
        self.A51 = -(gamma_2 - gamma_1) * torch.cos(phi_2 - phi_1)
        self.A52 = (gamma_2 - gamma_1) * torch.cos(phi_2 - phi_1) * self.c

        self.A60 = (gamma_2 - gamma_1) * torch.cos(phi_2 - phi_1)
        self.A61 = -(gamma_2 - gamma_1) * torch.sin(phi_2 - phi_1)
        self.A62 = self.c * (gamma_2 - gamma_1) * torch.sin(phi_2 - phi_1)

        self.A75 = self.e * self.m2 * gamma_2
        self.A76 = -(self.d + self.e) * self.k3 / v_x2
        self.A77 = (self.d + self.e) * self.k3 * self.d / v_x2
        self.A78 = (self.e * self.m2s * self.h2s * gamma_2 ** 2 - self.m2s * self.h2s * gamma_2 * v_y2)

        self.A89 = 1

        self.A93, self.A95, self.A96, self.A97, self.A98, self.A99 = self.k12, -(self.m2s * self.h2s * gamma_2+self.hh*self.m2*gamma_2), \
                                                                     self.hh * self.k3 / v_x2, -self.d * self.hh * self.k3 / v_x2, (
                    self.m2s * self.gravity * self.h2s - self.k_varphi2 - self.k12 + self.m2s * self.h2s ** 2 * gamma_2 ** 2 + self.I2yy * gamma_2 ** 2 - self.I2zz * gamma_2 ** 2-self.m2s*self.h2s*self.hh*gamma_2**2), -self.c_varphi2


        A_matrix[0, 2], A_matrix[0, 4], A_matrix[0, 7], A_matrix[0, 9] = \
            self.A02, self.A04, self.A07, self.A09
        A_matrix[1, 0], A_matrix[1, 1], A_matrix[1, 2], \
        A_matrix[1, 3], A_matrix[1, 5], A_matrix[1, 6], \
        A_matrix[1, 7], A_matrix[1, 8] = self.A10, self.A11, self.A12, \
                                         self.A13, self.A15, self.A16, \
                                         self.A17, self.A18

        A_matrix[2, 1], A_matrix[2, 2], A_matrix[2, 3], A_matrix[2, 5], \
        A_matrix[2, 6], A_matrix[2, 7], A_matrix[2, 8] \
            = self.A21, self.A22, self.A23, self.A25, self.A26, self.A27, self.A28

        A_matrix[3, 4] = self.A34

        A_matrix[4, 2], A_matrix[4, 3], A_matrix[4, 4], \
        A_matrix[4, 5], A_matrix[4, 6], A_matrix[4, 7], \
        A_matrix[4, 8] = \
            self.A42, self.A43, self.A44, self.A45, self.A46, self.A47, self.A48

        A_matrix[5, 0], A_matrix[5, 1], A_matrix[5, 2] = self.A50, self.A51, self.A52

        A_matrix[6, 0], A_matrix[6, 1], A_matrix[6, 2] = self.A60, self.A61, self.A62

        A_matrix[7, 5], A_matrix[7, 6], A_matrix[7, 7], \
        A_matrix[7, 8] = self.A75, self.A76, self.A77, self.A78
        A_matrix[8, 9] = self.A89

        A_matrix[9, 3], A_matrix[9, 5], A_matrix[9, 6], \
        A_matrix[9, 7], A_matrix[9, 8], A_matrix[9, 9] = \
            self.A93, self.A95, self.A96, self.A97, self.A98, self.A99

        B_matrix = torch.zeros((self.state_dim - 6, 2))
        self.B00 = - (self.m1 + self.m2)
        self.B11 = -self.k1
        self.B21 = -self.a * self.k1
        B_matrix[0, 0] = self.B00
        B_matrix[1, 1] = self.B11
        B_matrix[2, 1] = self.B21

        X_dot_batch = torch.zeros((self.batch_size, self.state_dim - 6))
        
        for batch in range(self.batch_size):
            X_dot = (torch.matmul(torch.matmul(torch.inverse(M_matrix), A_matrix), X[batch, :]) + torch.matmul(
                torch.inverse(M_matrix), torch.matmul(B_matrix, action[batch, :]))).squeeze()
            X_dot_batch[batch, :] = X_dot

        state_next[:, 0] = x_1 + delta_t * (v_x1 * torch.cos(phi_1.clone()) - v_y1 * torch.sin(phi_1.clone()))
        state_next[:, 1] = y_1 + delta_t * (v_y1 * torch.cos(phi_1.clone()) + v_x1 * torch.sin(phi_1.clone()))
        state_next[:, 2] = phi_1 + delta_t * gamma_1
        state_next[:, 3] = v_x1 + delta_t * X_dot_batch[:, 0]

        state_next[:, 4] = x_2 + delta_t * (v_x2 * torch.cos(-phi_2.clone()) - v_y2 * torch.sin(-phi_2.clone()))  # state_next[12] - self.b * torch.cos(phi_1) - self.e * torch.cos(phi_2)  # posx_trailer
        state_next[:, 5] = y_2 - delta_t * (v_y2 * torch.cos(-phi_2.clone()) + v_x2 * torch.sin(-phi_2.clone()))  # posy_trailer
        state_next[:, 6] = phi_2 + delta_t * gamma_2
        state_next[:, 7] = v_x2 + delta_t * X_dot_batch[:, 5]

        state_next[:, 8:12] = state[:, 8:12] + delta_t * X_dot_batch[:, 1:5]
        state_next[:, 12:] = state[:, 12:] + delta_t * X_dot_batch[:, 6:]
        return state_next

class Semitruckpu8dofModel(PythBaseModel):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    def __init__(
        self,
        ref_vx: float = 20,
        pre_horizon: int = 20,
        max_steer: float = 0.5,
        max_accel: float = 1.5,
        min_accel: float = -1.5,
        device: Union[torch.device, str, None] = None,
        **kwargs,
    ):
        self.vehicle_dynamics = VehicleDynamicsModel()
        self.state_dim = 16
        ego_obs_dim = 14
        ref_obs_dim = 6
        obs_scale_default = [1/100, 1/100, 1/10,1/100,
                             1/100, 1/100, 1/10,1/100,
                             1/100, 1, 1, 1,
                             1/100,1, 1, 1]
        self.obs_scale = np.array(kwargs.get('obs_scale', obs_scale_default))
        super().__init__(
            obs_dim=ego_obs_dim + ref_obs_dim * pre_horizon,
            action_dim=2,
            action_lower_bound=[min_accel, -max_steer],
            action_upper_bound=[max_accel, max_steer],
            dt=0.01,
            device=device,)

        self.ref_traj = Ref_Route(ref_vx)
        self.action_last = torch.zeros((1, 2))

    def forward(
            self,
            obs: torch.Tensor,
            action: torch.Tensor,
            done: torch.Tensor,
            info: InfoDict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, InfoDict]:
        state = info["state"]
        ref_x = info["ref_x"]
        ref_y = info["ref_y"]
        ref_x2 = info["ref_x2"]
        ref_y2 = info["ref_y2"]
        ref_points = info["ref_points"]
        target_speed = info["target_speed"]

        reward = self.compute_reward(obs, action)

        next_state = self.vehicle_dynamics.f_xu(state, action, self.dt)

        next_ref_points = ref_points.clone()
        next_ref_points[:, :-1] = ref_points[:, 1:]
        next_ref_x = ref_x + self.dt * target_speed
        next_ref_y = ref_y + self.dt * state[:, 14]
        new_ref_point = self.ref_traj.find_nearst_point(torch.stack([next_ref_x, next_ref_y], 1))

        next_ref_x2 = ref_x2 + self.dt * target_speed
        next_ref_y2 = ref_y2 + self.dt * state[:, 14]
        new_ref_point_2 = self.ref_traj.find_nearst_point(torch.stack([next_ref_x2, next_ref_y2], 1))
        next_ref_points[:, -1] = torch.cat((new_ref_point, new_ref_point_2), 1)

        next_obs = self.get_obs(next_state, next_ref_points)

        isdone = self.judge_done(next_state, next_ref_points)

        next_info = {}
        for key, value in info.items():
            next_info[key] = value.detach().clone()
        next_info.update({
            "state": next_state,
            "ref_x": next_ref_x,
            "ref_y": next_ref_y,
            "ref_x2": next_ref_x2,
            "ref_y2": next_ref_y2,
            "ref_points": next_ref_points
        })
        self.action_last = action.clone().detach()
        return next_obs, reward, isdone, next_info

    def get_obs(self, state, ref_points):
        ref_x_tf, ref_y_tf, ref_phi_tf, ref_vx_tf = \
            state_error_calculate(
                state[:, 0], state[:, 1], state[:, 2],state[:, 3],
                ref_points[..., 0], ref_points[..., 1], ref_points[..., 2], ref_points[..., 3]
            )

        ref_x2_tf, ref_y2_tf, ref_phi2_tf, ref_vx2_tf = \
            state_error_calculate(
                state[:, 4], state[:, 5], state[:, 6], state[:, 7],
                ref_points[..., 4], ref_points[..., 5], ref_points[..., 6], ref_points[..., 7]
            )

        # ego_obs: [
        # delta_y, delta_phi,delta_vx, delta_y2, delta_phi2 delta_vx2, (of the first reference point)
        # v, w, varphi (of ego vehicle, including tractor and trailer)
        # ]
        ego_obs = torch.stack((ref_y_tf[:, 0] * self.obs_scale[1], ref_phi_tf[:, 0] * self.obs_scale[2],
                     ref_vx_tf[:, 0] * self.obs_scale[3], ref_y2_tf[:, 0] * self.obs_scale[5],
                     ref_phi2_tf[:, 0] * self.obs_scale[6], ref_vx2_tf[:, 0] * self.obs_scale[7],
                     state[:, 8] * self.obs_scale[8], state[:, 9] * self.obs_scale[9], state[:, 10] * self.obs_scale[10],
                   state[:, 11] * self.obs_scale[11], state[:, 12] * self.obs_scale[12],
                       state[:, 13] * self.obs_scale[13], state[:, 14] * self.obs_scale[14], state[:, 15] * self.obs_scale[15]), dim=1)

        # ref_obs: [
        # delta_x, delta_y, delta_psi (of the second to last reference point)
        # ]
        ref_obs = torch.stack((ref_y_tf*self.obs_scale[1], ref_phi_tf*self.obs_scale[2],
              ref_vx_tf*self.obs_scale[3], ref_y2_tf*self.obs_scale[5],
              ref_phi2_tf*self.obs_scale[6], ref_vx2_tf*self.obs_scale[7]), 2)[
            :, 1:].reshape(ego_obs.shape[0], -1)
        return torch.concat((ego_obs, ref_obs), 1)

    def compute_reward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        accel, steer =  action[:, 0], action[:, 1]
        return -(
                1 * (0.04 * (obs[:, 0]/self.obs_scale[1]) ** 2)
                + 1 * (obs[:, 2]/self.obs_scale[3]) ** 2
                + 0.9 * (obs[:, 6]/self.obs_scale[7]) ** 2
                + 0.8 * (obs[:, 1]/self.obs_scale[2]) ** 2
                + 0.5 * (obs[:, 7]/self.obs_scale[8]) ** 2
                + 0.5 * (obs[:, 8]/self.obs_scale[9]) ** 2
                + 0.5 * (obs[:, 9]/self.obs_scale[10]) ** 2
                + 0.4 * steer ** 2
                + 0.4 * accel ** 2
                + 2.0 * (accel - self.action_last[:, 0]) ** 2
                + 2.0 * (steer - self.action_last[:, 1]) ** 2
        )

    def judge_done(self, state, ref_points) -> bool:
        done = ((abs(state[:, 1]-ref_points[0, 0, 1]) > 3)  # delta_y1
                + (abs(state[:, 3]-ref_points[0, 0, 3]) > 2) # delta_vx1
                + (abs(state[:, 8]) > 2)  # delta_vy1
                  + (abs(state[:, 2]-ref_points[0, 0, 2]) > torch.pi/2)  # delta_phi1
                  + (abs(state[:, 5]-ref_points[0, 0, 5]) > 3) # delta_y2
                  + (abs(state[:, 7]-ref_points[0, 0, 7]) > 2)  # delta_vx2
                  + (abs(state[:, 6]-ref_points[0, 0, 6]) > torch.pi / 2))  # delta_phi2
        return done

    @property
    def info(self) -> dict:
        return {
            "state": self.state.copy(),
            "ref_points": self.ref_points.copy(),
            "ref_x": self.ref_x,
            "ref_y": self.ref_y,
            "ref_x2": self.ref_x2,
            "ref_y2": self.ref_y2,
            "ref": self.ref_points[0].copy(),
            "target_speed": self.target_speed,
        }

    def Load_engine_data(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = current_dir + "/resources/Engine_Torque.csv"
        self.Torque_trucksim = pd.DataFrame(
            pd.read_csv(root_dir, header=None))
        self.Torque_throttle = torch.array(self.Torque_trucksim.iloc[0, 1:].dropna())  # 表的X坐标是throttle
        self.Torque_engspd = torch.array(self.Torque_trucksim.iloc[1:, 0])  # 表的Y坐标是engspd
        self.Torque = torch.array(self.Torque_trucksim.iloc[1:, 1:])  #

    def Calc_EngSpd(self, u1_k, ig):
        EngSpd_Lower = 400
        EngSpd_Upper = 2200
        i0 = 4.4
        rw = 0.51
        EngSpd = float(u1_k) * ig * i0 * 60 / (2 * rw * torch.pi)
        if EngSpd > EngSpd_Upper:
            EngSpd = EngSpd_Upper
        if EngSpd < EngSpd_Lower:
            EngSpd = EngSpd_Lower
        TransSpd = float(u1_k) * 60 / (2 * rw * torch.pi) * i0
        return EngSpd, TransSpd

    def Calc_Thr(self, EngSpd, EngTorque):
        Pedal_Percent = -999
        EngSpd_Lower = 400
        EngSpd_Upper = 2200
        Tecom_Lower = -181.8
        Tecom_Upper = 1364.5
        throttle = 0.0  # default
        torq_temp = [0] * 11
        if EngSpd < EngSpd_Lower or EngSpd > EngSpd_Upper:
            # print("ENGINE SPEED INPUT WRONG(thr_cal)!!", We)
            if EngSpd < EngSpd_Lower:
                EngSpd = EngSpd_Lower
            else:
                EngSpd = EngSpd_Upper
        if EngTorque < Tecom_Lower or EngTorque > Tecom_Upper:
            # print("TORQUE INPUT WRONG!!", Te)
            if EngTorque < Tecom_Lower:
                EngTorque = Tecom_Lower
            else:
                EngTorque = Tecom_Upper

        index_rpm = torch.argwhere(EngSpd >= self.Torque_engspd)[-1][0]
        if EngSpd == self.Torque_engspd[-1]:
            index_rpm = 16
            scale_rpm = 0
        else:
            scale_rpm = (EngSpd - self.Torque_engspd[index_rpm]) / (
                    self.Torque_engspd[index_rpm + 1] - self.Torque_engspd[index_rpm])

        for mm in range(len(self.Torque_throttle)):
            if EngSpd == self.Torque_engspd[-1]:
                torq_temp[mm] = self.Torque[index_rpm][mm]
            else:
                torq_temp[mm] = self.Torque[index_rpm][mm] + scale_rpm * (
                        self.Torque[index_rpm + 1][mm] - self.Torque[index_rpm][mm])

        for mm in range(10):
            if EngTorque >= torq_temp[mm] and EngTorque < torq_temp[mm + 1]:
                index_torq = mm
                break
        if EngTorque < torq_temp[0]:
            index_torq = 0
        if EngTorque >= torq_temp[10]:
            index_torq = 10

        if index_torq == 0:
            if torq_temp[1] - torq_temp[0] == 0.0:
                throttle = 0.0 + (EngTorque - torq_temp[0]) / (0.0000001) * (0.1 - 0)
            else:
                throttle = 0.0 + (EngTorque - torq_temp[0]) / (torq_temp[1] - torq_temp[0]) * (0.1 - 0)
        elif index_torq == 1:
            throttle = 0.1 + (EngTorque - torq_temp[1]) / (torq_temp[2] - torq_temp[1]) * (0.2 - 0.1)
        elif index_torq == 2:
            throttle = 0.2 + (EngTorque - torq_temp[2]) / (torq_temp[3] - torq_temp[2]) * (0.3 - 0.2)
        elif index_torq == 3:
            throttle = 0.3 + (EngTorque - torq_temp[3]) / (torq_temp[4] - torq_temp[3]) * (0.4 - 0.3)
        elif index_torq == 4:
            throttle = 0.4 + (EngTorque - torq_temp[4]) / (torq_temp[5] - torq_temp[4]) * (0.5 - 0.4)
        elif index_torq == 5:
            throttle = 0.5 + (EngTorque - torq_temp[5]) / (torq_temp[6] - torq_temp[5]) * (0.6 - 0.5)
        elif index_torq == 6:
            throttle = 0.6 + (EngTorque - torq_temp[6]) / (torq_temp[7] - torq_temp[6]) * (0.7 - 0.6)
        elif index_torq == 7:
            throttle = 0.7 + (EngTorque - torq_temp[7]) / (torq_temp[8] - torq_temp[7]) * (0.8 - 0.7)
        elif index_torq == 8:
            throttle = 0.8 + (EngTorque - torq_temp[8]) / (torq_temp[9] - torq_temp[8]) * (0.9 - 0.8)
        elif index_torq == 9:
            throttle = 0.9 + (EngTorque - torq_temp[8]) / (torq_temp[9] - torq_temp[8]) * (1.0 - 0.9)
        elif index_torq == 10:
            throttle = 1.0

        if throttle > 1.0:
            throttle = 1.0
        if throttle < 0.0:
            throttle = 0.0

        Pedal_Percent = throttle

        return Pedal_Percent

    def Calc_EngTe(self, u1_k, accel, ig):
        i0 = 4.4
        rw = 0.51
        mass = self.vehicle_dynamics.m1 + self.vehicle_dynamics.m2
        etaT = 0.92 * 0.92
        g = 9.81
        theta = 0
        f = 0.015
        Cd = 0.51
        A = 7.5
        rho = 1.206
        EngTorque = (mass * accel + 0.5 * Cd * A * rho * torch.power(u1_k, 2) + mass * g * (
                f * torch.cos(theta) + torch.sin(theta))) / (ig * i0 * etaT / (rw))
        return EngTorque

    def Threshold_LU(self, gear, Throttle):
        gear = int(gear)
        if gear == 1:  # 1-2
            up = torch.interp(Throttle, [0, 0.2, 0.8, 1], [98, 98, 190, 190])
            down = torch.interp(Throttle, [0, 0.2, 0.8, 1], [60, 60, 110, 110])
        elif gear == 2:
            up = torch.interp(Throttle, [0, 0.2, 0.8, 1], [132, 132, 250, 250])
            down = torch.interp(Throttle, [0, 0.2, 0.8, 1], [119, 119, 190, 190])
        elif gear == 3:
            up = torch.interp(Throttle, [0, 0.2, 0.8, 1], [178, 178, 340, 340])
            down = torch.interp(Throttle, [0, 0.2, 0.8, 1], [160, 160, 277, 277])
        elif gear == 4:
            up = torch.interp(Throttle, [0, 0.2, 0.8, 1], [241, 241, 460, 460])
            down = torch.interp(Throttle, [0, 0.2, 0.8, 1], [217, 217, 374, 374])
        elif gear == 5:
            up = torch.interp(Throttle, [0, 0.2, 0.8, 1], [325, 325, 625, 625])
            down = torch.interp(Throttle, [0, 0.2, 0.8, 1], [293, 293, 506, 506])
        elif gear == 6:
            up = torch.interp(Throttle, [0, 0.2, 0.8, 1], [440, 440, 850, 850])
            down = torch.interp(Throttle, [0, 0.2, 0.8, 1], [396, 396, 683, 683])
        elif gear == 7:
            up = torch.interp(Throttle, [0, 0.2, 0.8, 1], [593, 593, 1125, 1125])
            down = torch.interp(Throttle, [0, 0.2, 0.8, 1], [533, 533, 923, 923])
        elif gear == 8:
            up = torch.interp(Throttle, [0, 0.2, 0.8, 1], [800, 800, 1525, 1525])
            down = torch.interp(Throttle, [0, 0.7, 0.9, 1], [720, 720, 1244, 1244])
        elif gear == 9:
            up = torch.interp(Throttle, [0, 0.2, 0.8, 1], [229, 229, 563, 563])
            down = torch.interp(Throttle, [0, 0.75, 0.9, 1], [206, 206, 506, 506])
        else:
            up = torch.interp(Throttle, [0, 0.2, 0.8, 1], [1081, 1081, 2050, 2050])
            down = torch.interp(Throttle, [0, 0.7, 0.9, 1], [973, 973, 1400, 1400])
        return up, down

    def Gear_Ratio(self, gear):
        ig_dict = {'1': 11.06, '2': 8.2, '3': 6.06, '4': 4.49, '5': 3.32, '6': 2.46,
                   '7': 1.82, '8': 1.35, '9': 1, '10': 0.74}
        gear_ratio = ig_dict[str(int(gear))]
        return gear_ratio

    def Shift_Logic(self, gear_itorchut, AT_Speed, up_th, down_th):
        gear_output = gear_itorchut
        if gear_itorchut == 1:
            if AT_Speed < down_th:
                gear_output = 1
            elif down_th >= AT_Speed and AT_Speed < up_th:
                gear_output = 1
            elif AT_Speed >= up_th:
                gear_output = 2

        elif gear_itorchut == 2:
            if AT_Speed < down_th:
                gear_output = 1
            elif down_th >= AT_Speed and AT_Speed < up_th:
                gear_output = 2
            elif AT_Speed >= up_th:
                gear_output = 3

        elif gear_itorchut == 3:
            if AT_Speed < down_th:
                gear_output = 2
            elif down_th >= AT_Speed and AT_Speed < up_th:
                gear_output = 3
                # return gear
            elif AT_Speed >= up_th:
                gear_output = 4

        elif gear_itorchut == 4:
            if AT_Speed < down_th:
                gear_output = 3
            elif down_th >= AT_Speed and AT_Speed < up_th:
                gear_output = 4
            elif AT_Speed >= up_th:
                gear_output = 5

        elif gear_itorchut == 5:
            if AT_Speed < down_th:
                gear_output = 4
            elif down_th >= AT_Speed and AT_Speed < up_th:
                gear_output = 5
            elif AT_Speed >= up_th:
                gear_output = 6
        elif gear_itorchut == 6:
            if AT_Speed < down_th:
                gear_output = 5
            elif down_th >= AT_Speed and AT_Speed < up_th:
                gear_output = 6
            elif AT_Speed >= up_th:
                gear_output = 7
        elif gear_itorchut == 7:
            if AT_Speed < down_th:
                gear_output = 6
            elif down_th >= AT_Speed and AT_Speed < up_th:
                gear_output = 7
            elif AT_Speed >= up_th:
                gear_output = 8
        elif gear_itorchut == 8:
            if AT_Speed < down_th:
                gear_output = 7
            elif down_th >= AT_Speed and AT_Speed < up_th:
                gear_output = 8
            elif AT_Speed >= up_th:
                gear_output = 9
        elif gear_itorchut == 9:
            if AT_Speed < down_th:
                gear_output = 8
            elif down_th >= AT_Speed and AT_Speed < up_th:
                gear_output = 9
            elif AT_Speed >= up_th:
                gear_output = 10
        # elif gear_itorchut == 10:
        #     if AT_Speed < down_th:
        #         gear_output = 9
        #     elif down_th >= AT_Speed and AT_Speed < up_th:
        #         gear_output = 10
        #     elif AT_Speed >= up_th:
        #         gear_output = 11
        # elif gear_itorchut == 11:
        #     if AT_Speed < down_th:
        #         gear_output = 10
        #     elif down_th >= AT_Speed and AT_Speed < up_th:
        #         gear_output = 11
        #     elif AT_Speed >= up_th:
        #         gear_output = 12
        # elif gear_itorchut == 12:
        #     if AT_Speed < down_th:
        #         gear_output = 11
        #     elif down_th >= AT_Speed and AT_Speed < up_th:
        #         gear_output = 12
        #     elif AT_Speed >= up_th:
        #         gear_output = 13
        # elif gear_itorchut == 13:
        #     if AT_Speed < down_th:
        #         gear_output = 12
        #     elif down_th >= AT_Speed and AT_Speed < up_th:
        #         gear_output = 13
        #     elif AT_Speed >= up_th:
        #         gear_output = 14
        # elif gear_itorchut == 14:
        #     if AT_Speed < down_th:
        #         gear_output = 13
        #     elif down_th >= AT_Speed and AT_Speed < up_th:
        #         gear_output = 14
        #     elif AT_Speed >= up_th:
        #         gear_output = 15
        # elif gear_itorchut == 15:
        #     if AT_Speed < down_th:
        #         gear_output = 14
        #     elif down_th >= AT_Speed and AT_Speed < up_th:
        #         gear_output = 15
        #     elif AT_Speed >= up_th:
        #         gear_output = 16
        # elif gear_itorchut == 16:
        #     if AT_Speed < down_th:
        #         gear_output = 15
        #     elif down_th >= AT_Speed and AT_Speed < up_th:
        #         gear_output = 16
        #     elif AT_Speed >= up_th:
        #         gear_output = 17
        # elif gear_itorchut == 17:
        #     if AT_Speed < down_th:
        #         gear_output = 16
        #     elif down_th >= AT_Speed and AT_Speed < up_th:
        #         gear_output = 17
        #     elif AT_Speed >= up_th:
        #         gear_output = 18
        else:
            if AT_Speed < down_th:
                gear_output = 9
            else:
                gear_output = 10
        return gear_output

def state_error_calculate(
    ego_x: torch.Tensor,
    ego_y: torch.Tensor,
    ego_phi: torch.Tensor,
    ego_vx: torch.Tensor,
    ref_x: torch.Tensor,
    ref_y: torch.Tensor,
    ref_phi: torch.Tensor,
    ref_vx: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ego_x, ego_y, ego_phi, ego_vx = ego_x.unsqueeze(1), ego_y.unsqueeze(1), ego_phi.unsqueeze(1),ego_vx.unsqueeze(1)
    x_err = ego_x - ref_x
    y_err = ego_y - ref_y
    phi_err = angle_normalize(ego_phi - ref_phi)
    vx_err = ego_vx - ref_vx
    return x_err, y_err, phi_err, vx_err

def ego_vehicle_coordinate_transform(
    ego_x: torch.Tensor,
    ego_y: torch.Tensor,
    ego_phi: torch.Tensor,
    ego_vx: torch.Tensor,
    ref_x: torch.Tensor,
    ref_y: torch.Tensor,
    ref_phi: torch.Tensor,
    ref_vx: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ego_x, ego_y, ego_phi,ego_vx = ego_x.unsqueeze(1), ego_y.unsqueeze(1), ego_phi.unsqueeze(1), ego_vx.unsqueeze(1)
    cos_tf = torch.cos(-ego_phi)
    sin_tf = torch.sin(-ego_phi)
    ref_x_tf = (ref_x - ego_x) * cos_tf - (ref_y - ego_y) * sin_tf
    ref_y_tf = (ref_x - ego_x) * sin_tf + (ref_y - ego_y) * cos_tf
    ref_phi_tf = angle_normalize(ref_phi - ego_phi)
    ref_vx_tf = ref_vx - ego_vx
    return ref_x_tf, ref_y_tf, ref_phi_tf,ref_vx_tf

def env_model_creator(**kwargs):
    """
    make env model `pyth_veh3dofconti`
    """
    return Semitruckpu8dofModel(**kwargs)
