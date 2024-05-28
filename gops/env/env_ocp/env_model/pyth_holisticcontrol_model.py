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
from typing import Dict, Optional, Tuple, Union
import pandas as pd
import numpy as np
import torch
from gops.env.env_ocp.env_model.pyth_base_model import PythBaseModel
from gops.env.env_ocp.pyth_holisticcontrol import angle_normalize, VehicleDynamicsData
from gops.env.env_ocp.resources.ref_traj_model import MultiRefTrajModel
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
        self.preview_index = 2
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = current_dir+"/../resources/cury.csv"
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
        self.m = DynamicsData.vehicle_params["m"]
        self.ms = DynamicsData.vehicle_params["ms"]
        self.mu = DynamicsData.vehicle_params["mu"]
        self.g = DynamicsData.vehicle_params["g"]
        self.mu_r = DynamicsData.vehicle_params["mu_r"]
        self.Rw = DynamicsData.vehicle_params["Rw"]
        self.Iw = DynamicsData.vehicle_params["Iw"]

        self.lw = DynamicsData.vehicle_params["lw"]
        self.lf = DynamicsData.vehicle_params["lf"]  # Distance between the center of gravity (CG)and its front axle [m]
        self.lr = DynamicsData.vehicle_params["lr"]  # Distance between the CGand its rear axle [m]

        self.hs = DynamicsData.vehicle_params["hs"]  # Height of the CG of the sprung mass for the tractor [m]
        self.hr = DynamicsData.vehicle_params["hr"]  # Height of the CG of the sprung mass for the tractor [m]
        self.hu = DynamicsData.vehicle_params["hu"]  # Height of the CG of the sprung mass for the tractor [m]

        self.Izz = DynamicsData.vehicle_params["Izz"]  # Yaw moment of inertia of the whole mass[kg m^2]
        self.Ixx = DynamicsData.vehicle_params["Ixx"]  # Roll moment of inertia of the sprung mass[kg m^2]

        self.Ixz = DynamicsData.vehicle_params["Ixz"]  # Roll–yaw product of inertia of the sprung mass[kg m^2]

        self.k_alpha1 = DynamicsData.vehicle_params["k_alpha1"]  # Tire cornering stiffness of the 1st wheel[N/rad]
        self.k_alpha2 = DynamicsData.vehicle_params["k_alpha2"]  # Tire cornering stiffness of the 1st wheel[N/rad]
        self.k_alpha3 = DynamicsData.vehicle_params["k_alpha3"]  # Tire cornering stiffness of the rear axle[N/rad]
        self.k_alpha4 = DynamicsData.vehicle_params["k_alpha4"]  # Tire cornering stiffness of the rear axle[N/rad]
        self.C_slip1 = DynamicsData.vehicle_params["C_slip1"]
        self.C_slip2 = DynamicsData.vehicle_params["C_slip2"]
        self.C_slip3 = DynamicsData.vehicle_params["C_slip3"]
        self.C_slip4 = DynamicsData.vehicle_params["C_slip4"]

        self.K_varphi = DynamicsData.vehicle_params["K_varphi"]  # roll stiffness of tire [N-m/rad] /3.14*180
        self.C_varphi = DynamicsData.vehicle_params["C_varphi"]  # Roll damping of the suspension [N-m-s/rad]

        self.state_dim = DynamicsData.vehicle_params["state_dim"]

    def f_xu(self, state, action, delta_t):
        self.batch_size = len(state[:, 0])
        x, y, phi, v_x, v_y, gamma, varphi, varphi_dot = state[:, 0], state[:, 1],state[:, 2], state[0, 3],\
            state[:, 4], state[:, 5],state[:, 6], state[:, 7]
        # kappa1, kappa2, kappa3, kappa4  state[:, 8], state[:, 9],state[:, 10], state[:, 11]
        Q1, delta1, Q2, delta2, Q3, delta3, Q4, delta4, \
        dQ1, ddelta1, dQ2, ddelta2, dQ3, ddelta3, dQ4, ddelta4 = \
            action[:, 0], action[:, 1], action[:, 2], action[:, 3], \
            action[:, 4], action[:, 5], action[:, 6], action[:, 7],\
            action[:, 8], action[:, 9], action[:, 10], action[:, 11], \
            action[:, 12], action[:, 13], action[:, 14], action[:, 15]

        X = torch.tensor(state[:, 3:8])
        D = action[:, :8]
        U = action[:, 8:]

        state_next = torch.zeros_like(state)
        dividend = (self.m * self.Ixx * self.Izz - self.Izz * self.ms ** 2 * self.hs ** 2 - self.m * self.Ixz ** 2)
        A_matrix = torch.zeros((5, 5))

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

        B_matrix = torch.zeros((5, 3))
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

        Lc_matrix = torch.zeros((3, 8))
        Lc_matrix[0, 0], Lc_matrix[0, 2], Lc_matrix[0, 4], Lc_matrix[0, 6] = 1, 1, 1, 1

        Lc_matrix[1, 1], Lc_matrix[1, 3], Lc_matrix[1, 5], Lc_matrix[1, 7] = 1, 1, 1, 1

        Lc_matrix[2, 0], Lc_matrix[2, 1], Lc_matrix[2, 2], Lc_matrix[2, 3], \
        Lc_matrix[2, 4], Lc_matrix[2, 5], Lc_matrix[2, 6], Lc_matrix[2, 7] \
            = -self.lw / 2, self.lf, self.lw / 2, self.lf, \
              -self.lw / 2, -self.lr, self.lw / 2, -self.lr

        Ec_matrix = torch.eye(8)
        # active control bool matrix
        Ew_matrix = torch.eye(8) # torch.diag(torch.tensor([0., 0., 0., 0., 0., 0., 0., 0.]))

        A1_matrix = torch.zeros((8, 5))

        A1_matrix[1, 1], A1_matrix[1, 2] = -self.k_alpha1 / v_x, -self.k_alpha1 * self.lf / v_x

        A1_matrix[3, 1], A1_matrix[3, 2] = -self.k_alpha2 / v_x, -self.k_alpha2 * self.lf / v_x

        A1_matrix[5, 1], A1_matrix[5, 2] = -self.k_alpha3 / v_x, -self.k_alpha3 * (-self.lr) / v_x

        A1_matrix[7, 1], A1_matrix[7, 2] = -self.k_alpha4 / v_x, -self.k_alpha4 * (-self.lr) / v_x

        B1 = torch.tensor([1 / self.Rw, self.k_alpha1, 1 / self.Rw, self.k_alpha2,
              1 / self.Rw, self.k_alpha3, 1 / self.Rw, self.k_alpha4])

        B1_matrix = torch.diag(B1)
        X_dot_batch = torch.zeros((self.batch_size, 5))
        dt_matrix = torch.zeros((self.batch_size, 8))
        dt_matrix[:, 0], dt_matrix[:, 2], dt_matrix[:, 4], dt_matrix[:, 6] =\
            -1 / 4 * self.m * self.g * self.mu_r, -1 / 4 * self.m * self.g * self.mu_r, \
            -1 / 4 * self.m * self.g * self.mu_r, -1 / 4 * self.m * self.g * self.mu_r
        for batch in range(self.batch_size):
            temp = torch.matmul(A1_matrix, X[batch]) + torch.matmul(B1_matrix, D[batch]) + torch.matmul(
                torch.matmul(Ew_matrix, B1_matrix), U[batch])+dt_matrix[batch]
            Mw1 = torch.tensor([[torch.cos(delta1[batch]), -torch.sin(delta1[batch])],
                                [torch.sin(delta1[batch]), torch.cos(delta1[batch])]])
            Mw2 = torch.tensor([[torch.cos(delta2[batch]), -torch.sin(delta2[batch])],
                                [torch.sin(delta2[batch]), torch.cos(delta2[batch])]])
            Mw3 = torch.tensor([[torch.cos(delta3[batch]), -torch.sin(delta3[batch])],
                                [torch.sin(delta3[batch]), torch.cos(delta3[batch])]])
            Mw4 = torch.tensor([[torch.cos(delta4[batch]), -torch.sin(delta4[batch])],
                                [torch.sin(delta4[batch]), torch.cos(delta4[batch])]])

            Mw_matrix = torch.block_diag(Mw1, Mw2, Mw3, Mw4)

            X_dot = (torch.matmul(A_matrix, X[batch]) + torch.matmul(torch.matmul(torch.matmul(torch.matmul(
            B_matrix, Lc_matrix), Ec_matrix), Mw_matrix), temp)).squeeze()
            X_dot_batch[batch, :] = X_dot
        state_next[:, 0] = x + delta_t * (v_x * torch.cos(phi.clone()) - v_y * torch.sin(phi.clone()))
        state_next[:, 1] = y + delta_t * (v_y * torch.cos(phi.clone()) + v_x * torch.sin(phi.clone()))
        state_next[:, 2] = phi + delta_t * gamma
        state_next[:, 3:8] = state[:, 3:8] + delta_t * X_dot_batch

        # state_next[:, 8] = kappa1 + delta_t * (
        #             self.Rw * (Q1 - self.Rw * self.C_slip1 * kappa1) / (v_x * self.Iw) - (1 + kappa1) / (
        #                 self.m * v_x) * (self.C_slip1*kappa1+self.C_slip2*kappa2+self.C_slip3*kappa3+self.C_slip4*kappa4))
        # state_next[:, 9] = kappa2 + delta_t * (
        #             self.Rw * (Q2 - self.Rw * self.C_slip2 * kappa2) / (v_x * self.Iw) - (1 + kappa2) / (
        #                 self.m * v_x) * (self.C_slip1*kappa1+self.C_slip2*kappa2+self.C_slip3*kappa3+self.C_slip4*kappa4))
        # state_next[:, 10] = kappa3 + delta_t * (
        #             self.Rw * (Q3 - self.Rw * self.C_slip3 * kappa3) / (v_x * self.Iw) - (1 + kappa3) / (
        #                 self.m * v_x) * (self.C_slip1*kappa1+self.C_slip2*kappa2+self.C_slip3*kappa3+self.C_slip4*kappa4))
        # state_next[:, 11] = kappa4 + delta_t * (
        #             self.Rw * (Q4 - self.Rw * self.C_slip4 * kappa4) / (v_x * self.Iw) - (1 + kappa4) / (
        #                 self.m * v_x) * (self.C_slip1*kappa1+self.C_slip2*kappa2+self.C_slip3*kappa3+self.C_slip4*kappa4))

        return state_next

class FourwsdvehicleholisticcontrolModel(PythBaseModel):
    def __init__(
        self,
        ref_vx: float = 10,
        pre_horizon: int = 30,
        device: Union[torch.device, str, None] = None,
        max_torque: float = 298,
        max_steer: float = 0.5,
        max_delta_torque=10,
        max_delta_str=0.1,
        **kwargs,
    ):
        """
        you need to define parameters here
        """
        self.vehicle_dynamics = VehicleDynamicsModel()
        self.pre_horizon = pre_horizon
        self.state_dim = 8
        self.ref_vx = ref_vx
        ego_obs_dim = 7
        ref_obs_dim = 3
        obs_scale_default = [1/100, 1/100, 1/10,
                             1/100, 1/100, 1/10, 1/10, 1/50]
        self.obs_scale = np.array(kwargs.get('obs_scale', obs_scale_default))
        super().__init__(
            obs_dim=ego_obs_dim + ref_obs_dim * pre_horizon,
            action_dim=16,
            dt=0.01,
            action_lower_bound=[-max_torque, -max_steer]*4+[-max_delta_torque, -max_delta_str]*4,
            action_upper_bound=[max_torque, max_steer]*4+[max_delta_torque, max_delta_str]*4,
            device=device,
        )

        self.ref_traj = Ref_Route(self.ref_vx)
        self.action_last = torch.zeros((1, 16))

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
        ref_points = info["ref_points"]

        reward = self.compute_reward(state, action, ref_points)

        next_state = self.vehicle_dynamics.f_xu(state, action, self.dt)

        next_ref_points = ref_points.clone()
        next_ref_points[:, :-1] = ref_points[:, 1:]
        next_ref_x = ref_x + self.dt * state[:, 3]
        next_ref_y = ref_y + self.dt * state[:, 4]
        new_ref_point = self.ref_traj.find_nearst_point(torch.stack([next_ref_x, next_ref_y], 1))
        next_ref_points[:, -1] = new_ref_point


        next_obs = self.get_obs(next_state, next_ref_points)

        isdone = self.judge_done(next_obs)

        next_info = {}
        for key, value in info.items():
            next_info[key] = value.detach().clone()
        next_info.update({
            "state": next_state,
            "ref_x": next_ref_x,
            "ref_y": next_ref_y,
            "ref_points": next_ref_points
        })
        self.action_last = action.clone().detach()
        return next_obs, reward, isdone, next_info

    def get_obs(self, state, ref_points):
        ref_x_tf, ref_y_tf, ref_phi_tf, ref_vx_tf = \
            state_error_calculate(
                state[:, 0], state[:, 1], state[:, 2], state[:, 3],
                ref_points[..., 0], ref_points[..., 1], ref_points[..., 2], ref_points[..., 3],
            )

        ego_obs = torch.stack(
            (ref_y_tf[:, 0]*self.obs_scale[1], ref_phi_tf[:, 0]*self.obs_scale[2], ref_vx_tf[:, 0]*self.obs_scale[3],
             state[:, 4]*self.obs_scale[4], state[:, 5]*self.obs_scale[5], state[:, 6]*self.obs_scale[6], state[:, 7]*self.obs_scale[7],
             ), dim=1) # state[:, 8]*self.obs_scale[8], state[:, 9]*self.obs_scale[9], state[:, 10]*self.obs_scale[10], state[:, 11]*self.obs_scale[11]
        ref_obs = torch.stack((ref_y_tf*self.obs_scale[1], ref_phi_tf*self.obs_scale[2], ref_vx_tf*self.obs_scale[3]), 2)[
            :, 1:].reshape(ego_obs.shape[0], -1)
        return torch.concat((ego_obs, ref_obs), 1)

    def compute_reward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        ref_points: torch.Tensor
    ) -> torch.Tensor:
        px, py, phi, vx, vy, gamma, varphi, \
        varphi_dot = state[0, 0], state[0, 1],state[0, 2], state[0, 3],\
            state[0, 4], state[0, 5], state[0, 6], state[0, 7]
        #kappa1, kappa2, kappa3, kappa4 state[0, 8], state[0, 9], state[0, 10], state[0, 11]
        Q1, delta1, Q2, delta2, Q3, delta3, Q4, delta4, \
        dQ1, ddelta1, dQ2, ddelta2, dQ3, ddelta3, dQ4, ddelta4 = \
            action[0, 0], action[0, 1], action[0, 2], action[0, 3], \
            action[0, 4], action[0, 5], action[0, 6], action[0, 7],\
            action[0, 8], action[0, 9], action[0, 10], action[0, 11], \
            action[0, 12], action[0, 13], action[0, 14], action[0, 15]
        ref_x, ref_y, ref_phi, ref_vx = ref_points[:, 0, 0], ref_points[:, 0, 1], ref_points[:, 0, 2], ref_points[:, 0, 3]
        beta = torch.arctan(vy / vx)
        I_matrix = torch.tensor([[(self.vehicle_dynamics.k_alpha1 + self.vehicle_dynamics.k_alpha2 +
                               self.vehicle_dynamics.k_alpha3 + self.vehicle_dynamics.k_alpha4) / (
                                          self.vehicle_dynamics.m * vx),
                              self.vehicle_dynamics.lf * (
                                          self.vehicle_dynamics.k_alpha1 + self.vehicle_dynamics.k_alpha2) -
                              self.vehicle_dynamics.lr * (
                                          self.vehicle_dynamics.k_alpha3 + self.vehicle_dynamics.k_alpha4) / (
                                          self.vehicle_dynamics.m * vx ** 2)],
                             [self.vehicle_dynamics.lf * (
                                         self.vehicle_dynamics.k_alpha1 + self.vehicle_dynamics.k_alpha2) -
                              self.vehicle_dynamics.lr * (
                                          self.vehicle_dynamics.k_alpha3 + self.vehicle_dynamics.k_alpha4) / (
                                  self.vehicle_dynamics.Izz),
                              self.vehicle_dynamics.lf ** 2 * (
                                          self.vehicle_dynamics.k_alpha1 + self.vehicle_dynamics.k_alpha2) +
                              self.vehicle_dynamics.lr ** 2 * (
                                          self.vehicle_dynamics.k_alpha3 + self.vehicle_dynamics.k_alpha4) / (
                                          self.vehicle_dynamics.Izz * vx)]])

        k_matrix = torch.tensor([[-self.vehicle_dynamics.k_alpha1 / (self.vehicle_dynamics.m * vx),
                              -self.vehicle_dynamics.k_alpha2 / (self.vehicle_dynamics.m * vx),
                              -self.vehicle_dynamics.k_alpha3 / (self.vehicle_dynamics.m * vx),
                              -self.vehicle_dynamics.k_alpha4 / (self.vehicle_dynamics.m * vx)],
                             [-self.vehicle_dynamics.lf * self.vehicle_dynamics.k_alpha1 / self.vehicle_dynamics.Izz,
                              -self.vehicle_dynamics.lf * self.vehicle_dynamics.k_alpha2 / self.vehicle_dynamics.Izz,
                              self.vehicle_dynamics.lr * self.vehicle_dynamics.k_alpha3 / self.vehicle_dynamics.Izz,
                              self.vehicle_dynamics.lr * self.vehicle_dynamics.k_alpha4 / self.vehicle_dynamics.Izz]])
        delta_matrix = torch.tensor([[delta1], [delta2], [delta3], [delta4]])

        later_ref = torch.matmul(torch.matmul(torch.linalg.inv(I_matrix), k_matrix), delta_matrix)
        beta_ref = 0#later_ref[0][0]
        gamma_ref = 0#later_ref[1][0]
        C_varphi = 2 / (self.vehicle_dynamics.m * self.vehicle_dynamics.g * self.vehicle_dynamics.lw) * \
                   (self.vehicle_dynamics.K_varphi * (1 + (self.vehicle_dynamics.ms * self.vehicle_dynamics.hr +
                                                           self.vehicle_dynamics.mu * self.vehicle_dynamics.hu) /
                                                      (self.vehicle_dynamics.ms * self.vehicle_dynamics.hs)) - (
                                self.vehicle_dynamics.ms * self.vehicle_dynamics.hr +
                                self.vehicle_dynamics.mu * self.vehicle_dynamics.hu) * self.vehicle_dynamics.g)
        C_varphi_dot = 2 * C_varphi / (self.vehicle_dynamics.m * self.vehicle_dynamics.g * self.vehicle_dynamics.lw) * \
                       ((1 + (
                                   self.vehicle_dynamics.ms * self.vehicle_dynamics.hr + self.vehicle_dynamics.mu * self.vehicle_dynamics.hu) /
                         (self.vehicle_dynamics.ms * self.vehicle_dynamics.hs)))
        I_rollover = C_varphi * varphi + C_varphi_dot * varphi_dot
        kappa_constant = 0.15#vx / self.vehicle_dynamics.Rw
        r_action_Q = torch.sum((action[0, 0:8:2]) ** 2)
        r_action_str = torch.sum((action[0, 1:8:2]) ** 2)
        r_action_Qdot = torch.sum((action[0, 0:8:2] - self.action_last[0, 0:8:2]) ** 2)
        r_action_strdot = torch.sum((action[0, 1:8:2] - self.action_last[0, 1:8:2]) ** 2)

        r_action_deltaQ = torch.sum((action[0, 8:16:2]) ** 2)
        r_action_deltastr = torch.sum((action[0, 9:16:2]) ** 2)
        r_action_deltaQdot = torch.sum((action[0, 8:16:2] - self.action_last[0, 8:16:2]) ** 2)
        r_action_deltastrdot = torch.sum((action[0, 9:16:2] - self.action_last[0, 9:16:2]) ** 2)
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

    def judge_done(self, obs: torch.Tensor) -> torch.Tensor:
        delta_y, delta_phi, delta_vx = obs[:, 0]/self.obs_scale[1], obs[:, 1]/self.obs_scale[2],obs[:, 2]/self.obs_scale[3],
        done = (
                (torch.abs(delta_y) > 3)
                | (torch.abs(delta_vx) > 3)
                | (torch.abs(delta_phi) > np.pi/2)
        )
        return done



def state_error_calculate(
    ego_x: torch.Tensor,
    ego_y: torch.Tensor,
    ego_phi: torch.Tensor,
    ego_vx: torch.Tensor,
    ref_x: torch.Tensor,
    ref_y: torch.Tensor,
    ref_phi: torch.Tensor,
    ref_vx: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ego_x, ego_y, ego_phi, ego_vx = ego_x.unsqueeze(1), ego_y.unsqueeze(1), ego_phi.unsqueeze(1), ego_vx.unsqueeze(1)
    x_err = ego_x - ref_x
    y_err = ego_y - ref_y
    phi_err = angle_normalize(ego_phi - ref_phi)
    vx_err = ego_vx - ref_vx
    return x_err, y_err, phi_err, vx_err

def ego_vehicle_coordinate_transform(
    ego_x: torch.Tensor,
    ego_y: torch.Tensor,
    ego_phi: torch.Tensor,
    ref_x: torch.Tensor,
    ref_y: torch.Tensor,
    ref_phi: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ego_x, ego_y, ego_phi = ego_x.unsqueeze(1), ego_y.unsqueeze(1), ego_phi.unsqueeze(1)
    cos_tf = torch.cos(-ego_phi)
    sin_tf = torch.sin(-ego_phi)
    ref_x_tf = (ref_x - ego_x) * cos_tf - (ref_y - ego_y) * sin_tf
    ref_y_tf = (ref_x - ego_x) * sin_tf + (ref_y - ego_y) * cos_tf
    ref_phi_tf = angle_normalize(ref_phi - ego_phi)
    return ref_x_tf, ref_y_tf, ref_phi_tf


def env_model_creator(**kwargs):
    """
    make env model `pyth_veh3dofconti`
    """
    return FourwsdvehicleholisticcontrolModel(**kwargs)
