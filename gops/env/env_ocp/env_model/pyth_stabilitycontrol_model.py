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
from typing import Dict, Optional, Tuple, Union
import pandas as pd
import numpy as np
import torch
from gops.env.env_ocp.env_model.pyth_base_model import PythBaseModel
from gops.env.env_ocp.pyth_stabilitycontrol import angle_normalize, VehicleDynamicsData
from gops.env.env_ocp.resources.ref_traj_model import MultiRefTrajModel, MultiRoadSlopeModel
from gops.utils.gops_typing import InfoDict



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
        self.A = DynamicsData.vehicle_params["A"]
        self.rho = DynamicsData.vehicle_params["rho"]
        self.Cd = DynamicsData.vehicle_params["Cd"]

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

    def f_xu(self, state, action, delta_t, road_info):
        R = road_info
        self.batch_size = len(state[:, 0])
        x, y, phi, v_x, v_y, phi_dot, varphi, varphi_dot = state[:, 0], state[:, 1],state[:, 2], state[0, 3],\
            state[:, 4], state[:, 5], state[:, 6], state[:, 7]
        delta = action[:, 4]
        X = torch.tensor(state[:, 3:8])
        U = action
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

        R_matrix = torch.zeros((5, 2))
        R_matrix[0, 0] = -self.g
        R_matrix[1, 1] = (self.Izz * self.ms * self.hs * self.K_varphi -
                          self.g * self.m * (self.Ixx * self.Izz - self.Ixz ** 2)) / dividend

        R_matrix[2, 1] = (self.m * self.Ixz * self.K_varphi -
                          self.m * self.Ixz * self.ms * self.hs * self.g) / dividend

        R_matrix[4, 1] = (self.m * self.Izz * self.K_varphi - self.m * self.Izz * self.ms * self.hs * self.g) / dividend

        Lc_matrix = torch.zeros((3, 8))
        Lc_matrix[0, 0], Lc_matrix[0, 2], Lc_matrix[0, 4], Lc_matrix[0, 6] = 1, 1, 1, 1

        Lc_matrix[1, 1], Lc_matrix[1, 3], Lc_matrix[1, 5], Lc_matrix[1, 7] = 1, 1, 1, 1

        Lc_matrix[2, 0], Lc_matrix[2, 1], Lc_matrix[2, 2], Lc_matrix[2, 3], \
        Lc_matrix[2, 4], Lc_matrix[2, 5], Lc_matrix[2, 6], Lc_matrix[2, 7] \
            = -self.lw / 2, self.lf, self.lw / 2, self.lf, \
              -self.lw / 2, -self.lr, self.lw / 2, -self.lr


        At_matrix = torch.zeros((8, 5))

        At_matrix[1, 1], At_matrix[1, 2] = -self.k_alpha1 / v_x, -self.k_alpha1 * self.lf / v_x

        At_matrix[3, 1], At_matrix[3, 2] = -self.k_alpha2 / v_x, -self.k_alpha2 * self.lf / v_x

        At_matrix[5, 1], At_matrix[5, 2] = -self.k_alpha3 / v_x, -self.k_alpha3 * (-self.lr) / v_x

        At_matrix[7, 1], At_matrix[7, 2] = -self.k_alpha4 / v_x, -self.k_alpha4 * (-self.lr) / v_x

        Bt_matrix = torch.zeros((8, 5))
        Bt_matrix[0, 0], Bt_matrix[2, 1], Bt_matrix[4, 2], Bt_matrix[
            6, 3] = 1 / self.Rw, 1 / self.Rw, 1 / self.Rw, 1 / self.Rw
        Bt_matrix[1, 4], Bt_matrix[3, 4] = self.k_alpha1, self.k_alpha2

        X_dot_batch = torch.zeros((self.batch_size, 5))
        for batch in range(self.batch_size):
            temp = torch.matmul(At_matrix, X[batch]) + torch.matmul(Bt_matrix, U[batch])
            Mw1 = torch.tensor([[torch.cos(delta[batch]), -torch.sin(delta[batch])],
                                [torch.sin(delta[batch]), torch.cos(delta[batch])]])
            Mw2 = torch.tensor([[torch.cos(delta[batch]), -torch.sin(delta[batch])],
                                [torch.sin(delta[batch]), torch.cos(delta[batch])]])
            Mw3 = torch.tensor([[1, 0],
                                [0, 1]])
            Mw4 = torch.tensor([[1, 0],
                                [0, 1]])

            Mw_matrix = torch.block_diag(Mw1, Mw2, Mw3, Mw4)
            X_dot = (torch.matmul(A_matrix, X[batch]) + torch.matmul(torch.matmul(torch.matmul(
            B_matrix, Lc_matrix), Mw_matrix), temp)+torch.matmul(R_matrix, R[batch])).squeeze()
            X_dot_batch[batch, :] = X_dot
        state_next[:, 0] = x + delta_t * (v_x * torch.cos(phi.clone()) - v_y * torch.sin(phi.clone()))
        state_next[:, 1] = y + delta_t * (v_y * torch.cos(phi.clone()) + v_x * torch.sin(phi.clone()))
        state_next[:, 2] = phi + delta_t * phi_dot
        state_next[:, 2] = angle_normalize(state_next[:, 2])
        state_next[:, 3:8] = state[:, 3:8] + delta_t * X_dot_batch
        state_next[:, 8:13] = action

        return state_next

class FourwdstabilitycontrolModel(PythBaseModel):
    def __init__(
        self,
        pre_horizon: int = 30,
        device: Union[torch.device, str, None] = None,
        min_delta_torque: float = -10,
        max_delta_torque: float = 10,
        max_delta_steer: float = 0.05,
        path_para: Optional[Dict[str, Dict]] = None,
        u_para: Optional[Dict[str, Dict]] = None,
        slope_para: Optional[Dict[str, Dict]] = None,
        **kwargs,
    ):
        """
        you need to define parameters here
        """
        self.vehicle_dynamics = VehicleDynamicsModel()
        self.pre_horizon = pre_horizon
        self.state_dim = 13
        ego_obs_dim = 8
        ref_obs_dim = 4
        obs_scale_default = [1/100, 1/100, 1/10,
                             1/100, 1/100, 1/10, 1/10, 1/50, 1/(298*10), 1/10]
        self.obs_scale = np.array(kwargs.get('obs_scale', obs_scale_default))
        super().__init__(
            obs_dim=ego_obs_dim + ref_obs_dim * pre_horizon+2*pre_horizon+5,
            action_dim=5,
            dt=0.02,
            action_lower_bound=[min_delta_torque]*4+[-max_delta_steer],
            action_upper_bound=[max_delta_torque]*4+[max_delta_steer],
            device=device,
        )

        self.ref_traj = MultiRefTrajModel(path_para, u_para)
        self.road_slope = MultiRoadSlopeModel(slope_para)


    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        done: torch.Tensor,
        info: InfoDict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, InfoDict]:
        state = info["state"]
        ref_points = info["ref_points"]
        slope_points = info["slope_points"]
        path_num = info["path_num"]
        u_num = info["u_num"]
        slope_num = info["slope_num"]
        t = info["ref_time"]
        reward = self.compute_reward(obs, action, slope_points)
        action_psc = action + state[:, 8:13]
        action_lower_bound, action_upper_bound = torch.tensor([0] * 4 + [-0.5]), torch.tensor([298] * 4 + [0.5])
        action_psc = torch.clamp(action_psc, min=action_lower_bound, max=action_upper_bound)
        next_state = self.vehicle_dynamics.f_xu(state, action_psc, self.dt, slope_points[:, 1, :])

        next_t = t + self.dt

        next_ref_points = ref_points.clone()
        next_ref_points[:, :-1] = ref_points[:, 1:]

        next_slope_points = slope_points.clone()
        next_slope_points[:, :-1] = slope_points[:, 1:]
        new_ref_point = torch.stack(
            (
                self.ref_traj.compute_x(
                    next_t + self.pre_horizon * self.dt, path_num, u_num
                ),
                self.ref_traj.compute_y(
                    next_t + self.pre_horizon * self.dt, path_num, u_num
                ),
                self.ref_traj.compute_phi(
                    next_t + self.pre_horizon * self.dt, path_num, u_num
                ),
                self.ref_traj.compute_u(
                    next_t + self.pre_horizon * self.dt, path_num, u_num
                ),
            ),
            dim=1,
        )
        next_ref_points[:, -1] = new_ref_point

        new_slope_points = torch.stack(
            (
                self.road_slope.compute_longislope(
                    next_t + self.pre_horizon * self.dt, slope_num),
                self.road_slope.compute_latslope(
                    next_t + self.pre_horizon * self.dt, slope_num)
            ),
            dim=1,
        )
        next_slope_points[:, -1] = new_slope_points

        next_obs = self.get_obs(next_state, next_ref_points, next_slope_points)

        isdone = self.judge_done(next_obs)

        next_info = {}
        for key, value in info.items():
            next_info[key] = value.detach().clone()
        next_info.update({
            "state": next_state,
            "ref_points": next_ref_points,
            "path_num": path_num,
            "u_num": u_num,
            "ref_time": next_t,
            "slope_points": next_slope_points,
        })

        return next_obs, reward, isdone, next_info

    def get_obs(self, state, ref_points, slope_points):
        ref_x_tf, ref_y_tf, ref_phi_tf = \
            ego_vehicle_coordinate_transform(
                state[:, 0], state[:, 1], state[:, 2],
                ref_points[..., 0], ref_points[..., 1], ref_points[..., 2],
            )
        ref_u_tf = ref_points[..., 3] - state[:, 3].unsqueeze(1)
        ego_obs = torch.concat((torch.stack((ref_x_tf[:, 0]*self.obs_scale[0], ref_y_tf[:, 0]*self.obs_scale[1], ref_phi_tf[:, 0]*self.obs_scale[2], ref_u_tf[:, 0]*self.obs_scale[3]), dim=1),
                                torch.stack((state[:, 4]*self.obs_scale[4], state[:, 5]*self.obs_scale[5], state[:, 6]*self.obs_scale[6], state[:, 7]*self.obs_scale[7],
                                             state[:, 8]*self.obs_scale[8], state[:, 9]*self.obs_scale[8], state[:, 10]*self.obs_scale[8], state[:, 11]*self.obs_scale[8], state[:, 12]*self.obs_scale[9]), dim=1)), dim=1)
        ref_obs = torch.stack((ref_x_tf*self.obs_scale[0], ref_y_tf*self.obs_scale[1], ref_phi_tf*self.obs_scale[2], ref_u_tf*self.obs_scale[3], slope_points[:, :, 0], slope_points[:, :, 1]), 2)[
                  :, 1:].reshape(ego_obs.shape[0], -1)
        return torch.concat((ego_obs, ref_obs), 1)

    def compute_reward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        slope_points: torch.Tensor
    ) -> torch.Tensor:
        delta_x, delta_y, delta_phi, delta_vx, vy, phi_dot, varphi, varphi_dot = (
            obs[:, 0]/self.obs_scale[0], obs[:, 1]/self.obs_scale[1], obs[:, 2]/self.obs_scale[2], obs[:, 3]/self.obs_scale[3],\
            obs[:, 4]/self.obs_scale[4], obs[:, 5]/self.obs_scale[5], obs[:, 6]/self.obs_scale[6], obs[:, 7]/self.obs_scale[7])
        #kappa1, kappa2, kappa3, kappa4 state[0, 8], state[0, 9], state[0, 10], state[0, 11]
        # Q1, Q2, Q3, Q4, delta = action[:, 0], action[:, 1], action[:, 2], action[:, 3], \
        #     action[:, 4], action[:, 5]
        # delta1, delta2, delta3, delta4 = delta, delta, delta, delta

        # Q1, delta1, Q2, delta2, Q3, delta3, Q4, delta4 = \
        #     action[0, 0], action[0, 1], action[0, 2], action[0, 3], \
        #     action[0, 4], action[0, 5], action[0, 6], action[0, 7]
        # dQ1, ddelta1, dQ2, ddelta2, dQ3, ddelta3, dQ4, ddelta4 = ,\
        #     action[0, 8], action[0, 9], action[0, 10], action[0, 11], \
        #     action[0, 12], action[0, 13], action[0, 14], action[0, 15]
        # ref_x, ref_y, ref_phi, ref_vx = ref_points[:, 0, 0], ref_points[:, 0, 1], ref_points[:, 0, 2], ref_points[:, 0, 3]
        # beta = torch.arctan(vy / vx)
        # I_matrix = torch.tensor([[(self.vehicle_dynamics.k_alpha1 + self.vehicle_dynamics.k_alpha2 +
        #                        self.vehicle_dynamics.k_alpha3 + self.vehicle_dynamics.k_alpha4) / (
        #                                   self.vehicle_dynamics.m * vx),
        #                       self.vehicle_dynamics.lf * (
        #                                   self.vehicle_dynamics.k_alpha1 + self.vehicle_dynamics.k_alpha2) -
        #                       self.vehicle_dynamics.lr * (
        #                                   self.vehicle_dynamics.k_alpha3 + self.vehicle_dynamics.k_alpha4) / (
        #                                   self.vehicle_dynamics.m * vx ** 2)],
        #                      [self.vehicle_dynamics.lf * (
        #                                  self.vehicle_dynamics.k_alpha1 + self.vehicle_dynamics.k_alpha2) -
        #                       self.vehicle_dynamics.lr * (
        #                                   self.vehicle_dynamics.k_alpha3 + self.vehicle_dynamics.k_alpha4) / (
        #                           self.vehicle_dynamics.Izz),
        #                       self.vehicle_dynamics.lf ** 2 * (
        #                                   self.vehicle_dynamics.k_alpha1 + self.vehicle_dynamics.k_alpha2) +
        #                       self.vehicle_dynamics.lr ** 2 * (
        #                                   self.vehicle_dynamics.k_alpha3 + self.vehicle_dynamics.k_alpha4) / (
        #                                   self.vehicle_dynamics.Izz * vx)]])
        #
        # k_matrix = torch.tensor([[-self.vehicle_dynamics.k_alpha1 / (self.vehicle_dynamics.m * vx),
        #                       -self.vehicle_dynamics.k_alpha2 / (self.vehicle_dynamics.m * vx),
        #                       -self.vehicle_dynamics.k_alpha3 / (self.vehicle_dynamics.m * vx),
        #                       -self.vehicle_dynamics.k_alpha4 / (self.vehicle_dynamics.m * vx)],
        #                      [-self.vehicle_dynamics.lf * self.vehicle_dynamics.k_alpha1 / self.vehicle_dynamics.Izz,
        #                       -self.vehicle_dynamics.lf * self.vehicle_dynamics.k_alpha2 / self.vehicle_dynamics.Izz,
        #                       self.vehicle_dynamics.lr * self.vehicle_dynamics.k_alpha3 / self.vehicle_dynamics.Izz,
        #                       self.vehicle_dynamics.lr * self.vehicle_dynamics.k_alpha4 / self.vehicle_dynamics.Izz]])
        # delta_matrix = torch.tensor([[delta1], [delta2], [delta3], [delta4]])

        # later_ref = torch.matmul(torch.matmul(torch.linalg.inv(I_matrix), k_matrix), delta_matrix)
        # beta_ref = 0#later_ref[0][0]
        phi_dot_ref = 0#later_ref[1][0]
        C_varphi = 2 / (self.vehicle_dynamics.m * self.vehicle_dynamics.g * self.vehicle_dynamics.lw*torch.cos(slope_points[:, 0, 0])*torch.cos(slope_points[:, 0, 1])) * \
                   (self.vehicle_dynamics.K_varphi * (1 + (self.vehicle_dynamics.ms * self.vehicle_dynamics.hr +
                                                           self.vehicle_dynamics.mu * self.vehicle_dynamics.hu) /
                                                      (self.vehicle_dynamics.ms * self.vehicle_dynamics.hs)) - (
                                self.vehicle_dynamics.ms * self.vehicle_dynamics.hr +
                                self.vehicle_dynamics.mu * self.vehicle_dynamics.hu) * self.vehicle_dynamics.g*torch.cos(slope_points[:, 0, 1]))
        C_varphi_dot = 2 * C_varphi / (self.vehicle_dynamics.m * self.vehicle_dynamics.g * self.vehicle_dynamics.lw*torch.cos(slope_points[:, 0, 0])*torch.cos(slope_points[:, 0, 1])) * \
                       ((1 + (
                                   self.vehicle_dynamics.ms * self.vehicle_dynamics.hr + self.vehicle_dynamics.mu * self.vehicle_dynamics.hu) /
                         (self.vehicle_dynamics.ms * self.vehicle_dynamics.hs)))
        I_rollover = C_varphi * varphi + C_varphi_dot * varphi_dot
        # r_action_Q = torch.sum((action[:, 0:4]/254.8) ** 2)
        # print(r_action_Q)
        # r_action_str = torch.sum((action[:, 4:]) ** 2)
        # r_action_Qdot =  0.01*torch.sum((action[:, 0:4]/100) ** 2)
        r_action_Qdot = (action[:, 0] / 100) ** 2 + (action[:, 1] / 100) ** 2+(action[:, 2] / 100 )** 2+(action[:, 3] / 100 )** 2
        # print((delta_x ** 2 + delta_y ** 2))
        r_action_strdot = (action[:, 4]/0.02) ** 2
        # kappa_constant = 0.15#vx / self.vehicle_dynamics.Rw
        # r_action_deltaQ = torch.sum((action[0, 8:16:2]) ** 2)
        # r_action_deltastr = torch.sum((action[0, 9:16:2]) ** 2)
        # r_action_deltaQdot = torch.sum((action[0, 8:16:2] - self.action_last[0, 8:16:2]) ** 2)
        # r_action_deltastrdot = torch.sum((action[0, 9:16:2] - self.action_last[0, 9:16:2]) ** 2)

        return -(
                0.04 * (delta_x ** 2 + delta_y ** 2)
                + 0.4 * delta_vx ** 2
                + 0.02 * delta_phi ** 2
                + 0.01 * (phi_dot - phi_dot_ref) ** 2
                + 0.01 * I_rollover ** 2
                # + 0.01 * r_action_Q
                # + 0.01 * r_action_str
                + 0.05 * r_action_Qdot
                + 0.05 * r_action_strdot
                # + 0.5 * (beta - beta_ref) ** 2
                # + 1e-8 * r_action_deltaQ
                # + 1e-4 * r_action_deltastr
                # + 1e-4 * r_action_deltaQdot
                # + 1e-1 * r_action_deltastrdot
        )


    def judge_done(self, obs: torch.Tensor) -> torch.Tensor:
        delta_x, delta_y, delta_phi, delta_vx = obs[:, 0]/self.obs_scale[0], obs[:, 1]/self.obs_scale[1], obs[:, 2]/self.obs_scale[2], obs[:, 3]/self.obs_scale[3],
        done = ((torch.abs(delta_x) > 5)
                |(torch.abs(delta_y) > 3)
                | (torch.abs(delta_phi) > np.pi)
                | (torch.abs(delta_vx) > 3)

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
    return FourwdstabilitycontrolModel(**kwargs)
