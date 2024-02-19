#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: vehicle 3DOF model environment
#  Update Date: 2022-04-20, Jiaxin Gao: create environment


from typing import Dict, Optional, Tuple, Union
import pandas as pd
import numpy as np
import torch

from gops.env.env_ocp.env_model.pyth_base_model import PythBaseModel
from gops.env.env_ocp.pyth_veh3dofconti import angle_normalize, VehicleDynamicsData
from gops.env.env_ocp.resources.ref_traj_model import MultiRefTrajModel
from gops.utils.gops_typing import InfoDict




class VehicleDynamicsModel(VehicleDynamicsData):
    def __init__(self):
        self.v_x = 20
        self.m1 = 5760  # Total mass of the tractor [kg]
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
        self.state_dim = 15
        self.batch_size = len(states[:, 0])
        state_next = torch.zeros_like(states)
        M_matrix = torch.zeros((self.state_dim - 2, self.state_dim - 2))
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

        A_matrix = torch.zeros((self.state_dim - 2, self.state_dim - 2))
        A_matrix[0, 0], A_matrix[0, 1], A_matrix[1, 0], A_matrix[1, 1], A_matrix[1, 2], A_matrix[1, 3], A_matrix[
            1, 6] = \
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

        B_matrix = torch.zeros((self.state_dim - 2, 1))
        B_matrix[0, 0] = self.B11
        B_matrix[1, 0] = self.B21
        B_matrix[2, 0] = self.B31

        X_dot_batch = torch.zeros((self.batch_size, self.state_dim - 2))
        for batch in range(self.batch_size):
            X_dot = (torch.matmul(torch.matmul(torch.inverse(M_matrix), A_matrix),
                                  states[batch, :self.state_dim - 2]) +
                     torch.matmul(torch.inverse(M_matrix), torch.matmul(B_matrix, actions[batch, :]))).squeeze()
            X_dot_batch[batch, :] = X_dot
        # state_next[:, :self.state_dim - 2] = state_curr[:, :self.state_dim - 2] + delta_t * X_dot_batch
        state_next[:, :self.state_dim - 3] = states[:, :self.state_dim - 3] + delta_t * X_dot_batch[:,
                                            :self.state_dim - 3]
        state_next[:, 12] = state_next[:, 11].clone() - self.b * torch.sin(
            state_next[:, 8].clone()) - self.e * torch.sin(state_next[:, 9].clone())  # posy_trailer
        state_next[:, 13] = states[:, 13] + delta_t * self.v_x  # x_tractor
        state_next[:, 14] = state_next[:, 13].clone() - self.b * torch.cos(
            state_next[:, 8].clone()) - self.e * torch.cos(state_next[:, 9].clone())  # x_trailer
        return state_next


class SimuSemiTruck7dofModel(PythBaseModel):
    def __init__(
        self,
        pre_horizon: int = 30,
        device: Union[torch.device, str, None] = None,
        path_para: Optional[Dict[str, Dict]] = None,
        u_para: Optional[Dict[str, Dict]] = None,
        max_steer: float = np.pi / 6,
        **kwargs,
    ):
        """
        you need to define parameters here
        """
        self.vehicle_dynamics = VehicleDynamicsModel()
        self.pre_horizon = pre_horizon
        ego_obs_dim = 15
        ref_obs_dim = 3
        super().__init__(
            obs_dim=ego_obs_dim + ref_obs_dim * pre_horizon*2,
            action_dim=1,
            dt=0.01,
            action_lower_bound=[-max_steer],
            action_upper_bound=[max_steer],
            device=device,
        )
        self.ref_traj = MultiRefTrajModel(path_para, u_para)
        self.action_last = 0

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        done: torch.Tensor,
        info: InfoDict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, InfoDict]:
        state = info["state"]
        ref_points = info["ref_points"]
        path_num = info["path_num"]
        u_num = info["u_num"]
        t = info["ref_time"]

        reward = self.compute_reward(obs, action)

        next_state = self.vehicle_dynamics.f_xu(state, action, self.dt)
        next_t = t + self.dt
        next_ref_points = ref_points.clone()
        next_ref_points[:, :-1] = ref_points[:, 1:]
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

        next_obs = self.get_obs(next_state, next_ref_points)

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
        })
        # self.action_last = action
        return next_obs, reward, isdone, next_info

    def get_obs(self, state, ref_points):
        ref_x_tf, ref_y_tf, ref_phi_tf = \
            ego_vehicle_coordinate_transform(
                state[:, 13], state[:, 11], state[:, 8],
                ref_points[..., 0], ref_points[..., 1], ref_points[..., 2],
            )
        ref_x2_tf, ref_y2_tf, ref_phi2_tf = \
            ego_vehicle_coordinate_transform(
                state[:, 14], state[:, 12], state[:, 9],
                ref_points[..., 0], ref_points[..., 1], ref_points[..., 2],
            )

        ego_obs = torch.concat((torch.stack(
            (ref_x_tf[:, 0], ref_y_tf[:, 0], ref_phi_tf[:, 0], ref_x2_tf[:, 0], ref_y2_tf[:, 0], ref_phi2_tf[:, 0]), dim=1),
            state[:, 0:8], state[:, 10:11]), dim=1)
        ref_obs = torch.stack((ref_x_tf, ref_y_tf, ref_phi_tf, ref_x2_tf, ref_y2_tf, ref_phi2_tf), 2)[
            :, 1:].reshape(ego_obs.shape[0], -1)
        return torch.concat((ego_obs, ref_obs), 1)

    def compute_reward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        return -(
            1 * (obs[:, 0] ** 2 + obs[:, 1] ** 2)
            + 0.9 * obs[:, 14] ** 2
            + 0.8 * obs[:, 2] ** 2
            + 0.5 * obs[:, 8] ** 2
            + 0.5 * obs[:, 7] ** 2
            + 0.5 * obs[:, 9] ** 2
            + 0.5 * obs[:, 10] ** 2
            + 0.4 * action ** 2
            + 2.0 * (action - self.action_last) ** 2)

    def judge_done(self, obs: torch.Tensor) -> torch.Tensor:
        delta_x, delta_y, delta_phi = obs[:, 0], obs[:, 1], obs[:, 2]
        done = (
            (torch.abs(delta_x) > 3)
            | (torch.abs(delta_y) > 2)
            | (torch.abs(delta_phi) > np.pi)
        )
        return done


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
    return SimuSemiTruck7dofModel(**kwargs)
