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
from casadi import *
from torch.autograd.functional import jacobian
from gops.env.env_ocp.env_model.pyth_base_model import PythBaseModel
from gops.env.env_ocp.pyth_semitruck6dofin import angle_normalize, VehicleDynamicsData
from gops.env.env_ocp.resources.ref_traj_model import MultiRefTrajModel
from gops.utils.gops_typing import InfoDict


class VehicleDynamicsModel(VehicleDynamicsData):
    def __init__(self):
        DynamicsData = VehicleDynamicsData()
        self.m_tt = DynamicsData.vehicle_params["m_tt"]
        self.ms_tt = DynamicsData.vehicle_params["ms_tt"]
        self.m_tl = DynamicsData.vehicle_params["m_tl"]
        self.ms_tl = DynamicsData.vehicle_params["ms_tl"]
        self.gravity = DynamicsData.vehicle_params["gravity"]
        self.Rw = DynamicsData.vehicle_params["Rw"]
        self.lw = DynamicsData.vehicle_params["lw"]
        self.a = DynamicsData.vehicle_params["a"]
        self.b = DynamicsData.vehicle_params["b"]
        self.lhtt = DynamicsData.vehicle_params["lhtt"]
        self.lhtl = DynamicsData.vehicle_params["lhtl"]
        self.d = DynamicsData.vehicle_params["d"]
        self.hs_tt = DynamicsData.vehicle_params["hs_tt"]
        self.hs_tl = DynamicsData.vehicle_params["hs_tl"]
        self.hh_tt = DynamicsData.vehicle_params["hh_tt"]
        self.hh_tl = DynamicsData.vehicle_params["hh_tl"]
        self.Izz_tt = DynamicsData.vehicle_params["Izz_tt"]
        self.Ixx_tt = DynamicsData.vehicle_params["Ixx_tt"]
        self.Ixz_tt = DynamicsData.vehicle_params["Ixz_tt"]
        self.Izz_tl = DynamicsData.vehicle_params["Izz_tl"]
        self.Ixx_tl = DynamicsData.vehicle_params["Ixx_tl"]
        self.Ixz_tl = DynamicsData.vehicle_params["Ixz_tl"]

        self.kf = DynamicsData.vehicle_params["kf"]
        self.km = DynamicsData.vehicle_params["km"]
        self.kr = DynamicsData.vehicle_params["kr"]
        self.kvarphi_tt = DynamicsData.vehicle_params["kvarphi_tt"]
        self.kvarphi_tl = DynamicsData.vehicle_params["kvarphi_tl"]
        self.ka = DynamicsData.vehicle_params["ka"]
        self.cvarphi_tt = DynamicsData.vehicle_params["cvarphi_tt"]
        self.cvarphi_tl = DynamicsData.vehicle_params["cvarphi_tl"]
        self.state_dim = DynamicsData.vehicle_params["state_dim"]

        divided_tt = self.m_tt * self.Ixx_tt * self.Izz_tt - self.Izz_tt * self.ms_tt ** 2 * self.hs_tt ** 2 - self.m_tt * self.Ixz_tt ** 2
        divided_tl = self.m_tl * self.Ixx_tl * self.Izz_tl - self.Izz_tl * self.ms_tl ** 2 * self.hs_tl ** 2 - self.m_tl * self.Ixz_tl ** 2
        self.Att13 = -self.ms_tt * self.hs_tt * self.Izz_tt * (
                self.kvarphi_tt - self.ms_tt * self.gravity * self.hs_tt - self.ka) / divided_tt
        self.Att14 = -self.ms_tt * self.hs_tt * self.Izz_tt * self.cvarphi_tt / divided_tt

        self.Att23 = -self.m_tt * self.Ixz_tt * (
                self.kvarphi_tt - self.ms_tt * self.gravity * self.hs_tt - self.ka) / divided_tt
        self.Att24 = -self.m_tt * self.Ixz_tt * self.cvarphi_tt / divided_tt

        self.Att34 = 1
        self.Att43 = -self.m_tt * self.Izz_tt * (
                    self.kvarphi_tt - self.ms_tt * self.gravity * self.hs_tt - self.ka) / divided_tt
        self.Att44 = -self.m_tt * self.Izz_tt * self.cvarphi_tt / divided_tt

        self.Btt00 = 1 / self.m_tt

        self.Btt11 = (self.Ixx_tt * self.Izz_tt - self.Ixz_tt ** 2) / divided_tt
        self.Btt12 = (self.Ixz_tt * self.ms_tt * self.hs_tt) / divided_tt

        self.Btt21 = (self.Ixz_tt * self.ms_tt * self.hs_tt) / divided_tt
        self.Btt22 = (self.m_tt * self.Ixx_tt - self.ms_tt ** 2 * self.hs_tt ** 2) / divided_tt

        self.Btt41 = (self.Izz_tt * self.ms_tt * self.hs_tt) / divided_tt
        self.Btt42 = (self.Ixz_tt * self.m_tt) / divided_tt

        self.Ctt00 = 1 / self.m_tt

        self.Ctt11 = (-self.ms_tt * self.hs_tt * self.hh_tt * self.Izz_tt
                      - self.ms_tt * self.hs_tt * self.Ixz_tt * self.lhtt
                      + self.Ixx_tt * self.Izz_tt
                      - self.Ixz_tt ** 2) / divided_tt
        self.Ctt12 = (-self.ms_tt * self.hs_tt * self.Ixz_tt) / divided_tt

        self.Ctt21 = (self.ms_tt * self.hs_tt * self.Ixz_tt
                      - self.m_tt * self.hh_tt * self.Ixz_tt
                      - self.lhtt * self.m_tt * self.Ixx_tt
                      + self.lhtt * self.ms_tt ** 2 * self.hs_tt ** 2) / divided_tt
        self.Ctt22 = (self.ms_tt ** 2 * self.hs_tt ** 2 - self.m_tt * self.Ixx_tt) / divided_tt

        self.Ctt41 = (self.ms_tt * self.hs_tt * self.Izz_tt
                      - self.m_tt * self.hh_tt * self.Izz_tt
                      - self.lhtt * self.m_tt * self.Ixz_tt
                      + self.lhtt * self.ms_tt ** 2 * self.hs_tt ** 2) / divided_tt
        self.Ctt42 = -self.m_tt * self.Ixz_tt / divided_tt

        self.Gtt13 = -self.ms_tt * self.hs_tt * self.Izz_tt * self.ka / divided_tt

        self.Gtt23 = -self.m_tt * self.Ixz_tt * self.ka / divided_tt

        self.Gtt43 = -self.m_tt * self.Izz_tt * self.ka / divided_tt

        # ---------------------trilaer
        self.Atl13 = -self.ms_tl * self.hs_tl * self.Izz_tl * (
                self.kvarphi_tl - self.ms_tl * self.gravity * self.hs_tl - self.ka) / divided_tl
        self.Atl14 = -self.ms_tl * self.hs_tl * self.Izz_tl * self.cvarphi_tl / divided_tl

        self.Atl23 = -self.m_tl * self.Ixz_tl * (
                self.kvarphi_tl - self.ms_tl * self.gravity * self.hs_tl - self.ka) / divided_tl
        self.Atl24 = -self.m_tl * self.Ixz_tl * self.cvarphi_tl / divided_tl

        self.Atl34 = 1

        self.Atl43 = -self.m_tl * self.Izz_tl * (
                self.kvarphi_tl - self.ms_tl * self.gravity * self.hs_tl - self.ka) / divided_tl
        self.Atl44 = -self.m_tl * self.Izz_tl * self.cvarphi_tl / divided_tl

        self.Btl00 = 1 / self.m_tl

        self.Btl11 = (self.Ixx_tl * self.Izz_tl - self.Ixz_tl ** 2) / divided_tl
        self.Btl12 = (self.Ixz_tl * self.ms_tl * self.hs_tl) / divided_tl

        self.Btl21 = (self.Ixz_tl * self.ms_tl * self.hs_tl) / divided_tl
        self.Btl22 = (self.m_tl * self.Ixx_tl - self.ms_tl ** 2 * self.hs_tl ** 2) / divided_tl

        self.Btl41 = (self.Izz_tl * self.ms_tl * self.hs_tl) / divided_tl
        self.Btl42 = (self.Ixz_tl * self.m_tl) / divided_tl

        self.Ctl00 = 1 / self.m_tl

        self.Ctl11 = (-self.ms_tl * self.hs_tl * self.hh_tl * self.Izz_tl
                      - self.ms_tl * self.hs_tl * self.Ixz_tl * self.lhtl
                      + self.Ixx_tl * self.Izz_tl
                      - self.Ixz_tl ** 2) / divided_tl
        self.Ctl12 = (-self.ms_tl * self.hs_tl * self.Ixz_tl) / divided_tl

        self.Ctl21 = (self.ms_tl * self.hs_tl * self.Ixz_tl
                      - self.m_tl * self.hh_tl * self.Ixz_tl
                      - self.lhtl * self.m_tl * self.Ixx_tl
                      + self.lhtl * self.ms_tl ** 2 * self.hs_tl ** 2) / divided_tl
        self.Ctl22 = (self.ms_tl ** 2 * self.hs_tl ** 2 - self.m_tl * self.Ixx_tl) / divided_tl

        self.Ctl41 = (self.ms_tl * self.hs_tl * self.Izz_tl
                      - self.m_tl * self.hh_tl * self.Izz_tl
                      - self.lhtl * self.m_tl * self.Ixz_tl
                      + self.lhtl * self.ms_tl ** 2 * self.hs_tl ** 2) / divided_tl
        self.Ctl42 = -self.m_tl * self.Ixz_tl / divided_tl

        self.Gtl13 = -self.ms_tl * self.hs_tl * self.Izz_tl * self.ka / divided_tl

        self.Gtl23 = -self.m_tl * self.Ixz_tl * self.ka / divided_tl

        self.Gtl43 = -self.m_tl * self.Izz_tl * self.ka / divided_tl

        self.M00, self.M05 = 1, -1
        self.M11, self.M12, self.M16, self.M17 = 1, -self.lhtt, -1, -self.lhtl
        self.N00, self.N03 = 1, 1
        self.N11, self.N14, self.N22, self.N25 = 1, 1, 1, 1
        self.N35 = 1
        self.Q33, self.Q38 = -self.ka, self.ka

    def dynamic_func(self, states, actions):
        X_matrix = torch.cat((states[:, 3:4], states[:, 8:12], states[:, 7:8],
                              states[:, 12:16]), 1).reshape(10, 1)
        U_matrix = actions.reshape(3, 1)
        u_tt = states[:, 3]
        u_tl = states[:, 7]
        self.Att12 = -u_tt
        Att_matrix = torch.zeros((5, 5))
        Att_matrix[1, 2] = self.Att12
        Att_matrix[1, 3] = self.Att13
        Att_matrix[1, 4] = self.Att14

        Att_matrix[2, 3], Att_matrix[2, 4] = self.Att23, self.Att24

        Att_matrix[3, 4] = self.Att34

        Att_matrix[4, 3], Att_matrix[4, 4] = self.Att43, self.Att44

        Btt_matrix = torch.zeros((5, 3))
        Btt_matrix[0, 0] = self.Btt00

        Btt_matrix[1, 1], Btt_matrix[1, 2] = self.Btt11, self.Btt12

        Btt_matrix[2, 1], Btt_matrix[2, 2] = self.Btt21, self.Btt22
        Btt_matrix[4, 1], Btt_matrix[4, 2] = self.Btt41, self.Btt42

        Ctt_matrix = torch.zeros((5, 3))
        Ctt_matrix[0, 0] = self.Ctt00

        Ctt_matrix[1, 1], Ctt_matrix[1, 2] = self.Ctt11, self.Ctt12

        Ctt_matrix[2, 1], Ctt_matrix[2, 2] = self.Ctt21, self.Ctt22
        Ctt_matrix[4, 1], Ctt_matrix[4, 2] = self.Ctt41, self.Ctt42

        Gtt_matrix = torch.zeros((5, 5))
        Gtt_matrix[1, 3] = self.Gtt13

        Gtt_matrix[2, 3] = self.Gtt23

        Gtt_matrix[4, 3] = self.Gtt43

        # --------trailer---------
        self.Atl12 = -u_tl

        Atl_matrix = torch.zeros((5, 5))
        Atl_matrix[1, 2] = self.Atl12
        Atl_matrix[1, 3] = self.Atl13
        Atl_matrix[1, 4] = self.Atl14

        Atl_matrix[2, 3], Atl_matrix[2, 4] = self.Atl23, self.Atl24

        Atl_matrix[3, 4] = self.Atl34

        Atl_matrix[4, 3], Atl_matrix[4, 4] = self.Atl43, self.Atl44

        Btl_matrix = torch.zeros((5, 3))
        Btl_matrix[0, 0] = self.Btl00

        Btl_matrix[1, 1], Btl_matrix[1, 2] = self.Btl11, self.Btl12

        Btl_matrix[2, 1], Btl_matrix[2, 2] = self.Btl21, self.Btl22
        Btl_matrix[4, 1], Btl_matrix[4, 2] = self.Btl41, self.Btl42

        Ctl_matrix = torch.zeros((5, 3))
        Ctl_matrix[0, 0] = self.Ctl00

        Ctl_matrix[1, 1], Ctl_matrix[1, 2] = self.Ctl11, self.Ctl12

        Ctl_matrix[2, 1], Ctl_matrix[2, 2] = self.Ctl21, self.Ctl22
        Ctl_matrix[4, 1], Ctl_matrix[4, 2] = self.Ctl41, self.Ctl42

        Gtl_matrix = torch.zeros((5, 5))
        Gtl_matrix[1, 3] = self.Gtl13

        Gtl_matrix[2, 3] = self.Gtl23

        Gtl_matrix[4, 3] = self.Gtl43

        A_matrix = torch.block_diag(Att_matrix,
                                    Atl_matrix)

        B_matrix = torch.block_diag(Btt_matrix,
                                    Btl_matrix)

        C_matrix = torch.block_diag(Ctt_matrix,
                                    Ctl_matrix)

        G_matrix = torch.block_diag(Gtt_matrix, Gtl_matrix)


        M_matrix = torch.zeros((2, 10))
        M_matrix[0, 0], M_matrix[0, 5] = self.M00, self.M05
        # M_matrix[1, 0] = states[9]-states[13]
        M_matrix[1, 1], M_matrix[1, 2] = self.M11, self.M12
        M_matrix[1, 6], M_matrix[1, 7] = self.M16, self.M17

        self.P12, self.P17 = -u_tt, u_tt
        P_matrix = torch.zeros((2, 10))
        P_matrix[1, 2], P_matrix[1, 7] = self.P12, self.P17

        N_matrix = torch.zeros((4, 6))
        N_matrix[0, 0], N_matrix[0, 3] = self.N00, self.N03
        N_matrix[1, 1], N_matrix[1, 4] = self.N11, self.N14
        N_matrix[2, 2], N_matrix[2, 5] = self.N22, self.N25
        N_matrix[3, 5] = self.N35

        Q_matrix = torch.zeros((4, 10))
        Q_matrix[3, 3], Q_matrix[3, 8] = self.Q33, self.Q38

        J_matrix = torch.inverse(torch.vstack((N_matrix, torch.matmul(M_matrix, C_matrix))))

        K1_matrix = torch.vstack((Q_matrix,
                                  (P_matrix - torch.matmul(M_matrix, A_matrix)
                                   -torch.matmul(M_matrix, G_matrix))))
        K2_matrix = torch.vstack((torch.zeros((4, 6)), (torch.matmul(M_matrix, B_matrix))))

        Lc_matrix = torch.zeros((6, 12))
        Lc_matrix[0, 0], Lc_matrix[0, 2], \
        Lc_matrix[0, 4], Lc_matrix[0, 6] = 1, 1, 1, 1

        Lc_matrix[1, 1], Lc_matrix[1, 3], \
        Lc_matrix[1, 5], Lc_matrix[1, 7], = 1, 1, 1, 1

        Lc_matrix[2, 0], Lc_matrix[2, 1], Lc_matrix[2, 2], Lc_matrix[2, 3], \
        Lc_matrix[2, 4], Lc_matrix[2, 5], Lc_matrix[2, 6], Lc_matrix[2, 7] = \
            -self.lw / 2, self.a, self.lw / 2, self.a, \
            -self.lw / 2, -self.b, self.lw / 2, -self.b
        Lc_matrix[3, 8], Lc_matrix[3, 10] = 1, 1

        Lc_matrix[4, 9], Lc_matrix[4, 11] = 1, 1

        Lc_matrix[5, 8], Lc_matrix[5, 9], Lc_matrix[5, 10], Lc_matrix[5, 11] = \
            -self.lw / 2, -self.d, \
            self.lw / 2, -self.d

        delta = actions[:, 2]
        Mw1 = torch.tensor([[torch.cos(delta), -torch.sin(delta)],
                            [torch.sin(delta), torch.cos(delta)]])
        Mw2 = torch.tensor([[torch.cos(delta), -torch.sin(delta)],
                            [torch.sin(delta), torch.cos(delta)]])
        Mw3 = torch.tensor([[1, 0],
                            [0, 1]])
        Mw4 = torch.tensor([[1, 0],
                            [0, 1]])
        Mw5 = torch.tensor([[1, 0],
                            [0, 1]])
        Mw6 = torch.tensor([[1, 0],
                            [0, 1]])

        Mw_matrix = torch.block_diag(Mw1, Mw2, Mw3, Mw4, Mw5, Mw6)

        At_matrix = torch.zeros((12, 10))

        At_matrix[1, 1], At_matrix[1, 2] = -self.kf / u_tt, -self.kf * self.a / u_tt

        At_matrix[3, 1], At_matrix[3, 2] = -self.kf / u_tt, -self.kf * self.a / u_tt

        At_matrix[5, 1], At_matrix[5, 2] = -self.km / u_tt, -self.km * (-self.b) / u_tt

        At_matrix[7, 1], At_matrix[7, 2] = -self.km / u_tt, -self.km * (-self.b) / u_tt

        At_matrix[9, 6], At_matrix[9, 7] = -self.kr / u_tl, -self.kr * (-self.d) / u_tl

        At_matrix[11, 6], At_matrix[11, 7] = -self.kr / u_tl, -self.kr * (-self.d) / u_tl

        Bt_matrix = torch.zeros((12, 3))
        Bt_matrix[1, 1], Bt_matrix[3, 1] = self.kf, self.kf
        Bt_matrix[4, 0], Bt_matrix[6, 1] = 1 / self.Rw, 1 / self.Rw

        temp = torch.matmul(At_matrix, X_matrix) + torch.matmul(Bt_matrix, U_matrix)
        FCG = torch.matmul(Lc_matrix, torch.matmul(Mw_matrix, temp))
        X_dot = (torch.matmul((A_matrix + torch.matmul(C_matrix, torch.matmul(J_matrix, K1_matrix))), X_matrix)
                 + torch.matmul((B_matrix - torch.matmul(C_matrix, torch.matmul(J_matrix, K2_matrix))), FCG)).squeeze()
        X_all_dot = torch.concat((u_tt * torch.cos(states[:, 2].clone()) - states[:, 8].clone() * torch.sin(states[:, 2].clone()),
                                  u_tt * torch.sin(states[:, 2].clone()) + states[:, 8].clone() * torch.cos(states[:, 2].clone()),
                                  states[:, 9].clone(),
                                  X_dot[0], u_tl * torch.cos(states[:, 6].clone()) - states[:, 12].clone() * torch.sin(states[:, 6].clone()),
                                  u_tl * torch.sin(states[:, 6].clone()) + states[:, 12] * torch.cos(states[:, 6].clone()),
                                  states[:, 13],
                                  X_dot[5],
                                  X_dot[1:5],
                                  X_dot[6:10],
        ))
        return X_all_dot

    def f_xu(self, states, actions, delta_t):
        state_next = torch.zeros_like(states)
        # RK = 1
        # if RK == 1:
        #     X_all_dot = self.dynamic_func(states, actions)
        #     state_next = delta_t * X_all_dot
        #
        # elif RK == 2:
        #     X_all_dot = self.dynamic_func(states, actions)
        #     K1 = delta_t * X_all_dot
        #     x_2 = states + K1
        #     f_xu_2 = self.dynamic_func(x_2, actions)
        #     K2 = delta_t * f_xu_2
        #     state_next = states + (K1 + K2) / 2
        # elif RK == 4:
        #     X_all_dot = self.dynamic_func(states, actions)
        #     K1 = delta_t * X_all_dot
        #     x_2 = states + K1 / 2
        #     f_xu_2 = self.dynamic_func(x_2, actions)
        #     K2 = delta_t * f_xu_2
        #     x_3 = states + K2 / 2
        #     f_xu_3 = self.dynamic_func(x_3, actions)
        #     K3 = delta_t * f_xu_3
        #     x_4 = states + K3
        #     f_xu_4 = self.dynamic_func(x_4, actions)
        #     K4 = delta_t * f_xu_4
        #     state_next = states + (K1 + 2 * K2 + 2 * K3 + K4) / 6
        X_matrix = torch.cat((states[:, 3:4], states[:, 8:12], states[:, 7:8],
                              states[:, 12:16]), 1).reshape(10, 1)
        U_matrix = actions.reshape(3, 1)
        u_tt = states[:, 3]
        u_tl = states[:, 7]
        self.Att12 = -u_tt
        Att_matrix = torch.zeros((5, 5))
        Att_matrix[1, 2] = self.Att12
        Att_matrix[1, 3] = self.Att13
        Att_matrix[1, 4] = self.Att14

        Att_matrix[2, 3], Att_matrix[2, 4] = self.Att23, self.Att24

        Att_matrix[3, 4] = self.Att34

        Att_matrix[4, 3], Att_matrix[4, 4] = self.Att43, self.Att44

        Btt_matrix = torch.zeros((5, 3))
        Btt_matrix[0, 0] = self.Btt00

        Btt_matrix[1, 1], Btt_matrix[1, 2] = self.Btt11, self.Btt12

        Btt_matrix[2, 1], Btt_matrix[2, 2] = self.Btt21, self.Btt22
        Btt_matrix[4, 1], Btt_matrix[4, 2] = self.Btt41, self.Btt42

        Ctt_matrix = torch.zeros((5, 3))
        Ctt_matrix[0, 0] = self.Ctt00

        Ctt_matrix[1, 1], Ctt_matrix[1, 2] = self.Ctt11, self.Ctt12

        Ctt_matrix[2, 1], Ctt_matrix[2, 2] = self.Ctt21, self.Ctt22
        Ctt_matrix[4, 1], Ctt_matrix[4, 2] = self.Ctt41, self.Ctt42

        Gtt_matrix = torch.zeros((5, 5))
        Gtt_matrix[1, 3] = self.Gtt13

        Gtt_matrix[2, 3] = self.Gtt23

        Gtt_matrix[4, 3] = self.Gtt43

        # --------trailer---------
        self.Atl12 = -u_tl

        Atl_matrix = torch.zeros((5, 5))
        Atl_matrix[1, 2] = self.Atl12
        Atl_matrix[1, 3] = self.Atl13
        Atl_matrix[1, 4] = self.Atl14

        Atl_matrix[2, 3], Atl_matrix[2, 4] = self.Atl23, self.Atl24

        Atl_matrix[3, 4] = self.Atl34

        Atl_matrix[4, 3], Atl_matrix[4, 4] = self.Atl43, self.Atl44

        Btl_matrix = torch.zeros((5, 3))
        Btl_matrix[0, 0] = self.Btl00

        Btl_matrix[1, 1], Btl_matrix[1, 2] = self.Btl11, self.Btl12

        Btl_matrix[2, 1], Btl_matrix[2, 2] = self.Btl21, self.Btl22
        Btl_matrix[4, 1], Btl_matrix[4, 2] = self.Btl41, self.Btl42

        Ctl_matrix = torch.zeros((5, 3))
        Ctl_matrix[0, 0] = self.Ctl00

        Ctl_matrix[1, 1], Ctl_matrix[1, 2] = self.Ctl11, self.Ctl12

        Ctl_matrix[2, 1], Ctl_matrix[2, 2] = self.Ctl21, self.Ctl22
        Ctl_matrix[4, 1], Ctl_matrix[4, 2] = self.Ctl41, self.Ctl42

        Gtl_matrix = torch.zeros((5, 5))
        Gtl_matrix[1, 3] = self.Gtl13

        Gtl_matrix[2, 3] = self.Gtl23

        Gtl_matrix[4, 3] = self.Gtl43

        A_matrix = torch.block_diag(Att_matrix,
                                    Atl_matrix)

        B_matrix = torch.block_diag(Btt_matrix,
                                    Btl_matrix)

        C_matrix = torch.block_diag(Ctt_matrix,
                                    Ctl_matrix)

        G_matrix = torch.block_diag(Gtt_matrix, Gtl_matrix)

        M_matrix = torch.zeros((2, 10))
        M_matrix[0, 0], M_matrix[0, 5] = self.M00, self.M05
        # M_matrix[1, 0] = states[9]-states[13]
        M_matrix[1, 1], M_matrix[1, 2] = self.M11, self.M12
        M_matrix[1, 6], M_matrix[1, 7] = self.M16, self.M17

        self.P12, self.P17 = -u_tt, u_tt
        P_matrix = torch.zeros((2, 10))
        P_matrix[1, 2], P_matrix[1, 7] = self.P12, self.P17

        N_matrix = torch.zeros((4, 6))
        N_matrix[0, 0], N_matrix[0, 3] = self.N00, self.N03
        N_matrix[1, 1], N_matrix[1, 4] = self.N11, self.N14
        N_matrix[2, 2], N_matrix[2, 5] = self.N22, self.N25
        N_matrix[3, 5] = self.N35

        Q_matrix = torch.zeros((4, 10))
        Q_matrix[3, 3], Q_matrix[3, 8] = self.Q33, self.Q38

        J_matrix = torch.inverse(torch.vstack((N_matrix, torch.matmul(M_matrix, C_matrix))))

        K1_matrix = torch.vstack((Q_matrix,
                                  (P_matrix - torch.matmul(M_matrix, A_matrix)
                                   - torch.matmul(M_matrix, G_matrix))))
        K2_matrix = torch.vstack((torch.zeros((4, 6)), (torch.matmul(M_matrix, B_matrix))))

        Lc_matrix = torch.zeros((6, 12))
        Lc_matrix[0, 0], Lc_matrix[0, 2], \
        Lc_matrix[0, 4], Lc_matrix[0, 6] = 1, 1, 1, 1

        Lc_matrix[1, 1], Lc_matrix[1, 3], \
        Lc_matrix[1, 5], Lc_matrix[1, 7], = 1, 1, 1, 1

        Lc_matrix[2, 0], Lc_matrix[2, 1], Lc_matrix[2, 2], Lc_matrix[2, 3], \
        Lc_matrix[2, 4], Lc_matrix[2, 5], Lc_matrix[2, 6], Lc_matrix[2, 7] = \
            -self.lw / 2, self.a, self.lw / 2, self.a, \
            -self.lw / 2, -self.b, self.lw / 2, -self.b
        Lc_matrix[3, 8], Lc_matrix[3, 10] = 1, 1

        Lc_matrix[4, 9], Lc_matrix[4, 11] = 1, 1

        Lc_matrix[5, 8], Lc_matrix[5, 9], Lc_matrix[5, 10], Lc_matrix[5, 11] = \
            -self.lw / 2, -self.d, \
            self.lw / 2, -self.d

        delta = actions[:, 2]
        Mw1 = torch.tensor([[torch.cos(delta), -torch.sin(delta)],
                            [torch.sin(delta), torch.cos(delta)]])
        Mw2 = torch.tensor([[torch.cos(delta), -torch.sin(delta)],
                            [torch.sin(delta), torch.cos(delta)]])
        Mw3 = torch.tensor([[1, 0],
                            [0, 1]])
        Mw4 = torch.tensor([[1, 0],
                            [0, 1]])
        Mw5 = torch.tensor([[1, 0],
                            [0, 1]])
        Mw6 = torch.tensor([[1, 0],
                            [0, 1]])

        Mw_matrix = torch.block_diag(Mw1, Mw2, Mw3, Mw4, Mw5, Mw6)

        At_matrix = torch.zeros((12, 10))

        At_matrix[1, 1], At_matrix[1, 2] = -self.kf / u_tt, -self.kf * self.a / u_tt

        At_matrix[3, 1], At_matrix[3, 2] = -self.kf / u_tt, -self.kf * self.a / u_tt

        At_matrix[5, 1], At_matrix[5, 2] = -self.km / u_tt, -self.km * (-self.b) / u_tt

        At_matrix[7, 1], At_matrix[7, 2] = -self.km / u_tt, -self.km * (-self.b) / u_tt

        At_matrix[9, 6], At_matrix[9, 7] = -self.kr / u_tl, -self.kr * (-self.d) / u_tl

        At_matrix[11, 6], At_matrix[11, 7] = -self.kr / u_tl, -self.kr * (-self.d) / u_tl

        Bt_matrix = torch.zeros((12, 3))
        Bt_matrix[1, 1], Bt_matrix[3, 1] = self.kf, self.kf
        Bt_matrix[4, 0], Bt_matrix[6, 1] = 1 / self.Rw, 1 / self.Rw

        temp = torch.matmul(At_matrix, X_matrix) + torch.matmul(Bt_matrix, U_matrix)
        FCG = torch.matmul(Lc_matrix, torch.matmul(Mw_matrix, temp))
        X_dot = (torch.matmul((A_matrix + torch.matmul(C_matrix, torch.matmul(J_matrix, K1_matrix))), X_matrix)
                 + torch.matmul((B_matrix - torch.matmul(C_matrix, torch.matmul(J_matrix, K2_matrix))), FCG)).squeeze()
        state_next[:, 0] = states[:, 0] + delta_t * (
                    u_tt * torch.cos(states[:, 2].clone()) - states[:, 8].clone() * torch.sin(
                states[:, 2].clone()))  # x_tractor
        state_next[:, 1] = states[:, 1] + delta_t * (
                    u_tt * torch.sin(states[:, 2].clone()) + states[:, 8].clone() * torch.cos(states[:, 2].clone()))
        state_next[:, 2] = states[:, 2] + delta_t * states[:, 9].clone()
        state_next[:, 3] = states[:, 3] + delta_t * X_dot[0]  # u_tt
        state_next[:, 4] = states[:, 4] + delta_t * (
                    u_tl * torch.cos(states[:, 6].clone()) - states[:, 12].clone() * torch.sin(
                states[:, 6].clone()))  # px_tl
        state_next[:, 5] = states[:, 5] + delta_t * (
                    u_tl * torch.sin(states[:, 6].clone()) + states[:, 12] * torch.cos(states[:, 6].clone()))  # py_tt
        state_next[:, 6] = states[:, 6] + delta_t * states[:, 13]  # varphi_tl
        state_next[:, 7] = states[:, 7] + delta_t * X_dot[5]  # u_tl
        state_next[:, 8:12] = states[:, 8:12] + delta_t * X_dot[1:5]
        state_next[:, 12:16] = states[:, 12:16] + delta_t * X_dot[6:10]
        state_next[:, 16:19] = actions
        return state_next

    def auxvar_setting(self, delta_t=0.01, ref_traj_init=np.array([0, 0, 0, 0])):
        parameter = []
        self.wdis = SX.sym('wdis')
        parameter += [self.wdis]
        self.wu1 = SX.sym('wu1')
        parameter += [self.wu1]
        self.wphi1 = SX.sym('wphi1')
        parameter += [self.wphi1]
        self.wphidot1 = SX.sym('wphidot1')
        parameter += [self.wphidot1]
        self.wvarphi1 = SX.sym('wvarphi1')
        parameter += [self.wvarphi1]
        self.wvarphi1dot = SX.sym('wvarphi1dot')
        parameter += [self.wvarphi1dot]
        self.wstrdot = SX.sym('wstrdot')
        parameter += [self.wstrdot]
        self.wQdot = SX.sym('wQdot')
        parameter += [self.wQdot]
        self.cost_auxvar = vcat(parameter)

        self.x1, self.y1, self.phi1, self.u1, \
        self.x2,self.y2, self.phi2,self.u2, \
        self.v1, self.phi1dot, self.varphi1, self.varphi1dot, \
        self.v2, self.phi2dot, self.varphi2, self.varphi2dot = \
        SX.sym('x1'), SX.sym('y1'), SX.sym('phi1'),SX.sym('u1'), \
        SX.sym('x2'),SX.sym('y2'), SX.sym('phi2'), SX.sym('u2'), \
        SX.sym('v1'), SX.sym('phi1dot'), SX.sym('varphi1'), SX.sym('varphi1dot'), \
        SX.sym('v2'), SX.sym('phi2dot'), SX.sym('varphi2'), SX.sym('varphi2dot')

        X_matrix = vertcat(self.u1, self.v1, self.phi1dot, self.varphi1, self.varphi1dot,
                         self.u2, self.v2, self.phi2dot, self.varphi2, self.varphi2dot)
        self.steering = SX.sym('str')
        self.Q3 = SX.sym('Q3')
        self.Q4 = SX.sym('Q4')
        self.X = vertcat(self.x1, self.y1, self.phi1, self.u1,
                         self.x2, self.y2, self.phi2, self.u2,
                         self.v1, self.phi1dot, self.varphi1, self.varphi1dot,
                         self.v2, self.phi2dot, self.varphi2, self.varphi2dot, self.Q3, self.Q4, self.steering)

        self.U = vertcat(self.Q3, self.Q4, self.steering)

        self.Att12 = -self.m_tt * self.u1

        self.Att42 = self.ms_tt * self.hs_tt * self.u1

        Mtt_matrix = SX.zeros((5, 5))
        Mtt_matrix[0, 0] = self.Mtt00

        Mtt_matrix[1, 1] = self.Mtt11

        Mtt_matrix[1, 4] = self.Mtt14

        Mtt_matrix[2, 2] = self.Mtt22

        Mtt_matrix[3, 3] = self.Mtt33

        Mtt_matrix[4, 1] = self.Mtt41
        Mtt_matrix[4, 4] = self.Mtt44

        Att_matrix = SX.zeros((5, 5))
        Att_matrix[1, 2] = self.Att12
        Att_matrix[3, 4] = self.Att34
        Att_matrix[4, 2] = self.Att42
        Att_matrix[4, 3], Att_matrix[4, 4] = self.Att43, self.Att44

        Btt_matrix = SX.zeros((5, 3))
        Btt_matrix[0, 0] = self.Btt00
        Btt_matrix[1, 1], Btt_matrix[2, 2] = self.Btt11, self.Btt22

        Ctt_matrix = SX.zeros((5, 4))
        Ctt_matrix[0, 0], Ctt_matrix[1, 1] = self.Ctt00, self.Ctt11
        Ctt_matrix[2, 1], Ctt_matrix[2, 2] = self.Ctt21, self.Ctt22
        Ctt_matrix[4, 1], Ctt_matrix[4, 3] = self.Ctt41, self.Ctt43

        self.Atl12 = -self.m_tl * self.u2

        self.Atl42 = self.ms_tl * self.hs_tl * self.u2

        Mtl_matrix = SX.zeros((5, 5))
        Mtl_matrix[0, 0] = self.Mtl00

        Mtl_matrix[1, 1] = self.Mtl11

        Mtl_matrix[1, 4] = self.Mtl14

        Mtl_matrix[2, 2] = self.Mtl22

        Mtl_matrix[3, 3] = self.Mtl33

        Mtl_matrix[4, 1] = self.Mtl41
        Mtl_matrix[4, 4] = self.Mtl44

        Atl_matrix = SX.zeros((5, 5))
        Atl_matrix[1, 2] = self.Atl12
        Atl_matrix[3, 4] = self.Atl34
        Atl_matrix[4, 2] = self.Atl42
        Atl_matrix[4, 3], Atl_matrix[4, 4] = self.Atl43, self.Atl44

        Btl_matrix = SX.zeros((5, 3))
        Btl_matrix[0, 0] = self.Btl00

        Btl_matrix[1, 1], Btl_matrix[2, 2] = self.Btl11, self.Btl22

        Ctl_matrix = SX.zeros((5, 4))
        Ctl_matrix[0, 0], Ctl_matrix[1, 1] = self.Ctl00, self.Ctl11
        Ctl_matrix[2, 1], Ctl_matrix[2, 2] = self.Ctl21, self.Ctl22
        Ctl_matrix[4, 1], Ctl_matrix[4, 3] = self.Ctl41, self.Ctl43

        A_matrix = diagcat(mtimes(inv(Mtt_matrix), Att_matrix),
                                    mtimes(inv(Mtl_matrix), Atl_matrix))

        B_matrix = diagcat(mtimes(inv(Mtt_matrix), Btt_matrix),
                                    mtimes(inv(Mtl_matrix), Btl_matrix))

        C_matrix = diagcat(mtimes(inv(Mtt_matrix), Ctt_matrix),
                                    mtimes(inv(Mtl_matrix), Ctl_matrix))

        M_matrix = SX.zeros((2, 10))
        M_matrix[0, 0], M_matrix[0, 5] = self.M00, self.M05
        M_matrix[1, 1], M_matrix[1, 2] = self.M11, self.M12
        M_matrix[1, 6], M_matrix[1, 7] = self.M16, self.M17

        self.P12, self.P17 = -self.u1, self.u1
        P_matrix = SX.zeros((2, 10))
        P_matrix[1, 2], P_matrix[1, 7] = self.P12, self.P17

        N_matrix = SX.zeros((6, 8))
        N_matrix[0, 0], N_matrix[0, 4] = self.N00, self.N04
        N_matrix[1, 1], N_matrix[1, 5] = self.N11, self.N15
        N_matrix[2, 2], N_matrix[2, 6] = self.N22, self.N26
        N_matrix[3, 2] = self.N32
        N_matrix[4, 3], N_matrix[4, 7] = self.N43, self.N47
        N_matrix[5, 7] = self.N57

        Q_matrix = SX.zeros((6, 10))
        Q_matrix[5, 4], Q_matrix[5, 9] = self.Q54, self.Q59

        J_matrix = inv(vertcat((N_matrix, mtimes(M_matrix, C_matrix))))

        K1_matrix = vertcat((Q_matrix, (P_matrix - mtimes(M_matrix, A_matrix))))
        K2_matrix = vertcat((SX.zeros((6, 6)), (mtimes(M_matrix, B_matrix))))

        Lc_matrix = SX.zeros((6, 12))
        Lc_matrix[0, 0], Lc_matrix[0, 2], \
        Lc_matrix[0, 4], Lc_matrix[0, 6] = 1, 1, 1, 1

        Lc_matrix[1, 1], Lc_matrix[1, 3], \
        Lc_matrix[1, 5], Lc_matrix[1, 7], = 1, 1, 1, 1

        Lc_matrix[2, 0], Lc_matrix[2, 1], Lc_matrix[2, 2], Lc_matrix[2, 3], \
        Lc_matrix[2, 4], Lc_matrix[2, 5], Lc_matrix[2, 6], Lc_matrix[2, 7] = \
            -self.lw / 2, self.a, self.lw / 2, self.a, \
            -self.lw / 2, -self.b, self.lw / 2, -self.b
        Lc_matrix[3, 8], Lc_matrix[3, 10] = 1, 1

        Lc_matrix[4, 9], Lc_matrix[4, 11] = 1, 1

        Lc_matrix[5, 8], Lc_matrix[5, 9], Lc_matrix[5, 10], Lc_matrix[5, 11] = \
            -self.lw / 2, -self.d, \
            self.lw / 2, -self.d

        delta = self.U[2]
        Mw1 = horzcat(vertcat(cos(delta), -sin(delta)),
                    vertcat(sin(delta), cos(delta)))
        Mw2 = horzcat(vertcat(cos(delta), -sin(delta)),
                        vertcat(sin(delta), cos(delta)))
        Mw3 = DM.eye(2)
        Mw4 = DM.eye(2)
        Mw5 = DM.eye(2)
        Mw6 = DM.eye(2)

        Mw_matrix = diagcat(Mw1, Mw2, Mw3, Mw4, Mw5, Mw6)

        At_matrix = torch.zeros((12, 10))

        At_matrix[1, 1], At_matrix[1, 2] = -self.kf / self.u1, -self.kf * self.a / self.u1

        At_matrix[3, 1], At_matrix[3, 2] = -self.kf / self.u1, -self.kf * self.a / self.u1

        At_matrix[5, 1], At_matrix[5, 2] = -self.km / self.u1, -self.km * (-self.b) / self.u1

        At_matrix[7, 1], At_matrix[7, 2] = -self.km / self.u1, -self.km * (-self.b) / self.u1

        At_matrix[9, 6], At_matrix[9, 7] = -self.kr / self.u2, -self.kr * (-self.d) / self.u2

        At_matrix[11, 6], At_matrix[11, 7] = -self.kr / self.u2, -self.kr * (-self.d) / self.u2

        Bt_matrix = torch.zeros((12, 2))
        Bt_matrix[1, 1], Bt_matrix[3, 1] = self.kf, self.kf
        Bt_matrix[4, 0], Bt_matrix[6, 0] = 1 / self.Rw, 1 / self.Rw

        temp = mtimes(At_matrix, X_matrix) + mtimes(Bt_matrix, self.U)
        FCG = mtimes(Lc_matrix, mtimes(Mw_matrix, temp))
        X_dot = mtimes((A_matrix + mtimes(C_matrix, mtimes(J_matrix, K1_matrix))), X_matrix)\
                + mtimes((B_matrix - mtimes(C_matrix, mtimes(J_matrix, K2_matrix))), FCG)

        # X_dot = mtimes(mtimes(inv(M_matrix), A_matrix), X_matrix) + mtimes(mtimes(inv(M_matrix), B_matrix), self.U)
        # # px1, py1, phi1, px2, py2, phi2,
        # # beta1, phi1_dot, varphi1, varphi1_dot, beta2, phi2_dot, varphi2, varphi2_dot, vy1
        self.dyn = vertcat(self.X[0] + delta_t * (self.u1 * cos(self.X[2]) - self.X[8] * sin(self.X[2])),
                           self.X[1] + delta_t * (self.u1 * sin(self.X[2]) + self.X[8] * cos(self.X[2])),#
                           self.X[2] + delta_t * self.X[9],
                           self.X[3] + delta_t * X_dot[0],
                           self.X[4] + delta_t * (self.u2 * cos(self.X[6]) - self.X[12] * sin(self.X[6])),
                           self.X[5] + delta_t * (self.u2 * sin(self.X[6]) + self.X[12] * cos(self.X[6])),  #
                           self.X[6] + delta_t * self.X[13],
                           self.X[7] + delta_t * X_dot[5],
                           self.X[8:12] + delta_t * X_dot[1:5],
                           self.X[12:16] + delta_t * X_dot[6:10],
                           self.U
                           )
        #
        self.Path_Cost_Update(ref_traj_init)

    def Path_Cost_Update(self, ref):
        # ref:ref_y, ref_vx, ref_varphi, ref_x + (self.X[0] - ref[0]) ** 2
        self.path_cost = self.wdis * ((self.X[1] - ref[1]) ** 2 ) + \
                         self.wu1 * (self.X[3]-ref[3]) ** 2 + \
                         self.wphi1 * (self.X[2] - ref[2]) ** 2 \
                         + self.wphidot1 * (self.X[9]) ** 2  + \
                         self.wvarphi1 * (self.X[10]) ** 2 \
                         + self.wvarphi1dot * (self.X[11]) ** 2 + \
                         + self.wstrdot * (self.U[2]) ** 2\
                         + self.wQdot * (self.U[0] ** 2+self.U[1] ** 2)
        #+ (self.X[0] - ref[0]) ** 2
        self.final_cost = self.wdis * ((self.X[1] - ref[1]) ** 2 ) + \
                         self.wu1 * (self.X[3]-ref[3]) ** 2 + \
                         self.wphi1 * (self.X[2] - ref[2]) ** 2 \
                         + self.wphidot1 * (self.X[9]) ** 2  + \
                         self.wvarphi1 * (self.X[10]) ** 2 \
                         + self.wvarphi1dot * (self.X[11]) ** 2
        return self.path_cost, self.final_cost

    def stepPhysics_i(self, states, action):
        state_next = torch.empty_like(states)
        U_matrix = action.type(torch.float32)
        X_matrix = torch.cat((states[3], states[8:12], states[7],
                              states[12:16]))
        u_tt = states[3]
        u_tl = states[7]
        self.Att12 = -self.m_tt * u_tt

        self.Att42 = self.ms_tt * self.hs_tt * u_tt

        Mtt_matrix = torch.zeros((5, 5))
        Mtt_matrix[0, 0] = self.Mtt00

        Mtt_matrix[1, 1] = self.Mtt11

        Mtt_matrix[1, 4] = self.Mtt14

        Mtt_matrix[2, 2] = self.Mtt22

        Mtt_matrix[3, 3] = self.Mtt33

        Mtt_matrix[4, 1] = self.Mtt41
        Mtt_matrix[4, 4] = self.Mtt44

        Att_matrix = torch.zeros((5, 5))
        Att_matrix[1, 2] = self.Att12
        Att_matrix[3, 4] = self.Att34
        Att_matrix[4, 2] = self.Att42
        Att_matrix[4, 3], Att_matrix[4, 4] = self.Att43, self.Att44

        Btt_matrix = torch.zeros((5, 3))
        Btt_matrix[0, 0] = self.Btt00
        Btt_matrix[1, 1], Btt_matrix[2, 2] = self.Btt11, self.Btt22

        Ctt_matrix = torch.zeros((5, 4))
        Ctt_matrix[0, 0], Ctt_matrix[1, 1] = self.Ctt00, self.Ctt11
        Ctt_matrix[2, 1], Ctt_matrix[2, 2] = self.Ctt21, self.Ctt22
        Ctt_matrix[4, 1], Ctt_matrix[4, 3] = self.Ctt41, self.Ctt43

        # --------trailer---------
        self.Atl12 = -self.m_tl * u_tl

        self.Atl42 = self.ms_tl * self.hs_tl * u_tl

        Mtl_matrix = torch.zeros((5, 5))
        Mtl_matrix[0, 0] = self.Mtl00

        Mtl_matrix[1, 1] = self.Mtl11

        Mtl_matrix[1, 4] = self.Mtl14

        Mtl_matrix[2, 2] = self.Mtl22

        Mtl_matrix[3, 3] = self.Mtl33

        Mtl_matrix[4, 1] = self.Mtl41
        Mtl_matrix[4, 4] = self.Mtl44

        Atl_matrix = torch.zeros((5, 5))
        Atl_matrix[1, 2] = self.Atl12
        Atl_matrix[3, 4] = self.Atl34
        Atl_matrix[4, 2] = self.Atl42
        Atl_matrix[4, 3], Atl_matrix[4, 4] = self.Atl43, self.Atl44

        Btl_matrix = torch.zeros((5, 3))
        Btl_matrix[0, 0] = self.Btl00

        Btl_matrix[1, 1], Btl_matrix[2, 2] = self.Btl11, self.Btl22

        Ctl_matrix = torch.zeros((5, 4))
        Ctl_matrix[0, 0], Ctl_matrix[1, 1] = self.Ctl00, self.Ctl11
        Ctl_matrix[2, 1], Ctl_matrix[2, 2] = self.Ctl21, self.Ctl22
        Ctl_matrix[4, 1], Ctl_matrix[4, 3] = self.Ctl41, self.Ctl43

        A_matrix = torch.block_diag(torch.matmul(torch.inverse(Mtt_matrix), Att_matrix),
                                    torch.matmul(torch.inverse(Mtl_matrix), Atl_matrix))

        B_matrix = torch.block_diag(torch.matmul(torch.inverse(Mtt_matrix), Btt_matrix),
                                    torch.matmul(torch.inverse(Mtl_matrix), Btl_matrix))

        C_matrix = torch.block_diag(torch.matmul(torch.inverse(Mtt_matrix), Ctt_matrix),
                                    torch.matmul(torch.inverse(Mtl_matrix), Ctl_matrix))

        M_matrix = torch.zeros((2, 10))
        M_matrix[0, 0], M_matrix[0, 5] = self.M00, self.M05
        M_matrix[1, 1], M_matrix[1, 2] = self.M11, self.M12
        M_matrix[1, 6], M_matrix[1, 7] = self.M16, self.M17

        self.P12, self.P17 = -u_tt, u_tt
        P_matrix = torch.zeros((2, 10))
        P_matrix[1, 2], P_matrix[1, 7] = self.P12, self.P17

        N_matrix = torch.zeros((6, 8))
        N_matrix[0, 0], N_matrix[0, 4] = self.N00, self.N04
        N_matrix[1, 1], N_matrix[1, 5] = self.N11, self.N15
        N_matrix[2, 2], N_matrix[2, 6] = self.N22, self.N26
        N_matrix[3, 2] = self.N32
        N_matrix[4, 3], N_matrix[4, 7] = self.N43, self.N47
        N_matrix[5, 7] = self.N57

        Q_matrix = torch.zeros((6, 10))
        Q_matrix[5, 4], Q_matrix[5, 9] = self.Q54, self.Q59

        J_matrix = torch.inverse(torch.vstack((N_matrix, torch.matmul(M_matrix, C_matrix))))

        K1_matrix = torch.vstack((Q_matrix, (P_matrix - torch.matmul(M_matrix, A_matrix))))
        K2_matrix = torch.vstack((torch.zeros((6, 6)), (torch.matmul(M_matrix, B_matrix))))

        Lc_matrix = torch.zeros((6, 12))
        Lc_matrix[0, 0], Lc_matrix[0, 2], \
        Lc_matrix[0, 4], Lc_matrix[0, 6] = 1, 1, 1, 1

        Lc_matrix[1, 1], Lc_matrix[1, 3], \
        Lc_matrix[1, 5], Lc_matrix[1, 7], = 1, 1, 1, 1

        Lc_matrix[2, 0], Lc_matrix[2, 1], Lc_matrix[2, 2], Lc_matrix[2, 3], \
        Lc_matrix[2, 4], Lc_matrix[2, 5], Lc_matrix[2, 6], Lc_matrix[2, 7] = \
            -self.lw / 2, self.a, self.lw / 2, self.a, \
            -self.lw / 2, -self.b, self.lw / 2, -self.b
        Lc_matrix[3, 8], Lc_matrix[3, 10] = 1, 1

        Lc_matrix[4, 9], Lc_matrix[4, 11] = 1, 1

        Lc_matrix[5, 8], Lc_matrix[5, 9], Lc_matrix[5, 10], Lc_matrix[5, 11] = \
            -self.lw / 2, -self.d, \
            self.lw / 2, -self.d

        delta = action[1]
        Mw1 = torch.tensor([[torch.cos(delta), -torch.sin(delta)],
                            [torch.sin(delta), torch.cos(delta)]])
        Mw2 = torch.tensor([[torch.cos(delta), -torch.sin(delta)],
                            [torch.sin(delta), torch.cos(delta)]])
        Mw3 = torch.tensor([[1, 0],
                            [0, 1]])
        Mw4 = torch.tensor([[1, 0],
                            [0, 1]])
        Mw5 = torch.tensor([[1, 0],
                            [0, 1]])
        Mw6 = torch.tensor([[1, 0],
                            [0, 1]])

        Mw_matrix = torch.block_diag(Mw1, Mw2, Mw3, Mw4, Mw5, Mw6)

        At_matrix = torch.zeros((12, 10))

        At_matrix[1, 1], At_matrix[1, 2] = -self.kf / u_tt, -self.kf * self.a / u_tt

        At_matrix[3, 1], At_matrix[3, 2] = -self.kf / u_tt, -self.kf * self.a / u_tt

        At_matrix[5, 1], At_matrix[5, 2] = -self.km / u_tt, -self.km * (-self.b) / u_tt

        At_matrix[7, 1], At_matrix[7, 2] = -self.km / u_tt, -self.km * (-self.b) / u_tt

        At_matrix[9, 6], At_matrix[9, 7] = -self.kr / u_tl, -self.kr * (-self.d) / u_tl

        At_matrix[11, 6], At_matrix[11, 7] = -self.kr / u_tl, -self.kr * (-self.d) / u_tl

        Bt_matrix = torch.zeros((12, 2))
        Bt_matrix[1, 1], Bt_matrix[3, 1] = self.kf, self.kf
        Bt_matrix[4, 0], Bt_matrix[6, 1] = 1 / self.Rw, 1 / self.Rw

        temp = torch.matmul(At_matrix, X_matrix) + torch.matmul(Bt_matrix, U_matrix)
        FCG = torch.matmul(Lc_matrix, torch.matmul(Mw_matrix, temp))
        X_dot = (torch.matmul((A_matrix + torch.matmul(C_matrix, torch.matmul(J_matrix, K1_matrix))), X_matrix)
                 + torch.matmul((B_matrix - torch.matmul(C_matrix, torch.matmul(J_matrix, K2_matrix))), FCG)).squeeze()

        state_next[0] = states[0] + self.dt * (
                    u_tt * torch.cos(states[2].clone()) - states[8].clone() * torch.sin(
                states[2].clone()))  # x_tractor
        state_next[1] = states[1] + self.dt * (
                    u_tt * torch.sin(states[2].clone()) + states[8].clone() * torch.cos(states[2].clone()))
        state_next[2] = states[2] + self.dt * states[9].clone()
        state_next[3] = states[3] + self.dt * X_dot[0]  # u_tt
        state_next[4] = states[4] + self.dt * (
                    u_tl * torch.cos(states[6].clone()) - states[12].clone() * torch.sin(
                states[6].clone()))  # px_tl
        state_next[5] = states[5] + self.dt * (
                    u_tl * torch.sin(states[6].clone()) + states[12] * torch.cos(states[6].clone()))  # py_tt
        state_next[6] = states[6] + self.dt * states[13]  # varphi_tl
        state_next[7] = states[7] + self.dt * X_dot[5]  # u_tl
        state_next[8:12] = states[8:12] + self.dt * X_dot[1:5]
        state_next[12:16] = states[12:16] + self.dt * X_dot[6:10]
        state_next[16:19] = action
        return state_next

    def reward_func_i(self, states, action, ref, cost_paras):
        # to calculate dcx in jacobian API
        # #+(states[0] - ref_x) ** 2
        ref_x, ref_y, ref_phi, ref_u = ref[0], ref[1], ref[2], ref[3]
        reward = cost_paras[0] * ((states[1] - ref_y) ** 2) + \
                 cost_paras[1] * ((states[3]) ** 2-ref_u) + \
                 cost_paras[2] * ((states[2] - ref_phi) ** 2) + \
                 cost_paras[3] * (states[9]) ** 2 + \
                 cost_paras[4] * (states[10]) ** 2 + \
                 cost_paras[5] * (states[11]) ** 2 + \
                 cost_paras[6] * (action[2]) ** 2 + \
                 cost_paras[7] * (action[0] ** 2+action[1] ** 2)
        return reward

class Semitruckpu6dofModel(PythBaseModel):
    def __init__(
        self,
        pre_horizon: int = 20,
        device: Union[torch.device, str, None] = None,
        path_para: Optional[Dict[str, Dict]] = None,
        u_para: Optional[Dict[str, Dict]] = None,
        max_steer: float = 0.01,
        **kwargs,
    ):
        """
        you need to define parameters here
        """
        self.vehicle_dynamics = VehicleDynamicsModel()
        self.pre_horizon = pre_horizon
        self.batch_size = kwargs["replay_batch_size"]
        self.state_dim = self.vehicle_dynamics.state_dim
        self.cost_dim = 8
        ego_obs_dim = self.state_dim
        ref_obs_dim = 8
        max_torque = 2000
        obs_scale_default = [1/100, 1/100, 1/10, 1/10, 1/(max_torque*10), 1/10]
        self.obs_scale = np.array(kwargs.get('obs_scale', obs_scale_default))
        super().__init__(
            obs_dim=ego_obs_dim + ref_obs_dim * pre_horizon,
            action_dim=3,
            dt=0.01,
            action_lower_bound=[-50, -50, -max_steer],
            action_upper_bound=[50, 50, max_steer],
            device=device,
        )
        self.action_lower_bound, self.action_upper_bound = torch.tensor([0] * 2 + [-0.5]), torch.tensor([max_torque] * 2 + [0.5])
        self.ref_traj = MultiRefTrajModel(path_para, u_para)

        self.cost_paras = np.array([1, 1, 0.8, 0.5, 0.5, 0.5, 0.4, 0.4])#np.array(kwargs["cost_paras"])

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        done: torch.Tensor,
        info: InfoDict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, InfoDict]:
        state = info["state"]
        ref_points = info["ref_points"]
        t = info["ref_time"]
        t2 = info["ref_time2"]
        path_num = info["path_num"]
        u_num = info["u_num"]

        reward = self.compute_reward(obs, action)
        action_psc = action + state[:, 16:19]
        action_psc = torch.clamp(action_psc, min=self.action_lower_bound, max=self.action_upper_bound)
        next_state = self.vehicle_dynamics.f_xu(state, action_psc, self.dt)

        next_t = t + self.dt
        next_t2 = t2 + self.dt

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

        new_ref_point_2 = torch.stack(
            (
                self.ref_traj.compute_x(
                    next_t2 + self.pre_horizon * self.dt, path_num, u_num
                ),
                self.ref_traj.compute_y(
                    next_t2 + self.pre_horizon * self.dt, path_num, u_num
                ),
                self.ref_traj.compute_phi(
                    next_t2 + self.pre_horizon * self.dt, path_num, u_num
                ),
                self.ref_traj.compute_u(
                    next_t2 + self.pre_horizon * self.dt, path_num, u_num
                ),
            ),
            dim=1,
        )
        next_ref_points[:, -1] = torch.cat((new_ref_point, new_ref_point_2), 1)

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
            "ref_time2": next_t2,
        })
        return next_obs, reward, isdone, next_info

    def get_obs(self, state, ref_points):
        ref_x_tf, ref_y_tf, ref_phi_tf = \
            ego_vehicle_coordinate_transform(
                state[:, 0], state[:, 1], state[:, 2],
                ref_points[..., 0], ref_points[..., 1], ref_points[..., 2],
            )
        ref_u_tf = ref_points[..., 3]-state[:, 3]
        ref_x2_tf, ref_y2_tf, ref_phi2_tf = \
            ego_vehicle_coordinate_transform(
                state[:, 4], state[:, 5], state[:, 6],
                ref_points[..., 4], ref_points[..., 5], ref_points[..., 6],
            )
        ref_u2_tf = ref_points[..., 7] - state[:, 7]
        ego_obs = torch.concat((torch.stack((ref_x_tf[:, 0]*self.obs_scale[0], ref_y_tf[:, 0]*self.obs_scale[1], ref_phi_tf[:, 0]*self.obs_scale[2], ref_u_tf[:, 0]*self.obs_scale[3],
                                             ref_x2_tf[:, 0]*self.obs_scale[0], ref_y2_tf[:, 0]*self.obs_scale[1], ref_phi2_tf[:, 0]*self.obs_scale[2], ref_u2_tf[:, 0]*self.obs_scale[3]), dim=1),
                                state[:, 8:16], torch.stack((state[:, 16]*self.obs_scale[4], state[:, 17]*self.obs_scale[4], state[:, 18]*self.obs_scale[5]), dim=1)), dim=1)
        ref_obs = torch.stack((ref_x_tf*self.obs_scale[0], ref_y_tf*self.obs_scale[1], ref_phi_tf*self.obs_scale[2], ref_u_tf*self.obs_scale[3],
                               ref_x2_tf*self.obs_scale[0], ref_y2_tf*self.obs_scale[1], ref_phi2_tf*self.obs_scale[2],ref_u2_tf*self.obs_scale[3]), 2)[
            :, 1:].reshape(ego_obs.shape[0], -1)
        return torch.concat((ego_obs, ref_obs), 1)

    def compute_reward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor) -> torch.Tensor:
        delta_x, delta_y, delta_phi, delta_u, \
        delta_x2, delta_y2, delta_phi2, delta_u2 = \
            obs[:, 0] / self.obs_scale[0], obs[:, 1] / self.obs_scale[1], \
            obs[:, 2] / self.obs_scale[2], obs[:, 3] / self.obs_scale[3], \
            obs[:, 4] / self.obs_scale[0], obs[:, 5] / self.obs_scale[1], \
            obs[:, 6] / self.obs_scale[2], obs[:, 7] / self.obs_scale[3]
        return -(
            self.cost_paras[0] * delta_y ** 2
            + self.cost_paras[1] * delta_u ** 2
            + self.cost_paras[2] * delta_phi ** 2
            + self.cost_paras[3] * obs[:, 9] ** 2
            + self.cost_paras[4] * obs[:, 10] ** 2
            + self.cost_paras[5] * obs[:, 11] ** 2
            + self.cost_paras[6] * (action[:, 2]/0.02) ** 2
            + self.cost_paras[7] * ((action[:, 0]/100) ** 2+(action[:, 1]/100) ** 2))
    def judge_done(self, obs: torch.Tensor) -> torch.Tensor:
        delta_x, delta_y, delta_phi, delta_u, \
        delta_x2, delta_y2, delta_phi2, delta_u2 = \
            obs[:, 0]/self.obs_scale[0], obs[:, 1]/self.obs_scale[1], \
            obs[:, 2]/self.obs_scale[2], obs[:, 3]/self.obs_scale[3], \
            obs[:, 4]/self.obs_scale[0], obs[:, 5]/self.obs_scale[1], \
            obs[:, 6]/self.obs_scale[2], obs[:, 7]/self.obs_scale[3]
        done = ((torch.abs(delta_y) > 3)
                | (torch.abs(delta_phi) > np.pi)
                | (torch.abs(delta_u) > 5)
                | (torch.abs(delta_y2) > 3)
                | (torch.abs(delta_phi2) > np.pi)
        )
        # (torch.abs(delta_x) > 5)|
        #| (torch.abs(delta_x2) > 5)
        return done

    def update_cost_paras(self, cost_paras):
        self.cost_paras = cost_paras

    def Rollout_traj_NN(self, data, networks, step, cost_paras):
        o, a, r, o2, d = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )
        info = data
        # Forward Pass, step = horizon
        state_traj_opt, control_traj_opt, costate_traj_opt = np.zeros((step + 1, self.batch_size, self.state_dim)), \
                                                             np.zeros((step, self.batch_size,
                                                                       self.action_dim)), \
                                                             torch.zeros((step + 1, self.batch_size,
                                                                          self.state_dim, 1))
        path_num_rollout = torch.zeros((step + 1, self.batch_size))
        u_num_rollout = torch.zeros((step + 1, self.batch_size))
        ref_time2_rollout = torch.zeros((step + 1, self.batch_size))
        state = info["state"]
        ref_tractor = info["ref_points"]
        state_traj_opt[0, :, :] = state[:, :].clone().detach()
        costate_traj_opt[step, :, :] = 0
        ref_time2_rollout[0, :] = info["ref_time2"]
        path_num_rollout[0, :] = info["path_num"]
        u_num_rollout[0, :] = info["u_num"]
        a = networks.policy.forward_all_policy(o)
        for k in range(step):
            if k == 0:
                o2, r, d, info = self.forward(o, a[:, 0, :], d, info)
            else:
                o = o2
                o2, r, d, info = self.forward(o, a[:, k, :], d, info)

            a.requires_grad_(True)
            dfx_jacobian_list = []
            dcx_jacobian_list = []
            for i in range(self.batch_size):
                dfx_jacobian_i = jacobian(self.vehicle_dynamics.stepPhysics_i,
                                          (state[i, :self.state_dim], a[i, 0, :self.action_dim]+state[i, 0, 16:19]))
                dfx_jacobian_list.append(dfx_jacobian_i[0])
                dcx_jacobian_i = jacobian(self.vehicle_dynamics.reward_func_i, (
                    state[i, :self.state_dim], a[i, 0, :], ref_tractor[i, k, :4],
                    torch.from_numpy(cost_paras).type(torch.float32)))
                dcx_jacobian_list.append(dcx_jacobian_i[0])
                # Calculate the costate_traj
                costate_traj_opt[step - k - 1, i, :] = (torch.reshape(dcx_jacobian_i[0], (self.state_dim, 1))
                                                    + torch.mm(dfx_jacobian_i[0],
                                                               costate_traj_opt[step - k, i, :])).detach()

            control_traj_opt[k, :, :] = a[:, 0, :].detach()
            state_traj_opt[k+1, :, :] = state[:, :].detach()
            state = info["state"]
            ref_tractor = info["ref_points"]
            ref_time2_rollout[k+1, :] = info["ref_time2"]
            path_num_rollout[k+1, :] = info["path_num"]
            u_num_rollout[k+1, :] = info["u_num"]
        costate_traj_opt = torch.reshape(costate_traj_opt, (step + 1, self.batch_size, self.state_dim))
        traj = {"state_traj_opt": state_traj_opt,
                "control_traj_opt": control_traj_opt,
                "costate_traj_opt": costate_traj_opt.detach().numpy(),
                "ref_time2_rollout": ref_time2_rollout,
                "path_num_rollout": path_num_rollout,
                "u_num_rollout": u_num_rollout}
        return traj

    # def Rollout_Trajectory_PDP(self, obs, info, controller, step, cost_paras):
    #
    #     # Forward Pass, step = horizon
    #     state_traj_opt, control_traj_opt, costate_traj_opt = np.empty((1, step + 1, self.state_dim)), \
    #         np.empty((1, step, self.action_dim)), \
    #         np.empty((1, step, self.state_dim))
    #
    #     ref_tractor = info["ref_points"]
    #     state_traj_opt[0, 0, :] = state[0, :].clone().detach()
    #     ref_traj = []
    #     pos_x1, pos_y1 = state[:1, self.state_dim-2].clone(), state[:1, self.state_dim-4].clone()
    #
    #     action = controller(obs, info)
    #     state_traj_opt[0, :, :] = opt_sol["state_traj_opt"]
    #     control_traj_opt[0, :, :] = opt_sol["control_traj_opt"]
    #     costate_traj_opt[0, :, :] = opt_sol["costate_traj_opt"]
    #
    #     traj = {"state_traj_opt": state_traj_opt,
    #             "control_traj_opt": control_traj_opt,
    #             "costate_traj_opt": costate_traj_opt}
    #     return traj

def state_error_calculate(
    ego_x: torch.Tensor,
    ego_y: torch.Tensor,
    ego_varphi: torch.Tensor,
    ref_x: torch.Tensor,
    ref_y: torch.Tensor,
    ref_varphi: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ego_x, ego_y, ego_varphi = ego_x.unsqueeze(1), ego_y.unsqueeze(1), ego_varphi.unsqueeze(1)
    x_err = ego_x - ref_x
    y_err = ego_y - ref_y
    varphi_err = angle_normalize(ego_varphi - ref_varphi)
    return x_err, y_err, varphi_err

def ego_vehicle_coordinate_transform(
    ego_x: torch.Tensor,
    ego_y: torch.Tensor,
    ego_varphi: torch.Tensor,
    ref_x: torch.Tensor,
    ref_y: torch.Tensor,
    ref_varphi: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ego_x, ego_y, ego_varphi = ego_x.unsqueeze(1), ego_y.unsqueeze(1), ego_varphi.unsqueeze(1)
    cos_tf = torch.cos(-ego_varphi)
    sin_tf = torch.sin(-ego_varphi)
    ref_x_tf = (ref_x - ego_x) * cos_tf - (ref_y - ego_y) * sin_tf
    ref_y_tf = (ref_x - ego_x) * sin_tf + (ref_y - ego_y) * cos_tf
    ref_varphi_tf = angle_normalize(ref_varphi - ego_varphi)
    return ref_x_tf, ref_y_tf, ref_varphi_tf


def env_model_creator(**kwargs):
    """
    make env model `pyth_veh3dofconti`
    """
    return Semitruckpu6dofModel(**kwargs)
