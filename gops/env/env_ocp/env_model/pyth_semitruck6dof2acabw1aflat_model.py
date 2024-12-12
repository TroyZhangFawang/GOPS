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
from gops.env.env_ocp.pyth_semitruck6dof import angle_normalize, VehicleDynamicsData
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
        self.kvarvarphi_tt = DynamicsData.vehicle_params["kvarphi_tt"]
        self.kvarvarphi_tl = DynamicsData.vehicle_params["kvarphi_tl"]
        self.ka = DynamicsData.vehicle_params["ka"]
        self.cvarvarphi_tt = DynamicsData.vehicle_params["cvarphi_tt"]
        self.cvarvarphi_tl = DynamicsData.vehicle_params["cvarphi_tl"]
        self.state_dim = DynamicsData.vehicle_params["state_dim"]

        self.Mtt00 = self.m_tt

        self.Mtt11 = self.m_tt
        self.Mtt14 = -self.ms_tt * self.hs_tt

        self.Mtt22 = self.Izz_tt

        self.Mtt33 = 1

        self.Mtt41 = -self.ms_tt * self.hs_tt
        self.Mtt44 = self.Ixx_tt

        self.Att34 = 1

        self.Att43 = -self.kvarvarphi_tt + self.ms_tt * self.gravity * self.hs_tt
        self.Att44 = -self.cvarvarphi_tt

        self.Btt00 = 1
        self.Btt11 = 1
        self.Btt22 = 1

        self.Ctt00 = 1
        self.Ctt11 = 1
        self.Ctt21 = self.lhtt
        self.Ctt22 = 1
        self.Ctt41 = self.hh_tt
        self.Ctt43 = 1

        self.Mtl00 = self.m_tl

        self.Mtl11 = self.m_tl
        self.Mtl14 = -self.ms_tl * self.hs_tl

        self.Mtl22 = self.Izz_tl

        self.Mtl33 = 1

        self.Mtl41 = -self.ms_tl * self.hs_tl
        self.Mtl44 = self.Ixx_tl

        self.Atl34 = 1

        self.Atl43 = -self.kvarvarphi_tl + self.ms_tl * self.gravity * self.hs_tl
        self.Atl44 = -self.cvarvarphi_tl

        self.Btl00 = 1
        self.Btl11 = 1
        self.Btl22 = 1

        self.Ctl00 = 1
        self.Ctl11 = 1
        self.Ctl21 = self.lhtl
        self.Ctl22 = 1
        self.Ctl41 = self.hh_tl
        self.Ctl43 = 1

        self.M00, self.M05 = 1, -1
        self.M11, self.M12, self.M16, self.M17 = 1, -self.lhtt, -1, -self.lhtl

        self.N00, self.N04 = 1, 1
        self.N11, self.N15, self.N22, self.N26 = 1, 1, 1, 1
        self.N32 = 1
        self.N43, self.N47 = 1, 1
        self.N57 = 1

        self.Q54, self.Q59 = -self.ka, self.ka

    def f_xu(self, states, actions, delta_t):
        state_next = torch.zeros_like(states)
        X_matrix = torch.cat((states[:, 3:4], states[:, 8:12], states[:, 7:8],
                              states[:, 12:16]), 1).reshape(10, 1)
        U_matrix = actions.reshape(3,1)
        u_tt = states[:, 3]
        u_tl = states[:, 7]
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
        Bt_matrix[1, 2], Bt_matrix[3, 2] = self.kf, self.kf
        Bt_matrix[4, 0], Bt_matrix[6, 1] = 1 / self.Rw, 1 / self.Rw

        temp = torch.matmul(At_matrix, X_matrix) + torch.matmul(Bt_matrix, U_matrix)
        FCG = torch.matmul(Lc_matrix, torch.matmul(Mw_matrix, temp))
        X_dot = (torch.matmul((A_matrix + torch.matmul(C_matrix, torch.matmul(J_matrix, K1_matrix))), X_matrix)
                 + torch.matmul((B_matrix - torch.matmul(C_matrix, torch.matmul(J_matrix, K2_matrix))), FCG)).squeeze()

        # X_dot_batch = torch.zeros((self.batch_size, self.state_dim - 2))
        # for batch in range(self.batch_size):
        #     X_dot = (torch.matmul(torch.matmul(torch.inverse(M_matrix), A_matrix),
        #                           X_matrix[batch, :]) +
        #              torch.matmul(torch.inverse(M_matrix), torch.matmul(B_matrix, actions[batch, :]))).squeeze()
        #     X_dot_batch[batch, :] = X_dot

        state_next[:, 0] = states[:, 0] + delta_t * (u_tt * torch.cos(states[:, 2].clone()) - states[:, 8].clone() * torch.sin(states[:, 2].clone()))  # x_tractor
        state_next[:, 1] = states[:, 1] + delta_t * (u_tt * torch.sin(states[:, 2].clone()) + states[:, 8].clone() * torch.cos(states[:, 2].clone()))
        state_next[:, 2] = states[:, 2] + delta_t * states[:, 9].clone()
        state_next[:, 3] = states[:, 3] + delta_t * X_dot[0]  # u_tt
        state_next[:, 4] = states[:, 4] + delta_t * (u_tl * torch.cos(states[:, 6].clone()) - states[:, 12].clone() * torch.sin(states[:, 6].clone()))  # px_tl
        state_next[:, 5] = states[:, 5] + delta_t * (u_tl * torch.sin(states[:, 6].clone()) + states[:, 12] * torch.cos(states[:, 6].clone()))  # py_tt
        state_next[:, 6] = states[:, 6] + delta_t * states[:, 13]  # varphi_tl
        state_next[:, 7] = states[:, 7] + delta_t * X_dot[ 5]  # u_tl
        state_next[:, 8:12] = states[:, 8:12] + delta_t * X_dot[1:5]
        state_next[:, 12:16] = states[:, 12:16] + delta_t * X_dot[6:10]
        return state_next

    def auxvar_setting(self, delta_t=0.01, ref_traj_init=np.array([0, 0, 0]), action_last_np=0):
        parameter = []
        self.wdis = SX.sym('wdis')
        parameter += [self.wdis]
        self.wphi1 = SX.sym('wphi1')
        parameter += [self.wphi1]
        self.wphidot1 = SX.sym('wphidot1')
        parameter += [self.wphidot1]
        self.wvarphi1 = SX.sym('wvarphi1')
        parameter += [self.wvarphi1]
        self.wvarphi1dot = SX.sym('wvarphi1dot')
        parameter += [self.wvarphi1dot]
        self.wstr = SX.sym('wstr')
        parameter += [self.wstr]
        self.wstrdot = SX.sym('wstrdot')
        parameter += [self.wstrdot]
        self.wQ = SX.sym('wQ')
        parameter += [self.wQ]
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

        self.X = vertcat(self.x1, self.y1, self.phi1, self.u1,
                         self.x2, self.y2, self.phi2, self.u2,
                         self.v1, self.phi1dot, self.varphi1, self.varphi1dot,
                         self.v2, self.phi2dot, self.varphi2, self.varphi2dot)

        X_matrix = vertcat(self.u1, self.v1, self.phi1dot, self.varphi1, self.varphi1dot,
                         self.u2, self.v2, self.phi2dot, self.varphi2, self.varphi2dot)
        self.steering = SX.sym('str')
        self.Q = SX.sym('Q')
        self.U = vertcat(self.Q, self.Q, self.steering)

        self.Att12 = -self.m_tt * self.u1

        self.Att42 = -self.ms_tt * self.hs_tt * self.u1

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

        self.Atl42 = -self.ms_tl * self.hs_tl * self.u2

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

        Bt_matrix = torch.zeros((12, 3))
        Bt_matrix[1, 2], Bt_matrix[3, 2] = self.kf, self.kf
        Bt_matrix[4, 0], Bt_matrix[6, 1] = 1 / self.Rw, 1 / self.Rw

        temp = mtimes(At_matrix, X_matrix) + mtimes(Bt_matrix, self.U)
        FCG = mtimes(Lc_matrix, mtimes(Mw_matrix, temp))
        X_dot = mtimes((A_matrix + mtimes(C_matrix, mtimes(J_matrix, K1_matrix))), X_matrix)\
                + mtimes((B_matrix - mtimes(C_matrix, mtimes(J_matrix, K2_matrix))), FCG)

        # X_dot = mtimes(mtimes(inv(M_matrix), A_matrix), X_matrix) + mtimes(mtimes(inv(M_matrix), B_matrix), self.U)
        # # px1, py1, phi1, px2, py2, phi2,
        # # beta1, phi1_dot, varphi1, varphi1_dot, beta2, phi2_dot, varphi2, varphi2_dot, vy1
        # self.dyn = vertcat(self.X[0] + delta_t * (self.v_x * cos(self.X[2]) - self.X[14] * sin(self.X[2])),
        #                    self.X[1] + delta_t * X_dot[11],#(self.v_x * sin(self.X[2]) + self.X[14] * cos(self.X[2]))
        #                    self.X[2] + delta_t * X_dot[8],
        #                    self.X[0] - self.b * cos(self.X[2]) - self.e * cos(self.X[5]),
        #                    self.X[1] - self.b * sin(self.X[2]) - self.e * sin(self.X[5]),
        #                    self.X[5] + delta_t * X_dot[9],
        #                    self.X[6:14] + delta_t * X_dot[0:8],
        #                    self.X[14] + delta_t * X_dot[10])
        #
        # self.Path_Cost_Update(ref_traj_init, action_last_np)

    def Path_Cost_Update(self, ref, action_last_np):
        # ref:ref_y, ref_vx, ref_varphi, ref_x + (self.X[0] - ref[0]) ** 2
        self.path_cost = self.wdis * ((self.X[1] - ref[1]) ** 2 ) + \
                         self.wu1 * (self.X[14]) ** 2 + self.wphi1 * (self.X[2] - ref[2]) ** 2 \
                         + self.wphidot1 * (self.X[7]) ** 2 + self.wv1 * (self.X[6]) ** 2 + \
                         self.wvarphi1 * (self.X[8]) ** 2 + self.wvarphi1dot * (self.X[9]) ** 2 + \
                         self.wstr * (self.U[0]) ** 2 + self.wstrdot * (self.U[0] - action_last_np) ** 2
        #+ (self.X[0] - ref[0]) ** 2
        self.final_cost = self.wdis * ((self.X[1] - ref[1]) ** 2 ) + \
                         self.wu1 * (self.X[14]) ** 2 + self.wphi1 * (self.X[2] - ref[2]) ** 2 \
                         + self.wphidot1 * (self.X[7]) ** 2 + self.wv1 * (self.X[6]) ** 2 + \
                         self.wvarphi1 * (self.X[8]) ** 2 + self.wvarphi1dot * (self.X[9]) ** 2
        return self.path_cost, self.final_cost

    def stepPhysics_i(self, state_curr, action):
        state_next = torch.empty_like(state_curr)
        action = action.type(torch.float32)
        X_matrix = torch.cat((state_curr[6:14], state_curr[2:3], state_curr[5:6],
                              state_curr[14:], state_curr[1:2], state_curr[4:5]))
        M_matrix = torch.zeros((self.state_dim-2, self.state_dim-2))
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

        A_matrix = torch.zeros((self.state_dim-2, self.state_dim-2))
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

        B_matrix = torch.zeros((self.state_dim-2, 1))
        B_matrix[0, 0] = self.B11
        B_matrix[1, 0] = self.B21
        B_matrix[2, 0] = self.B31

        X_dot = (torch.matmul(torch.matmul(torch.inverse(M_matrix), A_matrix), X_matrix)  +
                 torch.matmul(torch.inverse(M_matrix), torch.matmul(B_matrix, action))).squeeze()

        state_next[0] = state_curr[0] + self.dt * (self.v_x * torch.cos(state_curr[2].clone()) - state_curr[14] * torch.sin(state_curr[2].clone()))
        state_next[1] = state_curr[1] + self.dt * X_dot[11]#(self.v_x * torch.sin(state_curr[2].clone()) + state_curr[14] * torch.cos(state_curr[2].clone()))
        state_next[2] = state_curr[2] + self.dt * X_dot[8]
        state_next[3] = state_next[0] - self.b * torch.cos(state_curr[2].clone()) - self.e * torch.cos(
            state_curr[5].clone())  # posx_trailer
        state_next[4] = state_next[1] - self.b * torch.sin(state_curr[2].clone()) - self.e * torch.sin(
            state_curr[5].clone())  # posy_trailer
        state_next[5] = state_curr[5] + self.dt * X_dot[9]
        state_next[6:14] = state_curr[6:14] + self.dt * X_dot[0:8]
        state_next[14] = state_curr[14] + self.dt * X_dot[10]
        return state_next

    def reward_func_i(self, state_curr, action, ref, cost_paras, action_last_i):
        # to calculate dcx in jacobian API
        # #+(state_curr[0] - ref_x) ** 2
        ref_x, ref_y, ref_phi,  = ref[0], ref[1], ref[2]
        reward = cost_paras[0] * ((state_curr[1] - ref_y) ** 2) + \
                 cost_paras[1] * ((state_curr[14]) ** 2) + \
                 cost_paras[2] * ((state_curr[2] - ref_phi) ** 2) + \
                 cost_paras[3] * (state_curr[7]) ** 2 + \
                 cost_paras[4] * (state_curr[6]) ** 2 + \
                 cost_paras[5] * (state_curr[8]) ** 2 + \
                 cost_paras[6] * (state_curr[9]) ** 2 + \
                 cost_paras[7] * (action[0] ** 2) + \
                 cost_paras[8] * (action[0]-action_last_i) ** 2
        return reward

class Semitruckpu6dofModel(PythBaseModel):
    def __init__(
        self,
        pre_horizon: int = 30,
        device: Union[torch.device, str, None] = None,
        path_para: Optional[Dict[str, Dict]] = None,
        u_para: Optional[Dict[str, Dict]] = None,
        max_steer: float = 0.5,
        **kwargs,
    ):
        """
        you need to define parameters here
        """
        self.vehicle_dynamics = VehicleDynamicsModel()
        self.pre_horizon = pre_horizon
        self.batch_size = kwargs["replay_batch_size"]
        self.state_dim = 16
        self.cost_dim = 10
        ego_obs_dim = 16
        ref_obs_dim = 8
        obs_scale_default = [1/100, 1/100, 1/10, 1/10]
        self.obs_scale = np.array(kwargs.get('obs_scale', obs_scale_default))
        super().__init__(
            obs_dim=ego_obs_dim + ref_obs_dim * pre_horizon,
            action_dim=3,
            dt=0.01,
            action_lower_bound=[50, 50, -max_steer],
            action_upper_bound=[400, 400, max_steer],
            device=device,
        )

        self.ref_traj = MultiRefTrajModel(path_para, u_para)
        self.action_last = torch.zeros((1, 3))
        self.action_last_np = np.zeros((3, ))
        self.cost_paras = np.array([1, 1, 0.8, 0.5, 0.5, 0.5, 0.4, 0.1, 0.4, 0.1])#np.array(kwargs["cost_paras"])

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

        next_state = self.vehicle_dynamics.f_xu(state, action, self.dt)

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
        self.action_last = action.clone().detach()
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
                                state[:, 8:16]), dim=1)
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
            + self.cost_paras[6] * action[:, 2] ** 2
            + self.cost_paras[7] * (action[:, 2] - self.action_last[:, 2]) ** 2
            + self.cost_paras[8] * ((action[:, 0]/400) ** 2+(action[:, 1]/400) ** 2)
            + self.cost_paras[9] * ((action[:, 0] - self.action_last[:, 0])/50 ** 2 + (action[:, 1] - self.action_last[:, 1])/50 ** 2))
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
                action_last_i = self.action_last[i, 0]
                dfx_jacobian_i = jacobian(self.vehicle_dynamics.stepPhysics_i,
                                          (state[i, :self.state_dim], a[i, 0, :self.action_dim]))
                dfx_jacobian_list.append(dfx_jacobian_i[0])
                dcx_jacobian_i = jacobian(self.vehicle_dynamics.reward_func_i, (
                    state[i, :self.state_dim], a[i, 0, :], ref_tractor[i, k, :],
                    torch.from_numpy(cost_paras).type(torch.float32), action_last_i))
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
