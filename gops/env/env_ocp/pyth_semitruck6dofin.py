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
from typing import Dict, Optional, Sequence, Tuple
import pandas as pd
import torch
from scipy.linalg import *
import gym
import numpy as np
from gops.env.env_ocp.pyth_base_env import PythBaseEnv
from gops.env.env_ocp.resources.ref_traj_data import MultiRefTrajData
from gops.utils.math_utils import angle_normalize


class VehicleDynamicsData:
    def __init__(self):
        self.vehicle_params = dict(
            state_dim=19,
            dt=0.01,
            m_tt=4455.+168+679, # Total mass of the tractor [kg]
            ms_tt=4455.,  # Sprung mass of the tractor [kg]
            m_tl=6000+434+5000,  # Total mass of the semitrailer [kg]
            ms_tl=6000+5000,  # Sprung mass of the semitrailer [kg]
            gravity=9.81,
            Rw=0.51,
            lw=2.03,  # vehicle width
            a=1.49634995,  # Distance between the center of gravity (CG) of the tractor and its front axle [m]
            b=3.9 - 1.1154211,  # Distance between the CG of the tractor and its rear axle [m]
            lhtt=3 - 1.1154211,  # Distance between the hitch point and the center of gravity (CG) of the tractor [m]
            lhtl=6.64822164 - 3,  # Distance between the hitch point and the CG of the semitrailer [m]
            d=9.3 - 6.64822164,  # Distance between the rear axle and the CG of the semitrailer [m]
            hs_tt=1.12317534 - 0.2,  # Height of the CG of the sprung mass to the roll center for the tractor [m]
            hs_tl=0.933400843 - 0.2,  # Height of the CG of the sprung mass to the roll center for the semitrailer [m]
            hh_tt=1.07 - 0.2,  # Height of hitch point to the roll center  for the tractor [m]
            hh_tl=1.07 - 0.2,  # Height of hitch point to the roll center for the semitrailer [m]
            Izz_tt=34802.6,  # Yaw moment of inertia of the whole mass of the tractor [kg m^2]
            Ixx_tt=2283.9,  # Roll moment of inertia of the sprung mass of the tractor [kg m^2]
            Ixz_tt=1626,  # Roll–yaw product of inertia of the sprung mass of the tractor [kg m^2]
            Izz_tl=179992,  # +6000Yaw moment of inertia of the whole mass of the semitrailer [kg m^2]
            Ixx_tl=9959.7,  #+900 Roll moment of inertia of the sprung mass of the semitrailer [kg m^2]
            Ixz_tl=0.0,  # Roll–yaw product of inertia of the sprung mass of the semitrailer [kg m^2]
            kf=0.12 * 1.6 * 2.354e4 / 3.14 * 180,  # Tire cornering stiffness of the front axle of the tractor [N/rad]
            km=0.12 * 1.6 * 2.354e4 / 3.14 * 180,  # Tire cornering stiffness of the rear axle of the tractor [N/rad]
            kr=0.12 * 1.6 * 2.354e4 / 3.14 * 180,  # Tire cornering stiffness of the rear axle of the trailer [N/rad]
            kvarphi_tt=(8500+1500) / 3.14 * 180*4,  # # roll stiffness of tire (front)[N/m]
            kvarphi_tl=3000 / 3.14 * 180*2,  # roll stiffness of tire (rear)[N/m]
            ka=-100000 / 3.14 * 180, # Roll stiffness of the articulation joint between the tractor and semitrailer [N m/rad]
            cvarphi_tt=0,  # Roll damping of the tractor's suspension [N-s/m]
            cvarphi_tl=0,  # Roll damping of the semitrailer's suspension [N-s/m]
        )
        self.dt = self.vehicle_params["dt"]
        self.m_tt = self.vehicle_params["m_tt"]
        self.ms_tt = self.vehicle_params["ms_tt"]
        self.m_tl = self.vehicle_params["m_tl"]
        self.ms_tl = self.vehicle_params["ms_tl"]
        self.gravity = self.vehicle_params["gravity"]
        self.Rw = self.vehicle_params["Rw"]
        self.lw = self.vehicle_params["lw"]
        self.a = self.vehicle_params["a"]
        self.b = self.vehicle_params["b"]
        self.lhtt = self.vehicle_params["lhtt"]
        self.lhtl = self.vehicle_params["lhtl"]
        self.d = self.vehicle_params["d"]
        self.hs_tt = self.vehicle_params["hs_tt"]
        self.hs_tl = self.vehicle_params["hs_tl"]
        self.hh_tt = self.vehicle_params["hh_tt"]
        self.hh_tl = self.vehicle_params["hh_tl"]
        self.Izz_tt = self.vehicle_params["Izz_tt"]
        self.Ixx_tt = self.vehicle_params["Ixx_tt"]
        self.Ixz_tt = self.vehicle_params["Ixz_tt"]
        self.Izz_tl = self.vehicle_params["Izz_tl"]
        self.Ixx_tl = self.vehicle_params["Ixx_tl"]
        self.Ixz_tl = self.vehicle_params["Ixz_tl"]

        self.kf = self.vehicle_params["kf"]
        self.km = self.vehicle_params["km"]
        self.kr = self.vehicle_params["kr"]
        self.kvarphi_tt = self.vehicle_params["kvarphi_tt"]
        self.kvarphi_tl = self.vehicle_params["kvarphi_tl"]
        self.ka = self.vehicle_params["ka"]
        self.cvarphi_tt = self.vehicle_params["cvarphi_tt"]
        self.cvarphi_tl = self.vehicle_params["cvarphi_tl"]
        self.state_dim = self.vehicle_params["state_dim"]

        divided_tt = self.m_tt * self.Ixx_tt * self.Izz_tt - self.Izz_tt * self.ms_tt ** 2 * self.hs_tt ** 2 - self.m_tt * self.Ixz_tt ** 2
        divided_tl = self.m_tl * self.Ixx_tl * self.Izz_tl - self.Izz_tl * self.ms_tl ** 2 * self.hs_tl ** 2 - self.m_tl * self.Ixz_tl ** 2
        self.Att13 = -self.ms_tt * self.hs_tt * self.Izz_tt * (
                    self.kvarphi_tt - self.ms_tt * self.gravity * self.hs_tt - self.ka) / divided_tt
        self.Att14 = -self.ms_tt * self.hs_tt * self.Izz_tt * self.cvarphi_tt / divided_tt

        self.Att23 = -self.m_tt * self.Ixz_tt * (
                    self.kvarphi_tt - self.ms_tt * self.gravity * self.hs_tt - self.ka) / divided_tt
        self.Att24 = -self.m_tt * self.Ixz_tt * self.cvarphi_tt / divided_tt

        self.Att34 = 1
        self.Att43 = -self.m_tt * self.Izz_tt * (self.kvarphi_tt-self.ms_tt*self.gravity*self.hs_tt-self.ka)/divided_tt
        self.Att44 = -self.m_tt*self.Izz_tt*self.cvarphi_tt/divided_tt

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

        #---------------------trilaer
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
        u_tt, u_tl = states[3], states[7]
        # x y phi utt xtl ytl phitl utl vtt phidot varphi varphidot,
        # u_tt, v_tt, phidot_tt, varphi_tt, varphidot_tt,  tl
        X_matrix = np.hstack((states[3], states[8:12], states[7], states[12:16])).reshape((10, 1))
        U_matrix = actions.reshape(3, 1)

        self.Att12 = -u_tt

        Att_matrix = np.zeros((5, 5))
        Att_matrix[1, 2] = self.Att12
        Att_matrix[1, 3] = self.Att13
        Att_matrix[1, 4] = self.Att14

        Att_matrix[2, 3], Att_matrix[2, 4] = self.Att23, self.Att24

        Att_matrix[3, 4] = self.Att34

        Att_matrix[4, 3], Att_matrix[4, 4] = self.Att43, self.Att44

        self.Atl12 = -u_tl
        Atl_matrix = np.zeros((5, 5))
        Atl_matrix[1, 2] = self.Atl12
        Atl_matrix[1, 3] = self.Atl13
        Atl_matrix[1, 4] = self.Atl14

        Atl_matrix[2, 3], Atl_matrix[2, 4] = self.Atl23, self.Atl24

        Atl_matrix[3, 4] = self.Atl34

        Atl_matrix[4, 3], Atl_matrix[4, 4] = self.Atl43, self.Atl44

        A_matrix = block_diag(Att_matrix, Atl_matrix)

        Btt_matrix = np.zeros((5, 3))
        Btt_matrix[0, 0] = self.Btt00

        Btt_matrix[1, 1], Btt_matrix[1, 2] = self.Btt11, self.Btt12

        Btt_matrix[2, 1], Btt_matrix[2, 2] = self.Btt21, self.Btt22
        Btt_matrix[4, 1], Btt_matrix[4, 2] = self.Btt41, self.Btt42

        Btl_matrix = np.zeros((5, 3))
        Btl_matrix[0, 0] = self.Btl00

        Btl_matrix[1, 1], Btl_matrix[1, 2] = self.Btl11, self.Btl12

        Btl_matrix[2, 1], Btl_matrix[2, 2] = self.Btl21, self.Btl22
        Btl_matrix[4, 1], Btl_matrix[4, 2] = self.Btl41, self.Btl42

        B_matrix = block_diag(Btt_matrix, Btl_matrix)

        Ctt_matrix = np.zeros((5, 3))
        Ctt_matrix[0, 0] = self.Ctt00

        Ctt_matrix[1, 1], Ctt_matrix[1, 2] = self.Ctt11, self.Ctt12

        Ctt_matrix[2, 1], Ctt_matrix[2, 2] = self.Ctt21, self.Ctt22
        Ctt_matrix[4, 1], Ctt_matrix[4, 2] = self.Ctt41, self.Ctt42

        Ctl_matrix = np.zeros((5, 3))
        Ctl_matrix[0, 0] = self.Ctl00

        Ctl_matrix[1, 1], Ctl_matrix[1, 2] = self.Ctl11, self.Ctl12

        Ctl_matrix[2, 1], Ctl_matrix[2, 2] = self.Ctl21, self.Ctl22
        Ctl_matrix[4, 1], Ctl_matrix[4, 2] = self.Ctl41, self.Ctl42

        C_matrix = block_diag(Ctt_matrix, Ctl_matrix)#np.vstack((block_diag(Ctt_matrix, Ctl_matrix), np.zeros((2, 6))))

        Gtt_matrix = np.zeros((5, 5))
        Gtt_matrix[1, 3] = self.Gtt13

        Gtt_matrix[2, 3] = self.Gtt23

        Gtt_matrix[4, 3] = self.Gtt43

        Gtl_matrix = np.zeros((5, 5))
        Gtl_matrix[1, 3] = self.Gtl13

        Gtl_matrix[2, 3] = self.Gtl23

        Gtl_matrix[4, 3] = self.Gtl43

        G_matrix = block_diag(Gtt_matrix, Gtl_matrix)


        M_matrix = np.zeros((2, 10))
        M_matrix[0, 0], M_matrix[0, 5] = self.M00, self.M05
        # M_matrix[1, 0] = states[9]-states[13]
        M_matrix[1, 1], M_matrix[1, 2] = self.M11, self.M12
        M_matrix[1, 6], M_matrix[1, 7] = self.M16, self.M17

        self.P12, self.P17 = -u_tt, u_tt
        P_matrix = np.zeros((2, 10))
        P_matrix[1, 2], P_matrix[1, 7] = self.P12, self.P17

        N_matrix = np.zeros((4, 6))
        N_matrix[0, 0], N_matrix[0, 3] = self.N00, self.N03
        N_matrix[1, 1], N_matrix[1, 4] = self.N11, self.N14
        N_matrix[2, 2], N_matrix[2, 5] = self.N22, self.N25
        N_matrix[3, 5] = self.N35

        Q_matrix = np.zeros((4, 10))
        Q_matrix[3, 3], Q_matrix[3, 8] = self.Q33, self.Q38
        J_matrix = np.linalg.inv(np.vstack((N_matrix, np.matmul(M_matrix, C_matrix))))

        K1_matrix = np.vstack((Q_matrix, (P_matrix - np.matmul(M_matrix, A_matrix)- np.matmul(M_matrix, G_matrix))))
        K2_matrix = np.vstack((np.zeros((4, 6)), (np.matmul(M_matrix, B_matrix))))

        Lc_matrix = np.zeros((6, 12))
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

        # 计算轮胎力， 控制量[Q3, Q4, delta]
        delta = actions[2]
        Mw1 = np.array([[np.cos(delta), -np.sin(delta)],
                        [np.sin(delta), np.cos(delta)]])
        Mw2 = np.array([[np.cos(delta), -np.sin(delta)],
                        [np.sin(delta), np.cos(delta)]])
        Mw3 = np.array([[1, 0],
                        [0, 1]])
        Mw4 = np.array([[1, 0],
                        [0, 1]])
        Mw5 = np.array([[1, 0],
                        [0, 1]])
        Mw6 = np.array([[1, 0],
                        [0, 1]])

        Mw_matrix = block_diag(Mw1, Mw2, Mw3, Mw4, Mw5, Mw6)

        At_matrix = np.zeros((12, 10))

        At_matrix[1, 1], At_matrix[1, 2] = -self.kf / u_tt, -self.kf * self.a / u_tt

        At_matrix[3, 1], At_matrix[3, 2] = -self.kf / u_tt, -self.kf * self.a / u_tt

        At_matrix[5, 1], At_matrix[5, 2] = -self.km / u_tt, -self.km * (-self.b) / u_tt

        At_matrix[7, 1], At_matrix[7, 2] = -self.km / u_tt, -self.km * (-self.b) / u_tt

        At_matrix[9, 6], At_matrix[9, 7] = -self.kr / u_tl, -self.kr * (-self.d) / u_tl

        At_matrix[11, 6], At_matrix[11, 7] = -self.kr / u_tl, -self.kr * (-self.d) / u_tl

        Bt_matrix = np.zeros((12, 3))
        Bt_matrix[1, 2], Bt_matrix[3, 2] = self.kf, self.kf
        Bt_matrix[4, 0], Bt_matrix[6, 1] = 1 / self.Rw, 1 / self.Rw

        temp = np.matmul(At_matrix, X_matrix) + np.matmul(Bt_matrix, U_matrix)
        FCG = np.matmul(Lc_matrix, np.matmul(Mw_matrix, temp))
        X_dot = (np.matmul((A_matrix+G_matrix + np.matmul(C_matrix, np.matmul(J_matrix, K1_matrix))), X_matrix)
                 + np.matmul((B_matrix - np.matmul(C_matrix, np.matmul(J_matrix, K2_matrix))), FCG)).squeeze()

        X_all_dot = np.concatenate(([u_tt * np.cos(states[2]) - states[8] * np.sin(states[2])],
                                    [u_tt * np.sin(states[2]) + states[8] * np.cos(states[2])],
                                    [states[9]],
                                    [X_dot[0]],
                                    [u_tl * np.cos(states[6]) - states[12] * np.sin(states[6])],
                                    [u_tl * np.sin(states[6]) + states[12] * np.cos(states[6])],
                                    [states[13]],
                                    [X_dot[5]],
                                    X_dot[1:5],
                                    X_dot[6:10]))
        return X_all_dot

    def f_xu(self, states, actions, delta_t):
        RK = 1
        state_next = np.empty_like(states)
        # u_tt, u_tl = states[3], states[7]
        # # x y phi utt xtl ytl phitl utl vtt phidot varphi varphidot,
        # # u_tt, v_tt, phidot_tt, varphi_tt, varphidot_tt,  tl
        # X_matrix = np.hstack((states[3], states[8:12], states[7], states[12:16])).reshape((10, 1))
        # U_matrix = actions.reshape(3, 1)
        #
        # self.Att12 = -self.m_tt*u_tt
        #
        # self.Att42 = self.ms_tt*self.hs_tt*u_tt
        #
        # Mtt_matrix = np.zeros((5, 5))
        # Mtt_matrix[0, 0] = self.Mtt00
        #
        # Mtt_matrix[1, 1] = self.Mtt11
        #
        # Mtt_matrix[1, 4] = self.Mtt14
        #
        # Mtt_matrix[2, 2] = self.Mtt22
        #
        # Mtt_matrix[3, 3] = self.Mtt33
        #
        # Mtt_matrix[4, 1] = self.Mtt41
        # Mtt_matrix[4, 4] = self.Mtt44
        #
        # Att_matrix = np.zeros((5, 5))
        # Att_matrix[1, 2] = self.Att12
        # Att_matrix[3, 4] = self.Att34
        # Att_matrix[4, 2] = self.Att42
        # Att_matrix[4, 3], Att_matrix[4, 4] = self.Att43, self.Att44
        #
        # Btt_matrix = np.zeros((5, 3))
        # Btt_matrix[0, 0] = self.Btt00
        # Btt_matrix[1, 1], Btt_matrix[2, 2] = self.Btt11, self.Btt22
        #
        # Ctt_matrix = np.zeros((5, 4))
        # Ctt_matrix[0, 0], Ctt_matrix[1, 1] = self.Ctt00, self.Ctt11
        # Ctt_matrix[2, 1], Ctt_matrix[2, 2] = self.Ctt21, self.Ctt22
        # Ctt_matrix[4, 1], Ctt_matrix[4, 3] = self.Ctt41, self.Ctt43
        #
        # #--------trailer---------
        # self.Atl12 = -self.m_tl * u_tl
        #
        # self.Atl42 = self.ms_tl * self.hs_tl * u_tl
        #
        # Mtl_matrix = np.zeros((5, 5))
        # Mtl_matrix[0, 0] = self.Mtl00
        #
        # Mtl_matrix[1, 1] = self.Mtl11
        #
        # Mtl_matrix[1, 4] = self.Mtl14
        #
        # Mtl_matrix[2, 2] = self.Mtl22
        #
        # Mtl_matrix[3, 3] = self.Mtl33
        #
        # Mtl_matrix[4, 1] = self.Mtl41
        # Mtl_matrix[4, 4] = self.Mtl44
        #
        # Atl_matrix = np.zeros((5, 5))
        # Atl_matrix[1, 2] = self.Atl12
        # Atl_matrix[3, 4] = self.Atl34
        # Atl_matrix[4, 2] = self.Atl42
        # Atl_matrix[4, 3], Atl_matrix[4, 4] = self.Atl43, self.Atl44
        #
        # Btl_matrix = np.zeros((5, 3))
        # Btl_matrix[0, 0] = self.Btl00
        #
        # Btl_matrix[1, 1], Btl_matrix[2, 2] = self.Btl11, self.Btl22
        #
        # Ctl_matrix = np.zeros((5, 4))
        # Ctl_matrix[0, 0], Ctl_matrix[1, 1] = self.Ctl00, self.Ctl11
        # Ctl_matrix[2, 1], Ctl_matrix[2, 2] = self.Ctl21, self.Ctl22
        # Ctl_matrix[4, 1], Ctl_matrix[4, 3] = self.Ctl41, self.Ctl43
        #
        # A_matrix = block_diag(np.matmul(np.linalg.inv(Mtt_matrix), Att_matrix), np.matmul(np.linalg.inv(Mtl_matrix), Atl_matrix))
        #
        # B_matrix = block_diag(np.matmul(np.linalg.inv(Mtt_matrix), Btt_matrix), np.matmul(np.linalg.inv(Mtl_matrix), Btl_matrix))
        #
        # C_matrix = block_diag(np.matmul(np.linalg.inv(Mtt_matrix), Ctt_matrix), np.matmul(np.linalg.inv(Mtl_matrix), Ctl_matrix))
        #
        # M_matrix = np.zeros((2, 10))
        # M_matrix[0, 0], M_matrix[0, 5] = self.M00, self.M05
        # M_matrix[1, 1], M_matrix[1, 2] = self.M11, self.M12
        # M_matrix[1, 6], M_matrix[1, 7] = self.M16, self.M17
        #
        # self.P12, self.P17 = -u_tt, u_tt
        # P_matrix = np.zeros((2, 10))
        # P_matrix[1, 2], P_matrix[1, 7] = self.P12, self.P17
        #
        #
        # N_matrix = np.zeros((6, 8))
        # N_matrix[0, 0], N_matrix[0, 4] = self.N00, self.N04
        # N_matrix[1, 1], N_matrix[1, 5] = self.N11, self.N15
        # N_matrix[2, 2], N_matrix[2, 6] = self.N22, self.N26
        # N_matrix[3, 2] = self.N32
        # N_matrix[4, 3], N_matrix[4, 7] = self.N43, self.N47
        # N_matrix[5, 7] = self.N57
        #
        # Q_matrix = np.zeros((6, 10))
        # Q_matrix[5, 4], Q_matrix[5, 9] = self.Q54, self.Q59
        #
        # J_matrix = np.linalg.inv(np.vstack((N_matrix, np.matmul(M_matrix, C_matrix))))
        #
        # K1_matrix = np.vstack((Q_matrix, (P_matrix-np.matmul(M_matrix, A_matrix))))
        # K2_matrix = np.vstack((np.zeros((6, 6)),(np.matmul(M_matrix, B_matrix))))
        #
        # Lc_matrix = np.zeros((6, 12))
        # Lc_matrix[0, 0], Lc_matrix[0, 2], \
        # Lc_matrix[0, 4], Lc_matrix[0, 6] = 1, 1, 1, 1
        #
        # Lc_matrix[1, 1], Lc_matrix[1, 3], \
        # Lc_matrix[1, 5], Lc_matrix[1, 7], = 1, 1, 1, 1
        #
        # Lc_matrix[2, 0], Lc_matrix[2, 1], Lc_matrix[2, 2], Lc_matrix[2, 3], \
        # Lc_matrix[2, 4], Lc_matrix[2, 5], Lc_matrix[2, 6], Lc_matrix[2, 7] = \
        #     -self.lw / 2, self.a, self.lw / 2, self.a, \
        #       -self.lw / 2, -self.b, self.lw / 2, -self.b
        # Lc_matrix[3, 8], Lc_matrix[3, 10] = 1, 1
        #
        # Lc_matrix[4, 9], Lc_matrix[4, 11] = 1, 1
        #
        # Lc_matrix[5, 8], Lc_matrix[5, 9], Lc_matrix[5, 10], Lc_matrix[5, 11] = \
        #     -self.lw / 2, -self.d, \
        #     self.lw / 2, -self.d
        #
        # # 计算轮胎力， 控制量[Q3, Q4, delta]
        # delta = actions[2]
        # Mw1 = np.array([[np.cos(delta), -np.sin(delta)],
        #                 [np.sin(delta), np.cos(delta)]])
        # Mw2 = np.array([[np.cos(delta), -np.sin(delta)],
        #                 [np.sin(delta), np.cos(delta)]])
        # Mw3 = np.array([[1, 0],
        #                 [0, 1]])
        # Mw4 = np.array([[1, 0],
        #                 [0, 1]])
        # Mw5 = np.array([[1, 0],
        #                 [0, 1]])
        # Mw6 = np.array([[1, 0],
        #                 [0, 1]])
        #
        # Mw_matrix = block_diag(Mw1, Mw2, Mw3, Mw4, Mw5, Mw6)
        #
        # At_matrix = np.zeros((12, 10))
        #
        # At_matrix[1, 1], At_matrix[1, 2] = -self.kf / u_tt, -self.kf * self.a / u_tt
        #
        # At_matrix[3, 1], At_matrix[3, 2] = -self.kf / u_tt, -self.kf * self.a / u_tt
        #
        # At_matrix[5, 1], At_matrix[5, 2] = -self.km / u_tt, -self.km * (-self.b) / u_tt
        #
        # At_matrix[7, 1], At_matrix[7, 2] = -self.km / u_tt, -self.km * (-self.b) / u_tt
        #
        # At_matrix[9, 6], At_matrix[9, 7] = -self.kr / u_tl, -self.kr * (-self.d) / u_tl
        #
        # At_matrix[11, 6], At_matrix[11, 7] = -self.kr / u_tl, -self.kr * (-self.d) / u_tl
        #
        # Bt_matrix = np.zeros((12, 3))
        # Bt_matrix[1, 2], Bt_matrix[3, 2] = self.kf, self.kf
        # Bt_matrix[4, 0], Bt_matrix[6, 1] = 1 / self.Rw, 1 / self.Rw
        #
        #
        # temp = np.matmul(At_matrix, X_matrix) + np.matmul(Bt_matrix, U_matrix)
        # FCG = np.matmul(Lc_matrix, np.matmul(Mw_matrix, temp))
        # X_dot = (np.matmul((A_matrix+np.matmul(C_matrix, np.matmul(J_matrix, K1_matrix))), X_matrix)
        #          + np.matmul((B_matrix-np.matmul(C_matrix, np.matmul(J_matrix, K2_matrix))), FCG)).squeeze()
        #
        # X_all_dot = np.concatenate(([u_tt * np.cos(states[2]) - states[8] * np.sin(states[2])],
        #                             [u_tt * np.sin(states[2]) + states[8] * np.cos(states[2])],
        #                             [states[9]],
        #                             [X_dot[0]],
        #                             [u_tl * np.cos(states[6]) - states[12] * np.sin(states[6])],
        #                             [u_tl * np.sin(states[6]) + states[12] * np.cos(states[6])],
        #                             [states[13]],
        #                             [X_dot[5]],
        #                             X_dot[1:5],
        #                             X_dot[6:10]))
        # if RK == 1:
        #     # state_next[0] = states[0] + delta_t * (u_tt * np.cos(states[2]) - states[8] * np.sin(states[2]))  # px_tt
        #     # state_next[1] = states[1] + delta_t * (u_tt * np.sin(states[2]) + states[8] * np.cos(states[2]))  # py_tt
        #     # state_next[2] = states[2] + delta_t * states[9]  # phi_tt
        #     # state_next[3] = states[3] + delta_t * X_dot[0]  # u_tt
        #     #
        #     # state_next[4] = states[4] + delta_t * (u_tl * np.cos(states[6]) - states[12] * np.sin(states[6]))  # px_tl
        #     # state_next[5] = states[5] + delta_t * (u_tl * np.sin(states[6]) + states[12] * np.cos(states[6]))  # py_tt
        #     # state_next[6] = states[6] + delta_t * states[13]  # phi_tl
        #     # state_next[7] = states[7] + delta_t * X_dot[5]  # u_tl
        #     #
        #     # state_next[8:12] = states[8:12] + delta_t * X_dot[1:5]
        #     # state_next[12:16] = states[12:16] + delta_t * X_dot[6:10]
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
        u_tt, u_tl = states[3], states[7]
        # x y phi utt xtl ytl phitl utl vtt phidot varphi varphidot,
        # u_tt, v_tt, phidot_tt, varphi_tt, varphidot_tt,  tl
        X_matrix = np.hstack((states[3], states[8:12], states[7], states[12:16])).reshape((10, 1))
        U_matrix = actions.reshape(3, 1)

        self.Att12 = -u_tt

        Att_matrix = np.zeros((5, 5))
        Att_matrix[1, 2] = self.Att12
        Att_matrix[1, 3] = self.Att13
        Att_matrix[1, 4] = self.Att14

        Att_matrix[2, 3], Att_matrix[2, 4] = self.Att23, self.Att24

        Att_matrix[3, 4] = self.Att34

        Att_matrix[4, 3], Att_matrix[4, 4] = self.Att43, self.Att44

        self.Atl12 = -u_tl
        Atl_matrix = np.zeros((5, 5))
        Atl_matrix[1, 2] = self.Atl12
        Atl_matrix[1, 3] = self.Atl13
        Atl_matrix[1, 4] = self.Atl14

        Atl_matrix[2, 3], Atl_matrix[2, 4] = self.Atl23, self.Atl24

        Atl_matrix[3, 4] = self.Atl34

        Atl_matrix[4, 3], Atl_matrix[4, 4] = self.Atl43, self.Atl44

        A_matrix = block_diag(Att_matrix, Atl_matrix)

        Btt_matrix = np.zeros((5, 3))
        Btt_matrix[0, 0] = self.Btt00

        Btt_matrix[1, 1], Btt_matrix[1, 2] = self.Btt11, self.Btt12

        Btt_matrix[2, 1], Btt_matrix[2, 2] = self.Btt21, self.Btt22
        Btt_matrix[4, 1], Btt_matrix[4, 2] = self.Btt41, self.Btt42

        Btl_matrix = np.zeros((5, 3))
        Btl_matrix[0, 0] = self.Btl00

        Btl_matrix[1, 1], Btl_matrix[1, 2] = self.Btl11, self.Btl12

        Btl_matrix[2, 1], Btl_matrix[2, 2] = self.Btl21, self.Btl22
        Btl_matrix[4, 1], Btl_matrix[4, 2] = self.Btl41, self.Btl42

        B_matrix = block_diag(Btt_matrix, Btl_matrix)

        Ctt_matrix = np.zeros((5, 3))
        Ctt_matrix[0, 0] = self.Ctt00

        Ctt_matrix[1, 1], Ctt_matrix[1, 2] = self.Ctt11, self.Ctt12

        Ctt_matrix[2, 1], Ctt_matrix[2, 2] = self.Ctt21, self.Ctt22
        Ctt_matrix[4, 1], Ctt_matrix[4, 2] = self.Ctt41, self.Ctt42

        Ctl_matrix = np.zeros((5, 3))
        Ctl_matrix[0, 0] = self.Ctl00

        Ctl_matrix[1, 1], Ctl_matrix[1, 2] = self.Ctl11, self.Ctl12

        Ctl_matrix[2, 1], Ctl_matrix[2, 2] = self.Ctl21, self.Ctl22
        Ctl_matrix[4, 1], Ctl_matrix[4, 2] = self.Ctl41, self.Ctl42

        C_matrix = block_diag(Ctt_matrix,
                              Ctl_matrix)  # np.vstack((block_diag(Ctt_matrix, Ctl_matrix), np.zeros((2, 6))))

        Gtt_matrix = np.zeros((5, 5))
        Gtt_matrix[1, 3] = self.Gtt13

        Gtt_matrix[2, 3] = self.Gtt23

        Gtt_matrix[4, 3] = self.Gtt43

        Gtl_matrix = np.zeros((5, 5))
        Gtl_matrix[1, 3] = self.Gtl13

        Gtl_matrix[2, 3] = self.Gtl23

        Gtl_matrix[4, 3] = self.Gtl43

        G_matrix = block_diag(Gtt_matrix, Gtl_matrix)

        M_matrix = np.zeros((2, 10))
        M_matrix[0, 0], M_matrix[0, 5] = self.M00, self.M05
        # M_matrix[1, 0] = states[9]-states[13]
        M_matrix[1, 1], M_matrix[1, 2] = self.M11, self.M12
        M_matrix[1, 6], M_matrix[1, 7] = self.M16, self.M17

        self.P12, self.P17 = -u_tt, u_tt
        P_matrix = np.zeros((2, 10))
        P_matrix[1, 2], P_matrix[1, 7] = self.P12, self.P17

        N_matrix = np.zeros((4, 6))
        N_matrix[0, 0], N_matrix[0, 3] = self.N00, self.N03
        N_matrix[1, 1], N_matrix[1, 4] = self.N11, self.N14
        N_matrix[2, 2], N_matrix[2, 5] = self.N22, self.N25
        N_matrix[3, 5] = self.N35

        Q_matrix = np.zeros((4, 10))
        Q_matrix[3, 3], Q_matrix[3, 8] = self.Q33, self.Q38
        J_matrix = np.linalg.inv(np.vstack((N_matrix, np.matmul(M_matrix, C_matrix))))

        K1_matrix = np.vstack((Q_matrix, (P_matrix - np.matmul(M_matrix, A_matrix) - np.matmul(M_matrix, G_matrix))))
        K2_matrix = np.vstack((np.zeros((4, 6)), (np.matmul(M_matrix, B_matrix))))

        Lc_matrix = np.zeros((6, 12))
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

        # 计算轮胎力， 控制量[Q3, Q4, delta]
        delta = actions[2]
        Mw1 = np.array([[np.cos(delta), -np.sin(delta)],
                        [np.sin(delta), np.cos(delta)]])
        Mw2 = np.array([[np.cos(delta), -np.sin(delta)],
                        [np.sin(delta), np.cos(delta)]])
        Mw3 = np.array([[1, 0],
                        [0, 1]])
        Mw4 = np.array([[1, 0],
                        [0, 1]])
        Mw5 = np.array([[1, 0],
                        [0, 1]])
        Mw6 = np.array([[1, 0],
                        [0, 1]])

        Mw_matrix = block_diag(Mw1, Mw2, Mw3, Mw4, Mw5, Mw6)

        At_matrix = np.zeros((12, 10))

        At_matrix[1, 1], At_matrix[1, 2] = -self.kf / u_tt, -self.kf * self.a / u_tt

        At_matrix[3, 1], At_matrix[3, 2] = -self.kf / u_tt, -self.kf * self.a / u_tt

        At_matrix[5, 1], At_matrix[5, 2] = -self.km / u_tt, -self.km * (-self.b) / u_tt

        At_matrix[7, 1], At_matrix[7, 2] = -self.km / u_tt, -self.km * (-self.b) / u_tt

        At_matrix[9, 6], At_matrix[9, 7] = -self.kr / u_tl, -self.kr * (-self.d) / u_tl

        At_matrix[11, 6], At_matrix[11, 7] = -self.kr / u_tl, -self.kr * (-self.d) / u_tl

        Bt_matrix = np.zeros((12, 3))
        Bt_matrix[1, 2], Bt_matrix[3, 2] = self.kf, self.kf
        Bt_matrix[4, 0], Bt_matrix[6, 1] = 1 / self.Rw, 1 / self.Rw

        temp = np.matmul(At_matrix, X_matrix) + np.matmul(Bt_matrix, U_matrix)
        FCG = np.matmul(Lc_matrix, np.matmul(Mw_matrix, temp))
        X_dot = (np.matmul((A_matrix + G_matrix + np.matmul(C_matrix, np.matmul(J_matrix, K1_matrix))), X_matrix)
                 + np.matmul((B_matrix - np.matmul(C_matrix, np.matmul(J_matrix, K2_matrix))), FCG)).squeeze()

        state_next[0] = states[0] + delta_t * (u_tt * np.cos(states[2]) - states[8] * np.sin(states[2]))  # px_tt
        state_next[1] = states[1] + delta_t * (u_tt * np.sin(states[2]) + states[8] * np.cos(states[2]))  # py_tt
        state_next[2] = states[2] + delta_t * states[9]  # phi_tt
        state_next[3] = states[3] + delta_t * X_dot[0]  # u_tt

        state_next[4] = states[4] + delta_t * (u_tl * np.cos(states[6]) - states[12] * np.sin(states[6]))  # px_tl
        state_next[5] = states[5] + delta_t * (u_tl * np.sin(states[6]) + states[12] * np.cos(states[6]))  # py_tt
        state_next[6] = states[6] + delta_t * states[13]  # phi_tl
        state_next[7] = states[7] + delta_t * X_dot[5]  # u_tl

        state_next[8:12] = states[8:12] + delta_t * X_dot[1:5]
        state_next[12:16] = states[12:16] + delta_t * X_dot[6:10]
        state_next[16:19] = actions
        return state_next

class Semitruckpu6dof(PythBaseEnv):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    def __init__(
        self,
        pre_horizon: int = 20,
        path_para: Optional[Dict[str, Dict]] = None,
        u_para: Optional[Dict[str, Dict]] = None,
        max_steer: float = 0.5,
        **kwargs,
    ):
        work_space = kwargs.pop("work_space", None)
        if work_space is None:
            # 用高斯分布去采样
            # delta_x_tt, delta_y_tt, delta_phi_tt, delta_u_tt，delta_x_tl, delta_y_tl, delta_phi_tl, delta_u_tl
            # v_tt, phidot_tt, varphi_tt, varphidot_tt， v_tl, phidot_tl, varphi_tl, varphidot_tl
            init_high = np.array([2, 2, 0.1, 2, 2, 2, 0.1, 2,
                                  0.1, 0.1, 0.01, 0.1, 0.1, 0.1, 0.01, 0.1, 0.1, 0.1, 0.01], dtype=np.float32)
            init_low = np.array([-2, -2,  -0.1, -2, -2, -2, -0.1, -2,
                                 -0.1, -0.1, -0.01, -0.1, -0.1, -0.1, -0.01, -0.1, -0.1, -0.1, -0.01], dtype=np.float32)
            work_space = np.stack((init_low, init_high))
        super(Semitruckpu6dof, self).__init__(work_space=work_space, **kwargs)

        self.vehicle_dynamics = VehicleDynamicsData()
        self.state_dim = self.vehicle_dynamics.state_dim
        self.ref_traj = MultiRefTrajData(path_para, u_para)
        self.pre_horizon = pre_horizon
        ego_obs_dim = self.state_dim
        ref_obs_dim = 8
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf] * (ego_obs_dim + ref_obs_dim * pre_horizon)),
            high=np.array([np.inf] * (ego_obs_dim + ref_obs_dim * pre_horizon)),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-50, -50, -max_steer]),
            high=np.array([50, 50, max_steer]),
            dtype=np.float32,
        )
        max_torque = 2000
        self.action_psc_space = gym.spaces.Box(
            low=np.array([0] * 2 + [-0.5]),
            high=np.array([max_torque] * 2 + [0.5]),
            dtype=np.float32,
        )
        obs_scale_default = [1/100, 1/100, 1/10, 1/10, 1/(max_torque*10), 1/10]
        self.obs_scale = np.array(kwargs.get('obs_scale', obs_scale_default))

        self.dt = self.vehicle_dynamics.dt
        self.max_episode_steps = 200

        self.state = None
        self.ref_points = None
        self.t = None
        self.t2 = None
        self.path_num = None
        self.u_num = None

        self.info_dict = {
            "state": {"shape": (self.state_dim,), "dtype": np.float32},
            "ref_points": {"shape": (self.pre_horizon + 1, 8), "dtype": np.float32},
            "ref": {"shape": (8,), "dtype": np.float32},
            "path_num": {"shape": (), "dtype": np.uint8},
            "u_num": {"shape": (), "dtype": np.uint8},
            "target_speed": {"shape": (), "dtype": np.float32},
            "ref_time": {"shape": (), "dtype": np.float32},
            "ref_time2": {"shape": (), "dtype": np.float32},
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
            self.t = 20.0 * self.np_random.uniform(0.0, 1.0)

        # Calculate path num and speed num: ref_num = [0, 1, 2,..., 7]
        if ref_num is None:
            path_num = None
            u_num = None
        else:
            path_num = int(ref_num / 2)
            u_num = int(ref_num % 2)

        # If no ref_num, then randomly select path and speed
        if path_num is not None:
            self.path_num = path_num
        else:
            self.path_num = self.np_random.choice([0, 1, 2, 3, 4, 5]) #, 6

        if u_num is not None:
            self.u_num = u_num
        else:
            self.u_num = self.np_random.choice([0, 1])

        ref_points = []
        for i in range(self.pre_horizon + 1):
            ref_x = self.ref_traj.compute_x(
                self.t + i * self.dt, self.path_num, self.u_num
            )
            ref_y = self.ref_traj.compute_y(
                self.t + i * self.dt, self.path_num, self.u_num
            )
            ref_phi = self.ref_traj.compute_phi(
                self.t + i * self.dt, self.path_num, self.u_num)
            ref_u = self.ref_traj.compute_u(
                self.t + i * self.dt, self.path_num, self.u_num
            )

            self.t2 = self.t - (self.vehicle_dynamics.lhtt+self.vehicle_dynamics.lhtl)/ref_u
            ref_x2 = self.ref_traj.compute_x(
                self.t2 + i * self.dt, self.path_num, self.u_num
            )
            ref_y2 = self.ref_traj.compute_y(
                self.t2 + i * self.dt, self.path_num, self.u_num
            )
            ref_phi2 = self.ref_traj.compute_phi(
                self.t2 + i * self.dt, self.path_num, self.u_num)
            ref_u2 = self.ref_traj.compute_u(
                self.t2 + i * self.dt, self.path_num, self.u_num)
            ref_points.append([ref_x, ref_y, ref_phi, ref_u, ref_x2, ref_y2, ref_phi2, ref_u2])
        self.ref_points = np.array(ref_points, dtype=np.float32)

        if init_state is not None:
            delta_state = np.array(init_state, dtype=np.float32)
        else:
            delta_state = self.sample_initial_state()
        torque = np.random.uniform(800, 2000)
        steer = np.random.uniform(-0.1, 0.1)
        action_psc = np.concatenate((torque + delta_state[16:18], steer + delta_state[18:]))
        self.state = np.concatenate(
            (self.ref_points[0] + delta_state[:8], delta_state[8:16], action_psc))#np.array([0,0,0,20,0,0,0,20])
        obs = self.get_obs()
        return obs, self.info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        action = np.clip(action, self.action_space.low, self.action_space.high)

        reward = self.compute_reward(action)
        action_psc = self.state[16:] + action
        action_psc = np.clip(action_psc, self.action_psc_space.low, self.action_psc_space.high)
        self.state = self.vehicle_dynamics.f_xu(self.state, action_psc, self.dt)
        print(self.ref_points[0][:4],self.state[:4])
        self.t = self.t + self.dt
        self.t2 = self.t2 + self.dt

        self.ref_points[:-1] = self.ref_points[1:]
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

        new_ref_point_2 = np.array(
            [
                self.ref_traj.compute_x(
                    self.t2 + self.pre_horizon * self.dt, self.path_num, self.u_num
                ),
                self.ref_traj.compute_y(
                    self.t2 + self.pre_horizon * self.dt, self.path_num, self.u_num
                ),
                self.ref_traj.compute_phi(
                    self.t2 + self.pre_horizon * self.dt, self.path_num, self.u_num
                ),
                self.ref_traj.compute_u(
                    self.t2 + self.pre_horizon * self.dt, self.path_num, self.u_num
                ),
            ],
            dtype=np.float32,
        )
        self.ref_points[-1] = np.hstack((new_ref_point, new_ref_point_2))

        obs = self.get_obs()
        self.done = self.judge_done()

        self.action_last = action
        return obs, reward, self.done, self.info

    def get_obs(self) -> np.ndarray:
        ref_x_tf, ref_y_tf, ref_phi_tf = \
            ego_vehicle_coordinate_transform(
                self.state[0], self.state[1], self.state[2],
                self.ref_points[:, 0], self.ref_points[:, 1], self.ref_points[:, 2],
            )
        ref_u_tf = self.ref_points[:, 3] - self.state[3]
        ref_x2_tf, ref_y2_tf, ref_phi2_tf = \
            ego_vehicle_coordinate_transform(
                self.state[4], self.state[5], self.state[6],
                self.ref_points[:, 4], self.ref_points[:, 5], self.ref_points[:, 6],
            )
        ref_u2_tf = self.ref_points[:, 7] - self.state[7]
        # ego_obs: [
        # delta_y, delta_phi, delta_y2, delta_phi2 (of the first reference point)
        # v, w, varphi (of ego vehicle, including tractor and trailer)
        # ]

        ego_obs = np.concatenate(
            ([ref_x_tf[0]*self.obs_scale[0], ref_y_tf[0]*self.obs_scale[1], ref_phi_tf[0]*self.obs_scale[2], ref_u_tf[0]*self.obs_scale[3],
              ref_x2_tf[0]*self.obs_scale[0], ref_y2_tf[0]*self.obs_scale[1], ref_phi2_tf[0]*self.obs_scale[2], ref_u2_tf[0]*self.obs_scale[3]],
             self.state[8:16], [self.state[16]*self.obs_scale[4], self.state[17]*self.obs_scale[4], self.state[18]*self.obs_scale[5]]))
        # ref_obs: [
        # delta_y, delta_phi (of the second to last reference point)
        # ]
        ref_obs = np.stack((ref_x_tf*self.obs_scale[0], ref_y_tf*self.obs_scale[1], ref_phi_tf*self.obs_scale[2], ref_u_tf*self.obs_scale[3],
                            ref_x2_tf*self.obs_scale[0],ref_y2_tf*self.obs_scale[1], ref_phi2_tf*self.obs_scale[2], ref_u2_tf*self.obs_scale[3]), 1)[1:].flatten()
        return np.concatenate((ego_obs, ref_obs))

    def compute_reward(self, action: np.ndarray) -> float:
        px1, py1, phi1, u1, px2, py2, phi2,u2, v1, phi1_dot, varphi1, varphi1_dot, \
        v2, phi2_dot, varphi2, varphi2_dot = self.state[:16]

        ref_x, ref_y, ref_phi, ref_u = self.ref_points[0][0:4]
        steer = action[2]
        return -(
            1 * ((py1 - ref_y) ** 2) #(px1 - ref_x) ** 2 +
            + 1 * ((u1 - ref_u) ** 2)
            + 0.8 * angle_normalize(phi1 - ref_phi) ** 2
            + 0.5 * phi1_dot ** 2
            + 0.5 * varphi1 ** 2
            + 0.5 * varphi1_dot ** 2
            + 0.4 * (steer/0.02) ** 2
            + 0.4 * ((action[0]/100) ** 2+(action[1]/100) ** 2)
        )

    def judge_done(self) -> bool:
        done = ((np.abs(self.state[1]-self.ref_points[0, 1]) > 3)  # delta_y1
                  + (np.abs(angle_normalize(self.state[2]-self.ref_points[0, 2])) > np.pi)  # delta_phi1
                  + (np.abs(self.state[3]-self.ref_points[0, 3]) > 5)  # delta_u1
                  + (np.abs(self.state[5]-self.ref_points[0, 5]) > 3) # delta_y2
                  + (np.abs(angle_normalize(self.state[6]-self.ref_points[0, 6])) > np.pi))  # delta_phi2
        #(np.abs(self.state[0]-self.ref_points[0, 0]) > 5)+   # delta_x1
        #+ (np.abs(self.state[3] - self.ref_points[0, 3]) > 5)  # delta_x2
        return done

    @property
    def info(self) -> dict:
        return {
            "state": self.state.copy(),
            "ref_points": self.ref_points.copy(),
            "ref": self.ref_points[0].copy(),
            "path_num": self.path_num,
            "u_num": self.u_num,
            "ref_time": self.t,
            "ref_time2": self.t2,
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
    make env `pyth_semitruckpu6dof`
    """
    return Semitruckpu6dof(**kwargs)

#
# if __name__ == "__main__":
#     env = env_creator()
#     env.reset()
#     for i in range(50):
#         # a = env.action_space.sample()
#         a = np.array([0.1, 0.0])
#         print(i)
#         obs, reward, done, info = env.step(a)
#         # print(obs)
#         # env.render()