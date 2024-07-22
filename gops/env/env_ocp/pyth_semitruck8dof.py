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
        root_dir = current_dir + "/resources/cury.csv"
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
        ref_vx = np.empty_like(ref_x) + self.ref_vx
        return np.stack([ref_x, ref_y, ref_heading, ref_vx], axis=1)

class VehicleDynamicsData:
    def __init__(self):
        self.vehicle_params = dict(
        state_dim=16,
        m1=4455. + 168 + 679, #  # Total mass of the tractor [kg]
        m1s=4455, # 4455.  # Sprung mass of the tractor [kg]
        m2=6000 + 445 , # 6000 + 445 + 91  # Total mass of the semitrailer [kg]
        m2s=6000, # 6000  # Sprung mass of the semitrailer [kg]
        gravity=9.81,
        a=1.10588255, # 1.10588255  # Distance between the center of gravity (CG) of the tractor and its front axle [m]
        b=3.9-1.10588255, # 3.9-1.10588255  # Distance between the CG of the tractor and its rear axle [m]
        c=3.0-1.10588255, # 3.0-1.10588255  # Distance between the hitch point and the center of gravity (CG) of the tractor [m]
        e=8.21291021-3.0, # 8.21291021-3.0  # Distance between the hitch point and the CG of the semitrailer [m]
        d=9.466 - 8.21291021, # 12.46502325 - 8.71811794  # Distance between the rear axle and the CG of the semitrailer [m]

        h1s=1.12977506- ((1.04-0.528)/2+0.528), # ((1.04-0.528)/2+0.528)  # Height of the CG of the sprung mass for the tractor [m]
        h2s=1.89301435-((1.04-0.528)/2+0.528), #   # Height of the CG of the sprung mass for the semitrailer [m]
        hh=1.07-((1.04-0.528)/2+0.528),  # 1.07-0.528  # Height of hitch point to the roll center of the sprung mass for the tractor [m]
        I1xx=2283.9,  # Roll moment of inertia of the sprung mass of the tractor [kg m^2]
        I1yy=35402.8,
        I1zz=34802.6 , # Yaw moment of inertia of the whole mass of the tractor [kg m^2]
        I1xz=1626 , # Roll–yaw product of inertia of the sprung mass of the tractor [kg m^2]
        I2zz=54000.0, # 54000.0  # Yaw moment of inertia of the whole mass of the semitrailer [kg m^2]
        I2xx=10140.0, # 10140.0  # Roll moment of inertia of the sprung mass of the semitrailer [kg m^2]
        I2xz=0.0, # Roll–yaw product of inertia of the sprung mass of the semitrailer [kg m^2]
        I2yy=54000.0,
        k1=-0.12*1.6* 2.354e+04/3.14 * 180,    # Tire cornering stiffness of the front axle of the tractor [N/rad]
        k2=-0.12*1.6* 2.354e+04/3.14 * 180,   # Tire cornering stiffness of the rear axle of the tractor [N/rad]
        k3=-0.12*1.6* 2.354e+04/3.14 * 180,  # Tire cornering stiffness of the rear axle of the trailer [N/rad]
        k_varphi1=1500/3.14*180*4,   # roll stiffness of tire (front)[N/m]
        k_varphi2=3000/3.14*180*2,   # roll stiffness of tire (rear)[N/m]
        k12=100000/3.14*180,  # Roll stiffness of the articulation joint between the tractor and semitrailer [N m/rad]
        c_varphi1=0,  # 0  # Roll damping of the tractor's suspension [N-s/m]
        c_varphi2=0,  # 0  # Roll damping of the semitrailer's suspension [N-s/m]
        )
        self.state_dim = self.vehicle_params["state_dim"]
        self.m1 = self.vehicle_params["m1"]  # Total mass of the tractor [kg]
        self.m1s = self.vehicle_params["m1s"]  # Sprung mass of the tractor [kg]
        self.m2 = self.vehicle_params["m2"]  # Total mass of the semitrailer [kg]
        self.m2s = self.vehicle_params["m2s"]  # Sprung mass of the semitrailer [kg]
        self.gravity = self.vehicle_params["gravity"]
        self.a = self.vehicle_params["a"]  # Distance between the center of gravity (CG) of the tractor and its front axle [m]
        self.b = self.vehicle_params["b"]  # Distance between the CG of the tractor and its rear axle [m]
        self.c = self.vehicle_params["c"]  # Distance between the hitch point and the center of gravity (CG) of the tractor [m]
        self.e = self.vehicle_params["e"]  # Distance between the hitch point and the CG of the semitrailer [m]
        self.d = self.vehicle_params["d"]  # Distance between the rear axle and the CG of the semitrailer [m]

        self.h1s = self.vehicle_params["h1s"]  # Height of the CG of the sprung mass for the tractor [m]
        self.h2s = self.vehicle_params["h2s"]  # Height of the CG of the sprung mass for the semitrailer [m]
        self.hh = self.vehicle_params["hh"]  # Height of hitch point to the roll center of the sprung mass for the tractor [m]

        self.I1xx = self.vehicle_params["I1xx"]  # Roll moment of inertia of the sprung mass of the tractor [kg m^2]
        self.I1yy = self.vehicle_params["I1yy"]
        self.I1zz = self.vehicle_params["I1zz"]  # Yaw moment of inertia of the whole mass of the tractor [kg m^2]
        self.I1xz = self.vehicle_params["I1xz"]  # Roll–yaw product of inertia of the sprung mass of the tractor [kg m^2]

        self.I2zz = self.vehicle_params["I2zz"]  # Yaw moment of inertia of the whole mass of the semitrailer [kg m^2]
        self.I2xx = self.vehicle_params["I2xx"]  # Roll moment of inertia of the sprung mass of the semitrailer [kg m^2]
        self.I2xz = self.vehicle_params["I2xz"]  # Roll–yaw product of inertia of the sprung mass of the semitrailer [kg m^2]
        self.I2yy = self.vehicle_params["I2yy"]
        self.k1 = self.vehicle_params["k1"]  # Tire cornering stiffness of the front axle of the tractor [N/rad]
        self.k2 = self.vehicle_params["k2"]  # Tire cornering stiffness of the rear axle of the tractor [N/rad]
        self.k3 = self.vehicle_params["k3"]  # Tire cornering stiffness of the rear axle of the trailer [N/rad]
        self.k_varphi1 = self.vehicle_params["k_varphi1"]  # roll stiffness of tire (front)[N/m]
        self.k_varphi2 = self.vehicle_params["k_varphi2"]  # roll stiffness of tire (rear)[N/m]
        self.k12 = self.vehicle_params["k12"]  # Roll stiffness of the articulation joint between the tractor and semitrailer [N m/rad]
        self.c_varphi1 = self.vehicle_params["c_varphi1"]  # Roll damping of the tractor's suspension [N-s/m]
        self.c_varphi2 = self.vehicle_params["c_varphi2"]  # Roll damping of the semitrailer's suspension [N-s/m]

    def f_xu(self, states, actions, delta_t):
        x_1, y_1, phi_1, v_x1, x_2, y_2, phi_2, v_x2, \
        v_y1, gamma_1, varphi_1, varphi_1dot, \
        v_y2, gamma_2, varphi_2, varphi_2dot = states
        state_next = np.empty_like(states)
        X = np.hstack([states[3], states[8:12], states[7], states[12:self.state_dim]])

        M_matrix = np.zeros((self.state_dim - 6, self.state_dim - 6))
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
        self.M26 = -self.m2 * self.c
        self.M29 = -self.m2s * self.h2s * self.c

        self.M33 = 1

        self.M41 = self.m1s * self.h1s
        self.M42 = -self.I1xz
        self.M44 = self.I1xx + 2*self.m1s * self.h1s ** 2
        self.M46 = -self.m2 * self.hh
        self.M49 = self.m2s * self.hh * self.h2s

        self.M50 = -np.cos(phi_2 - phi_1)
        self.M51 = np.sin(phi_2 - phi_1)
        self.M52 = -np.sin(phi_2 - phi_1) * self.c
        self.M55 = 1

        self.M60 = -np.sin(phi_2 - phi_1)
        self.M61 = -np.cos(phi_2 - phi_1)
        self.M62 = np.cos(phi_2 - phi_1) * self.c
        self.M66 = 1
        self.M67 = self.c

        self.M75 = -self.m2s * self.h2s * varphi_2
        self.M76 = -self.e * self.m2
        self.M77 = self.I2zz
        self.M79 = (self.m2s * self.h2s - self.I2xz)

        self.M88 = 1

        self.M96 = self.m2s * self.h2s+self.hh*self.m2
        self.M97 = -self.I2xz
        self.M99 = (self.I2xx + self.m2s * self.h2s ** 2-self.m2s*self.h2s*self.hh)

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

        A_matrix = np.zeros((self.state_dim - 6, self.state_dim - 6))
        self.A02 = -self.m1 * v_y1
        self.A04 = -self.m1s * self.h1s * 2 * gamma_1
        self.A07 = -self.m2 * v_y2
        self.A09 = -2 * self.m2s * self.h2s * gamma_2

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
        self.A43 = -(self.k_varphi1 + self.k12 - self.m1s * self.gravity * self.h1s  - self.I1yy * gamma_1 ** 2 + self.I1zz * gamma_1 ** 2)
        self.A44 = -self.c_varphi1
        self.A45 = (self.m2 * self.hh) * gamma_2
        self.A46 = -self.hh * self.k3 / v_x2
        self.A47 = self.d * self.k3 * self.hh / v_x2
        self.A48 = (self.k12 + self.m2s * self.h2s * self.hh * gamma_2 ** 2)

        self.A50 = -(gamma_2 - gamma_1) * np.sin(phi_2 - phi_1)
        self.A51 = -(gamma_2 - gamma_1) * np.cos(phi_2 - phi_1)
        self.A52 = (gamma_2 - gamma_1) * np.cos(phi_2 - phi_1) * self.c

        self.A60 = (gamma_2 - gamma_1) * np.cos(phi_2 - phi_1)
        self.A61 = -(gamma_2 - gamma_1) * np.sin(phi_2 - phi_1)
        self.A62 = self.c * (gamma_2 - gamma_1) * np.sin(phi_2 - phi_1)

        self.A75 = self.e * self.m2 * gamma_2
        self.A76 = -(self.d + self.e) * self.k3 / v_x2
        self.A77 = (self.d + self.e) * self.k3 * self.d / v_x2
        self.A78 = (self.e * self.m2s * self.h2s * gamma_2 ** 2 - self.m2s * self.h2s * gamma_2 * v_y2)

        self.A89 = 1

        self.A93, self.A95, self.A96, self.A97, self.A98, self.A99 = self.k12, -(self.m2s * self.h2s * gamma_2+self.hh*self.m2*gamma_2), \
                                                                     self.hh * self.k3 / v_x2, -self.d * self.hh * self.k3 / v_x2, (
                    self.m2s * self.gravity * self.h2s - self.k_varphi2 - self.k12 + self.I2yy * gamma_2 ** 2 - self.I2zz * gamma_2 ** 2-self.m2s*self.h2s*self.hh*gamma_2**2), -self.c_varphi2


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

        B_matrix = np.zeros((self.state_dim - 6, 2))
        self.B00 = - (self.m1 + self.m2)*1.0
        self.B11 = -self.k1
        self.B21 = -self.a * self.k1
        B_matrix[0, 0] = self.B00
        B_matrix[1, 1] = self.B11
        B_matrix[2, 1] = self.B21

        X_dot = (np.matmul(np.matmul(np.linalg.inv(M_matrix), A_matrix), X) + np.matmul(
            np.linalg.inv(M_matrix), np.matmul(B_matrix, actions))).squeeze()

        state_next[0] = x_1 + delta_t * (v_x1 * np.cos(phi_1) - v_y1 * np.sin(phi_1))
        state_next[1] = y_1 + delta_t * (v_y1 * np.cos(phi_1) + v_x1 * np.sin(phi_1))
        state_next[2] = phi_1 + delta_t * gamma_1
        state_next[3] = v_x1 + delta_t * X_dot[0]

        state_next[4] = x_2 + delta_t * (v_x2 * np.cos(-phi_2) - v_y2 * np.sin(-phi_2))  # state_next[12] - self.b * np.cos(phi_1) - self.e * np.cos(phi_2)  # posx_trailer
        state_next[5] = y_2 - delta_t * (v_y2 * np.cos(-phi_2) + v_x2 * np.sin(-phi_2))  # posy_trailer
        state_next[6] = phi_2 + delta_t * gamma_2
        state_next[7] = v_x2 + delta_t * X_dot[5]

        state_next[8:12] = states[8:12] + delta_t * X_dot[1:5]
        state_next[12:] = states[12:] + delta_t * X_dot[6:]
        return state_next

    def f_xu_0624(self, states, actions, delta_t):
        x_1, y_1, phi_1, v_x1, x_2, y_2, phi_2, v_x2, \
        v_y1, gamma_1, varphi_1, varphi_1dot, \
        v_y2, gamma_2, varphi_2, varphi_2dot = states
        state_next = np.empty_like(states)
        X = np.hstack([states[3], states[8:12], states[7], states[12:self.state_dim]])

        M_matrix = np.zeros((self.state_dim - 6, self.state_dim - 6))
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
        self.M26 = -self.m2 * self.c
        self.M29 = -self.m2s * self.h2s * self.c

        self.M33 = 1

        self.M41 = self.m1s * self.h1s
        self.M42 = self.I1xz
        self.M44 = -self.I1xx - 2 * self.m1s * self.h1s ** 2
        self.M46 = -self.m2 * self.hh
        self.M49 = self.m2s * self.hh * self.h2s

        self.M50 = -np.cos(phi_2 - phi_1)
        self.M51 = np.sin(phi_2 - phi_1)
        self.M52 = -np.sin(phi_2 - phi_1) * self.c
        self.M55 = 1

        self.M60 = -np.sin(phi_2 - phi_1)
        self.M61 = -np.cos(phi_2 - phi_1)
        self.M62 = np.cos(phi_2 - phi_1) * self.c
        self.M66 = 1
        self.M67 = self.c

        self.M75 = -self.m2s * self.h2s * varphi_2
        self.M76 = -self.e * self.m2
        self.M77 = self.I2zz
        self.M79 = (self.m2s * self.h2s - self.I2xz)

        self.M88 = 1

        self.M96 = self.m2s * self.h2s - self.hh * self.m2
        self.M97 = self.I2xz
        self.M99 = -(self.I2xx + 2 * self.m2s * self.h2s ** 2 - self.m2s * self.h2s * self.hh)

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

        M_matrix[5, 0], M_matrix[5, 1], M_matrix[5, 2], M_matrix[5,] = \
            self.M50, self.M51, self.M52, self.M55

        M_matrix[6, 0], M_matrix[6, 1], M_matrix[6, 2], M_matrix[6, 6], M_matrix[6, 7] = \
            self.M60, self.M61, self.M62, self.M66, self.M67

        M_matrix[7, 5], M_matrix[7, 6], M_matrix[7, 7], M_matrix[7, 9] = self.M75, self.M76, self.M77, self.M79

        M_matrix[8, 8] = self.M88

        M_matrix[9, 6], M_matrix[9, 7], M_matrix[9, 9] = self.M96, self.M97, self.M99

        A_matrix = np.zeros((self.state_dim - 6, self.state_dim - 6))
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
        self.A28 = -self.m2s * self.h2s * self.c * gamma_2 ** 2

        self.A34 = 1

        self.A42 = - self.m1s * self.h1s * v_x1
        self.A43 = (
                self.k_varphi1 + self.k12 - self.m1s * self.gravity * self.h1s - self.I1yy * gamma_1 ** 2 + self.I1zz * gamma_1 ** 2)
        self.A44 = self.c_varphi1
        self.A45 = (self.m2 * self.hh) * gamma_2
        self.A46 = self.hh * self.k3 / v_x2
        self.A47 = -self.d * self.k3 * self.hh / v_x2
        self.A48 = -(self.k12 - self.m2s * self.h2s * self.hh * gamma_2 ** 2)

        self.A50 = -(gamma_2 - gamma_1) * np.sin(phi_2 - phi_1)
        self.A51 = -(gamma_2 - gamma_1) * np.cos(phi_2 - phi_1)
        self.A52 = (gamma_2 - gamma_1) * np.cos(phi_2 - phi_1) * self.c

        self.A60 = (gamma_2 - gamma_1) * np.cos(phi_2 - phi_1)
        self.A61 = -(gamma_2 - gamma_1) * np.sin(phi_2 - phi_1)
        self.A62 = self.c * (gamma_2 - gamma_1) * np.sin(phi_2 - phi_1)

        self.A75 = self.e * self.m2 * gamma_2
        self.A76 = -(self.d + self.e) * self.k3 / v_x2
        self.A77 = (self.d + self.e) * self.k3 * self.d / v_x2
        self.A78 = (self.e * self.m2s * self.h2s * gamma_2 ** 2 - self.m2s * self.h2s * gamma_2 * v_y2)

        self.A89 = 1

        self.A93, self.A95, self.A96, self.A97, self.A98, self.A99 = self.k12, -(
                    self.m2s * self.h2s * gamma_2 + self.hh * self.m2 * gamma_2), \
                                                                     -self.hh * self.k3 / v_x2, self.d * self.hh * self.k3 / v_x2, -(
                self.m2s * self.gravity * self.h2s - self.k_varphi2 - self.k12 + self.I2yy * gamma_2 ** 2 - self.I2zz * gamma_2 ** 2 - self.m2s * self.h2s * self.hh * gamma_2 ** 2), self.c_varphi2

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

        B_matrix = np.zeros((self.state_dim - 6, 2))
        self.B00 = - (self.m1 + self.m2)
        self.B11 = -self.k1
        self.B21 = -self.a * self.k1
        B_matrix[0, 0] = self.B00
        B_matrix[1, 1] = self.B11
        B_matrix[2, 1] = self.B21

        X_dot = (np.matmul(np.matmul(np.linalg.inv(M_matrix), A_matrix), X) + np.matmul(
            np.linalg.inv(M_matrix), np.matmul(B_matrix, actions))).squeeze()

        state_next[0] = x_1 + delta_t * (v_x1 * np.cos(phi_1) - v_y1 * np.sin(phi_1))
        state_next[1] = y_1 + delta_t * (v_y1 * np.cos(phi_1) + v_x1 * np.sin(phi_1))
        state_next[2] = phi_1 + delta_t * gamma_1
        state_next[3] = v_x1 + delta_t * X_dot[0]

        state_next[4] = x_2 + delta_t * (v_x2 * np.cos(-phi_2) - v_y2 * np.sin(
            -phi_2))  # state_next[12] - self.b * np.cos(phi_1) - self.e * np.cos(phi_2)  # posx_trailer
        state_next[5] = y_2 - delta_t * (v_y2 * np.cos(-phi_2) + v_x2 * np.sin(-phi_2))  # posy_trailer
        state_next[6] = phi_2 + delta_t * gamma_2
        state_next[7] = v_x2 + delta_t * X_dot[5]

        state_next[8:12] = states[8:12] + delta_t * X_dot[1:5]
        state_next[12:] = states[12:] + delta_t * X_dot[6:]
        return state_next

    def f_xu_multi(self, states, actions, delta_t):
        x_1, y_1, phi_1, v_x1, x_2, y_2, phi_2, v_x2, \
        v_y1, gamma_1, varphi_1, varphi_1dot, \
        v_y2, gamma_2, varphi_2, varphi_2dot = states
        Q1, delta1, Q2, delta2, Q3, delta3, Q4, delta4 = actions

        D = np.array([Q1, delta1, Q2, delta2, Q3, delta3, Q4, delta4]).reshape(8, 1)
        X = np.array([states[3], states[8:12], states[7], states[12:self.state_dim]]).reshape(5, 1)

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

        Lc_matrix[1, 1], Lc_matrix[1, 3], Lc_matrix[1, 5], Lc_matrix[1, 7] = 1, 1, 1, 1

        Lc_matrix[2, 0], Lc_matrix[2, 1], Lc_matrix[2, 2], Lc_matrix[2, 3], \
        Lc_matrix[2, 4], Lc_matrix[2, 5], Lc_matrix[2, 6], Lc_matrix[2, 7] \
            = -self.lw / 2, self.lf, self.lw / 2, self.lf, \
              -self.lw / 2, -self.lr, self.lw / 2, -self.lr

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

        Ew = [1, 1, 1, 1, 1, 1, 1, 1]  # delta action bool matrix[0, 0, 0, 0, 0, 0, 0, 0]
        Ew_matrix = np.diag(Ew)

        A1_matrix = np.zeros((8, 5))

        A1_matrix[1, 1], A1_matrix[1, 2] = -self.k_alpha1 / v_x, -self.k_alpha1 * self.lf / v_x

        A1_matrix[3, 1], A1_matrix[3, 2] = -self.k_alpha2 / v_x, -self.k_alpha2 * self.lf / v_x

        A1_matrix[5, 1], A1_matrix[5, 2] = -self.k_alpha3 / v_x, -self.k_alpha3 * (-self.lr) / v_x

        A1_matrix[7, 1], A1_matrix[7, 2] = -self.k_alpha4 / v_x, -self.k_alpha4 * (-self.lr) / v_x

        B1 = [1 / self.Rw, self.k_alpha1, 1 / self.Rw, self.k_alpha2,
              1 / self.Rw, self.k_alpha3, 1 / self.Rw, self.k_alpha4]

        B1_matrix = np.diag(B1)
        dt_matrix = np.zeros((8, 1))
        dt_matrix[0], dt_matrix[2], dt_matrix[4], dt_matrix[
            6] = -1 / 4 * self.m * self.g * self.mu_r, -1 / 4 * self.m * self.g * self.mu_r, -1 / 4 * self.m * self.g * self.mu_r, -1 / 4 * self.m * self.g * self.mu_r
        temp = np.matmul(A1_matrix, X) + np.matmul(B1_matrix, D) + np.matmul(np.matmul(Ew_matrix, B1_matrix),
                                                                             U) + dt_matrix

        X_dot = (np.matmul(A_matrix, X) + np.matmul(np.matmul(np.matmul(np.matmul(
            B_matrix, Lc_matrix), Ec_matrix), Mw_matrix), temp)).squeeze()

        state_next[0] = x + delta_t * (v_x * np.cos(phi) - v_y * np.sin(phi))
        state_next[1] = y + delta_t * (v_y * np.cos(phi) + v_x * np.sin(phi))
        state_next[2] = phi + delta_t * gamma
        state_next[3:8] = states[3:8] + delta_t * X_dot

        # state_next[8] = kappa1+delta_t*(self.Rw*(Q1-self.Rw*self.C_slip1*kappa1)/(v_x*self.Iw)-(1+kappa1)/(self.m*v_x)*(self.C_slip1*kappa1+self.C_slip2*kappa2+self.C_slip3*kappa3+self.C_slip4*kappa4))
        # state_next[9] = kappa2+delta_t*(self.Rw*(Q2-self.Rw*self.C_slip2*kappa2)/(v_x*self.Iw)-(1+kappa2)/(self.m*v_x)*(self.C_slip1*kappa1+self.C_slip2*kappa2+self.C_slip3*kappa3+self.C_slip4*kappa4))
        # state_next[10] = kappa3+delta_t*(self.Rw*(Q3-self.Rw*self.C_slip3*kappa3)/(v_x*self.Iw)-(1+kappa3)/(self.m*v_x)*(self.C_slip1*kappa1+self.C_slip2*kappa2+self.C_slip3*kappa3+self.C_slip4*kappa4))
        # state_next[11] = kappa4+delta_t*(self.Rw*(Q4-self.Rw*self.C_slip4*kappa4)/(v_x*self.Iw)-(1+kappa4)/(self.m*v_x)*(self.C_slip1*kappa1+self.C_slip2*kappa2+self.C_slip3*kappa3+self.C_slip4*kappa4))
        return state_next

class Semitruck8dof(PythBaseEnv):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    def __init__(
        self,
        pre_horizon: int = 20,
        max_steer: float = 0.5,
        max_accel: float = 1.5,
        min_accel: float = -1.5,
        target_speed: float = 20.0,
        **kwargs,
    ):
        work_space = kwargs.pop("work_space", None)
        if work_space is None:
            # initial range of [x_1, y_1, phi_1, v_x1, x_2, y_2, phi_2, v_x2,
            # v_y1, gamma_1, varphi_1, varphi_1dot, v_y2, gamma_2, varphi_2, varphi_2dot]
            # 用高斯分布去采样
            init_high = np.array([200, 1, 0.1, 22, 200, 1, 22, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                  0.1, 0.1], dtype=np.float32)
            init_low = np.array([0, -1, -0.1, 18, 0,  -1, -0.1, 18,  -0.1, -0.1, -0.1, -0.1, -0.1, -0.1,
                                  -0.1, -0.1], dtype=np.float32)
            work_space = np.stack((init_low, init_high))
        super(Semitruck8dof, self).__init__(work_space=work_space, **kwargs)
        self.dt = 0.01
        self.target_speed = target_speed
        self.pre_horizon = pre_horizon
        self.vehicle_dynamics = VehicleDynamicsData()
        self.ref_traj = Ref_Route(self.target_speed)
        self.state_dim = 16
        ego_obs_dim = 14
        ref_obs_dim = 6
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf] * (ego_obs_dim + ref_obs_dim * pre_horizon)),
            high=np.array([np.inf] * (ego_obs_dim + ref_obs_dim * pre_horizon)),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.array([min_accel, -max_steer]),
            high=np.array([max_accel, max_steer]),
            dtype=np.float32,
        )
        obs_scale_default = [1/100, 1/100, 1/10,1/100,
                             1/100, 1/100, 1/10,1/100,
                             1/100, 1, 1, 1,
                             1/100,1, 1, 1]
        self.obs_scale = np.array(kwargs.get('obs_scale', obs_scale_default))


        self.max_episode_steps = 1500

        self.state = None
        self.ref_x = None
        self.ref_y = None
        self.ref_x2 = None
        self.ref_y2 = None
        self.ref_points = None

        self.info_dict = {
            "state": {"shape": (self.state_dim,), "dtype": np.float32},
            "ref_points": {"shape": (self.pre_horizon + 1, 8), "dtype": np.float32},
            "ref_x": {"shape": (), "dtype": np.float32},
            "ref_y": {"shape": (), "dtype": np.float32},
            "ref_x2": {"shape": (), "dtype": np.float32},
            "ref_y2": {"shape": (), "dtype": np.float32},
            "ref": {"shape": (8,), "dtype": np.float32},
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
        #[x_1, y_1, phi_1, v_x1, x_2, y_2, phi_2, v_x2,
         # v_y1, gamma_1, varphi_1, varphi_1dot, v_y2, gamma_2, varphi_2, varphi_2dot]

        # 训练用，run的时候注释掉
        state[0] = self.np_random.uniform(
            low=self.init_space[0][0], high=self.init_space[1][0]
        )

        state[3] = self.np_random.uniform(
            low=self.init_space[0][3], high=self.init_space[1][3]
        )

        state[0] = 10
        state[3] = 19

        state[4] = state[0] - self.vehicle_dynamics.b * np.cos(state[2]) - self.vehicle_dynamics.e * np.cos(
            state[6])  # posx_trailer

        state[5] = state[1] - self.vehicle_dynamics.b * np.sin(state[2]) - self.vehicle_dynamics.e * np.sin(
            state[6])  # posy_trailer
        state[7] = state[3]*np.cos(state[6]-state[2])-(state[8]-self.vehicle_dynamics.c*state[9])*np.sin(state[6]-state[2])

        self.state = state
        self.ref_x, self.ref_y = state[0], state[1]
        traj_points = [[self.ref_x, self.ref_y]]
        for k in range(self.pre_horizon):
            self.ref_x += state[3] * self.dt
            self.ref_y += state[8] * self.dt
            traj_points.append([self.ref_x, self.ref_y])
        
        self.ref_x2, self.ref_y2 = state[4], state[5]
        traj_points_2 = [[self.ref_x2, self.ref_y2]]
        for k in range(self.pre_horizon):
            self.ref_x2 += state[7] * self.dt
            self.ref_y2 += state[12] * self.dt
            traj_points_2.append([self.ref_x2, self.ref_y2])
        self.ref_points = np.hstack((self.ref_traj.find_nearest_point(np.array(traj_points)), self.ref_traj.find_nearest_point(np.array(traj_points_2))))  # x, y, phi, u,# x, y, phi, u

        obs = self.get_obs()
        self.action_last = np.zeros(2)
        return obs, self.info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        action = np.clip(action, self.action_space.low, self.action_space.high)

        reward = self.compute_reward(action)

        self.state = self.vehicle_dynamics.f_xu(self.state, action, self.dt)

        self.ref_x, self.ref_y = self.state[self.state_dim - 2], self.state[self.state_dim - 4]

        self.ref_points[:-1] = self.ref_points[1:]
        self.ref_x += self.state[3] * self.dt
        self.ref_y += self.state[8] * self.dt
        traj_points = [[self.ref_x, self.ref_y]]
        new_ref_point = self.ref_traj.find_nearest_point(np.array(traj_points))  # x, y, phi, u

        self.ref_x2 += self.state[7] * self.dt
        self.ref_y2 += self.state[12] * self.dt
        traj_points_2 = [[self.ref_x2, self.ref_y2]]
        new_ref_point_2 = self.ref_traj.find_nearest_point(np.array(traj_points_2))  # x, y, phi, u
        self.ref_points[-1] = np.hstack((new_ref_point, new_ref_point_2))

        obs = self.get_obs()
        self.done = self.judge_done()

        self.action_last = action
        return obs, reward, self.done, self.info

    def get_obs(self) -> np.ndarray:
        ref_x_tf, ref_y_tf, ref_phi_tf, ref_vx_tf = \
            state_error_calculate(
                self.state[0], self.state[1], self.state[2], self.state[3],
                self.ref_points[:, 0], self.ref_points[:, 1], self.ref_points[:, 2],self.ref_points[:, 3]
            )

        ref_x2_tf, ref_y2_tf, ref_phi2_tf, ref_vx2_tf = \
            state_error_calculate(
                self.state[4], self.state[5], self.state[6], self.state[7],
                self.ref_points[:, 4], self.ref_points[:, 5], self.ref_points[:, 6],self.ref_points[:, 7]
            )

        # ego_obs: [
        # delta_y, delta_phi,delta_vx, delta_y2, delta_phi2 delta_vx2, (of the first reference point)
        # v, w, varphi (of ego vehicle, including tractor and trailer)
        # ]

        ego_obs = np.concatenate(
            ([ref_y_tf[0]*self.obs_scale[1], ref_phi_tf[0]*self.obs_scale[2],
              ref_vx_tf[0]*self.obs_scale[3], ref_y2_tf[0]*self.obs_scale[5],
              ref_phi2_tf[0]*self.obs_scale[6], ref_vx2_tf[0]*self.obs_scale[7]],
             [self.state[8]*self.obs_scale[8]], self.state[9:12], [self.state[12]*self.obs_scale[12]], self.state[13:]))


        # ref_obs: [
        # delta_x, delta_y, delta_psi (of the second to last reference point)
        # ]
        ref_obs = np.stack((ref_y_tf*self.obs_scale[1], ref_phi_tf*self.obs_scale[2],
              ref_vx_tf*self.obs_scale[3], ref_y2_tf*self.obs_scale[5],
              ref_phi2_tf*self.obs_scale[6], ref_vx2_tf*self.obs_scale[7]), 1)[1:].flatten()
        return np.concatenate((ego_obs, ref_obs))

    def compute_reward(self, action: np.ndarray) -> float:
        p_x1, p_y1, phi_1, v_x1, p_x2, p_y2, phi_2, v_x2, \
        v_y1, gamma_1, varphi_1, varphi_1dot, \
        v_y2, gamma_2, varphi_2, varphi_2dot = self.state

        ref_x, ref_y, ref_psi, ref_vx = self.ref_points[0][0:4]
        accel, steer = action[0], action[1]
        return -(
            1 * ((p_x1 - ref_x) ** 2 + 0.04 * (p_y1 - ref_y) ** 2)
            + 1 * (v_x1 - ref_vx) ** 2
            + 0.9 * v_y1 ** 2
            + 0.8 * angle_normalize(phi_1 - ref_psi) ** 2
            + 0.5 * gamma_1 ** 2
            + 0.5 * varphi_1 ** 2
            + 0.5 * varphi_1dot ** 2
            + 0.4 * steer ** 2
            + 0.4 * accel ** 2
            + 2.0 * (accel - self.action_last[0]) ** 2
            + 2.0 * (steer - self.action_last[1]) ** 2
        )

    def judge_done(self) -> bool:
        done = ((abs(self.state[1]-self.ref_points[0, 1]) > 3)  # delta_y1
                +(abs(self.state[3] - self.ref_points[0, 3]) > 2)  # delta_vx1
                + (abs(self.state[8]) > 2)  # delta_vy1
                  + (abs(self.state[2]-self.ref_points[0, 2]) > np.pi/2)  # delta_phi1
                  + (abs(self.state[5]-self.ref_points[0, 5]) > 3) # delta_y2
                + (abs(self.state[7] - self.ref_points[0, 7]) > 2)  # delta_vx2
                  + (abs(self.state[6]-self.ref_points[0, 6]) > np.pi / 2))  # delta_phi2
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
        self.Torque_throttle = np.array(self.Torque_trucksim.iloc[0, 1:].dropna())  # 表的X坐标是throttle
        self.Torque_engspd = np.array(self.Torque_trucksim.iloc[1:, 0])  # 表的Y坐标是engspd
        self.Torque = np.array(self.Torque_trucksim.iloc[1:, 1:])  #

    def Calc_EngSpd(self, u1_k, ig):
        EngSpd_Lower = 400
        EngSpd_Upper = 2200
        i0 = 4.4
        rw = 0.51
        EngSpd = float(u1_k) * ig * i0 * 60 / (2 * rw * np.pi)
        if EngSpd > EngSpd_Upper:
            EngSpd = EngSpd_Upper
        if EngSpd < EngSpd_Lower:
            EngSpd = EngSpd_Lower
        TransSpd = float(u1_k) * 60 / (2 * rw * np.pi) * i0
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
        index_rpm = np.argwhere(EngSpd >= self.Torque_engspd)[-1][0]
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
        EngTorque = (mass * accel + 0.5 * Cd * A * rho * np.power(u1_k, 2) + mass * g * (
                f * np.cos(theta) + np.sin(theta))) / (ig * i0 * etaT / (rw))

        return EngTorque

    def Threshold_LU(self, gear, Throttle):
        gear = int(gear)
        if gear == 1:  # 1-2
            up = np.interp(Throttle, [0, 0.2, 0.8, 1], [98, 98, 190, 190])
            down = np.interp(Throttle, [0, 0.2, 0.8, 1], [60, 60, 110, 110])
        elif gear == 2:
            up = np.interp(Throttle, [0, 0.2, 0.8, 1], [132, 132, 250, 250])
            down = np.interp(Throttle, [0, 0.2, 0.8, 1], [119, 119, 190, 190])
        elif gear == 3:
            up = np.interp(Throttle, [0, 0.2, 0.8, 1], [178, 178, 340, 340])
            down = np.interp(Throttle, [0, 0.2, 0.8, 1], [160, 160, 277, 277])
        elif gear == 4:
            up = np.interp(Throttle, [0, 0.2, 0.8, 1], [241, 241, 460, 460])
            down = np.interp(Throttle, [0, 0.2, 0.8, 1], [217, 217, 374, 374])
        elif gear == 5:
            up = np.interp(Throttle, [0, 0.2, 0.8, 1], [325, 325, 625, 625])
            down = np.interp(Throttle, [0, 0.2, 0.8, 1], [293, 293, 506, 506])
        elif gear == 6:
            up = np.interp(Throttle, [0, 0.2, 0.8, 1], [440, 440, 850, 850])
            down = np.interp(Throttle, [0, 0.2, 0.8, 1], [396, 396, 683, 683])
        elif gear == 7:
            up = np.interp(Throttle, [0, 0.2, 0.8, 1], [593, 593, 1125, 1125])
            down = np.interp(Throttle, [0, 0.2, 0.8, 1], [533, 533, 923, 923])
        elif gear == 8:
            up = np.interp(Throttle, [0, 0.2, 0.8, 1], [800, 800, 1525, 1525])
            down = np.interp(Throttle, [0, 0.7, 0.9, 1], [720, 720, 1244, 1244])
        elif gear == 9:
            up = np.interp(Throttle, [0, 0.2, 0.8, 1], [229, 229, 563, 563])
            down = np.interp(Throttle, [0, 0.75, 0.9, 1], [206, 206, 506, 506])
        else:
            up = np.interp(Throttle, [0, 0.2, 0.8, 1], [1081, 1081, 2050, 2050])
            down = np.interp(Throttle, [0, 0.7, 0.9, 1], [973, 973, 1400, 1400])
        return up, down

    def Gear_Ratio(self, gear):
        ig_dict = {'1': 11.06, '2': 8.2, '3': 6.06, '4': 4.49, '5': 3.32, '6': 2.46,
                   '7': 1.82, '8': 1.35, '9': 1, '10': 0.74}
        gear_ratio = ig_dict[str(int(gear))]
        return gear_ratio

    def Shift_Logic(self, gear_input, AT_Speed, up_th, down_th):
        gear_output = gear_input
        if gear_input == 1:
            if AT_Speed < down_th:
                gear_output = 1
            elif down_th >= AT_Speed and AT_Speed < up_th:
                gear_output = 1
            elif AT_Speed >= up_th:
                gear_output = 2

        elif gear_input == 2:
            if AT_Speed < down_th:
                gear_output = 1
            elif down_th >= AT_Speed and AT_Speed < up_th:
                gear_output = 2
            elif AT_Speed >= up_th:
                gear_output = 3

        elif gear_input == 3:
            if AT_Speed < down_th:
                gear_output = 2
            elif down_th >= AT_Speed and AT_Speed < up_th:
                gear_output = 3
                # return gear
            elif AT_Speed >= up_th:
                gear_output = 4

        elif gear_input == 4:
            if AT_Speed < down_th:
                gear_output = 3
            elif down_th >= AT_Speed and AT_Speed < up_th:
                gear_output = 4
            elif AT_Speed >= up_th:
                gear_output = 5

        elif gear_input == 5:
            if AT_Speed < down_th:
                gear_output = 4
            elif down_th >= AT_Speed and AT_Speed < up_th:
                gear_output = 5
            elif AT_Speed >= up_th:
                gear_output = 6
        elif gear_input == 6:
            if AT_Speed < down_th:
                gear_output = 5
            elif down_th >= AT_Speed and AT_Speed < up_th:
                gear_output = 6
            elif AT_Speed >= up_th:
                gear_output = 7
        elif gear_input == 7:
            if AT_Speed < down_th:
                gear_output = 6
            elif down_th >= AT_Speed and AT_Speed < up_th:
                gear_output = 7
            elif AT_Speed >= up_th:
                gear_output = 8
        elif gear_input == 8:
            if AT_Speed < down_th:
                gear_output = 7
            elif down_th >= AT_Speed and AT_Speed < up_th:
                gear_output = 8
            elif AT_Speed >= up_th:
                gear_output = 9
        elif gear_input == 9:
            if AT_Speed < down_th:
                gear_output = 8
            elif down_th >= AT_Speed and AT_Speed < up_th:
                gear_output = 9
            elif AT_Speed >= up_th:
                gear_output = 10
        # elif gear_input == 10:
        #     if AT_Speed < down_th:
        #         gear_output = 9
        #     elif down_th >= AT_Speed and AT_Speed < up_th:
        #         gear_output = 10
        #     elif AT_Speed >= up_th:
        #         gear_output = 11
        # elif gear_input == 11:
        #     if AT_Speed < down_th:
        #         gear_output = 10
        #     elif down_th >= AT_Speed and AT_Speed < up_th:
        #         gear_output = 11
        #     elif AT_Speed >= up_th:
        #         gear_output = 12
        # elif gear_input == 12:
        #     if AT_Speed < down_th:
        #         gear_output = 11
        #     elif down_th >= AT_Speed and AT_Speed < up_th:
        #         gear_output = 12
        #     elif AT_Speed >= up_th:
        #         gear_output = 13
        # elif gear_input == 13:
        #     if AT_Speed < down_th:
        #         gear_output = 12
        #     elif down_th >= AT_Speed and AT_Speed < up_th:
        #         gear_output = 13
        #     elif AT_Speed >= up_th:
        #         gear_output = 14
        # elif gear_input == 14:
        #     if AT_Speed < down_th:
        #         gear_output = 13
        #     elif down_th >= AT_Speed and AT_Speed < up_th:
        #         gear_output = 14
        #     elif AT_Speed >= up_th:
        #         gear_output = 15
        # elif gear_input == 15:
        #     if AT_Speed < down_th:
        #         gear_output = 14
        #     elif down_th >= AT_Speed and AT_Speed < up_th:
        #         gear_output = 15
        #     elif AT_Speed >= up_th:
        #         gear_output = 16
        # elif gear_input == 16:
        #     if AT_Speed < down_th:
        #         gear_output = 15
        #     elif down_th >= AT_Speed and AT_Speed < up_th:
        #         gear_output = 16
        #     elif AT_Speed >= up_th:
        #         gear_output = 17
        # elif gear_input == 17:
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
    make env `pyth_semitruck8dof`
    """
    return Semitruck8dof(**kwargs)
