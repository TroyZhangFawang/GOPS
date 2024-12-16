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
from gops.env.env_ocp.pyth_veh3dofconti import angle_normalize, VehicleDynamicsData
from gops.env.env_ocp.resources.ref_traj_model import MultiRefTrajModel
from gops.utils.gops_typing import InfoDict


class VehicleDynamicsModel(VehicleDynamicsData):
    def __init__(self):

        self.dt = 0.01
        self.state_dim = 16
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

    def f_xu(self, states, actions, delta_t):
        self.batch_size = len(states[:, 0])
        state_next = torch.zeros_like(states)
        X_matrix = torch.cat((states[:, 7:15], states[:, 2:3], states[:, 6:7],
                              states[:, 15:], states[:, 1:2], states[:, 5:6]), 1)


        X_dot_batch = torch.zeros((self.batch_size, self.state_dim - 3))
        for batch in range(self.batch_size):
            vx = states[batch, 3]
            M11 = self.m1 * vx * self.c
            M12 = self.I1zz
            M13 = -self.m1s * self.h1c * self.c - self.I1xz

            M21 = self.m1 * vx * self.h1c - self.m1s * self.h1 * vx
            M22 = -self.I1xz
            M24 = self.I1xx + 2 * self.m1s * self.h1 * self.h1 - self.m1s * self.h1 * self.h1c

            M31 = self.m1 * vx
            M34 = -self.m1s * self.h1
            M35 = self.m2 * vx
            M38 = -self.m2s * self.h2

            M45 = self.m2 * vx * self.e
            M46 = -self.I2zz
            M48 = self.I2xz - self.m2s * self.h2 * self.e

            M55 = self.m2 * vx * self.h2c - self.m2s * self.h2 * vx
            M56 = -self.I2xz
            M58 = self.I2xx + 2 * self.m2s * self.h2 * self.h2 - self.m2s * self.h2 * self.h2c

            M61 = 1
            M62 = -self.c / vx
            M64 = -self.h1c / vx
            M65 = -1
            M66 = -self.e / vx
            M68 = self.h2c / vx

            M73 = 1

            M87 = 1

            M99 = 1
            M1010 = 1
            M111 = -vx
            M1111 = 1
            M1212 = 1
            M1313 = 1

            A11 = (self.c + self.a) * self.kf + (self.c - self.b) * self.km
            A12 = self.a * (self.c + self.a) * self.kf / vx - self.b * (
                    self.c - self.b) * self.km / vx - self.m1 * vx * self.c
            A21 = (self.kf + self.km) * self.h1c
            A22 = (self.a * self.kf - self.b * self.km) * self.h1c / vx + (
                    self.m1s * self.h1 - self.m1 * self.h1c) * vx
            A23 = self.m1s * self.gravity * self.h1 - self.kr1 - self.ka
            A24 = -self.c1
            A27 = self.ka

            A31 = self.kf + self.km
            A32 = (self.a * self.kf - self.b * self.km) / vx - self.m1 * vx
            A35 = self.kr
            A36 = -self.d * self.kr / vx - self.m2 * vx

            A45 = (self.e + self.d) * self.kr
            A46 = -self.d * (self.e + self.d) * self.kr / vx - self.m2 * vx * self.e
            A53 = self.ka
            A55 = self.kr * self.h2c
            A56 = (self.m2s * self.h2 - self.m2 * self.h2c) * vx - self.d * self.kr * self.h2c / vx
            A57 = self.m2s * self.gravity * self.h2 - self.kr2 - self.ka
            A58 = -self.c2
            A62 = -1
            A66 = 1
            A74 = 1
            A88, A92, A106 = 1, 1, 1
            A121, A129, A135, A1310 = vx, vx, vx, vx

            B11 = - (self.c + self.a) * self.kf
            B21 = -self.kf * self.h1c
            B31 = -self.kf

            M_matrix = torch.zeros((self.state_dim - 3, self.state_dim - 3))
            M_matrix[0, 0] = M11
            M_matrix[0, 1] = M12
            M_matrix[0, 2] = M13

            M_matrix[1, 0], M_matrix[1, 1], M_matrix[1, 3] = M21, M22, M24

            M_matrix[2, 0], M_matrix[2, 3], M_matrix[2, 4], M_matrix[2, 7] = \
                M31, M34, M35, M38

            M_matrix[3, 4], M_matrix[3, 5], M_matrix[3, 7], = M45, M46, M48

            M_matrix[4, 4], M_matrix[4, 5], M_matrix[4, 7] = M55, M56, M58

            M_matrix[5, 0], M_matrix[5, 1], M_matrix[5, 3], M_matrix[5, 4], M_matrix[5, 5], M_matrix[5, 7] = \
                M61, M62, M64, M65, M66, M68

            M_matrix[6, 2] = M73
            M_matrix[7, 6] = M87
            M_matrix[8, 8] = M99
            M_matrix[9, 9] = M1010
            M_matrix[10, 0], M_matrix[10, 10] = M111, M1111
            M_matrix[11, 11] = M1212
            M_matrix[12, 12] = M1313

            A_matrix = torch.zeros((self.state_dim - 3, self.state_dim - 3))
            A_matrix[0, 0], A_matrix[0, 1], A_matrix[1, 0], A_matrix[1, 1], A_matrix[1, 2], A_matrix[1, 3], A_matrix[
                1, 6] = \
                A11, A12, A21, A22, A23, A24, A27
            A_matrix[2, 0], A_matrix[2, 1], A_matrix[2, 4], A_matrix[2, 5] \
                = A31, A32, A35, A36

            A_matrix[3, 4], A_matrix[3, 5] = A45, A46
            A_matrix[4, 2], A_matrix[4, 4], A_matrix[4, 5], A_matrix[4, 6], A_matrix[4, 7] = \
                A53, A55, A56, A57, A58
            A_matrix[5, 1], A_matrix[5, 5], A_matrix[6, 3], A_matrix[7, 7], \
                A_matrix[8, 1], A_matrix[9, 5], A_matrix[11, 0], A_matrix[11, 8], \
                A_matrix[12, 4], A_matrix[12, 9] = A62, A66, A74, A88, \
                A92, A106, A121, A129, \
                A135, A1310

            B_matrix = torch.zeros((self.state_dim - 3, 1))
            B_matrix[0, 0] = B11
            B_matrix[1, 0] = B21
            B_matrix[2, 0] = B31

            X_dot = (torch.matmul(torch.matmul(torch.inverse(M_matrix), A_matrix),
                                  X_matrix[batch, :]) +
                     torch.matmul(torch.inverse(M_matrix), torch.matmul(B_matrix, actions[batch, 0:1]))).squeeze()
            X_dot_batch[batch, :] = X_dot

        state_next[:, 0] = states[:, 0] + delta_t * (vx * torch.cos(states[:, 2].clone()) - states[:, 15].clone() * torch.sin(states[:, 2].clone()))  # x_tractor
        state_next[:, 1] = states[:, 1] + delta_t * (vx * torch.sin(states[:, 2].clone()) + states[:, 15].clone() * torch.cos(states[:, 2].clone()))#X_dot_batch[:, 11]
        state_next[:, 2] = states[:, 2] + delta_t * states[:, 8]
        state_next[:, 3] = states[:, 3] + delta_t * actions[:, 1]
        state_next[:, 4] = state_next[:, 0].clone() - self.b * torch.cos(states[:, 2].clone()) - self.e * torch.cos(states[:, 6].clone())  # posx_trailer
        state_next[:, 5] = state_next[:, 1].clone() - self.b * torch.sin(states[:, 2].clone()) - self.e * torch.sin(states[:, 6].clone())  # posy_trailer
        state_next[:, 6] = states[:, 6] + delta_t * states[:, 12]
        state_next[:, 7:15] = states[:, 7:15] + delta_t * X_dot_batch[:, 0:8]
        state_next[:, 15] = states[:, 15] + delta_t * X_dot_batch[:, 10]
        return state_next

    def auxvar_setting(self, delta_t=0.01, ref_traj_init=np.array([0, 0, 0, 0]), action_last_np=np.zeros((2,))):
        parameter = []
        self.wdis = SX.sym('wdis')
        parameter += [self.wdis]
        self.wvx1 = SX.sym('wvx1')
        parameter += [self.wvx1]
        self.wpsi1 = SX.sym('wpsi1')
        parameter += [self.wpsi1]
        self.womega1 = SX.sym('womega1')
        parameter += [self.womega1]
        self.wbeta1 = SX.sym('wbeta1')
        parameter += [self.wbeta1]
        self.wphi1 = SX.sym('wphi1')
        parameter += [self.wphi1]
        self.wphi1dot = SX.sym('wphi1dot')
        parameter += [self.wphi1dot]
        self.wu = SX.sym('wu')
        parameter += [self.wu]
        self.wudot = SX.sym('wudot')
        parameter += [self.wudot]
        self.cost_auxvar = vcat(parameter)

        self.x1, self.y1, self.psi1,self.u1, self.x2,self.y2, self.psi2, self.beta1, self.psi1dot, self.phi1, self.phi1dot, \
            self.beta2, self.psi2dot, self.phi2, self.phi2dot, \
            self.v1  = \
            SX.sym('x1'), SX.sym('y1'), SX.sym('psi1'), SX.sym('u1'), SX.sym('x2'),SX.sym('y2'), SX.sym('psi2'), \
            SX.sym('beta1'), SX.sym('psi1dot'), SX.sym('phi1'), SX.sym('phi1dot'), \
                SX.sym('beta2'), SX.sym('psi2dot'), SX.sym('phi2'), SX.sym('phi2dot'), SX.sym('v1')

        self.X = vertcat(self.x1, self.y1, self.psi1, self.u1, self.x2,self.y2, self.psi2,
                         self.beta1, self.psi1dot, self.phi1, self.phi1dot,
                         self.beta2, self.psi2dot, self.phi2, self.phi2dot,
                         self.v1)
        X_matrix = vertcat(self.beta1, self.psi1dot, self.phi1, self.phi1dot,
                         self.beta2, self.psi2dot, self.phi2, self.phi2dot,
                           self.psi1, self.psi2, self.v1, self.y1, self.y2)
        self.ax = SX.sym('ax')
        self.steering = SX.sym('str')
        self.U = vertcat(self.steering, self.ax)
        vx = self.X[3]
        M11 = self.m1 * vx * self.c
        M12 = self.I1zz
        M13 = -self.m1s * self.h1c * self.c - self.I1xz

        M21 = self.m1 * vx * self.h1c - self.m1s * self.h1 * vx
        M22 = -self.I1xz
        M24 = self.I1xx + 2 * self.m1s * self.h1 * self.h1 - self.m1s * self.h1 * self.h1c

        M31 = self.m1 * vx
        M34 = -self.m1s * self.h1
        M35 = self.m2 * vx
        M38 = -self.m2s * self.h2

        M45 = self.m2 * vx * self.e
        M46 = -self.I2zz
        M48 = self.I2xz - self.m2s * self.h2 * self.e

        M55 = self.m2 * vx * self.h2c - self.m2s * self.h2 * vx
        M56 = -self.I2xz
        M58 = self.I2xx + 2 * self.m2s * self.h2 * self.h2 - self.m2s * self.h2 * self.h2c

        M61 = 1
        M62 = -self.c / vx
        M64 = -self.h1c / vx
        M65 = -1
        M66 = -self.e / vx
        M68 = self.h2c / vx

        M73 = 1

        M87 = 1

        M99 = 1
        M1010 = 1
        M111 = -vx
        M1111 = 1
        M1212 = 1
        M1313 = 1

        A11 = (self.c + self.a) * self.kf + (self.c - self.b) * self.km
        A12 = self.a * (self.c + self.a) * self.kf / vx - self.b * (
                self.c - self.b) * self.km / vx - self.m1 * vx * self.c
        A21 = (self.kf + self.km) * self.h1c
        A22 = (self.a * self.kf - self.b * self.km) * self.h1c / vx + (
                self.m1s * self.h1 - self.m1 * self.h1c) * vx
        A23 = self.m1s * self.gravity * self.h1 - self.kr1 - self.ka
        A24 = -self.c1
        A27 = self.ka

        A31 = self.kf + self.km
        A32 = (self.a * self.kf - self.b * self.km) / vx - self.m1 * vx
        A35 = self.kr
        A36 = -self.d * self.kr / vx - self.m2 * vx

        A45 = (self.e + self.d) * self.kr
        A46 = -self.d * (self.e + self.d) * self.kr / vx - self.m2 * vx * self.e
        A53 = self.ka
        A55 = self.kr * self.h2c
        A56 = (self.m2s * self.h2 - self.m2 * self.h2c) * vx - self.d * self.kr * self.h2c / vx
        A57 = self.m2s * self.gravity * self.h2 - self.kr2 - self.ka
        A58 = -self.c2
        A62 = -1
        A66 = 1
        A74 = 1
        A88, A92, A106 = 1, 1, 1
        A121, A129, A135, A1310 = vx, vx, vx, vx

        B11 = - (self.c + self.a) * self.kf
        B21 = -self.kf * self.h1c
        B31 = -self.kf

        M_matrix = SX.zeros((self.state_dim - 3, self.state_dim - 3))
        M_matrix[0, 0], M_matrix[0, 1], M_matrix[0, 2] = M11, M12, M13

        M_matrix[1, 0], M_matrix[1, 1], M_matrix[1, 3] = M21, M22, M24

        M_matrix[2, 0], M_matrix[2, 3], M_matrix[2, 4], M_matrix[2, 7] = \
            M31, M34, M35, M38

        M_matrix[3, 4], M_matrix[3, 5], M_matrix[3, 7], = M45, M46, M48

        M_matrix[4, 4], M_matrix[4, 5], M_matrix[4, 7] = M55, M56, M58

        M_matrix[5, 0], M_matrix[5, 1], M_matrix[5, 3], M_matrix[5, 4], M_matrix[5, 5], M_matrix[5, 7] = \
            M61, M62, M64, M65, M66, M68

        M_matrix[6, 2] = M73
        M_matrix[7, 6] = M87
        M_matrix[8, 8] = M99
        M_matrix[9, 9] = M1010
        M_matrix[10, 0], M_matrix[10, 10] = M111, M1111
        M_matrix[11, 11] = M1212
        M_matrix[12, 12] = M1313

        A_matrix = SX.zeros((self.state_dim - 3, self.state_dim - 3))
        A_matrix[0, 0], A_matrix[0, 1], A_matrix[1, 0], A_matrix[1, 1], A_matrix[1, 2], A_matrix[1, 3], A_matrix[1, 6] = \
            A11, A12, A21, A22, A23, A24, A27
        A_matrix[2, 0], A_matrix[2, 1], A_matrix[2, 4], A_matrix[2, 5] \
            = A31, A32, A35, A36

        A_matrix[3, 4], A_matrix[3, 5] = A45, A46
        A_matrix[4, 2], A_matrix[4, 4], A_matrix[4, 5], A_matrix[4, 6], A_matrix[4, 7] = \
            A53, A55, A56, A57, A58
        A_matrix[5, 1], A_matrix[5, 5], A_matrix[6, 3], A_matrix[7, 7], \
            A_matrix[8, 1], A_matrix[9, 5], A_matrix[11, 0], A_matrix[11, 8], \
            A_matrix[12, 4], A_matrix[12, 9] = A62, A66, A74, A88, \
            A92, A106, A121, A129, \
            A135, A1310

        B_matrix = SX.zeros((self.state_dim - 3, 1))
        B_matrix[0, 0] = B11
        B_matrix[1, 0] = B21
        B_matrix[2, 0] = B31
        # beta1dot, psi1_dotdot, phi1dot, phi1_dotdot,
        # beta2dot, psi2_dotdot, phi2dot, phi2_dotdot,
        # psi1dot, psi2dot, vy1dot py1dot,

        X_dot = mtimes(mtimes(inv(M_matrix), A_matrix), X_matrix) + mtimes(mtimes(inv(M_matrix), B_matrix), self.U[0])
        # px1, py1, psi1, px2, py2, psi2,
        # beta1, psi1_dot, phi1, phi1_dot, beta2, psi2_dot, phi2, phi2_dot, vy1
        self.dyn = vertcat(self.X[0] + delta_t * (self.X[3] * cos(self.X[2]) - self.X[15] * sin(self.X[2])),
                           self.X[1] + delta_t * (vx * sin(self.X[2]) + self.X[15] * cos(self.X[2])),#X_dot[11]
                           self.X[2] + delta_t * self.psi1dot,
                           self.X[3] + delta_t * self.U[1],
                           self.X[0] - self.b * cos(self.X[2]) - self.e * cos(self.X[6]),
                           self.X[1] - self.b * sin(self.X[2]) - self.e * sin(self.X[6]),
                           self.X[6] + delta_t * self.psi2dot,
                           self.X[7:15] + delta_t * X_dot[0:8],
                           self.X[15] + delta_t * X_dot[10])

        self.Path_Cost_Update(ref_traj_init, action_last_np)

    def Path_Cost_Update(self, ref, action_last_np):
        # ref:ref_y, ref_vx, ref_phi, ref_x + (self.X[0] - ref[0]) ** 2
        self.path_cost = self.wdis * ((self.X[1] - ref[1]) ** 2 ) + \
                         self.wvx1 * (self.X[3]-ref[3]) ** 2 + self.wpsi1 * (self.X[2] - ref[2]) ** 2 \
                         + self.womega1 * (self.X[8]) ** 2 + self.wbeta1 * (self.X[7]) ** 2 + \
                         self.wphi1 * (self.X[9]) ** 2 + self.wphi1dot * (self.X[10]) ** 2 + \
                         self.wu * (self.U[0]** 2+self.U[1]** 2) \
                         + self.wudot * ((self.U[0] - action_last_np[0]) ** 2+(self.U[1] - action_last_np[1]) ** 2)
        #+ (self.X[0] - ref[0]) ** 2
        self.final_cost = self.wdis * ((self.X[1] - ref[1]) ** 2 ) + \
                         self.wvx1 * (self.X[3]-ref[3]) ** 2 + self.wpsi1 * (self.X[2] - ref[2]) ** 2 \
                         + self.womega1 * (self.X[8]) ** 2 + self.wbeta1 * (self.X[7]) ** 2 + \
                         self.wphi1 * (self.X[9]) ** 2 + self.wphi1dot * (self.X[10]) ** 2
        return self.path_cost, self.final_cost

    def stepPhysics_i(self, states, action):
        state_next = torch.empty_like(states)
        action = action.type(torch.float32)
        X_matrix = torch.cat((states[7:15], states[2:3], states[6:7],
                              states[15:16], states[1:2], states[5:6]))
        vx = states[3]
        M11 = self.m1 * vx * self.c
        M12 = self.I1zz
        M13 = -self.m1s * self.h1c * self.c - self.I1xz

        M21 = self.m1 * vx * self.h1c - self.m1s * self.h1 * vx
        M22 = -self.I1xz
        M24 = self.I1xx + 2 * self.m1s * self.h1 * self.h1 - self.m1s * self.h1 * self.h1c

        M31 = self.m1 * vx
        M34 = -self.m1s * self.h1
        M35 = self.m2 * vx
        M38 = -self.m2s * self.h2

        M45 = self.m2 * vx * self.e
        M46 = -self.I2zz
        M48 = self.I2xz - self.m2s * self.h2 * self.e

        M55 = self.m2 * vx * self.h2c - self.m2s * self.h2 * vx
        M56 = -self.I2xz
        M58 = self.I2xx + 2 * self.m2s * self.h2 * self.h2 - self.m2s * self.h2 * self.h2c

        M61 = 1
        M62 = -self.c / vx
        M64 = -self.h1c / vx
        M65 = -1
        M66 = -self.e / vx
        M68 = self.h2c / vx

        M73 = 1

        M87 = 1

        M99 = 1
        M1010 = 1
        M111 = -vx
        M1111 = 1
        M1212 = 1
        M1313 = 1

        A11 = (self.c + self.a) * self.kf + (self.c - self.b) * self.km
        A12 = self.a * (self.c + self.a) * self.kf / vx - self.b * (
                self.c - self.b) * self.km / vx - self.m1 * vx * self.c
        A21 = (self.kf + self.km) * self.h1c
        A22 = (self.a * self.kf - self.b * self.km) * self.h1c / vx + (
                self.m1s * self.h1 - self.m1 * self.h1c) * vx
        A23 = self.m1s * self.gravity * self.h1 - self.kr1 - self.ka
        A24 = -self.c1
        A27 = self.ka

        A31 = self.kf + self.km
        A32 = (self.a * self.kf - self.b * self.km) / vx - self.m1 * vx
        A35 = self.kr
        A36 = -self.d * self.kr / vx - self.m2 * vx

        A45 = (self.e + self.d) * self.kr
        A46 = -self.d * (self.e + self.d) * self.kr / vx - self.m2 * vx * self.e
        A53 = self.ka
        A55 = self.kr * self.h2c
        A56 = (self.m2s * self.h2 - self.m2 * self.h2c) * vx - self.d * self.kr * self.h2c / vx
        A57 = self.m2s * self.gravity * self.h2 - self.kr2 - self.ka
        A58 = -self.c2
        A62 = -1
        A66 = 1
        A74 = 1
        A88, A92, A106 = 1, 1, 1
        A121, A129, A135, A1310 = vx, vx, vx, vx

        B11 = - (self.c + self.a) * self.kf
        B21 = -self.kf * self.h1c
        B31 = -self.kf
        
        
        M_matrix = torch.zeros((self.state_dim-3, self.state_dim-3))
        M_matrix[0, 0] = M11
        M_matrix[0, 1] = M12
        M_matrix[0, 2] = M13

        M_matrix[1, 0], M_matrix[1, 1], M_matrix[1, 3] = M21, M22, M24

        M_matrix[2, 0], M_matrix[2, 3], M_matrix[2, 4], M_matrix[2, 7] = \
            M31, M34, M35, M38

        M_matrix[3, 4], M_matrix[3, 5], M_matrix[3, 7], = M45, M46, M48

        M_matrix[4, 4], M_matrix[4, 5], M_matrix[4, 7] = M55, M56, M58

        M_matrix[5, 0], M_matrix[5, 1], M_matrix[5, 3], M_matrix[5, 4], M_matrix[5, 5], M_matrix[5, 7] = \
            M61, M62, M64, M65, M66, M68

        M_matrix[6, 2] = M73
        M_matrix[7, 6] = M87
        M_matrix[8, 8] = M99
        M_matrix[9, 9] = M1010
        M_matrix[10, 0], M_matrix[10, 10] = M111, M1111
        M_matrix[11, 11] = M1212
        M_matrix[12, 12] = M1313

        A_matrix = torch.zeros((self.state_dim-3, self.state_dim-3))
        A_matrix[0, 0], A_matrix[0, 1], A_matrix[1, 0], A_matrix[1, 1], A_matrix[1, 2], A_matrix[1, 3], A_matrix[1, 6] = \
            A11, A12, A21, A22, A23, A24, A27
        A_matrix[2, 0], A_matrix[2, 1], A_matrix[2, 4], A_matrix[2, 5] \
            = A31, A32, A35, A36

        A_matrix[3, 4], A_matrix[3, 5] = A45, A46
        A_matrix[4, 2], A_matrix[4, 4], A_matrix[4, 5], A_matrix[4, 6], A_matrix[4, 7] = \
            A53, A55, A56, A57, A58
        A_matrix[5, 1], A_matrix[5, 5], A_matrix[6, 3], A_matrix[7, 7], \
        A_matrix[8, 1], A_matrix[9, 5], A_matrix[11, 0], A_matrix[11, 8], \
        A_matrix[12, 4], A_matrix[12, 9] = A62, A66, A74, A88, \
                                           A92, A106, A121, A129, \
                                           A135, A1310

        B_matrix = torch.zeros((self.state_dim-3, 1))
        B_matrix[0, 0] = B11
        B_matrix[1, 0] = B21
        B_matrix[2, 0] = B31

        X_dot = (torch.matmul(torch.matmul(torch.inverse(M_matrix), A_matrix), X_matrix)  +
                 torch.matmul(torch.inverse(M_matrix), torch.matmul(B_matrix, action[0:1]))).squeeze()

        state_next[0] = states[0] + self.dt * (vx * torch.cos(states[2].clone()) - states[15] * torch.sin(states[2].clone()))
        state_next[1] = states[1] + self.dt * X_dot[11]#(vx * torch.sin(states[2].clone()) + states[14] * torch.cos(states[2].clone()))
        state_next[2] = states[2] + self.dt * X_dot[8]
        state_next[3] = states[3] + self.dt * action[1]
        state_next[4] = state_next[0] - self.b * torch.cos(states[2].clone()) - self.e * torch.cos(
            states[6].clone())  # posx_trailer
        state_next[5] = state_next[1] - self.b * torch.sin(states[2].clone()) - self.e * torch.sin(
            states[6].clone())  # posy_trailer
        state_next[6] = states[6] + self.dt * X_dot[9]
        state_next[7:15] = states[7:15] + self.dt * X_dot[0:8]
        state_next[15] = states[15] + self.dt * X_dot[10]
        return state_next

    def reward_func_i(self, states, action, ref, cost_paras, action_last_i):
        # to calculate dcx in jacobian API
        # #+(states[0] - ref_x) ** 2
        ref_x, ref_y, ref_phi, ref_u = ref[0], ref[1], ref[2], ref[3]
        reward = cost_paras[0] * ((states[1] - ref_y) ** 2) + \
                 cost_paras[1] * ((states[3] - ref_u) ** 2) + \
                 cost_paras[2] * ((states[2] - ref_phi) ** 2) + \
                 cost_paras[3] * (states[8]) ** 2 + \
                 cost_paras[4] * (states[7]) ** 2 + \
                 cost_paras[5] * (states[9]) ** 2 + \
                 cost_paras[6] * (states[10]) ** 2 + \
                 cost_paras[7] * (action[0] ** 2+action[1] ** 2) + \
                 cost_paras[8] * ((action[0]-action_last_i[0]) ** 2+(action[1]-action_last_i[1]) ** 2)
        return reward

class Semitruckpu7dofModel(PythBaseModel):
    def __init__(
        self,
        pre_horizon: int = 100,
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
        self.state_dim = self.vehicle_dynamics.state_dim
        self.cost_dim = 7+2
        ego_obs_dim = 16
        ref_obs_dim = 7
        obs_scale_default = [1/100, 1/100, 1/10, 1/100, 1/100, 1/100, 1/10,
                             1, 1, 1, 1,
                             1, 1, 1, 1,
                             1/100]
        self.obs_scale = np.array(kwargs.get('obs_scale', obs_scale_default))
        super().__init__(
            obs_dim=ego_obs_dim + ref_obs_dim * pre_horizon,
            action_dim=2,
            dt=0.01,
            action_lower_bound=[-max_steer, -3],
            action_upper_bound=[max_steer, 3],
            device=device,
        )
        self.ref_traj = MultiRefTrajModel(path_para, u_para)
        self.action_last = torch.zeros((self.batch_size, 2))
        self.action_last_np = np.zeros((2, ))
        self.cost_paras = np.array(kwargs["cost_paras"])
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
        ref_u_tf = ref_points[..., 3] - state[:, 3].unsqueeze(1)
        ref_x2_tf, ref_y2_tf, ref_phi2_tf = \
            ego_vehicle_coordinate_transform(
                state[:, 4], state[:, 5], state[:, 6],
                ref_points[..., 4], ref_points[..., 5], ref_points[..., 6],
            )
        ego_obs = torch.concat((torch.stack((ref_x_tf[:, 0]*self.obs_scale[0], ref_y_tf[:, 0]*self.obs_scale[1], ref_phi_tf[:, 0]*self.obs_scale[2], ref_u_tf[:, 0]*self.obs_scale[3],
                                             ref_x2_tf[:, 0]*self.obs_scale[4], ref_y2_tf[:, 0]*self.obs_scale[5], ref_phi2_tf[:, 0]*self.obs_scale[6]), dim=1),
                                state[:, 7:15], state[:, 15:]*self.obs_scale[15]), dim=1)
        ref_obs = torch.stack((ref_x_tf*self.obs_scale[0], ref_y_tf*self.obs_scale[1], ref_phi_tf*self.obs_scale[2], ref_u_tf*self.obs_scale[3],
                               ref_x2_tf*self.obs_scale[4], ref_y2_tf*self.obs_scale[5], ref_phi2_tf*self.obs_scale[6]), 2)[
            :, 1:].reshape(ego_obs.shape[0], -1)
        return torch.concat((ego_obs, ref_obs), 1)

    def compute_reward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor) -> torch.Tensor:
        # (obs[:, 0]/self.obs_scale[0]) ** 2+
        return -(
            self.cost_paras[0] * ((obs[:, 1]/self.obs_scale[1]) ** 2)
            + self.cost_paras[1] * (obs[:, 3]/self.obs_scale[3]) ** 2
            + self.cost_paras[2] * (obs[:, 2]/self.obs_scale[2]) ** 2
            + self.cost_paras[3] * obs[:, 8] ** 2
            + self.cost_paras[4] * obs[:, 7] ** 2
            + self.cost_paras[5] * obs[:, 9] ** 2
            + self.cost_paras[6] * obs[:, 10] ** 2
            + self.cost_paras[7] * (action[:, 0] ** 2+action[:, 1] ** 2)
            + self.cost_paras[8] * ((action[:, 0] - self.action_last[:, 0]) ** 2
                                    + (action[:, 1] - self.action_last[:, 1]) ** 2)
        )

    def judge_done(self, obs: torch.Tensor) -> torch.Tensor:
        delta_x, delta_y, delta_phi, delta_u, delta_x2, delta_y2, delta_phi2 = obs[:, 0]/self.obs_scale[0], obs[:, 1]/self.obs_scale[1], obs[:, 2]/self.obs_scale[2], obs[:, 3]/self.obs_scale[3], obs[:, 4]/self.obs_scale[4], obs[:, 5]/self.obs_scale[5], obs[:, 6]/self.obs_scale[6]
        done = ((torch.abs(delta_y) > 3)
                | (torch.abs(delta_u) > 3)
                | (torch.abs(delta_phi) > np.pi)
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
                action_last_i = self.action_last[i, :]
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
    #     pos_x1, pos_y1 = state[:1, self.state_dim-3].clone(), state[:1, self.state_dim-4].clone()
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
    ego_phi: torch.Tensor,
    ref_x: torch.Tensor,
    ref_y: torch.Tensor,
    ref_phi: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ego_x, ego_y, ego_phi = ego_x.unsqueeze(1), ego_y.unsqueeze(1), ego_phi.unsqueeze(1)
    x_err = ego_x - ref_x
    y_err = ego_y - ref_y
    phi_err = angle_normalize(ego_phi - ref_phi)
    return x_err, y_err, phi_err

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
    return Semitruckpu7dofModel(**kwargs)
