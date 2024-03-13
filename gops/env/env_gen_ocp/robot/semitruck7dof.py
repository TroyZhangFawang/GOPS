import math
from typing import NamedTuple

import numpy as np
from gym import spaces
from gops.env.env_gen_ocp.pyth_base import Robot
from gops.utils.math_utils import angle_normalize


class Semitruck7DoFParam(NamedTuple):
    state_dim = 15
    v_x = 20
    m1 = 5760.  # Total mass of the tractor [kg]
    m1s = 4455.  # Sprung mass of the tractor [kg]
    m2 = 20665  # Total mass of the semitrailer [kg]
    m2s = 20000  # Sprung mass of the semitrailer [kg]
    gravity = 9.81
    a = 1.1  # Distance between the center of gravity (CG) of the tractor and its front axle [m]
    b = 2.8  # Distance between the CG of the tractor and its rear axle [m]
    c = 1.9  # Distance between the hitch point and the center of gravity (CG) of the tractor [m]
    e = 1.24  # Distance between the hitch point and the CG of the semitrailer [m]
    d = 6.9  # Distance between the rear axle and the CG of the semitrailer [m]

    h1 = 1.175  # Height of the CG of the sprung mass for the tractor [m]
    h2 = 2.125  # Height of the CG of the sprung mass for the semitrailer [m]
    h1c = 1.1  # Height of hitch point to the roll center of the sprung mass for the tractor [m]
    h2c = 1.1  # Height of hitch point to the roll center of the sprung mass for the semitrailer [m]

    I1zz = 34802.6  # Yaw moment of inertia of the whole mass of the tractor [kg m^2]
    I1xx = 2283  # Roll moment of inertia of the sprung mass of the tractor [kg m^2]
    I1yy = 35402
    I1xz = 1626  # Roll–yaw product of inertia of the sprung mass of the tractor [kg m^2]
    I2zz = 250416  # Yaw moment of inertia of the whole mass of the semitrailer [kg m^2]
    I2xx = 22330  # Roll moment of inertia of the sprung mass of the semitrailer [kg m^2]
    I2xz = 0.0  # Roll–yaw product of inertia of the sprung mass of the semitrailer [kg m^2]

    kf = -4.0889e5  # Tire cornering stiffness of the front axle of the tractor [N/rad]
    km = -9.1361e5  # Tire cornering stiffness of the rear axle of the tractor [N/rad]
    kr = -6.5922e5  # Tire cornering stiffness of the rear axle of the trailer [N/rad]
    kr1 = 9.1731e5  # roll stiffness of tire (front)[N/m]
    kr2 = 2.6023e6  # roll stiffness of tire (rear)[N/m]
    ka = 3.5503e6  # Roll stiffness of the articulation joint between the tractor and semitrailer [N m/rad]
    c1 = 1.2727e6  # Roll damping of the tractor's suspension [N-s/m]
    c2 = 4.1745e5  # Roll damping of the semitrailer's suspension [N-s/m]

    M11 = m1 * v_x * c
    M12 = I1zz
    M13 = -m1s * h1c * c - I1xz

    M21 = m1 * v_x * h1c - m1s * h1 * v_x
    M22 = -I1xz
    M24 = I1xx + 2 * m1s * h1 * h1 - m1s * h1 * h1c

    M31 = m1 * v_x
    M34 = -m1s * h1
    M35 = m2 * v_x
    M38 = -m2s * h2

    M45 = m2 * v_x * e
    M46 = -I2zz
    M48 = I2xz - m2s * h2 * e

    M55 = m2 * v_x * h2c - m2s * h2 * v_x
    M56 = -I2xz
    M58 = I2xx + 2 * m2s * h2 * h2 - m2s * h2 * h2c

    M61 = 1
    M62 = -c / v_x
    M64 = -h1c / v_x
    M65 = -1
    M66 = -e / v_x
    M68 = h2c / v_x

    M73 = 1

    M87 = 1

    M99 = 1
    M1010 = 1
    M111 = -v_x
    M1111 = 1
    M1212 = 1
    M1313 = 1

    A11 = (c + a) * kf + (c - b) * km
    A12 = a * (c + a) * kf / v_x - b * (
            c - b) * km / v_x - m1 * v_x * c
    A21 = (kf + km) * h1c
    A22 = (a * kf - b * km) * h1c / v_x + (
            m1s * h1 - m1 * h1c) * v_x
    A23 = m1s * gravity * h1 - kr1 - ka
    A24 = -c1
    A27 = ka

    A31 = kf + km
    A32 = (a * kf - b * km) / v_x - m1 * v_x
    A35 = kr
    A36 = -d * kr / v_x - m2 * v_x

    A45 = (e + d) * kr
    A46 = -d * (e + d) * kr / v_x - m2 * v_x * e
    A53 = ka
    A55 = kr * h2c
    A56 = (m2s * h2 - m2 * h2c) * v_x - d * kr * h2c / v_x
    A57 = m2s * gravity * h2 - kr2 - ka
    A58 = -c2
    A62 = -1
    A66 = 1
    A74 = 1
    A88, A92, A106 = 1, 1, 1
    A121, A129, A135, A1310 = v_x, v_x, v_x, v_x

    B11 = - (c + a) * kf
    B21 = -kf * h1c
    B31 = -kf

class Semitruck7DoF(Robot):
    def __init__(
        self,
        *,
        dt: float = 0.01,
        max_steer: float = 0.5,
    ):
        self.param = Semitruck7DoFParam()
        self.dt = dt
        self.state = None
        self.param.state_dim = 15
        self.state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.param.state_dim,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-max_steer], dtype=np.float32),
            high=np.array([max_steer], dtype=np.float32),
        )

    def step(self, action: np.ndarray) -> np.ndarray:
        next_state = self.state.copy()
        M_matrix = np.zeros((self.param.state_dim - 2, self.param.state_dim - 2))
        M_matrix[0, 0] = self.param.M11
        M_matrix[0, 1] = self.param.M12
        M_matrix[0, 2] = self.param.M13

        M_matrix[1, 0], M_matrix[1, 1], M_matrix[1, 3] = self.param.M21, self.param.M22, self.param.M24

        M_matrix[2, 0], M_matrix[2, 3], M_matrix[2, 4], M_matrix[2, 7] = \
            self.param.M31, self.param.M34, self.param.M35, self.param.M38

        M_matrix[3, 4], M_matrix[3, 5], M_matrix[3, 7], = self.param.M45, self.param.M46, self.param.M48

        M_matrix[4, 4], M_matrix[4, 5], M_matrix[4, 7] = self.param.M55, self.param.M56, self.param.M58

        M_matrix[5, 0], M_matrix[5, 1], M_matrix[5, 3], M_matrix[5, 4], M_matrix[5, 5], M_matrix[5, 7] = \
            self.param.M61, self.param.M62, self.param.M64, self.param.M65, self.param.M66, self.param.M68

        M_matrix[6, 2] = self.param.M73
        M_matrix[7, 6] = self.param.M87
        M_matrix[8, 8] = self.param.M99
        M_matrix[9, 9] = self.param.M1010
        M_matrix[10, 0], M_matrix[10, 10] = self.param.M111, self.param.M1111
        M_matrix[11, 11] = self.param.M1212
        M_matrix[12, 12] = self.param.M1313

        A_matrix = np.zeros((self.param.state_dim - 2, self.param.state_dim - 2))
        A_matrix[0, 0], A_matrix[0, 1], A_matrix[1, 0], A_matrix[1, 1], A_matrix[1, 2], A_matrix[1, 3], A_matrix[1, 6] = \
            self.param.A11, self.param.A12, self.param.A21, self.param.A22, self.param.A23, self.param.A24, self.param.A27
        A_matrix[2, 0], A_matrix[2, 1], A_matrix[2, 4], A_matrix[2, 5] \
            = self.param.A31, self.param.A32, self.param.A35, self.param.A36

        A_matrix[3, 4], A_matrix[3, 5] = self.param.A45, self.param.A46
        A_matrix[4, 2], A_matrix[4, 4], A_matrix[4, 5], A_matrix[4, 6], A_matrix[4, 7] = \
            self.param.A53, self.param.A55, self.param.A56, self.param.A57, self.param.A58
        A_matrix[5, 1], A_matrix[5, 5], A_matrix[6, 3], A_matrix[7, 7], \
        A_matrix[8, 1], A_matrix[9, 5], A_matrix[11, 0], A_matrix[11, 8], \
        A_matrix[12, 4], A_matrix[12, 9] = self.param.A62, self.param.A66, self.param.A74, self.param.A88, \
                                           self.param.A92, self.param.A106, self.param.A121, self.param.A129, \
                                           self.param.A135, self.param.A1310

        B_matrix = np.zeros((self.param.state_dim - 2, 1))
        B_matrix[0, 0] = self.param.B11
        B_matrix[1, 0] = self.param.B21
        B_matrix[2, 0] = self.param.B31

        X_dot = (np.matmul(np.matmul(np.linalg.inv(M_matrix), A_matrix), self.state[:self.param.state_dim - 2]) + np.matmul(
            np.linalg.inv(M_matrix), np.matmul(B_matrix, action))).squeeze()
        next_state[:12] = self.state[:12] + self.dt * X_dot[:12]
        next_state[12] = next_state[11] - self.param.b * np.sin(next_state[8]) - self.param.e * np.sin(
            next_state[9])  # posy_trailer
        next_state[13] = self.state[13] + self.dt * self.param.v_x
        next_state[14] = next_state[13] - self.param.b * np.cos(self.state[8]) - self.param.e * np.cos(
            self.state[9])  # posx_trailer
        self.state = next_state
        return self.state

