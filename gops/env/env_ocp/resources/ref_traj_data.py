#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: reference trajectory for data environment
#  Update: 2022-11-16, Yujie Yang: create reference trajectory

from abc import ABCMeta, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

DEFAULT_PATH_PARAM = {
    "sine": {"A": 1.5, "omega": 2 * np.pi / 10, "phi": 0.0,},
    # "double_lane": {
    #     "t1": 1.35,
    #     "t2": 6.75,
    #     "t3": 8.1,
    #     "t4": 13.5,
    #     "y1": 0.0,
    #     "y2": 3.5,
    # },
    "double_lane": {
        "t1": 5.0,
        "t2": 9.0,
        "t3": 14.0,
        "t4": 18.0,
        "y1": 0.0,
        "y2": 3.5,
    },
    "triangle": {"A": 3.0, "T": 10.0, },
    "circle": {"r": 100.0, },
    "straight_lane": {"A": 0.0, "T": 100.0, },
    "u_turn": {"r": 50.0, "l1": 100.0,  "l2": 100.0},
    "figure_eight": {"a": 80.0, "b":80, "omega1":np.pi/100, "omega2":np.pi*2/100} #李萨如曲线
}

DEFAULT_SPEED_PARAM = {
    "constant": {"u": 10, },
    "sine": {"A": 3.0, "omega": 2 * np.pi / 10, "phi": 0.0, "b": 10.0,},
}

DEFAULT_SLOPE_PARAM = {
    "constant": {"longi_slope": 0.05, "lat_slope": 0.05},
    "sine": {"A": 0.05, "omega": 2 * np.pi / 10, "phi": 0.0, "b": 0.0,},
}

class MultiRefTrajData:
    def __init__(
        self,
        path_param: Optional[Dict[str, Dict]] = None,
        speed_param: Optional[Dict[str, Dict]] = None,
    ):
        self.path_param = deepcopy(DEFAULT_PATH_PARAM)
        if path_param is not None:
            for k, v in path_param.items():
                self.path_param[k].update(v)

        self.speed_param = deepcopy(DEFAULT_SPEED_PARAM)
        if speed_param is not None:
            for k, v in speed_param.items():
                self.speed_param[k].update(v)

        ref_speeds = [
            ConstantRefSpeedData(**self.speed_param["constant"]),
            SineRefSpeedData(**self.speed_param["sine"]),
        ]
        #

        self.ref_trajs: Sequence[RefTrajData] = [
            SineRefTrajData(ref_speeds, **self.path_param["sine"]),
            DoubleLaneRefTrajData(ref_speeds, **self.path_param["double_lane"]),
            TriangleRefTrajData(ref_speeds, **self.path_param["triangle"]),
            CircleRefTrajData(ref_speeds, **self.path_param["circle"]),
            TriangleRefTrajData(ref_speeds, **self.path_param["straight_lane"]),
            UTurnRefTrajData(ref_speeds, **self.path_param["u_turn"]),
            FigureEightRefTrajData(ref_speeds, **self.path_param["figure_eight"])
        ]

    def compute_x(self, t: float, path_num: int, speed_num: int) -> float:
        return self.ref_trajs[path_num].compute_x(t, speed_num)

    def compute_y(self, t: float, path_num: int, speed_num: int) -> float:
        return self.ref_trajs[path_num].compute_y(t, speed_num)

    def compute_u(self, t: float, path_num: int, speed_num: int) -> float:
        return self.ref_trajs[path_num].compute_u(t, speed_num)

    def compute_phi(self, t: float, path_num: int, speed_num: int) -> float:
        return self.ref_trajs[path_num].compute_phi(t, speed_num)


class MultiRoadSlopeData:
    def __init__(
        self,
        slope_param: Optional[Dict[str, Dict]] = None,
    ):
        self.slope_param = deepcopy(DEFAULT_SLOPE_PARAM)
        if slope_param is not None:
            for k, v in slope_param.items():
                self.slope_param[k].update(v)

        self.ref_slope = [
            ConstantRefSlopeData(**self.slope_param["constant"]),
            SineRefSlopeData(**self.slope_param["sine"]),
        ]

    def compute_longislope(self, t: float, slope_num: int) -> float:
        return self.ref_slope[slope_num].compute_longislope(t)

    def compute_latslope(self, t: float, slope_num: int) -> float:
        return self.ref_slope[slope_num].compute_latslope(t)


class RefSpeedData(metaclass=ABCMeta):
    @abstractmethod
    def compute_u(self, t: float) -> float:
        ...

    @abstractmethod
    def compute_integrate_u(self, t: float) -> float:
        ...


class RefSlopeData(metaclass=ABCMeta):
    @abstractmethod
    def compute_longislope(self, t: float) -> float:
        ...

    @abstractmethod
    def compute_latslope(self, t: float) -> float:
        ...

@dataclass
class ConstantRefSpeedData(RefSpeedData):
    u: float

    def compute_u(self, t: float) -> float:
        return self.u

    def compute_integrate_u(self, t: float) -> float:
        return self.u * t


@dataclass
class SineRefSpeedData(RefSpeedData):
    A: float
    omega: float
    phi: float
    b: float

    def compute_u(self, t: float) -> float:
        return self.A * np.sin(self.omega * t + self.phi) + self.b

    def compute_integrate_u(self, t: float) -> float:
        return (
            -self.A / self.omega * np.cos(self.omega * t + self.phi)
            + self.b * t
            + self.A / self.omega * np.cos(self.phi)
        )


@dataclass
class RefTrajData(metaclass=ABCMeta):
    ref_speeds: Sequence[RefSpeedData]

    @abstractmethod
    def compute_x(self, t: float, speed_num: int) -> float:
        ...

    @abstractmethod
    def compute_y(self, t: float, speed_num: int) -> float:
        ...

    def compute_u(self, t: float, speed_num: int) -> float:
        return self.ref_speeds[speed_num].compute_u(t)

    def compute_phi(self, t: float, speed_num: int) -> float:
        dt = 0.001
        dx = self.compute_x(t + dt, speed_num) - self.compute_x(t, speed_num)
        dy = self.compute_y(t + dt, speed_num) - self.compute_y(t, speed_num)
        return np.arctan2(dy, dx)


@dataclass
class SineRefTrajData(RefTrajData):
    A: float
    omega: float
    phi: float

    def compute_x(self, t: float, speed_num: int) -> float:
        return self.ref_speeds[speed_num].compute_integrate_u(t)

    def compute_y(self, t: float, speed_num: int) -> float:
        return self.A * np.sin(self.omega * t + self.phi)


@dataclass
class DoubleLaneRefTrajData(RefTrajData):
    t1: float
    t2: float
    t3: float
    t4: float
    y1: float
    y2: float

    def compute_x(self, t: float, speed_num: int) -> float:
        return self.ref_speeds[speed_num].compute_integrate_u(t)

    def compute_y(self, t: float, speed_num: int) -> float:
        if t <= self.t1:
            y = self.y1
        elif t <= self.t2:
            k = (self.y2 - self.y1) / (self.t2 - self.t1)
            y = k * (t - self.t1) + self.y1
        elif t <= self.t3:
            y = self.y2
        elif t <= self.t4:
            k = (self.y1 - self.y2) / (self.t4 - self.t3)
            y = k * (t - self.t3) + self.y2
        else:
            y = self.y1
        return y


@dataclass
class TriangleRefTrajData(RefTrajData):
    A: float
    T: float

    def compute_x(self, t: float, speed_num: int) -> float:
        return self.ref_speeds[speed_num].compute_integrate_u(t)

    def compute_y(self, t: float, speed_num: int) -> float:
        s = t % self.T
        if s <= self.T / 2:
            y = 2 * self.A / self.T * s
        else:
            y = -2 * self.A / self.T * (s - self.T)
        return y


@dataclass
class CircleRefTrajData(RefTrajData):
    r: float

    def compute_x(self, t: float, speed_num: int) -> float:
        arc_len = self.ref_speeds[speed_num].compute_integrate_u(t)
        return self.r * np.sin(arc_len / self.r)

    def compute_y(self, t: float, speed_num: int) -> float:
        arc_len = self.ref_speeds[speed_num].compute_integrate_u(t)
        return self.r * (np.cos(arc_len / self.r) - 1)


@dataclass
class UTurnRefTrajData(RefTrajData):
    r: float  # 转弯半径
    l1: float  # 第一段直线长度
    l2: float  # 第二段直线长度

    def compute_x(self, t: float, speed_num: int) -> float:
        distance = self.ref_speeds[speed_num].compute_integrate_u(t)
        return self._compute_x_from_distance(distance)

    def compute_y(self, t: float, speed_num: int) -> float:
        distance = self.ref_speeds[speed_num].compute_integrate_u(t)
        return self._compute_y_from_distance(distance)

    def _compute_x_from_distance(self, distance: float) -> float:
        if distance <= self.l1:  # 第一段直线
            return distance

        elif distance <= self.l1 + np.pi * self.r:  # 半圆弧
            arc_length = distance - self.l1
            return self.l1 + self.r * np.sin(arc_length / self.r)
        else:  # 第二段直线
            return self.l2 - (distance - self.l1 - np.pi * self.r)

    def _compute_y_from_distance(self, distance: float) -> float:
        if distance <= self.l1:  # 第一段直线
            return 0
        elif distance <= self.l1 + np.pi * self.r:  # 半圆弧
            arc_length = distance - self.l1
            return self.r * (1 - np.cos(arc_length / self.r))
        else:  # 第二段直线
            return 2 * self.r

@dataclass
class WaterDropRefTrajData(RefTrajData):
    a: float
    b: float
    def compute_x(self, t: float, speed_num: int) -> float:
        return -self.a * (np.cos(t)/(1+np.sin(t)**2)-1)

    def compute_y(self, t: float, speed_num: int) -> float:
        return -self.b * np.cos(t) * np.sin(t)/(1+np.sin(t)**2)


@dataclass
class FigureEightRefTrajData(RefTrajData):
    a: float  #表示在水平方向（𝑥）的振幅。
    b:float  #表示在垂直方向（𝑥）的振幅。
    omega1:float # 角频率 2pi/T, T为周期，T=10s, omega1 = pi/5
    omega2:float # 角频率 omega2 = 2 omega1
    def compute_x(self, t: float, speed_num: int) -> float:
        arc_len = self.ref_speeds[speed_num].compute_integrate_u(t)
        return self.a * np.sin(self.omega1*arc_len)

    def compute_y(self, t: float, speed_num: int) -> float:
        arc_len = self.ref_speeds[speed_num].compute_integrate_u(t)
        return self.b * np.sin(self.omega2*arc_len)


@dataclass
class ConstantRefSlopeData(RefSlopeData):
    longi_slope: float
    lat_slope: float

    def compute_longislope(self, t: float) -> float:
        return self.longi_slope

    def compute_latslope(self, t: float) -> float:
        return self.lat_slope


@dataclass
class SineRefSlopeData(RefSlopeData):
    A: float
    omega: float
    phi: float
    b: float

    def compute_longislope(self, t: float) -> float:
        return self.A * np.sin(self.omega * t + self.phi) + self.b

    def compute_latslope(self, t: float) -> float:
        return self.A * np.sin(self.omega * t + self.phi) + self.b


import matplotlib.pyplot as plt

def plot_traj(t, dt):
    path_para = None
    u_para = None
    ref_traj = MultiRefTrajData(path_para, u_para)
    path_num = 6
    u_num = 0
    x = []
    y = []
    for i in range(50):
        ref_x = ref_traj.compute_x(
            t + i * dt, path_num, u_num
        )
        ref_y = ref_traj.compute_y(
            t + i * dt, path_num, u_num
        )
        x.append(ref_x)
        y.append(ref_y)
    plt.figure()
    plt.plot(x, y)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.grid(True)
    plt.show()


def plot_slope(t, dt):
    slope_para = None
    ref_slope = MultiRoadSlopeData(slope_para)
    slope_num = 1
    longislope = []
    latslope = []
    time = []
    for i in range(10000):
        ref_longislope = ref_slope.compute_longislope(
            t + i * dt, slope_num
        )
        ref_latslope = ref_slope.compute_latslope(
            t + i * dt, slope_num
        )
        longislope.append(ref_longislope)
        latslope.append(ref_latslope)
        time.append(t + i * dt)
    plt.figure()
    plt.plot(time, longislope)
    plt.xlabel("Time")
    plt.ylabel("Longislope")
    plt.axis('equal')
    plt.grid(True)
    plt.show()

# plot_traj(t=0, dt=0.01)
# plot_slope(t=0, dt=0.01)