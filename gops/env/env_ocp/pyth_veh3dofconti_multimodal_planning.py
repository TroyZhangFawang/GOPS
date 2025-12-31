#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: vehicle 3DOF data environment with surrounding vehicles constraint
#  Update: 2023-01-08, Jiaxin Gao: create environment

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import gym
import numpy as np
from enum import Enum
from gops.env.env_ocp.pyth_base_env import PythBaseEnv
from gops.env.env_ocp.pyth_veh3dofcontiplanning import angle_normalize, ego_vehicle_coordinate_transform
from gops.env.env_ocp.resources.ref_traj_data import MultiRefTrajData

@dataclass
class DynamicObstacleData:
    x: float = 0.0
    y: float = 0.0
    phi: float = 0.0
    u: float = 0.0
    # front wheel angle
    delta: float = 0.0
    # distance from front axle to rear axle
    l: float = 3.0
    dt: float = 0.1
    veh_length: float = 4.8
    veh_width: float = 2.0

    def step(self):
        self.x = self.x + self.u * np.cos(self.phi) * self.dt
        self.y = self.y + self.u * np.sin(self.phi) * self.dt
        self.phi = self.phi + self.u * np.tan(self.delta) / self.l * self.dt
        self.phi = angle_normalize(self.phi)

class ObstacleType(Enum):
    """障碍物类型枚举"""
    STATIC = 0  # 静态障碍物
    DYNAMIC = 1  # 动态障碍物

@dataclass
class Obstacle:
    obs_id: str  # 障碍物ID
    x: float = 0.0 # 障碍物中心点x坐标
    y: float = 0.0 # 障碍物中心点y坐标
    phi: float = 0.0
    u: float = 0.0
    # front wheel angle
    delta: float = 0.0
    # distance from front axle to rear axle
    l: float = 3.0
    dt: float = 0.1
    height: float = 0.5  # 障碍物高度
    width: float = 1.8  # 障碍物宽度
    length: float = 1.8  # 障碍物长度
    obs_type: ObstacleType = 1  # 障碍物类型, 0-静态，1-动态

class VehicleDynamicsData:
    def __init__(self):
        self.vehicle_params = dict(
            k_f=-128915.5,  # front wheel cornering stiffness [N/rad]
            k_r=-85943.6,  # rear wheel cornering stiffness [N/rad]
            l_f=1.06,  # distance from CG to front axle [m]
            l_r=1.85,  # distance from CG to rear axle [m]
            m=1412.0,  # mass [kg]
            I_z=1536.7,  # Polar moment of inertia at CG [kg*m^2]
            miu=1.0,  # tire-road friction coefficient
            g=9.81,  # acceleration of gravity [m/s^2]
            veh_length=4.8,
            veh_width=2.0,
            ground_clearance=0.25,
            wheel_distance=1.8,
        )
        l_f, l_r, mass, g = (
            self.vehicle_params["l_f"],
            self.vehicle_params["l_r"],
            self.vehicle_params["m"],
            self.vehicle_params["g"],
        )
        F_zf, F_zr = l_r * mass * g / (l_f + l_r), l_f * mass * g / (l_f + l_r)
        self.vehicle_params.update(dict(F_zf=F_zf, F_zr=F_zr))

    def f_xu(self, states, actions, delta_t):
        x, y, phi, u, v, w = states
        steer, a_x = actions
        k_f = self.vehicle_params["k_f"]
        k_r = self.vehicle_params["k_r"]
        l_f = self.vehicle_params["l_f"]
        l_r = self.vehicle_params["l_r"]
        m = self.vehicle_params["m"]
        I_z = self.vehicle_params["I_z"]
        next_state = [
            x + delta_t * (u * np.cos(phi) - v * np.sin(phi)),
            y + delta_t * (u * np.sin(phi) + v * np.cos(phi)),
            phi + delta_t * w,
            u + delta_t * a_x,
            (
                m * v * u
                + delta_t * (l_f * k_f - l_r * k_r) * w
                - delta_t * k_f * steer * u
                - delta_t * m * np.square(u) * w
            )
            / (m * u - delta_t * (k_f + k_r)),
            (
                I_z * w * u
                + delta_t * (l_f * k_f - l_r * k_r) * v
                - delta_t * l_f * k_f * steer * u
            )
            / (I_z * u - delta_t * (np.square(l_f) * k_f + np.square(l_r) * k_r)),
        ]
        next_state[2] = angle_normalize(next_state[2])
        return np.array(next_state, dtype=np.float32)

class MultiObstacleProcessor:
    """多障碍物处理器"""

    def __init__(self):
        self.static_obstacles: Dict[str, Obstacle] = {}  # 静态障碍物集合
        self.dynamic_obstacles: Dict[str, Obstacle] = {}  # 动态障碍物集合

    def add_obstacle(self, obstacle: Obstacle):
        """添加障碍物到对应集合"""
        if obstacle.obs_type == ObstacleType.STATIC:
            self.static_obstacles[obstacle.obs_id] = obstacle
        else:
            self.dynamic_obstacles[obstacle.obs_id] = obstacle

    def clear_obstacles(self):
        """清空障碍物集合"""
        self.static_obstacles.clear()
        self.dynamic_obstacles.clear()

    def find_nearest_obstacle(self,
                              current_pos: np.ndarray,
                              obstacles: Dict[str, Obstacle]) -> Tuple[Optional[str], float]:
        """查找最近的障碍物"""
        if not obstacles:
            return None, float('inf')

        min_dist = float('inf')
        nearest_id = None
        for obs_id, obstacle in obstacles.items():
            # 计算到障碍物的距离
            obs_pos = np.array([obstacle.x, obstacle.y])
            dist = np.linalg.norm(current_pos - obs_pos)
            # 只考虑前方的障碍物
            if obs_pos[0] > current_pos[0] and dist < min_dist:
                min_dist = dist
                nearest_id = obs_id
        return nearest_id, min_dist
    def step(self, obstacle):
        obstacle.x = obstacle.x + obstacle.u * np.cos(obstacle.phi) * obstacle.dt
        obstacle.y = obstacle.y + obstacle.u * np.sin(obstacle.phi) * obstacle.dt
        obstacle.phi = obstacle.phi + obstacle.u * np.tan(obstacle.delta) / obstacle.l * obstacle.dt
        obstacle.phi = angle_normalize(obstacle.phi)

class BezierCurve:
    """二阶贝塞尔曲线类"""

    def __init__(self, p0: np.ndarray, p1: np.ndarray, p2: np.ndarray):
        self.p0 = p0  # 起点
        self.p1 = p1  # 控制点
        self.p2 = p2  # 终点

    def compute_point(self, t: float) -> np.ndarray:
        """计算贝塞尔曲线上的点
        Args:
            t: 参数t，范围[0,1]
        Returns:
            point: [x, y]坐标
        """
        return (1 - t) ** 2 * self.p0 + 2 * (1 - t) * t * self.p1 + t ** 2 * self.p2

    def compute_derivative(self, t: float) -> np.ndarray:
        """计算贝塞尔曲线在t处的导数"""
        return 2 * (1 - t) * (self.p1 - self.p0) + 2 * t * (self.p2 - self.p1)

class QuinticPolynomial:
    """五次多项式轨迹生成器"""

    def __init__(self, start_state, end_state, T):
        """
        初始化五次多项式曲线

        Args:
            start_state: 起点状态 [x, dx, ddx]，包含位置、速度、加速度
            end_state: 终点状态 [x, dx, ddx]，包含位置、速度、加速度
            T: 时间间隔
        """
        self.xs = start_state[0]  # 起点位置
        self.vxs = start_state[1]  # 起点速度
        self.axs = start_state[2]  # 起点加速度

        self.xe = end_state[0]  # 终点位置
        self.vxe = end_state[1]  # 终点速度
        self.axe = end_state[2]  # 终点加速度

        self.T = T

        # 计算多项式系数
        self.coeffs = self._solve_coefficients()

    def _solve_coefficients(self):
        """求解五次多项式系数"""
        A = np.array([
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 2, 0, 0],
            [self.T ** 5, self.T ** 4, self.T ** 3, self.T ** 2, self.T, 1],
            [5 * self.T ** 4, 4 * self.T ** 3, 3 * self.T ** 2, 2 * self.T, 1, 0],
            [20 * self.T ** 3, 12 * self.T ** 2, 6 * self.T, 2, 0, 0]
        ])

        b = np.array([
            self.xs, self.vxs, self.axs,
            self.xe, self.vxe, self.axe
        ])

        return np.linalg.solve(A, b)

    def compute_point(self, t):
        """计算时刻t的状态"""
        # 如果t超出范围，返回None表示需要使用全局轨迹
        # if t < 0 or t > self.T:
        #     return None, None, None

        pos = (self.coeffs[0] * t ** 5 + self.coeffs[1] * t ** 4 +
               self.coeffs[2] * t ** 3 + self.coeffs[3] * t ** 2 +
               self.coeffs[4] * t + self.coeffs[5])

        vel = (5 * self.coeffs[0] * t ** 4 + 4 * self.coeffs[1] * t ** 3 +
               3 * self.coeffs[2] * t ** 2 + 2 * self.coeffs[3] * t +
               self.coeffs[4])

        acc = (20 * self.coeffs[0] * t ** 3 + 12 * self.coeffs[1] * t ** 2 +
               6 * self.coeffs[2] * t + 2 * self.coeffs[3])

        return pos, vel, acc

class SimuVeh3dofcontiSurrCstr(PythBaseEnv):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }
    def __init__(
        self,
        pre_horizon: int = 10,
        path_para: Optional[Dict[str, Dict]] = None,
        u_para: Optional[Dict[str, Dict]] = None,
        max_steer: float = np.pi / 6,
        dynamic_obstacle_num: int = 1,
        static_obstacle_num: int = 1,
        d_pre: float = 20.0,  # 离障碍物多少远开始规划
        lateral_sample: float = 5.0,  # 横向采样距离
        forward_sample: float = 10.0,  # 纵向采样距离
        **kwargs: Any,
    ):
        work_space = kwargs.pop("work_space", None)
        if work_space is None:
            # initial range of [delta_x, delta_y, delta_phi, delta_u, v, w]
            init_high = np.array([2, 1, np.pi / 6, 2, 0.1, 0.1], dtype=np.float32)
            init_low = -init_high
            work_space = np.stack((init_low, init_high))
        super(SimuVeh3dofcontiSurrCstr, self).__init__(work_space=work_space, **kwargs)
        self.obstacle_processor = MultiObstacleProcessor()
        self.vehicle_dynamics = VehicleDynamicsData()
        self.ref_traj = MultiRefTrajData(path_para, u_para)
        self.state_dim = 6
        self.pre_horizon = pre_horizon
        ego_obs_dim = 6
        ref_obs_dim = 4
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(ego_obs_dim + ref_obs_dim * pre_horizon + (dynamic_obstacle_num+static_obstacle_num) * 4,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-max_steer, -3]),
            high=np.array([max_steer, 3]),
            dtype=np.float32,
        )
        self.dt = 0.01
        self.max_episode_steps = 300
        self.state = None
        self.path_num = None
        self.u_num = None
        self.t = None
        self.ref_points = None
        self.guide_traj = None
        self.guide_end = np.zeros((3, 4))
        self.guide_time_interval = 0
        self.t_start = self.t
        self.dynamic_obstacle_num = dynamic_obstacle_num
        self.static_obstacle_num = static_obstacle_num
        self.dynamic_state = np.zeros((dynamic_obstacle_num, 5), dtype=np.float32)
        self.static_state = np.zeros((static_obstacle_num, 4), dtype=np.float32)
        self.veh_length = self.vehicle_dynamics.vehicle_params["veh_length"]
        self.veh_width = self.vehicle_dynamics.vehicle_params["veh_width"]
        self.wheel_distance = self.vehicle_dynamics.vehicle_params["wheel_distance"]
        self.ground_clearance = self.vehicle_dynamics.vehicle_params["ground_clearance"]
        self.d_pre = d_pre
        self.lateral_sample = lateral_sample
        self.forward_sample = forward_sample
        self.info_dict = {
            "state": {"shape": (self.state_dim,), "dtype": np.float32},
            "ref_points": {"shape": (self.pre_horizon + 1, 4), "dtype": np.float32},
            "path_num": {"shape": (), "dtype": np.uint8},
            "u_num": {"shape": (), "dtype": np.uint8},
            "ref_time": {"shape": (), "dtype": np.float32},
            "ref": {"shape": (4,), "dtype": np.float32},
            "dynamic_state": {"shape": (dynamic_obstacle_num, 5), "dtype": np.float32},
            "constraint": {"shape": (dynamic_obstacle_num,), "dtype": np.float32},
            "static_state": {"shape": (static_obstacle_num, 4), "dtype": np.float32},
            "generate_guide":{"shape": (), "dtype": np.uint8},
            "guide_start":{"shape": (4, ), "dtype": np.float32},
            "guide_end":{"shape": (3, 4), "dtype": np.float32},
            "guide_time_interval": {"shape": (), "dtype": np.float32},
            "t_start":{"shape": (), "dtype": np.float32},
        }
        self.seed()

    def reset(
        self,
        init_state: Optional[Sequence] = None,
        ref_time: Optional[float] = None,
        ref_num: Optional[int] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, dict]:
        # super().reset(init_state, ref_time, ref_num, **kwargs)
        # todo 正式训练前记得把初始化改回随机版本
        if ref_time is not None:
            self.t = ref_time
            self.t_start = self.t
        else:
            self.t = 0#20.0 * self.np_random.uniform(0.0, 1.0)
            self.t_start = self.t

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
            self.path_num = 4#self.np_random.choice([0, 1, 2, 3])

        if u_num is not None:
            self.u_num = u_num
        else:
            self.u_num = 0#self.np_random.choice([0, 1])

        ref_points = []
        for i in range(self.pre_horizon + 1):
            ref_x = self.ref_traj.compute_x(
                self.t + i * self.dt, self.path_num, self.u_num
            )
            ref_y = self.ref_traj.compute_y(
                self.t + i * self.dt, self.path_num, self.u_num
            )
            ref_phi = self.ref_traj.compute_phi(
                self.t + i * self.dt, self.path_num, self.u_num
            )
            ref_u = self.ref_traj.compute_u(
                self.t + i * self.dt, self.path_num, self.u_num
            )
            ref_points.append([ref_x, ref_y, ref_phi, ref_u])
        self.ref_points = np.array(ref_points, dtype=np.float32)
        if init_state is not None:
            delta_state = np.array(init_state, dtype=np.float32)
        else:
            delta_state = np.array([0, 0, 0, 0, 0, 0])#self.sample_initial_state()
        self.state = np.concatenate(
            (self.ref_points[0] + delta_state[:4], delta_state[4:])
        )
        print("ref pos", self.ref_points[:2])
        if self.path_num == 3:
            # circle path
            dynamic_delta = -np.arctan2(Obstacle.l, self.ref_traj.ref_trajs[3].r)
        else:
            dynamic_delta = 0.0
        # add dynamic obstacle
        obstacles = []
        for i_dynamic in range(self.dynamic_obstacle_num):
            # avoid ego vehicle
            delta_t = 1#self.np_random.uniform(2, 4)
            dynamic_phi = self.ref_traj.compute_phi(self.t + delta_t, self.path_num, self.u_num)
            delta_lon = 0#1.0 * self.np_random.uniform(-1, 1)
            delta_lat = 0#1.0 * self.np_random.uniform(-1, 1)
            dynamic_x = self.ref_traj.compute_x(self.t + delta_t, self.path_num, self.u_num) + delta_lon
            dynamic_y = self.ref_traj.compute_y(self.t + delta_t, self.path_num, self.u_num) + delta_lat
            print("dynamic pos", [dynamic_x, dynamic_x])
            dynamic_u = np.random.uniform(0, 10, self.dynamic_obstacle_num)
            obstacles.append(
                Obstacle("D{}".format(i_dynamic),
                    x=dynamic_x,
                    y=dynamic_y,
                    phi=dynamic_phi,
                    u=dynamic_u[i_dynamic],
                    delta=dynamic_delta,
                    dt=self.dt,
                    height=1.85,
                    length=self.veh_length,
                    width=self.veh_width,
                    obs_type=ObstacleType.DYNAMIC
                )
            )

        # add static obstacle
        for i_static in range(self.static_obstacle_num):
            delta_t = 3#self.np_random.uniform(1, 4)
            static_obs_phi = self.ref_traj.compute_phi(self.t + delta_t, self.path_num, self.u_num)
            delta_lon = 0#1.0 * self.np_random.uniform(-1, 1)
            delta_lat = 0#1.0 * self.np_random.uniform(-1, 1)
            static_obs_x = self.ref_traj.compute_x(self.t + delta_t, self.path_num, self.u_num) + delta_lon
            static_obs_y = self.ref_traj.compute_y(self.t + delta_t, self.path_num, self.u_num) + delta_lat
            print("static pos", [static_obs_x, static_obs_y])
            self.static_length = np.random.uniform(0, 2, self.static_obstacle_num)
            self.static_width = np.array([2, 0.5])#np.random.uniform(0, 2, self.static_obstacle_num)
            self.static_height = np.array([0.5, 0.1])#np.random.uniform(0, 1, self.static_obstacle_num)
            obstacles.append(
                Obstacle("S{}".format(i_static),
                    x=static_obs_x,
                    y=static_obs_y,
                    phi=static_obs_phi,
                    u=0,
                    delta=dynamic_delta,
                    dt=self.dt,
                    length=self.static_length[i_static],
                    height=self.static_height[i_static],
                    width=self.static_width[i_static],
                    obs_type=ObstacleType.STATIC
                )
            )
        # 将障碍物添加到处理器中
        for obs in obstacles:
            self.obstacle_processor.add_obstacle(obs)

        self.update_dynamic_state()
        self.update_static_state()
        # 初始时刻添加完障碍物之后就先判断下是否需要生成引导轨迹
        self.obstacle, self.generate_guide = self.is_generate_guide()
        if self.generate_guide and self.obstacle != None:
            curves = self.generate_quintic_curves(self.obstacle) # 生成3条五次多项式，每条包含2个元素，为横向、纵向位置多项式
            self.guide_traj, can_cross = self.can_cross_decision(self.obstacle, curves)
            guide_points = [self.state[:4]]
            for i in range(1, self.pre_horizon + 1):
                ref_x = self.ref_traj.compute_x(self.t + i * self.dt, self.path_num, self.u_num)
                guide_point = self.get_guide_points(self.guide_traj, self.t + i * self.dt, self.t_start)
                t = (ref_x - guide_point[0]) / guide_point[3]
                while guide_point[0] < ref_x:
                    guide_point = self.get_guide_points(self.guide_traj, t,
                                                          self.t_start)
                    t += self.dt
                # 超出范围，使用全局轨迹点
                if guide_point[0] > self.obstacle.x:
                    self.generate_guide = 0
                    guide_point = np.array([
                        self.ref_traj.compute_x(self.t + i * self.dt, self.path_num, self.u_num),
                        self.ref_traj.compute_y(self.t + i * self.dt, self.path_num, self.u_num),
                        self.ref_traj.compute_phi(self.t + i * self.dt, self.path_num, self.u_num),
                        self.ref_traj.compute_u(self.t + i * self.dt, self.path_num, self.u_num),
                    ], dtype=np.float32
                    )
                guide_points.append(guide_point)
            self.ref_points = np.array(guide_points, dtype=np.float32)
        self.state_full = np.zeros((self.pre_horizon, self.state_dim))
        return self.get_obs(), self.info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # reward = self.compute_reward(action)
        # self.state = self.vehicle_dynamics.f_xu(self.state, action, self.dt)
        self.t = self.t + self.dt
        # # 新加的##########----要想这部分成功run起来，需要在mlp里的finitehorizonfull 的forward函数把取第一个action给注释掉
        self.state = self.vehicle_dynamics.f_xu(self.state, action[0, :], self.dt)
        self.state_full = np.empty((self.pre_horizon, self.state_dim))
        self.state_full[0, :] = self.state
        reward = self.compute_reward(action[0, :])
        # self.action = action

        state = self.state
        for i in range(1, self.pre_horizon):
                state = self.vehicle_dynamics.f_xu(state, action[i, :], self.dt)
                self.state_full[i, :] = state
        # #############

        self.ref_points[:-1] = self.ref_points[1:]

        if self.generate_guide and self.obstacle != None:
            ref_x = self.ref_traj.compute_x(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num)
            new_ref_point = self.get_guide_points(self.guide_traj, self.t + self.pre_horizon * self.dt, self.t_start)
            t = (ref_x - new_ref_point[0]) / new_ref_point[3]
            while new_ref_point[0] <= ref_x:
                new_ref_point = self.get_guide_points(self.guide_traj, t + self.pre_horizon * self.dt,
                                                      self.t_start)
                t += self.dt
            # if new_ref_point[0] >= self.guide_end[self.curve_index][0]:
            #     new_ref_point = self.guide_end[self.curve_index]
            # print("data new_ref_point", new_ref_point, "obstacle.x", self.obstacle.x)
            if new_ref_point[0] >= self.obstacle.x:
                self.generate_guide = 0
                new_ref_point = np.array([
                    self.ref_traj.compute_x(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num),
                    self.ref_traj.compute_y(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num),
                    self.ref_traj.compute_phi(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num),
                    self.ref_traj.compute_u(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num),
                ], dtype=np.float32
                )
        else:
            # 判断下是否需要生成引导轨迹
            self.obstacle, self.generate_guide = self.is_generate_guide()
            if self.generate_guide and self.obstacle != None:
                self.t_start = self.t
                curves = self.generate_quintic_curves(self.obstacle)  # 生成3条五次多项式，每条包含2个元素，为横向、纵向位置多项式
                self.guide_traj, can_cross = self.can_cross_decision(self.obstacle, curves)
                new_ref_point = self.get_guide_points(self.guide_traj, self.t + self.pre_horizon * self.dt, self.t_start)
                # 确保从引导轨迹生成的点在原参考轨迹的前面
                ref_x = self.ref_traj.compute_x(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num)
                t = (ref_x - new_ref_point[0])/new_ref_point[3]
                while new_ref_point[0] <= ref_x:
                    new_ref_point = self.get_guide_points(self.guide_traj, t + self.pre_horizon * self.dt,
                                                          self.t_start)
                    t += self.dt
                if new_ref_point[0] >= self.guide_end[self.curve_index][0]:
                    new_ref_point = self.guide_end[self.curve_index]
                # 超出范围，使用全局轨迹点
                if new_ref_point[0] > self.obstacle.x:
                    self.generate_guide = 0
                    new_ref_point = np.array([
                        self.ref_traj.compute_x(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num),
                        self.ref_traj.compute_y(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num),
                        self.ref_traj.compute_phi(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num),
                        self.ref_traj.compute_u(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num),
                    ], dtype=np.float32
                    )
            else:
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
        self.ref_points[-1] = new_ref_point

        for obstacle_id, dynamic_obs in self.obstacle_processor.dynamic_obstacles.items():
            self.obstacle_processor.step(dynamic_obs)
        self.update_dynamic_state()
        self.update_static_state()
        done = self.judge_done()
        if done:
            reward = reward - 1000
        obs = self.get_obs()
        return obs, reward, done, self.info

    def compute_reward(self, action: np.ndarray) -> float:
        # x, y, phi, u, _, w = self.state
        # ref_x, ref_y, ref_phi, ref_u = self.ref_points[0]
        obs = self.get_obs()
        delta_x, delta_y, delta_phi, delta_u, v, w = obs[0], obs[1], obs[2], obs[3], obs[4], obs[5]
        steer, a_x = action
        # dis = circle center distance - 2 * radius
        dis = - self.get_constraint()[0]
        collision_bound = 0.5
        dis_to_tanh = np.maximum(8 - 8 * dis / collision_bound, 0)
        punish_dis = np.tanh(dis_to_tanh - 4) + 1

        return -(
                1.0 * delta_x ** 2
                + 1.0 * delta_y ** 2
                + 0.1 * delta_phi ** 2
                + 0.1 * delta_u ** 2
                + 0.5 * v ** 2
                + 0.5 * w ** 2
                + 0.5 * steer ** 2
                + 0.5 * a_x ** 2
                + 15.0 * punish_dis
        )

    def judge_done(self) -> bool:
        x, y, phi = self.state[:3]
        ref_x, ref_y, ref_phi = self.ref_points[0, :3]
        dis = - self.get_constraint()
        done = (
                # (np.abs(x - ref_x) > 10)
                (np.abs(y - ref_y) > 5)
                | (np.abs(angle_normalize(phi - ref_phi)) > np.pi)
                | (np.any(dis < 0))
        )
        return done

    def get_obs(self) -> np.ndarray:
        ref_x_tf, ref_y_tf, ref_phi_tf = \
            ego_vehicle_coordinate_transform(
                self.state[0], self.state[1], self.state[2],
                self.ref_points[:, 0], self.ref_points[:, 1], self.ref_points[:, 2],
            )
        ref_u_tf = self.ref_points[:, 3] - self.state[3]
        # ego_obs: [
        # delta_x, delta_y, delta_phi, delta_u, (of the first reference point)v, w (of ego vehicle) ]
        ego_obs = np.concatenate(
            ([ref_x_tf[0], ref_y_tf[0], ref_phi_tf[0], ref_u_tf[0]], self.state[4:]))

        dynamic_x_tf, dynamic_y_tf, dynamic_phi_tf = \
            ego_vehicle_coordinate_transform(
                self.state[0], self.state[1], self.state[2],
                self.dynamic_state[:, 0], self.dynamic_state[:, 1], self.dynamic_state[:, 2],
            )
        dynamic_u_tf = self.dynamic_state[:, 3] - self.state[3]
        dynamic_obs = np.concatenate(
            (dynamic_x_tf, dynamic_y_tf, dynamic_phi_tf, dynamic_u_tf))

        static_x_tf, static_y_tf, static_phi_tf = \
            ego_vehicle_coordinate_transform(
                self.state[0], self.state[1], self.state[2],
                self.static_state[:, 0], self.static_state[:, 1], self.static_state[:, 2],
            )
        static_u_tf = self.static_state[:, 3] - self.state[3]

        static_obs = np.concatenate((static_x_tf, static_y_tf, static_phi_tf, static_u_tf))
        ref_obs = np.stack((ref_x_tf, ref_y_tf, ref_phi_tf, ref_u_tf), 1)[1:].flatten()
        return np.concatenate((ego_obs, ref_obs, dynamic_obs, static_obs))

    def get_constraint(self) -> np.ndarray:
        # collision detection using bicircle model
        # distance from vehicle center to front/rear circle center
        d = (self.veh_length - self.veh_width) / 2
        # circle radius
        r = np.sqrt(2) / 2 * self.veh_width
        x, y, phi = self.state[:3]
        ego_center = np.array(
            [
                [x + d * np.cos(phi), y + d * np.sin(phi)],
                [x - d * np.cos(phi), y - d * np.sin(phi)],
            ],
            dtype=np.float32,
        )
        dynamic_x = self.dynamic_state[:, 0]
        dynamic_y = self.dynamic_state[:, 1]
        dynamic_phi = self.dynamic_state[:, 2]
        # n * 2 * 2 first 2 is front and rear circle the second 2 is x and y position
        dynamic_center = np.stack(
            (
                # front circle
                # n * 2, n is num of dynamic obs, 2 is x and y position
                np.stack(
                    ((dynamic_x + d * np.cos(dynamic_phi)), dynamic_y + d * np.sin(dynamic_phi)),
                    axis=1,
                ),
                np.stack(
                    ((dynamic_x - d * np.cos(dynamic_phi)), dynamic_y - d * np.sin(dynamic_phi)),
                    axis=1,
                ),
            ),
            axis=1,
        )

        # static_x = self.static_state[:, 0]
        # static_y = self.static_state[:, 1]
        # static_phi = self.static_state[:, 2]
        # # n * 2 * 2 first 2 is front and rear circle the second 2 is x and y position
        # static_center = np.stack(
        #     (
        #         # front circle
        #         # n * 2, n is num of static, 2 is x and y position
        #         np.stack(
        #             ((static_x + d * np.cos(static_phi)), static_y + d * np.sin(static_phi)),
        #             axis=1,
        #         ),
        #         np.stack(
        #             ((static_x - d * np.cos(static_phi)), static_y - d * np.sin(static_phi)),
        #             axis=1,
        #         ),
        #     ),
        #     axis=1,
        # )

        min_dist = np.inf * np.ones(self.dynamic_obstacle_num, dtype=np.float32)
        for i in range(2):
            # front and rear circle of ego vehicle
            for j in range(2):
                # front and rear circle of dynamic vehicles
                dist = np.linalg.norm(
                    ego_center[np.newaxis, i] - dynamic_center[:, j], axis=1
                )
                min_dist = np.minimum(min_dist, dist)
        # min_dist_static = np.inf * np.ones(self.static_obstacle_num, dtype=np.float32)
        # for i in range(2):
        #     # front and rear circle of ego vehicle
        #     for j in range(2):
        #         # front and rear circle of static obstacles
        #         dist = np.linalg.norm(
        #             ego_center[np.newaxis, i] - static_center[:, j], axis=1
        #         )
        #         min_dist_static = np.minimum(min_dist_static, dist)
        # min_dist = np.minimum(min_dist_dynamic, min_dist_static)
        # dynamic_obstacle_num dist: between ego_veh and sur_veh min dis
        # ego_to_veh_violation = 2 * r - min_dist
        return 2 * r - min_dist

    def update_dynamic_state(self):
        for i, dynamic_obs in self.obstacle_processor.dynamic_obstacles.items():
            self.dynamic_state[int(i[1])] = np.array(
                [dynamic_obs.x, dynamic_obs.y, dynamic_obs.phi, dynamic_obs.u, dynamic_obs.delta],
                dtype=np.float32,
            )

    def update_static_state(self):
        for i, static_obs in self.obstacle_processor.static_obstacles.items():
            self.static_state[int(i[1])] = np.array(
                [static_obs.x, static_obs.y, static_obs.phi, 0],
                dtype=np.float32,
            )

    def is_generate_guide(self)-> Tuple[Optional[Obstacle], bool]:
        current_pos = self.state[:2]
        ego_velocity = self.state[3]  # 自车速度

        # 查找最近的动态和静态障碍物
        # nearest_dynamic_id, dynamic_dist = self.obstacle_processor.find_nearest_obstacle(
        #     current_pos, self.obstacle_processor.dynamic_obstacles)
        nearest_static_id, static_dist = self.obstacle_processor.find_nearest_obstacle(
            current_pos, self.obstacle_processor.static_obstacles)

        # 确定处理顺序
        # if dynamic_dist == float('inf') and static_dist == float('inf'):
        #     return None, 0

        # 判断是处理动态还是静态障碍物

        # if dynamic_dist <= static_dist:
        #     if dynamic_dist <= self.d_pre:
        #         obstacle = self.obstacle_processor.dynamic_obstacles[nearest_dynamic_id]
        #         # 检查动态障碍物的速度
        #         if obstacle.u >= ego_velocity:
        #             return None, 0
        #         else:
        #             return obstacle, 1
        #     else:
        #         return None, 0
        # else:
        #     if static_dist <= self.d_pre:
        #         obstacle = self.obstacle_processor.static_obstacles[nearest_static_id]
        #         return obstacle, 1
        #     else:
        #         return None, 0
        if static_dist <= self.d_pre:
            obstacle = self.obstacle_processor.static_obstacles[nearest_static_id]
            return obstacle, 1
        else:
            return None, 0

    def can_cross_decision(self, obstacle: Obstacle, curves: List) -> Tuple[QuinticPolynomial, bool]:
        """跨/绕决策
        Args:
            obstacle: 障碍物信息
        Returns:
            can_cross: 是否可以跨越
        """

        can_cross = False

        # 如果是静态障碍物，则判断尺寸是否满足跨越条件
        if (obstacle.height < self.ground_clearance and
                obstacle.width < self.wheel_distance):
            can_cross = True
        self.curve_index = None
        if can_cross:
            self.curve_index = 1
            best_curve = curves[self.curve_index]  # 可跨就选跨
        else:
            self.curve_index = 0
            best_curve = curves[self.curve_index]  # 不可跨就选择左绕

        return best_curve, can_cross

    def generate_bezier_curves(self,
                               current_pos: np.ndarray,
                               obstacle: Obstacle) -> List[BezierCurve]:
        """生成贝塞尔曲线
        Args:
            current_pos: 当前位置[x, y]
            obstacle: 障碍物信息
        Returns:
            curves: 生成的贝塞尔曲线列表
        """
        curves = []

        # 横向采样点
        lateral_points = [
            np.array([obstacle.x, obstacle.y - self.lateral_sample]),  # 左侧
            np.array([obstacle.x, obstacle.y]),  # 中心
            np.array([obstacle.x, obstacle.y + self.lateral_sample])  # 右侧
        ]

        # 纵向采样点
        longitudinal_points = [
            np.array([obstacle.x + self.forward_sample, obstacle.y])  # 前方位置
        ]

        # 生成贝塞尔曲线
        for i in lateral_points:
            # 使用当前位置作为起点，障碍物位置作为控制点
            curve = BezierCurve(
                current_pos,
                lateral_points[i],
                longitudinal_points[0]
            )
            curves.append(curve)

        return curves

    def generate_quintic_curves(self,
                               obstacle: Obstacle) -> List:
        """生成五次多项式曲线
                Args:
                    current_pos: 当前位置[x, y]
                    obstacle: 障碍物信息
                Returns:
                    curves: 生成的五次多项式曲线列表
                """
        def _create_single_trajectory(start_point, end_point, T) -> Tuple[QuinticPolynomial, QuinticPolynomial]:
            """创建单条轨迹的x和y方向多项式"""
            # x方向多项式
            poly_x = QuinticPolynomial(
                [start_point[0], start_point[3] * np.cos(start_point[2]), 0], # 位置、速度、加速度
                [end_point[0], end_point[3] * np.cos(end_point[2]), 0],# 位置、速度、加速度
                T
            )
            # y方向多项式
            poly_y = QuinticPolynomial(
                [start_point[1], start_point[3] * np.sin(start_point[2]), 0],
                [end_point[1], end_point[3] * np.sin(end_point[2]), 0],
                T
            )
            return (poly_x, poly_y)

        curves = []
        # 横向采样点
        lateral_points = [
            np.array([obstacle.x, obstacle.y - self.lateral_sample, obstacle.phi, obstacle.u]),  # 左侧
            np.array([obstacle.x, obstacle.y, obstacle.phi, obstacle.u]),  # 中心
            np.array([obstacle.x, obstacle.y + self.lateral_sample, obstacle.phi, obstacle.u])  # 右侧
        ]
        # 生成五次多项式曲线
        for i in range(len(lateral_points)):
            curve = _create_single_trajectory(self.state, lateral_points[i], abs((obstacle.x-self.state[0])/self.state[3]))
            # 使用当前位置作为起点，障碍物位置作为控制点
            curves.append(curve)
        self.guide_end = lateral_points
        self.guide_time_interval = abs((obstacle.x-self.state[0])/self.state[3])
        return curves

    def get_guide_points(self, guide_traj, current_t: float,
                              t_start: float) -> np.array:
        """
        获取指定轨迹的预测点

        Args:
            current_t: 当前时间
            t_start: 当前曲线的起点时间

        Returns:
            np.ndarray: 预测轨迹点 shape=(1, 4) [x, y, phi, v]
        """
        poly_x, poly_y = guide_traj
        local_t = current_t - t_start
        x_state = poly_x.compute_point(local_t)
        y_state = poly_y.compute_point(local_t)
        x, y = x_state[0], y_state[0]
        vx, vy = x_state[1], y_state[1]
        phi = np.arctan2(vy, vx)
        return np.array([x, y, phi, vx], dtype=np.float32)

    @property
    def info(self):
        return {
            "state": self.state.copy(),
            "ref_points": self.ref_points.copy(),
            "path_num": self.path_num,
            "u_num": self.u_num,
            "ref_time": self.t,
            "ref": self.ref_points[0].copy(),
            "constraint": self.get_constraint(),
            "dynamic_state": self.dynamic_state.copy(),
            "static_state": self.static_state.copy(),
            "generate_guide": self.generate_guide,
            "guide_start": self.state[:4],
            "guide_end": self.guide_end,
            "guide_time_interval": self.guide_time_interval,
            "t_start": self.t_start,
        }

    def render(self, mode="human"):
        import matplotlib.pyplot as plt

        fig = plt.figure(num=0, figsize=(6.4, 3.2))
        plt.clf()
        ego_x, ego_y = self.state[:2]
        ax = plt.axes(xlim=(ego_x - 5, ego_x + 30), ylim=(ego_y - 10, ego_y + 10))
        ax.set_aspect('equal')

        self._render(ax)

        plt.tight_layout()

        if mode == "rgb_array":
            fig.canvas.draw()
            image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_from_plot = image_from_plot.reshape(
                fig.canvas.get_width_height()[::-1] + (3,)
            )
            plt.pause(0.01)
            return image_from_plot
        elif mode == "human":
            plt.pause(0.01)
            plt.show()

    def _render(self, ax, veh_length=4.8, veh_width=2.0):
        import matplotlib.patches as pc
        legend_label = ['Ego', 'Global', 'Local']

        # draw ego vehicle
        ego_x, ego_y, phi = self.state[:3]
        x_offset = veh_length / 2 * np.cos(phi) - veh_width / 2 * np.sin(phi)
        y_offset = veh_length / 2 * np.sin(phi) + veh_width / 2 * np.cos(phi)
        ax.add_patch(pc.Rectangle(
            (ego_x - x_offset, ego_y - y_offset),
            veh_length,
            veh_width,
            angle=np.rad2deg(phi),
            facecolor='w',
            edgecolor='r',
            zorder=1
        ))

        # draw reference paths
        # ref_x = []
        # ref_y = []

        # for i in np.arange(1, 60):
        #     ref_x.append(self.ref_traj.compute_x(
        #         self.t + i * self.dt, self.path_num, self.u_num
        #     ))
        #     ref_y.append(self.ref_traj.compute_y(
        #         self.t + i * self.dt, self.path_num, self.u_num
        #     ))
        ref_x = self.ref_points[1:, 0]
        ref_y = self.ref_points[1:, 1]
        ax.plot(ref_x, ref_y, 'b--', lw=1, zorder=2)
        # draw planning paths
        plan_x = []
        plan_y = []

        for i in range(self.pre_horizon):
            plan_x.append(self.state_full[i, 0])
            plan_y.append(self.state_full[i, 1])
        ax.plot(plan_x, plan_y, 'g', lw=1, zorder=2)
        # draw surrounding vehicles
        for i in range(self.dynamic_obstacle_num):
            dynamicx, dynamicy, dynamicphi = self.dynamic_state[i, :3]
            ax.add_patch(pc.Rectangle(
                (dynamicx - self.veh_length / 2, dynamicy - self.veh_width / 2),
                self.veh_length,
                self.veh_width,
                angle=dynamicphi * 180 / np.pi,
                facecolor='w',
                edgecolor='k',
                zorder=1
            ))
            legend_label.append('Dynamic_{}'.format(i))

        # draw static obstacles
        for i_static in range(self.static_obstacle_num):
            static_x, static_y, static_phi = self.static_state[i_static, :3]
            ax.add_patch(pc.Rectangle(
                (static_x - self.veh_length / 2, static_y - self.veh_width / 2),
                self.static_length[i_static],
                self.static_width[i_static],
                angle=static_phi * 180 / np.pi,
                facecolor='gray',
                edgecolor='gray',
                zorder=1
            ))
            legend_label.append('Static Obstacle_{}'.format(i_static))
        ax.legend(legend_label, ncol=2, loc=2)

def env_creator(**kwargs):
    return SimuVeh3dofcontiSurrCstr(**kwargs)

# if __name__ == "__main__":
#     env = env_creator()
#     env.reset()
#     for i in range(100):
#         a = env.action_space.sample()
#         obs, reward, done, info = env.step(a)
#         print(reward)
#         # env.render()
