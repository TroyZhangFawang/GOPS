#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: vehicle 3DOF data environment with surrounding vehicles constraint
#  Update: 2024-12-13, fawang zhang: create environment

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import gym
import numpy as np
from gops.env.env_ocp.pyth_veh3dofcontiplanning import SimuVeh3dofconti, angle_normalize, ego_vehicle_coordinate_transform

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
    veh_length: float = 4.8,
    veh_width: float = 2.0

    def step(self):
        self.x = self.x + self.u * np.cos(self.phi) * self.dt
        self.y = self.y + self.u * np.sin(self.phi) * self.dt
        self.phi = self.phi + self.u * np.tan(self.delta) / self.l * self.dt
        self.phi = angle_normalize(self.phi)

@dataclass
class StaticObstacle:
    obs_id: int  # 障碍物ID
    x: float = 0.0 # 障碍物中心点x坐标
    y: float = 0.0 # 障碍物中心点y坐标
    phi: float = 0.0
    height: float = 0.5  # 障碍物高度
    width: float = 1.8  # 障碍物宽度
    length: float = 1.8  # 障碍物长度

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

class SimuVeh3dofBimodalPlanning(SimuVeh3dofconti):
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
        static_obstacle_num: int = 2,
        d_pre: float = 20.0,  # 离障碍物多少远开始规划
        lateral_sample: float = 8.0,  # 横向采样距离
        forward_sample: float = 12.0,  # 纵向采样距离
        veh_length: float = 4.8,
        veh_width: float = 2.0,
        ground_clearance=0.25,
        wheel_distance=1.8,
        **kwargs: Any,
    ):
        super().__init__(pre_horizon, path_para, u_para, **kwargs)
        ego_obs_dim = 6
        ref_obs_dim = 4
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(ego_obs_dim + ref_obs_dim * pre_horizon + (dynamic_obstacle_num+static_obstacle_num) * 4,),
            dtype=np.float32,
        )
        self.generate_guide = 0
        self.guide_end = np.zeros((3, 4))
        self.guide_time_interval = 0
        self.t_start = 0
        self.t_end = 0.1
        self.dynamic_obstacle_num = dynamic_obstacle_num
        self.static_obstacle_num = static_obstacle_num
        self.dynamic_state = np.zeros((dynamic_obstacle_num, 5), dtype=np.float32)
        self.static_state = np.zeros((static_obstacle_num, 7), dtype=np.float32)
        self.veh_length = veh_length
        self.veh_width = veh_width
        self.wheel_distance = wheel_distance
        self.ground_clearance = ground_clearance
        self.d_pre = d_pre
        self.lateral_sample = lateral_sample
        self.forward_sample = forward_sample
        self.guide_traj = None
        self.info_dict.update(
            {
                "dynamic_state": {"shape": (dynamic_obstacle_num, 5), "dtype": np.float32},
                "static_state": {"shape": (static_obstacle_num, 7), "dtype": np.float32},
                "constraint": {"shape": (dynamic_obstacle_num,), "dtype": np.float32},
                "generate_guide": {"shape": (), "dtype": np.uint8},
                "guide_start": {"shape": (4, ), "dtype": np.float32},
                "guide_end": {"shape": (3, 4), "dtype": np.float32},
                "guide_time_interval": {"shape": (), "dtype": np.float32},
                "t_start": {"shape": (), "dtype": np.float32},
                "t_end": {"shape": (), "dtype": np.float32},
                "state": {"shape": (6, ), "dtype": np.float32},
            }
        )

    def reset(
        self,
        init_state: Optional[Sequence] = None,
        ref_time: Optional[float] = None,
        ref_num: Optional[int] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(init_state, ref_time, ref_num, **kwargs)
        # todo 正式训练前记得把初始化改回随机版本

        if self.path_num == 3:
            # circle path
            dynamic_delta = -np.arctan2(DynamicObstacleData.l, self.ref_traj.ref_trajs[3].r)
        else:
            dynamic_delta = 0.0

        # add dynamic obstacle
        self.dynamic_obss = []
        for i_dynamic in range(self.dynamic_obstacle_num):
            # avoid ego vehicle
            delta_t = self.np_random.uniform(3, 5)
            dynamic_phi = self.ref_traj.compute_phi(self.t + delta_t, self.path_num, self.u_num)
            delta_lon = 0#1.0 * self.np_random.uniform(-1, 1)
            delta_lat = 0#1.0 * self.np_random.uniform(-1, 1)
            dynamic_x = self.ref_traj.compute_x(self.t + delta_t, self.path_num, self.u_num) + delta_lon
            dynamic_y = self.ref_traj.compute_y(self.t + delta_t, self.path_num, self.u_num) + delta_lat
            # print("dynamic pos", [dynamic_x, dynamic_y])
            dynamic_u = np.random.uniform(0, 10, self.dynamic_obstacle_num)
            self.dynamic_obss.append(
                DynamicObstacleData(
                    x=dynamic_x,
                    y=dynamic_y,
                    phi=dynamic_phi,
                    u=dynamic_u[i_dynamic],
                    delta=dynamic_delta,
                    dt=self.dt
                )
            )
        self.static_obss = []
        # add static obstacle
        time = [2, 6]
        for i_static in range(self.static_obstacle_num):
            delta_t = self.np_random.uniform(2, 7) #time[i_static]#
            static_obs_phi = self.ref_traj.compute_phi(self.t + delta_t, self.path_num, self.u_num)
            delta_lon = 0#1.0 * self.np_random.uniform(-1, 1)
            delta_lat = 0#1.0 * self.np_random.uniform(-1, 1)
            static_obs_x = self.ref_traj.compute_x(self.t + delta_t, self.path_num, self.u_num) + delta_lon
            static_obs_y = self.ref_traj.compute_y(self.t + delta_t, self.path_num, self.u_num) + delta_lat
            # print("static pos", [static_obs_x, static_obs_y])
            self.static_length = np.random.uniform(0, 2, self.static_obstacle_num)
            self.static_width = np.array([3, 0.5])#np.random.uniform(0, 2, self.static_obstacle_num)
            self.static_height = np.array([0.5, 0.23])#np.random.uniform(0, 1, self.static_obstacle_num)
            self.static_obss.append(
                StaticObstacle(
                    obs_id=i_static,
                    x=static_obs_x,
                    y=static_obs_y,
                    phi=static_obs_phi,
                    length=self.static_length[i_static],
                    width=self.static_width[i_static],
                    height=self.static_height[i_static],
                )
            )

        self.update_dynamic_state()
        self.update_static_state()

        # 初始时刻添加完障碍物之后就先判断下是否需要生成引导轨迹
        obstacle, generate_guide = self.is_generate_guide()
        if generate_guide and obstacle != None:
            self.generate_guide = generate_guide
            self.obstacle = obstacle
            # quintic_curves = self.generate_quintic_curves(self.obstacle) # 生成3条五次多项式，每条包含2个元素，为横向、纵向位置多项式
            bezier_curves = self.generate_bezier_curves(self.obstacle)  # 生成3条贝塞尔曲线，每条包含2个元素，为横向、纵向位置多项式
            self.guide_traj, can_cross = self.can_cross_decision(self.obstacle, bezier_curves)
            self.t_start = self.t
            self.t_end = self.t + (self.obstacle.x + self.forward_sample - self.state[0]) / self.state[3]
            guide_points = [self.state[:4]]
            for i in range(1, self.pre_horizon + 1):
                ref_x = self.ref_traj.compute_x(self.t + i * self.dt, self.path_num, self.u_num)
                guide_point = self.get_bezier_guide_points(self.guide_traj, self.t + i * self.dt, self.t_start, self.t_end)
                t = (ref_x - guide_point[0]) / self.state[3]
                while guide_point[0] < ref_x and t < self.t_end and t > 0:
                    # guide_point = self.get_guide_points(self.guide_traj, t, self.t_start)
                    guide_point = self.get_bezier_guide_points(self.guide_traj, t, self.t_start, self.t_end)
                    t += self.dt
                # 超出范围，使用全局轨迹点, 若bezier 曲线，范围调到障碍物前方采样点
                if guide_point[0] >= obstacle.x+self.forward_sample:
                    if self.state[0] > obstacle.x + obstacle.length / 2:  # 确保车辆完全通过
                        self.processed_obstacles.add(obstacle.obs_id)
                    # self.processed_obstacles.add(self.obstacle.obs_id)
                    self.generate_guide = 0
                    self.guide_traj = None
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
        _, reward, _, _ = super().step(action)
        for dynamic_veh in self.dynamic_obss:
            dynamic_veh.step()
        self.update_dynamic_state()

        # if self.generate_guide == 1 and self.guide_traj == None:
        #     obstacle, generate_guide = self.is_generate_guide()
        #     if generate_guide and self.guide_traj != None:
        #         self.generate_guide = generate_guide
        #         self.obstacle = obstacle
        #         # quintic_curves = self.generate_quintic_curves(self.obstacle)  # 生成3条五次多项式，每条包含2个元素，为横向、纵向位置多项式
        #         bezier_curves = self.generate_bezier_curves(self.obstacle)  # 生成3条五次多项式，每条包含2个元素，为横向、纵向位置多项式
        #         self.guide_traj, can_cross = self.can_cross_decision(self.obstacle, bezier_curves)
        #         self.t_start = self.t
        #         self.t_end = self.t + (self.obstacle.x + self.forward_sample - self.state[0]) / self.state[3]
        #         for i in range(1, self.pre_horizon + 1):
        #             ref_x = self.ref_traj.compute_x(self.t + i * self.dt, self.path_num, self.u_num)
        #             guide_point = self.get_bezier_guide_points(self.guide_traj, self.t + i * self.dt, self.t_start,
        #                                                        t_end=self.t_end)
        #             t = (ref_x - guide_point[0]) / self.state[3]
        #             while guide_point[0] < ref_x and t < self.t_end and t > 0:
        #                 # guide_point = self.get_guide_points(self.guide_traj, t, self.t_start)
        #                 guide_point = self.get_bezier_guide_points(self.guide_traj, t, self.t_start, self.t_end)
        #                 t += self.dt
        #             # 超出范围，使用全局轨迹点, 若bezier 曲线，范围调到障碍物前方采样点
        #             if guide_point[0] >= self.obstacle.x + self.forward_sample:
        #                 self.generate_guide = 0
        #                 guide_point = np.array([
        #                     self.ref_traj.compute_x(self.t + i * self.dt, self.path_num, self.u_num),
        #                     self.ref_traj.compute_y(self.t + i * self.dt, self.path_num, self.u_num),
        #                     self.ref_traj.compute_phi(self.t + i * self.dt, self.path_num, self.u_num),
        #                     self.ref_traj.compute_u(self.t + i * self.dt, self.path_num, self.u_num),
        #                 ], dtype=np.float32
        #                 )
        #             self.ref_points[i] = guide_point
        #         new_ref_point = guide_point

        if self.generate_guide == 1 and self.guide_traj != None:
            ref_x = self.ref_traj.compute_x(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num)
            # new_ref_point = self.get_quintic_guide_points(self.guide_traj, self.t + self.pre_horizon * self.dt, self.t_start)
            new_ref_point = self.get_bezier_guide_points(self.guide_traj, self.t + self.pre_horizon * self.dt, self.t_start, self.t_end)
            t = (ref_x - new_ref_point[0]) / new_ref_point[3]
            while new_ref_point[0] <= ref_x and t <= self.t_end and t > 0:
                # new_ref_point = self.get_quintic_guide_points(self.guide_traj, t + self.pre_horizon * self.dt, self.t_start)
                new_ref_point = self.get_bezier_guide_points(self.guide_traj, t + self.pre_horizon * self.dt, self.t_start, self.t_end)
                t += self.dt
            if new_ref_point[0] >= self.obstacle.x + self.forward_sample:
                if self.state[0] > self.obstacle.x + self.obstacle.length / 2:  # 确保车辆完全通过
                    self.processed_obstacles.add(self.obstacle.obs_id)
                # self.processed_obstacles.add(self.obstacle.obs_id)
                self.generate_guide = 0
                self.guide_traj = None
                new_ref_point = np.array([
                    self.ref_traj.compute_x(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num),
                    self.ref_traj.compute_y(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num),
                    self.ref_traj.compute_phi(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num),
                    self.ref_traj.compute_u(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num),
                ], dtype=np.float32
                )

        # elif self.generate_guide == 0 and self.guide_traj != None:
        #     new_ref_point = np.array(
        #         [
        #             self.ref_traj.compute_x(
        #                 self.t + self.pre_horizon * self.dt, self.path_num, self.u_num
        #             ),
        #             self.ref_traj.compute_y(
        #                 self.t + self.pre_horizon * self.dt, self.path_num, self.u_num
        #             ),
        #             self.ref_traj.compute_phi(
        #                 self.t + self.pre_horizon * self.dt, self.path_num, self.u_num
        #             ),
        #             self.ref_traj.compute_u(
        #                 self.t + self.pre_horizon * self.dt, self.path_num, self.u_num
        #             ),
        #         ],
        #         dtype=np.float32,
        #     )
        #     if self.state[0] >= self.obstacle.x:
        #         self.guide_traj = None

        else:
            # 判断下是否需要生成引导轨迹
            obstacle, generate_guide = self.is_generate_guide()
            if generate_guide and obstacle != None:
                self.generate_guide = generate_guide
                self.obstacle = obstacle
                # quintic_curves = self.generate_quintic_curves(self.obstacle)  # 生成3条五次多项式，每条包含2个元素，为横向、纵向位置多项式
                bezier_curves = self.generate_bezier_curves(obstacle)  # 生成3条贝塞尔曲线
                self.guide_traj, can_cross = self.can_cross_decision(self.obstacle, bezier_curves)
                self.t_start = self.t
                self.t_end = self.t + (self.obstacle.x + self.forward_sample - self.state[0]) / self.state[3]
                for i in range(1, self.pre_horizon+1):
                    ref_x = self.ref_traj.compute_x(self.t + i * self.dt, self.path_num, self.u_num)

                    # Check if guide_traj is None before proceeding
                    if self.guide_traj is None and self.generate_guide == 0:
                        # Use reference trajectory directly if guide trajectory is not available
                        guide_point = np.array([
                            self.ref_traj.compute_x(self.t + i * self.dt, self.path_num, self.u_num),
                            self.ref_traj.compute_y(self.t + i * self.dt, self.path_num, self.u_num),
                            self.ref_traj.compute_phi(self.t + i * self.dt, self.path_num, self.u_num),
                            self.ref_traj.compute_u(self.t + i * self.dt, self.path_num, self.u_num),
                        ], dtype=np.float32)
                    else:
                        guide_point = self.get_bezier_guide_points(self.guide_traj, self.t + i * self.dt, self.t_start,
                                                                   t_end=self.t_end)

                        t = (ref_x - guide_point[0]) / self.state[3]
                        while guide_point[0] < ref_x and t < self.t_end and t > 0:
                            # guide_point = self.get_guide_points(self.guide_traj, t, self.t_start)
                            guide_point = self.get_bezier_guide_points(self.guide_traj, t, self.t_start, self.t_end)
                            t += self.dt
                        # 超出范围，使用全局轨迹点, 若bezier 曲线，范围调到障碍物前方采样点
                        if guide_point[0] >= self.obstacle.x + self.forward_sample:
                            if self.state[0] > self.obstacle.x + self.obstacle.length / 2:  # 确保车辆完全通过
                                self.processed_obstacles.add(self.obstacle.obs_id)
                            # self.processed_obstacles.add(self.obstacle.obs_id)
                            self.generate_guide = 0
                            self.guide_traj = None
                            guide_point = np.array([
                                self.ref_traj.compute_x(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num),
                                self.ref_traj.compute_y(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num),
                                self.ref_traj.compute_phi(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num),
                                self.ref_traj.compute_u(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num),
                            ], dtype=np.float32
                            )
                    self.ref_points[i] = guide_point
                new_ref_point = guide_point
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
        self.update_dynamic_state()
        self.update_static_state()
        done = self.judge_done()
        return self.get_obs(), reward, done, self.info

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
        # delta_x, delta_y, delta_phi, delta_u, (of the first reference point)
        # v, w (of ego vehicle)
        # ]
        ego_obs = np.concatenate(
            ([ref_x_tf[0], ref_y_tf[0], ref_phi_tf[0], ref_u_tf[0]], self.state[4:]))
        # ref_obs: [
        # delta_x, delta_y, delta_phi, delta_u (of the second to last reference point)
        # ]
        ref_obs = np.stack((ref_x_tf, ref_y_tf, ref_phi_tf, ref_u_tf), 1)[1:].flatten()
        obs = np.concatenate((ego_obs, ref_obs))

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
                self.static_state[:, 1], self.static_state[:, 2], self.static_state[:, 3],
            )
        static_u_tf = np.zeros((self.static_obstacle_num, )) - self.state[3]
        static_obs = np.concatenate((static_x_tf, static_y_tf, static_phi_tf, static_u_tf))
        return np.concatenate((obs, dynamic_obs, static_obs))

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
        for i, dynamic_veh in enumerate(self.dynamic_obss):
            self.dynamic_state[i] = np.array(
                [dynamic_veh.x, dynamic_veh.y, dynamic_veh.phi, dynamic_veh.u, dynamic_veh.delta],
                dtype=np.float32,
            )

    def update_static_state(self):
        for i, static_obs in enumerate(self.static_obss):
            self.static_state[i] = np.array(
                [static_obs.obs_id, static_obs.x, static_obs.y, static_obs.phi, static_obs.length, static_obs.width, static_obs.height],
                dtype=np.float32,
            )

    def find_nearest_obstacle(self,
                              current_pos: np.ndarray,
                              static_obstacles) -> Tuple[Optional[int], float]:
        """查找最近的障碍物"""
        if not static_obstacles:
            return None, float('inf')

        min_dist = float('inf')
        nearest_id = None
        for obs_id, static_obstacle in enumerate(static_obstacles):
            # 计算到障碍物的距离
            obs_pos = np.array([static_obstacle.x, static_obstacle.y])
            dist = np.linalg.norm(current_pos - obs_pos)
            # 只考虑前方的障碍物
            if obs_pos[0] > current_pos[0] and dist < min_dist:
                min_dist = dist
                nearest_id = obs_id
        return nearest_id, min_dist

    # def is_generate_guide(self) -> Tuple[Optional[StaticObstacle], int]:
    #     current_pos = self.state[:2]
    #
    #     nearest_static_id, static_dist = self.find_nearest_obstacle(
    #         current_pos, self.static_obss)
    #
    #     if static_dist <= self.d_pre:
    #         obstacle = self.static_obss[nearest_static_id]
    #         return obstacle, 1
    #     else:
    #         return None, 0
    def is_generate_guide(self) -> Tuple[Optional[StaticObstacle], int]:
        current_pos = self.state[:2]
        obstacles_in_range = []

        # 找出所有在规划范围内的障碍物
        for obstacle in self.static_obss:
            obs_pos = np.array([obstacle.x, obstacle.y])
            dist = np.linalg.norm(current_pos - obs_pos)
            # 只考虑前方的障碍物且在规划范围内
            if obs_pos[0] > current_pos[0] and dist <= self.d_pre:
                obstacles_in_range.append((obstacle, dist))

        if not obstacles_in_range:
            return None, 0

        # 按距离排序，选择最近的障碍物
        obstacles_in_range.sort(key=lambda x: x[1])
        nearest_obstacle = obstacles_in_range[0][0]

        # 检查是否已经处理过这个障碍物
        if hasattr(self, 'processed_obstacles'):
            if nearest_obstacle.obs_id in self.processed_obstacles:
                # 如果已经处理过，检查是否还有其他障碍物需要处理
                for obstacle, _ in obstacles_in_range[1:]:
                    if obstacle.obs_id not in self.processed_obstacles:
                        return obstacle, 1
                return None, 0
        else:
            self.processed_obstacles = set()

        return nearest_obstacle, 1

    def can_cross_decision(self, obstacle: StaticObstacle, curves: List) -> Tuple[Optional[BezierCurve], bool]:
        """跨/绕决策
        Args:
            obstacle: 障碍物信息
        Returns:
            can_cross: 是否可以跨越
        """

        # 判断是否满足跨越条件
        can_cross = (obstacle.height < self.ground_clearance and
                     obstacle.width < self.wheel_distance)

        self.curve_index = None
        if can_cross:
            best_curve = curves[1]  # 中间曲线用于跨越
            self.curve_index = 1
        else:
            # 不可跨越时选择绕行曲线
            best_curve = curves[2]  # 默认选择左绕
            self.curve_index = 2

        return best_curve, can_cross

    def generate_bezier_curves(self, obstacle: StaticObstacle) -> List[BezierCurve]:
        """生成贝塞尔曲线
        Args:
            current_pos: 当前位置[x, y]
            obstacle: 障碍物信息
        Returns:
            curves: 生成的贝塞尔曲线列表
        """
        curves = []
        current_pos = self.state[:2]
        # 横向采样点
        lateral_offsets = [-self.lateral_sample-obstacle.width/2-self.veh_width/2, 0, self.lateral_sample+obstacle.width/2+self.veh_width/2]
        lateral_points = [
            np.array([obstacle.x, obstacle.y + offset]) for offset in lateral_offsets
        ]

        # 终点位置（统一使用前方采样点）
        end_point = np.array([obstacle.x + self.forward_sample+obstacle.length*1.5, obstacle.y])

        # 生成三条候选曲线
        for control_point in lateral_points:
            curve = BezierCurve(current_pos, control_point, end_point)
            curves.append(curve)
        return curves

    # def generate_quintic_curves(self,
    #                            obstacle: StaticObstacle) -> List:
    #     """生成五次多项式曲线
    #             Args:
    #                 current_pos: 当前位置[x, y]
    #                 obstacle: 障碍物信息
    #             Returns:
    #                 curves: 生成的五次多项式曲线列表
    #             """
    #     def _create_single_trajectory(start_point, end_point, T) -> Tuple[QuinticPolynomial, QuinticPolynomial]:
    #         """创建单条轨迹的x和y方向多项式"""
    #         # x方向多项式
    #         poly_x = QuinticPolynomial(
    #             [start_point[0], start_point[3] * np.cos(start_point[2]), 0], # 位置、速度、加速度
    #             [end_point[0], end_point[3] * np.cos(end_point[2]), 0],# 位置、速度、加速度
    #             T
    #         )
    #         # y方向多项式
    #         poly_y = QuinticPolynomial(
    #             [start_point[1], start_point[3] * np.sin(start_point[2]), 0],
    #             [end_point[1], end_point[3] * np.sin(end_point[2]), 0],
    #             T
    #         )
    #         return (poly_x, poly_y)
    #
    #     curves = []
    #     # 横向采样点
    #     lateral_points = [
    #         np.array([obstacle.x, obstacle.y - self.lateral_sample, obstacle.phi, self.state[3]]),  # 左侧
    #         np.array([obstacle.x, obstacle.y, obstacle.phi, self.state[3]]),  # 中心
    #         np.array([obstacle.x, obstacle.y + self.lateral_sample, obstacle.phi, self.state[3]])  # 右侧
    #     ]
    #     # 生成五次多项式曲线
    #     for i in range(len(lateral_points)):
    #         curve = _create_single_trajectory(self.state, lateral_points[i], abs((obstacle.x-self.state[0])/self.state[3]))
    #         # 使用当前位置作为起点，障碍物位置作为控制点
    #         curves.append(curve)
    #     self.guide_end = lateral_points
    #     self.guide_time_interval = abs((obstacle.x-self.state[0])/self.state[3])
    #     return curves
    #
    # def get_quintic_guide_points(self, guide_traj, current_t: float,
    #                           t_start: float) -> np.array:
    #     """
    #     获取指定轨迹的预测点
    #
    #     Args:
    #         current_t: 当前时间
    #         t_start: 当前曲线的起点时间
    #
    #     Returns:
    #         np.ndarray: 预测轨迹点 shape=(1, 4) [x, y, phi, v]
    #     """
    #     poly_x, poly_y = guide_traj
    #     local_t = current_t - t_start
    #     x_state = poly_x.compute_point(local_t)
    #     y_state = poly_y.compute_point(local_t)
    #     x, y = x_state[0], y_state[0]
    #     vx, vy = x_state[1], y_state[1]
    #     phi = np.arctan2(vy, vx)
    #     return np.array([x, y, phi, vx], dtype=np.float32)

    # def get_bezier_guide_points(self, guide_traj, t_interpolate: float, t_start: float, t_end:float):
    #     if t_end <= t_start:
    #         # print("please check the time end and time start data")
    #         t_end = t_start + self.dt
    #     # 计算归一化时间参数
    #     t = np.clip((t_interpolate - t_start) / (t_end - t_start), 0, 1)
    #     # 计算位置和导数
    #     point = guide_traj.compute_point(t)
    #     derivative = guide_traj.compute_derivative(t)
    #     # 计算航向角和速度
    #     phi = np.arctan2(derivative[1], derivative[0])
    #     u = self.ref_traj.compute_u(t_interpolate, self.path_num, self.u_num)
    #     traj_point = np.array([point[0], point[1], phi, u], dtype=np.float32)
    #     return traj_point
    def get_bezier_guide_points(self, guide_traj, t_interpolate: float, t_start: float, t_end: float):
        if t_end <= t_start:
            t_end = t_start + self.dt

        # 确保时间参数在有效范围内
        t = np.clip((t_interpolate - t_start) / (t_end - t_start), 0, 1)

        point = guide_traj.compute_point(t)
        derivative = guide_traj.compute_derivative(t)

        # 确保导数不为零向量
        if np.linalg.norm(derivative) < 1e-6:
            derivative = np.array([1e-6, 0])  # 小量向前

        phi = np.arctan2(derivative[1], derivative[0])
        global_phi = self.ref_traj.compute_phi(t_interpolate, self.path_num, self.u_num)
        blend_ratio = np.clip(t, 0.2, 0.8)  # 渐进混合系数
        phi = angle_normalize(blend_ratio * phi + (1 - blend_ratio) * global_phi)

        u = self.ref_traj.compute_u(t_interpolate, self.path_num, self.u_num)

        # 确保速度方向与航向一致
        if u < 0:
            phi = angle_normalize(phi + np.pi)
            u = abs(u)

        traj_point = np.array([point[0], point[1], phi, u], dtype=np.float32)
        return traj_point

    def update_loca_traj(self, local_traj):
        self.planned_traj = local_traj

    @property
    def info(self):
        info = super().info
        info.update({
            "constraint": self.get_constraint(),
            "dynamic_state": self.dynamic_state.copy(),
            "static_state": self.static_state.copy(),
            "generate_guide": self.generate_guide,
            "guide_start": self.state[:4],
            "guide_end": self.guide_end,
            "guide_time_interval": self.guide_time_interval,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "location":self.state
        })
        return info

    # def _render(self, ax):
    #     super()._render(ax, self.veh_length, self.veh_width)
    #     import matplotlib.patches as pc
    #     legend_label = ['Ego', 'Global', 'Local']
    #     # draw reference paths
    #     # ref_x = []
    #     # ref_y = []
    #
    #     # for i in np.arange(1, 60):
    #     #     ref_x.append(self.ref_traj.compute_x(
    #     #         self.t + i * self.dt, self.path_num, self.u_num
    #     #     ))
    #     #     ref_y.append(self.ref_traj.compute_y(
    #     #         self.t + i * self.dt, self.path_num, self.u_num
    #     #     ))
    #     # ref_x = self.ref_points[1:, 0]
    #     # ref_y = self.ref_points[1:, 1]
    #     # ax.plot(ref_x, ref_y, 'b--', lw=1, zorder=2)
    #     # # draw planning paths
    #     # plan_x = []
    #     # plan_y = []
    #     #
    #     # for i in range(self.pre_horizon):
    #     #     plan_x.append(self.state_full[i, 0])
    #     #     plan_y.append(self.state_full[i, 1])
    #     # ax.plot(plan_x, plan_y, 'g', lw=1, zorder=2)
    #     # draw surrounding vehicles
    #     for i in range(self.dynamic_obstacle_num):
    #         dynamicx, dynamicy, dynamicphi = self.dynamic_state[i, :3]
    #         ax.add_patch(pc.Rectangle(
    #             (dynamicx - self.veh_length / 2, dynamicy - self.veh_width / 2),
    #             self.veh_length,
    #             self.veh_width,
    #             angle=dynamicphi * 180 / np.pi,
    #             facecolor='w',
    #             edgecolor='k',
    #             zorder=1
    #         ))
    #         legend_label.append('Dynamic_{}'.format(i))
    #
    #     # draw static obstacles
    #     for i_static in range(self.static_obstacle_num):
    #         static_x, static_y, static_phi = self.static_state[i_static, 1:4]
    #         ax.add_patch(pc.Rectangle(
    #             (static_x - self.static_length[i_static] / 2, static_y - self.static_width[i_static] / 2),
    #             self.static_length[i_static],
    #             self.static_width[i_static],
    #             angle=static_phi * 180 / np.pi,
    #             facecolor='gray',
    #             edgecolor='gray',
    #             zorder=1
    #         ))
    #         legend_label.append('Static Obstacle_{}'.format(i_static))
    #     ax.legend(legend_label, ncol=2, loc=2)
    def _render(self, ax):
        super()._render(ax, self.veh_length, self.veh_width)
        import matplotlib.patches as pc
        legend_label = ['Ego', 'Global', 'Local']

        # 新增绘制逻辑 -------------------------------------------------
        if hasattr(self, 'guide_traj') and self.guide_traj is not None:
            # 绘制贝塞尔曲线
            t_values = np.linspace(0, 1, 20)
            curve_points = [self.guide_traj.compute_point(t) for t in t_values]
            x_coords = [p[0] for p in curve_points]
            y_coords = [p[1] for p in curve_points]
            ax.plot(x_coords, y_coords, 'c--', linewidth=1.5, zorder=3, label='Bezier Trajectory')

            # 绘制控制点
            control_points = [
                self.guide_traj.p0,
                self.guide_traj.p1,
                self.guide_traj.p2
            ]
            colors = ['ro', 'go', 'bo']  # 红:起点, 绿:控制点, 蓝:终点
            labels = ['Start Point', 'Control Point', 'End Point']

            for idx, (point, color, label) in enumerate(zip(control_points, colors, labels)):
                ax.plot(point[0], point[1], color, markersize=8, zorder=4)
                ax.text(point[0] + 0.5, point[1] + 0.5, label, fontsize=8, color=color[0])

            # 连接控制点辅助线
            ax.plot([self.guide_traj.p0[0], self.guide_traj.p1[0]],
                    [self.guide_traj.p0[1], self.guide_traj.p1[1]],
                    'g:', linewidth=0.8, zorder=2)
            ax.plot([self.guide_traj.p1[0], self.guide_traj.p2[0]],
                    [self.guide_traj.p1[1], self.guide_traj.p2[1]],
                    'b:', linewidth=0.8, zorder=2)
            legend_label.extend(['Bezier Curve', 'Control Lines'])
        # ------------------------------------------------------------

        # 原有障碍物绘制逻辑
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
            legend_label.append(f'Dynamic_{i}')

        # 绘制静态障碍物（原有逻辑保持不变）
        for i_static in range(self.static_obstacle_num):
            static_x = self.static_state[i_static, 1]
            static_y = self.static_state[i_static, 2]
            static_phi = self.static_state[i_static, 3]
            static_length = self.static_state[i_static, 4]
            static_width = self.static_state[i_static, 5]

            ax.add_patch(pc.Rectangle(
                (static_x - static_length / 2, static_y - static_width / 2),
                static_length,
                static_width,
                angle=static_phi * 180 / np.pi,
                facecolor='gray',
                edgecolor='gray',
                zorder=1
            ))
            legend_label.append(f'Static_{i_static}')

        # 更新图例
        ax.legend(legend_label, ncol=3, loc='upper left', fontsize=6)
def env_creator(**kwargs):
    return SimuVeh3dofBimodalPlanning(**kwargs)

# if __name__ == "__main__":
#     env = env_creator()
#     env.reset()
#     for i in range(100):
#         a = env.action_space.sample()
#         obs, reward, done, info = env.step(a)
#         print(reward)
#         # env.render()
