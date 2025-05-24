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
from gops.utils.planner_benchmark.elements.map import RoutedLocalMap, Lane
from gops.utils.planner_benchmark.elements.box import TrackingBoxList, TrackingBox
from gops.utils.planner_benchmark.elements.vehicle import VehicleState
import numpy as np

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
        dynamic_obstacle_num: int = 2,
        static_obstacle_num: int = 3,
        d_pre: float = 20.0,  # 离障碍物多少远开始规划
        lateral_sample: float = 3.5,  # 横向采样距离
        forward_sample: float = 10.0,  # 纵向采样距离
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
        self.veh_width = self.vehicle_dynamics.vehicle_params["veh_width"]
        self.veh_length = self.vehicle_dynamics.vehicle_params["veh_length"]
        self.wheel_distance = self.vehicle_dynamics.vehicle_params["wheel_distance"]
        self.ground_clearance = self.vehicle_dynamics.vehicle_params["ground_clearance"]
        self.d_pre = d_pre
        self.lateral_sample = lateral_sample
        self.forward_sample = forward_sample
        self.best_curve = None
        self.obstacle = None
        self.last_curve_index = None
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
                "best_curve_isnone": {"shape": (), "dtype": bool},
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
        self.obstacle_trackingbox = []
        dynamic_time = [5, 5]
        for i_dynamic in range(self.dynamic_obstacle_num):
            # avoid ego vehicle
            delta_t = dynamic_time[i_dynamic]#self.np_random.uniform(3, 5)
            dynamic_phi = np.array([np.pi*5/4, np.pi/4])#self.ref_traj.compute_phi(self.t + delta_t, self.path_num, self.u_num)
            delta_lon = 0#1.0 * self.np_random.uniform(-1, 1)
            delta_lat = np.array([-3.5, 3.5])#1.0 * self.np_random.uniform(-1, 1)
            dynamic_x = self.ref_traj.compute_x(self.t + delta_t, self.path_num, self.u_num) + delta_lon
            dynamic_y = 0#self.ref_traj.compute_y(self.t + delta_t, self.path_num, self.u_num)
            dynamic_u = np.array([-4, -4])#np.random.uniform(0, 10, self.dynamic_obstacle_num)
            self.dynamic_obss.append(
                DynamicObstacleData(
                    x=dynamic_x,
                    y=dynamic_y + delta_lat[i_dynamic],
                    phi=dynamic_phi[i_dynamic],
                    u=dynamic_u[i_dynamic],
                    delta=dynamic_delta,
                    dt=self.dt
                )
            )
            obstacle = TrackingBox(obb=(dynamic_x, dynamic_y+ delta_lat[i_dynamic], self.veh_length, self.veh_width, dynamic_phi[i_dynamic], 2), vx=dynamic_u[i_dynamic], vy=0.0, id=i_dynamic)
            self.obstacle_trackingbox.append(obstacle)
        self.static_obss = []
        # add static obstacle
        static_time = [3, 7, 9]
        for i_static in range(self.static_obstacle_num):
            delta_t = static_time[i_static]#self.np_random.uniform(2, 7)
            static_obs_phi = self.ref_traj.compute_phi(self.t + delta_t, self.path_num, self.u_num)
            delta_lon = 0#1.0 * self.np_random.uniform(-1, 1)
            delta_lat = np.array([0., 0, 0])#1.0 * self.np_random.uniform(-1, 1)
            static_obs_x = self.ref_traj.compute_x(self.t + delta_t, self.path_num, self.u_num) + delta_lon
            static_obs_y = 0#self.ref_traj.compute_y(self.t + delta_t, self.path_num, self.u_num)
            self.static_length = np.array([0.5, 3, 5])#np.random.uniform(0, 2, self.static_obstacle_num)
            self.static_width = np.array([1.5, 2.0, 1.3]) #np.random.uniform(0, 2, self.static_obstacle_num)#, 0.5
            self.static_height = np.array([0.2, 0.5, 0.1]) #np.random.uniform(0, 1, self.static_obstacle_num)#, 0.23
            self.static_obss.append(
                StaticObstacle(
                    obs_id=i_static,
                    x=static_obs_x,
                    y=static_obs_y+ delta_lat[i_static],
                    phi=static_obs_phi,
                    length=self.static_length[i_static],
                    width=self.static_width[i_static],
                    height=self.static_height[i_static],
                )
            )
            obstacle = TrackingBox(obb=(static_obs_x, static_obs_y+ delta_lat[i_static], self.static_length[i_static], self.static_width[i_static], static_obs_phi, self.static_height[i_static]),
                                   vx=0, vy=0.0, id=i_static)
            self.obstacle_trackingbox.append(obstacle)

        self.update_dynamic_state()
        self.update_static_state()
        self.ego_veh_state = VehicleState.from_kine_states(self.state[0], self.state[1], self.state[2], vx=self.state[3], vy=self.state[4],
                                                       length=self.veh_length, width=self.veh_width)
        self.local_map = RoutedLocalMap()
        for idx, yy in enumerate([-3.5, 0, 3.5]):
            xs = []
            ys = []
            for t in range(1, self.max_episode_steps, ):
                x = self.ref_traj.compute_x(t*self.dt, self.path_num, self.u_num)
                y = self.ref_traj.compute_y(t*self.dt, self.path_num, self.u_num)
                xs.append(x)
                ys.append(y)
            center_line = np.column_stack((np.array(xs), np.array(ys)+yy))  # 中心线的x y
            lane = Lane(idx, center_line, width=3.5, speed_limit=80 / 3.6)
            self.local_map.lanes.append(lane)
        self.obstaclesBox = TrackingBoxList(self.obstacle_trackingbox)

        # 初始时刻添加完障碍物之后就先判断下是否需要生成引导轨迹
        obstacle, generate_guide = self.is_generate_guide()
        if generate_guide and obstacle != None:
            self.generate_guide = generate_guide
            self.obstacle = obstacle
            # quintic_curves = self.generate_quintic_curves(self.obstacle) # 生成3条五次多项式，每条包含2个元素，为横向、纵向位置多项式
            bezier_curves = self.generate_bezier_curves(self.obstacle)  # 生成3条贝塞尔曲线，每条包含2个元素，为横向、纵向位置多项式
            self.best_curve, can_cross = self.can_cross_decision(self.obstacle, bezier_curves)
            self.t_start = self.t
            self.t_end = self.t + (self.obstacle.x + self.forward_sample - self.state[0]) / self.state[3]
            guide_points = [self.state[:4]]
            for i in range(1, self.pre_horizon + 1):
                ref_x = self.ref_traj.compute_x(self.t + i * self.dt, self.path_num, self.u_num)
                if self.best_curve is None:
                    # Use reference trajectory directly if guide trajectory is not available
                    guide_point = np.array([
                        self.ref_traj.compute_x(self.t + i * self.dt, self.path_num, self.u_num),
                        self.ref_traj.compute_y(self.t + i * self.dt, self.path_num, self.u_num),
                        self.ref_traj.compute_phi(self.t + i * self.dt, self.path_num, self.u_num),
                        self.ref_traj.compute_u(self.t + i * self.dt, self.path_num, self.u_num),
                    ], dtype=np.float32)
                else:
                    guide_point = self.get_bezier_guide_points(self.best_curve, self.t + i * self.dt, self.t_start, self.t_end)
                    t_gap = (ref_x - guide_point[0]) / self.state[3]
                    while guide_point[0] < ref_x and t_gap < self.t_end and t_gap > 0:
                        # guide_point = self.get_guide_points(self.best_curve, t, self.t_start)
                        guide_point = self.get_bezier_guide_points(self.best_curve, self.t+t_gap, self.t_start, self.t_end)
                        t_gap += self.dt
                    # 超出范围，使用全局轨迹点, 若bezier 曲线，范围调到障碍物前方采样点
                    if guide_point[0] >= obstacle.x+self.forward_sample:
                        if self.state[0] > obstacle.x:  # 确保车辆完全通过
                            self.processed_obstacles.add(obstacle.obs_id)
                            self.generate_guide = 0
                        self.best_curve = None
                        guide_point = np.array([
                            self.ref_traj.compute_x(self.t + i * self.dt, self.path_num, self.u_num),
                            self.ref_traj.compute_y(self.t + i * self.dt, self.path_num, self.u_num),
                            self.ref_traj.compute_phi(self.t + i * self.dt, self.path_num, self.u_num),
                            self.ref_traj.compute_u(self.t + i * self.dt, self.path_num, self.u_num),
                        ], dtype=np.float32)
                guide_points.append(guide_point)
            self.ref_points = np.array(guide_points, dtype=np.float32)
        self.state_full = np.zeros((self.pre_horizon, self.state_dim))
        return self.get_obs(), self.info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        _, reward, _, _ = super().step(action)
        for dynamic_veh in self.dynamic_obss:
            dynamic_veh.step()
        self.update_dynamic_state()
        for tb in self.obstaclesBox:
            if tb.vx != 0:
                tb.set_obb([tb.x + tb.vx *np.cos(tb.box_heading) * self.dt, tb.y + tb.vx *np.sin(tb.box_heading)* self.dt, tb.length, tb.width, tb.box_heading, tb.height])
        if self.obstacle != None:
            if self.generate_guide == 1 and self.best_curve == None:
                new_ref_point = np.array([
                                    self.ref_traj.compute_x(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num),
                                    self.ref_traj.compute_y(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num),
                                    self.ref_traj.compute_phi(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num),
                                    self.ref_traj.compute_u(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num),
                                ], dtype=np.float32
                                )

            #     obstacle, generate_guide = self.is_generate_guide()
            #     if generate_guide and self.best_curve != None:
            #         self.generate_guide = generate_guide
            #         self.obstacle = obstacle
            #         # quintic_curves = self.generate_quintic_curves(self.obstacle)  # 生成3条五次多项式，每条包含2个元素，为横向、纵向位置多项式
            #         bezier_curves = self.generate_bezier_curves(self.obstacle)  # 生成3条五次多项式，每条包含2个元素，为横向、纵向位置多项式
            #         self.best_curve, can_cross = self.can_cross_decision(self.obstacle, bezier_curves)
            #         self.t_start = self.t
            #         self.t_end = self.t + (self.obstacle.x + self.forward_sample - self.state[0]) / self.state[3]
            #         for i in range(1, self.pre_horizon + 1):
            #             ref_x = self.ref_traj.compute_x(self.t + i * self.dt, self.path_num, self.u_num)
            #             guide_point = self.get_bezier_guide_points(self.best_curve, self.t + i * self.dt, self.t_start,
            #                                                        t_end=self.t_end)
            #             t = (ref_x - guide_point[0]) / self.state[3]
            #             while guide_point[0] < ref_x and t < self.t_end and t > 0:
            #                 # guide_point = self.get_guide_points(self.best_curve, t, self.t_start)
            #                 guide_point = self.get_bezier_guide_points(self.best_curve, t, self.t_start, self.t_end)
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

            elif self.generate_guide == 1 and self.best_curve != None:
                # ref_x = self.ref_traj.compute_x(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num)
                # new_ref_point = self.get_quintic_guide_points(self.best_curve, self.t + self.pre_horizon * self.dt, self.t_start)
                new_ref_point = self.get_bezier_guide_points(self.best_curve, self.t + self.pre_horizon * self.dt, self.t_start, self.t_end)
                # t_gap = (ref_x - new_ref_point[0]) / new_ref_point[3]
                # if t_gap > self.t_end and t_gap > 0:
                #     new_ref_point = np.array([
                #         self.ref_traj.compute_x(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num),
                #         self.ref_traj.compute_y(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num),
                #         self.ref_traj.compute_phi(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num),
                #         self.ref_traj.compute_u(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num),
                #     ], dtype=np.float32
                #     )
                # else:
                #     while new_ref_point[0] <= ref_x and t_gap <= self.t_end and t_gap > 0:
                #         # new_ref_point = self.get_quintic_guide_points(self.best_curve, t + self.pre_horizon * self.dt, self.t_start)
                #         new_ref_point = self.get_bezier_guide_points(self.best_curve, t_gap + self.pre_horizon * self.dt, self.t_start, self.t_end)
                #         t_gap += self.dt
                #     if new_ref_point[0] >= self.obstacle.x + self.forward_sample:
                #         if self.state[0] > self.obstacle.x + self.obstacle.length / 2:  # 确保车辆完全通过
                #             self.processed_obstacles.add(self.obstacle.obs_id)
                #             self.generate_guide = 0
                #             self.best_curve = None
                #         new_ref_point = np.array([
                #             self.ref_traj.compute_x(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num),
                #             self.ref_traj.compute_y(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num),
                #             self.ref_traj.compute_phi(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num),
                #             self.ref_traj.compute_u(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num),
                #         ], dtype=np.float32
                #         )

            if self.state[0] > self.obstacle.x:  # 确保车辆完全通过
                self.processed_obstacles.add(self.obstacle.obs_id)
                self.generate_guide = 0
                self.obstacle = None
        else:
            # 判断下是否需要生成引导轨迹
            obstacle, generate_guide = self.is_generate_guide()
            if generate_guide and obstacle != None:
                self.generate_guide = generate_guide
                self.obstacle = obstacle
                # quintic_curves = self.generate_quintic_curves(self.obstacle)  # 生成3条五次多项式，每条包含2个元素，为横向、纵向位置多项式
                bezier_curves = self.generate_bezier_curves(obstacle)  # 生成3条贝塞尔曲线
                self.best_curve, can_cross = self.can_cross_decision(self.obstacle, bezier_curves)
                self.t_start = self.t
                self.t_end = self.t + (self.obstacle.x + self.forward_sample+obstacle.width/2 - self.state[0]) / self.state[3]
                for i in range(1, self.pre_horizon+1):
                    ref_x = self.ref_traj.compute_x(self.t + i * self.dt, self.path_num, self.u_num)
                    # # Check if guide_traj is None before proceeding
                    if self.best_curve is None:
                        # Use reference trajectory directly if guide trajectory is not available
                        guide_point = np.array([
                            self.ref_traj.compute_x(self.t + i * self.dt, self.path_num, self.u_num),
                            self.ref_traj.compute_y(self.t + i * self.dt, self.path_num, self.u_num),
                            self.ref_traj.compute_phi(self.t + i * self.dt, self.path_num, self.u_num),
                            self.ref_traj.compute_u(self.t + i * self.dt, self.path_num, self.u_num),
                        ], dtype=np.float32)
                    else:
                        guide_point = self.get_bezier_guide_points(self.best_curve, self.t + i * self.dt, self.t_start,self.t_end)
                        t_gap = (ref_x - guide_point[0]) / self.state[3]
                        while guide_point[0] < ref_x and t_gap < self.t_end and t_gap > 0:
                            # guide_point = self.get_guide_points(self.best_curve, t, self.t_start)
                            guide_point = self.get_bezier_guide_points(self.best_curve, self.t+t_gap, self.t_start, self.t_end)
                            t_gap += self.dt
                        # 超出范围，使用全局轨迹点, 若bezier 曲线，范围调到障碍物前方采样点
                        if guide_point[0] >= self.obstacle.x + self.forward_sample:
                            if self.state[0] > obstacle.x:  # 确保车辆完全通过
                                self.processed_obstacles.add(obstacle.obs_id)
                                self.generate_guide = 0
                                self.obstacle = None
                            # self.processed_obstacles.add(self.obstacle.obs_id)
                            self.best_curve = None
                            guide_point = np.array([
                                ref_x,
                                self.ref_traj.compute_y(self.t + i * self.dt, self.path_num, self.u_num),
                                self.ref_traj.compute_phi(self.t + i * self.dt, self.path_num, self.u_num),
                                self.ref_traj.compute_u(self.t + i * self.dt, self.path_num, self.u_num),
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
        self.ego_veh_state = VehicleState.from_kine_states(self.state[0], self.state[1], self.state[2], vx=self.state[3], vy=self.state[4],
                                                       length=self.veh_length, width=self.veh_width)
        # done = False
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
        # if np.any(dis < 0):
        #     print("cillision with dynamic obstacle")
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


        min_dist = np.inf * np.ones(self.dynamic_obstacle_num, dtype=np.float32)
        for i in range(2):
            # front and rear circle of ego vehicle
            for j in range(2):
                # front and rear circle of dynamic vehicles
                dist = np.linalg.norm(
                    ego_center[np.newaxis, i] - dynamic_center[:, j], axis=1
                )
                min_dist = np.minimum(min_dist, dist)

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

    # def can_cross_decision(self, obstacle: StaticObstacle, curves: List) -> Tuple[Optional[BezierCurve], bool]:
    #     """跨/绕决策
    #     Args:
    #         obstacle: 障碍物信息
    #     Returns:
    #         can_cross: 是否可以跨越
    #     """
    #
    #     # 判断是否满足跨越条件
    #     can_cross = (obstacle.height < self.ground_clearance and
    #                  obstacle.width < self.wheel_distance)
    #     # todo 待增加一个规则的决策，如果绕过一个动/静态，再跨过一个静态，再绕过一个动/静态的场景，
    #     #  那么静态障碍物可以直接忽略，保证轨迹的平顺性,问题是如何判断未来行为是不是绕行
    #     self.curve_index = None
    #     if can_cross:
    #         best_curve = curves[1]  # 中间曲线用于跨越
    #         self.curve_index = 1
    #     else:
    #         # 不可跨越时选择绕行曲线
    #         best_curve = curves[2]  # 默认选择右绕
    #         self.curve_index = 2
    #     return best_curve, can_cross

    # def can_cross_decision(self, obstacle: StaticObstacle, curves: List) -> Tuple[Optional[BezierCurve], bool]:
    #     can_cross = (obstacle.height < self.ground_clearance and
    #                  obstacle.width < self.wheel_distance)
    #
    #     # 评估曲线平滑性
    #     def evaluate_curve(curve):
    #         # 采样曲线上的点
    #         ts = np.linspace(0, 1, 10)
    #         points = np.array([curve.compute_point(t) for t in ts])
    #
    #         # 计算曲率变化
    #         dx = np.gradient(points[:, 0])
    #         dy = np.gradient(points[:, 1])
    #         ddx = np.gradient(dx)
    #         ddy = np.gradient(dy)
    #         curvature = np.abs(dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** 1.5
    #
    #         return np.mean(curvature)  # 返回平均曲率
    #
    #     # 选择最平滑的曲线
    #     if can_cross:
    #         best_curve = min(curves, key=evaluate_curve)
    #     else:
    #         # 选择绕行时，优先选择与当前方向更一致的曲线
    #         current_heading = self.state[2]
    #         heading_diffs = []
    #         valid_indices = [0, 2]  # 对应的原始索引
    #         valid_curves = [curves[0], curves[2]]
    #         for curve in valid_curves:
    #             end_heading = np.arctan2(curve.p2[1] - curve.p1[1], curve.p2[0] - curve.p1[0])
    #             heading_diffs.append(abs(angle_normalize(end_heading - current_heading)))
    #         # 选择差异最小的曲线
    #         best_sub_idx = np.argmin(heading_diffs)
    #         best_curve = valid_curves[best_sub_idx]#
    #         # 额外检查：如果选择的曲线与障碍物太近，选择另一条
    #         # if self._is_too_close_to_obstacle(best_curve, obstacle):
    #         #     # 选择另一条曲线
    #         #     self.curve_index = valid_indices[1 - best_sub_idx]
    #         #     best_curve = valid_curves[1 - best_sub_idx]
    #     return best_curve, can_cross
    def can_cross_decision(self, obstacle: StaticObstacle, curves: List) -> Tuple[Optional[BezierCurve], bool]:
        can_cross = (obstacle.height < self.ground_clearance and
                     obstacle.width < self.wheel_distance)

        # 轨迹评估函数
        def evaluate_curve(curve, is_crossing):
            # 采样曲线上的点
            ts = np.linspace(0, 1, 20)  # 增加采样点数量
            points = np.array([curve.compute_point(t) for t in ts])

            # 计算参数化曲线的一阶和二阶导数
            dx = np.gradient(points[:, 0])
            dy = np.gradient(points[:, 1])
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)

            # 1. 曲率评估
            curvature = np.abs(dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** 1.5
            mean_curvature = np.mean(curvature)
            max_curvature = np.max(curvature)

            # 2. 横向动力学评估
            # 假设速度为v (可根据实际情况调整)
            v = self.state[4]
            # 横向加速度 (ay = v^2 * curvature)
            lateral_acc = v ** 2 * curvature
            max_lateral_acc = np.max(lateral_acc)

            # 3. 向心加速度评估 (与横向加速度相同)
            centripetal_acc = lateral_acc

            # 4. 横向jerk评估 (加速度变化率)
            jerk = np.gradient(lateral_acc)
            max_jerk = np.max(np.abs(jerk))

            # 5. 方向变化评估 (用于减少频繁切换)
            if hasattr(self, 'last_curve_index'):
                current_curve_index = curves.index(curve)
                is_switching = current_curve_index != self.last_curve_index
            else:
                is_switching = False

            # 6. 轨迹长度评估 (避免不必要的绕行)
            curve_length = np.sum(np.sqrt(np.diff(points[:, 0]) ** 2 + np.diff(points[:, 1]) ** 2))

            # 权重设置 (可根据实际需求调整)
            weights = {
                'mean_curvature': 0.3,
                'max_curvature': 0.2,
                'max_lateral_acc': 0.2,
                'max_jerk': 0.15,
                'is_switching': 0.1 if is_switching else 0,
                'curve_length': 0.05
            }

            # 归一化处理 (假设这些是最大允许值)
            max_values = {
                'mean_curvature': 0.5,
                'max_curvature': 1.0,
                'max_lateral_acc': 2.0,  # m/s^2
                'max_jerk': 0.5,  # m/s^3
                'is_switching': 1,
                'curve_length': 20.0  # meters
            }

            # 计算加权得分 (得分越低越好)
            score = 0
            score += weights['mean_curvature'] * (mean_curvature / max_values['mean_curvature'])
            score += weights['max_curvature'] * (max_curvature / max_values['max_curvature'])
            score += weights['max_lateral_acc'] * (max_lateral_acc / max_values['max_lateral_acc'])
            score += weights['max_jerk'] * (max_jerk / max_values['max_jerk'])
            score += weights['is_switching'] * is_switching
            score += weights['curve_length'] * (curve_length / max_values['curve_length'])

            return score, {
                'mean_curvature': mean_curvature,
                'max_curvature': max_curvature,
                'max_lateral_acc': max_lateral_acc,
                'max_jerk': max_jerk,
                'is_switching': is_switching,
                'curve_length': curve_length
            }

        # 选择最佳曲线
        if can_cross:
            valid_indices = [0, 1, 2]  # 对应的原始索引
            valid_curves = [curves[0],curves[1], curves[2]]
            # 可跨越时，评估所有曲线
            scored_curves = [(evaluate_curve(curve, True), curve) for curve in curves]
            scored_curves.sort(key=lambda x: x[0][0])  # 按得分排序
            best_score, metrics = scored_curves[0][0]
            best_curve = scored_curves[0][1]

            # 检查是否满足安全约束
            # if (metrics['max_lateral_acc'] > 2.5 or  # 超过最大允许横向加速度
            #         metrics['max_jerk'] > 1.0 or  # 超过最大允许jerk
            #         metrics['max_curvature'] > 1.5):  # 超过最大允许曲率
            #     can_cross = False  # 即使物理上可以跨越，动力学上也不安全
            # 记录当前选择的曲线索引
            current_index = valid_indices[valid_curves.index(best_curve)]

            # # 检查是否需要切换轨迹
            # if hasattr(self, 'last_curve_index') and self.last_curve_index != None:
            #     if current_index != self.last_curve_index:
            #         # 如果切换轨迹，需要确保新轨迹明显更好
            #         if best_score > scored_curves[1][0][0] * 0.8:  # 新轨迹优势不明显时保持原轨迹
            #             best_curve = valid_curves[valid_indices.index(self.last_curve_index)]
            #
            # # 更新最后选择的曲线索引
            # self.last_curve_index = current_index

        if not can_cross:
            # 绕行时，只考虑特定曲线（如你原始代码中的valid_curves）
            valid_indices = [0, 2]  # 对应的原始索引
            valid_curves = [curves[0], curves[2]]

            # 评估候选曲线
            scored_curves = [(evaluate_curve(curve, False), curve) for curve in valid_curves]
            scored_curves.sort(key=lambda x: x[0][0])

            # 选择得分最低的曲线
            best_score, metrics = scored_curves[0][0]
            best_curve = scored_curves[0][1]

            # 记录当前选择的曲线索引
            current_index = valid_indices[valid_curves.index(best_curve)]

            # # 检查是否需要切换轨迹
            # if hasattr(self, 'last_curve_index') and self.last_curve_index != None:
            #     if current_index != self.last_curve_index:
            #         # 如果切换轨迹，需要确保新轨迹明显更好
            #         if best_score > scored_curves[1][0][0] * 0.8:  # 新轨迹优势不明显时保持原轨迹
            #             best_curve = valid_curves[valid_indices.index(self.last_curve_index)]
            #
            # # 更新最后选择的曲线索引
            # self.last_curve_index = current_index

        return best_curve, can_cross

    def _is_too_close_to_obstacle(self, curve: BezierCurve, obstacle: StaticObstacle) -> bool:
        """检查曲线是否离障碍物太近(NumPy版本)"""
        # 采样曲线上的点
        ts = np.linspace(0, 1, 5)
        points = np.array([curve.compute_point(t) for t in ts])

        # 计算到障碍物的距离
        obs_center = np.array([obstacle.x, obstacle.y])
        distances = np.linalg.norm(points - obs_center, axis=1)

        # 安全距离阈值
        safe_distance = max(obstacle.width, obstacle.length) * 1.2

        return np.any(distances < safe_distance)

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
        mid_points = [
            np.array([obstacle.x, obstacle.y + offset]) for offset in lateral_offsets
        ]

        # 终点位置（统一使用前方采样点）
        end_point = [
            np.array([obstacle.x + self.forward_sample+obstacle.length/2, obstacle.y + offset]) for offset in lateral_offsets
        ]

        # 生成三条候选曲线
        for i in range(len(mid_points)):
            curve = BezierCurve(current_pos, mid_points[i], end_point[i])
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
        # 确保时间间隔合理
        time_interval = t_end - t_start
        # 确保时间参数在有效范围内
        t = np.clip((t_interpolate - t_start) / time_interval, 0, 1)

        point = guide_traj.compute_point(t)
        derivative = guide_traj.compute_derivative(t)

        # 确保导数不为零向量
        if np.linalg.norm(derivative) < 1e-6:
            derivative = np.array([1e-6, 0])  # 小量向前

        phi = np.arctan2(derivative[1], derivative[0])
        # global_phi = self.ref_traj.compute_phi(t_interpolate, self.path_num, self.u_num)
        # blend_ratio = np.clip(t, 0.2, 0.8)  # 渐进混合系数
        # phi = angle_normalize(blend_ratio * phi + (1 - blend_ratio) * global_phi)

        u = self.ref_traj.compute_u(t_interpolate, self.path_num, self.u_num)

        # 确保速度方向与航向一致
        if u < 0:
            phi = angle_normalize(phi + np.pi)
            u = abs(u)

        traj_point = np.array([point[0], point[1], phi, u], dtype=np.float32)
        return traj_point

    def update_loca_traj(self, local_traj):
        self.planned_traj = local_traj

    def conduct_trajectory(self, trajectory):
        traj = trajectory
        # 控制+定位，假设完美控制到下一个轨迹点
        print("[x, y]:",traj.x[1], traj.y[1])
        self.ego_veh_state.transform.location.x, self.ego_veh_state.transform.location.y, self.ego_veh_state.transform.rotation.yaw \
            = traj.x[1], traj.y[1], traj.heading[1]
        self.ego_veh_state.kinematics.speed, self.ego_veh_state.kinematics.acceleration, self.ego_veh_state.kinematics.curvature \
            = traj.v[1], traj.a[1], traj.curvature[1]
        for tb in self.obstaclesBox:
            tb.set_obb([tb.x + tb.vx * traj.dt, tb.y + tb.vy * traj.dt, tb.length, tb.width, tb.box_heading])
    # todo 完善完美执行规划轨迹的step
    def step_w_perfectaction(self, trajectory: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        traj = trajectory
        # 控制+定位，假设完美控制到下一个轨迹点
        print("[x, y]:", traj.x[1], traj.y[1])
        self.ego_veh_state.transform.location.x, self.ego_veh_state.transform.location.y, self.ego_veh_state.transform.rotation.yaw \
            = traj.x[1], traj.y[1], traj.heading[1]
        self.ego_veh_state.kinematics.speed, self.ego_veh_state.kinematics.acceleration, self.ego_veh_state.kinematics.curvature \
            = traj.v[1], traj.a[1], traj.curvature[1]
        for tb in self.obstaclesBox:
            tb.set_obb([tb.x + tb.vx * traj.dt, tb.y + tb.vy * traj.dt, tb.length, tb.width, tb.box_heading])
        for dynamic_veh in self.dynamic_obss:
            dynamic_veh.step()
        self.update_dynamic_state()
        for tb in self.obstaclesBox:
            tb.set_obb([tb.x + tb.vx * self.dt, tb.y + tb.vy * self.dt, tb.length, tb.width, tb.box_heading])


        if self.obstacle != None:
            if self.generate_guide == 1 and self.best_curve == None:
                new_ref_point = np.array([
                    self.ref_traj.compute_x(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num),
                    self.ref_traj.compute_y(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num),
                    self.ref_traj.compute_phi(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num),
                    self.ref_traj.compute_u(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num),
                ], dtype=np.float32
                )

            #     obstacle, generate_guide = self.is_generate_guide()
            #     if generate_guide and self.best_curve != None:
            #         self.generate_guide = generate_guide
            #         self.obstacle = obstacle
            #         # quintic_curves = self.generate_quintic_curves(self.obstacle)  # 生成3条五次多项式，每条包含2个元素，为横向、纵向位置多项式
            #         bezier_curves = self.generate_bezier_curves(self.obstacle)  # 生成3条五次多项式，每条包含2个元素，为横向、纵向位置多项式
            #         self.best_curve, can_cross = self.can_cross_decision(self.obstacle, bezier_curves)
            #         self.t_start = self.t
            #         self.t_end = self.t + (self.obstacle.x + self.forward_sample - self.state[0]) / self.state[3]
            #         for i in range(1, self.pre_horizon + 1):
            #             ref_x = self.ref_traj.compute_x(self.t + i * self.dt, self.path_num, self.u_num)
            #             guide_point = self.get_bezier_guide_points(self.best_curve, self.t + i * self.dt, self.t_start,
            #                                                        t_end=self.t_end)
            #             t = (ref_x - guide_point[0]) / self.state[3]
            #             while guide_point[0] < ref_x and t < self.t_end and t > 0:
            #                 # guide_point = self.get_guide_points(self.best_curve, t, self.t_start)
            #                 guide_point = self.get_bezier_guide_points(self.best_curve, t, self.t_start, self.t_end)
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

            elif self.generate_guide == 1 and self.best_curve != None:
                # ref_x = self.ref_traj.compute_x(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num)
                # new_ref_point = self.get_quintic_guide_points(self.best_curve, self.t + self.pre_horizon * self.dt, self.t_start)
                new_ref_point = self.get_bezier_guide_points(self.best_curve, self.t + self.pre_horizon * self.dt,
                                                             self.t_start, self.t_end)
                # t_gap = (ref_x - new_ref_point[0]) / new_ref_point[3]
                # if t_gap > self.t_end and t_gap > 0:
                #     new_ref_point = np.array([
                #         self.ref_traj.compute_x(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num),
                #         self.ref_traj.compute_y(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num),
                #         self.ref_traj.compute_phi(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num),
                #         self.ref_traj.compute_u(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num),
                #     ], dtype=np.float32
                #     )
                # else:
                #     while new_ref_point[0] <= ref_x and t_gap <= self.t_end and t_gap > 0:
                #         # new_ref_point = self.get_quintic_guide_points(self.best_curve, t + self.pre_horizon * self.dt, self.t_start)
                #         new_ref_point = self.get_bezier_guide_points(self.best_curve, t_gap + self.pre_horizon * self.dt, self.t_start, self.t_end)
                #         t_gap += self.dt
                #     if new_ref_point[0] >= self.obstacle.x + self.forward_sample:
                #         if self.state[0] > self.obstacle.x + self.obstacle.length / 2:  # 确保车辆完全通过
                #             self.processed_obstacles.add(self.obstacle.obs_id)
                #             self.generate_guide = 0
                #             self.best_curve = None
                #         new_ref_point = np.array([
                #             self.ref_traj.compute_x(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num),
                #             self.ref_traj.compute_y(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num),
                #             self.ref_traj.compute_phi(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num),
                #             self.ref_traj.compute_u(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num),
                #         ], dtype=np.float32
                #         )

            if self.state[0] > self.obstacle.x:  # 确保车辆完全通过
                self.processed_obstacles.add(self.obstacle.obs_id)
                self.generate_guide = 0
                self.obstacle = None
        else:
            # 判断下是否需要生成引导轨迹
            obstacle, generate_guide = self.is_generate_guide()
            if generate_guide and obstacle != None:
                self.generate_guide = generate_guide
                self.obstacle = obstacle
                # quintic_curves = self.generate_quintic_curves(self.obstacle)  # 生成3条五次多项式，每条包含2个元素，为横向、纵向位置多项式
                bezier_curves = self.generate_bezier_curves(obstacle)  # 生成3条贝塞尔曲线
                self.best_curve, can_cross = self.can_cross_decision(self.obstacle, bezier_curves)
                self.t_start = self.t
                self.t_end = self.t + (self.obstacle.x + self.forward_sample + obstacle.width / 2 - self.state[0]) / \
                             self.state[3]
                for i in range(1, self.pre_horizon + 1):
                    ref_x = self.ref_traj.compute_x(self.t + i * self.dt, self.path_num, self.u_num)
                    # # Check if guide_traj is None before proceeding
                    if self.best_curve is None:
                        # Use reference trajectory directly if guide trajectory is not available
                        guide_point = np.array([
                            self.ref_traj.compute_x(self.t + i * self.dt, self.path_num, self.u_num),
                            self.ref_traj.compute_y(self.t + i * self.dt, self.path_num, self.u_num),
                            self.ref_traj.compute_phi(self.t + i * self.dt, self.path_num, self.u_num),
                            self.ref_traj.compute_u(self.t + i * self.dt, self.path_num, self.u_num),
                        ], dtype=np.float32)
                    else:
                        guide_point = self.get_bezier_guide_points(self.best_curve, self.t + i * self.dt, self.t_start,
                                                                   self.t_end)
                        t_gap = (ref_x - guide_point[0]) / self.state[3]
                        while guide_point[0] < ref_x and t_gap < self.t_end and t_gap > 0:
                            # guide_point = self.get_guide_points(self.best_curve, t, self.t_start)
                            guide_point = self.get_bezier_guide_points(self.best_curve, self.t + t_gap, self.t_start,
                                                                       self.t_end)
                            t_gap += self.dt
                        # 超出范围，使用全局轨迹点, 若bezier 曲线，范围调到障碍物前方采样点
                        if guide_point[0] >= self.obstacle.x + self.forward_sample:
                            if self.state[0] > obstacle.x:  # 确保车辆完全通过
                                self.processed_obstacles.add(obstacle.obs_id)
                                self.generate_guide = 0
                                self.obstacle = None
                            # self.processed_obstacles.add(self.obstacle.obs_id)
                            self.best_curve = None
                            guide_point = np.array([
                                ref_x,
                                self.ref_traj.compute_y(self.t + i * self.dt, self.path_num, self.u_num),
                                self.ref_traj.compute_phi(self.t + i * self.dt, self.path_num, self.u_num),
                                self.ref_traj.compute_u(self.t + i * self.dt, self.path_num, self.u_num),
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
        self.ego_veh_state = VehicleState.from_kine_states(self.state[0], self.state[1], self.state[2],
                                                           vx=self.state[3], vy=self.state[4],
                                                           length=self.veh_length, width=self.veh_width)
        reward = 0

        return self.get_obs(), reward, done, self.info

    def visualize(self,traj):
        '''
        qzl: 要修改，统一格式
        '''
        import gops.utils.planner_benchmark.visualize as vis
        import matplotlib.pyplot as plt
        # for lane in self.local_map.lanes:
        #     plt.plot(lane.centerline[:, 0], lane.centerline[:, 1], color='gray', linestyle='--', lw=1.5)  # 画地图
        # vis.draw_ego_vehicle(self.ego_veh_state, color='green', fill=True, alpha=0.2, linestyle='-', linewidth=1.5) # 画自车
        legend_label = []
        for tb in self.obstaclesBox:
            can_cross = (tb.height < self.ground_clearance and
                         tb.width < self.wheel_distance)
            if tb.vx != 0:
                vis.draw_boundingbox(tb, color='red', fill=True, alpha=0.1, linestyle='-', linewidth=1.5)  # 画他车
                legend_label.append(f'Dynamic_{tb.id}')
            else:
                color = 'lime' if can_cross else 'darkviolet'
                vis.draw_boundingbox(tb, color=color, fill=True, alpha=0.1, linestyle='-', linewidth=1.5)  # 画他车
                legend_label.append(f'Static_{tb.id}({"Cross" if can_cross else "Avoid"})')
        #     # 画他车预测轨迹
        #     tb_pred_traj = np.column_stack((tb.x + np.asarray(traj.t) * tb.vx, tb.y + np.asarray(traj.t) * tb.vy))
        #     vis.draw_polyline(tb_pred_traj, show_buffer=True, buffer_dist=tb.width * 0.5, buffer_alpha=0.1,
        #                       color='C3')

        # vis.draw_ego_history(self.ego_veh_state, '-', lw=1, color='gray')  # 画自车历史
        vis.draw_trajectory(traj, '.-', show_footprint=True, color='pink')  # 画轨迹
        if "control_points" in traj.debug_info:  # bezier planner
            pts = traj.debug_info["control_points"]
            plt.plot(pts[:, 0], pts[:, 1], 'or')
        # if "corridor" in traj.debug_info: # optimizer planner
        #     vis.draw_corridor(traj.debug_info["corridor"], color='green', linewidth=0.5)
        if "initial_trajectory" in traj.debug_info:
            vis.draw_trajectory(traj.debug_info["initial_trajectory"], '--', color="black", show_footprint=False)

        vis.draw_ego_vehicle(self.ego_veh_state, color='magenta', fill=True, alpha=0.3, linestyle='-', linewidth=1.5)  # 画自车
        legend_label.append('Planned traj')
        # plt.axis('equal')
        plt.tight_layout()
        vis.ego_centric_view(self.ego_veh_state.x(), self.ego_veh_state.y(), [-20, 80], [-10, 10])
        # plt.xlim([ego_veh_state.x() - 20, ego_veh_state.x() + 80])
        # plt.ylim([ego_veh_state.y() - 5, ego_veh_state.y() + 5])
        # plt.pause(0.001)
        plt.legend(legend_label, ncol=6, loc='upper left', fontsize=6, bbox_to_anchor= (0, 1.2))

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
            "location":self.state,
            "best_curve_isnone": self.best_curve==None,
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
            static_height = self.static_state[i_static, 6]
            # 判断是否可跨越
            can_cross = (static_height < self.ground_clearance and
                         static_width < self.wheel_distance)

            # 设置颜色 - 可跨越为黄色，不可跨越为灰色
            color = 'yellow' if can_cross else 'gray'

            ax.add_patch(pc.Rectangle(
                (static_x - static_length / 2, static_y - static_width / 2),
                static_length,
                static_width,
                angle=static_phi * 180 / np.pi,
                facecolor=color,
                edgecolor=color,  # 边缘色更深
                zorder=1
            ))

            # 在图例中标注是否可跨越
            legend_label.append(f'Static_{i_static}({"Cross" if can_cross else "Avoid"})')


        # 新增绘制逻辑 -------------------------------------------------
        if hasattr(self, 'guide_traj') and self.best_curve is not None:
            # 绘制贝塞尔曲线
            t_values = np.linspace(0, 1, 20)
            curve_points = [self.best_curve.compute_point(t) for t in t_values]
            x_coords = [p[0] for p in curve_points]
            y_coords = [p[1] for p in curve_points]
            ax.plot(x_coords, y_coords, 'c--', linewidth=1.5, zorder=3, label='Bezier Trajectory')

            # 绘制控制点
            control_points = [
                self.best_curve.p0,
                self.best_curve.p1,
                self.best_curve.p2
            ]
            colors = ['ro', 'go', 'bo']  # 红:起点, 绿:控制点, 蓝:终点
            labels = ['Start Point', 'Control Point', 'End Point']

            for idx, (point, color, label) in enumerate(zip(control_points, colors, labels)):
                ax.plot(point[0], point[1], color, markersize=8, zorder=4)
                ax.text(point[0] + 0.5, point[1] + 0.5, label, fontsize=8, color=color[0])

            # 连接控制点辅助线
            ax.plot([self.best_curve.p0[0], self.best_curve.p1[0]],
                    [self.best_curve.p0[1], self.best_curve.p1[1]],
                    'g:', linewidth=0.8, zorder=2)
            ax.plot([self.best_curve.p1[0], self.best_curve.p2[0]],
                    [self.best_curve.p1[1], self.best_curve.p2[1]],
                    'b:', linewidth=0.8, zorder=2)
            legend_label.extend(['Bezier Curve', 'Control Lines'])
        # ------------------------------------------------------------
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
