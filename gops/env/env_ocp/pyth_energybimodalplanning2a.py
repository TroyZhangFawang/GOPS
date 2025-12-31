#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: vehicle 4DOF data environment


from typing import Dict, Optional, Sequence, Tuple
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import gym
import numpy as np
from gops.env.env_ocp.pyth_base_env import PythBaseEnv
from gops.env.env_ocp.resources.ref_traj_data import MultiRefTrajData, MultiRoadSlopeData
from gops.utils.math_utils import angle_normalize


@dataclass
class StaticObstacle:
    obs_id: int  # 障碍物ID
    material: str  # 障碍物材质 0为soft,直接跨，1为hard，需要结合尺寸进行判断
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

class VehicleDynamicsData:
    def __init__(self):
        self.vehicle_params = dict(
            k_f=47900,  # front wheel cornering stiffness [N/rad]
            k_r=47900,  # rear wheel cornering stiffness [N/rad]
            l_f=1.544,  # distance from CG to front axle [m]
            l_r=1.456,  # distance from CG to rear axle [m]
            m=2060.0,  # total mass [kg]
            ms=1700.0,  # sprung mass [kg]
            h_cg=0.25,  # 质心高度-侧倾中心高度
            I_z=2250,  # z Polar moment of inertia at CG [kg*m^2]
            I_x=700.7,  # x Polar moment of inertia at CG [kg*m^2]
            miu=0.85,  # tire-road friction coefficient
            g=9.81,  # acceleration of gravity [m/s^2]
            k_psi=135330,  # roll cornering stiffness [N/rad]
            C_psi=8000,  # roll damper
            ground_clearance=0.25,
            wheel_distance=1.8,
            veh_length=4.8,
            veh_width=2.0,
        )
        l_f, l_r, mass, g = (
            self.vehicle_params["l_f"],
            self.vehicle_params["l_r"],
            self.vehicle_params["m"],
            self.vehicle_params["g"],
        )
        F_zf, F_zr = l_r * mass * g / (l_f + l_r), l_f * mass * g / (l_f + l_r)
        self.vehicle_params.update(dict(F_zf=F_zf, F_zr=F_zr))

    def f_xu(self, states, actions, disturb, delta_t):
        x, y, phi, u, v, w, psi, psi_dot = states
        steer, a_x = actions
        psi_i, psi_b = disturb

        k_f = self.vehicle_params["k_f"]
        k_r = self.vehicle_params["k_r"]
        l_f = self.vehicle_params["l_f"]
        l_r = self.vehicle_params["l_r"]
        m = self.vehicle_params["m"]
        ms = self.vehicle_params["ms"]
        h_cg = self.vehicle_params["h_cg"]
        I_z = self.vehicle_params["I_z"]
        I_x = self.vehicle_params["I_x"]
        k_psi = self.vehicle_params["k_psi"]
        C_psi = self.vehicle_params["C_psi"]
        g = self.vehicle_params["g"]

        next_state = np.empty(states.shape)
        # 计算侧偏角，侧向力
        alpha_f = (v + w * l_f) / u - steer
        Fyf = -k_f * alpha_f
        alpha_r = (v - l_r * w) / u
        Fyr = -k_r * alpha_r
        # 状态更新公式
        next_state[0] = x + delta_t * (u * np.cos(phi) - v * np.sin(phi))
        next_state[1] = y + delta_t * (u * np.sin(phi) + v * np.cos(phi))
        temp1 = ms * h_cg * \
                (ms * h_cg * u * w + ms * g * h_cg * psi - k_psi * (psi - psi_b) - C_psi * psi_dot) \
                / (m * (I_x + ms * h_cg ** 2))
        temp2 = 1 - ms ** 2 * h_cg ** 2 / (m * (I_x + ms * h_cg ** 2))
        v_dot = (-u * w + temp1 + 2 * (Fyf + Fyr) / m - g * psi_b) / temp2
        next_state[3] = u + delta_t * (a_x + v * w - 2 * Fyf * steer / m - g * psi_i)
        next_state[4] = v + delta_t * v_dot
        next_state[2] = phi + delta_t * w
        next_state[5] = w + delta_t * 2 * (l_f * Fyf - l_r * Fyr) / I_z
        next_state[6] = psi + delta_t * psi_dot
        next_state[7] = psi_dot + delta_t * ((ms * h_cg * (v_dot + u * w)
                                              + ms * g * h_cg * psi - k_psi * (psi - psi_b)
                                              - C_psi * psi_dot) / (I_x + ms * h_cg ** 2))
        next_state[2] = angle_normalize(next_state[2])
        return np.array(next_state, dtype=np.float32)

class PythEnergybimodalplanning2a(PythBaseEnv):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    def __init__(
        self,
        pre_horizon: int = 30,
        path_para: Optional[Dict[str, Dict]] = None,
        u_para: Optional[Dict[str, Dict]] = None,
        slope_para: Optional[Dict[str, Dict]] = None,
        max_steer: float = np.pi / 6,
        static_obstacle_num: int = 20,
        d_pre: float = 20.0,  # 离障碍物多少远开始规划
        lateral_sample: float = 3.5,  # 横向采样距离
        forward_sample: float = 10.0,  # 纵向采样距离
        **kwargs,
    ):
        work_space = kwargs.pop("work_space", None)
        if work_space is None:
            # initial range of [delta_x, delta_y, delta_phi, delta_u, v, w, psi, psi_dot]
            init_high = np.array([2, 1, np.pi / 6, 2, 0.1, 0.1, np.pi / 36, 0.1], dtype=np.float32)
            init_low = -init_high
            work_space = np.stack((init_low, init_high))
        super(PythEnergybimodalplanning2a, self).__init__(work_space=work_space, **kwargs)

        self.vehicle_dynamics = VehicleDynamicsData()
        self.ref_traj = MultiRefTrajData(path_para, u_para)
        self.road_slope = MultiRoadSlopeData(slope_para)
        self.state_dim = 8
        self.pre_horizon = pre_horizon
        ego_obs_dim = 8
        ref_obs_dim = 6
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf] * (ego_obs_dim + ref_obs_dim * pre_horizon + static_obstacle_num * 4)),
            high=np.array([np.inf] * (ego_obs_dim + ref_obs_dim * pre_horizon + static_obstacle_num * 4)),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-max_steer, -3]),
            high=np.array([max_steer, 3]),
            dtype=np.float32,
        )
        self.dt = 0.05
        self.max_episode_steps = 500

        self.state = None
        self.path_num = None
        self.u_num = None
        self.t = None
        self.ref_points = None

        self.generate_guide = 0
        self.guide_end = np.zeros((3, 4))
        self.guide_time_interval = 0
        self.t_start = 0
        self.t_end = 0.1
        self.static_obstacle_num = static_obstacle_num
        self.static_state = np.zeros((static_obstacle_num, 8), dtype=np.float32)
        self.veh_width = self.vehicle_dynamics.vehicle_params["veh_width"]
        self.veh_length = self.vehicle_dynamics.vehicle_params["veh_length"]
        self.wheel_distance = self.vehicle_dynamics.vehicle_params["wheel_distance"]
        self.ground_clearance = self.vehicle_dynamics.vehicle_params["ground_clearance"]
        self.d_pre = d_pre
        self.lateral_sample = lateral_sample
        self.forward_sample = forward_sample
        self.best_curve = None
        self.obstacle = None
        self.seed()
        self.info_dict = {
            "state": {"shape": (self.state_dim,), "dtype": np.float32},
            "ref_points": {"shape": (self.pre_horizon + 1, 6), "dtype": np.float32},
            "path_num": {"shape": (), "dtype": np.uint8},
            "u_num": {"shape": (), "dtype": np.uint8},
            "ref_time": {"shape": (), "dtype": np.float32},
            "ref": {"shape": (6,), "dtype": np.float32},
            "static_state": {"shape": (static_obstacle_num, 7), "dtype": np.float32},
            "generate_guide": {"shape": (), "dtype": np.uint8},
            "guide_start": {"shape": (4,), "dtype": np.float32},
            "guide_end": {"shape": (3, 4), "dtype": np.float32},
            "guide_time_interval": {"shape": (), "dtype": np.float32},
            "t_start": {"shape": (), "dtype": np.float32},
            "t_end": {"shape": (), "dtype": np.float32},
            "best_curve_isnone": {"shape": (), "dtype": bool},
        }

    def reset(
        self,
        init_state: Optional[Sequence] = None,
        ref_time: Optional[float] = None,
        ref_num: Optional[int] = None,
        u_num: Optional[int] = None,
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
            slope_num = None
        else:
            path_num = int(ref_num / 2)
            u_num = int(ref_num % 2)
            slope_num = int(ref_num % 2)

        # If no ref_num, then randomly select path and speed
        if path_num is not None:
            self.path_num = path_num
        else:
            self.path_num = self.np_random.choice([7])

        if u_num is not None:
            self.u_num = u_num
        else:
            self.u_num = self.np_random.choice([0])

        if slope_num is not None:
            self.slope_num = slope_num
        else:
            self.slope_num = self.np_random.choice([0])


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
            road_longi = self.road_slope.compute_longislope(self.t + i * self.dt, self.slope_num)
            road_lat = self.road_slope.compute_latslope(self.t + i * self.dt, self.slope_num)
            ref_points.append([ref_x, ref_y, ref_phi, ref_u, road_longi, road_lat])
        self.ref_points = np.array(ref_points, dtype=np.float32)

        if init_state is not None:
            delta_state = np.array(init_state, dtype=np.float32)
        else:
            delta_state = self.sample_initial_state()
        self.state = np.concatenate(
            (self.ref_points[0, :4] + delta_state[:4], delta_state[4:])
        )


        # add static obstacle
        self.static_obss = []

        static_time = [3, 15, 22, 27.5, 35, 43.5, 52, 57.5, 65, 76.5]
        #,53, 58, 65, 67.5, 75, 80.5, 92, 97.5, 102, 105.5,
                       # 113, 120, 128, 137.5, 145, 146.5, 147, 153.5, 155, 157.5,
        for i_static in range(self.static_obstacle_num):
            delta_t = self.np_random.uniform(3, 70) #static_time[i_static]#
            static_obs_phi = self.ref_traj.compute_phi(self.t + delta_t, self.path_num, self.u_num)
            delta_lon = 0#1.0 * self.np_random.uniform(-1, 1)
            delta_lat = 0#1.0 * self.np_random.uniform(-1, 1)
            static_obs_x = self.ref_traj.compute_x(self.t + delta_t, self.path_num, self.u_num) + delta_lon
            static_obs_y = self.ref_traj.compute_y(self.t + delta_t, self.path_num, self.u_num) + delta_lat
            self.static_length = np.random.uniform(0, 10, self.static_obstacle_num)#np.array([0.5, 3, 5,0.5, 3, 5,0.5, 3, 5, 7])#
            self.static_width = np.random.uniform(0, 10, self.static_obstacle_num)#, 0.5 np.array([5.0, 4, 3.0, 5.6, 4.7, 3.9,1.7, 1.4, 3.2,  6.7]) #
            self.static_height = np.random.uniform(0, 2, self.static_obstacle_num)#, 0.23 np.array([0.2, 0.5, 0.5, 0.2, 0.5, 0.5,0.2, 0.1, 0.5, 1.0]) #
            self.static_material = self.np_random.choice([0, 1])#np.array([0, 0, 1, 0, 0, 0, 1, 1, 0 , 0])
            self.static_obss.append(
                StaticObstacle(
                    obs_id=i_static,
                    x=static_obs_x,
                    y=static_obs_y,
                    phi=static_obs_phi,
                    length=self.static_length[i_static], #
                    width=self.static_width[i_static],
                    height=self.static_height[i_static],
                    material=self.static_material
                )
            )

        self.update_static_state()
        # 初始时刻添加完障碍物之后就先判断下是否需要生成引导轨迹
        obstacle, generate_guide = self.is_generate_guide()
        if generate_guide and obstacle != None:
            self.generate_guide = generate_guide
            self.obstacle = obstacle
            bezier_curves = self.generate_bezier_curves(self.obstacle)  # 生成3条贝塞尔曲线，每条包含2个元素，为横向、纵向位置多项式
            self.best_curve, can_cross = self.can_cross_decision(self.obstacle, bezier_curves)
            self.t_start = self.t
            self.t_end = self.t + (self.obstacle.x + self.forward_sample - self.state[0]) / self.state[3]
            guide_points = [self.state[:4]]
            self.pos_x = self.state[0]
            self.pos_y = self.state[1]
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
                    guide_point = self.get_bezier_guide_points(self.best_curve, self.t + i * self.dt, self.t_start,
                                                               self.t_end)
                    t_gap = (ref_x - guide_point[0]) / self.state[3]
                    while guide_point[0] < ref_x and t_gap < self.t_end and t_gap > 0:
                        # guide_point = self.get_guide_points(self.best_curve, t, self.t_start)
                        guide_point = self.get_bezier_guide_points(self.best_curve, self.t + t_gap, self.t_start,
                                                                   self.t_end)
                        t_gap += self.dt
                    # 超出范围，使用全局轨迹点, 若bezier 曲线，范围调到障碍物前方采样点
                    if guide_point[0] >= obstacle.x + self.forward_sample:
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
        action = np.clip(action, self.action_space.low, self.action_space.high)
        reward = self.compute_reward(action)
        disturb = self.ref_points[1, 4:]
        self.state = self.vehicle_dynamics.f_xu(self.state, action, disturb, self.dt)

        # # # 新加的##########----要想这部分成功run起来，NN based需要在mlp里的finitehorizonfull 的forward函数把取第一个action给注释掉
        # # MPC 需要在opt_controller.py文件里把 163行的0改为：，同时需要把sys_run 中 run_an_episode 中action的第0个存到action_list中
        # # 如果要plot，还得在sys run里修改action_list
        # disturb = self.ref_points[1, 4:]
        # self.state = self.vehicle_dynamics.f_xu(self.state, action[0, :], disturb,self.dt)
        # self.state_full = np.empty((self.pre_horizon, self.state_dim))
        # self.state_full[0, :] = self.state
        # reward = self.compute_reward(action[0, :])
        # self.action = action
        # state = self.state
        # for i in range(1, self.pre_horizon):
        #         state = self.vehicle_dynamics.f_xu(state, action[i, :], self.ref_points[i+1, 4:], self.dt)
        #         self.state_full[i, :] = state


        self.t = self.t + self.dt

        self.ref_points[:-1] = self.ref_points[1:]

        if self.obstacle != None:
            if self.generate_guide == 1 and self.best_curve == None:
                new_ref_point = np.array([
                                    self.ref_traj.compute_x(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num),
                                    self.ref_traj.compute_y(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num),
                                    self.ref_traj.compute_phi(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num),
                                    self.ref_traj.compute_u(self.t + self.pre_horizon * self.dt, self.path_num, self.u_num),
                                ], dtype=np.float32
                                )

            elif self.generate_guide == 1 and self.best_curve != None:
                new_ref_point = self.get_bezier_guide_points(self.best_curve, self.t + self.pre_horizon * self.dt, self.t_start, self.t_end)

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
                self.pos_x, self.pos_y = self.state[0], self.state[1]
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
                    self.ref_points[i, :4] = guide_point
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

        new_slope_point = np.array([self.road_slope.compute_longislope(self.t+self.pre_horizon*self.dt, self.slope_num),
                                    self.road_slope.compute_latslope(self.t+self.pre_horizon*self.dt, self.slope_num)])
        new_ref_point = np.concatenate([new_ref_point, new_slope_point])

        self.ref_points[-1] = new_ref_point

        self.done = self.judge_done()
        if self.done:
            reward = reward - 1000

        return self.get_obs(), reward, self.done, self.info

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
        ref_obs = np.stack((ref_x_tf, ref_y_tf, ref_phi_tf, ref_u_tf, self.ref_points[:, 4], self.ref_points[:, 5]), 1)[1:].flatten()

        static_x_tf, static_y_tf, static_phi_tf = \
            ego_vehicle_coordinate_transform(
                self.state[0], self.state[1], self.state[2],
                self.static_state[:, 1], self.static_state[:, 2], self.static_state[:, 3],
            )
        static_u_tf = np.zeros((self.static_obstacle_num,)) - self.state[3]
        static_obs = np.concatenate((static_x_tf/(static_x_tf).max(), static_y_tf/(static_y_tf).max(),
                                     static_phi_tf/(static_phi_tf).max(), static_u_tf/(static_u_tf).max()))
        # static_obs = np.concatenate((static_x_tf, static_y_tf, static_phi_tf, static_u_tf))
        return np.concatenate((ego_obs, ref_obs, static_obs))

    def compute_reward(self, action: np.ndarray) -> float:
        x, y, phi, u, v, w, psi, psi_dot = self.state
        ref_x, ref_y, ref_phi, ref_u = self.ref_points[0, :4]
        steer, a_x = action
        return -(
            0.04 * (x - ref_x) ** 2
            + 0.02 * (y - ref_y) ** 2
            + 0.02 * angle_normalize(phi - ref_phi) ** 2
            + 0.02 * (u - ref_u) ** 2
            + 0.01 * w ** 2
            + 0.05 * steer ** 2
            + 0.01 * a_x ** 2
        )

    def judge_done(self) -> bool:
        x, y, phi = self.state[:3]
        ref_x, ref_y, ref_phi = self.ref_points[0, :3]
        done = ((np.abs(x - ref_x) > 7.5)
                | (np.abs(y - ref_y) > 7.5)
                | (np.abs(angle_normalize(phi - ref_phi)) > np.pi)
                | (self.get_constraint() > 0)
                 )

        # if done:
        #     # print((np.abs(x - ref_x) > 10), np.abs(y - ref_y) > 10, np.abs(angle_normalize(phi - ref_phi)) > np.pi)
        return done

    def update_static_state(self):
        for i, static_obs in enumerate(self.static_obss):
            self.static_state[i] = np.array(
                [static_obs.obs_id, static_obs.x, static_obs.y, static_obs.phi, static_obs.length, static_obs.width, static_obs.height, static_obs.material],
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

    def can_cross_decision(self, obstacle: StaticObstacle, curves: List) -> Tuple[Optional[BezierCurve], bool]:
        if obstacle.material == 0:
            can_cross = True
        elif obstacle.material == 1:
            can_cross = (obstacle.height < self.ground_clearance and
                         obstacle.width < self.wheel_distance)
        # can_cross = False
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
            # 可跨越时，评估所有曲线
            # scored_curves = [(evaluate_curve(curve, True), curve) for curve in curves]
            # scored_curves.sort(key=lambda x: x[0][0])  # 按得分排序
            # best_score, metrics = scored_curves[0][0]
            # # best_curve = scored_curves[0][1]
            best_curve = curves[1]
            # # 检查是否满足安全约束
            # if (metrics['max_lateral_acc'] > 2.5 or  # 超过最大允许横向加速度
            #         metrics['max_jerk'] > 1.0 or  # 超过最大允许jerk
            #         metrics['max_curvature'] > 1.5):  # 超过最大允许曲率
            #     can_cross = False  # 即使物理上可以跨越，动力学上也不安全

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
            self.last_curve_index = valid_indices[valid_curves.index(best_curve)]

            # 检查是否需要切换轨迹
            if hasattr(self, 'last_curve_index'):
                current_index = valid_indices[valid_curves.index(best_curve)]
                if current_index != self.last_curve_index:
                    # 如果切换轨迹，需要确保新轨迹明显更好
                    if best_score > scored_curves[1][0][0] * 0.8:  # 新轨迹优势不明显时保持原轨迹
                        best_curve = valid_curves[valid_indices.index(self.last_curve_index)]

            # 更新最后选择的曲线索引
            self.last_curve_index = valid_indices[valid_curves.index(best_curve)]

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
        lateral_offsets = [(-self.lateral_sample-obstacle.width/2-self.veh_width/2)*np.cos(obstacle.phi),
                           0,
                           (self.lateral_sample+obstacle.width/2+self.veh_width/2)*np.cos(obstacle.phi)]
        longitudinal_offsets = [(self.lateral_sample + obstacle.width / 2 + self.veh_width / 2) * np.sin(obstacle.phi),
                                0,
                            (-self.lateral_sample - obstacle.width / 2 - self.veh_width / 2) * np.sin(obstacle.phi)]

        mid_points = [
            np.array([obstacle.x + longitudinal_offsets[0], obstacle.y + lateral_offsets[0]]),
            np.array([obstacle.x + longitudinal_offsets[1], obstacle.y + lateral_offsets[1]]),
            np.array([obstacle.x + longitudinal_offsets[2], obstacle.y + lateral_offsets[2]])
                    ]

        # 终点位置（统一使用前方采样点）
        end_point = [np.array([obstacle.x + (self.forward_sample+obstacle.length/2)*np.cos(obstacle.phi)+longitudinal_offsets[0],
                               obstacle.y + (self.forward_sample+obstacle.length/2)*np.sin(obstacle.phi)+lateral_offsets[0]]),
                    np.array([obstacle.x + (self.forward_sample+obstacle.length/2)*np.cos(obstacle.phi)+longitudinal_offsets[1],
                              obstacle.y + (self.forward_sample+obstacle.length/2)*np.sin(obstacle.phi)+lateral_offsets[1]]) ,
                     np.array([obstacle.x + (self.forward_sample + obstacle.length / 2) * np.cos(obstacle.phi)+longitudinal_offsets[2],
                               obstacle.y + (self.forward_sample + obstacle.length / 2) * np.sin(obstacle.phi)+lateral_offsets[2]])
                    ]

        # 生成三条候选曲线
        for i in range(len(mid_points)):
            curve = BezierCurve(current_pos, mid_points[i], end_point[i])
            curves.append(curve)
        return curves

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

    def get_constraint(self) -> np.ndarray:
        # collision detection using bicircle model
        # distance from vehicle center to front/rear circle center
        d_ego = (self.veh_length - self.veh_width) / 2
        d_static = (self.static_state[:, 4] - self.static_state[:, 5]) / 2
        # circle radius
        r = 0.5 * self.veh_width

        x, y, phi = self.state[:3]
        ego_center = np.array(
            [
                [x + d_ego * np.cos(phi), y + d_ego * np.sin(phi)],
                [x - d_ego * np.cos(phi), y - d_ego * np.sin(phi)],
            ],
            dtype=np.float32,
        )

        static_x = self.static_state[:, 1]
        static_y = self.static_state[:, 2]
        static_phi = self.static_state[:, 3]
        static_center = np.stack(
            (
                np.stack(
                    ((static_x + d_static * np.cos(static_phi)), static_y + d_static * np.sin(static_phi)),
                    axis=1,
                ),
                np.stack(
                    ((static_x - d_static * np.cos(static_phi)), static_y - d_static * np.sin(static_phi)),
                    axis=1,
                ),
            ),
            axis=1,
        )

        min_dist = np.inf
        for i in range(2):
            # front and rear circle of ego vehicle
            for j in range(2):
                # front and rear circle of staticounding vehicles
                dist = np.linalg.norm(
                    ego_center[np.newaxis, i] - static_center[:, j], axis=1
                )
                min_dist = min(min_dist, np.min(dist))
        ego_to_veh_violation = 2 * r - min_dist

        # road boundary violation
        ego_upper_y = max(ego_center[0, 1], ego_center[1, 1]) + r
        ego_lower_y = min(ego_center[0, 1], ego_center[1, 1]) - r
        # upper_bound_violation = ego_upper_y - self.upper_bound
        # lower_bound_violation = self.lower_bound - ego_lower_y
        return np.array([ego_to_veh_violation], dtype=np.float32)

    @property
    def info(self) -> dict:
        return {
            "state": self.state.copy(),
            "ref_points": self.ref_points.copy(),
            "path_num": self.path_num,
            "u_num": self.u_num,
            "slope_num": self.slope_num,
            "ref_time": self.t,
            "ref": self.ref_points[0].copy(),
            "static_state": self.static_state.copy(),
            "generate_guide": self.generate_guide,
            "guide_start": self.state[:4],
            "guide_end": self.guide_end,
            "guide_time_interval": self.guide_time_interval,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "location": self.state,
            "best_curve_isnone": self.best_curve == None,
        }

    def render(self, mode="human"):
        import matplotlib.pyplot as plt

        fig = plt.figure(num=0, figsize=(6.4, 3.2), dpi=300)
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
        ref_x = []
        ref_y = []
        for i in np.arange(1, 60):
            ref_x.append(self.ref_traj.compute_x(
                self.t + i * self.dt, self.path_num, self.u_num
            ))
            ref_y .append(self.ref_traj.compute_y(
                self.t + i * self.dt, self.path_num, self.u_num
            ))
        ax.plot(ref_x, ref_y, 'b--', lw=1, zorder=2)

        # # draw planning paths
        plan_x = []
        plan_y = []

        for i in range(self.pre_horizon):
            plan_x.append(self.state_full[i, 0])
            plan_y.append(self.state_full[i, 1])
        ax.plot(plan_x, plan_y, 'r', lw=2, zorder=2, marker="o", markersize='3', markeredgecolor='g',markeredgewidth=5)

        # draw texts
        left_x = ego_x - 5
        top_y = ego_y + 15
        delta_y = 2
        ego_speed = self.state[3] * 3.6  # [km/h]
        ref_speed = self.ref_points[0, 3] * 3.6  # [km/h]

        # 绘制静态障碍物（原有逻辑保持不变）
        for i_static in range(self.static_obstacle_num):
            static_x = self.static_state[i_static, 1]
            static_y = self.static_state[i_static, 2]
            static_phi = self.static_state[i_static, 3]
            static_length = self.static_state[i_static, 4]
            static_width = self.static_state[i_static, 5]
            static_height = self.static_state[i_static, 6]
            static_material = self.static_state[i_static, 7]
            # 判断是否可跨越
            if static_material == 0:
                can_cross = True
            elif static_material == 1:
                can_cross = (static_height < self.ground_clearance and
                             static_width < self.wheel_distance)

            # 设置颜色 - 可跨越为黄色，不可跨越为灰色
            color = 'lime' if can_cross else 'darkviolet'

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

        # ax.text(left_x, top_y, f'time: {self.t:.1f}s')
        # ax.text(left_x, top_y - delta_y, f'speed: {ego_speed:.1f}km/h')
        # ax.text(left_x, top_y - 2 * delta_y, f'ref speed: {ref_speed:.1f}km/h')
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
        ax.legend(legend_label, ncol=3, loc='upper left', fontsize=6, bbox_to_anchor= (0., 1.4))

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
    make env `pyth_veh4dofconti`
    """
    return PythEnergybimodalplanning2a(**kwargs)
