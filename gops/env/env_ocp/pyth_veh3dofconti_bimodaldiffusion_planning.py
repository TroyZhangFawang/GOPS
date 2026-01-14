import gym
import numpy as np
import math
from gym import spaces
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
from gops.utils.planner_benchmark.control import SimpleController
from gops.env.env_ocp.pyth_base_env import PythBaseEnv
from gops.env.env_ocp.resources.ref_traj_data import MultiRefTrajData, MultiRoadSlopeData
from gops.utils.math_utils import angle_normalize
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

# ==========================================
# 辅助类定义
@dataclass
class DynamicObstacleData:
    """动态障碍物数据结构 (从专家环境移植)"""
    x: float = 0.0
    y: float = 0.0
    phi: float = 0.0
    u: float = 0.0
    delta: float = 0.0  # 前轮转角
    l: float = 3.0  # 轴距
    dt: float = 0.1
    road_width: float = 7.0  # 新增道路半宽

    def step(self):
        # 简单的运动学模型更新
        self.x = self.x + self.u * np.cos(self.phi) * self.dt
        self.y = self.y + self.u * np.sin(self.phi) * self.dt
        self.phi = self.phi + self.u * np.tan(self.delta) / self.l * self.dt
        self.phi = angle_normalize(self.phi)
        limit_y = self.road_width - 1.8/2
        self.y = np.clip(self.y, -limit_y, limit_y)

@dataclass
class Obstacle:
    """统一障碍物接口，兼容静态和动态"""
    x: float
    y: float
    l: float = 2.0
    w: float = 2.0
    phi: float = 0.0
    u: float = 0.0
    type: str = "static" # "static" or "dynamic"
    can_cross: bool = False
    id: int = 0
    # 动态障碍物特有句柄
    dynamic_data: Optional[DynamicObstacleData] = None

class BezierGenerator:
    """
    二阶贝塞尔曲线生成器 (Quadratic Bezier)
    输入: P0(起点), P1(控制点), P2(终点)
    """
    @staticmethod
    def generate(p0, p1, p2, num_points=20):
        t = np.linspace(0, 1, num_points)
        t = t[:, np.newaxis]
        # 二阶贝塞尔公式: B(t) = (1-t)^2 * P0 + 2(1-t)t * P1 + t^2 * P2
        curve = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2
        return curve

class VelocityGenerator:
    """
    考虑曲率和车辆动力学约束的速度规划器 (修正版)
    """

    @staticmethod
    def generate_profile(path_points, v_start, v_end, dt,
                         max_lat_acc=2.5,  # 侧向加速度限制
                         max_lon_acc=2.5,  # 纵向加速限制 (稍微调低一点，更真实)
                         max_lon_dec=3.0):  # 纵向减速限制
        """
        输入:
            path_points: (N, 2) [x, y]
            v_start: 起始速度
            v_end: 期望末端速度
        """
        # --- 1. 基础几何计算 ---
        x = path_points[:, 0]
        y = path_points[:, 1]
        dx = np.gradient(x)
        dy = np.gradient(y)

        # 航向角
        phis = np.arctan2(dy, dx)
        phis = np.unwrap(phis)

        # 距离与累计路程
        dists = np.sqrt(dx ** 2 + dy ** 2)
        s_cum = np.cumsum(dists)

        # --- 2. 计算曲率 ---
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        curvature = np.abs(dx * ddy - dy * ddx) / (np.power(dx ** 2 + dy ** 2, 1.5) + 1e-6)

        # --- 3. 计算速度上界 (Speed Limits) ---

        # A. 物理/曲率限制
        v_limit_lat = np.sqrt(max_lat_acc / (curvature + 1e-6))

        # B. 绝对硬约束
        max_hard_speed = 15.0  # 可以设高一点，由 v_end 控制实际速度
        min_hard_speed = 1.0
        v_limit_lat = np.clip(v_limit_lat, min_hard_speed, max_hard_speed)

        # C. 【关键修改】 目标速度约束
        # 基础 Profile 不再是斜线，而是恒定的 v_end (受限于物理瓶颈)
        # 意思：只要弯道允许，我就想开到 v_end
        v_profile = np.minimum(v_end, v_limit_lat)

        # --- 4. 纵向平滑 (S-Curve) ---

        # (1) Forward Pass (加速能力限制)
        # 保证从 v_start 开始，加速度不超过 max_lon_acc
        v_forward = np.zeros_like(v_profile)
        v_forward[0] = v_start  # 锚定起点

        for i in range(1, len(v_profile)):
            ds = dists[i]
            # 物理允许达到的最大速度 (v^2 = u^2 + 2as)
            v_phys_max = np.sqrt(v_forward[i - 1] ** 2 + 2 * max_lon_acc * ds)

            # 最终速度是：意图/曲率限制 和 物理加速能力 的较小值
            v_forward[i] = min(v_profile[i], v_phys_max)

        # (2) Backward Pass (减速能力限制)
        # 保证为了满足未来的低速点，提前减速
        v_final = np.zeros_like(v_forward)

        # 【微调】末端速度初始化
        # 如果 v_end 比 forward[-1] 还小，说明还没减速到位，要从 v_end 开始反推
        # 通常取 Forward 的结果即可，但在特定边界条件下取 min 更稳健
        v_final[-1] = v_forward[-1]

        for i in range(len(v_profile) - 2, -1, -1):
            ds = dists[i + 1]
            # 物理允许的最大刹车前速度 (v_curr^2 = v_next^2 + 2*dec*s)
            v_phys_max = np.sqrt(v_final[i + 1] ** 2 + 2 * max_lon_dec * ds)

            # 取 Forward pass (加速限制) 和 Backward pass (减速限制) 的较小值
            v_final[i] = min(v_forward[i], v_phys_max)

        # --- 5. 组合输出 ---
        traj_with_v = np.stack([
            x,
            y,
            phis,
            v_final
        ], axis=1)

        return traj_with_v

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

    def f_xu(self, states, actions, road_info, delta_t):
        x, y, phi, u, v, w = states
        theta_r, varphi_r = road_info
        steer, a_x = actions
        k_f = self.vehicle_params["k_f"]
        k_r = self.vehicle_params["k_r"]
        l_f = self.vehicle_params["l_f"]
        l_r = self.vehicle_params["l_r"]
        m = self.vehicle_params["m"]
        I_z = self.vehicle_params["I_z"]
        g = self.vehicle_params["g"]
        next_state = [
            x + delta_t * (u * np.cos(phi) - v * np.sin(phi)),
            y + delta_t * (u * np.sin(phi) + v * np.cos(phi)),
            phi + delta_t * w,
            u + delta_t * (a_x - g * np.sin(theta_r)),
            (
                m * v * u
                + delta_t * (l_f * k_f - l_r * k_r) * w
                - delta_t * k_f * steer * u
                - delta_t * m * np.square(u) * w
                # 假设坐标系为: x前, y左, z上。
                # 如果 varphi_r > 0 代表车身向右倾斜(右低左高)，重力分力指向右侧(-y方向)
                - delta_t * m * g * np.sin(varphi_r) * u
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

class SimuVeh3dofcontiBimodalDiffusion(PythBaseEnv):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }
    def __init__(self,
                 pre_horizon: int = 20,
                 path_para: Optional[Dict[str, Dict]] = None,
                 u_para: Optional[Dict[str, Dict]] = None,
                 slope_para: Optional[Dict[str, Dict]] = None,
                 max_steer: float = np.pi / 6,
                 max_accel: float = 3.0,
                 dynamic_obstacle_num: int = 2,
                 static_obstacle_num: int = 3,
                 d_planning: float = 20.0,  # 离障碍物多少远开始规划
                 lateral_sample: float = 3.5,  # 横向采样距离
                 forward_sample: float = 20.0,  # 纵向采样距离
                 max_speed_dev: float = 2, # 最大允许偏离参考速度
                 **kwargs):
        work_space = kwargs.pop("work_space", None)
        if work_space is None:
            # initial range of [delta_x, delta_y, delta_phi, delta_u, v, w]
            init_high = np.array([2, 1, np.pi / 6, 2, 0.1, 0.1], dtype=np.float32)
            init_low = -init_high
            work_space = np.stack((init_low, init_high))
        super(SimuVeh3dofcontiBimodalDiffusion, self).__init__(work_space=work_space, **kwargs)
        self.vehicle_dynamics = VehicleDynamicsData()
        self.ref_traj = MultiRefTrajData(path_para, u_para)
        self.state_dim = 6
        self.pre_horizon = pre_horizon
        self.dt = 0.1
        self.max_episode_steps = 200
        # 控制模式开关, "planning": 输出轨迹; "control": 输入 [steer, acc] , 直接控制
        self.control_mode = kwargs.get("control_mode", "planning")
        self.max_steer = max_steer
        self.max_accel = max_accel
        self.num_speed_points = 5
        # --- 1. 预测时域与动作空间 ---
        if self.control_mode == "planning":
            self.action_dim = 1+self.num_speed_points
            low_act = np.array([-lateral_sample] + [-max_speed_dev] * self.num_speed_points, dtype=np.float32)
            high_act = np.array([lateral_sample] + [max_speed_dev] * self.num_speed_points, dtype=np.float32)
            self.action_space = spaces.Box(low=low_act, high=high_act, dtype=np.float32)

            self.controller = SimpleController(max_steer, max_accel)
        else:
            # SAC/PPO 输出直接控制量: [steer, acc]
            self.action_dim = 2
            self.action_space = spaces.Box(
                low=np.array([-self.max_steer, -self.max_accel]),
                high=np.array([self.max_steer, self.max_accel]),
                dtype=np.float32
            )
        # --- 2. 观测空间设计 (融合全局跟踪信息 + 引导信息) ---
        # A. 基础参考路径参数 (用于生成 ref_points)
        self.ref_horizon = pre_horizon
        self.ref_traj = MultiRefTrajData(path_para, u_para)
        self.road_slope = MultiRoadSlopeData(slope_para)
        # B. 障碍物参数
        self.max_obs_num = 3  # 观测中最多包含的障碍物数量
        self.dynamic_obstacle_num = dynamic_obstacle_num
        self.static_obstacle_num = static_obstacle_num
        self.n_obs_pred = 5  # 预测点数量
        self.obs_feat_dim = 8 + self.n_obs_pred * 2  # [x, y, phi, u, l, w, type, ttc]
        # C. 维度计算
        self.dim_ego = 6
        self.dim_ref = self.ref_horizon * 6
        self.dim_obstacles = self.max_obs_num * self.obs_feat_dim
        self.perception_range = 60.0
        self.guide_traj_num = 3
        self.dim_prompts = self.guide_traj_num * self.ref_horizon * 2
        self.total_obs_dim = self.dim_ego + self.dim_ref + self.dim_obstacles + self.dim_prompts
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.total_obs_dim,),
            dtype=np.float32
        )
        obs_scale_default = [1 / 100, 1 / 100, 1 / 10,
                             1 / 100, 1 / 100, 1 / 10, 1 / 10, 1 / 50, 1 / (max_accel * 100), 1 / 10]
        self.obs_scale = np.array(kwargs.get('obs_scale', obs_scale_default))
        # --- 3. 物理模型初始化 ---
        self.d_planning = d_planning  # 障碍物感知前瞻距离
        self.lateral_sample = lateral_sample  # 横向偏移量
        self.forward_sample = forward_sample
        self.safe_dist2obs = 0.2
        self.max_road_width = 7.0
        self.max_lat_acc = 2.5  # 越野环境最大横向加速度
        self.veh_width = self.vehicle_dynamics.vehicle_params["veh_width"]
        self.veh_length = self.vehicle_dynamics.vehicle_params["veh_length"]
        self.wheel_distance = self.vehicle_dynamics.vehicle_params["wheel_distance"]
        self.ground_clearance = self.vehicle_dynamics.vehicle_params["ground_clearance"]
        # 内部状态
        self.obstacles: List[Obstacle] = []
        self.guide_trajectories: List[np.ndarray] = []
        self.current_diffusion_traj = None
        self.current_planning_traj = None
        self.history_traj = []
        self.seed()
        self.state = None
        self.path_num = None
        self.u_num = None
        self.t = None
        self.info_dict = {
            "state": {"shape": (self.state_dim,), "dtype": np.float32},
            "ref_points": {"shape": (self.pre_horizon + 1, 6), "dtype": np.float32},
            "path_num": {"shape": (), "dtype": np.uint8},
            "u_num": {"shape": (), "dtype": np.uint8},
            "slope_num": {"shape": (), "dtype": np.uint8},
            "ref_time": {"shape": (), "dtype": np.float32},
            "ref": {"shape": (6,), "dtype": np.float32},
        }
        # --- 数据记录容器 ---
        self.log_data = {
            'actual_x': [],
            'actual_y': [],
            'actual_phi': [],
            'actual_u': [],
            'ref_x': [],
            'ref_y': [],
            'ref_phi': [],
            'ref_u': [],
            'err_lat': [],  # 横向误差
            'err_phi': [],  # 航向误差
            'err_u': [],  # 速度误差
            'plan_traj': []  # 记录某一帧的规划轨迹用于debug(可选)
        }

    def reset(self,
              init_state: list = None,
              ref_time: float = None,
              ref_num: int = None,
              slope_num: int = None,
              **kwargs) -> Tuple[np.ndarray, dict]:
        if ref_time is not None:
            self.t = ref_time
        else:
            self.t = 20.0 * self.np_random.uniform(0.0, 1.0)
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
            self.path_num = self.np_random.choice([0, 1, 2, 4])

        if u_num is not None:
            self.u_num = u_num
        else:
            self.u_num = 0#self.np_random.choice([0, 1])

        if slope_num is not None:
            self.slope_num = slope_num
        else:
            self.slope_num = 1#self.np_random.choice([0, 1])
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
        # 4. 生成越野障碍
        self.obstacles = self._generate_offroad_obstacles()
        self.guide_trajectories = self._generate_guidance_prompts()

        # 初始化渲染变量
        self.current_diffusion_traj = None
        self.current_planning_traj = None
        self.history_traj = [self.state[:2]] # 记录起点
        self.step_self = 0
        self.last_best_guide_idx = -1
        # 重置记录
        self.log_data = {key: [] for key in self.log_data}
        # 记录初始点
        self._log_step_data()
        return self._get_obs(), self.info

    def step(self, action):
        if np.any(np.isnan(action)):
            print("【致命错误】Agent输出了 NaN Action!")
            action = np.zeros_like(action)

        if self.control_mode == "planning":
            action = np.clip(action, self.action_space.low, self.action_space.high)
            # 1. 坐标系对齐：全部在 Ego Frame 下工作
            ego_x, ego_y, ego_phi, ego_u = self.state[0], self.state[1], self.state[2], self.state[3]
            self.speed_deltas = action[1:]

            # 1. 寻找最近的威胁障碍物
            nearest_obs = None
            min_dist = float('inf')
            for obs in self.obstacles:
                # 计算到障碍物的距离
                dist = np.linalg.norm([obs.x - ego_x, obs.y - ego_y])
                # 判定条件：
                # 1. 距离在感知范围内 (d_planning) 2. 障碍物在车前方 (简单判断 x)
                if dist < self.d_planning and obs.x > ego_x:
                    if dist < min_dist:
                        min_dist = dist
                        nearest_obs = obs
            # 2. 对轨迹规划
            # === 情况 A: 存在威胁障碍物 ===
            if nearest_obs is not None:
                y_target = nearest_obs.y + action[0]
                p0 = np.array([ego_x, ego_y])
                # 控制点 p1: 在障碍物横向截面处
                p1 = np.array([nearest_obs.x, y_target])
                # 终点 p2: 在障碍物后方，回归目标 y
                p2 = np.array([nearest_obs.x + self.forward_sample + nearest_obs.l / 2, y_target])
                # Global 坐标系轨迹
                full_curve = BezierGenerator.generate(p0, p1, p2, num_points=self.pre_horizon + 1)
                # self.current_planning_traj = full_curve
            # === 情况 B: 无威胁，自由行驶 ===
            else:
                # 获取全局参考路径在 Ego 系下的投影 (作为基准)
                ref_ego_x, ref_ego_y, _ = ego_vehicle_coordinate_transform(
                    ego_x, ego_y, ego_phi,
                    self.ref_points[:, 0], self.ref_points[:, 1], self.ref_points[:, 2]
                )
                # ==========================================
                # 2. 确定基准点 (Residual Learning)
                mid_idx = min(len(ref_ego_x) - 1, int(self.pre_horizon / 2))
                end_idx = min(len(ref_ego_x) - 1, self.pre_horizon - 1)
                p1_base = np.array([ref_ego_x[mid_idx], ref_ego_y[mid_idx]])
                p2_base = np.array([ref_ego_x[end_idx], ref_ego_y[end_idx]])
                # ==========================================
                # 3. 生成局部轨迹 (Ego Frame)
                # ==========================================
                p0 = np.array([0.0, 0.0])  # 【关键】P0 必须是局部原点
                p1 = p1_base + np.array([0.0, action[0]])  # 动作是相对于基准的偏移
                p2 = p2_base + np.array([0.0, action[0]])
                bezier_curve = BezierGenerator.generate(p0, p1, p2, num_points=self.pre_horizon + 1)
                # 立即转为 Global 轨迹
                full_curve = self._ego_to_global(
                    bezier_curve, self.state
                )
            # 3. 速度规划
            # A. 速度插值：将 5 个关键点扩展为 20 个点 (pre_horizon)
            # 使用简单的线性插值或三次样条插值
            t_knots = np.linspace(0, 1, self.num_speed_points)
            t_query = np.linspace(0, 1, self.pre_horizon + 1)

            # 插值得到每个点的 delta_v
            delta_v_profile = np.interp(t_query, t_knots, self.speed_deltas)

            # B. 叠加到全局参考速度
            # 假设 self.ref_points 是全局参考
            # 注意：取对应时间步的参考速度，而不是只取第0个
            base_ref_vs = self.ref_points[:self.pre_horizon + 1, 3]
            raw_target_vs = base_ref_vs + delta_v_profile

            # C. 速度物理安全约束
            # 就算网络想飞，物理也不允许。我们计算几何路径的曲率限制。
            # C1. 计算曲率
            dx = np.gradient(full_curve[:, 0])
            dy = np.gradient(full_curve[:, 1])
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)
            curvature = np.abs(dx * ddy - dy * ddx) / (np.power(dx ** 2 + dy ** 2, 1.5) + 1e-6)

            # C2. 计算侧向加速度限制速度
            v_limit_curve = np.sqrt(self.max_lat_acc / (curvature + 1e-6))

            # C2. 计算纵向障碍物限速
            # 如果路径前方有障碍物，这个 profile 会在障碍物前降为 0
            # v_limit_obs = self._calculate_lon_safety_profile(full_curve, self.obstacles)

            # C3. 最终融合：取所有限制的最小值
            # 逻辑：Min(网络意图, 曲率限制, 障碍物刹车限制)
            final_v_profile = np.minimum(raw_target_vs, v_limit_curve)
            # final_v_profile = np.minimum(final_v_profile, v_limit_obs)  # 加入障碍物限制

            # C4. 物理极速硬截断
            final_v_profile = np.clip(final_v_profile, 0.0, 15.0)

            # 4. 组合最终轨迹 (x, y, phi, u)
            phis = np.arctan2(dy, dx)
            phis = np.unwrap(phis)
            self.current_planning_traj = np.stack([
                full_curve[:, 0],
                full_curve[:, 1],
                phis,
                final_v_profile  # <--- 这就是包含了地形、意图和安全的最终速度
            ], axis=1)

            # ==========================================
            # 4. 追踪控制器,全局坐标系， current_planning_traj 包含[x, y, phi, u]
            # real_action = self._tracking_controller(full_curve[1:],  self.current_planning_traj[1, 3])
            lookahead_time = 1.0  # 预瞄 1.0 秒
            lookahead_step = int(lookahead_time / self.dt)
            #
            # 防止索引越界
            target_idx = min(lookahead_step, len(self.current_planning_traj) - 1)

            # 获取较远处的规划速度
            preview_target_v = self.current_planning_traj[target_idx, 3]
            real_action = self.controller.get_control(self.current_planning_traj[:, :2], preview_target_v, self.state[:3], self.state[3])
            reward = self._compute_reward(real_action)
            self.state = self.vehicle_dynamics.f_xu(self.state, real_action, self.ref_points[1,4:], self.dt)
            # ----------  完美执行 ----------
            # next_pt = self.current_planning_traj[1]
            #
            # target_x = next_pt[0]
            # target_y = next_pt[1]
            # target_phi = next_pt[2]
            # target_u = next_pt[3]
            #
            # # 估算角速度 w (用于 Observation 和 Reward 计算)
            # # w = (phi_next - phi_now) / dt
            # w_estimated = angle_normalize(target_phi - self.state[2]) / self.dt
            #
            # # 估算侧向速度 v (假设完美跟踪，侧滑角为0，v=0)
            # v_estimated = 0.0
            #
            # # --- 直接更新状态 (Teleport) ---
            # self.state = np.array([
            #     target_x,
            #     target_y,
            #     target_phi,
            #     target_u,
            #     v_estimated,
            #     w_estimated
            # ], dtype=np.float32)
            #
            # # 构造一个虚拟的 action 用于计算 Reward (防止报错)
            # # 实际上在完美执行模式下，Reward 应该主要看轨迹质量，而不是动作平滑度
            # real_action = np.array([0.0, 0.0], dtype=np.float32)
            #
            # # 计算奖励 (此时 self.state 已经是完美执行后的状态了)
            # reward = self._compute_reward(real_action)

        else:
            action = np.clip(action, self.action_space.low, self.action_space.high)
            # --- 分支 B: SAC/PPO/DSACT Controller 模式 ---
            self.current_diffusion_traj = None  # 无轨迹可视化
            self.state = self.vehicle_dynamics.f_xu(self.state, action, self.ref_points[1,4:], self.dt)
            reward = self._compute_reward(action)

        # self.t = self.t + self.dt
        # === 自适应时间步进 ===
        # 障碍物步进 (动态障碍物更新)
        self.step_self += 1
        for obs in self.obstacles:
            if obs.type == "dynamic" and obs.dynamic_data is not None:
                obs.dynamic_data.step()
                # 同步回 Obstacle 对象
                obs.x = obs.dynamic_data.x
                obs.y = obs.dynamic_data.y
                obs.phi = obs.dynamic_data.phi
                obs.u = obs.dynamic_data.u
        # 根据自车实际行驶距离，计算在参考轨迹上的投影时间流逝
        # 这样，如果车停了，参考点也会停下来等车
        dist_traveled = self.state[3] * self.dt
        # 获取当前路段的参考速度
        current_ref_v = max(self.ref_points[0, 3], 0.1)
        # 将距离换算回参考轨迹的时间进度
        self.t += dist_traveled / current_ref_v
        # 更新参考轨迹
        self._update_ref_points()

        # 5. 更新引导
        self.guide_trajectories = self._generate_guidance_prompts()
        obs = self._get_obs()
        # 6. 终止条件
        done = self.judge_done()
        if done:
            reward = reward - 100
        # 7. 时间限制 (Truncated: 步数超限)
        is_truncated = self.step_self >= self.max_episode_steps
        current_info = self.info
        if is_truncated:
            current_info["TimeLimit.truncated"] = True
            reward += 100.0  # 存活奖励
            done = True
        else:
            current_info["TimeLimit.truncated"] = False
        # 8. 记录历史轨迹用于渲染 (在 __init__ 中记得初始化 self.history_traj = [])
        if not hasattr(self, 'history_traj'): self.history_traj = []
        self.history_traj.append(self.state[:2])
        if len(self.history_traj) > self.max_episode_steps: self.history_traj.pop(0)

        if np.isnan(reward) or np.isinf(reward):
            print("Warning: Reward is NaN/Inf! Resetting to -10.0")
            # reward = -10.0  # 兜底防止崩盘

        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            print("Warning: Obs contains NaN/Inf!")
            # obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        # --- 在 step 函数末尾，return 之前，加入数据记录 ---
        self._log_step_data()
        return obs, reward, done, current_info

    def _get_obs(self):
        # 1. 坐标变换: 全局参考点 -> 局部 Ego 坐标系
        ref_x_tf, ref_y_tf, ref_phi_tf = ego_vehicle_coordinate_transform(
            self.state[0], self.state[1], self.state[2],
            self.ref_points[:, 0], self.ref_points[:, 1], self.ref_points[:, 2],
        )
        ref_u_tf = self.ref_points[:, 3] - self.state[3]

        # 2. Ego Obs
        ego_obs = np.concatenate(
            ([ref_x_tf[0]*self.obs_scale[0], ref_y_tf[0]*self.obs_scale[1], ref_phi_tf[0]*self.obs_scale[2], ref_u_tf[0]*self.obs_scale[3]], self.state[4:])
        )
        # 3. Ref Preview Obs
        ref_obs = np.stack((ref_x_tf*self.obs_scale[0], ref_y_tf*self.obs_scale[1], ref_phi_tf*self.obs_scale[2], ref_u_tf*self.obs_scale[3], self.ref_points[:, 4], self.ref_points[:, 5]), 1)[1:].flatten()

        # 4. Obstacle Obs
        obs_feats = []
        dists = []

        # 设定感知半径 (例如 60米)
        # 只有在视野内的障碍物才值得关注

        for obs in self.obstacles:
            d = np.sqrt((obs.x - self.state[0]) ** 2 + (obs.y - self.state[1]) ** 2)
            if d < self.perception_range and obs.x > self.state[0] and abs(obs.y - self.state[1]) < 3.5:
                dists.append((d, obs))

        # 按距离排序，优先关注最近的
        dists.sort(key=lambda x: x[0])

        # 定义预测的时间步长索引 (例如: 取未来 20 步中的 5 个均匀点)
        # 假设 pre_horizon=20, dt=0.1, 则预测未来 2.0s
        # pred_indices = [4, 8, 12, 16, 20] (对应 0.4s, 0.8s, ..., 2.0s)
        pred_indices = np.linspace(1, self.pre_horizon, self.n_obs_pred, dtype=int)

        # 填充特征向量
        for i in range(self.max_obs_num):
            if i < len(dists):
                d, obs = dists[i]
                ox_tf, oy_tf, ophi_tf = ego_vehicle_coordinate_transform(
                    self.state[0], self.state[1], self.state[2],
                    np.array([obs.x]), np.array([obs.y]), np.array([obs.phi])
                )
                # 增加 u 的相对速度
                u_rel = obs.u - self.state[3]
                #TTC (Time-To-Collision) 特征
                ttc = 100.0
                if (ox_tf[0] > 0 and u_rel < -0.1) or (ox_tf[0] < 0 and u_rel > 0.1):
                    ttc = d / (abs(u_rel) + 1e-5)
                ttc = np.clip(ttc, 0, 10.0)
                # [x, y, phi, u, l, w, type, dist]
                base_feat = [ox_tf[0]/self.perception_range, oy_tf[0]/self.perception_range, ophi_tf[0]*self.obs_scale[2],
                        u_rel*self.obs_scale[3], obs.l*self.obs_scale[2], obs.w*self.obs_scale[2], 0.0 if obs.can_cross else 1.0, ttc*self.obs_scale[2]]

                # --- B. 未来轨迹预测特征 (核心修改) ---
                pred_feats = []
                for k in pred_indices:
                    delta_t = k * self.dt
                    if obs.type == "dynamic":
                        # 恒定速度模型预测 (Constant Velocity)
                        pred_x_global = obs.x + obs.u * np.cos(obs.phi) * delta_t
                        pred_y_global = obs.y + obs.u * np.sin(obs.phi) * delta_t
                    else:
                        # 静态障碍物位置不变
                        pred_x_global = obs.x
                        pred_y_global = obs.y

                    # 关键：必须转换到 *当前* 自车坐标系下
                    # (Agent 需要知道：相对于我现在的位置，它未来会在哪里)
                    pred_x_tf, pred_y_tf, _ = ego_vehicle_coordinate_transform(
                        self.state[0], self.state[1], self.state[2],
                        np.array([pred_x_global]), np.array([pred_y_global]), np.array([0.0])  # 角度不重要
                    )

                    # 归一化并加入特征
                    pred_feats.append(pred_x_tf[0] * self.obs_scale[0])
                    pred_feats.append(pred_y_tf[0] * self.obs_scale[1])

                # 合并基础特征和预测特征
                feat = base_feat + pred_feats

            else:
                # 如果视野内没有障碍物，或者不足 max_obs_num 个，用 0 填充
                # 这告诉网络：“这里没有东西，是安全的”
                feat = [0.0] * self.obs_feat_dim

            obs_feats.extend(feat)

        obstacle_obs = np.array(obs_feats, dtype=np.float32)

        # 5. Prompts Obs
        prompt_feats = []
        for traj in self.guide_trajectories:
            px_tf, py_tf, _ = ego_vehicle_coordinate_transform(
                self.state[0], self.state[1], self.state[2],
                traj[:, 0], traj[:, 1], np.zeros(len(traj))
            )
            prompt_feats.append(np.stack([px_tf*self.obs_scale[0], py_tf*self.obs_scale[1]], axis=1).flatten())

        while len(prompt_feats) < self.guide_traj_num:
            # 不可跨越，补充一维0
            prompt_feats.append(np.zeros(self.ref_horizon * 2))
        prompt_obs = np.concatenate(prompt_feats)

        return np.concatenate((ego_obs, ref_obs, obstacle_obs, prompt_obs))

    def _update_ref_points(self):
        # 整体前移一位
        self.ref_points[:-1] = self.ref_points[1:]
        # 计算新点
        future_t = self.t + self.ref_horizon * self.dt
        new_ref_point = np.array([
            self.ref_traj.compute_x(future_t, self.path_num, self.u_num),
            self.ref_traj.compute_y(future_t, self.path_num, self.u_num),
            self.ref_traj.compute_phi(future_t, self.path_num, self.u_num),
            self.ref_traj.compute_u(future_t, self.path_num, self.u_num),
            self.road_slope.compute_longislope(self.t + self.pre_horizon * self.dt, self.slope_num),
            self.road_slope.compute_latslope(self.t + self.pre_horizon * self.dt, self.slope_num)
        ], dtype=np.float32)

        self.ref_points[-1] = new_ref_point

    def _compute_reward(self, action):
        if self.control_mode == "planning":
            if self.current_planning_traj is None:
                return -1.0

            agent_planning_traj = self.current_planning_traj

            # --- 1. 智能引导奖励 (Smart Guidance + Heading Alignment) ---
            max_guide_score = -float('inf')
            valid_guide_found = False
            best_guide_traj = None
            ego_pos = self.state[:2]
            current_best_idx = -1
            r_heading = 0.0

            for i, guide_traj in enumerate(self.guide_trajectories):
                # --- A. 安全性检查 ---
                if self._check_traj_collision(guide_traj, margin=1.5, return_cost=False):
                    continue

                valid_guide_found = True

                # --- B. 评分机制 ---
                # [修正2] 只计算引导线起始位置与自车的距离，作为选择依据
                # 这样能选出"入口"离我最近的那条线
                dist_to_start = np.linalg.norm(guide_traj[0, :2] - ego_pos)

                # 基础分：距离越近分越高
                score = -dist_to_start

                if score > max_guide_score:
                    max_guide_score = score
                    best_guide_traj = guide_traj
                    current_best_idx = i

            if not valid_guide_found:
                r_guidance = -1.0  # [建议] 没路走的时候给一个固定的较大惩罚，而不是-1
                r_heading = -1.0
            elif best_guide_traj is not None:
                # 截取长度对齐
                min_len = min(len(agent_planning_traj), len(best_guide_traj))

                # 位置偏差
                diff_vec = agent_planning_traj[:min_len, :2] - best_guide_traj[:min_len, :2]

                # [修正3] 建议使用 mean 而不是 sum，防止步长变化影响 reward 尺度
                # 如果保持 sum，系数 -0.5 可能偏小（取决于单位），改成 mean 后 -0.5 代表平均偏离1米扣0.5分
                # 这里我建议稍微加大权重，因为这是核心任务
                cum_dist_mean = np.mean(np.linalg.norm(diff_vec, axis=1))

                # 航向偏差
                agent_diff = np.diff(agent_planning_traj[:min_len, :2], axis=0)
                guide_diff = np.diff(best_guide_traj[:min_len, :2], axis=0)

                agent_phi = np.arctan2(agent_diff[:, 1], agent_diff[:, 0])
                guide_phi = np.arctan2(guide_diff[:, 1], guide_diff[:, 0])
                phi_error = angle_normalize(agent_phi - guide_phi)

                # --- 奖励赋值 ---
                # [修正1] 致命错误修复：这里必须用 cum_dist_mean (或者 cum_dist)，绝对不能用 min_cum_dist_score
                r_guidance = -0.5 * cum_dist_mean  # 权重建议加到 2.0，强调贴合

                # 航向奖励
                r_heading = -0.1 * np.mean(np.square(phi_error))  # 同样建议用 mean
            # # --- 2. 轨迹一致性奖励  ---
            r_consistency = 0.0
            if self.last_best_guide_idx != -1 and current_best_idx != -1:
                if current_best_idx == self.last_best_guide_idx:
                    # 情况 A: 保持在同一条线 -> 给一点点奖励鼓励稳定
                    r_consistency = 0.5
                else:
                    # 情况 B: 发生了切换-> 给惩罚
                    # 这个惩罚不能太大，否则遇到障碍物它也不敢换道
                    # 也不能太小，否则它会左右横跳
                    r_consistency = -0.5
                    # [关键] 更新状态供下一帧使用
            self.last_best_guide_idx = current_best_idx
            
            # --- 3. 效率奖励 (Efficiency Reward) ---
            # 如果前方很近有障碍物，降低对速度的要求
            # 简单做法：计算与最近前方障碍物的距离
            min_front_dist = float('inf')
            for obs in self.obstacles:
                # 简单筛选前方障碍物
                if 0 < obs.x - self.state[0] < 40 and abs(obs.y - self.state[1]) < 3.0:
                    dist = np.linalg.norm([obs.x - self.state[0], obs.y - self.state[1]])
                    if dist < min_front_dist:
                        min_front_dist = dist

            # 动态调整期望速度权重
            if min_front_dist < 15.0:
                # 如果前方15米有障碍物，速度奖励降权，或者不惩罚低速
                r_efficiency = 0.0
            else:
                # 前方开阔，鼓励加速
                r_efficiency = 0.1 * self.state[3]


            # ----4. 稳定性奖励(Lateral Stability Reward) -------
            # a_lat = u^2 * curvature. 近似为 u * w (角速度)
            # 惩罚高速过弯
            ego_u = self.state[3]
            ego_w = self.state[5]
            lat_acc = ego_u * ego_w
            r_stability = -0.5 * (lat_acc ** 2)

            # --- 5. 非碰撞奖励-------
            # violation > 0: 撞了
            # violation = -2.0: 离障碍物边界还有 2米
            traj_violation = self._check_traj_collision(
                agent_planning_traj,
                margin=0.0,  # 这里的margin给0，让网络自己学距离
                return_cost=True
            )

            # 2. 设计 Soft Barrier Reward
            # 逻辑：
            # 当 violation < -3.0 (非常远) -> 惩罚 ≈ 0
            # 当 violation = -1.0 (靠近了) -> 惩罚开始增加
            # 当 violation > 0.0 (撞了)   -> 惩罚巨大

            # 裁剪一下，防止太远的地方产生微小梯度干扰
            # 只关注 3米以内的风险
            effective_violation = max(traj_violation, -0.5)

            # 使用 tanh 平滑过渡
            # tanh(x + 3) 使得 -3 时为 0，0 时为 0.99
            r_collision_risk = np.tanh(effective_violation + 0.5)

            # 如果真的预测会撞上 (violation > 0)，额外叠加重罚
            if traj_violation > 0:
                r_collision_risk += traj_violation * 10.0  # 线性增加，撞得越深罚得越重

            # 3. 最终加权
            # 给一个较大的系数，比如 -20
            r_collision_traj = -0.5 * r_collision_risk

            # 获取侵入量 (Violation)
            # 正值表示危险，负值表示安全
            violation = self._check_ego_collision(return_cost=True)
            # --- Soft Barrier Penalty (核心修改) ---
            # 设定一个“关注范围” (Attention Region)
            # 例如：当 violation > -3.0 (即距离障碍物 3米以内) 时开始给压力

            # 归一化因子：控制惩罚上升的陡峭程度
            # 这种设计下：
            # violation = -5.0 -> punish ≈ 0
            # violation = -1.0 -> punish ≈ 0.5 (开始警告)
            # violation =  0.0 -> punish = 1.0 (进入 Buffer)
            # violation >  0.0 -> punish > 1.0 (撞击)

            # 使用你提供的公式变体:
            # punish = tanh(violation + offset) + 1
            # 使得在很远的地方 punish 为 0，靠近时平滑上升

            # 1. 裁剪一下，太远了就不算了，防止 exp 溢出或梯度消失
            # 我们只关心 violation > -2.0 的情况
            effective_violation = max(violation, -0.5)

            # 2. 计算惩罚项 (范围 0 ~ 2.0)
            # 当 violation = -2 时，tanh(-2 + 2) = 0 -> punish = 0
            # 当 violation = 0 时，tanh(0 + 2) = 0.96 -> punish ≈ 1.0
            r_collision_risk = np.tanh(effective_violation + 0.5)

            # 如果真的撞了 (violation > 0)，额外叠加线性惩罚
            if violation > 0:
                r_collision_risk += violation * 5.0  # 撞得越深罚得越重

            # 3. 加权 (系数根据需要调整，建议给大一点)
            r_collision_cost = -1.2 * r_collision_risk
            # print(r_collision_ego)
            # --- 6. 坡度安全奖励 ---
            # 坡度大时鼓励减速
            current_slope = self.ref_points[0, 4]  # 纵坡
            v = self.state[3]
            # 下坡(slope < 0)且超速 -> 重罚
            if current_slope < -0.1 and v > 15:
                r_slope_safety = -1.0 * (v - 15) ** 2
            # 上坡(slope > 0)且龟速 -> 惩罚 (鼓励加油)
            elif current_slope > 0.1 and v < 3:
                r_slope_safety = -0.5 * (3 - v) ** 2
            else:
                r_slope_safety = 0.0

            # --- 7.速度平滑性 --
            diffs = np.diff(self.speed_deltas)
            r_speed_smooth = -0.05 * np.sum(np.square(diffs))

            # ----8.道路边界惩罚----
            # 计算自车相对于参考线的横向偏差
            ref_y = self.ref_points[0, 1]
            ego_y = self.state[1]
            lat_dev = abs(ego_y - ref_y)
            # 定义安全区 (比如 6.0m)。超过 6.0m 就开始扣分
            r_boundary = 0.0
            if lat_dev > self.max_road_width - 1.0:
                # 二次惩罚：越远扣得越狠
                r_boundary = -1.5 * ((lat_dev - (self.max_road_width - 1.0)) ** 2)
            return r_efficiency + r_guidance + r_heading + r_collision_cost+r_collision_traj +r_boundary+r_speed_smooth+r_consistency+r_stability+r_slope_safety
        else:
            x, y, phi, u, _, w = self.state
            ref_x, ref_y, ref_phi, ref_u = self.ref_points[0]
            steer, a_x = action
            return -(
                    0.04 * (x - ref_x) ** 2
                    + 0.04 * (y - ref_y) ** 2
                    + 0.02 * angle_normalize(phi - ref_phi) ** 2
                    + 0.02 * (u - ref_u) ** 2
                    + 0.01 * w ** 2
                    + 0.01 * steer ** 2
                    + 0.01 * a_x ** 2
            )

    def _generate_offroad_obstacles(self):
        obs_list = []
        # 1. 动态障碍物
        if self.path_num == 3:
            # circle path
            dynamic_delta = -np.arctan2(DynamicObstacleData.l, self.ref_traj.ref_trajs[3].r)
        else:
            dynamic_delta = 0.0

        for i_dynamic in range(self.dynamic_obstacle_num):
            delta_t = self.np_random.uniform(3, 8)  # 稍微放宽范围

            delta_lon = 1.0 * self.np_random.uniform(-1, 1)
            delta_lat = 1.0 * self.np_random.uniform(-2.0, 2.0)  # 限制在路宽范围内
            delta_phi = 1.0 * self.np_random.uniform(0, np.pi)
            dynamic_x = self.ref_traj.compute_x(self.t + delta_t, self.path_num, self.u_num) + delta_lon
            dynamic_y = self.ref_traj.compute_y(self.t + delta_t, self.path_num, self.u_num) + delta_lat
            dynamic_u = self.np_random.uniform(2, 8)  # 速度随机
            dynamic_phi = self.ref_traj.compute_phi(self.t + delta_t, self.path_num, self.u_num) + delta_phi
            dyn_data = DynamicObstacleData(
                x=dynamic_x, y=dynamic_y, phi=dynamic_phi, u=dynamic_u, delta=dynamic_delta, dt=self.dt
            )
            obs_list.append(Obstacle(
                x=dynamic_x, y=dynamic_y, phi=dynamic_phi, u=dynamic_u,
                l=4.8, w=2.0, type="dynamic", can_cross=False,
                id=i_dynamic, dynamic_data=dyn_data
            ))

        # 2. 静态障碍物
        for i_static in range(self.static_obstacle_num):
            delta_t = self.np_random.uniform(3, 10)  # 放在稍远一点
            static_obs_phi = self.ref_traj.compute_phi(self.t + delta_t, self.path_num, self.u_num)

            delta_lon = 1.0 * self.np_random.uniform(-2, 2)
            delta_lat = 1.0 * self.np_random.uniform(-3.0, 3.0)
            static_obs_x = self.ref_traj.compute_x(self.t + delta_t, self.path_num, self.u_num) + delta_lon
            static_obs_y = self.ref_traj.compute_y(self.t + delta_t, self.path_num, self.u_num) + delta_lat

            static_length = self.np_random.uniform(1.0, 3.0)
            static_width = self.np_random.uniform(0.2, 2.0)
            static_height = self.np_random.uniform(0.1, 0.6)

            # 判断可跨越性
            can_cross = (static_height < self.ground_clearance and static_width < self.wheel_distance)

            obs_list.append(Obstacle(
                x=static_obs_x, y=static_obs_y, phi=static_obs_phi, u=0.0,
                l=static_length, w=static_width, type="static",
                can_cross=can_cross, id=self.dynamic_obstacle_num + i_static
            ))

        return obs_list

    def _generate_offroad_obstacles_fix(self):
        obs_list = []
        ego_x = self.state[0]

        # 1. 静态障碍 - 不可跨 (大石头)
        obs_list.append(Obstacle(
            x=ego_x + 30, y=1.5, l=2.0, w=2.0,
            type="static", can_cross=False, id=1
        )) # 30, 0,

        # 2. 静态障碍 - 可跨 (小土堆/水坑)
        obs_list.append(Obstacle(
            x=ego_x + 100, y=0.0, l=3.0, w=3.0,
            type="static", can_cross=True, id=2
        )) # 60, 0

        # 3. 动态障碍 (同向慢车 - 视为不可跨)
        dyn_data = DynamicObstacleData(
            x=ego_x + 80, y=0, phi=np.pi, u=8.0
        )# 10, 0, 8.0
        obs_list.append(Obstacle(
            x=dyn_data.x, y=dyn_data.y, l=4.0, w=2.0,
            u=dyn_data.u, type="dynamic", can_cross=False, id=3,
            dynamic_data=dyn_data
        ))
        return obs_list

    # def _generate_guidance_prompts(self):
    #     """
    #     智能引导生成逻辑 (几何增强版)：
    #     基于障碍物局部坐标系生成精确的绕行/跨越轨迹，并进行道路边界过滤。
    #     """
    #     ego_x, ego_y = self.state[0], self.state[1]
    #
    #     # 1. 寻找最近的威胁障碍物
    #     nearest_obs = None
    #     min_dist = float('inf')
    #     for obs in self.obstacles:
    #         # 计算到障碍物的距离
    #         dist = np.linalg.norm([obs.x - ego_x, obs.y - ego_y])
    #         # 判定条件：
    #         # 1. 距离在感知范围内
    #         # 2. 障碍物在车前方 (简单判断 x)
    #         # 3. 障碍物在横向威胁范围内 (比如左右各3.5m内)
    #         if dist < self.d_planning and obs.x > ego_x and abs(obs.y - ego_y) < 3.5:
    #             if dist < min_dist:
    #                 min_dist = dist
    #                 nearest_obs = obs
    #
    #
    #     prompts = []
    #
    #     # === 情况 A: 存在威胁障碍物 ===
    #     if nearest_obs is not None:
    #         # --- 1. 定义局部坐标系下的关键点偏移 ---
    #         # 格式: (local_x_offset, local_y_offset)
    #         # local_x 沿障碍物朝向，local_y 垂直于障碍物朝向(左正右负)
    #
    #         offset_dist = self.lateral_sample + nearest_obs.w / 2 + self.veh_width / 2
    #
    #         # 候选偏移列表：[右绕, (直行), 左绕]
    #         local_candidates = []
    #
    #         # A. 右绕 (Right)
    #         local_candidates.append({
    #             "type": "right",
    #             "p1_local": np.array([0.0, -offset_dist]),
    #             "p2_local": np.array([self.forward_sample + nearest_obs.l / 2, -offset_dist])
    #         })
    #
    #         # B. 直行/跨越 (Center) - 仅当可跨越时添加
    #         if nearest_obs.can_cross:
    #             local_candidates.append({
    #                 "type": "center",
    #                 "p1_local": np.array([0.0, 0.0]),
    #                 "p2_local": np.array([self.forward_sample + nearest_obs.l / 2, 0.0])
    #             })
    #
    #         # C. 左绕 (Left)
    #         local_candidates.append({
    #             "type": "left",
    #             "p1_local": np.array([0.0, offset_dist]),
    #             "p2_local": np.array([self.forward_sample + nearest_obs.l / 2, offset_dist])
    #         })
    #
    #         # --- 2. 坐标变换 (Local -> Global) 并过滤 ---
    #         # 旋转矩阵
    #         c, s = np.cos(nearest_obs.phi), np.sin(nearest_obs.phi)
    #         rot_mat = np.array([[c, -s], [s, c]])
    #         obs_center = np.array([nearest_obs.x, nearest_obs.y])
    #
    #         valid_candidates = []
    #
    #         # 道路边界阈值 (硬约束)
    #         road_limit = self.max_road_width - 1.0  # 留1m余量
    #
    #         for item in local_candidates:
    #             # 变换 P1 (控制点)
    #             # Global = Obs_Center + Rot * Local
    #             p1_global = obs_center + rot_mat @ item["p1_local"]
    #
    #             # 变换 P2 (终点)
    #             p2_global = obs_center + rot_mat @ item["p2_local"]
    #
    #             # 保存变换后的点
    #             item["p1_global"] = p1_global
    #             item["p2_global"] = p2_global
    #
    #             # 【关键修改】基于 Global Y 坐标进行边界过滤
    #             # 检查 P1 和 P2 是否都在道路范围内
    #             if abs(p1_global[1]) < road_limit and abs(p2_global[1]) < road_limit:
    #                 valid_candidates.append(item)
    #
    #             # --- 3. 兜底策略 ---
    #             # 如果所有路径都被封死 (比如左右都出界)，保留道路中心线作为引导
    #             if not valid_candidates:
    #                 # 我们直接从 self.ref_points 中提取点
    #                 # ref_points 包含了未来的参考路径点 [x, y, phi, u, ...]
    #
    #                 # 计算索引 (防止越界)
    #                 idx_mid = min(len(self.ref_points) - 1, int(self.pre_horizon / 2))
    #                 idx_end = min(len(self.ref_points) - 1, self.pre_horizon)
    #
    #                 # P1: 取参考路径的中段点
    #                 p1_center = self.ref_points[idx_mid, :2]
    #                 # P2: 取参考路径的末端点
    #                 p2_center = self.ref_points[idx_end, :2]
    #
    #                 # 构造 fallback item
    #                 fallback_item = {
    #                     "type": "fallback_center",
    #                     "p1_local": np.zeros(2),  # 占位，兜底策略不依赖局部坐标
    #                     "p2_local": np.zeros(2),  # 占位
    #                     "p1_global": p1_center,
    #                     "p2_global": p2_center
    #                 }
    #                 valid_candidates.append(fallback_item)
    #
    #         # --- 4. 生成贝塞尔曲线 ---
    #         for item in valid_candidates:
    #             p0 = np.array([ego_x, ego_y])
    #             p1 = item["p1_global"]
    #             p2 = item["p2_global"]
    #
    #             curve = BezierGenerator.generate(p0, p1, p2, num_points=self.ref_horizon)
    #             prompts.append(curve)
    #
    #     # === 情况 B: 无威胁，自由行驶 ===
    #     else:
    #         # 优化：生成左、中、右三条车道线
    #         lane_offsets = [self.lateral_sample, 0.0, -self.lateral_sample]
    #         for offset in lane_offsets:
    #             traj_points = []
    #             for i in range(self.ref_horizon):
    #                 rx = self.ref_points[i, 0]
    #                 ry = self.ref_points[i, 1]
    #                 rphi = self.ref_points[i, 2]
    #                 # 法向平移
    #                 nx = -np.sin(rphi)
    #                 ny = np.cos(rphi)
    #                 px = rx + nx * offset
    #                 py = ry + ny * offset
    #                 traj_points.append([px, py])
    #             prompts.append(np.array(traj_points))
    #
    #     return prompts

    def _generate_guidance_prompts(self):
        """
        智能引导生成逻辑 (带边界过滤版)：
        1. 寻找最近的威胁障碍物。
        2. 生成绕行轨迹。
        3. 【新增】过滤掉超出道路边界的引导线，强制 Agent 走那边。
        """
        ego_x, ego_y = self.state[0], self.state[1]

        # 1. 寻找最近的威胁障碍物
        nearest_obs = None
        min_dist = float('inf')
        for obs in self.obstacles:
            # 计算到障碍物的距离
            dist = np.linalg.norm([obs.x - ego_x, obs.y - ego_y])
            # 判定条件：
            # 1. 距离在感知范围内 (d_planning) 2. 障碍物在车前方 (简单判断 x)
            if dist < self.d_planning and obs.x > ego_x and abs(obs.y - ego_y) < 3.5:
                if dist < min_dist:
                    min_dist = dist
                    nearest_obs = obs


        prompts = []

        # === 情况 A: 存在威胁障碍物 ===
        if nearest_obs is not None:
            # 基础偏移量，这里我们基于障碍物的 y 中心生成偏移
            obs_y = nearest_obs.y
            # 可跨越则添加直行
            if nearest_obs.can_cross:
                # 【可跨越】：生成 左、中、右 三条
                lateral_offsets = [
                    obs_y + (- self.lateral_sample - nearest_obs.w / 2 - self.veh_width / 2)*np.cos(nearest_obs.phi),  # 右绕
                    obs_y,  # 直接跨越
                    obs_y + (self.lateral_sample + nearest_obs.w / 2 + self.veh_width / 2)*np.cos(nearest_obs.phi)  # 左绕
                ]

            else:
                # 【不可跨】：只生成 左、右 两条
                lateral_offsets = [
                    obs_y +(- self.lateral_sample-nearest_obs.w/2-self.veh_width/2)*np.cos(nearest_obs.phi),  # 右绕
                    obs_y + (self.lateral_sample + nearest_obs.w / 2 + self.veh_width / 2)*np.cos(nearest_obs.phi)  # 左绕
                ]

            # 【关键修改】过滤掉超出道路边界的偏移量
            valid_offsets = [y for y in lateral_offsets if abs(y) < self.max_road_width-1.0]#offsets#

            # --- 3. 兜底策略 ---
            # # 如果所有路径都被封死 (比如左右都出界)，保留道路中心线作为引导
            # --- 3. 兜底策略 ---
            # 如果所有路径都被封死 (比如左右都出界)，保留道路中心线作为引导
            if not valid_offsets:
                # 这里的 ref_points 是当前时刻往后预测的参考点数组，形状 (pre_horizon+1, 6)
                # 我们直接截取一段作为引导线。
                # ref_points 的列结构通常是: [x, y, phi, u, slope_lon, slope_lat]

                # 提取前 pre_horizon 个点的位置坐标 [x, y]
                # 这种方式生成的引导线天然就是沿着道路中心线的
                fallback_curve = self.ref_points[:self.ref_horizon, :2]

                prompts.append(fallback_curve)

            else:
                # 如果有合法的偏移量，则正常生成贝塞尔曲线
                for y_target in valid_offsets:
                    p0 = np.array([ego_x, ego_y])
                    p1 = np.array([nearest_obs.x, y_target])
                    # 终点 p2: 在障碍物后方
                    p2 = np.array([nearest_obs.x + self.forward_sample + nearest_obs.l / 2, y_target])

                    curve = BezierGenerator.generate(p0, p1, p2, num_points=self.ref_horizon)
                    prompts.append(curve)

        # === 情况 B: 无威胁，自由行驶 ===
        else:
            # 优化：生成左、中、右三条车道线，让 Agent 自己选一条最近的去跟
            # 这样如果它刚绕过障碍物在左边，它会自然地选择左车道线继续开，而不是被强行拉回中间
            lane_offsets = [self.lateral_sample, 0.0, -self.lateral_sample]
            for offset in lane_offsets:
                traj_points = []
                for i in range(self.ref_horizon):
                    rx = self.ref_points[i, 0]
                    ry = self.ref_points[i, 1]
                    rphi = self.ref_points[i, 2]
                    # 法向平移
                    nx = -np.sin(rphi)
                    ny = np.cos(rphi)
                    px = rx + nx * offset
                    py = ry + ny * offset
                    traj_points.append([px, py])
                prompts.append(np.array(traj_points))
        return prompts

    def _check_ego_collision(self, return_cost=False):
        """
        计算碰撞侵入量 (Violation)。
        返回:
            max_violation: float.
                           > 0 表示碰撞深度；
                           < 0 表示最小安全距离的负值 (越接近0越危险)。
        """
        # --- 1. 准备自车双圆中心 ---
        x, y, phi = self.state[:3]
        # 自车双圆半径 (覆盖矩形)
        r_ego = np.sqrt(2) / 2 * self.veh_width
        # 圆心偏移量
        d_ego = (self.veh_length - self.veh_width) / 2

        ego_centers = np.array([
            [x + d_ego * np.cos(phi), y + d_ego * np.sin(phi)],  # 前圆
            [x - d_ego * np.cos(phi), y - d_ego * np.sin(phi)]  # 后圆
        ])  # shape: (2, 2)

        # 初始化最大侵入量为负无穷 (表示非常安全)
        # 我们寻找的是最危险的情况 (距离最小 -> violation 最大)
        max_violation = -np.inf
        is_collision = False

        for obs in self.obstacles:
            if obs.can_cross:
                continue

            # --- 2. 准备障碍物双圆中心 ---
            # 假设障碍物也是车辆形状，用同样的双圆模型
            # 如果障碍物尺寸不同，需要单独计算它的 d 和 r
            r_obs = np.sqrt(obs.l ** 2 + obs.w ** 2) / 4.0  # 简化估算，假设障碍物用两个圆覆盖
            d_obs = (obs.l - obs.w) / 2.0 if obs.l > obs.w else 0.0

            obs_x, obs_y, obs_phi = obs.x, obs.y, obs.phi
            obs_centers = np.array([
                [obs_x + d_obs * np.cos(obs_phi), obs_y + d_obs * np.sin(obs_phi)],
                [obs_x - d_obs * np.cos(obs_phi), obs_y - d_obs * np.sin(obs_phi)]
            ])

            # --- 3. 计算双圆对双圆的距离矩阵 (2x2) ---
            # ego_centers: (2, 2), obs_centers: (2, 2)
            # 扩展维度进行广播计算
            # dists shape: (2, 2) -> [ego_front/rear, obs_front/rear]
            dists = np.linalg.norm(
                ego_centers[:, np.newaxis, :] - obs_centers[np.newaxis, :, :],
                axis=2
            )

            # 找到自车和该障碍物之间最近的一对圆的距离
            min_dist_pair = np.min(dists)

            # 侵入量 = (自车半径 + 障碍物半径 + 安全余量) - 实际距离
            # 正值代表危险/碰撞，负值代表安全
            # 这里我们把 safe_dist2obs 加进去，表示进入这个 buffer 就算 violation > 0
            # 动态buffer：如果需要考虑速度
            safety_buffer = 0.0
            if obs.type == "dynamic":
                safety_buffer = 0.3 * abs(obs.u - self.state[3])

            current_violation = (r_ego + r_obs + self.safe_dist2obs + safety_buffer) - min_dist_pair

            # 更新全局最大威胁
            if current_violation > max_violation:
                max_violation = current_violation

            # 判断是否真的发生了物理碰撞 (不包含 buffer)
            if min_dist_pair < (r_ego + r_obs):
                is_collision = True

        # --- 4. 返回逻辑 ---
        if return_cost:
            # 返回最危险的那个 violation 值（可以是负数）给 Reward 函数用
            # 如果没有任何障碍物 (max_violation仍为-inf)，返回一个足够安全的负值 (比如 -10.0)
            if max_violation == -np.inf:
                return -10.0
            return max_violation
        else:
            # Done 判定逻辑：只关心是否发生了物理碰撞
            return is_collision

    def _check_traj_collision(self, traj_points, margin=0.0, return_cost=False):
        """
        计算轨迹的碰撞风险值。

        Args:
            traj_points: 规划出的轨迹点 [N, ...]
            margin: 安全边界余量
            return_cost:
                False -> 仅检测是否碰撞 (返回 Bool)
                True  -> 返回具体的风险数值 (float)

        Returns:
            如果 return_cost=True:
                返回 max_violation (float)。
                > 0 : 表示轨迹中至少有一个点会撞，数值越大撞得越深。
                < 0 : 表示轨迹全程安全，数值代表“最小安全距离”的负值。
            如果 return_cost=False:
                返回 True (撞了) / False (没撞)
        """
        if traj_points is None or len(traj_points) == 0:
            if return_cost: return -10.0  # 默认安全
            return False

        # 生成时间步 (N,)
        t_steps = np.linspace(0, self.pre_horizon * self.dt, len(traj_points))

        # 初始化最大侵入量为负无穷 (代表极度安全)
        # 我们要找的是“最危险的那一刻”，即 violation 最大的值
        global_max_violation = -np.inf
        has_collision = False

        for obs in self.obstacles:
            if obs.can_cross:
                continue

            # --- 1. 预测障碍物未来轨迹 ---
            # 简单匀速直线运动预测
            obs_future_x = obs.x + obs.u * np.cos(obs.phi) * t_steps
            obs_future_y = obs.y + obs.u * np.sin(obs.phi) * t_steps

            # 道路边界截断 (防止预测跑到路外面去)
            limit_y = self.max_road_width
            obs_future_y = np.clip(obs_future_y, -limit_y, limit_y)

            # 障碍物未来轨迹 [N, 2]
            obs_future_traj = np.stack([obs_future_x, obs_future_y], axis=1)

            # --- 2. 计算距离矩阵 [N,] ---
            # 计算规划轨迹每个点 到 对应时刻障碍物位置 的距离
            dists = np.linalg.norm(traj_points[:, :2] - obs_future_traj, axis=1)

            # --- 3. 计算阈值 ---
            # 动态安全阈值
            base_threshold = self.veh_width / 2.0 + obs.w / 2.0 + margin
            safety_buffer = 0.0
            if obs.type == "dynamic":
                # 动态障碍物额外加一点 buffer
                safety_buffer = 0.5 * abs(obs.u - self.state[3])

            safe_threshold = base_threshold + safety_buffer

            # --- 4. 计算侵入量 (Violation) ---
            # Violation = 阈值 - 实际距离
            # V > 0: 危险/碰撞 (距离 < 阈值)
            # V < 0: 安全 (距离 > 阈值)
            violation_array = safe_threshold - dists

            # 找出针对当前这个障碍物，最危险的那个时刻
            current_obs_max_risk = np.max(violation_array)

            # 更新全局最大风险
            if current_obs_max_risk > global_max_violation:
                global_max_violation = current_obs_max_risk

            # 如果是检测模式，只要发现大于0，就可以提前退出了
            if not return_cost and current_obs_max_risk > 0:
                return True

        # 如果没有不可跨越的障碍物，给一个默认的安全值
        if global_max_violation == -np.inf:
            global_max_violation = -10.0

        # --- 5. 返回结果 ---
        if return_cost:
            # 返回风险值 (连续的浮点数)
            return global_max_violation
        else:
            # 返回布尔值
            return global_max_violation > 0
    def judge_done(self) -> bool:
        """
        判断 Episode 是否结束
        结束条件：
        1. 偏离参考线太远 (横向误差过大)
        2. 车头朝向严重偏离 (航向误差过大)
        3. 发生碰撞 (物理碰撞)
        4. 动态约束破坏 (可选，如速度过低或过高)
        """
        x, y, phi = self.state[:3]
        ref_x, ref_y, ref_phi = self.ref_points[0, :3]

        # 1. 横向误差限制 (6米约等于两个车道宽，可以维持)
        lat_error_done = np.abs(y - ref_y) > self.max_road_width
        # 2. 航向误差限制 (建议收紧到 90度 或 60度)
        # 超过 90度 (np.pi / 2) 意味着车已经横过来了，基本无法恢复正常行驶
        heading_error_done = np.abs(angle_normalize(phi - ref_phi)) > (np.pi / 2)
        # 3. 碰撞检测 (复用更准确的逻辑)
        collision_done = self._check_ego_collision()
        # 4. (可选) 纵向落后限制
        # 如果 x 轴严重落后于参考点(说明倒车或者停滞不前)，也可以 done
        dist_longi_done = (x - ref_x) < -10.0
        done = lat_error_done | heading_error_done | collision_done #| dist_longi_done
        # if collision_done:
        #     print(self._check_ego_collision(return_cost=True))
        return bool(done)

    def _tracking_controller(self, traj_ego, target_v):
        """
        鲁棒控制器 (Pure Pursuit + P-Speed)
        输入: traj_ego (N, 2) - 自车坐标系下的规划轨迹点 [x, y]
        输出: [steer, acc]
        """
        # 1. 动态预瞄距离 (Lookahead Distance)
        # 速度越快看越远，最小 2.0m，最大 15.0m
        k_v = 0.8  # 增益系数
        lookahead_dist = np.clip(self.state[3] * k_v, 2.0, 15.0)

        # 2. 搜索预瞄点 (第一个超出预瞄距离的点)
        target_point = traj_ego[-1]  # 默认兜底为最后一个点
        target_idx = len(traj_ego) - 1

        for i, pt in enumerate(traj_ego):
            dist = np.linalg.norm(pt)
            if dist > lookahead_dist:
                target_point = pt
                target_idx = i
                break

        # 3. 横向控制: Pure Pursuit
        # target_point 是 (x, y)，在自车系下，x为纵向，y为横向
        # alpha 是目标点相对于车身纵轴的夹角
        alpha = math.atan2(target_point[1], target_point[0])

        # 纯追踪公式: delta = atan(2 * L * sin(alpha) / Ld)
        steer = math.atan2(2.0 * self.wheel_distance * math.sin(alpha), lookahead_dist)

        # 4. 纵向控制: 带有前馈的速度跟踪
        # 获取规划轨迹对应点的期望速度 (这里近似取全局参考线的速度)
        # 注意：这里假设 ref_points 依然是全局参考，且长度够长
        # ref_idx = min(target_idx, len(self.ref_points) - 1)

        # P 控制器
        k_p_acc = 5.0
        acc = k_p_acc * (target_v - self.state[3])
        steer = np.clip(steer, -self.max_steer, self.max_steer)
        acc = np.clip(acc, -self.max_accel, self.max_accel)
        print("target_speed",target_v, "current_speed", self.state[3], "acc",acc)
        return np.array([steer, acc], dtype=np.float32)
    @property
    def info(self) -> dict:
        return {
            "state": self.state.copy(),
            "ref_points": self.ref_points.copy(),
            "path_num": self.path_num,
            "u_num": self.u_num,
            "slope_num": self.slope_num,
            "ref": self.ref_points[0].copy(),
            "ref_time": self.t,
        }

    @property
    def additional_info(self) -> Dict[str, Dict]:
        return self.info_dict

    def _ego_to_global(self, points, vehicle_state):
        """将局部坐标点转换为全局坐标"""
        ego_x, ego_y, ego_phi = vehicle_state[0], vehicle_state[1], vehicle_state[2]
        c, s = np.cos(ego_phi), np.sin(ego_phi)
        x_global = points[:, 0] * c - points[:, 1] * s + ego_x
        y_global = points[:, 0] * s + points[:, 1] * c + ego_y
        return np.stack([x_global, y_global], axis=1)

    def _calculate_lon_safety_profile(self, path_points, obstacles, max_brake=4.0, safety_margin=2.0):
        """
        计算纵向障碍物安全速度限制
        path_points: (N, 2) 规划出的几何路径
        obstacles: 障碍物列表
        """
        n_points = len(path_points)
        # 初始化为无穷大（无限制）
        v_limit_obs = np.full(n_points, 15.0, dtype=np.float32)

        # 计算路径累计距离
        diffs = np.diff(path_points, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        s_cum = np.concatenate(([0], np.cumsum(dists)))  # (N,)

        # 寻找路径上最近的碰撞点
        min_coll_s = float('inf')

        # 简单的碰撞检测：检查路径点是否在障碍物范围内
        # 为了效率，这里简化为点到圆心的距离检查，你也可以用更精确的矩形检测
        for obs in obstacles:
            # 只考虑前方的障碍物
            if obs.x < self.state[0] - 2.0:
                continue

            # 计算路径所有点到该障碍物的距离
            # obs_pos = np.array([obs.x, obs.y])
            # p_dists = np.linalg.norm(path_points - obs_pos, axis=1)

            # 更精确的碰撞判定：考虑车宽和障碍物尺寸
            # 简单版：距离小于阈值视为碰撞
            # 阈值 = (车宽 + 障碍物宽)/2 + 缓冲
            threshold = (self.veh_width + obs.w) / 2.0 + 0.5

            # 向量化计算距离
            dx = path_points[:, 0] - obs.x
            dy = path_points[:, 1] - obs.y
            p_dists = np.sqrt(dx ** 2 + dy ** 2)

            collision_indices = np.where(p_dists < threshold)[0]

            if len(collision_indices) > 0:
                # 找到该障碍物在路径上的第一个碰撞点索引
                first_idx = collision_indices[0]
                coll_s = s_cum[first_idx]

                # 更新最近碰撞距离
                if coll_s < min_coll_s:
                    min_coll_s = coll_s

        # 如果发现路径上有障碍物
        if min_coll_s != float('inf'):
            # 计算刹车限制曲线
            # v_max^2 = 2 * a_brake * distance_to_stop
            # distance_to_stop = collision_s - current_s - safety_margin

            dist_to_obj = min_coll_s - s_cum - safety_margin
            dist_to_obj = np.maximum(dist_to_obj, 0.0)  # 防止负数

            # v = sqrt(2 * a * s)
            v_brake_profile = np.sqrt(2.0 * max_brake * dist_to_obj)

            # 取最小值
            v_limit_obs = np.minimum(v_limit_obs, v_brake_profile)

        return v_limit_obs

    # --- 新增辅助函数：计算误差并记录 ---
    def _log_step_data(self):
        # 1. 获取当前自车状态
        x, y, phi, u = self.state[:4]

        # 2. 默认值
        target_x, target_y, target_phi, target_u = self.ref_points[0, :4]
        lat_error = 0.0
        phi_error = 0.0

        # 3. 计算相对于“规划轨迹”的几何误差
        if hasattr(self, 'current_planning_traj') and \
                self.current_planning_traj is not None and \
                len(self.current_planning_traj) > 1:

            traj = self.current_planning_traj
            traj_points = traj[:, :2]  # [N, 2]

            # 计算当前车位置到轨迹上所有点的距离
            dists = np.linalg.norm(traj_points - np.array([x, y]), axis=1)

            # 找到最近点的索引
            min_idx = np.argmin(dists)

            # 取出最近点的状态作为“即时目标”
            # 注意：如果 min_idx 是最后一个点，方向可能要取前一个点的切向
            closest_pt = traj[min_idx]
            target_x, target_y, target_phi, target_u = closest_pt[:4]

            # --- 计算横向误差 (CTE) ---
            # 向量：从最近点指向当前车
            dx = x - target_x
            dy = y - target_y

            # 将误差投影到路径法线方向
            # sin(target_phi) 和 cos(target_phi) 定义了切向
            # 法向是 (-sin, cos)
            # Cross Track Error: 距离向量在法向上的投影
            lat_error = -dx * np.sin(target_phi) + dy * np.cos(target_phi)

            # --- 计算航向误差 ---
            phi_error = angle_normalize(phi - target_phi)

            # --- 计算速度误差 ---
            # 注意：速度误差通常还是跟时间绑定的，但跟最近点比也行
            u_error = u - target_u

        else:
            # 如果没有规划轨迹，退化为跟全局参考点比
            # (这种情况一般只在刚reset的第一帧出现)
            ref_x, ref_y, ref_phi, ref_u = self.ref_points[0, :4]
            dx = x - ref_x
            dy = y - ref_y
            lat_error = -dx * np.sin(ref_phi) + dy * np.cos(ref_phi)
            phi_error = angle_normalize(phi - ref_phi)
            u_error = u - ref_u

        # --- 4. 存入 Log (这部分保持不变) ---
        self.log_data['actual_x'].append(x)
        self.log_data['actual_y'].append(y)
        self.log_data['actual_phi'].append(phi)
        self.log_data['actual_u'].append(u)

        # 记录用于对比的那个“最近点”
        self.log_data['ref_x'].append(target_x)
        self.log_data['ref_y'].append(target_y)
        self.log_data['ref_phi'].append(target_phi)
        self.log_data['ref_u'].append(target_u)

        self.log_data['err_lat'].append(lat_error)
        self.log_data['err_phi'].append(phi_error)
        self.log_data['err_u'].append(u_error)

    def render_tracking_controller(self, mode='human'):
        import matplotlib.pyplot as plt

        # 1. 强制设置 Agg 后端 (解决服务器无头模式报错)
        if mode == 'rgb_array':
            plt.switch_backend('agg')

        # 2. 画布初始化
        if not hasattr(self, 'fig') or self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(12, 4.5), dpi=600)

        # 3. 清理并绘图
        self.ax.cla()

        # 调用内部绘图逻辑，传入 self.ax
        self._render(self.ax)

        # 4. 根据模式返回
        if mode == 'rgb_array':
            self.fig.canvas.draw()
            # 获取 Buffer 并 Reshape
            try:
                data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
                width, height = self.fig.canvas.get_width_height()
                data = data.reshape((height, width, 3))
                return data
            except Exception as e:
                print(f"Render Error: {e}")
                return None
        elif mode == 'human':
            try:
                plt.pause(0.01)
            except:
                pass
            return self.fig

    def render(self, mode='human'):
        if mode == 'rgb_array':
            # 确保返回正确的 numpy array
            fig = self.render_tracking_controller(mode='rgb_array')
            if fig is not None:
                return fig
            else:
                # 备用方案：创建一个简单的图像
                return np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            return self.render_tracking_controller(mode=mode)

    def _render(self, ax, veh_length=4.8, veh_width=2.0):
        """
        核心绘图逻辑：修复车辆显示、中文字体问题及图例挤压问题
        """
        import matplotlib.patches as pc
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        from matplotlib.patches import Polygon
        from matplotlib.lines import Line2D  # 引入Line2D用于自定义图例

        # --- 0. 设置中文字体 (保持不变) ---
        try:
            import matplotlib
            chinese_font_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                'gops', 'utils', 'SIMSUN.ttf')

            if not os.path.exists(chinese_font_path):
                current_dir = os.path.dirname(os.path.abspath(__file__))
                for _ in range(4):
                    current_dir = os.path.dirname(current_dir)
                    test_path = os.path.join(current_dir, 'gops', 'utils', 'SIMSUN.ttf')
                    if os.path.exists(test_path):
                        chinese_font_path = test_path
                        break

            if os.path.exists(chinese_font_path):
                matplotlib.font_manager.fontManager.addfont(chinese_font_path)
                font_name = matplotlib.font_manager.FontProperties(fname=chinese_font_path).get_name()
                matplotlib.rcParams['font.sans-serif'] = [font_name]
                matplotlib.rcParams['axes.unicode_minus'] = False
            else:
                matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
        except Exception as e:
            print(f"加载中文字体时出错: {e}")

        for spine in ax.spines.values():
            spine.set_linewidth(3)

        # --- 辅助函数：旋转矩形 ---
        def get_rotated_rect(x, y, phi, l, w):
            cos_phi, sin_phi = np.cos(phi), np.sin(phi)
            half_l, half_w = l / 2, w / 2
            corners = np.array([
                [-half_l, -half_w], [half_l, -half_w],
                [half_l, half_w], [-half_l, half_w]
            ])
            rot_mat = np.array([[cos_phi, -sin_phi], [sin_phi, cos_phi]])
            rotated = corners.dot(rot_mat.T)
            rotated[:, 0] += x
            rotated[:, 1] += y
            return rotated

        # === 【关键修改 1】初始化图例句柄列表 ===
        # 我们将手动控制哪些元素进入图例，防止重复和挤压
        legend_handles = []
        # 用于记录已添加的图例标签，防止重复
        added_labels = set()

        # --- 1. 绘制自车 (Ego) ---
        ego_x, ego_y, phi = self.state[:3]
        ego_corners = get_rotated_rect(ego_x, ego_y, phi, veh_length, veh_width)

        # 绘制自车
        ego_polygon = Polygon(
            ego_corners, closed=True, facecolor='magenta', edgecolor='darkmagenta',
            alpha=0.9, linewidth=2, zorder=20
        )
        ax.add_patch(ego_polygon)

        # 手动添加自车到图例
        if '自车' not in added_labels:
            legend_handles.append(ego_polygon)  # 注意：这里直接用patch作为handle，label在legend函数里赋或者patch里赋
            ego_polygon.set_label('自车')
            added_labels.add('自车')

        # 自车箭头
        ax.arrow(ego_x, ego_y, veh_length * 0.8 * np.cos(phi), veh_length * 0.8 * np.sin(phi),
                 head_width=0.8, head_length=1.2, fc='red', ec='red', alpha=0.8, zorder=21)

        # --- 2. 绘制参考轨迹 (Global) ---
        if hasattr(self, 'ref_points') and self.ref_points is not None:
            # 2.1 绘制全局轨迹
            line_ref, = ax.plot(self.ref_points[:, 0], self.ref_points[:, 1], 'b--', lw=5, zorder=2)
            # 手动添加图例
            if '全局轨迹' not in added_labels:
                line_ref.set_label('全局轨迹')
                legend_handles.append(line_ref)
                added_labels.add('全局轨迹')
            # 2.2 【修改】绘制崎岖的土路边界 (Rugged Road Boundaries)
            rx = self.ref_points[:, 0]
            ry = self.ref_points[:, 1]
            rphi = self.ref_points[:, 2]
            # === [Step A] 向后延长参考线 ===
            # 计算起始点的切向，向后倒推 20m，确保线能覆盖到车身后面
            dx0 = rx[1] - rx[0]
            dy0 = ry[1] - ry[0]
            head0 = np.arctan2(dy0, dx0)

            back_len = 10.0  # 向后延长的长度
            p_back_x = rx[0] - back_len * np.cos(head0)
            p_back_y = ry[0] - back_len * np.sin(head0)

            # 拼接新数据：[延长点, 原轨迹点...]
            rx_ext = np.concatenate(([p_back_x], rx))
            ry_ext = np.concatenate(([p_back_y], ry))
            # === [Step C] 计算并绘制左右边界 ===
            # 重新计算延长后的航向角
            # 使用梯度计算每个点的切向
            dx_ext = np.gradient(rx_ext)
            dy_ext = np.gradient(ry_ext)
            phi_ext = np.arctan2(dy_ext, dx_ext)

            # 计算法向量
            nx = -np.sin(phi_ext)
            ny = np.cos(phi_ext)

            offset = 7.0

            # # 左边界 (Left Boundary)
            # lx = rx_ext + offset * nx
            # ly = ry_ext + offset * ny

            # 右边界 (Right Boundary)
            rx_b = rx_ext - offset * nx
            ry_b = ry_ext - offset * ny
            num_pts = len(rx_b)

            # --- 生成崎岖感 ---
            # 1. 生成随机噪声 (范围例如 +/- 0.8m)
            noise_l = np.random.uniform(-0.8, 0.8, num_pts)
            noise_r = np.random.uniform(-0.8, 0.8, num_pts)

            # 2. 简单平滑处理，让边界呈波浪状而不是锯齿状
            # 使用移动平均窗口
            window_size = 5  # 窗口越大越平滑
            window = np.ones(window_size) / window_size
            # 为了简单演示，这里用numpy的convolve，边缘可能会有一点点不完美，但视觉足够
            noise_l_smooth = np.convolve(noise_l, window, mode='same')
            noise_r_smooth = np.convolve(noise_r, window, mode='same')

            # 3. 应用带有噪声的偏移
            lx = rx_ext + (offset + noise_l_smooth) * nx
            ly = ry_ext + (offset + noise_l_smooth) * ny
            r_bound_x = rx_b - (offset + noise_r_smooth) * nx
            r_bound_y = ry_b - (offset + noise_r_smooth) * ny

            # 4. 绘制线条 (使用深棕色实线，增加一点透明度)
            # 越野土色：'#8B4513' (SaddleBrown) 或 '#A0522D' (Sienna)
            earth_color = '#8B4513'
            ax.plot(lx, ly, color=earth_color, linestyle='--', linewidth=3, alpha=0.8, zorder=1)
            ax.plot(r_bound_x, r_bound_y, color=earth_color, linestyle='--', linewidth=3, alpha=0.8, zorder=1)
            # 添加边界图例
            if '道路边界' not in added_labels:
                # 图例也用同样的颜色和样式
                proxy_bound = Line2D([], [], color=earth_color, linestyle='-', linewidth=3, alpha=0.6,
                                     label='道路边界')
                legend_handles.append(proxy_bound)
                added_labels.add('道路边界')

            # --- [新增] 绘制实际行驶的历史轨迹 (Controller 实际走出来的路) ---
            # 使用 Log 中的数据，比 self.history_traj 更全
            if len(self.log_data['actual_x']) > 1:
                ax.plot(self.log_data['actual_x'], self.log_data['actual_y'],
                        color='black', linestyle='-', linewidth=2.5, alpha=0.9, label='实际轨迹 (Actual)')

            # --- [修改] 绘制当前规划轨迹 (Agent 每一帧规划出的路) ---
            traj_global = getattr(self, 'current_planning_traj', None)
            if traj_global is not None and len(traj_global) > 0:
                pts_x = traj_global[:, 0]
                pts_y = traj_global[:, 1]

                # 绘制连线 (粉色)
                ax.plot(pts_x, pts_y, color='deeppink', linewidth=3.0, linestyle='--', alpha=0.8,
                        label='当前规划 (Planning)')

                # 绘制速度颜色点 (可选)
                if traj_global.shape[1] >= 4:
                    pts_u = traj_global[:, 3]
                    sc = ax.scatter(pts_x, pts_y, c=pts_u, cmap='cool', vmin=0, vmax=15.0,
                                    s=20, zorder=15)
                    # Colorbar 逻辑保持你原有的即可

            # # --- 3. 绘制规划轨迹 (Planning) ---
            # traj_global = getattr(self, 'current_planning_traj', None)
            # if traj_global is not None and len(traj_global) > 0:
            #     pts_x = traj_global[:, 0]
            #     pts_y = traj_global[:, 1]
            #
            #     if traj_global.shape[1] >= 4:
            #         pts_u = traj_global[:, 3]
            #         # A. 绘制底线
            #         ax.plot(pts_x, pts_y, color='pink', linewidth=4.0, alpha=0.5, zorder=14)
            #         # B. 绘制散点
            #         sc = ax.scatter(pts_x, pts_y, c=pts_u, cmap='jet', vmin=0, vmax=12.0,
            #                         s=30, edgecolors='none', zorder=15)
            #
                    # 手动创建一个 "点线" 样式的图例 Handle
                    if '规划轨迹' not in added_labels:
                        proxy_line = Line2D([], [], color='pink', linewidth=4, linestyle='-',
                                            marker='o', markersize=10, markerfacecolor='cyan', markeredgecolor='none')
                        proxy_line.set_label('规划轨迹')
                        legend_handles.append(proxy_line)
                        added_labels.add('规划轨迹')

                    # C. Colorbar (保持不变)
                    try:
                        from mpl_toolkits.axes_grid1 import make_axes_locatable
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="2%", pad=0.1)
                        cbar = plt.colorbar(sc, cax=cax, orientation="vertical")
                        cbar.set_label(r"速度 $(\mathrm{m/s})$", fontsize=25, labelpad=10)
                        cbar.ax.tick_params(labelsize=21)
                        cbar.set_ticks([0, 3, 6, 9, 12])
                    except:
                        pass
                else:
                    # 兼容旧代码
                    line_plan, = ax.plot(pts_x, pts_y, 'pink', marker='.', markersize=15,
                                         markeredgecolor='deeppink', markeredgewidth=0.5, linewidth=5, alpha=0.5,
                                         zorder=15)
                    if '规划轨迹' not in added_labels:
                        line_plan.set_label('规划轨迹')
                        legend_handles.append(line_plan)
                        added_labels.add('规划轨迹')

        # --- 4. 绘制历史轨迹 ---
        # if hasattr(self, 'history_traj') and self.history_traj:
        #     history_array = np.array(self.history_traj)
        #     if len(history_array) > 1:
        #         line_hist, = ax.plot(history_array[:, 0], history_array[:, 1], 'gray', linestyle='-', linewidth=4.0,
        #                              alpha=0.8, zorder=5)
        #         if '历史轨迹' not in added_labels:
        #             line_hist.set_label('历史轨迹')
        #             legend_handles.append(line_hist)
        #             added_labels.add('历史轨迹')

        # --- 5. 绘制障碍物 (核心修复) ---
        if hasattr(self, 'obstacles'):
            # 排序：静态在下，动态在上
            sorted_obs = sorted(self.obstacles, key=lambda o: 0 if o.type == 'static' else 1)

            for obs in sorted_obs:
                can_cross = getattr(obs, 'can_cross', False)
                is_dynamic = getattr(obs, 'type', 'static') == 'dynamic'

                # 确定颜色和标签
                label_text = None
                color = 'gray'
                z_order = 10

                if is_dynamic:
                    color = 'red'
                    z_order = 12
                    label_text = '动态障碍物'
                elif can_cross:
                    color = 'lime'
                    z_order = 10
                    label_text = '可跨静态障碍物'
                else:
                    color = 'orange'
                    z_order = 10
                    label_text = '不可跨静态障碍物'

                # 绘制多边形
                obs_corners = get_rotated_rect(obs.x, obs.y, obs.phi, obs.l, obs.w)
                # 注意：这里我们不再在 Polygon 里传 label 参数了，防止 Matplotlib 自动收集
                obs_poly = Polygon(obs_corners, closed=True, facecolor=color, edgecolor='black',
                                   alpha=0.9, linewidth=2, zorder=z_order)
                ax.add_patch(obs_poly)

                # 【核心逻辑】如果这个类型的标签还没添加过，手动创建一个 Handle 加进去
                if label_text and label_text not in added_labels:
                    # 创建一个看不见的 Proxy Patch 用于显示图例，颜色要和障碍物一致
                    proxy_patch = Polygon([[0, 0]], closed=True, facecolor=color, edgecolor='black', alpha=0.9,
                                          linewidth=2)
                    proxy_patch.set_label(label_text)
                    legend_handles.append(proxy_patch)
                    added_labels.add(label_text)

                # 动态障碍物箭头
                if is_dynamic and abs(obs.u) > 0.1:
                    ax.arrow(obs.x, obs.y, obs.u * 0.5 * np.cos(obs.phi), obs.u * 0.5 * np.sin(obs.phi),
                             head_width=0.5, head_length=0.8, fc='darkred', ec='darkred', alpha=0.6, zorder=z_order + 1)

        # --- 6. 绘制引导线 ---
        if hasattr(self, 'guide_trajectories') and self.guide_trajectories:
            for i, guide_traj in enumerate(self.guide_trajectories):
                if len(guide_traj) > 1:
                    line_guide, = ax.plot(guide_traj[:, 0], guide_traj[:, 1], color='cyan', ls='--', lw=4.0, alpha=0.8,
                                          zorder=3)
                    # 只添加一次图例
                    if '引导线' not in added_labels:
                        line_guide.set_label('引导线')
                        legend_handles.append(line_guide)
                        added_labels.add('引导线')

        # --- 8. 设置视野与动态刻度 ---
        # A. Y轴
        y_center = ego_y
        y_min, y_max = y_center - 10, y_center + 10
        y_ticks = [y_center - 10, y_center, y_center + 10]
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(y_ticks)

        # B. X轴
        x_start = ego_x - 21
        x_end = ego_x + 59
        x_ticks = np.linspace(x_start, x_end, 5)
        ax.set_xlim(x_start, x_end)
        ax.set_xticks(x_ticks)

        # C. 格式化
        from matplotlib.ticker import FormatStrFormatter
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

        # D. 样式
        ax.set_aspect('equal')
        ax.tick_params(labelsize=28)
        ax.tick_params(axis='x', direction='in')
        ax.tick_params(axis='y', direction='in')

        ax.set_xlabel(r'纵向位置 $p_x\, (\mathrm{m})$', fontsize=30)
        ax.set_ylabel(r'横向位置 $p_y\, (\mathrm{m})$', fontsize=30)

        # --- 10. 创建图例 (使用我们手动收集的 handles) ---
        # 【关键修改】handles=legend_handles
        # 这样无论外面画了多少个障碍物，图例里只会有 unique 的几个
        if legend_handles:
            ax.legend(
                handles=legend_handles,  # <--- 强制指定
                loc='upper center',
                bbox_to_anchor=(0.54, 1.50),
                ncol=4,
                fontsize=21,
                frameon=False,
                fancybox=False
            )

        # --- 11. 布局调整 ---
        # 使用 rect 参数给顶部的 legend 留出空间，防止 tight_layout 挤压主图
        # rect=[left, bottom, right, top]
        ax.figure.tight_layout()


def env_creator(**kwargs):
    return SimuVeh3dofcontiBimodalDiffusion(**kwargs)