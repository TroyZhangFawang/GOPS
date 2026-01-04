import gym
import numpy as np
import math
from gym import spaces
from gops.env.env_ocp.pyth_base_env import PythBaseEnv
from gops.env.env_ocp.resources.ref_traj_data import MultiRefTrajData
from gops.utils.math_utils import angle_normalize
from gops.env.env_ocp.pyth_veh3dofcontiplanning import SimuVeh3dofconti, angle_normalize, ego_vehicle_coordinate_transform
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
from gops.utils.planner_benchmark.control import SimpleController

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

    def step(self):
        # 简单的运动学模型更新
        self.x = self.x + self.u * np.cos(self.phi) * self.dt
        self.y = self.y + self.u * np.sin(self.phi) * self.dt
        self.phi = self.phi + self.u * np.tan(self.delta) / self.l * self.dt
        self.phi = angle_normalize(self.phi)
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

class SimuVeh3dofcontiBimodalDiffusion(SimuVeh3dofconti):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }
    def __init__(self,
                 pre_horizon: int = 20,
                 path_para: Optional[Dict[str, Dict]] = None,
                 u_para: Optional[Dict[str, Dict]] = None,
                 max_steer: float = np.pi / 6,
                 max_accel: float = 3.0,
                 dynamic_obstacle_num: int = 1,
                 static_obstacle_num: int = 2,
                 d_pre: float = 20.0,  # 离障碍物多少远开始规划
                 lateral_sample: float = 3.5,  # 横向采样距离
                 forward_sample: float = 20.0,  # 纵向采样距离
                 **kwargs):
        super().__init__(pre_horizon, path_para, u_para, **kwargs)
        self.max_episode_steps = 150
        self.controller = SimpleController()
        self.is_adversary = kwargs.get("is_adversary", False)
        self.is_constraint = kwargs.get("is_constraint", False)
        # 控制模式开关, "planning": "control": 输入 [steer, acc] , 直接控制
        self.control_mode = kwargs.get("control_mode", "planning")
        self.max_steer = max_steer
        self.max_accel = max_accel
        self.action_dim = 2
        # --- 1. 预测时域与动作空间 ---
        self.pred_horizon = pre_horizon
        if self.control_mode == "planning":
            self.action_space = spaces.Box(
                low=np.array([-lateral_sample], dtype=np.float32),
                high=np.array([lateral_sample], dtype=np.float32),
                dtype=np.float32
            )
        else:
            # SAC/PPO 输出直接控制量: [steer, acc]
            self.action_space = spaces.Box(
                low=np.array([-self.max_steer, -self.max_accel]),
                high=np.array([self.max_steer, self.max_accel]),
                dtype=np.float32
            )
        # --- 2. 观测空间设计 (融合全局跟踪信息 + 引导信息) ---
        # A. 基础参考路径参数 (用于生成 ref_points)
        self.ref_horizon = pre_horizon
        # B. 障碍物参数
        self.max_obs_num = dynamic_obstacle_num+static_obstacle_num  # 观测中最多包含的障碍物数量
        self.dynamic_obstacle_num = dynamic_obstacle_num
        self.static_obstacle_num = static_obstacle_num
        self.obs_feat_dim = 8  # [x, y, phi, u, l, w, type, dist]
        # C. 维度计算
        self.dim_ego = 6
        self.dim_ref = self.ref_horizon * 4
        self.dim_obstacles = self.max_obs_num * self.obs_feat_dim
        self.perception_range = 60.0
        self.guide_traj_num = 3
        self.dim_prompts = self.guide_traj_num * self.ref_horizon * 2
        self.total_obs_dim = self.dim_ego + self.dim_ref + self.dim_obstacles + self.dim_prompts
        obs_scale_default = [1 / 100, 1 / 100, 1 / 10,
                             1 / 100, 1 / 100, 1 / 10, 1 / 10, 1 / 50, 1 / (max_accel * 100), 1 / 10]
        self.obs_scale = np.array(kwargs.get('obs_scale', obs_scale_default))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.total_obs_dim,),
            dtype=np.float32
        )
        # --- 3. 物理模型初始化 ---
        self.d_pre = d_pre  # 障碍物感知前瞻距离
        self.lateral_sample = lateral_sample  # 横向偏移量
        self.forward_sample = forward_sample
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
        self.info_dict = {
            "state": {"shape": (self.state_dim,), "dtype": np.float32},
            "ref_points": {"shape": (self.pre_horizon + 1, 4), "dtype": np.float32},
            "path_num": {"shape": (), "dtype": np.uint8},
            "u_num": {"shape": (), "dtype": np.uint8},
            "ref_time": {"shape": (), "dtype": np.float32},
            "ref": {"shape": (4,), "dtype": np.float32},
        }

    def reset(self,
              init_state: list = None,
              ref_time: float = None,
              ref_num: int = None,
              **kwargs) -> Tuple[np.ndarray, dict]:
        super().reset(init_state, ref_time, ref_num, **kwargs)
        # 4. 生成越野特有元素
        self.obstacles = self._generate_offroad_obstacles()
        self.guide_trajectories = self._generate_guidance_prompts()

        # 初始化渲染变量
        self.current_diffusion_traj = None
        self.current_planning_traj = None
        self.history_traj = [self.state[:2]] # 记录起点
        self.step_self = 0
        self.info["TimeLimit.truncated"] = False
        return self._get_obs(), self.info

    def _ego_to_global(self, points, vehicle_state):
        """将局部坐标点转换为全局坐标"""
        ego_x, ego_y, ego_phi = vehicle_state[0], vehicle_state[1], vehicle_state[2]
        c, s = np.cos(ego_phi), np.sin(ego_phi)
        x_global = points[:, 0] * c - points[:, 1] * s + ego_x
        y_global = points[:, 0] * s + points[:, 1] * c + ego_y
        return np.stack([x_global, y_global], axis=1)

    def step(self, action):
        if np.any(np.isnan(action)):
            print("【致命错误】Agent输出了 NaN Action!")
            action = np.zeros_like(action)
        if self.control_mode == "planning":
            # 1. 坐标系对齐：全部在 Ego Frame 下工作
            ego_x, ego_y, ego_phi = self.state[0], self.state[1], self.state[2]

            # 1. 寻找最近的威胁障碍物
            nearest_obs = None
            min_dist = float('inf')
            for obs in self.obstacles:
                # 计算到障碍物的距离
                dist = np.linalg.norm([obs.x - ego_x, obs.y - ego_y])
                # 判定条件：
                # 1. 距离在感知范围内 (d_pre) 2. 障碍物在车前方 (简单判断 x)
                if dist < self.d_pre and obs.x > ego_x:
                    if dist < min_dist:
                        min_dist = dist
                        nearest_obs = obs
            # === 情况 A: 存在威胁障碍物 ===
            if nearest_obs is not None:
                y_target = nearest_obs.y + action[0] + nearest_obs.w / 2 + self.veh_width / 2
                p0 = np.array([ego_x, ego_y])
                # 控制点 p1: 在障碍物横向截面处
                p1 = np.array([nearest_obs.x, y_target])
                # 终点 p2: 在障碍物后方，回归目标 y
                p2 = np.array([nearest_obs.x + self.forward_sample + nearest_obs.l / 2, y_target])
                # Global 坐标系轨迹
                full_curve = BezierGenerator.generate(p0, p1, p2, num_points=self.pred_horizon + 1)
                self.current_planning_traj = full_curve
            # === 情况 B: 无威胁，自由行驶 ===
            else:
                # 获取全局参考路径在 Ego 系下的投影 (作为基准)
                ref_ego_x, ref_ego_y, _ = ego_vehicle_coordinate_transform(
                    ego_x, ego_y, ego_phi,
                    self.ref_points[:, 0], self.ref_points[:, 1], self.ref_points[:, 2]
                )
                # ==========================================
                # 2. 确定基准点 (Residual Learning)
                mid_idx = min(len(ref_ego_x) - 1, int(self.pred_horizon / 2))
                end_idx = min(len(ref_ego_x) - 1, self.pred_horizon - 1)
                p1_base = np.array([ref_ego_x[mid_idx], ref_ego_y[mid_idx]])
                p2_base = np.array([ref_ego_x[end_idx], ref_ego_y[end_idx]])
                # ==========================================
                # 3. 生成局部轨迹 (Ego Frame)
                # ==========================================
                p0 = np.array([0.0, 0.0])  # 【关键】P0 必须是局部原点
                p1 = p1_base + np.array([0.0, action[0]])  # 动作是相对于基准的偏移
                p2 = p2_base + np.array([0.0, action[0]])
                full_curve = BezierGenerator.generate(p0, p1, p2, num_points=self.pred_horizon + 1)
                # 立即转为 Global 轨迹 (仅用于 Reward 计算和 Render 绘图)
                self.current_planning_traj = self._ego_to_global(
                    full_curve[1:], self.state
                )
            # ==========================================
            # 4. 追踪控制器 (使用局部坐标)
            # real_action = self._tracking_controller(full_curve[1:])
            real_action = self.controller.get_control(self.current_planning_traj, self.ref_points[0, 3], self.state[:3], self.state[3])
            reward = self._compute_reward(real_action)
            _, _, _, _ = super().step(real_action)
        else:
            # --- 分支 B: SAC/PPO/DSACT Controller 模式 ---
            steer = np.clip(action[0], -self.max_steer, self.max_steer)
            acc = np.clip(action[1], -self.max_accel, self.max_accel)
            real_action = np.array([steer, acc], dtype=np.float32)
            self.current_diffusion_traj = None  # 无轨迹可视化
            _, _, _, _ = super().step(real_action)
            reward = self._compute_reward(real_action)
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
        # 5. 更新引导
        self.guide_trajectories = self._generate_guidance_prompts()
        obs = self._get_obs()
        # 6. 终止条件
        done = self.judge_done()
        if done:
            reward -= 10
        # 7. 时间限制 (Truncated: 步数超限)
        is_truncated = self.step_self >= self.max_episode_steps
        if is_truncated:
            self.info["TimeLimit.truncated"] = True
            reward += 10.0  # 存活奖励
            done = True
        else:
            self.info["TimeLimit.truncated"] = False
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
        return obs, reward, done, self.info

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
        ref_obs = np.stack((ref_x_tf*self.obs_scale[0], ref_y_tf*self.obs_scale[1], ref_phi_tf*self.obs_scale[2], ref_u_tf*self.obs_scale[3]), 1)[1:].flatten()

        # 4. Obstacle Obs
        obs_feats = []
        dists = []

        # 设定感知半径 (例如 60米)
        # 只有在视野内的障碍物才值得关注

        for obs in self.obstacles:
            d = np.sqrt((obs.x - self.state[0]) ** 2 + (obs.y - self.state[1]) ** 2)

            # 【关键修改】增加距离过滤
            # 只有距离小于感知半径，且不是那种已经被甩在身后很远(例如身后15米外)的障碍物才加入
            # 这里简单用欧氏距离过滤，你也可以加 (obs.x - ego_x) > -10 这样的逻辑
            if d < self.perception_range:
                dists.append((d, obs))

        # 按距离排序，优先关注最近的
        dists.sort(key=lambda x: x[0])

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
                feat = [ox_tf[0]/self.perception_range, oy_tf[0]/self.perception_range, ophi_tf[0]*self.obs_scale[2],
                        u_rel*self.obs_scale[3], obs.l*self.obs_scale[2], obs.w*self.obs_scale[2], 1.0 if obs.type == "dynamic" else 0.0, ttc*self.obs_scale[2]]
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
        ], dtype=np.float32)
        self.ref_points[-1] = new_ref_point

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
            dynamic_phi = self.ref_traj.compute_phi(self.t + delta_t, self.path_num, self.u_num)

            delta_lon = 1.0 * self.np_random.uniform(-1, 1)
            delta_lat = 1.0 * self.np_random.uniform(-2.0, 2.0)  # 限制在路宽范围内

            dynamic_x = self.ref_traj.compute_x(self.t + delta_t, self.path_num, self.u_num) + delta_lon
            dynamic_y = self.ref_traj.compute_y(self.t + delta_t, self.path_num, self.u_num) + delta_lat
            dynamic_u = self.np_random.uniform(2, 8)  # 速度随机

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
            delta_t = self.np_random.uniform(5, 15)  # 放在稍远一点
            static_obs_phi = self.ref_traj.compute_phi(self.t + delta_t, self.path_num, self.u_num)

            delta_lon = 1.0 * self.np_random.uniform(-2, 2)
            delta_lat = 1.0 * self.np_random.uniform(-3.0, 3.0)

            static_obs_x = self.ref_traj.compute_x(self.t + delta_t, self.path_num, self.u_num) + delta_lon
            static_obs_y = self.ref_traj.compute_y(self.t + delta_t, self.path_num, self.u_num) + delta_lat

            static_length = self.np_random.uniform(1.0, 3.0)
            static_width = self.np_random.uniform(1.0, 3.0)
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
            x=ego_x + 30, y=0.0, l=2.0, w=2.0,
            type="static", can_cross=False, id=1
        ))

        # 2. 静态障碍 - 可跨 (小土堆/水坑)
        obs_list.append(Obstacle(
            x=ego_x + 60, y=0.0, l=3.0, w=3.0,
            type="static", can_cross=True, id=2
        ))

        # 3. 动态障碍 (同向慢车 - 视为不可跨)
        dyn_data = DynamicObstacleData(
            x=ego_x + 10, y=0, phi=0, u=8.0
        )
        obs_list.append(Obstacle(
            x=dyn_data.x, y=dyn_data.y, l=4.0, w=2.0,
            u=dyn_data.u, type="dynamic", can_cross=False, id=3,
            dynamic_data=dyn_data
        ))
        return obs_list

    def _generate_guidance_prompts(self):
        """
                智能引导生成逻辑 (区分可跨/不可跨)：
                1. 寻找最近的威胁障碍物。
                2. 若无威胁 -> 生成 3 条全局参考线跟随 (左/中/右车道)。
                3. 若有威胁 -> 判断障碍物属性：
                   - 可跨越 (can_cross=True) -> 生成 3 条引导: [左绕, 跨越, 右绕]
                   - 不可跨 (can_cross=False) -> 生成 2 条引导: [左绕, 右绕]
                """
        ego_x, ego_y = self.state[0], self.state[1]

        # 1. 寻找最近的威胁障碍物
        nearest_obs = None
        min_dist = float('inf')
        for obs in self.obstacles:
            # 计算到障碍物的距离
            dist = np.linalg.norm([obs.x - ego_x, obs.y - ego_y])
            # 判定条件：
            # 1. 距离在感知范围内 (d_pre) 2. 障碍物在车前方 (简单判断 x)
            if dist < self.d_pre and obs.x > ego_x:
                if dist < min_dist:
                    min_dist = dist
                    nearest_obs = obs

        prompts = []
        # === 情况 A: 存在威胁障碍物 ===
        if nearest_obs is not None:
            # 基础偏移量，这里我们基于障碍物的 y 中心生成偏移
            obs_y = nearest_obs.y
            if nearest_obs.can_cross:
                # 【可跨越】：生成 左、中、右 三条
                offsets = [
                    obs_y - self.lateral_sample - nearest_obs.w / 2 - self.veh_width / 2,  # 右绕
                    obs_y,  # 直接跨越
                    obs_y + self.lateral_sample + nearest_obs.w / 2 + self.veh_width / 2  # 左绕
                ]
            else:
                # 【不可跨】：只生成 左、右 两条
                offsets = [
                    obs_y - self.lateral_sample-nearest_obs.w/2-self.veh_width/2,  # 右绕
                    obs_y + self.lateral_sample + nearest_obs.w / 2 + self.veh_width / 2  # 左绕
                ]

            # 使用贝塞尔曲线生成平滑轨迹
            for y_target in offsets:
                # 控制点 p0: 自车当前位置
                p0 = np.array([ego_x, ego_y])
                # 控制点 p1: 在障碍物横向截面处
                p1 = np.array([nearest_obs.x, y_target])
                # 终点 p2: 在障碍物后方，回归目标 y
                p2 = np.array([nearest_obs.x + self.forward_sample+nearest_obs.l/2, y_target])

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

    def _compute_reward(self, action):
        if self.control_mode == "planning":
            if self.current_planning_traj is None:
                return -1.0

            agent_planning_traj = self.current_planning_traj

            # --- 1. 智能引导奖励 (Smart Guidance + Heading Alignment) ---
            min_cum_dist_score = float('inf')
            valid_guide_found = False
            best_guide_traj = None  # [新增] 用于存储匹配到的最佳引导线

            for guide_traj in self.guide_trajectories:
                # 1. 安全性检查 (只跟踪不撞墙的线)
                # margin=1.0 意味着引导线必须离障碍物有1米以上的缓冲
                if self._check_traj_collision(guide_traj, margin=1.0):
                    continue

                valid_guide_found = True

                # 2. 截取长度对齐
                min_len = min(len(agent_planning_traj), len(guide_traj))

                # 3. 计算逐点欧氏距离 (位置误差)
                diff_vec = agent_planning_traj[:min_len, :2] - guide_traj[:min_len, :2]
                point_wise_dists = np.linalg.norm(diff_vec, axis=1)

                # 4. 计算累计位置误差
                cum_dist = np.sum(point_wise_dists)

                # 寻找累计误差最小的那条安全线
                if cum_dist < min_cum_dist_score:
                    min_cum_dist_score = cum_dist
                    best_guide_traj = guide_traj[:min_len]  # [新增] 保存最佳轨迹片段

            # --- 计算 Guidance 和 Heading 奖励 ---
            r_heading = 0.0  # [新增] 航向奖励
            if not valid_guide_found:
                r_guidance = -1.0
                # 如果没路走了，heading 也没意义，给个惩罚
                r_heading = -1.0
            else:
                # A. 位置引导奖励
                r_guidance = -0.05 * min_cum_dist_score

                # B. [新增] 航向一致性奖励 (Heading Alignment)
                if best_guide_traj is not None:
                    # 截取 agent 轨迹以匹配长度
                    agent_match = agent_planning_traj[:len(best_guide_traj)]

                    # 计算切向量 (x_{i+1} - x_i, y_{i+1} - y_i)
                    # axis=0 表示沿时间维做差分
                    agent_diff = np.diff(agent_match, axis=0)
                    guide_diff = np.diff(best_guide_traj, axis=0)

                    # 计算每一段的航向角 phi = arctan2(dy, dx)
                    agent_phi = np.arctan2(agent_diff[:, 1], agent_diff[:, 0])
                    guide_phi = np.arctan2(guide_diff[:, 1], guide_diff[:, 0])

                    # 计算角度误差，并归一化到 [-pi, pi]
                    phi_error = agent_phi - guide_phi
                    phi_error = (phi_error + np.pi) % (2 * np.pi) - np.pi

                    # 累计航向误差惩罚
                    # 权重建议：因为弧度值较小 (0.1~0.5)，且是 Sum，建议系数给大一点点
                    r_heading = -0.05 * np.sum(np.square(phi_error))

            # --- 2. 任务奖励 ---
            ego_u = self.state[3]
            ref_u = self.ref_points[0, 3]
            # 缩小了系数，避免数值过大掩盖几何奖励
            r_velocity = -0.5 * np.square(ego_u - ref_u)

            # --- 3. 自身碰撞检测
            r_collision = 0.0
            t_steps = np.linspace(0, self.pred_horizon * self.dt, len(agent_planning_traj))

            for obs in self.obstacles:
                # 动态预测
                obs_future_x = obs.x + obs.u * np.cos(obs.phi) * t_steps
                obs_future_y = obs.y + obs.u * np.sin(obs.phi) * t_steps
                obs_future_traj = np.stack([obs_future_x, obs_future_y], axis=1)

                dists = np.linalg.norm(agent_planning_traj[:, :2] - obs_future_traj, axis=1)

                # 势场半径
                base_threshold = self.veh_width / 2.0 + obs.w / 2.0 + 1.0
                safety_buffer = 0.0
                if obs.type == "dynamic":
                    safety_buffer = 0.5 * abs(obs.u - self.state[3])

                safe_threshold = base_threshold + safety_buffer

                in_field_mask = dists < safe_threshold
                if np.any(in_field_mask):
                    intrusions = safe_threshold - dists[in_field_mask]
                    # 系数 1.1 配合平方项
                    r_collision -= np.sum(np.square(intrusions))* 2.0
            # [B] 真实物理碰撞惩罚
            if self._check_ego_collision():
                r_collision -= 10.0

            # 生存奖励，保持正向激励
            r_living = 0.1

            return r_velocity + r_guidance + r_heading + r_collision + r_living
        else:
            return super().compute_reward(action)

    def _check_ego_collision(self):
        ego_x, ego_y = self.state[0], self.state[1]
        for obs in self.obstacles:
            dist = np.sqrt((ego_x - obs.x) ** 2 + (ego_y - obs.y) ** 2)
            safe_dist = (self.veh_width + obs.w) / 2.0 + 0.2
            if dist < safe_dist:
                return True
        return False

    def _check_traj_collision(self, traj_points, margin=0.0):
        """
        margin: 额外的安全边界。检查 Prompt 时给 1.0，检查自身碰撞时给 0.0
        """
        if traj_points is None or len(traj_points) == 0:
            return True

        t_steps = np.linspace(0, self.pred_horizon * self.dt, len(traj_points))

        for obs in self.obstacles:
            obs_future_x = obs.x + obs.u * np.cos(obs.phi) * t_steps
            obs_future_y = obs.y + obs.u * np.sin(obs.phi) * t_steps
            obs_future_traj = np.stack([obs_future_x, obs_future_y], axis=1)

            dists = np.linalg.norm(traj_points[:, :2] - obs_future_traj, axis=1)

            # 动态安全阈值
            base_threshold = self.veh_width / 2.0 + obs.w / 2.0 + 0.5 + margin
            # 如果是动态障碍物，增加额外的纵向安全距离 (Longitudinal Safety Buffer)
            # 这就是防止周车撞我的关键！即使横向没撞，如果纵向太近也不行。
            safety_buffer = 0.0
            if obs.type == "dynamic":
                # 计算相对速度
                # 简单近似：如果障碍物比我快且在我后面，或者比我慢且在我前面，都需要额外距离
                # 这里简化为统一加一个速度相关的 buffer
                safety_buffer = 0.5 * abs(obs.u - self.state[3])

            collision_threshold = base_threshold + safety_buffer
            if np.any(dists < collision_threshold):
                return True
        return False

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
        lat_error_done = np.abs(y - ref_y) > 7.0
        # 2. 航向误差限制 (建议收紧到 90度 或 60度)
        # 超过 90度 (np.pi / 2) 意味着车已经横过来了，基本无法恢复正常行驶
        heading_error_done = np.abs(angle_normalize(phi - ref_phi)) > (np.pi / 2)
        # 3. 碰撞检测 (复用更准确的逻辑)
        # 注意：这里直接调用 _check_ego_collision，它考虑了由车宽/长定义的矩形/圆形安全域
        collision_done = self._check_ego_collision()
        # 4. (可选) 纵向落后限制
        # 如果 x 轴严重落后于参考点(说明倒车或者停滞不前)，也可以 done
        dist_longi_done = (x - ref_x) < -10.0
        done = lat_error_done | heading_error_done | collision_done | dist_longi_done
        return bool(done)

    def _tracking_controller(self, traj_ego):
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
        ref_idx = min(target_idx, len(self.ref_points) - 1)
        target_v = self.ref_points[ref_idx, 3]

        # P 控制器
        k_p_acc = 5.0
        acc = k_p_acc * (target_v - self.state[3])
        steer = np.clip(steer, -self.max_steer, self.max_steer)
        acc = np.clip(acc, -self.max_accel, self.max_accel)
        return np.array([steer, acc], dtype=np.float32)
    @property
    def info(self):
        info = super().info
        # info.update({
        #     "is_success": self.is_success,
        # })
        return info

    def render_tracking_controller(self, mode='human'):
        import matplotlib.pyplot as plt

        # 1. 强制设置 Agg 后端 (解决服务器无头模式报错)
        if mode == 'rgb_array':
            plt.switch_backend('agg')

        # 2. 画布初始化
        if not hasattr(self, 'fig') or self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 6), dpi=80)

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
        核心绘图逻辑：修复车辆显示和中文字体问题 (精简图例版)
        """
        import matplotlib.patches as pc
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        from matplotlib.patches import Polygon

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

        # --- 1. 绘制自车 (Ego) ---
        ego_x, ego_y, phi = self.state[:3]
        ego_corners = get_rotated_rect(ego_x, ego_y, phi, veh_length, veh_width)

        ego_polygon = Polygon(
            ego_corners, closed=True, facecolor='magenta', edgecolor='darkmagenta',
            alpha=0.3, linewidth=2, zorder=20, label='自车'  # Label
        )
        ax.add_patch(ego_polygon)

        # 自车箭头 (不加图例)
        ax.arrow(ego_x, ego_y, veh_length * 0.8 * np.cos(phi), veh_length * 0.8 * np.sin(phi),
                 head_width=0.8, head_length=1.2, fc='red', ec='red', alpha=0.8, zorder=21)

        # --- 2. 绘制参考轨迹 (Global) ---
        if hasattr(self, 'ref_points') and self.ref_points is not None:
            ax.plot(self.ref_points[:, 0], self.ref_points[:, 1], 'b--', lw=2, zorder=2, label='全局轨迹')  # Label

        # --- 3. 绘制规划轨迹 (Planning) ---
        traj_global = getattr(self, 'current_planning_traj', None)
        if traj_global is not None and len(traj_global) > 0:
            ax.plot(traj_global[:, 0], traj_global[:, 1], 'pink', marker='.', markersize=7,
                    markeredgecolor='deeppink', markeredgewidth=0.5, linewidth=2, alpha=0.8, zorder=15,
                    label='规划轨迹')  # Label

        # --- 4. 绘制历史轨迹 ---
        if hasattr(self, 'history_traj') and len(self.history_traj) > 1:
            h_arr = np.array(self.history_traj)
            # ax.plot(h_arr[:, 0], h_arr[:, 1], 'gray', ls='-', lw=1.0, alpha=0.5, zorder=5, label='历史轨迹') # 不需要

        # --- 5. 绘制障碍物 (精简图例) ---
        # 标志位，确保每种类型只添加一次图例
        has_label_dynamic = False
        has_label_static_cross = False
        has_label_static_nocross = False

        if hasattr(self, 'obstacles'):
            for obs in self.obstacles:
                can_cross = getattr(obs, 'can_cross', False)
                is_dynamic = getattr(obs, 'type', 'static') == 'dynamic'

                # 确定样式和图例标签
                label = None
                if is_dynamic:
                    color = 'red'
                    if not has_label_dynamic:
                        label = '动态障碍物'
                        has_label_dynamic = True
                elif can_cross:
                    color = 'lime'
                    if not has_label_static_cross:
                        label = '静态障碍物(可跨)'
                        has_label_static_cross = True
                else:
                    color = 'gray'
                    if not has_label_static_nocross:
                        label = '静态障碍物(不可跨)'
                        has_label_static_nocross = True

                # 绘制
                obs_corners = get_rotated_rect(obs.x, obs.y, obs.phi, obs.l, obs.w)
                obs_poly = Polygon(obs_corners, closed=True, facecolor=color, edgecolor='black',
                                   alpha=0.1, linewidth=2, zorder=10, label=label)  # 仅第一次有label
                ax.add_patch(obs_poly)

                # 动态障碍物箭头
                if is_dynamic and abs(obs.u) > 0.1:
                    ax.arrow(obs.x, obs.y, obs.u * 0.5 * np.cos(obs.phi), obs.u * 0.5 * np.sin(obs.phi),
                             head_width=0.5, head_length=0.8, fc='darkred', ec='darkred', alpha=0.6, zorder=11)

        # --- 6. 绘制引导线 (合并图例) ---
        has_label_guide = False
        if hasattr(self, 'guide_trajectories') and self.guide_trajectories:
            colors = ['cyan', 'orange', 'purple']
            for i, guide_traj in enumerate(self.guide_trajectories):
                if len(guide_traj) > 1:
                    label = '引导线' if not has_label_guide else None
                    has_label_guide = True

                    ax.plot(guide_traj[:, 0], guide_traj[:, 1], color='cyan',
                            ls='--', lw=2.0, alpha=0.5, zorder=3, label=label)

        # --- 7. 绘制贝塞尔曲线 (合并图例) ---
        # if hasattr(self, 'best_curve') and self.best_curve is not None:
        #     # ... (省略具体计算，如有需要可保留)
        #     ax.plot(cx, cy, 'c--', lw=2, zorder=3, label='贝塞尔曲线')

        # --- 8. 设置视野 ---
        view_x_min, view_x_max = ego_x - 20, ego_x + 80
        view_y_min, view_y_max = ego_y - 10, ego_y + 10
        ax.set_xlim(view_x_min, view_x_max)
        ax.set_ylim(view_y_min, view_y_max)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=20)
        ax.tick_params(axis='x', direction='in')
        ax.tick_params(axis='y', direction='in')
        # --- 9. 设置坐标轴 ---
        # ego_speed = self.state[3] * 3.6
        # ax.set_title(f"时间: {self.t:.1f}s | 速度: {ego_speed:.1f} km/h", fontsize=14)
        ax.set_xlabel(r'纵向位置 $p_x$ (m)', fontsize=20)
        ax.set_ylabel(r'横向位置 $p_y$ (m)', fontsize=20)
        # ax.grid(False, alpha=0.3, ls='--', lw=0.5)

        # --- 10. 创建图例 (自动去重) ---
        # 因为我们在 plot/patch 时已经控制了 label 的唯一性，
        # 所以直接调用 legend() 即可，它会自动忽略 label=None 的对象
        ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, 1.45),  # 放在顶部居中
            ncol=4,  # 一行4个，不够换行
            fontsize=15,
            frameon=False,
            framealpha=0.8,
            fancybox=False
        )

        # --- 11. 布局调整 ---
        ax.figure.tight_layout()
def env_creator(**kwargs):
    return SimuVeh3dofcontiBimodalDiffusion(**kwargs)