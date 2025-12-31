# pyth_veh3dofconti_bimodaldiffusion_planning.py

import gym
import numpy as np
import math
from gym import spaces
from gops.env.env_ocp.pyth_veh3dofcontiplanning import SimuVeh3dofconti, angle_normalize
from gops.env.env_ocp.pyth_veh3dofcontiplanning import ego_vehicle_coordinate_transform
from gops.env.env_ocp.resources.ref_traj_data import MultiRefTrajData
from dataclasses import dataclass
from typing import List


# ==========================================
# 辅助类定义
# ==========================================

@dataclass
class Obstacle:
    """定义越野环境中的障碍物"""
    x: float
    y: float
    l: float = 2.0
    w: float = 2.0
    phi: float = 0.0
    u: float = 0.0  # 纵向速度
    type: str = "static"
    can_cross: bool = False


class BezierGenerator:
    @staticmethod
    def generate(p0, p1, p2, p3, num_points=20):
        t = np.linspace(0, 1, num_points)
        t = t[:, np.newaxis]
        curve = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3
        return curve


# ==========================================
# 核心环境类
# ==========================================

class SimuVeh3dofcontiBimodalDiffusion(gym.Env):
    def __init__(self, **kwargs):
        self.is_adversary = kwargs.get("is_adversary", False)
        self.is_constraint = kwargs.get("is_constraint", False)

        # --- 1. 预测时域与动作空间 ---
        self.pred_horizon = kwargs.get("pred_horizon", 20)
        self.action_dim = 2
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.pred_horizon * self.action_dim,),
            dtype=np.float32
        )

        # --- 2. 观测空间设计 (融合全局跟踪信息 + 引导信息) ---

        # A. 基础参考路径参数 (用于生成 ref_points)
        self.ref_horizon = 20  # 观测中包含的参考点数量

        # B. 障碍物参数
        self.max_obs_num = 5  # 观测中最多包含的障碍物数量
        self.obs_feat_dim = 8  # [x, y, phi, u, l, w, type, dist]

        # C. 维度计算
        self.dim_ego = 6
        self.dim_ref = (self.ref_horizon - 1) * 4
        self.dim_obstacles = self.max_obs_num * self.obs_feat_dim
        self.dim_prompts = 60

        self.total_obs_dim = self.dim_ego + self.dim_ref + self.dim_obstacles + self.dim_prompts
        print(f"Diffusion Environment Obs Dim: {self.total_obs_dim}")

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.total_obs_dim,),
            dtype=np.float32
        )

        # --- 3. 物理模型初始化 ---
        self.vehicle_dynamics = SimuVeh3dofconti(**kwargs)
        self.ref_traj = MultiRefTrajData()  # 加载参考轨迹数据
        self.dt = 0.1
        self.max_episode_steps = 200

        # 内部状态
        self.steps = 0
        self.time = 0.0
        self.obstacles: List[Obstacle] = []
        self.guide_trajectories: List[np.ndarray] = []
        self.current_diffusion_traj = None
        self.ref_points = None

        # 【关键修复】初始化随机种子
        self.seed()

    # =======================================================
    # 【关键修复】添加 seed 方法
    # =======================================================
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        # 同时也设置内部物理模型的随机种子
        if hasattr(self.vehicle_dynamics, 'seed'):
            self.vehicle_dynamics.seed(seed)
        return [seed]

    def reset(self):
        self.steps = 0
        self.time = 0.0

        # 1. 随机选择参考轨迹
        # 使用 self.np_random 替代 np.random 以支持 seed
        # self.path_num = self.np_random.randint(0, 3)
        # self.u_num = self.np_random.randint(0, 2)

        # 安全测试模式：
        self.path_num = 0
        self.u_num = 0

        # 2. 初始化参考点
        self.ref_points = np.zeros((self.ref_horizon, 4), dtype=np.float32)
        for i in range(self.ref_horizon):
            curr_t = self.time + i * self.dt
            self.ref_points[i, 0] = self.ref_traj.compute_x(curr_t, self.path_num, self.u_num)
            self.ref_points[i, 1] = self.ref_traj.compute_y(curr_t, self.path_num, self.u_num)
            self.ref_points[i, 2] = self.ref_traj.compute_phi(curr_t, self.path_num, self.u_num)
            self.ref_points[i, 3] = self.ref_traj.compute_u(curr_t, self.path_num, self.u_num)

        # 3. 重置车辆
        init_x = self.ref_points[0, 0]
        init_y = self.ref_points[0, 1]
        init_phi = self.ref_points[0, 2]
        init_u = self.ref_points[0, 3]

        self.vehicle_dynamics.reset()
        # 加上一点随机扰动，使用 self.np_random
        self.vehicle_dynamics.state = np.array([
            init_x,
            init_y + self.np_random.uniform(-0.5, 0.5),
            init_phi + self.np_random.uniform(-0.05, 0.05),
            init_u, 0, 0
        ], dtype=np.float32)

        self.state = self.vehicle_dynamics.state.copy()

        # 4. 生成障碍物
        self.obstacles = self._generate_offroad_obstacles()

        # 5. 生成引导轨迹 (Prompt)
        self.guide_trajectories = self._generate_guidance_prompts()

        return self._get_obs()

    def step(self, action):
        self.steps += 1
        self.time += self.dt
        # 1. 解析 Diffusion Action
        self.current_diffusion_traj = action.reshape(self.pred_horizon, self.action_dim)

        # 2. 更新参考轨迹
        self._update_ref_points()

        # 3. 轨迹跟踪控制 (P Controller)
        target_dx = self.current_diffusion_traj[0, 0]
        target_dy = self.current_diffusion_traj[0, 1]

        # 获取参考速度
        ref_u = self.ref_points[0, 3]
        u_current = self.state[3]

        # 纵向控制
        acc = 1.0 * (ref_u - u_current) + 0.5 * target_dx
        acc = np.clip(acc, -3.0, 2.0)

        # 横向
        if abs(target_dx) < 0.1:
            steer = 0.0
        else:
            steer = math.atan2(target_dy, target_dx)
        steer = np.clip(steer, -0.4, 0.4)

        real_action = np.array([steer, acc], dtype=np.float32)

        # 4. 物理步进
        self.vehicle_dynamics.step(real_action)
        self.state = self.vehicle_dynamics.state.copy()

        # 5. 更新引导
        self.guide_trajectories = self._generate_guidance_prompts()

        obs = self._get_obs()
        reward = self._compute_reward(real_action)
        done = self.steps >= self.max_episode_steps

        # 出界判定
        lat_error = abs(self.state[1] - self.ref_points[0, 1])
        if lat_error > 5.0:
            done = True
            reward -= 100.0

        return obs, reward, done, {}

    def _update_ref_points(self):
        # 整体前移一位
        self.ref_points[:-1] = self.ref_points[1:]

        # 计算新点
        future_t = self.time + (self.ref_horizon - 1) * self.dt
        new_ref_point = np.array([
            self.ref_traj.compute_x(future_t, self.path_num, self.u_num),
            self.ref_traj.compute_y(future_t, self.path_num, self.u_num),
            self.ref_traj.compute_phi(future_t, self.path_num, self.u_num),
            self.ref_traj.compute_u(future_t, self.path_num, self.u_num),
        ], dtype=np.float32)
        self.ref_points[-1] = new_ref_point

    def _get_obs(self):
        # 1. 坐标变换: 全局参考点 -> 局部 Ego 坐标系
        ref_x_tf, ref_y_tf, ref_phi_tf = ego_vehicle_coordinate_transform(
            self.state[0], self.state[1], self.state[2],
            self.ref_points[:, 0], self.ref_points[:, 1], self.ref_points[:, 2],
        )
        ref_u_tf = self.ref_points[:, 3] - self.state[3]

        # 2. Ego Obs
        ego_obs = np.concatenate(
            ([ref_x_tf[0], ref_y_tf[0], ref_phi_tf[0], ref_u_tf[0]], self.state[4:])
        )

        # 3. Ref Preview Obs
        ref_obs = np.stack((ref_x_tf, ref_y_tf, ref_phi_tf, ref_u_tf), 1)[1:].flatten()

        # 4. Obstacle Obs
        obs_feats = []
        dists = []
        for obs in self.obstacles:
            d = np.sqrt((obs.x - self.state[0]) ** 2 + (obs.y - self.state[1]) ** 2)
            dists.append((d, obs))
        dists.sort(key=lambda x: x[0])

        for i in range(self.max_obs_num):
            if i < len(dists):
                d, obs = dists[i]
                ox_tf, oy_tf, ophi_tf = ego_vehicle_coordinate_transform(
                    self.state[0], self.state[1], self.state[2],
                    np.array([obs.x]), np.array([obs.y]), np.array([obs.phi])
                )
                ou_tf = obs.u - self.state[3]
                type_code = 1.0 if obs.can_cross else 0.0
                feat = [ox_tf[0], oy_tf[0], ophi_tf[0], ou_tf, obs.l, obs.w, type_code, d]
            else:
                feat = [0.0] * self.obs_feat_dim
            obs_feats.extend(feat)

        obstacle_obs = np.array(obs_feats, dtype=np.float32)

        # 5. Prompts Obs
        prompt_feats = []
        for traj in self.guide_trajectories:
            indices = np.linspace(0, len(traj) - 1, 10, dtype=int)
            sampled = traj[indices]
            px_tf, py_tf, _ = ego_vehicle_coordinate_transform(
                self.state[0], self.state[1], self.state[2],
                sampled[:, 0], sampled[:, 1], np.zeros(10)
            )
            prompt_feats.append(np.stack([px_tf, py_tf], axis=1).flatten())

        prompt_obs = np.concatenate(prompt_feats)
        return np.concatenate((ego_obs, ref_obs, obstacle_obs, prompt_obs))

    def _generate_offroad_obstacles(self):
        obs = []
        ref_x = self.ref_points[-1, 0]  # 远端参考点附近生成
        ref_y = self.ref_points[-1, 1]

        # 1. 静态大石头 (不可跨)
        obs.append(Obstacle(x=ref_x, y=ref_y + 1.0, l=2.0, w=2.0, can_cross=False))
        # 2. 倒伏树木 (可跨)
        obs.append(Obstacle(x=self.state[0] + 30, y=self.state[1] - 0.5, l=1.0, w=3.0, can_cross=True))
        return obs

    def _generate_guidance_prompts(self):
        prompts = []
        ego_x, ego_y, ego_phi = self.state[0], self.state[1], self.state[2]

        target_ref_idx = self.pred_horizon - 1
        ref_target = self.ref_points[target_ref_idx]
        tx, ty, tphi = ref_target[0], ref_target[1], ref_target[2]

        offsets = [3.0, 0.0, -3.0]  # Left, Center, Right

        for lat_offset in offsets:
            p0 = np.array([ego_x, ego_y])

            # 计算目标点 (沿参考线法向偏移)
            nx, ny = -np.sin(tphi), np.cos(tphi)
            p3 = np.array([tx + nx * lat_offset, ty + ny * lat_offset])

            # 控制点
            dist = np.linalg.norm(p3 - p0)
            p1 = p0 + np.array([np.cos(ego_phi), np.sin(ego_phi)]) * (dist * 0.4)
            p2 = p3 - np.array([np.cos(tphi), np.sin(tphi)]) * (dist * 0.4)

            curve = BezierGenerator.generate(p0, p1, p2, p3, num_points=self.pred_horizon)
            prompts.append(curve)
        return prompts

    def _compute_reward(self, real_action):
        ego_x, ego_y, ego_u = self.state[0], self.state[1], self.state[3]
        steer, acc = real_action
        ref_u = self.ref_points[0, 3]

        # 1. 速度追踪
        r_velocity = -1.0 * (ego_u - ref_u) ** 2
        if ego_u < 0.1: r_velocity -= 5.0

        # 2. 引导一致性
        min_prompt_dist = float('inf')
        for traj in self.guide_trajectories:
            prompt_pt = traj[0]
            dist_y = np.linalg.norm([ego_x - prompt_pt[0], ego_y - prompt_pt[1]])
            if dist_y < min_prompt_dist:
                min_prompt_dist = dist_y
        r_lateral = -1.0 * max(0, min_prompt_dist - 1.5) ** 2

        # 3. 避障
        r_collision = 0.0
        for obs in self.obstacles:
            dist = np.sqrt((ego_x - obs.x) ** 2 + (ego_y - obs.y) ** 2)
            danger_radius = 2.5
            if dist < danger_radius:
                if obs.can_cross:
                    r_collision -= 5.0 * (danger_radius - dist)
                else:
                    r_collision -= 50.0 * (danger_radius - dist)

        r_smooth = -0.1 * steer ** 2 - 0.01 * acc ** 2
        return r_velocity + r_lateral + r_collision + r_smooth

    def render(self, mode='human'):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        if not hasattr(self, 'fig') or self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.cla()

        ego_x, ego_y, ego_phi = self.state[0], self.state[1], self.state[2]

        # 1. 参考线
        if self.ref_points is not None:
            self.ax.plot(self.ref_points[:, 0], self.ref_points[:, 1], 'b--', alpha=0.3, label='Global Ref')

        # 2. 障碍
        for obs in self.obstacles:
            color = 'lime' if obs.can_cross else 'gray'
            rect = patches.Rectangle((obs.x - obs.l / 2, obs.y - obs.w / 2), obs.l, obs.w, angle=np.degrees(obs.phi),
                                     facecolor=color, edgecolor='k')
            self.ax.add_patch(rect)

        # 3. Prompt
        for traj in self.guide_trajectories:
            self.ax.plot(traj[:, 0], traj[:, 1], 'g:', alpha=0.5)

        # 4. Action
        if self.current_diffusion_traj is not None:
            diff = self.current_diffusion_traj
            c, s = np.cos(ego_phi), np.sin(ego_phi)
            gx = ego_x + diff[:, 0] * c - diff[:, 1] * s
            gy = ego_y + diff[:, 0] * s + diff[:, 1] * c
            self.ax.plot(gx, gy, 'r-', lw=2, label='Diffusion')

        # 5. Ego
        car = patches.Rectangle((ego_x - 2, ego_y - 1), 4, 2, angle=np.degrees(ego_phi), facecolor='blue')
        self.ax.add_patch(car)

        self.ax.set_xlim(ego_x - 10, ego_x + 50)
        self.ax.set_ylim(ego_y - 15, ego_y + 15)
        self.ax.legend()
        plt.pause(0.01)
def env_creator(**kwargs):
    return SimuVeh3dofcontiBimodalDiffusion(**kwargs)