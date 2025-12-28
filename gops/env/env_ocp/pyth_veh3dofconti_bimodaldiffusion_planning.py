# pyth_veh3dofconti_bimodaldiffusion_planning.py

import gym
import numpy as np
import math
from gym import spaces
from gops.env.env_ocp.pyth_veh3dofcontiplanning import SimuVeh3dofconti, angle_normalize
# 引入必要的坐标变换函数
from gops.env.env_ocp.pyth_veh3dofcontiplanning import ego_vehicle_coordinate_transform
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
        self.ref_horizon = 20  # 观测中包含的参考点数量 (和 pred_horizon 可以不同，通常一致)

        # B. 障碍物参数
        self.max_obs_num = 5  # 观测中最多包含的障碍物数量
        self.obs_feat_dim = 8  # [x, y, phi, u, l, w, type, dist]

        # C. 维度计算
        # Part 1: Ego Tracking (6) -> [dx, dy, dphi, du, v, w] (相对于第1个参考点)
        self.dim_ego = 6

        # Part 2: Ref Preview (N-1) * 4 -> [dx, dy, dphi, du] (相对于自车)
        # 注意：user snippet 中是 ref_obs = np.stack(...)[1:].flatten()，所以是 (ref_horizon - 1) * 4
        self.dim_ref = (self.ref_horizon - 1) * 4

        # Part 3: Obstacles -> max_obs_num * 8
        self.dim_obstacles = self.max_obs_num * self.obs_feat_dim

        # Part 4: Prompts -> 3条 * 10点 * 2坐标 = 60
        self.dim_prompts = 60

        self.total_obs_dim = self.dim_ego + self.dim_ref + self.dim_obstacles + self.dim_prompts
        print(f"Diffusion Environment Obs Dim: {self.total_obs_dim}")
        # 结果大约是: 6 + 76 + 40 + 60 = 182 维 (不再强行凑106，信息全更重要)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.total_obs_dim,),
            dtype=np.float32
        )

        # --- 3. 物理模型初始化 ---
        self.vehicle_dynamics = SimuVeh3dofconti(**kwargs)
        self.dt = 0.1
        self.max_episode_steps = 200

        # 内部状态
        self.steps = 0
        self.time = 0.0
        self.obstacles: List[Obstacle] = []
        self.guide_trajectories: List[np.ndarray] = []
        self.current_diffusion_traj = None

        # 参考路径缓存 (模拟全局路径)
        self.ref_points = None

    def reset(self):
        self.steps = 0
        self.time = 0.0

        # 1. 重置车辆
        self.vehicle_dynamics.reset()
        self.state = self.vehicle_dynamics.state.copy()

        # 2. 初始化全局参考路径 (模拟一条直线或正弦曲线)
        # 这里为了演示，生成一条 y=0, u=10 的直线参考路径
        self._update_ref_points()

        # 3. 生成障碍物
        self.obstacles = self._generate_offroad_obstacles()

        # 4. 生成引导轨迹 (Prompt)
        self.guide_trajectories = self._generate_guidance_prompts()

        return self._get_obs()

    def step(self, action):
        self.steps += 1
        self.time += self.dt
        # 1. 解析 Diffusion Action
        self.current_diffusion_traj = action.reshape(self.pred_horizon, self.action_dim)

        # 2. 轨迹跟踪控制 (P Controller)
        target_dx = self.current_diffusion_traj[0, 0]
        target_dy = self.current_diffusion_traj[0, 1]

        # 纵向: 追踪参考速度 10m/s
        u_current = self.state[3]
        acc = 1.0 * (10.0 - u_current) + 1.0 * target_dx
        acc = np.clip(acc, -3.0, 2.0)

        # 横向
        if abs(target_dx) < 0.1:
            steer = 0.0
        else:
            steer = math.atan2(target_dy, target_dx)
        steer = np.clip(steer, -0.4, 0.4)

        real_action = np.array([steer, acc], dtype=np.float32)

        # 3. 物理步进
        self.vehicle_dynamics.step(real_action)
        self.state = self.vehicle_dynamics.state.copy()

        # 4. 更新环境信息
        self._update_ref_points()  # 更新参考点窗口
        self.guide_trajectories = self._generate_guidance_prompts()

        obs = self._get_obs()
        reward = self._compute_reward(real_action)
        done = self.steps >= self.max_episode_steps

        return obs, reward, done, {}

    def _update_ref_points(self):
        """
        更新当前的参考点窗口 (Global Frame)
        模拟车辆在一条 y=0 的无限长直路上行驶
        """
        # 生成未来 ref_horizon 个点
        ref_x = []
        ref_y = []
        ref_phi = []
        ref_u = []

        # 简单的直线路径逻辑：参考点始终在车前方沿 X 轴延伸
        # 在实际 GOPS 中，这里会读取预加载的 Path 文件
        start_x = self.state[0]
        for i in range(self.ref_horizon):
            dist = i * self.dt * 10.0  # 假设参考速度 10m/s
            ref_x.append(start_x + dist)  # 这里的逻辑可以改为绝对坐标路径
            ref_y.append(0.0)  # 始终保持在 y=0
            ref_phi.append(0.0)
            ref_u.append(10.0)

        self.ref_points = np.stack([ref_x, ref_y, ref_phi, ref_u], axis=1)

    def _get_obs(self):
        """
        构建融合观测向量，严格包含用户要求的 ref_obs 和 dynamic_obs
        """
        # ==========================================
        # Part 1: Base Obs (基于你的代码片段)
        # ==========================================

        # 1. 坐标变换: 全局参考点 -> 局部 Ego 坐标系
        # self.ref_points shape: (N, 4) -> x, y, phi, u
        ref_x_tf, ref_y_tf, ref_phi_tf = ego_vehicle_coordinate_transform(
            self.state[0], self.state[1], self.state[2],
            self.ref_points[:, 0], self.ref_points[:, 1], self.ref_points[:, 2],
        )
        ref_u_tf = self.ref_points[:, 3] - self.state[3]

        # 2. 构建 ego_obs (6维)
        # [delta_x, delta_y, delta_phi, delta_u (of 1st ref point), v, w]
        ego_obs = np.concatenate(
            ([ref_x_tf[0], ref_y_tf[0], ref_phi_tf[0], ref_u_tf[0]], self.state[4:])
        )

        # 3. 构建 ref_obs (未来路径预览)
        # 剔除第1个点，扁平化剩余点
        # shape: (N-1, 4) -> flattened
        ref_obs = np.stack((ref_x_tf, ref_y_tf, ref_phi_tf, ref_u_tf), 1)[1:].flatten()

        # 4. 构建 Obstacle Obs
        # 我们这里不区分 dynamic/static 变量名，而是统一处理成 fixed size vector
        # 寻找最近的 max_obs_num 个障碍物
        obs_feats = []

        # 计算距离
        dists = []
        for obs in self.obstacles:
            d = np.sqrt((obs.x - self.state[0]) ** 2 + (obs.y - self.state[1]) ** 2)
            dists.append((d, obs))
        dists.sort(key=lambda x: x[0])

        for i in range(self.max_obs_num):
            if i < len(dists):
                d, obs = dists[i]
                # 坐标变换: 全局 -> 局部
                ox_tf, oy_tf, ophi_tf = ego_vehicle_coordinate_transform(
                    self.state[0], self.state[1], self.state[2],
                    np.array([obs.x]), np.array([obs.y]), np.array([obs.phi])
                )
                ou_tf = obs.u - self.state[3]  # <--- 这里计算出来是标量 (float)

                # 编码: [rel_x, rel_y, rel_phi, rel_u, l, w, type, dist]
                # type 1.0cross, 0.0=
                type_code = 1.0 if obs.can_cross else 0.0
                feat = [ox_tf[0], oy_tf[0], ophi_tf[0], ou_tf, obs.l, obs.w, type_code, d]
            else:
                feat = [0.0] * self.obs_feat_dim  # Padding
            obs_feats.extend(feat)

        obstacle_obs = np.array(obs_feats, dtype=np.float32)

        # ==========================================
        # Part 2: Prompts (引导轨迹)

        prompt_feats = []
        for traj in self.guide_trajectories:
            # traj: (20, 2) global
            # 降采样到 10 个点
            indices = np.linspace(0, len(traj) - 1, 10, dtype=int)
            sampled = traj[indices]

            # 全局 -> 局部
            px_tf, py_tf, _ = ego_vehicle_coordinate_transform(
                self.state[0], self.state[1], self.state[2],
                sampled[:, 0], sampled[:, 1], np.zeros(10)  # phi 不重要
            )

            prompt_feats.append(np.stack([px_tf, py_tf], axis=1).flatten())

        prompt_obs = np.concatenate(prompt_feats)

        # ==========================================
        # Part 3: 拼接总观测
        # ==========================================
        total_obs = np.concatenate((ego_obs, ref_obs, obstacle_obs, prompt_obs))

        return total_obs

    def _generate_offroad_obstacles(self):
        obs = []
        ego_x = self.state[0]
        # 1. 静态大石头 (不可跨)
        obs.append(Obstacle(x=ego_x + 25, y=0.0, l=2.0, w=2.0, can_cross=False))
        # 2. 动态车辆 (假设同向低速)
        obs.append(Obstacle(x=ego_x + 40, y=-3.0, l=4.0, w=1.8, u=2.0, type="dynamic"))
        # 3. 倒伏树木 (可跨)
        obs.append(Obstacle(x=ego_x + 50, y=0.0, l=1.0, w=3.0, can_cross=True))
        return obs

    def _generate_guidance_prompts(self):
        # ... (保持之前的贝塞尔生成逻辑不变) ...
        prompts = []
        ego_x, ego_y, ego_phi = self.state[0], self.state[1], self.state[2]
        lookahead = 20.0
        offsets = [3.0, 0.0, -3.0]

        for lat_offset in offsets:
            p0 = np.array([ego_x, ego_y])
            gx = ego_x + lookahead * np.cos(ego_phi) - lat_offset * np.sin(ego_phi)
            gy = ego_y + lookahead * np.sin(ego_phi) + lat_offset * np.cos(ego_phi)
            p3 = np.array([gx, gy])
            p1 = p0 + np.array([np.cos(ego_phi), np.sin(ego_phi)]) * (lookahead * 0.4)
            p2 = p3 - np.array([np.cos(ego_phi), np.sin(ego_phi)]) * (lookahead * 0.4)
            curve = BezierGenerator.generate(p0, p1, p2, p3, num_points=self.pred_horizon)
            prompts.append(curve)
        return prompts

    def _compute_reward(self, real_action):
        # 简单奖励用于测试
        return -abs(self.state[1])  # 保持在中心线

    def render(self, mode='human'):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        if not hasattr(self, 'fig') or self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.cla()

        ego_x, ego_y, ego_phi = self.state[0], self.state[1], self.state[2]

        # 1. 画路
        self.ax.plot([ego_x - 10, ego_x + 60], [4, 4], 'k-')
        self.ax.plot([ego_x - 10, ego_x + 60], [-4, -4], 'k-')

        # 2. 画参考线
        if self.ref_points is not None:
            self.ax.plot(self.ref_points[:, 0], self.ref_points[:, 1], 'b--', alpha=0.3, label='Global Ref')

        # 3. 画障碍
        for obs in self.obstacles:
            color = 'lime' if obs.can_cross else 'gray'
            rect = patches.Rectangle((obs.x - obs.l / 2, obs.y - obs.w / 2), obs.l, obs.w, angle=np.degrees(obs.phi),
                                     facecolor=color, edgecolor='k')
            self.ax.add_patch(rect)

        # 4. 画 Prompt
        for traj in self.guide_trajectories:
            self.ax.plot(traj[:, 0], traj[:, 1], 'g:', alpha=0.5)

        # 5. 画 Action
        if self.current_diffusion_traj is not None:
            # Local -> Global
            diff = self.current_diffusion_traj
            c, s = np.cos(ego_phi), np.sin(ego_phi)
            gx = ego_x + diff[:, 0] * c - diff[:, 1] * s
            gy = ego_y + diff[:, 0] * s + diff[:, 1] * c
            self.ax.plot(gx, gy, 'r-', lw=2, label='Diffusion')

        # 6. 画 Ego
        car = patches.Rectangle((ego_x - 2, ego_y - 1), 4, 2, angle=np.degrees(ego_phi), facecolor='blue')
        self.ax.add_patch(car)

        self.ax.set_xlim(ego_x - 10, ego_x + 50)
        self.ax.set_ylim(-10, 10)
        self.ax.legend()
        plt.pause(0.01)


# # Debug
# if __name__ == "__main__":
#     env = SimuVeh3dofcontiBimodalDiffusion()
#     env.reset()
#     for _ in range(20):
#         env.step(np.random.randn(40))
#         env.render()