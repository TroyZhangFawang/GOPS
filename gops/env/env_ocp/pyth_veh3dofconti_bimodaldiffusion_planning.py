# pyth_veh3dofconti_bimodaldiffusion_planning.py

import gym
import numpy as np
import math
from gym import spaces
from gops.env.env_ocp.pyth_veh3dofcontiplanning import SimuVeh3dofconti, angle_normalize
from gops.env.env_ocp.pyth_veh3dofcontiplanning import ego_vehicle_coordinate_transform
from gops.env.env_ocp.resources.ref_traj_data import MultiRefTrajData  # 核心引入
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
    u: float = 0.0
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

        # --- 2. 观测空间设计 ---
        # 必须与 ref_traj 保持一致
        self.ref_horizon = 20

        # 障碍物配置
        self.max_obs_num = 5
        self.obs_feat_dim = 8

        # 维度计算
        self.dim_ego = 6
        self.dim_ref = (self.ref_horizon - 1) * 4  # 剔除第一个点后的相对预览
        self.dim_obstacles = self.max_obs_num * self.obs_feat_dim
        self.dim_prompts = 60  # 3条 * 10点 * 2坐标

        self.total_obs_dim = self.dim_ego + self.dim_ref + self.dim_obstacles + self.dim_prompts
        print(f"Diffusion Environment Obs Dim: {self.total_obs_dim}")

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.total_obs_dim,),
            dtype=np.float32
        )

        # --- 3. 物理模型与参考轨迹 ---
        self.vehicle_dynamics = SimuVeh3dofconti(**kwargs)
        self.ref_traj = MultiRefTrajData()  # 加载参考轨迹数据
        self.dt = 0.1
        self.max_episode_steps = 200

        # 内部状态
        self.steps = 0
        self.t = 0.0
        self.path_num = 0  # 选择哪条路 (正弦/双移线等)
        self.u_num = 0  # 选择哪个速度配置

        self.obstacles: List[Obstacle] = []
        self.guide_trajectories: List[np.ndarray] = []
        self.current_diffusion_traj = None
        self.ref_points = None

    def reset(self):
        self.steps = 0
        self.t = 0.0

        # 1. 随机选择一条参考轨迹 (Path & Speed)
        # 假设 MultiRefTrajData 提供了多种路径类型
        self.path_num = np.random.randint(0, 3)  # 随机选路径 geometry
        self.u_num = np.random.randint(0, 2)  # 随机选速度 profile

        # 2. 初始化参考点 (Initial Reference Points)
        self.ref_points = np.zeros((self.ref_horizon, 4), dtype=np.float32)
        for i in range(self.ref_horizon):
            curr_t = self.t + i * self.dt
            self.ref_points[i, 0] = self.ref_traj.compute_x(curr_t, self.path_num, self.u_num)
            self.ref_points[i, 1] = self.ref_traj.compute_y(curr_t, self.path_num, self.u_num)
            self.ref_points[i, 2] = self.ref_traj.compute_phi(curr_t, self.path_num, self.u_num)
            self.ref_points[i, 3] = self.ref_traj.compute_u(curr_t, self.path_num, self.u_num)

        # 3. 重置车辆 (根据参考轨迹起点初始化，加一点随机扰动)
        init_x = self.ref_points[0, 0]
        init_y = self.ref_points[0, 1]
        init_phi = self.ref_points[0, 2]
        init_u = self.ref_points[0, 3]  # 初始速度跟随参考速度

        # 这里的 reset 需要根据 SimuVeh3dofconti 的实现来调整，假设支持 init_state
        # 为了简单，我们手动设置 vehicle_dynamics 的内部状态
        self.vehicle_dynamics.reset()
        self.vehicle_dynamics.state = np.array([
            init_x, init_y + np.random.uniform(-0.5, 0.5),  # 初始横向误差
                    init_phi + np.random.uniform(-0.05, 0.05),  # 初始航向误差
            init_u, 0, 0
        ], dtype=np.float32)

        self.state = self.vehicle_dynamics.state.copy()

        # 4. 生成障碍物
        self.obstacles = self._generate_offroad_obstacles()

        # 5. 生成引导轨迹
        self.guide_trajectories = self._generate_guidance_prompts()

        return self._get_obs()

    def step(self, action):
        self.steps += 1
        self.t += self.dt

        # 1. 解析 Diffusion Action
        self.current_diffusion_traj = action.reshape(self.pred_horizon, self.action_dim)

        # 2. 更新参考轨迹 (Rolling Update - 用户指定逻辑)
        self._update_ref_points()

        # 3. 轨迹跟踪控制 (P Controller)
        target_dx = self.current_diffusion_traj[0, 0]
        target_dy = self.current_diffusion_traj[0, 1]

        # 【关键修改】获取当前时刻参考点的期望速度 ref_u
        # ref_points[0] 是当前时刻的参考点
        ref_u = self.ref_points[0, 3]

        u_current = self.state[3]
        # 纵向控制: 追踪 ref_u，并结合 diffusion 输出的纵向趋势 target_dx
        acc = 1.0 * (ref_u - u_current) + 0.5 * target_dx
        acc = np.clip(acc, -3.0, 2.0)

        # 横向控制
        if abs(target_dx) < 0.1:
            steer = 0.0
        else:
            steer = math.atan2(target_dy, target_dx)
        steer = np.clip(steer, -0.4, 0.4)

        real_action = np.array([steer, acc], dtype=np.float32)

        # 4. 物理步进
        self.vehicle_dynamics.step(real_action)
        self.state = self.vehicle_dynamics.state.copy()

        # 5. 更新环境信息
        # 障碍物相对位置变了，Prompt 需要重新基于新的 Ref Path 生成
        self.guide_trajectories = self._generate_guidance_prompts()

        obs = self._get_obs()
        reward = self._compute_reward(real_action)
        done = self.steps >= self.max_episode_steps

        # 简单的出界判断 (如果偏离参考线太远)
        lat_error = abs(self.state[1] - self.ref_points[0, 1])
        if lat_error > 5.0:
            done = True
            reward -= 100.0

        return obs, reward, done, {"ref_u": ref_u}

    def _update_ref_points(self):
        """
        按照用户要求的滚动更新逻辑更新参考点
        """
        # 1. 整体前移一位
        self.ref_points[:-1] = self.ref_points[1:]

        # 2. 计算新的末尾点 (t + horizon * dt)
        # 注意：用户代码里用了 self.pre_horizon，这里统一用 self.pred_horizon (或 self.ref_horizon)
        # 假设 pred_horizon == ref_horizon
        future_t = self.t + (self.ref_horizon - 1) * self.dt

        new_ref_point = np.array(
            [
                self.ref_traj.compute_x(future_t, self.path_num, self.u_num),
                self.ref_traj.compute_y(future_t, self.path_num, self.u_num),
                self.ref_traj.compute_phi(future_t, self.path_num, self.u_num),
                self.ref_traj.compute_u(future_t, self.path_num, self.u_num),
            ],
            dtype=np.float32,
        )
        self.ref_points[-1] = new_ref_point

    def _get_obs(self):
        # 1. 参考点坐标变换 (Global -> Ego)
        ref_x_tf, ref_y_tf, ref_phi_tf = ego_vehicle_coordinate_transform(
            self.state[0], self.state[1], self.state[2],
            self.ref_points[:, 0], self.ref_points[:, 1], self.ref_points[:, 2],
        )
        ref_u_tf = self.ref_points[:, 3] - self.state[3]  # speed error

        # 2. Ego Obs (包含相对第一个点的误差)
        ego_obs = np.concatenate(
            ([ref_x_tf[0], ref_y_tf[0], ref_phi_tf[0], ref_u_tf[0]], self.state[4:])
        )

        # 3. Ref Preview Obs (剩余点)
        ref_obs = np.stack((ref_x_tf, ref_y_tf, ref_phi_tf, ref_u_tf), 1)[1:].flatten()

        # 4. Obstacles Obs
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
        # 基于当前参考路径生成障碍物 (让障碍物出现在路上，而不是随机荒野)
        # 简单起见，我们在当前车前方一定距离生成
        obs = []
        ref_x = self.ref_points[-1, 0]  # 远端
        ref_y = self.ref_points[-1, 1]

        # 静态大石头 (不可跨)
        obs.append(Obstacle(x=ref_x, y=ref_y + 1.0, l=2.0, w=2.0, can_cross=False))
        # 倒伏树木 (可跨)
        obs.append(Obstacle(x=self.state[0] + 30, y=self.state[1] - 0.5, l=1.0, w=3.0, can_cross=True))
        return obs

    def _generate_guidance_prompts(self):
        """
        基于【全局参考轨迹】生成引导线 (Left/Center/Right)
        确保引导线顺着路的弯曲方向
        """
        prompts = []
        ego_x, ego_y, ego_phi = self.state[0], self.state[1], self.state[2]

        # 获取远端参考点 (Lookahead)
        # 比如取 Ref Horizon 的中间或末尾
        target_ref_idx = self.pred_horizon - 1
        ref_target = self.ref_points[target_ref_idx]  # [x, y, phi, u]

        tx, ty, tphi = ref_target[0], ref_target[1], ref_target[2]

        # 定义相对于参考路径的横向偏移 (Frenet Frame 下的 d)
        lat_offsets = [3.0, 0.0, -3.0]  # Left, Center, Right

        for lat_offset in lat_offsets:
            p0 = np.array([ego_x, ego_y])
            # 目标点 P3：在参考点的基础上，沿着法线方向偏移
            # 法线方向 = tphi + 90度
            nx = -np.sin(tphi)
            ny = np.cos(tphi)

            p3 = np.array([
                tx + nx * lat_offset,
                ty + ny * lat_offset
            ])

            # 控制点 P1, P2
            # P1 顺着当前车头
            dist = np.linalg.norm(p3 - p0)
            p1 = p0 + np.array([np.cos(ego_phi), np.sin(ego_phi)]) * (dist * 0.4)

            # P2 顺着目标点的切向 (即参考路径方向)
            p2 = p3 - np.array([np.cos(tphi), np.sin(tphi)]) * (dist * 0.4)

            curve = BezierGenerator.generate(p0, p1, p2, p3, num_points=self.pred_horizon)
            prompts.append(curve)

        return prompts

    def _compute_reward(self, real_action):
        """
        更新后的 Reward: 考虑 ref_u 追踪
        """
        ego_x, ego_y, ego_u = self.state[0], self.state[1], self.state[3]
        ref_u = self.ref_points[0, 3]  # 当前参考速度
        steer, acc = real_action

        # 1. 速度追踪奖励 (Velocity Tracking)
        # 惩罚与参考速度的偏差
        r_velocity = -1.0 * (ego_u - ref_u) ** 2

        # 2. 引导一致性奖励,只要靠近任意一条 Prompt 就不惩罚
        min_prompt_dist = float('inf')
        for traj in self.guide_trajectories:
            # 找到最近的 Prompt y (近似)
            # 更精确的做法是算点到曲线距离，这里简化为终点距离或采样点距离
            # 为了效率，我们比较当前位置和 Prompt 上对应点的距离
            # Prompt[0] 是起点，Prompt[1] 是下一步...
            # 取 Prompt[0] 附近的点比较
            prompt_pt = traj[0]
            d = np.linalg.norm([ego_x - prompt_pt[0], ego_y - prompt_pt[1]])
            if d < min_prompt_dist:
                min_prompt_dist = d

        r_lateral = -1.0 * max(0, min_prompt_dist - 1.0) ** 2

        # 3. 避障奖励
        r_collision = 0.0
        for obs in self.obstacles:
            dist = np.sqrt((ego_x - obs.x) ** 2 + (ego_y - obs.y) ** 2)
            if dist < 2.5:
                if obs.can_cross:
                    r_collision -= 5.0 * (2.5 - dist)
                else:
                    r_collision -= 50.0 * (2.5 - dist)

        r_smooth = -0.1 * steer ** 2 - 0.1 * acc ** 2

        return r_velocity + r_lateral + r_collision + r_smooth

    def render(self, mode='human'):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        if not hasattr(self, 'fig') or self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.cla()

        ego_x, ego_y, ego_phi = self.state[0], self.state[1], self.state[2]

        # 画参考轨迹 (蓝色虚线)
        if self.ref_points is not None:
            self.ax.plot(self.ref_points[:, 0], self.ref_points[:, 1], 'b--', linewidth=2, label='Global Ref')

        # 画障碍
        for obs in self.obstacles:
            color = 'lime' if obs.can_cross else 'gray'
            rect = patches.Rectangle((obs.x - obs.l / 2, obs.y - obs.w / 2), obs.l, obs.w, angle=np.degrees(obs.phi),
                                     facecolor=color, edgecolor='k')
            self.ax.add_patch(rect)

        # 画 Prompts
        for traj in self.guide_trajectories:
            self.ax.plot(traj[:, 0], traj[:, 1], 'g:', alpha=0.5)

        # 画 Diffusion Action
        if self.current_diffusion_traj is not None:
            diff = self.current_diffusion_traj
            c, s = np.cos(ego_phi), np.sin(ego_phi)
            gx = ego_x + diff[:, 0] * c - diff[:, 1] * s
            gy = ego_y + diff[:, 0] * s + diff[:, 1] * c
            self.ax.plot(gx, gy, 'r-', lw=2, label='Diffusion')

        # 画 Ego
        car = patches.Rectangle((ego_x - 2, ego_y - 1), 4, 2, angle=np.degrees(ego_phi), facecolor='blue')
        self.ax.add_patch(car)

        self.ax.set_xlim(ego_x - 10, ego_x + 50)
        self.ax.set_ylim(ego_y - 20, ego_y + 20)  # 视野扩大一点看弯道
        self.ax.legend()
        plt.pause(0.01)