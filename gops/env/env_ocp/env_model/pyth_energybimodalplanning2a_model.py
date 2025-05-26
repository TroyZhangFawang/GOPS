#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: vehicle 4DOF model environment

from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
from gops.env.env_ocp.env_model.pyth_base_model import PythBaseModel
from gops.env.env_ocp.pyth_energybimodalplanning2a import angle_normalize, VehicleDynamicsData, StaticObstacle
from gops.env.env_ocp.resources.ref_traj_model import MultiRefTrajModel, MultiRoadSlopeModel
from gops.utils.gops_typing import InfoDict

class VehicleDynamicsModel(VehicleDynamicsData):
    l: float = 3.0
    dt: float = 0.1
    obs_length: float = 4.8
    obs_width: float = 2.0
    def f_xu(self, states, actions, disturb, delta_t):
        x, y, phi, u, v, w, psi, psi_dot = (
            states[:, 0],
            states[:, 1],
            states[:, 2],
            states[:, 3],
            states[:, 4],
            states[:, 5],
            states[:, 6],
            states[:, 7],
        )
        steer, a_x = actions[:, 0], actions[:, 1]
        psi_i, psi_b = disturb[:, 0], disturb[:, 1]
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
        
        # 计算侧偏角，侧向力
        alpha_f = (v + w * l_f) / u - steer
        Fyf = -k_f * alpha_f
        alpha_r = (v - l_r * w) / u
        Fyr = -k_r * alpha_r
        # 状态更新公式
        next_state = torch.empty_like(states)
        next_state[:, 0] = x + delta_t * (u * torch.cos(phi) - v * torch.sin(phi))
        next_state[:, 1] = y + delta_t * (u * torch.sin(phi) + v * torch.cos(phi))
        temp1 = ms * h_cg * \
                (ms * h_cg * u * w + ms * g * h_cg * psi - k_psi * (psi - psi_b) - C_psi * psi_dot) \
                / (m * (I_x + ms * h_cg ** 2))
        temp2 = 1 - ms ** 2 * h_cg ** 2 / (m * (I_x + ms * h_cg ** 2))
        v_dot = (-u * w + temp1 + 2 * (Fyf + Fyr) / m - g * psi_b) / temp2
        next_state[:, 3] = u + delta_t * (a_x + v * w - 2 * Fyf * steer / m - g * psi_i)
        next_state[:, 4] = v + delta_t * v_dot
        next_state[:, 2] = phi + delta_t * w
        next_state[:, 5] = w + delta_t * 2 * (l_f * Fyf - l_r * Fyr) / I_z
        next_state[:, 6] = psi + delta_t * psi_dot
        next_state[:, 7] = psi_dot + delta_t * ((ms * h_cg * (v_dot + u * w)
                                              + ms * g * h_cg * psi - k_psi * (psi - psi_b)
                                              - C_psi * psi_dot) / (I_x + ms * h_cg ** 2))
        next_state[:, 2] = angle_normalize(next_state[:, 2])
        return next_state

class BezierCurve:
    """二阶贝塞尔曲线类"""

    def __init__(self, p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor,device: Union[torch.device, str, None] = None,):
        self.device = device
        p0=p0.to(device=self.device)
        p1 = p1.to(device=self.device)
        p2 = p2.to(device=self.device)
        self.p0 = p0  # 起点
        self.p1 = p1  # 控制点
        self.p2 = p2  # 终点

    def compute_point(self, t: torch.Tensor) -> torch.Tensor:
        """计算贝塞尔曲线上的点
        Args:
            t: 参数t，范围[0,1]
        Returns:
            point: [x, y]坐标的tensor
        """
        t = t.to(device=self.device)
        return (1 - t) ** 2 * self.p0 + 2 * (1 - t) * t * self.p1 + t ** 2 * self.p2

    # def compute_point(self, t: torch.Tensor) -> torch.Tensor:
    #     """支持批量t输入"""
    #     t = t.unsqueeze(-1) if t.dim() == 0 else t
    #     return ((1 - t) ** 2 * self.p0 +
    #             2 * (1 - t) * t * self.p1 +
    #             t ** 2 * self.p2)
    def compute_derivative(self, t: torch.Tensor) -> torch.Tensor:
        """计算贝塞尔曲线在t处的导数"""
        return 2 * (1 - t) * (self.p1 - self.p0) + 2 * t * (self.p2 - self.p1)

class PythEnergybimodalplanning2aModel(PythBaseModel):
    def __init__(
        self,
        pre_horizon: int = 30,
        device: Union[torch.device, str, None] = None,
        path_para: Optional[Dict[str, Dict]] = None,
        u_para: Optional[Dict[str, Dict]] = None,
        slope_para: Optional[Dict[str, Dict]] = None,
        max_steer: float = np.pi / 6,
        static_obstacle_num: int = 10,
        d_pre: float = 20.0,  # 离障碍物多少远开始规划
        lateral_sample: float = 3.5,  # 横向采样距离
        forward_sample: float = 10.0,  # 纵向采样距离
        **kwargs,
    ):
        """
        you need to define parameters here
        """
        self.vehicle_dynamics = VehicleDynamicsModel()
        self.pre_horizon = pre_horizon
        ego_obs_dim = 8
        ref_obs_dim = 6
        super().__init__(
            obs_dim=ego_obs_dim + ref_obs_dim * pre_horizon+static_obstacle_num*4,
            action_dim=2,
            dt=0.05,
            action_lower_bound=[-max_steer, -3],
            action_upper_bound=[max_steer, 3],
            device=device,
        )
        self.ref_traj = MultiRefTrajModel(path_para, u_para)
        self.road_slope = MultiRoadSlopeModel(slope_para)
        self.static_obstacle_num = static_obstacle_num
        self.wheel_distance = self.vehicle_dynamics.vehicle_params["wheel_distance"]
        self.ground_clearance = self.vehicle_dynamics.vehicle_params["ground_clearance"]
        self.veh_width = self.vehicle_dynamics.vehicle_params["veh_width"]
        self.veh_length = self.vehicle_dynamics.vehicle_params["veh_length"]
        self.d_pre = d_pre
        self.lateral_sample = lateral_sample
        self.forward_sample = forward_sample
        self.best_curve = None
        self.obstacle = None
        self.last_curve_index = None

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        done: torch.Tensor,
        info: InfoDict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, InfoDict]:
        batch_size = obs.shape[0]
        state = info["state"]
        ref_points = info["ref_points"]
        path_num = info["path_num"]
        u_num = info["u_num"]
        slope_num = info["slope_num"]
        t = info["ref_time"]
        static_state = info["static_state"]
        generate_guide = info["generate_guide"]
        self.guide_start = info["guide_start"]
        self.guide_end = info["guide_end"]
        guide_time_interval = info["guide_time_interval"]
        t_start = info["t_start"]
        t_end = info["t_end"]
        best_curve_isnone = info["best_curve_isnone"]

        # 初始化处理标记
        if not hasattr(self, 'processed_obstacles'):
            self.processed_obstacles = [set() for _ in range(batch_size)]

        self.static_obss = []
        for i_static in range(static_state.size(1)):
            self.static_obss.append(
                StaticObstacle(
                    obs_id=static_state[:, i_static, 0],
                    x=static_state[:, i_static, 1],
                    y=static_state[:, i_static, 2],
                    phi=static_state[:, i_static, 3],
                    length=static_state[:, i_static, 4],
                    width=static_state[:, i_static, 5],
                    height=static_state[:, i_static, 6],
                    material=static_state[:, i_static, 7],
                )
            )

        reward = self.compute_reward(obs, action)
        disturb = ref_points[:, 1, 4:6]
        next_state = self.vehicle_dynamics.f_xu(state, action, disturb, self.dt)

        next_t = t + self.dt

        static_x_tf, static_y_tf, static_phi_tf = \
            ego_vehicle_coordinate_transform(
                state[:, 0], state[:, 1], state[:, 2],
                static_state[..., 1], static_state[..., 2], static_state[..., 3],
            )
        static_u_tf = torch.zeros_like(static_state[..., 1]) - state[:, 3].unsqueeze(1)
        next_static_obs = torch.stack((static_x_tf, static_y_tf, static_phi_tf, static_u_tf), 1).squeeze(2).reshape(
            (-1, self.static_obstacle_num * 4))

        next_ref_points = ref_points.clone()
        next_ref_points[:, :-1] = ref_points[:, 1:]

        # 生成新参考点（批量处理）
        next_ref_points, generate_guide, t_start, t_end, best_curve_isnone = self._generate_new_ref_point(
            batch_size, state, path_num, u_num, slope_num, next_t, t_start, t_end, generate_guide, next_ref_points,
            best_curve_isnone)

        next_ego_obs = self.get_obs(next_state, next_ref_points)
        next_obs = torch.cat((next_ego_obs, next_static_obs), dim=1)
        isdone = self.judge_done(next_obs)

        next_info = {}
        for key, value in info.items():
            next_info[key] = value.detach().clone()
        next_info.update({
            "state": next_state,
            "ref_points": next_ref_points,
            "path_num": path_num,
            "u_num": u_num,
            "ref_time": next_t,
            "static_state": static_state,
            "generate_guide": generate_guide,
            "guide_start": self.guide_start,
            "guide_end": self.guide_end,
            "guide_time_interval": guide_time_interval,
            "t_start": t_start,
            "t_end": t_end,
            "best_curve_isnone": best_curve_isnone
        })
        return next_obs, reward, isdone, next_info

    def get_obs(self, state, ref_points):
        ref_x_tf, ref_y_tf, ref_phi_tf = \
            ego_vehicle_coordinate_transform(
                state[:, 0], state[:, 1], state[:, 2],
                ref_points[..., 0], ref_points[..., 1], ref_points[..., 2],
            )
        ref_u_tf = ref_points[..., 3] - state[:, 3].unsqueeze(1)
        ego_obs = torch.concat((torch.stack(
            (ref_x_tf[:, 0], ref_y_tf[:, 0], ref_phi_tf[:, 0], ref_u_tf[:, 0]), dim=1),
            state[:, 4:]), dim=1)
        ref_obs = torch.stack((ref_x_tf, ref_y_tf, ref_phi_tf, ref_u_tf, ref_points[..., 4], ref_points[..., 5]), 2)[
            :, 1:].reshape(ego_obs.shape[0], -1)
        return torch.concat((ego_obs, ref_obs), 1)

    def compute_reward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        delta_x, delta_y, delta_phi, delta_u = obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3]
        w = obs[:, 5]
        steer, a_x = action[:, 0], action[:, 1]
        return -(
            0.04 * delta_x ** 2
            + 0.2 * delta_y ** 2
            + 0.02 * delta_phi ** 2
            + 0.25 * delta_u ** 2
            + 0.01 * w ** 2
            + 0.5 * steer ** 2
            + 0.01 * a_x ** 2
        )

    def judge_done(self, obs: torch.Tensor) -> torch.Tensor:
        delta_x, delta_y, delta_phi = obs[:, 0], obs[:, 1], obs[:, 2]
        done = (
            (torch.abs(delta_x) > 10)
            | (torch.abs(delta_y) > 10)
            | (torch.abs(delta_phi) > np.pi)
        )
        return done

    def _generate_new_ref_point(self, batch_size, state, path_num, u_num, slope_num, t, t_start, t_end, generate_guide, next_ref_points, best_curve_isnone):
        guide_mask = torch.tensor(generate_guide, dtype=torch.bool)
        # 预计算全局参考点
        global_refs = self._get_global_reference(
            torch.arange(batch_size),
            t + self.pre_horizon * self.dt,
            path_num,
            u_num
        )
        # 初始化guide_points为全局参考点
        guide_points = global_refs.clone()

        # 只处理需要引导的batch
        guide_indices = torch.where(guide_mask)[0]
        for b in guide_indices:
            current_pos = state[b, :2]
            if self.obstacle != None:
                if guide_mask[b, ] == True and best_curve_isnone[b, ]==0:
                    guide_point = self._get_global_reference(
                        torch.tensor([b]),
                        t[b] + self.pre_horizon * self.dt,
                        path_num[b],
                        u_num[b]
                    )

                elif guide_mask[b, ]==True and best_curve_isnone[b, ]==1:
                    guide_point = global_refs[b, :]
                if state[b, 0] > self.obstacle.x[b]:
                    # 标记障碍物为已处理
                    self.processed_obstacles[b].add(self.obstacle.obs_id[b].item())
                    guide_mask[b] = False
                    self.obstacle = None
            else:
                self.obstacle, self.generate_guide_b = self.is_generate_guide(current_pos, b)
                if self.generate_guide_b and self.obstacle != None:
                    guide_mask[b] = self.generate_guide_b
                    # 为当前batch生成轨迹
                    curves = self.generate_bezier_curves(state[b], self.obstacle, b)
                    self.best_curve, can_cross = self.can_cross_decision(self.obstacle, curves, b, state[b])
                    # 计算当前batch的时间参数
                    t_start_b = t[b]
                    t_start[b] = t_start_b
                    t_end_b = t[b] + (self.obstacle.x[b] + self.forward_sample+self.obstacle.width[b]/2 - state[b, 0]) / state[b, 3]
                    t_end[b] = t_end_b
                    # 更新当前batch的参考点
                    for i in range(1, self.pre_horizon + 1):
                        ref_x = self.ref_traj.compute_x(t[b] + i * self.dt, path_num[b], u_num[b])
                        if self.best_curve is None:
                            guide_point = self._get_global_reference(
                                torch.tensor([b]),
                                t[b]+i*self.dt,
                                path_num[b],
                                u_num[b]
                            )
                            best_curve_isnone[b] = 0
                        else:
                            # 获取引导点
                            guide_point = self.get_bezier_guide_point(
                                self.best_curve, t[b] + i * self.dt, state[b], t_start_b, t_end_b, path_num[b], u_num[b]
                            )
                            # 保证guide_point在x方向上和ref_x对齐
                            t_gap = (ref_x - guide_point[0]) / state[b, 3]
                            while guide_point[0] < ref_x and t_gap < t_end[b, ] and t_gap > 0:
                                guide_point = self.get_bezier_guide_point(
                                    self.best_curve, t[b]+t_gap, state[b], t_start[b], t_end[b], path_num[b], u_num[b]
                                )
                                t_gap += self.dt

                            if guide_point[0] >= self.obstacle.x[b] + self.forward_sample and guide_point[0] < ref_x:
                                if state[b, 0] > self.obstacle.x[b]:
                                    # 标记障碍物为已处理
                                    self.processed_obstacles[b].add(self.obstacle.obs_id[b].item())
                                    self.obstacle = None
                                    guide_mask[b] = False
                                self.best_curve = None
                                best_curve_isnone[b] = 0
                                guide_point = self._get_global_reference(
                                    torch.tensor([b]),
                                    t[b] + i * self.dt,
                                    path_num[b],
                                    u_num[b]
                                )
                        # next_ref_points[b, i, :4] = guide_point
                else:
                    guide_point = self._get_global_reference(
                        torch.tensor([b]),
                        t[b] + self.pre_horizon * self.dt,
                        path_num[b],
                        u_num[b]
                    )

                guide_points[b] = guide_point

        longislope = self.road_slope.compute_longislope(t + self.pre_horizon * self.dt, slope_num)
        latslope = self.road_slope.compute_latslope(t + self.pre_horizon * self.dt, slope_num)
        slope_points = torch.tensor([longislope, latslope]).reshape(1, -1)
        next_ref_points[:, -1] = torch.concatenate((guide_points, slope_points), dim=1,)
        return next_ref_points, guide_mask, t_start, t_end, best_curve_isnone

    def _get_global_reference(self, batch_indices: torch.Tensor, t: torch.Tensor,
                              path_num: torch.Tensor, u_num: torch.Tensor) -> torch.Tensor:
        """
        优化版批量获取全局参考轨迹点
        改进点：
        1. 支持部分batch索引
        2. 自动处理不同形状输入
        3. 添加数值稳定性检查
        """
        # 输入校验
        assert batch_indices.dim() == 1, "batch_indices应为1D Tensor"

        # 获取有效batch数量
        batch_size = batch_indices.shape[0]

        # 处理标量输入（兼容单样本情况）
        t = t.expand(batch_size) if t.numel() == 1 else t[batch_indices]
        path_num = path_num.expand(batch_size) if path_num.numel() == 1 else path_num[batch_indices]
        u_num = u_num.expand(batch_size) if u_num.numel() == 1 else u_num[batch_indices]

        # 批量计算各维度（避免重复调用compute_x等）
        ref_data = [
            self.ref_traj.compute_x(t, path_num, u_num),
            self.ref_traj.compute_y(t, path_num, u_num),
            self.ref_traj.compute_phi(t, path_num, u_num),
            self.ref_traj.compute_u(t, path_num, u_num)
        ]

        # 堆叠并确保形状正确 [B,4]
        ref_points = torch.stack(ref_data, dim=1)
        assert ref_points.shape == (batch_size, 4), f"形状错误: {ref_points.shape}"

        return ref_points
    def is_generate_guide(self, current_pos: torch.Tensor, batch_idx: int) -> Tuple[Optional[StaticObstacle], bool]:
        # 批量计算所有障碍物距离
        obs_positions = torch.stack([torch.stack([obs.x[batch_idx], obs.y[batch_idx]])
                                     for obs in self.static_obss])
        dists = torch.norm(current_pos - obs_positions, dim=1)

        # 筛选前方未处理的障碍物
        forward_mask = torch.tensor(obs_positions[:, 0] > current_pos[0], device=self.device)
        unprocessed_mask = torch.tensor([obs.obs_id[batch_idx] not in self.processed_obstacles[batch_idx]
                                         for obs in self.static_obss], device=self.device)
        valid_mask = torch.tensor(dists < self.d_pre,  device=self.device) & forward_mask & unprocessed_mask
        if not torch.any(valid_mask):
            return None, False

        # 找到最近的有效障碍物
        data = dists * valid_mask.float()
        min_idx = torch.argmin(torch.where(data == 0, torch.inf, data))
        return self.static_obss[min_idx], True

    def _is_too_close_to_obstacle(self, curve: BezierCurve, obstacle: StaticObstacle, batch_idx: int) -> bool:
        """检查曲线是否离障碍物太近"""
        # 采样曲线上的点
        ts = torch.linspace(0, 1, 5, device=self.device)
        points = torch.stack([curve.compute_point(t) for t in ts])

        # 计算到障碍物的距离
        obs_center = torch.tensor([obstacle.x[batch_idx], obstacle.y[batch_idx]],
                                  device=self.device)
        distances = torch.norm(points - obs_center, dim=1)

        # 安全距离阈值
        safe_distance = max(obstacle.width[batch_idx], obstacle.length[batch_idx]) * 1.2

        return torch.any(distances < safe_distance)

    def can_cross_decision(self, obstacle: StaticObstacle, curves: list, batch_idx: int, state: torch.Tensor) -> Tuple[
        Optional[BezierCurve], bool]:
        # 判断是否满足跨越条件
        if obstacle.material[batch_idx] == 0:
            can_cross = True
        elif obstacle.material[batch_idx] == 1:
            can_cross = (obstacle.height[batch_idx] < self.ground_clearance and
                         obstacle.width[batch_idx] < self.wheel_distance)
        # can_cross = False
        # 评估曲线平滑性 (Tensor版本)
        def evaluate_curve(curve: BezierCurve, is_crossing) -> torch.Tensor:
            # 采样曲线上的点 (使用Tensor操作)
            ts = torch.linspace(0, 1, 10, device=self.device)
            points = torch.stack([curve.compute_point(t) for t in ts])

            # 计算导数 (使用自动微分)
            dx = torch.gradient(points[:, 0])[0]
            dy = torch.gradient(points[:, 1])[0]
            ddx = torch.gradient(dx)[0]
            ddy = torch.gradient(dy)[0]

            # 1. 曲率评估
            curvature = torch.abs(dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** 1.5
            mean_curvature = torch.mean(curvature)
            max_curvature = torch.max(curvature)

            # 2. 横向动力学评估
            # 假设速度为v (可根据实际情况调整)
            v = state[3, ]
            # 横向加速度 (ay = v^2 * curvature)
            lateral_acc = v ** 2 * curvature
            max_lateral_acc = torch.max(lateral_acc)

            # 3. 向心加速度评估 (与横向加速度相同)
            centripetal_acc = lateral_acc

            # 4. 横向jerk评估 (加速度变化率)
            jerk = torch.gradient(lateral_acc)[0]
            max_jerk = torch.abs(jerk).max()

            # 5. 方向变化评估 (用于减少频繁切换)
            if hasattr(self, 'last_curve_index'):
                current_curve_index = curves.index(curve)
                is_switching = current_curve_index != self.last_curve_index
            else:
                is_switching = False

            # 6. 轨迹长度评估 (避免不必要的绕行)
            curve_length = torch.sum(torch.sqrt(torch.diff(points[:, 0]) ** 2 + torch.diff(points[:, 1]) ** 2))

            # # 计算曲率
            # denominator = (dx.pow(2) + dy.pow(2)).pow(1.5) + 1e-6  # 避免除零
            # curvature = (dx * ddy - dy * ddx).abs() / denominator

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
            # 可跨越时评估所有曲线
            # scored_curves = []
            # for curve in curves:
            #     score, metrics = evaluate_curve(curve, True)
            #     scored_curves.append((score, metrics, curve))
            #
            # # 按得分排序
            # scored_curves.sort(key=lambda x: x[0])
            # best_score, best_metrics, best_curve = scored_curves[0]
            best_curve = curves[1]
            # # 检查安全约束
            # if (best_metrics['max_lateral_acc'] > 2.5 or
            #         best_metrics['max_jerk'] > 1.0 or
            #         best_metrics['max_curvature'] > 1.5):
            #     can_cross = False

        if not can_cross:
            # 绕行时只考虑特定曲线
            valid_indices = [0, 2]
            valid_curves = [curves[0], curves[2]]

            scored_curves = []
            for curve in valid_curves:
                score, metrics = evaluate_curve(curve, False)
                scored_curves.append((score, metrics, curve))

            scored_curves.sort(key=lambda x: x[0])
            best_score, best_metrics, best_curve = scored_curves[0]

            # 记录当前选择的曲线索引
            current_index = valid_indices[valid_curves.index(best_curve)]

            # 检查是否需要切换轨迹
            if self.last_curve_index is not None:
                if current_index != self.last_curve_index:
                    # 如果切换轨迹，确保新轨迹明显更好
                    if len(scored_curves) > 1 and best_score > scored_curves[1][0] * 0.8:
                        best_curve = valid_curves[valid_indices.index(self.last_curve_index)]
                        current_index = self.last_curve_index

            self.last_curve_index = current_index

        return best_curve, can_cross

    def get_bezier_guide_point(self, curve, t_interpolate, state, t_start: torch.Tensor, t_end: torch.Tensor, path_num, u_num):
        # 确保时间参数在有效范围内
        if t_end <= t_start:
            t_end = t_start + self.dt
        # 计算归一化时间参数
        t_norm = torch.clamp((t_interpolate - t_start) / (t_end - t_start), 0, 1)
        point = curve.compute_point(t_norm)
        derivative = curve.compute_derivative(t_norm)
        # 确保导数不为零向量
        if torch.norm(derivative) < 1e-6 or len(derivative) < 2:
            derivative = torch.tensor([1e-6, 0], device=state.device)
        phi = torch.arctan2(derivative[1], derivative[0])

        # # 混合全局方向
        # global_phi = self.ref_traj.compute_phi(t_start + t_norm * (t_end - t_start), path_num, u_num)
        # blend_ratio = torch.clamp(t_norm, 0.2, 0.8)
        # phi = angle_normalize(blend_ratio * phi + (1 - blend_ratio) * global_phi)

        # 速度方向修正
        u = self.ref_traj.compute_u(t_start + t_norm * (t_end - t_start), path_num, u_num)
        if u < 0:
            phi += torch.pi
            u = abs(u)

        return torch.tensor([point[0], point[1], phi, u], device=state.device)

    def generate_bezier_curves(self, state:torch.Tensor, obstacle: StaticObstacle, batch_idx: int) -> list:
        """生成贝塞尔曲线(Tensor版本)
        Args:
            obstacle: 障碍物信息
        Returns:
            curves: 生成的贝塞尔曲线列表
        """
        # # 确保状态和采样参数为tensor
        state = torch.tensor(state[:2], dtype=torch.float32) \
            if not isinstance(state, torch.Tensor) else state[:2]
        lateral_sample = torch.tensor(self.lateral_sample, dtype=torch.float32) \
            if not isinstance(self.lateral_sample, torch.Tensor) else self.lateral_sample
        forward_sample = torch.tensor(self.forward_sample, dtype=torch.float32) \
            if not isinstance(self.forward_sample, torch.Tensor) else self.forward_sample
        curves = []

        # 横向采样点 - 使用tensor
        mid_points = torch.stack([
            torch.tensor([obstacle.x[batch_idx, ]+ (lateral_sample+obstacle.width[batch_idx, ]/2+self.veh_width/2)*torch.sin(obstacle.phi[batch_idx, ]),
                          obstacle.y[batch_idx, ]+ (- lateral_sample-obstacle.width[batch_idx,]/2-self.veh_width/2)*torch.cos(obstacle.phi[batch_idx, ])], dtype=torch.float32),  # 左侧
            torch.tensor([obstacle.x[batch_idx, ], obstacle.y[batch_idx, ]], dtype=torch.float32),  # 中心
            torch.tensor([obstacle.x[batch_idx, ]+ (- lateral_sample-obstacle.width[batch_idx,]/2-self.veh_width/2)*torch.sin(obstacle.phi[batch_idx, ]),
                          obstacle.y[batch_idx, ]+(lateral_sample+obstacle.width[batch_idx, ]/2+self.veh_width/2)*torch.cos(obstacle.phi[batch_idx, ])], dtype=torch.float32)  # 右侧
                                ])

        # 纵向采样点 - 使用tensor
        end_points = torch.stack([
            torch.tensor([obstacle.x[batch_idx,] + (forward_sample + obstacle.length[batch_idx,] / 2)*torch.cos(obstacle.phi[batch_idx,])+(lateral_sample+obstacle.width[batch_idx, ]/2+self.veh_width/2)*torch.sin(obstacle.phi[batch_idx, ]),
                          obstacle.y[batch_idx,]+(forward_sample+obstacle.length[batch_idx, ]/2)*torch.sin(obstacle.phi[batch_idx,])+(- lateral_sample-obstacle.width[batch_idx,]/2-self.veh_width/2)*torch.cos(obstacle.phi[batch_idx,])],dtype=torch.float32),
            torch.tensor([obstacle.x[batch_idx, ] + (forward_sample+obstacle.length[batch_idx, ]/2)*torch.cos(obstacle.phi[batch_idx,]),
                          obstacle.y[batch_idx, ]+(forward_sample+obstacle.length[batch_idx, ]/2)*torch.sin(obstacle.phi[batch_idx,])], dtype=torch.float32),
            torch.tensor([obstacle.x[batch_idx,] + forward_sample + obstacle.length[batch_idx,] / 2-(lateral_sample+obstacle.width[batch_idx, ]/2+self.veh_width/2)*torch.sin(obstacle.phi[batch_idx, ]),
                          obstacle.y[batch_idx,]+(forward_sample+obstacle.length[batch_idx, ]/2)*torch.sin(obstacle.phi[batch_idx,])+ (lateral_sample+obstacle.width[batch_idx, ]/2+self.veh_width/2)*torch.cos(obstacle.phi[batch_idx,])], dtype=torch.float32)
            ])

        # 生成贝塞尔曲线
        for i in range(mid_points.shape[0]):
            # 使用当前位置作为起点，障碍物位置作为控制点
            curve = BezierCurve(
                state,
                mid_points[i],
                end_points[i],
                self.device
            )
            curves.append(curve)
        return curves

def ego_vehicle_coordinate_transform(
    ego_x: torch.Tensor,
    ego_y: torch.Tensor,
    ego_phi: torch.Tensor,
    ref_x: torch.Tensor,
    ref_y: torch.Tensor,
    ref_phi: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ego_x, ego_y, ego_phi = ego_x.unsqueeze(1), ego_y.unsqueeze(1), ego_phi.unsqueeze(1)
    cos_tf = torch.cos(-ego_phi)
    sin_tf = torch.sin(-ego_phi)
    ref_x_tf = (ref_x - ego_x) * cos_tf - (ref_y - ego_y) * sin_tf
    ref_y_tf = (ref_x - ego_x) * sin_tf + (ref_y - ego_y) * cos_tf
    ref_phi_tf = angle_normalize(ref_phi - ego_phi)
    return ref_x_tf, ref_y_tf, ref_phi_tf


def env_model_creator(**kwargs):
    """
    make env model `pyth_veh4dofconti`
    """
    return PythEnergybimodalplanning2aModel(**kwargs)
