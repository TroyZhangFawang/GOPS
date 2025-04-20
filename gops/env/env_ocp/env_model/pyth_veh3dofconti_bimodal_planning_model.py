#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: vehicle 3DOF model environment with surrounding vehicles constraint
#  Update: 2024-11-01, Fawang Zhang: create environment

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from gops.env.env_ocp.env_model.pyth_veh3dofconti_model import (
    VehicleDynamicsModel,
    Veh3dofcontiModel,
    angle_normalize,
)
from gops.env.env_ocp.pyth_veh3dofconti_bimodal_planning import StaticObstacle
from gops.env.env_ocp.resources.ref_traj_model import MultiRefTrajModel
from gops.utils.gops_typing import InfoDict


@dataclass
class DynamicObstacleModel:
    # distance from front axle to rear axle
    l: float = 3.0
    dt: float = 0.1
    obs_length: float = 4.8
    obs_width: float = 2.0
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x, y, phi, u, delta = state.split(1, dim=-1)
        next_x = x + u * torch.cos(phi) * self.dt
        next_y = y + u * torch.sin(phi) * self.dt
        next_phi = phi + u * torch.tan(delta) / self.l * self.dt
        next_phi = angle_normalize(next_phi)
        return torch.cat((next_x, next_y, next_phi, u, delta), dim=-1)

class QuinticPolynomial:
    """五次多项式轨迹生成器 - Tensor版本"""

    def __init__(self, start_state, end_state, T):
        """
        初始化五次多项式曲线

        Args:
            start_state: 起点状态 [x, dx, ddx]，包含位置、速度、加速度
            end_state: 终点状态 [x, dx, ddx]，包含位置、速度、加速度
            T: 时间间隔
        """
        # 确保输入为tensor类型
        if not isinstance(start_state[0], torch.Tensor):
            start_state = [torch.tensor(x, dtype=torch.float32) for x in start_state]
        if not isinstance(end_state[0], torch.Tensor):
            end_state = [torch.tensor(x, dtype=torch.float32) for x in end_state]
        if not isinstance(T, torch.Tensor):
            T = torch.tensor(T, dtype=torch.float32)

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
        # 构建系数矩阵
        A = torch.tensor([
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 2, 0, 0],
            [self.T ** 5, self.T ** 4, self.T ** 3, self.T ** 2, self.T, 1],
            [5 * self.T ** 4, 4 * self.T ** 3, 3 * self.T ** 2, 2 * self.T, 1, 0],
            [20 * self.T ** 3, 12 * self.T ** 2, 6 * self.T, 2, 0, 0]
        ], dtype=torch.float32)

        # 构建等式右边的向量
        b = torch.tensor([
            self.xs, self.vxs, self.axs,
            self.xe, self.vxe, self.axe
        ], dtype=torch.float32)

        # 求解线性方程组
        return torch.linalg.solve(A, b)

    def compute_point(self, t):
        """
        计算时刻t的状态

        Args:
            t: 时间点，可以是标量或tensor

        Returns:
            tuple: (位置, 速度, 加速度), 如果t超出范围则返回(None, None, None)
        """
        # 确保t是tensor类型
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32)

        # # 检查时间是否在有效范围内
        # if t < 0 or t > self.T:
        #     return None, None, None

        # 计算位置
        pos = (self.coeffs[0] * t ** 5 + self.coeffs[1] * t ** 4 +
               self.coeffs[2] * t ** 3 + self.coeffs[3] * t ** 2 +
               self.coeffs[4] * t + self.coeffs[5])

        # 计算速度
        vel = (5 * self.coeffs[0] * t ** 4 + 4 * self.coeffs[1] * t ** 3 +
               3 * self.coeffs[2] * t ** 2 + 2 * self.coeffs[3] * t +
               self.coeffs[4])

        # 计算加速度
        acc = (20 * self.coeffs[0] * t ** 3 + 12 * self.coeffs[1] * t ** 2 +
               6 * self.coeffs[2] * t + 2 * self.coeffs[3])

        return pos, vel, acc

    def compute_trajectory(self, t_samples):
        """
        计算一系列时间点的轨迹

        Args:
            t_samples: 时间点tensor，shape=(n,)

        Returns:
            tuple: (positions, velocities, accelerations)，每个元素shape=(n,)
        """
        if not isinstance(t_samples, torch.Tensor):
            t_samples = torch.tensor(t_samples, dtype=torch.float32)

        # 过滤掉无效的时间点
        mask = (t_samples >= 0) & (t_samples <= self.T)
        valid_t = t_samples[mask]

        if len(valid_t) == 0:
            return None, None, None

        # 计算位置矩阵
        positions = (self.coeffs[0] * valid_t.pow(5).unsqueeze(1) +
                     self.coeffs[1] * valid_t.pow(4).unsqueeze(1) +
                     self.coeffs[2] * valid_t.pow(3).unsqueeze(1) +
                     self.coeffs[3] * valid_t.pow(2).unsqueeze(1) +
                     self.coeffs[4] * valid_t.unsqueeze(1) +
                     self.coeffs[5])

        # 计算速度矩阵
        velocities = (5 * self.coeffs[0] * valid_t.pow(4).unsqueeze(1) +
                      4 * self.coeffs[1] * valid_t.pow(3).unsqueeze(1) +
                      3 * self.coeffs[2] * valid_t.pow(2).unsqueeze(1) +
                      2 * self.coeffs[3] * valid_t.unsqueeze(1) +
                      self.coeffs[4])

        # 计算加速度矩阵
        accelerations = (20 * self.coeffs[0] * valid_t.pow(3).unsqueeze(1) +
                         12 * self.coeffs[1] * valid_t.pow(2).unsqueeze(1) +
                         6 * self.coeffs[2] * valid_t.unsqueeze(1) +
                         2 * self.coeffs[3])

        return positions, velocities, accelerations

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

class Veh3dofBimodalPlanningModel(Veh3dofcontiModel):
    def __init__(
        self,
        pre_horizon: int,
        device: Union[torch.device, str, None] = None,
        path_para: Optional[Dict[str, Dict]] = None,
        u_para: Optional[Dict[str, Dict]] = None,
        dynamic_obstacle_num: int = 2,
        dynamic_length: float = 4.8,
        dynamic_width: float = 2.0,
        static_obstacle_num: int = 3,
        d_pre: float = 20.0,  # 离障碍物多少远开始规划
        lateral_sample: float = 3.5,  # 横向采样距离
        forward_sample: float = 10.0,  # 纵向采样距离
        **kwargs: Any,
    ):
        self.state_dim = 6
        self.ego_obs_dim = 6
        self.ref_obs_dim = 4
        super(Veh3dofcontiModel, self).__init__(
            obs_dim=self.ego_obs_dim + self.ref_obs_dim * pre_horizon + (dynamic_obstacle_num+static_obstacle_num) * 4,
            action_dim=2,
            dt=0.1,
            action_lower_bound=[-np.pi / 6, -3],
            action_upper_bound=[np.pi / 6, 3],
            device=device,
        )

        self.vehicle_dynamics = VehicleDynamicsModel()
        self.dynamic_obs_model = DynamicObstacleModel()
        self.ref_traj = MultiRefTrajModel(path_para, u_para)
        self.pre_horizon = pre_horizon
        self.dynamic_obstacle_num = dynamic_obstacle_num
        self.static_obstacle_num = static_obstacle_num
        self.dynamic_length = dynamic_length
        self.dynamic_width = dynamic_width
        self.wheel_distance = self.vehicle_dynamics.vehicle_params["wheel_distance"]
        self.ground_clearance = self.vehicle_dynamics.vehicle_params["ground_clearance"]
        self.veh_width = self.vehicle_dynamics.vehicle_params["veh_width"]
        self.veh_length = self.vehicle_dynamics.vehicle_params["veh_length"]
        self.d_pre = d_pre
        self.lateral_sample = lateral_sample
        self.forward_sample = forward_sample
        self.best_curve = None
        self.obstacle = None

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
        t = info["ref_time"]
        dynamic_state = info["dynamic_state"]
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
                         )
                                )
        reward = self.compute_reward(obs, action, info)
        next_state = self.vehicle_dynamics.f_xu(state, action, self.dt)
        next_t = t + self.dt

        next_dynamic_state = self.dynamic_obs_model.forward(dynamic_state)
        dynamic_x_tf, dynamic_y_tf, dynamic_phi_tf = \
            ego_vehicle_coordinate_transform(
                state[:, 0], state[:, 1], state[:, 2],
                next_dynamic_state[..., 0], next_dynamic_state[..., 1], next_dynamic_state[..., 2],
            )
        dynamic_u_tf = next_dynamic_state[..., 3] - state[:, 3].unsqueeze(1)
        next_dynamic_obs = torch.stack((dynamic_x_tf, dynamic_y_tf, dynamic_phi_tf, dynamic_u_tf), 1).squeeze(2).reshape((-1, self.dynamic_obstacle_num * 4))

        static_x_tf, static_y_tf, static_phi_tf = \
            ego_vehicle_coordinate_transform(
                state[:, 0], state[:, 1], state[:, 2],
                static_state[..., 1], static_state[..., 2], static_state[..., 3],
            )
        static_u_tf = torch.zeros_like(static_state[..., 1]) - state[:, 3].unsqueeze(1)
        next_static_obs = torch.stack((static_x_tf, static_y_tf, static_phi_tf, static_u_tf), 1).squeeze(2).reshape((-1, self.static_obstacle_num * 4))

        next_ref_points = ref_points.clone()  # [batch_size,N,4]
        next_ref_points[:, :-1] = ref_points[:, 1:]
        # 生成新参考点（批量处理）
        next_ref_points, generate_guide, t_start, t_end, best_curve_isnone = self._generate_new_ref_point(
            batch_size, state, path_num, u_num, next_t, t_start, t_end, generate_guide, next_ref_points, best_curve_isnone)

        # next_ref_points[:, -1] = new_ref_point
        next_ego_obs = self.get_obs(next_state, next_ref_points)
        next_obs = torch.cat((next_ego_obs, next_dynamic_obs, next_static_obs), dim=1)
        next_info = {}
        for key, value in info.items():
            next_info[key] = value.detach().clone()

        next_info.update({
            "state": next_state,
            "ref_points": next_ref_points,
            "path_num": path_num,
            "u_num": u_num,
            "ref_time": next_t,
            "constraint": self.get_constraint(next_obs, next_info),
            "dynamic_state": next_dynamic_state,
            "static_state": static_state,
            "generate_guide": generate_guide,
            "guide_start": self.guide_start,
            "guide_end": self.guide_end,
            "guide_time_interval": guide_time_interval,
            "t_start": t_start,
            "t_end": t_end,
            "best_curve_isnone":best_curve_isnone
        })

        next_done = self.judge_done(next_obs, next_info)
        return next_obs, reward, next_done, next_info

    def _generate_new_ref_point(self, batch_size, state, path_num, u_num, t, t_start, t_end, generate_guide, next_ref_points, best_curve_isnone):
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
                if guide_mask[b, ]==True and best_curve_isnone[b, ]==0:
                    guide_point = self._get_global_reference(
                        torch.tensor([b]),
                        t[b] + self.pre_horizon * self.dt,
                        path_num[b],
                        u_num[b]
                    )
                    # if self.generate_guide_b and self.obstacle != None:
                    #     guide_mask[b] = self.generate_guide_b
                    #     curves = self.generate_bezier_curves(state[b], self.obstacle, b)
                    #     self.best_curve, can_cross = self.can_cross_decision(self.obstacle, curves, b, state[b])
                    #     ref_x = global_refs[b, 0]
                    #     t_start_b = t[b]
                    #     t_start[b] = t_start_b
                    #     t_end_b = t[b] + (
                    #                 self.obstacle.x[b] + self.forward_sample + self.obstacle.width[b] / 2 - state[b, 0]) / \
                    #               state[b, 3]
                    #     t_end[b] = t_end_b
                    #     guide_point = self.get_bezier_guide_point(
                    #         self.best_curve, t[b] + self.pre_horizon * self.dt, state[b], t_start_b, t_end_b, path_num[b], u_num[b]
                    #     )
                    #     # 保证guide_point在x方向上和ref_x对齐
                    #     t_gap = (ref_x - guide_point[0]) / state[b, 3]
                    #     while guide_point[0] < ref_x and t_gap < t_end[b,] and t_gap > 0:
                    #         guide_point = self.get_bezier_guide_point(
                    #             self.best_curve, t[b]+t_gap, state[b], t_start[b], t_end[b], path_num[b], u_num[b]
                    #         )
                    #         t_gap += self.dt
                    #
                    #     if guide_point[0] >= self.obstacle.x[b] + self.forward_sample and guide_point[0] < ref_x:
                    #         if state[b, 0] > self.obstacle.x[b]:
                    #             # 标记障碍物为已处理
                    #             self.processed_obstacles[b].add(self.obstacle.obs_id[b].item())
                    #             self.obstacle = None
                    #             guide_mask[b] = False
                    #         self.best_curve = None
                    #         best_curve_isnone[b] = 0
                    #         guide_point = self._get_global_reference(
                    #             torch.tensor([b]),
                    #             t[b] + self.pre_horizon * self.dt,
                    #             path_num[b],
                    #             u_num[b]
                    #         )

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
                    # if self.best_curve != None:
                    #     # 获取引导点
                    #     # ref_x = global_refs[b, 0]
                    #     guide_point = self.get_bezier_guide_point(
                    #         self.best_curve, t[b]+self.pre_horizon*self.dt, state[b], t_start[b], t_end[b], path_num[b], u_num[b]
                    #     )
                    #     # 检查是否需要切换回全局参考轨迹
                    #     # t_gap = (ref_x - guide_point[0]) / state[b, 3]
                    #     # if t_gap > t_end[b] and t_gap > 0:
                    #     #     guide_point = self._get_global_reference(
                    #     #         torch.tensor([b]),
                    #     #         t[b] + self.pre_horizon * self.dt,
                    #     #         path_num[b],
                    #     #         u_num[b]
                    #     #     )
                    #     # else:
                    #     #     while guide_point[0] <= ref_x and t_gap <= t_end[b] and t_gap > 0:
                    #     #         guide_point = self.get_bezier_guide_point(
                    #     #             self.best_curve, t_gap + self.pre_horizon * self.dt, state[b], t_start[b], t_end[b], path_num[b], u_num[b]
                    #     #         )
                    #     #         t_gap += self.dt
                    #     #     if guide_point[0] >= obstacle.x[b] + self.forward_sample:
                    #     #         if state[b, 0] > obstacle.x[b] + obstacle.length[b] / 2:
                    #     #             # 标记障碍物为已处理
                    #     #             self.processed_obstacles[b].add(obstacle.obs_id[b].item())
                    #     #             guide_mask[b] = False
                    #     #             self.best_curve = None
                    #     #         # 使用全局参考轨迹
                    #     #         guide_point = self._get_global_reference(
                    #     #             torch.tensor([b]),
                    #     #             t[b] + self.pre_horizon * self.dt,
                    #     #             path_num[b],
                    #     #             u_num[b]
                    #     #         )
                    # else:
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
                        next_ref_points[b, i] = guide_point
                else:
                    guide_point = self._get_global_reference(
                        torch.tensor([b]),
                        t[b] + self.pre_horizon * self.dt,
                        path_num[b],
                        u_num[b]
                    )

                guide_points[b] = guide_point

        # # 处理非引导轨迹的batch
        # if torch.any(~guide_mask):
        #     new_ref_points[~guide_mask, :-1] = ref_points[~guide_mask, 1:]
        #     non_guide_indices = torch.where(~guide_mask)[0]
        #     global_refs = self._get_global_reference(
        #         non_guide_indices,
        #         t + self.pre_horizon * self.dt,

        #         path_num,
        #         u_num
        #     )
        #     new_ref_points[non_guide_indices, -1] = global_refs
        next_ref_points[:, -1] = guide_points
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

    def compute_reward(
            self,
            obs: torch.Tensor,
            action: torch.Tensor,
            info: InfoDict,
    ) -> torch.Tensor:
        delta_x, delta_y, delta_phi, delta_u = obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3]
        v, w = obs[:, 4], obs[:, 5]
        steer, a_x = action[:, 0], action[:, 1]
        dis = torch.min(- self.get_constraint(obs, info), dim=1)[0]# sum or min?
        collision_bound = 1.0
        dis_to_tanh = torch.maximum(8 - 8 * dis / collision_bound, torch.zeros_like(dis))
        punish_dis = torch.tanh(dis_to_tanh - 4) + 1

        return -(
                1.0 * delta_x ** 2
                + 1.0 * delta_y ** 2
                + 0.1 * delta_phi ** 2
                + 0.1 * delta_u ** 2
                + 0.5 * v ** 2
                + 0.5 * w ** 2
                + 0.5 * steer ** 2
                + 0.5 * a_x ** 2
                + 20.0 * punish_dis.squeeze()
        )


    def get_constraint(self, obs: torch.Tensor, info: InfoDict) -> torch.Tensor:
        # collision detection using bicircle model
        # distance from vehicle center to front/rear circle center
        d = (self.dynamic_length - self.dynamic_width) / 2
        # circle radius
        r = np.sqrt(2) / 2 * self.dynamic_width

        x, y, phi = info["state"][:, :3].split(1, dim=1)
        ego_center = torch.stack(
            (
                torch.cat((x + d * torch.cos(phi), y + d * torch.sin(phi)), dim=1),
                torch.cat((x - d * torch.cos(phi), y - d * torch.sin(phi)), dim=1),
            ),
            dim=1,
        )

        dynamic_x, dynamic_y, dynamic_phi = info["dynamic_state"][..., :3].split(1, dim=2)
        dynamic_center = torch.stack(
            (
                torch.cat(
                    (
                        (dynamic_x + d * torch.cos(dynamic_phi)),
                        dynamic_y + d * torch.sin(dynamic_phi),
                    ),
                    dim=2,
                ),
                torch.cat(
                    (
                        (dynamic_x - d * torch.cos(dynamic_phi)),
                        dynamic_y - d * torch.sin(dynamic_phi),
                    ),
                    dim=2,
                ),
            ),
            dim=2,
        )
        
        min_dist = np.finfo(np.float32).max * torch.ones_like(dynamic_x).squeeze(-1)

        for i in range(2):
            # front and rear circle of ego vehicle
            for j in range(2):
                # front and rear circle of surrounding vehicles
                dist = torch.linalg.norm(
                    ego_center[:, i].unsqueeze(1) - dynamic_center[..., j, :], dim=2
                )

                min_dist = torch.minimum(
                    min_dist, dist
                )
        return 2 * r - min_dist

    def judge_done(self, obs: torch.Tensor, info: InfoDict) -> torch.Tensor:
        # delta_x, delta_y, delta_phi = obs[:, 0], obs[:, 1], obs[:, 2]
        # dis = - self.get_constraint(obs, info)
        # done = (
        #     (torch.abs(delta_x) > 5)
        #     | (torch.abs(delta_y) > 2)
        #     | (torch.abs(delta_phi) > np.pi)
        #     # | (torch.any(dis < 0., dim=1))
        # )
        done = torch.zeros(obs.shape[0], device=self.device).bool()
        return done

    # def is_generate_guide(self, current_pos: torch.Tensor, batch_idx: int) -> Tuple[Optional[StaticObstacle], bool]:
    #     min_dist = torch.tensor(float('inf'), device=current_pos.device)
    #     nearest_obstacle = None
    #
    #     for obs in self.static_obss:
    #         obs_pos = torch.stack([obs.x[batch_idx], obs.y[batch_idx]])
    #         dist = torch.norm(current_pos - obs_pos)
    #
    #         # 只考虑前方障碍物且未处理的
    #         if (obs_pos[0] > current_pos[0] and
    #                 dist < self.d_pre and
    #                 obs.obs_id[batch_idx] not in self.processed_obstacles[batch_idx]):
    #             if dist < min_dist:
    #                 min_dist = dist
    #                 nearest_obstacle = obs
    #
    #     return nearest_obstacle, min_dist < self.d_pre
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

    # def generate_bezier_curves(self, state: torch.Tensor, obstacle: StaticObstacle, batch_idx: int) -> list:
    #     """生成更平滑的贝塞尔曲线"""
    #     # 获取车辆当前位置和方向
    #     current_pos = state[:2]
    #     current_heading = state[2]
    #
    #     # 计算障碍物边界
    #     obs_left = obstacle.y[batch_idx] - obstacle.width[batch_idx] / 2 - self.veh_width / 2 - 0.5  # 额外安全距离
    #     obs_right = obstacle.y[batch_idx] + obstacle.width[batch_idx] / 2 + self.veh_width / 2 + 0.5
    #
    #     # 定义终点 (左右两侧)
    #     end_x = obstacle.x[batch_idx] + self.forward_sample + obstacle.length[batch_idx] / 2
    #     left_end = torch.tensor([end_x, obstacle.y[batch_idx]], device=self.device)
    #     right_end = torch.tensor([end_x, obstacle.y[batch_idx]], device=self.device)
    #
    #     # 计算中间控制点 (确保曲线平滑)
    #     mid_x = obstacle.x[batch_idx]#(current_pos[0] + end_x) / 2
    #     left_mid = torch.tensor([mid_x, obs_left - self.lateral_sample], device=self.device)
    #     right_mid = torch.tensor([mid_x, obs_right + self.lateral_sample], device=self.device)
    #
    #     # 生成三条曲线: 左绕、跨越(直线)、右绕
    #     curves = [
    #         # 左绕曲线
    #         BezierCurve(
    #             current_pos,
    #             left_mid,
    #             left_end
    #         ),
    #         # 跨越曲线 (直线)
    #         BezierCurve(
    #             current_pos,
    #             torch.tensor([mid_x, obstacle.y[batch_idx]], device=self.device),
    #             torch.tensor([end_x, obstacle.y[batch_idx]], device=self.device)
    #         ),
    #         # 右绕曲线
    #         BezierCurve(
    #             current_pos,
    #             right_mid,
    #             right_end
    #         )
    #     ]
    #
    #     return curves

    # def can_cross_decision(self, obstacle: StaticObstacle, curves: list, batch_idx: int, state: torch.Tensor) -> Tuple[
    #     Optional[BezierCurve], bool]:
    #     # 判断是否满足跨越条件
    #     can_cross = (obstacle.height[batch_idx] < self.ground_clearance and
    #                  obstacle.width[batch_idx] < self.wheel_distance)
    #
    #     # 如果不可以跨越，则选择最安全的绕行路线
    #     if not can_cross:
    #         # 计算车辆到左右绕行路线的初始转向角度
    #         current_heading = state[2]
    #
    #         # 左绕曲线初始转向角度
    #         left_init_angle = torch.atan2(curves[0].p1[1] - curves[0].p0[1],
    #                                       curves[0].p1[0] - curves[0].p0[0])
    #         left_angle_diff = torch.abs(angle_normalize(left_init_angle - current_heading))
    #
    #         # 右绕曲线初始转向角度
    #         right_init_angle = torch.atan2(curves[2].p1[1] - curves[2].p0[1],
    #                                        curves[2].p1[0] - curves[2].p0[0])
    #         right_angle_diff = torch.abs(angle_normalize(right_init_angle - current_heading))
    #
    #         # 选择转向角度变化较小的路线
    #         if left_angle_diff < right_angle_diff:
    #             return curves[0], False  # 左绕
    #         else:
    #             return curves[2], False  # 右绕
    #     else:
    #         # 可以跨越时选择直线
    #         return curves[1], True

    # def get_bezier_guide_point(self, curve, t_norm, state, t_start: torch.Tensor, t_end: torch.Tensor, path_num, u_num):
    #     # 确保时间参数在有效范围内
    #     t_norm = torch.clamp(t_norm, 0, 1)
    #     point = curve.compute_point(t_norm)
    #
    #     # 计算导数时添加平滑处理
    #     t_samples = torch.linspace(max(0, t_norm - 0.1), min(1, t_norm + 0.1), 5, device=self.device)
    #     points = torch.stack([curve.compute_point(t) for t in t_samples])
    #
    #     # 使用线性回归计算方向，避免突变
    #     x = points[:, 0].unsqueeze(1)
    #     y = points[:, 1]
    #     A = torch.cat([x, torch.ones_like(x)], dim=1)
    #     coeffs = torch.linalg.lstsq(A, y).solution
    #     phi = torch.atan(coeffs[0])  # 使用线性拟合的角度
    #
    #     # 混合全局方向
    #     global_phi = self.ref_traj.compute_phi(
    #         t_start + t_norm * (t_end - t_start), path_num, u_num
    #     )
    #     blend_ratio = torch.clamp(t_norm, 0.2, 0.8)
    #     phi = angle_normalize(blend_ratio * phi + (1 - blend_ratio) * global_phi)
    #
    #     # 速度方向修正
    #     u = self.ref_traj.compute_u(t_start + t_norm * (t_end - t_start), path_num, u_num)
    #
    #     return torch.tensor([point[0], point[1], phi, u], device=state.device)

    def get_quintic_guide_point(self, guide_traj, current_t: float, t_start: float) -> torch.Tensor:
        """
        获取指定轨迹的预测点
        """
        poly_x, poly_y = guide_traj
        local_t = current_t - t_start
        # 从五次多项式获取点
        x_state = poly_x.compute_point(local_t)
        y_state = poly_y.compute_point(local_t)

        x, y = x_state[0], y_state[0]
        vx, vy = x_state[1], y_state[1]
        phi = torch.atan2(vy, vx)

        return torch.stack([x, y, phi, vx], dim=1)

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

    def generate_quintic_curves(self, state:torch.Tensor,obstacle: StaticObstacle) -> list:
        def _create_single_trajectory(start_point, end_point, T) -> Tuple[QuinticPolynomial, QuinticPolynomial]:
            # x方向多项式
            poly_x = QuinticPolynomial(
                [start_point[:, 0].item(), start_point[:, 3].item() * torch.cos(start_point[:, 2]).item(), 0],
                [end_point[:, 0].item(), end_point[:, 3].item() * torch.cos(end_point[:, 2]).item(), 0],
                T
            )
            # y方向多项式
            poly_y = QuinticPolynomial(
                [start_point[:, 1].item(), start_point[:, 3].item() * torch.sin(start_point[:, 2]).item(), 0],
                [end_point[:, 1].item(), end_point[:, 3].item() * torch.sin(end_point[:, 2]).item(), 0],
                T
            )
            return (poly_x, poly_y)
        guide_start = state[:, :4]
        # 横向采样点
        guide_end = torch.stack(
            (torch.tensor([obstacle.x, obstacle.y - self.lateral_sample, obstacle.phi, state[:, 3].item()]),  # 左侧
             torch.tensor([obstacle.x, obstacle.y, obstacle.phi, state[:, 3]]),  # 中心
             torch.tensor([obstacle.x, obstacle.y + self.lateral_sample, obstacle.phi, state[:, 3].item()])),  # 右侧
            dim=1
        ).reshape((-1, 3, 4))
        guide_time_interval = abs((obstacle.x - state[:, 0]) / state[:, 3])
        curves = []
        # 生成五次多项式曲线
        for i in range(3):
            curve = _create_single_trajectory(
                guide_start,
                guide_end[:, i, :].reshape(-1, 4),
                guide_time_interval
            )
            curves.append(curve)
        return curves

    # def can_cross_decision(self, obstacle: StaticObstacle, curves: list, batch_idx: int) -> Tuple[Optional[BezierCurve], bool]:
    #     can_cross = False
    #
    #     if (obstacle.height[batch_idx, ] < self.ground_clearance and
    #             obstacle.width[batch_idx, ] < self.wheel_distance):
    #         can_cross = True
    #     self.curve_index = None
    #     if can_cross:
    #         self.curve_index = 1
    #         best_curve = curves[self.curve_index]  # 可跨就选跨
    #     else:
    #         self.curve_index = 2
    #         best_curve = curves[self.curve_index]  # 不可跨就选择左绕
    #     return best_curve, can_cross

    def can_cross_decision(self, obstacle: StaticObstacle, curves: list, batch_idx: int, state: torch.Tensor) -> Tuple[
        Optional[BezierCurve], bool]:
        # 判断是否满足跨越条件
        can_cross = (obstacle.height[batch_idx] < self.ground_clearance and
                     obstacle.width[batch_idx] < self.wheel_distance)

        # 评估曲线平滑性 (Tensor版本)
        def evaluate_curve(curve: BezierCurve) -> torch.Tensor:
            # 采样曲线上的点 (使用Tensor操作)
            ts = torch.linspace(0, 1, 10, device=self.device)
            points = torch.stack([curve.compute_point(t) for t in ts])

            # 计算导数 (使用自动微分)
            dx = torch.gradient(points[:, 0])[0]
            dy = torch.gradient(points[:, 1])[0]
            ddx = torch.gradient(dx)[0]
            ddy = torch.gradient(dy)[0]

            # 计算曲率
            denominator = (dx.pow(2) + dy.pow(2)).pow(1.5) + 1e-6  # 避免除零
            curvature = (dx * ddy - dy * ddx).abs() / denominator

            return curvature.mean()  # 返回平均曲率

        if can_cross:
            # 可跨越时，选择最平滑的曲线
            curve_scores = torch.tensor([evaluate_curve(curve) for curve in curves],
                                        device=self.device)
            best_idx = torch.argmin(curve_scores)
            best_curve = curves[best_idx]
            self.curve_index = best_idx
        else:
            # 不可跨越时，从左右两条曲线选择与当前航向最一致的绕行曲线
            current_heading = state[2]
            heading_diffs = []
            valid_indices = [0, 2]  # 对应的原始索引
            valid_curves = [curves[0], curves[2]]
            for curve in valid_curves:
                # 计算曲线终点方向
                end_heading = torch.atan2(curve.p2[1] - curve.p1[1],
                                          curve.p2[0] - curve.p1[0])
                # 计算角度差异
                diff = (end_heading - current_heading).abs()
                diff = torch.min(diff, 2 * np.pi - diff)  # 取最小角度差
                heading_diffs.append(diff)

            # 选择差异最小的曲线
            heading_diffs = torch.stack(heading_diffs)
            best_idx = torch.argmin(heading_diffs)
            self.curve_index = valid_indices[best_idx]  # 使用原始索引
            best_curve = valid_curves[best_idx]#

            # # # 额外检查：如果选择的曲线与障碍物太近，选择另一条
            # if self._is_too_close_to_obstacle(best_curve, obstacle, batch_idx):
            #     # 排除当前选择，选次优的
            #     mask = torch.ones_like(heading_diffs, dtype=torch.bool)
            #     mask[best_idx] = False
            #     remaining_diffs = heading_diffs[mask]
            #     if len(remaining_diffs) > 0:
            #         second_best_idx = torch.argmin(remaining_diffs)
            #         # 转换为原始索引
            #         original_indices = torch.arange(len(curves), device=self.device)[mask]
            #         best_idx = original_indices[second_best_idx]
            #         # 如果首选曲线不安全，选择另一条
            #         self.curve_index = valid_indices[1 - best_idx]
            #         best_curve = valid_curves[1 - best_idx]

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
            torch.tensor([obstacle.x[batch_idx, ], obstacle.y[batch_idx, ] - lateral_sample-obstacle.width[batch_idx,]/2-self.veh_width/2], dtype=torch.float32),  # 左侧
            torch.tensor([obstacle.x[batch_idx, ], obstacle.y[batch_idx, ]], dtype=torch.float32),  # 中心
            torch.tensor([obstacle.x[batch_idx, ], obstacle.y[batch_idx, ] + lateral_sample+obstacle.width[batch_idx, ]/2+self.veh_width/2], dtype=torch.float32)  # 右侧
        ])

        # 纵向采样点 - 使用tensor
        end_points = torch.stack([
            torch.tensor([obstacle.x[batch_idx,] + forward_sample + obstacle.length[batch_idx,] / 2, obstacle.y[batch_idx,]- lateral_sample-obstacle.width[batch_idx,]/2-self.veh_width/2],
                dtype=torch.float32),
            torch.tensor([obstacle.x[batch_idx, ] + forward_sample+obstacle.length[batch_idx, ]/2, obstacle.y[batch_idx, ]], dtype=torch.float32),
            torch.tensor([obstacle.x[batch_idx,] + forward_sample + obstacle.length[batch_idx,] / 2, obstacle.y[batch_idx,]+ lateral_sample+obstacle.width[batch_idx, ]/2+self.veh_width/2],
                dtype=torch.float32)
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
    return Veh3dofBimodalPlanningModel(**kwargs)


def test_batch_processing():
    model = Veh3dofBimodalPlanningModel(pre_horizon=10)
    batch_size = 32
    obs = torch.randn(batch_size, model.obs_dim)
    action = torch.randn(batch_size, 2)
    done = torch.zeros(batch_size, dtype=torch.bool)
    info = {
        "state": torch.randn(batch_size, 6),
        "ref_points": torch.randn(batch_size, 10, 4),
        "dynamic_state": torch.randn(batch_size, 1, 5),
        "static_state": torch.randn(batch_size, 2, 7)
    }

    with torch.no_grad():
        next_obs, reward, next_done, next_info = model(obs, action, done, info)

    assert next_obs.shape == (batch_size, model.obs_dim)
    assert reward.shape == (batch_size,)
    assert next_done.shape == (batch_size,)
    print("Batch processing test passed!")