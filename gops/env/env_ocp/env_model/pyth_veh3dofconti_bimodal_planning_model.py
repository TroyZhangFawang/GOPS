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

    def __init__(self, p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor):
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
        return (1 - t) ** 2 * self.p0 + 2 * (1 - t) * t * self.p1 + t ** 2 * self.p2

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
        dynamic_obstacle_num: int = 1,
        dynamic_length: float = 4.8,
        dynamic_width: float = 2.0,
        static_obstacle_num: int = 2,
        d_pre: float = 20.0,  # 离障碍物多少远开始规划
        lateral_sample: float = 5.0,  # 横向采样距离
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
        self.d_pre = d_pre
        self.lateral_sample = lateral_sample
        self.forward_sample = forward_sample
        self.guide_traj = None
        self.obstacle = None
    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        done: torch.Tensor,
        info: InfoDict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, InfoDict]:
        state = info["state"]
        ref_points = info["ref_points"]
        path_num = info["path_num"]
        u_num = info["u_num"]
        t = info["ref_time"]
        dynamic_state = info["dynamic_state"]
        static_state = info["static_state"]
        generate_guide = info["generate_guide"]
        guide_start = info["guide_start"]
        guide_end = info["guide_end"]
        guide_time_interval = info["guide_time_interval"]
        t_start = info["t_start"]
        t_end = info["t_end"]
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

        next_ref_points = ref_points.clone() # [batch_size,N,4]
        next_ref_points[:, :-1] = ref_points[:, 1:]
        # 判断是否需要生成引导轨迹，生成参考轨迹点
        # if torch.any(generate_guide==1) and self.guide_traj == None:
        #     indices = generate_guide.nonzero().squeeze(-1)
        #     # mask = generate_guide.unsqueeze(-1).expand_as(state)
        #     # state = state*mask
        #     for index in indices:
        #         obstacle, generate_guide_int = self.is_generate_guide(state[index, :])
        #         if generate_guide_int and obstacle != None:
        #             self.obstacle = obstacle
        #             # curves = self.generate_quintic_curves(guide_start, guide_end, guide_time_interval)
        #             curves = self.generate_bezier_curves(state[index, :2], obstacle)
        #             self.guide_traj, can_cross = self.can_cross_decision(obstacle, curves)
        #             # new_ref_point = self.get_quintic_guide_points(guide_traj, next_t + self.pre_horizon * self.dt, t_start)
        #             new_ref_point = self.get_bezier_guide_points(self.guide_traj, next_t + self.pre_horizon * self.dt,
        #                                                          t_start, t_end, path_num, u_num)
        #             ref_x = self.ref_traj.compute_x(next_t + self.pre_horizon * self.dt, path_num, u_num)
        #             t = (ref_x - new_ref_point[index, 0]) / state[index, 3]
        #             while new_ref_point[index, 0] <= ref_x and t < t_end and t > 0:
        #                 # new_ref_point = self.get_quintic_guide_points(guide_traj, t + self.pre_horizon * self.dt, t_start)
        #                 new_ref_point = self.get_bezier_guide_points(self.guide_traj, t + self.pre_horizon * self.dt,
        #                                                              t_start, t_end, path_num, u_num)
        #                 t += self.dt
        #
        #             if new_ref_point[index, 0] > obstacle.x + self.forward_sample:
        #                 generate_guide = 0
        #                 new_ref_point = torch.stack((
        #                     self.ref_traj.compute_x(next_t + self.pre_horizon * self.dt, path_num, u_num),
        #                     self.ref_traj.compute_y(next_t + self.pre_horizon * self.dt, path_num, u_num),
        #                     self.ref_traj.compute_phi(next_t + self.pre_horizon * self.dt, path_num, u_num),
        #                     self.ref_traj.compute_u(next_t + self.pre_horizon * self.dt, path_num, u_num),
        #                 ),
        #                     dim=1,
        #                 )
        #         else:
        #             new_ref_point = torch.stack(
        #                 (
        #                     self.ref_traj.compute_x(
        #                         next_t + self.pre_horizon * self.dt, path_num, u_num
        #                     ),
        #                     self.ref_traj.compute_y(
        #                         next_t + self.pre_horizon * self.dt, path_num, u_num
        #                     ),
        #                     self.ref_traj.compute_phi(
        #                         next_t + self.pre_horizon * self.dt, path_num, u_num
        #                     ),
        #                     self.ref_traj.compute_u(
        #                         next_t + self.pre_horizon * self.dt, path_num, u_num
        #                     ),
        #                 ),
        #                 dim=1,
        #             )

        if torch.any(generate_guide == 1) and self.guide_traj != None:
            indices = generate_guide.nonzero().squeeze(-1)
            # new_ref_point = self.get_quintic_guide_points(guide_traj, next_t + self.pre_horizon * self.dt, t_start)
            for index in indices:
                new_ref_point = self.get_bezier_guide_points(self.guide_traj, next_t + self.pre_horizon * self.dt,
                                                             t_start, t_end, path_num, u_num)
                ref_x = self.ref_traj.compute_x(next_t + self.pre_horizon * self.dt, path_num, u_num)
                t = (ref_x - new_ref_point[index, 0]) / next_state[index, 3]
                while new_ref_point[index, 0] < ref_x and t < t_end[index] and t > 0:
                    # new_ref_point = self.get_quintic_guide_points(guide_traj, t + self.pre_horizon * self.dt, t_start)
                    new_ref_point = self.get_bezier_guide_points(self.guide_traj, t + self.pre_horizon * self.dt,
                                                                 t_start, t_end, path_num, u_num)
                    t += self.dt
                if new_ref_point[index, 0] > self.obstacle.x+self.forward_sample:
                        if state[:, 0] > self.obstacle.x + self.obstacle.length / 2:
                            self.processed_obstacles.add(self.obstacle.obs_id)
                        generate_guide = 0
                        self.guide_traj = None
                        new_ref_point = torch.stack((
                            self.ref_traj.compute_x(next_t + self.pre_horizon * self.dt, path_num, u_num),
                            self.ref_traj.compute_y(next_t + self.pre_horizon * self.dt, path_num, u_num),
                            self.ref_traj.compute_phi(next_t + self.pre_horizon * self.dt, path_num, u_num),
                            self.ref_traj.compute_u(next_t + self.pre_horizon * self.dt, path_num, u_num),
                        ),
                        dim=1,
                        )

        # elif torch.all(generate_guide==0) and self.guide_traj != None:
        #     new_ref_point = torch.stack(
        #         (
        #             self.ref_traj.compute_x(
        #                 next_t + self.pre_horizon * self.dt, path_num, u_num
        #             ),
        #             self.ref_traj.compute_y(
        #                 next_t + self.pre_horizon * self.dt, path_num, u_num
        #             ),
        #             self.ref_traj.compute_phi(
        #                 next_t + self.pre_horizon * self.dt, path_num, u_num
        #             ),
        #             self.ref_traj.compute_u(
        #                 next_t + self.pre_horizon * self.dt, path_num, u_num
        #             ),
        #         ),
        #         dim=1,
        #     )
        #     if next_state[:, 0] >= self.obstacle.x:
        #         self.guide_traj = None

        else: # generate_guide==0  self.guide_traj==None
            # for batch in range(len(next_state[:,0])):
            obstacle, generate_guide = self.is_generate_guide(next_state)
            if generate_guide and obstacle != None:
                self.obstacle = obstacle
                # quintic_curves = self.generate_quintic_curves(state, obstacle)
                bezier_curves = self.generate_bezier_curves(next_state, obstacle)
                guide_traj, can_cross = self.can_cross_decision(obstacle, bezier_curves)
                self.guide_traj = guide_traj

                t_start = next_t
                t_end = next_t + (obstacle.x + self.forward_sample - next_state[:, 0]) / next_state[:, 3]
                for i in range(1, self.pre_horizon+1):
                    ref_x = self.ref_traj.compute_x(next_t + i * self.dt, path_num, u_num)
                    if self.guide_traj is None and generate_guide == 0:
                        # Use reference trajectory directly if guide trajectory is not available
                        guide_point = torch.stack(
                            (
                                self.ref_traj.compute_x(
                                    next_t + i * self.dt, path_num, u_num
                                ),
                                self.ref_traj.compute_y(
                                    next_t + i * self.dt, path_num, u_num
                                ),
                                self.ref_traj.compute_phi(
                                    next_t + i * self.dt, path_num, u_num
                                ),
                                self.ref_traj.compute_u(
                                    next_t + i * self.dt, path_num, u_num
                                ),
                            ),
                            dim=1,
                        )
                    else:
                        guide_point = self.get_bezier_guide_points(guide_traj, next_t + i * self.dt, t_start, t_end, path_num, u_num)
                        t = (ref_x - guide_point[:, 0]) / guide_point[:, 3]
                        while guide_point[:, 0] < ref_x and t < t_end and t > 0:
                            # guide_point = self.get_guide_points(self.guide_traj, t, self.t_start)
                            guide_point = self.get_bezier_guide_points(guide_traj, t, t_start, t_end, path_num, u_num)
                            t += self.dt
                        # 超出范围，使用全局轨迹点, 若bezier 曲线，范围调到障碍物前方采样点
                        if guide_point[:, 0] > obstacle.x + self.forward_sample:
                            if state[:, 0] > obstacle.x + obstacle.length / 2:
                                self.processed_obstacles.add(self.obstacle.obs_id)
                            generate_guide = 0
                            self.guide_traj = None
                            guide_point = torch.stack(
                                (
                                    self.ref_traj.compute_x(
                                        next_t + self.pre_horizon * self.dt, path_num, u_num
                                    ),
                                    self.ref_traj.compute_y(
                                        next_t + self.pre_horizon * self.dt, path_num, u_num
                                    ),
                                    self.ref_traj.compute_phi(
                                        next_t + self.pre_horizon * self.dt, path_num, u_num
                                    ),
                                    self.ref_traj.compute_u(
                                        next_t + self.pre_horizon * self.dt, path_num, u_num
                                    ),
                                ),
                                dim=1,
                            )
                    next_ref_points[:, i, :] = guide_point
                new_ref_point = guide_point
            else:
                new_ref_point = torch.stack(
                    (
                        self.ref_traj.compute_x(
                            next_t + self.pre_horizon * self.dt, path_num, u_num
                        ),
                        self.ref_traj.compute_y(
                            next_t + self.pre_horizon * self.dt, path_num, u_num
                        ),
                        self.ref_traj.compute_phi(
                            next_t + self.pre_horizon * self.dt, path_num, u_num
                        ),
                        self.ref_traj.compute_u(
                            next_t + self.pre_horizon * self.dt, path_num, u_num
                        ),
                    ),
                    dim=1,
                )
        next_ref_points[:, -1] = new_ref_point
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
            "generate_guide": torch.tensor([generate_guide]),
            "guide_start": guide_start,
            "guide_end": guide_end,
            "guide_time_interval": guide_time_interval,
            "t_start": t_start,
            "t_end": t_end
        })

        next_done = self.judge_done(next_obs, next_info)
        return next_obs, reward, next_done, next_info

    def compute_reward(
            self,
            obs: torch.Tensor,
            action: torch.Tensor,
            info: InfoDict,
    ) -> torch.Tensor:
        delta_x, delta_y, delta_phi, delta_u = obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3]
        v, w = obs[:, 4], obs[:, 5]
        steer, a_x = action[:, 0], action[:, 1]
        dis = - self.get_constraint(obs, info)#.min()
        # dis = torch.min(dis)
        collision_bound = 0.5
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
                + 15.0 * punish_dis.squeeze()
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
        done = torch.zeros(obs.shape[0]).bool()
        return done

    def find_nearest_obstacle(self,
                              current_pos: torch.Tensor,
                              static_obstacles) -> Tuple[Optional[int], float]:
        """查找最近的障碍物"""
        if not static_obstacles:
            return None, float('inf')

        min_dist = float('inf')
        nearest_id = None
        for obs_id, static_obstacle in enumerate(static_obstacles):
            for i in range(len(static_obstacle.x)):
                # 转换障碍物位置为tensor
                obs_pos = torch.tensor([static_obstacle.x[i], static_obstacle.y[i]]).unsqueeze(0)

                # 计算每个点与障碍物之间的差值
                diff = current_pos - obs_pos  # 结果形状为[64, 2]

                # 计算欧氏距离（L2范数）
                dist = torch.norm(diff, dim=1)  # 结果形状为[64]
                # 计算到障碍物的距离
                # dist = torch.norm(current_pos - obs_pos) # [batch_size, 2]

                # 只考虑前方的障碍物
                if obs_pos[0, 0] > current_pos[:, 0] and dist < min_dist:
                    min_dist = dist.item()
                    nearest_id = obs_id

        return nearest_id, min_dist

    # def is_generate_guide(self, state) -> Tuple[Optional[StaticObstacle], int]:
    #     current_pos = state[:, :2]
    #     nearest_static_id, static_dist = self.find_nearest_obstacle(
    #         current_pos, self.static_obss)
    #
    #     if static_dist <= self.d_pre:
    #         obstacle = self.static_obss[nearest_static_id]
    #         return obstacle, 1
    #     else:
    #         return None, 0

    def is_generate_guide(self, state) -> Tuple[Optional[StaticObstacle], int]:
        current_pos = state[:, :2]
        obstacles_in_range = []
        # 找出所有在规划范围内的障碍物
        for obstacle in self.static_obss:
            obs_pos = torch.tensor([obstacle.x, obstacle.y])
            dist = torch.norm(current_pos - obs_pos)
            # 只考虑前方的障碍物且在规划范围内
            if obs_pos[0] > current_pos[:, 0] and dist.item() <= self.d_pre:
                obstacles_in_range.append((obstacle, dist))
            # 没有障碍物需要处理
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

    def can_cross_decision(self, obstacle: StaticObstacle, curves: list) -> Tuple[Optional[BezierCurve], bool]:
        can_cross = False

        if (obstacle.height < self.ground_clearance and
                obstacle.width < self.wheel_distance):
            can_cross = True
        self.curve_index = None
        if can_cross:
            self.curve_index = 1
            best_curve = curves[self.curve_index]  # 可跨就选跨
        else:
            self.curve_index = 2
            best_curve = curves[self.curve_index]  # 不可跨就选择左绕
        return best_curve, can_cross

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

    def generate_bezier_curves(self, state:torch.Tensor,obstacle: StaticObstacle) -> list:
        """生成贝塞尔曲线(Tensor版本)
        Args:
            obstacle: 障碍物信息
        Returns:
            curves: 生成的贝塞尔曲线列表
        """
        # 确保状态和采样参数为tensor
        state = torch.tensor(state[:, :2], dtype=torch.float32) \
            if not isinstance(state, torch.Tensor) else state[:, :2]
        lateral_sample = torch.tensor(self.lateral_sample, dtype=torch.float32) \
            if not isinstance(self.lateral_sample, torch.Tensor) else self.lateral_sample
        forward_sample = torch.tensor(self.forward_sample, dtype=torch.float32) \
            if not isinstance(self.forward_sample, torch.Tensor) else self.forward_sample
        curves = []

        # 横向采样点 - 使用tensor
        lateral_points = torch.stack([
            torch.tensor([obstacle.x, obstacle.y - lateral_sample-obstacle.width/2-1.0], dtype=torch.float32),  # 左侧
            torch.tensor([obstacle.x, obstacle.y], dtype=torch.float32),  # 中心
            torch.tensor([obstacle.x, obstacle.y + lateral_sample+obstacle.width/2+1.0], dtype=torch.float32)  # 右侧
        ])

        # 纵向采样点 - 使用tensor
        longitudinal_points = torch.tensor(
            [[obstacle.x + forward_sample+obstacle.length*1.5, obstacle.y]],
            dtype=torch.float32
        )

        # 生成贝塞尔曲线
        for i in range(lateral_points.shape[0]):
            # 使用当前位置作为起点，障碍物位置作为控制点
            curve = BezierCurve(
                state,
                lateral_points[i],
                longitudinal_points[0]
            )
            curves.append(curve)
        return curves

    def get_quintic_guide_points(self, guide_traj, current_t: float, t_start: float) -> torch.Tensor:
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

    def get_bezier_guide_points(self, guide_traj, t_interpolate: torch.Tensor, t_start: torch.Tensor, t_end: torch.Tensor, path_num, u_num)-> torch.Tensor:
        if t_end <= t_start:
            t_end = t_start + self.dt
            print("please check the time end and time start model")
        t = (t_interpolate - t_start) / (t_end - t_start)
        point = guide_traj.compute_point(t)
        derivative = guide_traj.compute_derivative(t)
        # 确保导数不为零向量
        if torch.norm(derivative) < 1e-6:
            derivative = torch.tensor([1e-6, 0]).reshape((-1, 2))  # 小量向前
        phi = torch.arctan2(derivative[:, 1], derivative[:, 0])
        global_phi = self.ref_traj.compute_phi(t_interpolate, path_num, u_num)
        blend_ratio = torch.clip(t, 0.2, 0.8)  # 渐进混合系数
        phi = angle_normalize(blend_ratio * phi + (1 - blend_ratio) * global_phi)

        u = self.ref_traj.compute_u(t_interpolate, path_num, u_num)

        # 确保速度方向与航向一致
        if torch.any(u<0):
            phi = angle_normalize(phi+torch.pi)
            u = torch.abs(u)
        traj_point = torch.stack([point[:, 0], point[:, 1], phi, u]).reshape((-1, 4))
        return traj_point

    # def generate_bezier_curves(self, obstacle: StaticObstacle) -> List[BezierCurve]:
    #     """生成贝塞尔曲线
    #     Args:
    #         current_pos: 当前位置[x, y]
    #         obstacle: 障碍物信息
    #     Returns:
    #         curves: 生成的贝塞尔曲线列表
    #     """
    #     curves = []
    #
    #     # 横向采样点
    #     lateral_points = [
    #         np.array([obstacle.x, obstacle.y - self.lateral_sample]),  # 左侧
    #         np.array([obstacle.x, obstacle.y]),  # 中心
    #         np.array([obstacle.x, obstacle.y + self.lateral_sample])  # 右侧
    #     ]
    #
    #     # 纵向采样点
    #     longitudinal_points = [
    #         np.array([obstacle.x + self.forward_sample, obstacle.y])  # 前方位置
    #     ]
    #
    #     # 生成贝塞尔曲线
    #     for i in lateral_points:
    #         # 使用当前位置作为起点，障碍物位置作为控制点
    #         curve = BezierCurve(
    #             self.state[:2],
    #             lateral_points[i],
    #             longitudinal_points[0]
    #         )
    #         curves.append(curve)
    #
    #     return curves
    #
    # def is_generate_guide(self) -> Tuple[Optional[Obstacle], bool]:
    #     generate_guide = False
    #     current_pos = self.state[:2]
    #     ego_velocity = self.state[3]  # 自车速度
    #
    #     # 查找最近的动态和静态障碍物
    #     nearest_dynamic_id, dynamic_dist = self.obstacle_processor.find_nearest_obstacle(
    #         current_pos, self.obstacle_processor.dynamic_obstacles)
    #     nearest_static_id, static_dist = self.obstacle_processor.find_nearest_obstacle(
    #         current_pos, self.obstacle_processor.static_obstacles)
    #
    #     # 确定处理顺序
    #     if dynamic_dist == float('inf') and static_dist == float('inf'):
    #         return None, generate_guide
    #
    #     # 判断是处理动态还是静态障碍物
    #     if dynamic_dist <= static_dist:
    #         if dynamic_dist <= self.d_pre:
    #             obstacle = self.obstacle_processor.dynamic_obstacles[nearest_dynamic_id]
    #             # 检查动态障碍物的速度
    #             if obstacle.u >= ego_velocity:
    #                 return None, generate_guide
    #             else:
    #                 generate_guide = True
    #                 return obstacle, generate_guide
    #         else:
    #             return None, generate_guide
    #     else:
    #         if static_dist <= self.d_pre:
    #             obstacle = self.obstacle_processor.static_obstacles[nearest_static_id]
    #             generate_guide = True
    #             return obstacle, generate_guide
    #         else:
    #             return None, generate_guide
    #
    # def can_cross_decision(self, obstacle: Obstacle, curves: List) -> Tuple[QuinticPolynomial, bool]:
    #     """跨/绕决策
    #     Args:
    #         obstacle: 障碍物信息
    #     Returns:
    #         can_cross: 是否可以跨越
    #     """
    #
    #     can_cross = False
    #
    #     # 如果是静态障碍物，则判断尺寸是否满足跨越条件
    #     if (obstacle.height < self.ground_clearance and
    #             obstacle.width < self.wheel_distance):
    #         can_cross = True
    #
    #     if can_cross:
    #         best_curve = curves[1]  # 可跨就选跨
    #     else:
    #         best_curve = curves[0]  # 不可跨就选择左绕
    #
    #     return best_curve, can_cross
    #
    # def generate_quintic_curves(self,
    #                             obstacle: Obstacle) -> List:
    #     """生成五次多项式曲线
    #             Args:
    #                 current_pos: 当前位置[x, y]
    #                 obstacle: 障碍物信息
    #             Returns:
    #                 curves: 生成的五次多项式曲线列表
    #             """
    #
    #     def _create_single_trajectory(start_point, end_point, T) -> Tuple[QuinticPolynomial, QuinticPolynomial]:
    #         """创建单条轨迹的x和y方向多项式"""
    #         # x方向多项式
    #         poly_x = QuinticPolynomial(
    #             [start_point[0], start_point[3] * np.cos(start_point[2]), 0],  # 位置、速度、加速度
    #             [end_point[0], end_point[3] * np.cos(end_point[2]), 0],  # 位置、速度、加速度
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
    #         np.array([obstacle.x, obstacle.y - self.lateral_sample, obstacle.phi, obstacle.u]),  # 左侧
    #         np.array([obstacle.x, obstacle.y, obstacle.phi, obstacle.u]),  # 中心
    #         np.array([obstacle.x, obstacle.y + self.lateral_sample, obstacle.phi, obstacle.u])  # 右侧
    #     ]
    #     # 生成五次多项式曲线
    #     for i in lateral_points:
    #         curve = _create_single_trajectory(self.state, lateral_points[i],
    #                                           (obstacle.x - self.state[0]) / self.state[2])
    #         # 使用当前位置作为起点，障碍物位置作为控制点
    #         curves.append(curve)
    #     return curves
    #
    # def get_guide_points(self, guide_traj, current_t: float,
    #                      t_start: float) -> np.array:
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
    #     # 尝试从五次多项式获取点
    #     x_state = poly_x.calc_point(local_t)
    #     y_state = poly_y.calc_point(local_t)
    #     x, y = x_state[0], y_state[0]
    #     vx, vy = x_state[1], y_state[1]
    #     phi = np.arctan2(vy, vx)
    #     return np.array([x, y, phi, vx], dtype=np.float32)

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
