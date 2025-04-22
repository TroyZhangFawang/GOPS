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
from gops.env.env_ocp.env_model.pyth_base_model import PythBaseModel
import numpy as np
from gops.env.env_ocp.pyth_veh3dofconti_multimodal_planning  import (
    VehicleDynamicsData,
    angle_normalize,
    ObstacleType,
    Obstacle
)
from gops.env.env_ocp.resources.ref_traj_model import MultiRefTrajModel
from gops.utils.gops_typing import InfoDict
import torch
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

@dataclass
class Obstacle:
    obs_id: str  # 障碍物ID
    x: float = 0.0  # 障碍物中心点x坐标
    y: float = 0.0  # 障碍物中心点y坐标
    phi: float = 0.0
    u: float = 0.0
    # front wheel angle
    delta: float = 0.0
    # distance from front axle to rear axle
    l: float = 3.0
    dt: float = 0.1
    height: float = 0.5  # 障碍物高度
    width: float = 1.8  # 障碍物宽度
    obs_type: ObstacleType = 1  # 障碍物类型, 0-静态，1-动态

class ObstacleType(Enum):
    """障碍物类型枚举"""
    STATIC = 0  # 静态障碍物
    DYNAMIC = 1  # 动态障碍物

class VehicleDynamicsModel(VehicleDynamicsData):
    def f_xu(self, states, actions, delta_t):
        x, y, phi, u, v, w = (
            states[:, 0],
            states[:, 1],
            states[:, 2],
            states[:, 3],
            states[:, 4],
            states[:, 5],
        )
        steer, a_x = actions[:, 0], actions[:, 1]
        k_f = self.vehicle_params["k_f"]
        k_r = self.vehicle_params["k_r"]
        l_f = self.vehicle_params["l_f"]
        l_r = self.vehicle_params["l_r"]
        m = self.vehicle_params["m"]
        I_z = self.vehicle_params["I_z"]
        next_state = [
            x + delta_t * (u * torch.cos(phi) - v * torch.sin(phi)),
            y + delta_t * (u * torch.sin(phi) + v * torch.cos(phi)),
            phi + delta_t * w,
            u + delta_t * a_x,
            (
                m * v * u
                + delta_t * (l_f * k_f - l_r * k_r) * w
                - delta_t * k_f * steer * u
                - delta_t * m * torch.square(u) * w
            )
            / (m * u - delta_t * (k_f + k_r)),
            (
                I_z * w * u
                + delta_t * (l_f * k_f - l_r * k_r) * v
                - delta_t * l_f * k_f * steer * u
            )
            / (I_z * u - delta_t * (l_f ** 2 * k_f + l_r ** 2 * k_r)),
        ]
        next_state[2] = angle_normalize(next_state[2])
        return torch.stack(next_state, 1)

class MultiObstacleProcessor:
    """多障碍物处理器"""

    def __init__(self):
        self.static_obstacles: Dict[str, Obstacle] = {}  # 静态障碍物集合
        self.dynamic_obstacles: Dict[str, Obstacle] = {}  # 动态障碍物集合
        self.dt = 0.1
        self.l = 3.0

    def add_obstacle(self, obstacle: Obstacle):
        """添加障碍物到对应集合"""
        if obstacle.obs_type == ObstacleType.STATIC:
            self.static_obstacles[obstacle.obs_id] = obstacle
        else:
            self.dynamic_obstacles[obstacle.obs_id] = obstacle

    def clear_obstacles(self):
        """清空障碍物集合"""
        self.static_obstacles.clear()
        self.dynamic_obstacles.clear()

    def find_nearest_obstacle(self,
                              current_pos: torch.Tensor,
                              obstacles: Dict[str, Obstacle]) -> Tuple[Optional[str], float]:
        """查找最近的障碍物"""
        if not obstacles:
            return None, float('inf')

        min_dist = float('inf')
        nearest_id = None

        for obs_id, obstacle in obstacles.items():
            # 转换障碍物位置为tensor
            obs_pos = torch.tensor([obstacle.x, obstacle.y])
            # 计算到障碍物的距离
            dist = torch.norm(current_pos - obs_pos)

            # 只考虑前方的障碍物
            if obs_pos[0] > current_pos[:, 0] and dist.item() < min_dist:
                min_dist = dist.item()
                nearest_id = obs_id

        return nearest_id, min_dist

    def step(self, state):
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

    def compute_point(self, t: float) -> torch.Tensor:
        """计算贝塞尔曲线上的点
        Args:
            t: 参数t，范围[0,1]
        Returns:
            point: [x, y]坐标的tensor
        """
        t = torch.tensor(t)
        return (1 - t) ** 2 * self.p0 + 2 * (1 - t) * t * self.p1 + t ** 2 * self.p2

    def compute_derivative(self, t: float) -> torch.Tensor:
        """计算贝塞尔曲线在t处的导数"""
        t = torch.tensor(t)
        return 2 * (1 - t) * (self.p1 - self.p0) + 2 * t * (self.p2 - self.p1)

class Veh3dofcontiSurrCstrModel(PythBaseModel):
    def __init__(
        self,
        pre_horizon: int,
        device: Union[torch.device, str, None] = None,
        path_para: Optional[Dict[str, Dict]] = None,
        u_para: Optional[Dict[str, Dict]] = None,
        dynamic_obstacle_num: int = 1,
        static_obstacle_num: int = 1,
        d_pre: float = 10.0,  # 离障碍物多少远开始规划
        lateral_sample: float = 5.0,  # 横向采样距离
        forward_sample: float = 10.0,  # 纵向采样距离
        **kwargs: Any,
    ):
        self.state_dim = 6
        self.ego_obs_dim = 6
        self.ref_obs_dim = 4
        super().__init__(
            obs_dim=self.ego_obs_dim + self.ref_obs_dim * pre_horizon + (dynamic_obstacle_num+static_obstacle_num) * 4,
            action_dim=2,
            dt=0.01,
            action_lower_bound=[-np.pi / 6, -3],
            action_upper_bound=[np.pi / 6, 3],
            device=device,
        )
        self.vehicle_dynamics = VehicleDynamicsModel()
        self.obstacle_processor = MultiObstacleProcessor()
        self.ref_traj = MultiRefTrajModel(path_para, u_para)
        self.pre_horizon = pre_horizon
        self.dynamic_obstacle_num = dynamic_obstacle_num
        self.static_obstacle_num = static_obstacle_num
        self.veh_length = self.vehicle_dynamics.vehicle_params["veh_length"]
        self.veh_width = self.vehicle_dynamics.vehicle_params["veh_width"]
        self.wheel_distance = self.vehicle_dynamics.vehicle_params["wheel_distance"]
        self.ground_clearance = self.vehicle_dynamics.vehicle_params["ground_clearance"]
        self.d_pre = d_pre
        self.lateral_sample = lateral_sample
        self.forward_sample = forward_sample
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
        obstacles = []
        # for i_dynamic in range(self.dynamic_obstacle_num):
        #     obstacles.append(
        #         Obstacle("D{}".format(i_dynamic),
        #                  x=dynamic_state[0, i_dynamic, 0],
        #                  y=dynamic_state[0, i_dynamic, 1],
        #                  phi=dynamic_state[0, i_dynamic, 2],
        #                  u=dynamic_state[0, i_dynamic, 3],
        #                  delta=dynamic_state[0, i_dynamic, 4],
        #                  dt=self.dt,
        #                  height=1.85,
        #                  width=1.93,
        #                  obs_type=ObstacleType.DYNAMIC
        #                  )
        #     )


        reward = self.compute_reward(obs, action, info)
        next_state = self.vehicle_dynamics.f_xu(state, action, self.dt)

        next_dynamic_state = self.obstacle_processor.step(dynamic_state)
        # dynamic_x_tf, dynamic_y_tf, dynamic_phi_tf = \
        #     ego_vehicle_coordinate_transform(
        #         state[:, 0], state[:, 1], state[:, 2],
        #         next_dynamic_state[..., 0], next_dynamic_state[..., 1], next_dynamic_state[..., 2],
        #     )
        # dynamic_u_tf = next_dynamic_state[..., 3] - state[:, 3].unsqueeze(1)
        next_dynamic_obs = next_dynamic_state[..., :4] - next_state[:, :4].unsqueeze(1)
        next_dynamic_obs = next_dynamic_obs.reshape((-1, self.dynamic_obstacle_num * 4))

        # static obstacle
        # static_x_tf, static_y_tf, static_phi_tf = \
        #     ego_vehicle_coordinate_transform(
        #         state[:, 0], state[:, 1], state[:, 2],
        #         static_state[..., 0], static_state[..., 1], static_state[..., 2],
        #     )
        # static_u_tf = static_state[..., 3]- state[:, 3].unsqueeze(1)
        # next_static_obs = torch.stack((static_x_tf, static_y_tf, static_phi_tf, static_u_tf), 1).squeeze(2)
        next_static_obs = static_state[..., :4] - next_state[:, :4].unsqueeze(1)
        next_static_obs = next_static_obs.reshape((-1, self.static_obstacle_num * 4))
        next_t = t + self.dt
        next_ref_points = ref_points.clone()
        next_ref_points[:, :-1] = ref_points[:, 1:]
        for i_static in range(self.static_obstacle_num):
            obstacles.append(
                Obstacle("S{}".format(i_static),
                         x=static_state[0, i_static, 0],
                         y=static_state[0, i_static, 1],
                         phi=static_state[0, i_static, 2],
                         u=0,
                         delta=static_state[0, i_static, 3],
                         dt=self.dt,
                         height=1.85,
                         width=1.93,
                         obs_type=ObstacleType.STATIC
                         )
            )
        # 将障碍物添加到处理器中
        for obstacle in obstacles:
            self.obstacle_processor.add_obstacle(obstacle)
        if generate_guide.item() == 1:
            obstacle, generate_guide = self.is_generate_guide(state)
            if generate_guide and obstacle != None:
                curves = self.generate_quintic_curves(guide_start, guide_end, guide_time_interval)
                guide_traj, can_cross = self.can_cross_decision(obstacle, curves)
                new_ref_point = self.get_guide_points(guide_traj, next_t + self.pre_horizon * self.dt, t_start)
                ref_x = self.ref_traj.compute_x(
                    next_t + self.pre_horizon * self.dt, path_num, u_num
                )
                t = (ref_x - new_ref_point[0][0]) / new_ref_point[0][3]
                while new_ref_point[0][0] <= ref_x:
                    new_ref_point = self.get_guide_points(guide_traj, t + self.pre_horizon * self.dt, t_start)
                    t += self.dt
                if new_ref_point[0][0] >= guide_end[:, self.curve_index, 0]:
                    new_ref_point = guide_end[:, self.curve_index, :].reshape(-1, 4)
                # print("0 new_ref_point",new_ref_point, "obstacle.x",obstacle.x)
                if new_ref_point[0][0] > obstacle.x:
                    generate_guide = 0
                    new_ref_point = torch.stack((
                        self.ref_traj.compute_x(next_t + self.pre_horizon * self.dt, path_num, u_num),
                        self.ref_traj.compute_y(next_t + self.pre_horizon * self.dt, path_num, u_num),
                        self.ref_traj.compute_phi(next_t + self.pre_horizon * self.dt, path_num, u_num),
                        self.ref_traj.compute_u(next_t + self.pre_horizon * self.dt, path_num, u_num),
                    ),
                    dim=1,
                    )
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
        else:
            obstacle, generate_guide = self.is_generate_guide(next_state)
            if generate_guide and obstacle != None:
                guide_start = next_state[:, :4]
                # 横向采样点
                guide_end = torch.stack(
                    (torch.tensor([obstacle.x, obstacle.y - self.lateral_sample, obstacle.phi, obstacle.u]),  # 左侧
                    torch.tensor([obstacle.x, obstacle.y, obstacle.phi, obstacle.u]),  # 中心
                    torch.tensor([obstacle.x, obstacle.y + self.lateral_sample, obstacle.phi, obstacle.u])),  # 右侧
                dim=1
                ).reshape((-1, 3, 4))
                guide_time_interval = abs((obstacle.x - next_state[:, 0]) / next_state[:, 3])
                curves = self.generate_quintic_curves(guide_start, guide_end, guide_time_interval)
                guide_traj, can_cross = self.can_cross_decision(obstacle, curves)
                t_start = next_t
                new_ref_point = self.get_guide_points(guide_traj, next_t + self.pre_horizon * self.dt, t_start)
                # print("1new_ref_point",new_ref_point, "obstacle.x",obstacle.x)
                ref_x = self.ref_traj.compute_x(
                            next_t + self.pre_horizon * self.dt, path_num, u_num
                        )
                t = (ref_x - new_ref_point[0][0]) / new_ref_point[0][3]
                while new_ref_point[0][0] <= ref_x:
                    new_ref_point = self.get_guide_points(guide_traj, t + self.pre_horizon * self.dt, t_start)
                    t += self.dt
                if new_ref_point[0][0] >= guide_end[:, self.curve_index, 0]:
                    new_ref_point = guide_end[:, self.curve_index, :].reshape(-1, 4)
                # 超出范围，使用全局轨迹点
                if next_state[0][0] > obstacle.x:
                    generate_guide = 0
                    new_ref_point = torch.stack((
                        self.ref_traj.compute_x(next_t + self.pre_horizon * self.dt, path_num, u_num),
                        self.ref_traj.compute_y(next_t + self.pre_horizon * self.dt, path_num, u_num),
                        self.ref_traj.compute_phi(next_t + self.pre_horizon * self.dt, path_num, u_num),
                        self.ref_traj.compute_u(next_t + self.pre_horizon * self.dt, path_num, u_num),
                    ),

                        dim=1,
                    )
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
        next_obs = self.get_obs(next_state, next_ref_points, next_dynamic_obs, next_static_obs)

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
            "t_start": t_start
        })

        next_done = self.judge_done(next_obs, next_info)
        return next_obs, reward, next_done, next_info

    def get_obs(self, state, ref_points, next_dynamic_obs, next_static_obs):
        ref_x_tf, ref_y_tf, ref_phi_tf = \
            ego_vehicle_coordinate_transform(
                state[:, 0], state[:, 1], state[:, 2],
                ref_points[..., 0], ref_points[..., 1], ref_points[..., 2],
            )
        ref_u_tf = ref_points[..., 3] - state[:, 3].unsqueeze(1)
        ego_obs = torch.concat((torch.stack(
            (ref_x_tf[:, 0], ref_y_tf[:, 0], ref_phi_tf[:, 0], ref_u_tf[:, 0]), dim=1),
            state[:, 4:]), dim=1)
        ref_obs = torch.stack((ref_x_tf, ref_y_tf, ref_phi_tf, ref_u_tf), 2)[
            :, 1:].reshape(ego_obs.shape[0], -1)

        next_obs = torch.cat((ego_obs, ref_obs, next_dynamic_obs, next_static_obs), dim=1)
        return next_obs

    def compute_reward(
            self,
            obs: torch.Tensor,
            action: torch.Tensor,
            info: InfoDict,
    ) -> torch.Tensor:
        delta_x, delta_y, delta_phi, delta_u = obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3]
        v, w = obs[:, 4], obs[:, 5]
        steer, a_x = action[:, 0], action[:, 1]
        dis = - self.get_constraint(obs, info)
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
        # return -(
        #         0.5 * delta_x ** 2
        #         + 0.5 * delta_y ** 2
        #         + 0.1 * delta_phi ** 2
        #         + 0.1 * delta_u ** 2
        #         + 0.5 * v ** 2
        #         + 0.5 * w ** 2
        #         + 0.5 * steer ** 2
        #         + 0.5 * a_x ** 2
        #         + 15.0 * punish_dis.squeeze()
        # )

    def get_constraint(self, obs: torch.Tensor, info: InfoDict) -> torch.Tensor:
        # collision detection using bicircle model
        # distance from vehicle center to front/rear circle center
        d = (self.veh_length - self.veh_width) / 2
        # circle radius
        r = np.sqrt(2) / 2 * self.veh_width

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
        # static
        # static_x, static_y, static_phi = info["static_state"][..., :3].split(1, dim=2)
        # static_center = torch.stack(
        #     (
        #         torch.cat(
        #             (
        #                 (static_x + d * torch.cos(static_phi)),
        #                 static_y + d * torch.sin(static_phi),
        #             ),
        #             dim=2,
        #         ),
        #         torch.cat(
        #             (
        #                 (static_x - d * torch.cos(static_phi)),
        #                 static_y - d * torch.sin(static_phi),
        #             ),
        #             dim=2,
        #         ),
        #     ),
        #     dim=2,
        # )

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
        # min_dist_static = np.finfo(np.float32).max * torch.ones_like(static_x).squeeze(-1)
        # for i in range(2):
        #     # front and rear circle of ego vehicle
        #     for j in range(2):
        #         # front and rear circle of surrounding vehicles
        #         dist = torch.linalg.norm(
        #             ego_center[:, i].unsqueeze(1) - static_center[..., j, :], dim=2
        #         )
        #         min_dist_static = torch.minimum(min_dist_static, torch.min(dist, dim=1, keepdim=True).values)
        # min_dist = torch.minimum(min_dist_dynamic, min_dist_static)
        # min_dist = torch.tensor([torch.min(min_dist)])
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

    def is_generate_guide(self, state) -> Tuple[Optional[Obstacle], bool]:
        current_pos = state[:, :2]
        ego_velocity = state[:, 3]  # 自车速度

        # 查找最近的动态和静态障碍物
        # nearest_dynamic_id, dynamic_dist = self.obstacle_processor.find_nearest_obstacle(
        #     current_pos, self.obstacle_processor.dynamic_obstacles)
        nearest_static_id, static_dist = self.obstacle_processor.find_nearest_obstacle(
            current_pos, self.obstacle_processor.static_obstacles)

        # 确定处理顺序
        # if torch.isinf(torch.tensor(dynamic_dist)) and torch.isinf(torch.tensor(static_dist)):
        #     return None, 0

        if static_dist <= self.d_pre:
            obstacle = self.obstacle_processor.static_obstacles[nearest_static_id]
            return obstacle, 1
        else:
            return None, 0

        # # 判断是处理动态还是静态障碍物
        # if dynamic_dist <= static_dist:
        #     if dynamic_dist <= self.d_pre:
        #         obstacle = self.obstacle_processor.dynamic_obstacles[nearest_dynamic_id]
        #         if obstacle.u >= ego_velocity:
        #             return None, 0
        #         else:
        #             return obstacle, 1
        #     else:
        #         return None, 0
        # else:
        #     if static_dist <= self.d_pre:
        #         obstacle = self.obstacle_processor.static_obstacles[nearest_static_id]
        #         return obstacle, 1
        #     else:
        #         return None, 0

    def can_cross_decision(self, obstacle: Obstacle, curves: list) -> Tuple[QuinticPolynomial, bool]:
        can_cross = False

        if (obstacle.height < self.ground_clearance and
                obstacle.width < self.wheel_distance):
            can_cross = True
        self.curve_index = None
        if can_cross:
            self.curve_index = 1
            best_curve = curves[self.curve_index]  # 可跨就选跨
        else:
            self.curve_index = 0
            best_curve = curves[self.curve_index]  # 不可跨就选择左绕
        return best_curve, can_cross

    def generate_quintic_curves(self, guide_start, guide_end, time_interval) -> list:
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

        curves = []
        # 生成五次多项式曲线
        for i in range(3):
            curve = _create_single_trajectory(
                guide_start,
                guide_end[:, i, :].reshape(-1, 4),
                time_interval
            )
            curves.append(curve)
        return curves

    def generate_bezier_curves(self,
                               current_pos: torch.Tensor,
                               obstacle: Obstacle) -> torch.Tensor:
        """生成贝塞尔曲线的控制点
        Args:
            current_pos: 当前位置[x, y]
            obstacle: 障碍物信息
        Returns:
            curves_tensor: 形状为(3, 3, 2)的tensor, 包含3条曲线的控制点
                          第一维: 3条不同的曲线(左绕、穿过、右绕)
                          第二维: 每条曲线的3个控制点(起点、控制点、终点)
                          第三维: 每个点的x,y坐标
        """
        # 转换障碍物位置为tensor
        obs_pos = torch.tensor([obstacle.x, obstacle.y], device=self.device)

        # 创建一个(3, 3, 2)的tensor来存储所有曲线的控制点
        curves_tensor = torch.zeros((3, 3, 2), device=self.device)

        # 设置所有曲线的起点(current_pos)
        curves_tensor[:, 0] = current_pos

        # 设置三个横向采样点作为控制点
        curves_tensor[0, 1] = torch.tensor([obstacle.x, obstacle.y - self.lateral_sample], device=self.device)  # 左侧
        curves_tensor[1, 1] = obs_pos  # 中心
        curves_tensor[2, 1] = torch.tensor([obstacle.x, obstacle.y + self.lateral_sample], device=self.device)  # 右侧

        # 设置终点(所有曲线共用同一个前方点)
        forward_point = torch.tensor([obstacle.x + self.forward_sample, obstacle.y], device=self.device)
        curves_tensor[:, 2] = forward_point

        return curves_tensor

    def get_guide_points(self, guide_traj, current_t: float, t_start: float) -> torch.Tensor:
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
    return Veh3dofcontiSurrCstrModel(**kwargs)
