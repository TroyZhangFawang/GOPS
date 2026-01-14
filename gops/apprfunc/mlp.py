#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Multilayer Perceptron (MLP)
#  Update: 2021-03-05, Wenjun Zou: create MLP function
#  Update: 2023-07-28, Jiaxin Gao: add FiniteHorizonFullPolicy function
#  Update: 2023-10-25, Wenxuan Wang: add DSAC-T algorithm


__all__ = [
    "DetermPolicy",
    "FiniteHorizonPolicy",
    "FiniteHorizonFullPolicy",
    "DiffusionPolicyeasy",
    "StochaPolicy",
    "EncodingStochaPolicy",
    "EncodingStochaPolicy2",
    "DiffusionEncondingNet",
    "DSACTCriticEncodingNet",
    "ActionValue",
    "ActionValueDis",
    "ActionValueDistri",
    "StochaPolicyDis",
    "StateValue",
]

import numpy as np
import torch
import warnings
import torch.nn as nn
from gops.utils.common_utils import get_activation_func
from gops.utils.act_distribution_cls import Action_Distribution
from gops.utils.diffusion_helpers import (
    cosine_beta_schedule,
    linear_beta_schedule,
    vp_beta_schedule,
    extract,
    Losses,
    SinusoidalPosEmb,
)
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import torch
import torch.nn as nn
import numpy as np
import math


# ==========================================
# 1. 基础组件：时间步编码 (Sinusoidal Positional Embedding)
# ==========================================
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# ==========================================
# 2. 改造后的 Actor：Diffusion Noise Predictor
# ==========================================
class DiffusionEncondingNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]  # e.g., [256, 256, 256]

        # === A. 维度定义 (照搬你原来的逻辑) ===
        self.ego_dim = 6
        self.ref_dim = 20 * 6
        self.obs_dim = 3 * 8
        self.num_prompts = 3
        self.points_per_prompt = 20

        # === B. 状态编码器 (Encoders) ===
        self.ref_encoder = nn.Sequential(nn.Linear(self.ref_dim, 128), nn.ReLU(), nn.Linear(128, 64))
        self.obstacle_encoder = nn.Sequential(nn.Linear(self.obs_dim, 128), nn.ReLU(), nn.Linear(128, 64))
        # 假设 PromptEncoder 是你自定义的类，这里保留接口
        # self.prompt_encoder = PromptEncoder(input_dim=2, output_dim=32)
        # 为了演示，我用 Linear 代替，请换回你自己的 PromptEncoder
        self.prompt_encoder = nn.Sequential(nn.Linear(2, 32), nn.ReLU())

        self.ego_encoder = nn.Sequential(nn.Linear(self.ego_dim, 64), nn.ReLU())

        # 状态特征总维度: 64(ego) + 64(ref) + 64(obs) + 32*3(prompt) = 288
        self.state_feat_dim = 64 + 64 + 64 + (32 * 3)

        # === C. Diffusion 特有组件 ===
        # 1. 时间步编码
        self.time_dim = 32
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(self.time_dim),
            nn.Linear(self.time_dim, self.time_dim * 2),
            nn.Mish(),
            nn.Linear(self.time_dim * 2, self.time_dim),
        )

        # 2. 动作编码
        self.action_mlp = nn.Sequential(nn.Linear(act_dim, 32), nn.Mish())

        # === D. 主干网络 (Backbone) ===
        # 输入 = 状态特征 + 时间特征 + 动作特征
        input_dim = self.state_feat_dim + self.time_dim + 32

        layers = []
        last_dim = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(last_dim, size))
            layers.append(nn.Mish())  # Diffusion常用Mish激活函数
            last_dim = size
        layers.append(nn.Linear(last_dim, act_dim))  # 输出噪声，维度与 action 相同

        self.net = nn.Sequential(*layers)

    def forward(self, obs, act, t):
        """
        obs: 原始观测 [B, obs_dim]
        act: 加噪后的动作 (x_t) [B, act_dim]
        t:   时间步 [B]
        """
        # 1. 状态编码 (照搬原逻辑)
        ego = obs[:, :self.ego_dim]
        ref = obs[:, self.ego_dim: self.ego_dim + self.ref_dim]
        obst = obs[:, self.ego_dim + self.ref_dim: self.ego_dim + self.ref_dim + self.obs_dim]
        prompts = obs[:, self.ego_dim + self.ref_dim + self.obs_dim:]

        ego_feat = self.ego_encoder(ego)
        ref_feat = self.ref_encoder(ref)
        obs_feat = self.obstacle_encoder(obst)

        prompts = prompts.reshape(-1, self.num_prompts, self.points_per_prompt, 2)
        prompt_feats = []
        for i in range(self.num_prompts):
            # 注意处理 batch 维度，这里简单处理
            p_feat = self.prompt_encoder(prompts[:, i, :, :])
            # 如果 PromptEncoder 输出是 (B, N, D) 需要 flatten 或 pool，假设你原来的输出是 (B, 32)
            if p_feat.dim() > 2: p_feat = p_feat.mean(dim=1)  # 简单示例
            prompt_feats.append(p_feat)
        prompt_combined = torch.cat(prompt_feats, dim=-1)

        state_feat = torch.cat([ego_feat, ref_feat, obs_feat, prompt_combined], dim=-1)

        # 2. 时间编码
        time_feat = self.time_mlp(t)

        # 3. 动作编码
        act_feat = self.action_mlp(act)

        # 4. 融合与输出
        combined = torch.cat([state_feat, time_feat, act_feat], dim=-1)
        noise_pred = self.net(combined)

        return noise_pred


# ==========================================
# 3. 改造后的 Critic：支持编码的 DSACT Critic
# ==========================================
class DSACTCriticEncodingNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]

        # === 维度与编码器 (与 Actor 共享结构，但不共享权重) ===
        # ... (这里省略重复定义的 self.ego_encoder 等，代码同上 Actor) ...
        # 建议：为了代码整洁，可以将编码器部分封装成一个单独的 class StateEncoder(nn.Module)

        # 简单起见，这里假设你已经复制了上面的编码器定义
        self.ego_dim = 6
        self.ref_dim = 20 * 6
        self.obs_dim = 3 * 8
        self.num_prompts = 3
        self.points_per_prompt = 20
        self.ref_encoder = nn.Sequential(nn.Linear(self.ref_dim, 128), nn.ReLU(), nn.Linear(128, 64))
        self.obstacle_encoder = nn.Sequential(nn.Linear(self.obs_dim, 128), nn.ReLU(), nn.Linear(128, 64))
        self.prompt_encoder = nn.Sequential(nn.Linear(2, 32), nn.ReLU())
        self.ego_encoder = nn.Sequential(nn.Linear(self.ego_dim, 64), nn.ReLU())
        self.state_feat_dim = 64 + 64 + 64 + (32 * 3)

        # === Q 网络主干 ===
        # 输入 = 状态特征 + 动作
        input_dim = self.state_feat_dim + act_dim

        # Mean Network
        layers_mean = []
        last_dim = input_dim
        for size in hidden_sizes:
            layers_mean.append(nn.Linear(last_dim, size))
            layers_mean.append(nn.ReLU())
            last_dim = size
        layers_mean.append(nn.Linear(last_dim, 1))  # 输出 Q mean
        self.mean_net = nn.Sequential(*layers_mean)

        # Std Network (DSACT 特有)
        layers_std = []
        last_dim = input_dim
        for size in hidden_sizes:
            layers_std.append(nn.Linear(last_dim, size))
            layers_std.append(nn.ReLU())
            last_dim = size
        layers_std.append(nn.Linear(last_dim, 1))  # 输出 Q std
        self.std_net = nn.Sequential(*layers_std)

        self.min_log_std = -5.0  # 防止除零
        self.max_log_std = 2.0

    def forward(self, obs, act):
        # 1. 状态编码
        ego = obs[:, :self.ego_dim]
        ref = obs[:, self.ego_dim: self.ego_dim + self.ref_dim]
        obst = obs[:, self.ego_dim + self.ref_dim: self.ego_dim + self.ref_dim + self.obs_dim]
        prompts = obs[:, self.ego_dim + self.ref_dim + self.obs_dim:]

        ego_feat = self.ego_encoder(ego)
        ref_feat = self.ref_encoder(ref)
        obs_feat = self.obstacle_encoder(obst)

        prompts = prompts.reshape(-1, self.num_prompts, self.points_per_prompt, 2)
        prompt_feats = []
        for i in range(self.num_prompts):
            # 注意：这里的 prompt_encoder 实现需要与你实际的维度匹配
            p_feat = self.prompt_encoder(prompts[:, i, :, :])
            if p_feat.dim() > 2: p_feat = p_feat.mean(dim=1)
            prompt_feats.append(p_feat)
        prompt_combined = torch.cat(prompt_feats, dim=-1)

        state_feat = torch.cat([ego_feat, ref_feat, obs_feat, prompt_combined], dim=-1)

        # 2. 拼接动作
        combined = torch.cat([state_feat, act], dim=-1)

        # 3. 输出
        q_mean = self.mean_net(combined)
        q_log_std = self.std_net(combined)

        # 限制范围
        q_log_std = torch.clamp(q_log_std, self.min_log_std, self.max_log_std)
        q_std = q_log_std.exp()

        # 返回拼接的 (mean, std) 以适配你的算法代码
        return torch.cat([q_mean, q_std], dim=-1)


class ObstacleEncoder(nn.Module):
    def __init__(self, obstacle_dim=4, hidden_dim=64, nhead=4):
        super().__init__()
        self.obstacle_proj = nn.Linear(obstacle_dim, hidden_dim)
        encoder_layer = TransformerEncoderLayer(hidden_dim, nhead, hidden_dim * 4)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=2)

        # 注意力池化层
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, obstacles):
        """
        obstacles: (batch_size, num_obstacles, obstacle_dim)
        返回: (batch_size, hidden_dim)
        """
        # 投影到隐藏空间
        x = self.obstacle_proj(obstacles)  # (B, N, D)

        # Transformer处理
        x = x.transpose(0, 1)  # (N, B, D)
        x = self.transformer(x)
        x = x.transpose(0, 1)  # (B, N, D)

        # 注意力池化
        attn_weights = F.softmax(self.attention_pool(x), dim=1)  # (B, N, 1)
        pooled = torch.sum(attn_weights * x, dim=1)  # (B, D)

        return pooled

# Define MLP function
def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


# Count parameter number of MLP
def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


# Deterministic policy
class DetermPolicy(nn.Module, Action_Distribution):
    """
    Approximated function of deterministic policy.
    Input: observation.
    Output: action.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]

        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(
            pi_sizes,
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]))
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]))
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(
            self.pi(obs)
        ) + (self.act_high_lim + self.act_low_lim) / 2
        return action


class FiniteHorizonPolicy(nn.Module, Action_Distribution):
    """
    Approximated function of deterministic policy for finite-horizon.
    Input: observation, time step.
    Output: action.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"] + 1
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]

        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(
            pi_sizes,
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]))
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]))
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs, virtual_t=1):
        virtual_t = virtual_t * torch.ones(
            size=[obs.shape[0], 1], dtype=torch.float32, device=obs.device
        )
        expand_obs = torch.cat((obs, virtual_t), 1)
        action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(
            self.pi(expand_obs)
        ) + (self.act_high_lim + self.act_low_lim) / 2
        return action


class FiniteHorizonFullPolicy(nn.Module, Action_Distribution):
    """
    Approximated function of deterministic policy for finite-horizon.
    Input: observation, time step.
    Output: action.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        self.act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.pre_horizon = kwargs["pre_horizon"]
        pi_sizes = [obs_dim] + list(hidden_sizes) + [self.act_dim * self.pre_horizon]

        self.pi = mlp(
            pi_sizes,
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]).float())
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]).float())
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        return self.forward_all_policy(obs)[0, :, :]

    def forward_all_policy(self, obs):
        actions = self.pi(obs).reshape(obs.shape[0], self.pre_horizon, self.act_dim)
        action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(actions) \
                 + (self.act_high_lim + self.act_low_lim) / 2
        return action

# Stochastic Policy
class StochaPolicy(nn.Module, Action_Distribution):
    """
    Approximated function of stochastic policy.
    Input: observation.
    Output: parameters of action distribution.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.std_type = kwargs["std_type"]

        # mean and log_std are calculated by different MLP
        if self.std_type == "mlp_separated":
            pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
            self.mean = mlp(
                pi_sizes,
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
            )
            self.log_std = mlp(
                pi_sizes,
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
            )
        # mean and log_std are calculated by same MLP
        elif self.std_type == "mlp_shared":
            pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim * 2]
            self.policy = mlp(
                pi_sizes,
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
            )
        # mean is calculated by MLP, and log_std is learnable parameter
        elif self.std_type == "parameter":
            pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
            self.mean = mlp(
                pi_sizes,
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
            )
            self.log_std = nn.Parameter(-0.5*torch.ones(1, act_dim))

        self.min_log_std = kwargs["min_log_std"]
        self.max_log_std = kwargs["max_log_std"]
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]))
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]))
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        if self.std_type == "mlp_separated":
            action_mean = self.mean(obs)
            action_std = torch.clamp(
                self.log_std(obs), self.min_log_std, self.max_log_std
            ).exp()
        elif self.std_type == "mlp_shared":
            logits = self.policy(obs)
            action_mean, action_log_std = torch.chunk(
                logits, chunks=2, dim=-1
            )  # output the mean
            action_std = torch.clamp(
                action_log_std, self.min_log_std, self.max_log_std
            ).exp()
        elif self.std_type == "parameter":
            action_mean = self.mean(obs)
            action_log_std = self.log_std + torch.zeros_like(action_mean)
            action_std = torch.clamp(
                action_log_std, self.min_log_std, self.max_log_std
            ).exp()

        return torch.cat((action_mean, action_std), dim=-1)


class PromptEncoder(nn.Module):
    """专门用于编码引导轨迹 (Prompts)，增加了 BatchNorm 防止梯度爆炸"""

    def __init__(self, input_dim=2, hidden_dim=32, output_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            # 第一层卷积 + BN + ReLU
            nn.Conv1d(input_dim, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),  # 【关键修复】归一化中间特征
            nn.ReLU(),

            # 第二层卷积 + BN + ReLU
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),  # 【关键修复】
            nn.ReLU(),

            # 池化
            nn.AdaptiveMaxPool1d(1)
        )
        self.fc = nn.Linear(32, output_dim)

    def forward(self, x):
        # x shape: (Batch, Points, 2)
        # 手动归一化输入：假设视野约为 30m，除以 30 将数值缩放到 [0, 1] 附近
        # 这比依赖 BN 更稳健，特别是对于坐标数据
        x = x / 30.0

        x = x.transpose(1, 2)  # -> (Batch, 2, Points)
        feat = self.net(x)
        feat = feat.squeeze(-1)
        return self.fc(feat)

# Stochastic Policy
class EncodingStochaPolicy(nn.Module, Action_Distribution):
    """
    Approximated function of stochastic policy.
    Input: observation.
    Output: parameters of action distribution.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.std_type = kwargs["std_type"]
        # 观测分解维度
        self.ego_dim = 8
        self.ref_dim = 180  # 30*6
        self.obs_dim = 16  # 3*4

        # 1. 参考轨迹编码器 (LSTM)
        self.ref_encoder = nn.LSTM(
            input_size=6,  # 每个参考点的维度
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        # 2. 障碍物编码器
        self.obstacle_encoder = ObstacleEncoder(
            obstacle_dim=4,
            hidden_dim=64
        )

        # 3. 自车状态编码
        self.ego_encoder = nn.Sequential(
            nn.Linear(self.ego_dim, 64),
            nn.ReLU()
        )

        # 4. 特征融合后的策略头
        total_feat_dim = 64 * 3  # ego + ref + obstacle

        # mean和std的网络构建
        if self.std_type == "mlp_separated":
            self.mean = mlp(
                [total_feat_dim] + list(hidden_sizes) + [act_dim],
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"])
            )
            self.log_std = mlp(
                [total_feat_dim] + list(hidden_sizes) + [act_dim],
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"])
            )
        elif self.std_type == "mlp_shared":
            self.policy = mlp(
                [total_feat_dim] + list(hidden_sizes) + [act_dim * 2],
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"])
            )
        elif self.std_type == "parameter":
            self.mean = mlp(
                [total_feat_dim] + list(hidden_sizes) + [act_dim],
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"])
            )
            self.log_std = nn.Parameter(-0.5 * torch.ones(1, act_dim))

        self.min_log_std = kwargs["min_log_std"]
        self.max_log_std = kwargs["max_log_std"]
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]))
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]))
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        # 分解观测
        ego_state = obs[:, :self.ego_dim]  # (B, 8)
        ref_traj = obs[:, self.ego_dim:self.ego_dim + self.ref_dim]  # (B, 180)
        obstacles = obs[:, -self.obs_dim:]  # (B, 16)

        # 1. 编码参考轨迹
        ref_traj = ref_traj.view(-1, 30, 6)  # (B, 30, 6)
        ref_feat, _ = self.ref_encoder(ref_traj)  # (B, 30, 64)
        ref_feat = ref_feat[:, -1, :]  # 取最后时刻的特征 (B, 64)

        # 2. 编码障碍物
        obstacles = obstacles.view(-1, 4, 4)  # (B, 4, 4)
        obs_feat = self.obstacle_encoder(obstacles)  # (B, 64)

        # 3. 编码自车状态
        ego_feat = self.ego_encoder(ego_state)  # (B, 64)

        # 4. 特征融合
        combined = torch.cat([ego_feat, ref_feat, obs_feat], dim=-1)  # (B, 192)
        # 5. 策略输出
        if self.std_type == "mlp_separated":
            action_mean = self.mean(combined)
            action_std = torch.clamp(
                self.log_std(combined), self.min_log_std, self.max_log_std
            ).exp()
        elif self.std_type == "mlp_shared":
            logits = self.policy(combined)
            action_mean, action_log_std = torch.chunk(logits, 2, dim=-1)
            action_std = torch.clamp(
                action_log_std, self.min_log_std, self.max_log_std
            ).exp()
        elif self.std_type == "parameter":
            action_mean = self.mean(combined)
            action_log_std = self.log_std + torch.zeros_like(action_mean)
            action_std = torch.clamp(
                action_log_std, self.min_log_std, self.max_log_std
            ).exp()

        return torch.cat((action_mean, action_std), dim=-1)

class EncodingStochaPolicy2(nn.Module, Action_Distribution):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.std_type = kwargs["std_type"]

        # === 维度定义 ===
        self.ego_dim = 6
        self.ref_dim = 20 * 6
        self.obs_dim = 3 * 18
        # 注意：这里不需要定义 prompt_dim，因为我们后面是按 shape view 的

        self.num_prompts = 3
        self.points_per_prompt = 20
        # 计算剩下的维度给 Prompt
        self.known_dim = self.ego_dim + self.ref_dim + self.obs_dim

        # 1. Encoders (保持维度匹配)
        self.ref_encoder = nn.Sequential(
            nn.Linear(self.ref_dim, 128), nn.ReLU(), nn.Linear(128, 64)
        )
        self.obstacle_encoder = nn.Sequential(
            nn.Linear(self.obs_dim, 128), nn.ReLU(), nn.Linear(128, 64)
        )
        self.prompt_encoder = PromptEncoder(input_dim=2, output_dim=32)
        self.ego_encoder = nn.Sequential(
            nn.Linear(self.ego_dim, 64), nn.ReLU()
        )

        # 5. 融合维度: 64*3 + 32*3 = 288
        total_feat_dim = 64 + 64 + 64 + (32 * 3)

        # 策略头
        # 使用正交初始化 (Orthogonal Initialization) 有助于 RL 收敛
        self.mean_net = mlp([total_feat_dim] + list(hidden_sizes) + [act_dim],
                            get_activation_func(kwargs["hidden_activation"]),
                            get_activation_func(kwargs["output_activation"]))

        self.log_std_net = mlp([total_feat_dim] + list(hidden_sizes) + [act_dim],
                               get_activation_func(kwargs["hidden_activation"]),
                               get_activation_func(kwargs["output_activation"]))

        self.min_log_std = kwargs["min_log_std"]
        self.max_log_std = kwargs["max_log_std"]
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]))
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]))
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        # 1. 鲁棒的切分逻辑
        ego = obs[:, :self.ego_dim]
        ref = obs[:, self.ego_dim: self.ego_dim + self.ref_dim]
        obst = obs[:, self.ego_dim + self.ref_dim: self.ego_dim + self.ref_dim + self.obs_dim]
        prompts = obs[:, self.ego_dim + self.ref_dim + self.obs_dim:]

        # 2. 编码
        ego_feat = self.ego_encoder(ego)
        ref_feat = self.ref_encoder(ref)
        obs_feat = self.obstacle_encoder(obst)

        # 处理 Prompts
        prompts = prompts.reshape(-1, self.num_prompts, self.points_per_prompt, 2)
        prompt_feats = []
        for i in range(self.num_prompts):
            p_feat = self.prompt_encoder(prompts[:, i, :, :])
            prompt_feats.append(p_feat)
        prompt_combined = torch.cat(prompt_feats, dim=-1)

        # 3. 融合
        combined = torch.cat([ego_feat, ref_feat, obs_feat, prompt_combined], dim=-1)

        # 4. 输出
        action_mean = self.mean_net(combined)
        action_log_std = self.log_std_net(combined)

        # 【关键修复】严格限制 Log Std 的范围，防止 NaN
        action_log_std = torch.clamp(action_log_std, self.min_log_std, self.max_log_std)
        action_std = action_log_std.exp()

        return torch.cat((action_mean, action_std), dim=-1)


class ActionValue(nn.Module, Action_Distribution):
    """
    Approximated function of action-value function.
    Input: observation, action.
    Output: action-value.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.q = mlp(
            [obs_dim + act_dim] + list(hidden_sizes) + [1],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)


class ActionValueDis(nn.Module, Action_Distribution):
    """
    Approximated function of action-value function for discrete action space.
    Input: observation.
    Output: action-value for all action.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_num = kwargs["act_num"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.q = mlp(
            [obs_dim] + list(hidden_sizes) + [act_num],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        return self.q(obs)


class ActionValueDistri(nn.Module):
    """
    Approximated function of distributed action-value function.
    Input: observation.
    Output: parameters of action-value distribution.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.q = mlp(
            [obs_dim + act_dim] + list(hidden_sizes) + [2],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        if "min_log_std"  in kwargs or "max_log_std" in kwargs:
            warnings.warn("min_log_std and max_log_std are deprecated in ActionValueDistri.")

    def forward(self, obs, act):
        logits = self.q(torch.cat([obs, act], dim=-1))
        value_mean, value_std = torch.chunk(logits, chunks=2, dim=-1)
        value_log_std = torch.nn.functional.softplus(value_std) 
        
        return torch.cat((value_mean, value_log_std), dim=-1)


class StochaPolicyDis(ActionValueDis, Action_Distribution):
    """
    Approximated function of stochastic policy for discrete action space.
    Input: observation.
    Output: parameters of action distribution.
    """

    pass


class StateValue(nn.Module, Action_Distribution):
    """
    Approximated function of state-value function.
    Input: observation, action.
    Output: state-value.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.v = mlp(
            [obs_dim] + list(hidden_sizes) + [1],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        v = self.v(obs)
        return torch.squeeze(v, -1)


# Diffusion Policy
class DiffusionMLP(nn.Module):
    """
    MLP Model
    """

    def __init__(self, state_dim, action_dim, hidden_dim, t_dim=16):#, device
        super(DiffusionMLP, self).__init__()
        # self.device = device
        self.t_dim = t_dim
        self.a_dim = action_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = state_dim + action_dim + t_dim
        self.mid_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )

        self.final_layer = nn.Linear(hidden_dim, action_dim)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, time, state, **kwargs):
        t = self.time_mlp(time)
        # x = x.to(self.device)
        # t = t.to(self.device)
        # state = state.to(self.device)
        x = torch.cat([x, t, state], dim=1)
        x = self.mid_layer(x)

        return self.final_layer(x)

class DiffusionPolicy(nn.Module,Action_Distribution):
    """
    Approximated function of stochastic policy.
    Input: observation.
    Output: parameters of action distribution.
    """

    def __init__(
        self,
        beta_schedule="linear",
        loss_type="l2",
        clip_denoised=True,
        predict_epsilon=True,
        **kwargs
    ):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.std_type = kwargs["std_type"]
        self.device = torch.device(kwargs["device"])
        self.action_distribution_cls = kwargs["action_distribution_cls"]

        self.min_log_std = kwargs["min_log_std"]
        self.max_log_std = kwargs["max_log_std"]
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]))
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]))

        self.w = kwargs["policy_w"]
        self.T = kwargs["policy_T"]
        self.state_dim = kwargs["obs_dim"]
        self.action_dim = kwargs["act_dim"]
        self.max_action = kwargs["act_high_lim"][0]
        self.model = DiffusionMLP(obs_dim, act_dim, hidden_sizes, self.device).to(self.device)

        if beta_schedule == "linear":
            betas = linear_beta_schedule(self.T)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(self.T)
        elif beta_schedule == "vp":
            betas = vp_beta_schedule(self.T)
        betas = betas.to(self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, device=self.device), alphas_cumprod[:-1]])

        self.n_timesteps = self.T
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        self.loss_fn = Losses[loss_type]()

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        """
        if self.predict_epsilon, model output is (scaled) noise;
        otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, s, **kwargs):
        eps = self.model(x, t, s)
        x_recon = self.predict_start_from_noise(x, t=t, noise=eps)

        if self.clip_denoised:
            x_recon.clamp_(-self.max_action, self.max_action)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, x, t, s, **kwargs):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, s=s, **kwargs
        )
        noise = 0.5 * torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_sample_loop(
        self, state, shape, verbose=False, return_diffusion=False, **kwargs
    ):
        batch_size = shape[0]
        # x = torch.full(shape, 1.0, device=self.device, requires_grad=True)
        x = 0.5 * torch.randn(shape, device=self.device, requires_grad=True)

        if return_diffusion:
            diffusion = [x]

        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full(
                (batch_size,), i, device=self.device, dtype=torch.long
            )
            x = self.p_sample(x, timesteps, state, **kwargs)

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    def p_sample_approximate(self, state, action, **kwargs):
        # EDP sampling, one step to approximate the action
        batch_size = len(action)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=self.device).long()
        x_noisy = self.q_sample(x_start=action, t=t)
        if torch.allclose(
            kwargs["cemb"],
            torch.zeros_like(kwargs["cemb"]),
            rtol=1e-05,
            atol=1e-08,
            equal_nan=False,
        ):
            x_approx = self.predict_start_from_noise(
                x_t=x_noisy, t=t, noise=self.model(x_noisy, t, state)
            )
        else:
            cemb_shape = kwargs["cemb"].shape
            pred_eps_cond = self.model(x_noisy, t, state, **kwargs)
            kwargs["cemb"] = torch.zeros(cemb_shape, device=self.device)
            pred_eps_uncond = self.model(x_noisy, t, state, **kwargs)
            eps = pred_eps_uncond + self.w * (pred_eps_cond - pred_eps_uncond)
            x_approx = self.predict_start_from_noise(x_t=x_noisy, t=t, noise=eps)

        return x_approx

    def sample(self, state, *args, **kwargs):
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        if "edp" in kwargs and kwargs["edp"] == True:
            assert "action" in kwargs
            action = self.p_sample_approximate(
                state=state, shape=shape, *args, **kwargs
            )
        else:
            action = self.p_sample_loop(state, shape, *args, **kwargs)
        return action.clamp_(-1, 1)

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = 0.5 * torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, state, t, weights=1.0):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_recon = self.model(x_noisy, t, state)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss = self.loss_fn(x_recon, noise, weights)
        else:
            loss = self.loss_fn(x_recon, x_start, weights)

        return loss

    def loss(self, x, state, weights=1.0):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=self.device).long()
        return self.p_losses(x, state, t, weights)

    def forward(self, obs, **kwargs):
        obs = obs.to(self.device)
        return self.sample(obs, **kwargs)

class DiffusionPolicyeasy(nn.Module,Action_Distribution):
    """
    Approximated function of stochastic policy.
    Input: observation.
    Output: parameters of action distribution.
    """

    def __init__(
        self,
        beta_schedule="vp",
        loss_type="l2",
        clip_denoised=True,
        predict_epsilon=True,
        **kwargs
    ):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.std_type = kwargs["std_type"]
        # self.device = torch.device(kwargs["device"])
        self.action_distribution_cls = kwargs["action_distribution_cls"]
        self.min_log_std = kwargs["min_log_std"]
        self.max_log_std = kwargs["max_log_std"]
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]))
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]))

        # self.w = kwargs["policy_w"]
        self.T = kwargs["T"]
        self.state_dim = kwargs["obs_dim"]
        self.action_dim = kwargs["act_dim"]
        self.max_action = kwargs["act_high_lim"][0]
        # self.model = DiffusionMLP(obs_dim, act_dim, hidden_sizes, self.device).to(self.device)
        self.model = DiffusionMLP(obs_dim, act_dim, hidden_sizes)
        if beta_schedule == "linear":
            betas = linear_beta_schedule(self.T)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(self.T)
        elif beta_schedule == "vp":
            betas = vp_beta_schedule(self.T)
        # betas = betas.to(self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])#, device=self.device

        self.n_timesteps = self.T
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        self.loss_fn = Losses[loss_type]()
    # ------------------------------------------ sampling ------------------------------------------#
    def sample(self, state, *args, **kwargs):
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        action = self.p_sample_loop(state, shape, *args, **kwargs)
        return action.clamp_(-1, 1)

    def p_sample_loop(
        self, state, shape, verbose=False, return_diffusion=False, **kwargs
    ):
        batch_size = shape[0]
        # x = torch.full(shape, -1.0, device=self.device, requires_grad=True)
        x = torch.zeros(shape, requires_grad=True)#, device=self.device
        # x = 0.5 * torch.randn(shape, device=self.device, requires_grad=True)
        if return_diffusion:
            diffusion = [x]
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full(
                (batch_size,), i, dtype=torch.long
            )#, device=self.device
            # device = x.device
            eps = self.model(x, timesteps, state)
            x_0_pred = extract(self.sqrt_recip_alphas_cumprod, timesteps, x.shape) * x- extract(self.sqrt_recipm1_alphas_cumprod, timesteps, x.shape) * eps
            x_0_pred = x_0_pred.clamp_(-self.max_action, self.max_action)
            x = (
                extract(self.posterior_mean_coef1, timesteps, x.shape) * x_0_pred
                + extract(self.posterior_mean_coef2, timesteps, x.shape) * x
            )
            x = x.clamp(-1, 1)
            if return_diffusion:
                diffusion.append(x)
        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    def forward(self, obs, **kwargs):
        # obs = obs.to(self.device)
        return self.sample(obs, **kwargs)

# class doublemlpDiffusionPolicy(nn.Module,Action_Distribution):
#     """
#     Approximated function of stochastic policy.
#     Input: observation.
#     Output: parameters of action distribution.
#     """
#
#     def __init__(
#         self,
#         beta_schedule="linear",
#         loss_type="l2",
#         clip_denoised=True,
#         predict_epsilon=True,
#         **kwargs
#     ):
#         super().__init__()
#         obs_dim = kwargs["obs_dim"]
#         act_dim = kwargs["act_dim"]
#         hidden_sizes = kwargs["hidden_sizes"]
#         self.std_type = kwargs["std_type"]
#         self.device = torch.device(kwargs["device"])
#         self.action_distribution_cls = kwargs["action_distribution_cls"]
#
#         self.min_log_std = kwargs["min_log_std"]
#         self.max_log_std = kwargs["max_log_std"]
#         self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]))
#         self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]))
#
#         self.w = kwargs["policy_w"]
#         self.T = kwargs["policy_T"]
#         self.state_dim = kwargs["obs_dim"]
#         self.action_dim = kwargs["act_dim"]
#         self.max_action = kwargs["act_high_lim"][0]
#         self.model = DiffusionMLP(obs_dim, act_dim, hidden_sizes, self.device).to(self.device)
#         #########################################################################################
#         pi_sizes = [obs_dim] + list([256, 256]) + [act_dim * 2]
#         self.policy = mlp(
#             pi_sizes,
#             get_activation_func("gelu"),
#             get_activation_func(kwargs["output_activation"]),
#         ).to(self.device)
#         #########################################################################################
#
#         if beta_schedule == "linear":
#             betas = linear_beta_schedule(self.T)
#         elif beta_schedule == "cosine":
#             betas = cosine_beta_schedule(self.T)
#         elif beta_schedule == "vp":
#             betas = vp_beta_schedule(self.T)
#         betas = betas.to(self.device)
#         alphas = 1.0 - betas
#         alphas_cumprod = torch.cumprod(alphas, axis=0)
#         alphas_cumprod_prev = torch.cat([torch.ones(1, device=self.device), alphas_cumprod[:-1]])
#
#         self.n_timesteps = self.T
#         self.clip_denoised = clip_denoised
#         self.predict_epsilon = predict_epsilon
#
#         self.register_buffer("betas", betas)
#         self.register_buffer("alphas" , alphas)
#         self.register_buffer("alphas_cumprod", alphas_cumprod)
#         self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
#
#         # calculations for diffusion q(x_t | x_{t-1}) and others
#         self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
#         self.register_buffer(
#             "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
#         )
#         self.register_buffer(
#             "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
#         )
#         self.register_buffer(
#             "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
#         )
#         self.register_buffer(
#             "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
#         )
#
#         # calculations for posterior q(x_{t-1} | x_t, x_0)
#         posterior_variance = (
#             betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
#         )
#         self.register_buffer("posterior_variance", posterior_variance)
#
#         ## log calculation clipped because the posterior variance
#         ## is 0 at the beginning of the diffusion chain
#         self.register_buffer(
#             "posterior_log_variance_clipped",
#             torch.log(torch.clamp(posterior_variance, min=1e-20)),
#         )
#         self.register_buffer(
#             "posterior_mean_coef1",
#             betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
#         )
#         self.register_buffer(
#             "posterior_mean_coef2",
#             (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
#         )
#
#         self.loss_fn = Losses[loss_type]()
#     # ------------------------------------------ sampling ------------------------------------------#
#     def sample(self, state, *args, **kwargs):
#         batch_size = state.shape[0]
#         shape = (batch_size, self.action_dim)
#         action = self.p_sample_loop(state, shape, *args, **kwargs)
#         return action.clamp_(-1, 1)
#
#     def p_sample_loop(
#         self, state, shape, verbose=False, return_diffusion=False, **kwargs
#     ):
#         batch_size = shape[0]
#         logits = self.policy(state)
#         action_mean, action_log_std = torch.chunk(logits, chunks=2, dim=-1)
#         # output the mean
#         action_std = torch.clamp(action_log_std, self.min_log_std, self.max_log_std).exp()
#         epsilon = torch.randn_like(action_mean)
#         x = action_mean + action_std * epsilon
#         # x = 0.5 * torch.randn(shape, device=self.device, requires_grad=True)
#         if return_diffusion:
#             diffusion = [x]
#         for i in reversed(range(0, self.n_timesteps)):
#             timesteps = torch.full(
#                 (batch_size,), i, device=self.device, dtype=torch.long
#             )
#             device = x.device
#             eps = self.model(x, timesteps, state)
#             x_0_pred = extract(self.sqrt_recip_alphas_cumprod, timesteps, x.shape) * x- extract(self.sqrt_recipm1_alphas_cumprod, timesteps, x.shape) * eps
#             x_0_pred = x_0_pred.clamp_(-self.max_action, self.max_action)
#             x = (
#                 extract(self.posterior_mean_coef1, timesteps, x.shape) * x_0_pred
#                 + extract(self.posterior_mean_coef2, timesteps, x.shape) * x
#             )
#             x = x.clamp(-1, 1)
#             if return_diffusion:
#                 diffusion.append(x)
#         if return_diffusion:
#             return x, torch.stack(diffusion, dim=1)
#         else:
#             return x
#
#     def forward(self, obs, **kwargs):
#         obs = obs.to(self.device)
#         return self.sample(obs, **kwargs)