# 文件路径: gops/apprfunc/apprfunc_diffusion.py

__all__ = ["diffusion"]  # 告诉GOPS我们要注册的名字是 "diffusion"

import torch
import torch.nn as nn
import numpy as np
from gops.utils.common_utils import get_activation_func


# 1. 正弦位置编码
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# 2. Diffusion MLP Actor
class DiffusionMLP(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs.get("hidden_sizes", [256, 256, 256])
        hidden_activation = kwargs.get("hidden_activation", "relu")

        # 时间步编码维度 (与第一层隐藏层对齐)
        time_dim = hidden_sizes[0]
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            get_activation_func(hidden_activation),
            nn.Linear(time_dim, time_dim),
        )

        # 主干网络构建
        layers = []
        # 输入: Obs + NoisyAction + TimeEmb
        input_dim = obs_dim + act_dim + time_dim

        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(get_activation_func(hidden_activation))
            input_dim = h  # 下一层的输入

        self.mlp = nn.Sequential(*layers)

        # 输出层 (预测噪声，无激活函数)
        self.last_layer = nn.Linear(hidden_sizes[-1], act_dim)

    def forward(self, obs, act, t):
        """
        obs: [B, obs_dim]
        act: [B, act_dim] (Noisy Action)
        t:   [B] (Time steps)
        """
        # 1. 处理时间编码
        if t.dim() == 0:  # 如果 t 是标量
            t = t.unsqueeze(0).repeat(obs.shape[0])

        t_emb = self.time_mlp(t)

        # 2. 拼接输入
        x = torch.cat([obs, act, t_emb], dim=-1)

        # 3. 通过 MLP
        feat = self.mlp(x)

        # 4. 预测噪声
        noise_pred = self.last_layer(feat)
        return noise_pred


diffusion = DiffusionMLP