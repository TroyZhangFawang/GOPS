# 文件路径: gops/apprfunc/diffusion.py (原 apprfunc_diffusion.py)

# 【关键】修改 __all__ 为 ["mlp"]，这样自动注册时 name 就是 "mlp"
# 文件名是 "diffusion.py"，type 就是 "diffusion"
# 组合起来 ID 就是 "diffusion_mlp"，完美匹配你的参数
__all__ = ["mlp"]

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

        # 获取激活函数类并实例化
        act_func_class = get_activation_func(kwargs.get("hidden_activation", "relu"))

        # 时间步编码维度
        time_dim = hidden_sizes[0]
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            act_func_class(),
            nn.Linear(time_dim, time_dim),
        )

        # 主干网络构建
        layers = []
        input_dim = obs_dim + act_dim + time_dim

        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(act_func_class())
            input_dim = h

        self.mlp = nn.Sequential(*layers)

        # 输出层
        self.last_layer = nn.Linear(hidden_sizes[-1], act_dim)

    def forward(self, obs, act, t):
        if t.dim() == 0:
            t = t.unsqueeze(0).repeat(obs.shape[0])
        t_emb = self.time_mlp(t)
        x = torch.cat([obs, act, t_emb], dim=-1)
        feat = self.mlp(x)
        noise_pred = self.last_layer(feat)
        return noise_pred


# 【关键】建立别名 mlp，对应 __all__ 中的名字
mlp = DiffusionMLP