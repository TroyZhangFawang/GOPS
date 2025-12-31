# gops/algorithm/dsact_diffusion.py

__all__ = ["ApproxContainer", "DSACTDiffusion"]

import time
from copy import deepcopy
from typing import Tuple, Dict
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np

from gops.algorithm.base import AlgorithmBase, ApprBase
from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.utils.tensorboard_setup import tb_tags
from gops.utils.gops_typing import DataDict
from gops.utils.common_utils import get_apprfunc_dict


# ==========================================
# 辅助类：Diffusion 采样结果包装
# ==========================================
class ActionDistResult:
    def __init__(self, action):
        self.action = action

    def sample(self):
        # 返回动作和假的 logp
        return self.action, torch.zeros((self.action.shape[0],), device=self.action.device)


# ==========================================
# Diffusion 数学工具类 (改为 nn.Module 以支持自动设备管理)
# ==========================================
class DiffusionScheduler(nn.Module):
    def __init__(self, num_steps=100, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.num_steps = num_steps

        # 使用 register_buffer 注册常量，这样它们会自动跟随 .to(device)
        betas = torch.linspace(beta_start, beta_end, num_steps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod)

    def add_noise(self, x_start, noise, t):
        # 确保 x_start 和 internal buffers 在同一设备
        # t 是索引，自动适配
        sqrt_alpha = self.sqrt_alphas_cumprod[t].reshape(-1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1)
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

    def sample_timesteps(self, batch_size):
        # 使用 self.betas.device 获取当前设备
        return torch.randint(0, self.num_steps, (batch_size,), device=self.betas.device).long()


# ==========================================
# 辅助类：Diffusion Policy 包装器
# ==========================================
class DiffusionPolicyWrapper(nn.Module):
    def __init__(self, mlp, scheduler, act_dim, act_max=1.0, act_min=-1.0):
        super().__init__()
        self.mlp = mlp
        self.scheduler = scheduler  # 引用传入的 scheduler
        self.act_dim = act_dim
        self.act_max = act_max
        self.act_min = act_min

    def forward(self, obs, act=None, t=None):
        if act is not None and t is not None:
            # 【训练模式】
            return self.mlp(obs, act, t)
        else:
            # 【推理模式】
            return self.sample_action(obs)

    def sample_action(self, obs):
        device = obs.device
        batch_size = obs.shape[0]

        # 1. 初始化纯噪声
        x = torch.randn((batch_size, self.act_dim), device=device)

        # 2. 逆向去噪循环
        for i in reversed(range(self.scheduler.num_steps)):
            t_tensor = torch.full((batch_size,), i, device=device, dtype=torch.long)

            # 预测噪声
            noise_pred = self.mlp(obs, x, t_tensor)

            # 获取当前步的系数 (它们现在会自动在正确的设备上)
            alpha = self.scheduler.alphas[i]
            alpha_hat = self.scheduler.alphas_cumprod[i]
            beta = self.scheduler.betas[i]

            # mean
            x = (1 / torch.sqrt(alpha)) * (
                    x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * noise_pred
            )

            # add noise
            if i > 0:
                noise = torch.randn_like(x)
                x = x + torch.sqrt(beta) * noise

        return x.clamp(self.act_min, self.act_max)


# ==========================================
# ApproxContainer (网络容器)
# ==========================================
class ApproxContainer(ApprBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # 1. 提取 Diffusion 参数
        self.diffusion_steps = kwargs.get("diffusion_steps", 20)

        # 【关键修复】不再强制指定 device，让它默认为 CPU
        # 当 Trainer 调用 .cuda() 时，它会自动转到 GPU
        self.scheduler = DiffusionScheduler(num_steps=self.diffusion_steps)

        # 2. 构建 Policy 网络
        policy_args = get_apprfunc_dict("policy", **kwargs)
        mlp_policy = create_apprfunc(**policy_args)

        act_dim = kwargs["action_dim"]
        act_high = kwargs.get("action_high_limit", np.array([1.0]))
        act_low = kwargs.get("action_low_limit", np.array([-1.0]))
        act_max = float(np.max(act_high))
        act_min = float(np.min(act_low))

        # Wrapper 包装
        self.policy = DiffusionPolicyWrapper(
            mlp_policy, self.scheduler, act_dim, act_max, act_min
        )
        self.policy_target = deepcopy(self.policy)

        # 3. 构建 Critic 网络
        q_args = get_apprfunc_dict("q", **kwargs)
        self.q1 = create_apprfunc(**q_args)
        self.q2 = create_apprfunc(**q_args)
        self.q1_target = deepcopy(self.q1)
        self.q2_target = deepcopy(self.q2)

        # 4. 优化器
        # 注意：这里我们只优化 MLP 部分的参数
        self.policy_optimizer = Adam(self.policy.mlp.parameters(), lr=kwargs["policy_learning_rate"])
        self.q1_optimizer = Adam(self.q1.parameters(), lr=kwargs["q_learning_rate"])
        self.q2_optimizer = Adam(self.q2.parameters(), lr=kwargs["q_learning_rate"])

        self.tau = kwargs["tau"]

    def create_action_distributions(self, logits):
        return ActionDistResult(logits)


# ==========================================
# DSACT_Diffusion (算法逻辑)
# ==========================================
class DSACTDiffusion(AlgorithmBase):
    def __init__(self, index=0, **kwargs):
        super().__init__(index, **kwargs)
        self.networks = ApproxContainer(**kwargs)
        self.gamma = kwargs["gamma"]
        self.delay_update = kwargs["delay_update"]

    @property
    def adjustable_parameters(self):
        return ("gamma", "tau", "delay_update")

    def local_update(self, data: DataDict, iteration: int) -> dict:
        tb_info = self.__compute_gradient(data, iteration)
        self.__update(iteration)
        return tb_info

    def __update(self, iteration: int):
        self.networks.q1_optimizer.step()
        self.networks.q2_optimizer.step()

        if iteration % self.delay_update == 0:
            self.networks.policy_optimizer.step()
            with torch.no_grad():
                polyak = 1 - self.networks.tau
                for p, p_targ in zip(self.networks.q1.parameters(), self.networks.q1_target.parameters()):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)
                for p, p_targ in zip(self.networks.q2.parameters(), self.networks.q2_target.parameters()):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)
                # 更新 policy target (wrapper 里的 mlp)
                for p, p_targ in zip(self.networks.policy.mlp.parameters(),
                                     self.networks.policy_target.mlp.parameters()):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

    def __compute_gradient(self, data: DataDict, iteration: int):
        self.networks.q1_optimizer.zero_grad()
        self.networks.q2_optimizer.zero_grad()
        self.networks.policy_optimizer.zero_grad()

        loss_q, loss_q_info = self.__compute_loss_critic(data)
        loss_q.backward()
        tb_info = loss_q_info

        if iteration % self.delay_update == 0:
            loss_policy, loss_policy_info = self.__compute_loss_policy(data)
            loss_policy.backward()
            tb_info.update(loss_policy_info)

        return tb_info

    def __compute_loss_critic(self, data: DataDict):
        obs, act, rew, obs_next, done = data["obs"], data["act"], data["rew"], data["obs2"], data["done"]
        q1 = self.networks.q1(obs, act)
        q2 = self.networks.q2(obs, act)

        with torch.no_grad():
            next_act = self.networks.policy_target(obs_next)
            q1_next = self.networks.q1_target(obs_next, next_act)
            q2_next = self.networks.q2_target(obs_next, next_act)
            q_next = torch.min(q1_next, q2_next)
            target_q = rew + self.gamma * (1 - done) * q_next

        loss_q = nn.MSELoss()(q1, target_q) + nn.MSELoss()(q2, target_q)
        return loss_q, {tb_tags["loss_critic"]: loss_q.item(), "q1_val": q1.mean().item()}

    def __compute_loss_policy(self, data: DataDict):
        obs, act_real = data["obs"], data["act"]

        with torch.no_grad():
            q1 = self.networks.q1(obs, act_real)
            q2 = self.networks.q2(obs, act_real)
            adv = torch.min(q1, q2) - torch.min(q1, q2).mean()
            weights = torch.exp(adv / 3.0).clamp(max=20.0)

        t = self.networks.scheduler.sample_timesteps(obs.shape[0])
        noise = torch.randn_like(act_real)
        act_noisy = self.networks.scheduler.add_noise(act_real, noise, t)

        noise_pred = self.networks.policy(obs, act_noisy, t)

        loss_mse = nn.MSELoss(reduction='none')(noise_pred, noise).mean(dim=1)
        loss_diff = (loss_mse * weights).mean()

        return loss_diff, {tb_tags["loss_actor"]: loss_diff.item()}

    def get_action(self, obs):
        with torch.no_grad():
            if obs.dim() == 1: obs = obs.unsqueeze(0)
            action = self.networks.policy(obs)
            return action.squeeze(0)