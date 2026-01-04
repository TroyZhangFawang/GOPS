
__all__ = ["ApproxContainer", "DSACTDiffusion2", "DsactDiffusion2"]

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
        return self.action, torch.zeros((self.action.shape[0],), device=self.action.device)
    def mode(self):
        return self.action

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
        # Diffusion Policy 部分
        self.diffusion_steps = kwargs.get("diffusion_steps", 20)
        self.scheduler = DiffusionScheduler(num_steps=self.diffusion_steps)
        policy_args = get_apprfunc_dict("policy", **kwargs)
        mlp_policy = create_apprfunc(**policy_args)

        act_dim = kwargs["action_dim"]
        act_high = kwargs.get("action_high_limit", np.array([1.0]))
        act_low = kwargs.get("action_low_limit", np.array([-1.0]))
        act_max = float(np.max(act_high))
        act_min = float(np.min(act_low))

        self.policy = DiffusionPolicyWrapper(mlp_policy, self.scheduler, act_dim, act_max, act_min)
        self.policy_target = deepcopy(self.policy)

        q_args = get_apprfunc_dict("value", **kwargs)

        self.q1 = create_apprfunc(**q_args)
        self.q2 = create_apprfunc(**q_args)
        self.q1_target = deepcopy(self.q1)
        self.q2_target = deepcopy(self.q2)

        self.policy_optimizer = Adam(self.policy.mlp.parameters(), lr=kwargs["policy_learning_rate"])
        self.q1_optimizer = Adam(self.q1.parameters(), lr=kwargs["q_learning_rate"])
        self.q2_optimizer = Adam(self.q2.parameters(), lr=kwargs["q_learning_rate"])

        self.tau = kwargs["tau"]

    def create_action_distributions(self, logits):
        return ActionDistResult(logits)


class DSACTDiffusion2(AlgorithmBase):
    def __init__(self, index=0, **kwargs):
        super().__init__(index, **kwargs)
        self.networks = ApproxContainer(**kwargs)
        self.gamma = kwargs["gamma"]
        self.delay_update = kwargs["delay_update"]

        # 【DSACT 特有参数】
        self.tau_b = kwargs.get("tau_b", 0.01)  # 用于 std 的软更新
        self.mean_std1 = None  # 动态记录 std 均值
        self.mean_std2 = None

    @property
    def adjustable_parameters(self):
        return ("gamma", "tau", "delay_update", "tau_b")

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
                for p, p_targ in zip(self.networks.policy.mlp.parameters(),
                                     self.networks.policy_target.mlp.parameters()):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

    def __compute_gradient(self, data: DataDict, iteration: int):
        self.networks.q1_optimizer.zero_grad()
        self.networks.q2_optimizer.zero_grad()
        self.networks.policy_optimizer.zero_grad()

        # 使用 DSACT 的 Critic Loss
        loss_q, q1, q2, std1, std2 = self.__compute_loss_q(data)
        loss_q.backward()

        tb_info = {
            tb_tags["loss_critic"]: loss_q.item(),
            "DDSACT/critic_avg_q1-RL iter": q1.item(),
            "DDSACT/critic_avg_q2-RL iter": q2.item(),
            "DDSACT/critic_avg_std1-RL iter": std1.item(),
            "DDSACT/critic_avg_std2-RL iter": std2.item(),
        }

        if iteration % self.delay_update == 0:
            loss_policy, loss_policy_info = self.__compute_loss_policy(data)
            loss_policy.backward()
            tb_info.update(loss_policy_info)

        return tb_info

    def __q_evaluate(self, obs, act, qnet):
        output = qnet(obs, act)
        # 1. 维度处理
        if output.dim() == 1:
            output = output.unsqueeze(-1)

        # 2. 维度检查
        if output.shape[-1] != 2:
            raise ValueError(
                f"Q Network output dim is {output.shape[-1]}, expected 2 (Mean + Std).\n"
                f"Please check your config: --q_func_name must be 'ActionValueDistri'!"
            )

        # 3. 解析 Mean 和 Std
        mean, std = torch.chunk(output, chunks=2, dim=-1)
        # 4. 稳健性处理：虽然 softplus 恒正，但为了防止数值过小导致除零，加上一个极小值
        std = torch.clamp(std, min=1e-6, max=100.0)
        # 5. DSACT 的采样逻辑
        normal = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(std))
        z = normal.sample()
        z = torch.clamp(z, -3, 3)
        q_sample = mean + torch.mul(z, std)
        return mean, std, q_sample

    def __compute_loss_q(self, data: DataDict):
        obs, act, rew, obs_next, done = data["obs"], data["act"], data["rew"], data["obs2"], data["done"]

        if rew.dim() == 1: rew = rew.unsqueeze(-1)
        if done.dim() == 1: done = done.unsqueeze(-1)

        # 1. 计算当前的 Q(s, a) 分布
        # 【修改】接收 3 个返回值
        q1_mean, q1_std, _ = self.__q_evaluate(obs, act, self.networks.q1)
        q2_mean, q2_std, _ = self.__q_evaluate(obs, act, self.networks.q2)

        # 2. 动量更新 std 均值
        with torch.no_grad():
            if self.mean_std1 is None:
                self.mean_std1 = torch.mean(q1_std)
            else:
                self.mean_std1 = (1 - self.tau_b) * self.mean_std1 + self.tau_b * torch.mean(q1_std)

            if self.mean_std2 is None:
                self.mean_std2 = torch.mean(q2_std)
            else:
                self.mean_std2 = (1 - self.tau_b) * self.mean_std2 + self.tau_b * torch.mean(q2_std)

        # 3. 计算 Target Q
        with torch.no_grad():
            next_act = self.networks.policy_target(obs_next)

            # 【修改】接收 3 个返回值，我们需要 q_sample 来计算 bound
            q1_next_mean, _, q1_next_sample = self.__q_evaluate(obs_next, next_act, self.networks.q1_target)
            q2_next_mean, _, q2_next_sample = self.__q_evaluate(obs_next, next_act, self.networks.q2_target)

            # (C) Min Q (保守估计)
            # 使用 Mean 进行 Min 操作
            q_next = torch.min(q1_next_mean, q2_next_mean)

            # 使用 Sample 进行 Bound 计算
            # 这里的逻辑是：如果 Q1 的均值更小，我们就用 Q1 的采样值作为未来的参考
            q_next_sample = torch.where(q1_next_mean < q2_next_mean, q1_next_sample, q2_next_sample)

            # (D) 计算 Target
            # 传入 q_next (均值) 和 q_next_sample (采样值)
            target_q1_mean, target_q1_bound = self.__compute_target_q(
                rew, done, q1_mean, self.mean_std1, q_next, q_next_sample
            )
            target_q2_mean, target_q2_bound = self.__compute_target_q(
                rew, done, q2_mean, self.mean_std2, q_next, q_next_sample
            )

        # 4. 计算 Loss
        bias = 0.1
        q1_loss = self.__dsact_loss_func(q1_mean, q1_std, target_q1_mean, target_q1_bound, self.mean_std1, bias)
        q2_loss = self.__dsact_loss_func(q2_mean, q2_std, target_q2_mean, target_q2_bound, self.mean_std2, bias)

        loss_q = q1_loss + q2_loss
        return loss_q, q1_mean.detach().mean(), q2_mean.detach().mean(), q1_std.detach().mean(), q2_std.detach().mean()

    def __compute_target_q(self, r, done, q_current, q_std_avg, q_next, q_next_sample):
        """计算 DSACT 的 Target，去掉了 Entropy 项"""
        # 标准 Bellman Target
        target_q = r + (1 - done) * self.gamma * q_next

        # Target Bound: 使用 Next Sample 计算 (这是 DSACT 论文的核心细节)
        target_q_sample = r + (1 - done) * self.gamma * q_next_sample

        # Bound 限制
        td_bound = 3 * q_std_avg
        difference = torch.clamp(target_q_sample - q_current, -td_bound, td_bound)
        target_q_bound = q_current + difference

        return target_q, target_q_bound

    def __dsact_loss_func(self, q, q_std, target_q, target_q_bound, mean_std, bias=0.1):
        """DSACT 复杂的 Loss 函数"""
        # 第一项：Mean Error weighted by Variance
        # 如果方差(std)很大，说明这里不确定，梯度就小一点
        # 如果方差很小，说明这里很确定，梯度就大一点
        term1 = -(target_q - q).detach() / (torch.pow(q_std, 2) + bias) * q

        # 第二项：Variance Estimation Error
        # 让 q_std 去拟合真实的 TD Error
        term2 = -((torch.pow(q.detach() - target_q_bound, 2) - torch.pow(q_std, 2)) / (
                    torch.pow(q_std, 3) + bias)) * q_std

        # 缩放因子
        weight = torch.pow(mean_std, 2) + bias
        loss = weight * torch.mean(term1 + term2)
        return loss

    def __compute_loss_policy(self, data: DataDict):
        # 保持之前的 Weighted BC 逻辑，这是适配 Diffusion 的最佳方案
        obs, act_real = data["obs"], data["act"]

        with torch.no_grad():
            # 使用 DSACT 的 Mean Q 来计算优势
            q1_mean, _, _ = self.__q_evaluate(obs, act_real, self.networks.q1)
            q2_mean, _, _ = self.__q_evaluate(obs, act_real, self.networks.q2)

            # Advantage Normalization
            q_min = torch.min(q1_mean, q2_mean)
            adv = q_min - q_min.mean()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            weights = torch.exp(adv * 3.0).clamp(max=100.0)

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


DsactDiffusion2 = DSACTDiffusion2