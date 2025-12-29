
__all__ = ["ApproxContainer", "DSACT_Diffusion", "DSACTDiffusion"]
import time
from copy import deepcopy
from typing import Tuple, Dict
import torch
import torch.nn as nn
from torch.optim import Adam

from gops.algorithm.base import AlgorithmBase, ApprBase
from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.utils.tensorboard_setup import tb_tags
from gops.utils.gops_typing import DataDict
from gops.apprfunc.apprfunc_diffusion import DiffusionMLP


# ==========================================
# Diffusion 数学工具类
# ==========================================
class DiffusionScheduler:
    def __init__(self, num_steps=100, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.num_steps = num_steps
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, num_steps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def add_noise(self, x_start, noise, t):
        """前向加噪: q(x_t | x_0)"""
        # 扩展维度以匹配 batch
        sqrt_alpha = self.sqrt_alphas_cumprod[t].reshape(-1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1)
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

    def sample_timesteps(self, batch_size):
        return torch.randint(0, self.num_steps, (batch_size,), device=self.device).long()


# ==========================================
# ApproxContainer (网络容器)
# ==========================================
class ApproxContainer(ApprBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # 1. 提取参数
        # 注意：这里我们假设 policy_args 里已经有了 DiffusionMLP 所需的参数
        self.diffusion_steps = kwargs.get("diffusion_steps", 100)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 2. 创建 Diffusion Scheduler
        self.scheduler = DiffusionScheduler(num_steps=self.diffusion_steps, device=device)

        # 3. 创建 Policy (Actor)
        # 关键点：我们在训练脚本里会把 "mlp_diffusion" 注册进 GOPS
        # 所以这里可以直接调用 create_apprfunc
        self.policy = create_apprfunc(**kwargs["policy_args"])

        # 4. 创建 Critic (Q Networks) - 保持 DSACT 的结构 (MLP)
        self.q1 = create_apprfunc(**kwargs["q_args"])
        self.q2 = create_apprfunc(**kwargs["q_args"])

        # 5. Target Networks
        self.q1_target = deepcopy(self.q1)
        self.q2_target = deepcopy(self.q2)

        # 6. Optimizers
        self.policy_optimizer = Adam(self.policy.parameters(), lr=kwargs["policy_learning_rate"])
        self.q1_optimizer = Adam(self.q1.parameters(), lr=kwargs["q_learning_rate"])
        self.q2_optimizer = Adam(self.q2.parameters(), lr=kwargs["q_learning_rate"])

        # 7. 其他参数
        self.tau = kwargs["tau"]

    # 这是一个占位符，防止 create_alg 调用时报错
    def create_action_distributions(self, logits):
        return logits


# ==========================================
# DSACT_Diffusion (算法逻辑)
# ==========================================
class DSACT_Diffusion(AlgorithmBase):
    def __init__(self, index=0, **kwargs):
        super().__init__(index, **kwargs)
        self.networks = ApproxContainer(**kwargs)
        self.gamma = kwargs["gamma"]
        self.delay_update = kwargs["delay_update"]

        # 扩散采样参数
        self.act_dim = kwargs["policy_args"]["act_dim"]
        # 假设 Action 在 Env wrapper 里已经归一化到 [-1, 1]
        self.act_max = 1.0
        self.act_min = -1.0

    @property
    def adjustable_parameters(self):
        return ("gamma", "tau", "delay_update")

    def local_update(self, data: DataDict, iteration: int) -> dict:
        tb_info = self.__compute_gradient(data, iteration)
        self.__update(iteration)
        return tb_info

    # ... (省略 remote_update 相关代码，Off-serial 模式用不到，或者可以直接复制 standard implementation) ...

    def __update(self, iteration: int):
        self.networks.q1_optimizer.step()
        self.networks.q2_optimizer.step()

        if iteration % self.delay_update == 0:
            self.networks.policy_optimizer.step()

            # Target Network Soft Update
            with torch.no_grad():
                polyak = 1 - self.networks.tau
                for p, p_targ in zip(self.networks.q1.parameters(), self.networks.q1_target.parameters()):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)
                for p, p_targ in zip(self.networks.q2.parameters(), self.networks.q2_target.parameters()):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

    def __compute_gradient(self, data: DataDict, iteration: int):
        start_time = time.time()

        # 1. Zero Gradients
        self.networks.q1_optimizer.zero_grad()
        self.networks.q2_optimizer.zero_grad()
        self.networks.policy_optimizer.zero_grad()

        # 2. Compute Loss
        loss_q, loss_q_info = self.__compute_loss_critic(data)
        loss_q.backward()
        tb_info = loss_q_info

        # 3. Compute Policy Loss (Delayed)
        if iteration % self.delay_update == 0:
            loss_policy, loss_policy_info = self.__compute_loss_policy(data)
            loss_policy.backward()
            tb_info.update(loss_policy_info)

        end_time = time.time()
        tb_info[tb_tags["alg_time"]] = (end_time - start_time) * 1000
        return tb_info

    def __compute_loss_critic(self, data: DataDict):
        obs, act, rew, obs_next, done = (
            data["obs"], data["act"], data["rew"], data["obs_next"], data["done"]
        )

        # MSE Bellman Loss
        q1 = self.networks.q1(obs, act)
        q2 = self.networks.q2(obs, act)

        with torch.no_grad():
            # Target Policy Action (Sampling from Diffusion)
            next_act = self._sample_action(obs_next)

            q1_next = self.networks.q1_target(obs_next, next_act)
            q2_next = self.networks.q2_target(obs_next, next_act)
            q_next = torch.min(q1_next, q2_next)

            target_q = rew + self.gamma * (1 - done) * q_next

        loss_q1 = nn.MSELoss()(q1, target_q)
        loss_q2 = nn.MSELoss()(q2, target_q)
        loss_q = loss_q1 + loss_q2

        return loss_q, {
            tb_tags["loss_critic"]: loss_q.item(),
            "q1_val": q1.mean().item()
        }

    def __compute_loss_policy(self, data: DataDict):
        """
        Online RL 核心修改：Q-Weighted Diffusion Loss
        """
        obs = data["obs"]
        act_real = data["act"]  # 来自 ReplayBuffer 的实际动作

        # ====================================================
        # Part 1: 计算 Q-Guidance 权重 (Weighting)
        # ====================================================
        with torch.no_grad():
            # 1. 计算当前动作的 Q 值
            q1 = self.networks.q1(obs, act_real)
            q2 = self.networks.q2(obs, act_real)
            q_current = torch.min(q1, q2)  # 保守估计

            # 2. 计算基线 V(s) (Baseline)
            # 为了数值稳定，通常用 batch 内的均值或分位数作为 V(s) 的近似
            v_baseline = q_current.mean()

            # 3. 计算优势 Advantage
            adv = q_current - v_baseline

            # 4. 计算权重 w = exp(adv / temperature)
            # temperature 越小，对高分动作的筛选越严格（只学最好的）
            # 建议 temperature 设为 3.0 ~ 10.0 之间
            temperature = 3.0
            weights = torch.exp(adv / temperature)

            # 截断权重防止数值爆炸 (Max Clip)
            weights = torch.clamp(weights, max=20.0)

        # ====================================================
        # Part 2: Diffusion Training (Weighted MSE)
        # ====================================================

        # 1. 采样时间步 t
        t = self.networks.scheduler.sample_timesteps(obs.shape[0])

        # 2. 采样噪声
        noise = torch.randn_like(act_real)

        # 3. 前向加噪
        act_noisy = self.networks.scheduler.add_noise(act_real, noise, t)

        # 4. 网络预测
        noise_pred = self.networks.policy(obs, act_noisy, t)

        # 5. 计算加权 MSE Loss
        # loss shape: [Batch, Action_Dim] -> [Batch]
        mse_loss = nn.MSELoss(reduction='none')(noise_pred, noise).mean(dim=1)

        # 应用 Q 权重
        loss_diff = (mse_loss * weights).mean()

        return loss_diff, {
            tb_tags["loss_actor"]: loss_diff.item(),
            "train/weight_mean": weights.mean().item(),
            "train/weight_max": weights.max().item(),
            "train/adv_mean": adv.mean().item()
        }

    def _sample_action(self, obs):
        """推理/采样函数：从纯噪声开始去噪"""
        device = obs.device
        batch_size = obs.shape[0]
        act_dim = self.act_dim

        # 1. 初始化纯噪声
        x = torch.randn((batch_size, act_dim), device=device)

        # 2. 逆向去噪循环
        scheduler = self.networks.scheduler
        for i in reversed(range(scheduler.num_steps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)

            # 预测噪声
            noise_pred = self.networks.policy(obs, x, t)

            # 计算 x_{t-1}
            alpha = scheduler.alphas[i]
            alpha_hat = scheduler.alphas_cumprod[i]
            beta = scheduler.betas[i]

            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (1 / torch.sqrt(alpha)) * (
                    x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * noise_pred
            ) + torch.sqrt(beta) * noise

        # 3. Clip Action
        x = x.clamp(self.act_min, self.act_max)
        return x

    # 兼容 Evaluator 调用
    def get_action(self, obs):
        with torch.no_grad():
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            action = self._sample_action(obs)
            return action.squeeze(0)

DSACTDiffusion = DSACT_Diffusion