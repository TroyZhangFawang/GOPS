#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Soft Actor-Critic (SAC) algorithm
#  Reference: Haarnoja T, Zhou A, Abbeel P et al (2018) 
#             Soft actor-critic: off-policy maximum entropy deep reinforcement learning with a stochastic actor. 
#             ICML, Stockholm, Sweden.
#  Update: 2021-03-05, Yujie Yang: create SAC algorithm

__all__ = ["ApproxContainer", "SACFpi2"]

import time
from copy import deepcopy
from typing import Any, Optional, Tuple
from gops.utils.act_distribution_cls import Action_Distribution
from gops.utils.common_utils import get_activation_func
import torch
import torch.nn as nn
from torch.optim import Adam
import math
from gops.algorithm.base import AlgorithmBase, ApprBase
from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.utils.tensorboard_setup import tb_tags
from gops.utils.gops_typing import DataDict
from gops.utils.common_utils import get_apprfunc_dict


class ApproxContainer(ApprBase):
    """Approximate function container for SAC.

    Contains one policy and two action values.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # create q networks
        q_args = get_apprfunc_dict("value", **kwargs)
        self.q1: nn.Module = create_apprfunc(**q_args)
        self.q2: nn.Module = create_apprfunc(**q_args)
        self.qf: nn.Module = create_apprfunc(**q_args)

        # create policy network
        policy_args = get_apprfunc_dict("policy", **kwargs)
        self.policy: nn.Module = create_apprfunc(**policy_args)

        # create target networks
        self.q1_target = deepcopy(self.q1)
        self.q2_target = deepcopy(self.q2)
        self.qf_target = deepcopy(self.qf)

        self.multiplier: nn.Module = Multiplier(**kwargs)

        # set target networks gradients
        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False
        for p in self.qf_target.parameters():
            p.requires_grad = False

        # create entropy coefficient
        self.log_alpha = nn.Parameter(torch.tensor(1, dtype=torch.float32))

        # create optimizers
        self.q1_optimizer = Adam(self.q1.parameters(), lr=kwargs["q_learning_rate"])
        self.q2_optimizer = Adam(self.q2.parameters(), lr=kwargs["q_learning_rate"])
        self.qf_optimizer = Adam(self.qf.parameters(), lr=kwargs["q_learning_rate"])

        self.multiplier_optimizer = Adam(self.multiplier.parameters(), lr=1e-5)

        self.policy_optimizer = Adam(
            self.policy.parameters(), lr=kwargs["policy_learning_rate"]
        )
        self.alpha_optimizer = Adam([self.log_alpha], lr=kwargs["alpha_learning_rate"])

    def create_action_distributions(self, logits):
        return self.policy.get_act_dist(logits)


class SACFpi2(AlgorithmBase):
    """Soft Actor-Critic (SAC) algorithm

    Paper: https://arxiv.org/abs/1801.01290

    :param float gamma: discount factor.
    :param float tau: param for soft update of target network.
    :param bool auto_alpha: whether to adjust temperature automatically.
    :param float alpha: initial temperature.
    :param Optional[float] target_entropy: target entropy for automatic
        temperature adjustment.
    """

    def __init__(
        self,
        index: int = 0,
        gamma: float = 0.99,
        tau: float = 0.005,
        auto_alpha: bool = True,
        alpha: float = 0.2,
        target_entropy: Optional[float] = None,
        pf: float = 0.1,
        eps: float = 1e-6,
        init_t: float = 1.0,
        multiplier_update_delay: int = 10,
        max_t: Optional[float] = None,
        **kwargs: Any,
    ):
        super().__init__(index, **kwargs)
        self.networks = ApproxContainer(**kwargs)
        self.gamma = gamma
        self.tau = tau
        self.auto_alpha = auto_alpha
        self.alpha = alpha
        if target_entropy is None:
            target_entropy = -kwargs["action_dim"]
        self.target_entropy = target_entropy
        self.pf_logit = -math.log(1 / pf - 1)
        self.eps = eps
        self.multiplier_update_delay = multiplier_update_delay
        self.t = init_t
        if max_t is None:
            self.max_t = math.inf
        else:
            self.max_t = max_t
        self.step = 0
        self.update_multiplier = False


    @property
    def adjustable_parameters(self):
        return ("gamma", "tau", "auto_alpha", "alpha", "target_entropy")

    def local_update(self, data: DataDict, iteration: int) -> dict:
        tb_info = self.__compute_gradient(data, iteration)
        self.__update(iteration)
        return tb_info

    def get_remote_update_info(
        self, data: DataDict, iteration: int
    ) -> Tuple[dict, dict]:
        tb_info = self.__compute_gradient(data, iteration)

        update_info = {
            "q1_grad": [p.grad for p in self.networks.q1.parameters()],
            "q2_grad": [p.grad for p in self.networks.q2.parameters()],
            "policy_grad": [p.grad for p in self.networks.policy.parameters()],
            "iteration": iteration,
        }
        if self.auto_alpha:
            update_info.update({"log_alpha_grad":self.networks.log_alpha.grad})

        return tb_info, update_info

    def remote_update(self, update_info: dict):
        iteration = update_info["iteration"]
        qf_grad = update_info["qf_grad"]
        q1_grad = update_info["q1_grad"]
        q2_grad = update_info["q2_grad"]
        policy_grad = update_info["policy_grad"]

        for p, grad in zip(self.networks.qf.parameters(), qf_grad):
            p._grad = grad
        for p, grad in zip(self.networks.q1.parameters(), q1_grad):
            p._grad = grad
        for p, grad in zip(self.networks.q2.parameters(), q2_grad):
            p._grad = grad
        for p, grad in zip(self.networks.policy.parameters(), policy_grad):
            p._grad = grad
        if self.auto_alpha:
            self.networks.log_alpha._grad = update_info["log_alpha_grad"]

        self.__update(iteration)

    def __get_alpha(self, requires_grad: bool = False):
        if self.auto_alpha:
            alpha = self.networks.log_alpha.exp()
            if requires_grad:
                return alpha
            else:
                return alpha.item()
        else:
            return self.alpha

    def __compute_gradient(self, data: DataDict, iteration: int):
        start_time = time.time()

        obs = data["obs"]
        logits = self.networks.policy(obs)
        act_dist = self.networks.create_action_distributions(logits)
        new_act, new_logp = act_dist.rsample()
        data.update({"new_act": new_act, "new_logp": new_logp})

        self.networks.q1_optimizer.zero_grad()
        self.networks.q2_optimizer.zero_grad()
        self.networks.qf_optimizer.zero_grad()
        loss_q, loss_qf, q1, q2, qf = self.__compute_loss_q(data)
        loss_qf.backward()
        loss_q.backward()

        for p in self.networks.q1.parameters():
            p.requires_grad = False
        for p in self.networks.q2.parameters():
            p.requires_grad = False
        for p in self.networks.qf.parameters():
            p.requires_grad = False

        self.networks.policy_optimizer.zero_grad()
        loss_policy, entropy, policy_loss1, policy_loss2, log_barrier, feasible, lamada = self.__compute_loss_policy(data)
        loss_policy.backward()

        for p in self.networks.q1.parameters():
            p.requires_grad = True
        for p in self.networks.q2.parameters():
            p.requires_grad = True
        for p in self.networks.qf.parameters():
            p.requires_grad = True

        if self.auto_alpha:
            self.networks.alpha_optimizer.zero_grad()
            loss_alpha = self.__compute_loss_alpha(data)
            loss_alpha.backward()

        self.step += 1
        if self.step % self.multiplier_update_delay == 0:
            self.update_multiplier = True
        else:
            self.update_multiplier = False

        tb_info = {
            tb_tags["loss_critic"]: loss_q.item(),
            tb_tags["loss_actor"]: loss_policy.item(),
            "SAC/critic_avg_q1-RL iter": q1.item(),
            "SAC/critic_avg_q2-RL iter": q2.item(),
            "SAC/entropy-RL iter": entropy.item(),
            "SAC/alpha-RL iter": self.__get_alpha(),
            'SAC/qf_loss': loss_qf,
            'SAC/qf': torch.sigmoid(qf).mean(),
            'SAC/feasible_policy_loss': policy_loss1.mean(),
            'SAC/infeasible_policy_loss': policy_loss2.mean(),
            'SAC/log_barrier': log_barrier.mean(),
            'SAC/feasible_ratio': feasible.to(dtype=torch.float32).mean(),
            'SAC/multiplier': lamada,
            tb_tags["alg_time"]: (time.time() - start_time) * 1000,
        }

        return tb_info

    def masked_mean(self, x: torch.Tensor, mask: torch.Tensor, axis: Optional[int] = None) -> torch.Tensor:
        mask = mask.to(dtype=x.dtype)
        masked_sum = torch.sum(mask * x, dim=axis)
        masked_count = torch.sum(mask, dim=axis)
        masked_count = torch.clamp(masked_count, min=1)
        return masked_sum / masked_count

    def __compute_loss_q(self, data: DataDict):
        obs, act, rew, obs2, done, next_cost = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
            data["constraint"]
        )
        next_cost = (torch.clamp_min(next_cost, 0).sum(1)) > 0

        q1 = self.networks.q1(obs, act)
        q2 = self.networks.q2(obs, act)
        qf_logits = self.networks.qf(obs, act)
        qf_mask = qf_logits - self.pf_logit < -self.eps
        with torch.no_grad():
            next_logits = self.networks.policy(obs2)
            next_act_dist = self.networks.create_action_distributions(next_logits)
            next_act, next_logp = next_act_dist.rsample()
            next_q1 = self.networks.q1_target(obs2, next_act)
            next_q2 = self.networks.q2_target(obs2, next_act)
            next_qf_logits = self.networks.qf_target(obs2, next_act)
            next_qf_target = torch.sigmoid(next_qf_logits)
            next_q = torch.min(next_q1, next_q2)
            qf_backup = next_cost + (1 - done) * (~next_cost) * self.gamma * next_qf_target
            backup = rew + (1 - done) * self.gamma * (
                next_q - self.__get_alpha() * next_logp
            )
        loss_qf = torch.nn.functional.binary_cross_entropy_with_logits(qf_logits, qf_backup)
        loss_q1 = self.masked_mean(((q1 - backup) ** 2), qf_mask)
        loss_q2 = self.masked_mean(((q2 - backup) ** 2), qf_mask)
        return loss_q1 + loss_q2, loss_qf, q1.detach().mean(), q2.detach().mean(), qf_logits.detach().mean()


    def __compute_loss_policy(self, data: DataDict):
        obs, new_act, new_logp = data["obs"], data["new_act"], data["new_logp"]
        q1 = self.networks.q1(obs, new_act)
        q2 = self.networks.q2(obs, new_act)
        qf_logits = self.networks.qf(obs, new_act)
        log_barrier = qf_logits - self.pf_logit
        # log_barrier = -torch.log(-torch.minimum(qf_logits - self.pf_logit, torch.tensor(-self.eps)))
        feasible = qf_logits - self.pf_logit < -self.eps

        lamada = self.networks.multiplier(obs)

        policy_loss1 = feasible * (self.__get_alpha() * new_logp - torch.min(q1, q2) +
                                   lamada * log_barrier).mean()
        # policy_loss1 = feasible * (self.__get_alpha() * new_logp - torch.min(q1, q2)+
        #                                    1 / self.t * log_barrier).mean()
        policy_loss2 = ~feasible * qf_logits

        new_logp = self.masked_mean(new_logp, feasible)
        data.update({"new_logp": new_logp})
        entropy = -new_logp.detach().mean()
        loss_policy = (policy_loss1 + policy_loss2).mean()
        return loss_policy, entropy, policy_loss1, policy_loss2, log_barrier, feasible, lamada.mean()

    def __compute_loss_alpha(self, data: DataDict):
        new_logp = data["new_logp"]
        loss_alpha = (
            -self.networks.log_alpha * (new_logp.detach() + self.target_entropy).mean()
        )
        return loss_alpha

    def __update(self, iteration: int):
        self.networks.qf_optimizer.step()
        self.networks.q1_optimizer.step()
        self.networks.q2_optimizer.step()

        self.networks.policy_optimizer.step()

        if self.auto_alpha:
            self.networks.alpha_optimizer.step()
        if self.update_multiplier:

            for param in self.networks.multiplier.parameters():
                param.grad = -param.grad  # 反转梯度
            self.networks.multiplier_optimizer.step()  # 使用反转后的梯度进行更新

        with torch.no_grad():
            polyak = 1 - self.tau
            for p, p_targ in zip(
                    self.networks.qf.parameters(), self.networks.qf_target.parameters()
            ):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)
            for p, p_targ in zip(
                self.networks.q1.parameters(), self.networks.q1_target.parameters()
            ):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)
            for p, p_targ in zip(
                self.networks.q2.parameters(), self.networks.q2_target.parameters()
            ):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class Multiplier(nn.Module, Action_Distribution):
    """
    Approximated function of state-value function.
    Input: observation, action.
    Output: state-value.
    """
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obsv_dim"]
        hidden_sizes = kwargs["value_hidden_sizes"]
        self.mul = mlp(
            [obs_dim] + list(hidden_sizes) + [1],
            get_activation_func(kwargs["value_hidden_activation"]),
            get_activation_func(kwargs["value_output_activation"]),
        )
    def forward(self, obs):
        muliplier = self.mul(obs)
        muliplier_log = torch.nn.functional.softplus(muliplier)
        clamped_multiplier_log = torch.clamp(muliplier_log, max=10000)
        return torch.squeeze(clamped_multiplier_log, -1)