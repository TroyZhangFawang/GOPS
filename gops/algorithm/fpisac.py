#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Soft Actor-Critic (SAC) algorithm equipped with feasible policy iteration (FPI)
#  Update: 2024-05-16, Zhilong Zheng: create FPI-SAC algorithm

__all__ = ["ApproxContainer", "FPISAC"]

import time
import math
from copy import deepcopy
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam

from gops.algorithm.sac import ApproxContainer as SACApproxContainer
from gops.algorithm.sac import SAC
from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.utils.tensorboard_setup import tb_tags
from gops.utils.gops_typing import DataDict
from gops.utils.common_utils import get_apprfunc_dict

EPSILON = 1e-6


class ApproxContainer(SACApproxContainer):
    """Approximate function container for SAC.

    Contains one policy and two action values.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # create qf network (for the constraint decay function in FPI)
        qf_args = get_apprfunc_dict("value", **kwargs)
        qf_args["value_output_activation"] = "linear"  # always use linear output activation for qf since we may need logits
        self.qf: nn.Module = create_apprfunc(**qf_args)

        # create target network for qf
        self.qf_target = deepcopy(self.qf)

        # set target network gradients
        for p in self.qf_target.parameters():
            p.requires_grad = False

        # create optimizer for qf
        self.qf_optimizer = Adam(self.qf.parameters(), lr=kwargs["q_learning_rate"])


class FPISAC(SAC):
    """Soft Actor-Critic (SAC) algorithm equipped with feasible policy iteration (FPI)

    Paper: https://arxiv.org/pdf/2304.08845

    :param float gamma: discount factor.
    :param float tau: param for soft update of target network.
    :param float alpha: initial temperature.
    :param bool auto_alpha: whether to adjust temperature automatically.
    :param Optional[float] target_entropy: target entropy for automatic
        temperature adjustment.
    :param float pf: feasibility threshold in (0, 1).
    :param float init_t: initial value of param t in the interior-point method.
    :param float t_increase_factor: factor for increasing t which should be greater than 1.
    :param int t_update_delay: delay for increasing t.
    :param Optional[float] max_t: maximum value of t.
    """

    def __init__(
        self,
        index: int = 0,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = math.e,
        auto_alpha: bool = True,
        target_entropy: Optional[float] = None,
        pf: float = 0.1,
        init_t: float = 1.0,
        t_increase_factor: float = 1.1,
        t_update_delay: int = 1000,
        max_t: Optional[float] = None,
        **kwargs: Any,
    ):
        super().__init__(
            index=index, 
            gamma=gamma,
            tau=tau,
            alpha=alpha,
            auto_alpha=auto_alpha,
            target_entropy=target_entropy,
            **kwargs
        )
        self.networks = ApproxContainer(**kwargs)
        self.networks.log_alpha.data.fill_(math.log(alpha))
        self.pf_logits = -math.log(1 / pf - 1)
        self.t = init_t
        assert t_increase_factor > 1, "t_increase_factor should be greater than 1"
        self.t_increase_factor = t_increase_factor
        self.t_update_delay = t_update_delay
        if max_t is None:
            self.max_t = math.inf
        else:
            self.max_t = max_t

    @property
    def adjustable_parameters(self):
        return super().adjustable_parameters + ("t_increase_factor", "t_update_delay", "max_t")

    def get_remote_update_info(
        self, data: DataDict, iteration: int
    ) -> Tuple[dict, dict]:
        tb_info, update_info = super().get_remote_update_info(data, iteration)
        update_info.update({"qf_grad": [p.grad for p in self.networks.qf.parameters()]})
        return tb_info, update_info

    def remote_update(self, update_info: dict):
        qf_grad = update_info["qf_grad"]

        for p, grad in zip(self.networks.qf.parameters(), qf_grad):
            p._grad = grad

        super().remote_update(update_info)

    def _compute_gradient(self, data: DataDict, iteration: int):
        start_time = time.time()

        obs = data["obs"]
        logits = self.networks.policy(obs)
        act_dist = self.networks.create_action_distributions(logits)
        new_act, new_logp = act_dist.rsample()
        data.update({"new_act": new_act, "new_logp": new_logp})

        self.networks.q1_optimizer.zero_grad()
        self.networks.q2_optimizer.zero_grad()
        self.networks.qf_optimizer.zero_grad()
        loss_q, q1, q2 = self._compute_loss_q(data)
        loss_q.backward()

        loss_qf, qf = self._compute_loss_qf(data)
        loss_qf.backward()

        for p in self.networks.q1.parameters():
            p.requires_grad = False
        for p in self.networks.q2.parameters():
            p.requires_grad = False
        for p in self.networks.qf.parameters():
            p.requires_grad = False

        self.networks.policy_optimizer.zero_grad()
        loss_policy, entropy, loss_policy_feas, loss_policy_infeas, log_barrier, feasible_ratio = self._compute_loss_policy(data)
        loss_policy.backward()

        for p in self.networks.q1.parameters():
            p.requires_grad = True
        for p in self.networks.q2.parameters():
            p.requires_grad = True
        for p in self.networks.qf.parameters():
            p.requires_grad = True

        if self.auto_alpha:
            self.networks.alpha_optimizer.zero_grad()
            loss_alpha = self._compute_loss_alpha(data)
            loss_alpha.backward()

        tb_info = {
            tb_tags["loss_critic"]: loss_q.item(),
            tb_tags["loss_actor"]: loss_policy.item(),
            tb_tags["loss_scenery"]: loss_qf.item(),
            "FPISAC/actor_loss_feasible-RL iter": loss_policy_feas.item(),
            "FPISAC/actor_loss_infeasible-RL iter": loss_policy_infeas.item(),
            "FPISAC/critic_avg_q1-RL iter": q1.item(),
            "FPISAC/critic_avg_q2-RL iter": q2.item(),
            "FPISAC/scenery_avg_qf-RL iter": qf.item(),
            "FPISAC/entropy-RL iter": entropy.item(),
            "FPISAC/alpha-RL iter": self._get_alpha(),
            "FPISAC/feasible_ratio-RL iter": feasible_ratio.item(),
            "FPISAC/log_barrier-RL iter": log_barrier.item(),
            "FPISAC/t-RL iter": self.t,
            tb_tags["alg_time"]: (time.time() - start_time) * 1000,
        }

        return tb_info
    
    def _compute_loss_qf(self, data: DataDict):
        obs, act, obs2, done = (
            data["obs"],
            data["act"],
            data["obs2"],
            data["done"],
        )
        next_cost = (data["next_constraint"] > 0).any(-1).float()
        qf = torch.sigmoid(self.networks.qf(obs, act))
        with torch.no_grad():
            next_logits = self.networks.policy(obs2)
            next_act_dist = self.networks.create_action_distributions(next_logits)
            next_act, _ = next_act_dist.rsample()
            next_qf = torch.sigmoid(self.networks.qf_target(obs2, next_act))
            backup = next_cost + (1 - done) * (1 - next_cost) * self.gamma * next_qf
        loss_qf = torch.nn.functional.binary_cross_entropy(qf, backup).mean()
        return loss_qf, qf.detach().mean()

    def _compute_loss_policy(self, data: DataDict):
        obs, new_act, new_logp = data["obs"], data["new_act"], data["new_logp"]
        q1 = self.networks.q1(obs, new_act)
        q2 = self.networks.q2(obs, new_act)
        qf_logits = self.networks.qf(obs, new_act)
        log_barrier = -torch.log(torch.clamp(self.pf_logits - qf_logits, min=EPSILON))
        feas_mask = (self.pf_logits - qf_logits) > EPSILON

        loss_policy1 = feas_mask * (self._get_alpha() * new_logp - torch.min(q1, q2) + log_barrier / self.t)
        loss_policy2 = ~feas_mask * qf_logits
        loss_policy = (loss_policy1 + loss_policy2).mean()
        entropy = -(new_logp * feas_mask).detach().mean()
        return loss_policy, entropy, loss_policy1.mean(), loss_policy2.mean(), log_barrier.mean(), feas_mask.float().mean()

    def _update(self, iteration: int):
        super()._update(iteration)

        self.networks.qf_optimizer.step()
        
        if (iteration + 1) % self.t_update_delay == 0:
            self.t = min(self.t * self.t_increase_factor, self.max_t)

        with torch.no_grad():
            polyak = 1 - self.tau
            for p, p_targ in zip(
                self.networks.qf.parameters(), self.networks.qf_target.parameters()
            ):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)