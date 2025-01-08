#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Approximate Dynamic Program Algorithm for Finity Horizon (FHADP)
#  Reference: Li SE (2023)
#             Reinforcement Learning for Sequential Decision and Optimal Control. Springer, Singapore.
#  create: 2023-07-28, Jiaxin Gao: create full horizon action fhadp algorithm

__all__ = ["TRANSStolenMpcLagrangian"]

import math
import torch
import torch.nn as nn
from torch.optim import Adam
from gops.utils.tensorboard_setup import tb_tags
from gops.algorithm.trans_stolen_mpc import ApproxContainer, TRANSStolenMpc




class TRANSStolenMpcLagrangian(TRANSStolenMpc):
    def __init__(self,
                 *,
                 multiplier: float = 1.0,
                 multiplier_lr: float = 1e-3,
                 multiplier_delay: int = 10,
                 index: int = 0,
                 **kwargs):
        super().__init__(index, **kwargs)
        # inverse of softplus function
        self.multiplier_param = nn.Parameter(torch.tensor(
            math.log(math.exp(multiplier) - 1), dtype=torch.float32))
        self.multiplier_optim = Adam([self.multiplier_param], lr=multiplier_lr)
        self.multiplier_delay = multiplier_delay
        self.update_step = 0

    @property
    def adjustable_parameters(self):
        return (
            *super().adjustable_parameters,
            "multiplier",
            "multiplier_lr",
            "multiplier_delay",
        )

    def _compute_loss_policy(self, data):
        o, a, r, o2, d = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )
        # torch.nn.utils.clip_grad_norm_(self.networks.policy.parameters(), max_norm=1.0)
        random_len = torch.randint(1, self.forward_step+1, (self.batch_size,))
        seq_range = torch.arange(self.forward_step).expand(self.batch_size, self.forward_step)
        key_padding_mask = seq_range >= random_len.unsqueeze(1)
        steps = torch.arange(self.forward_step).expand(self.batch_size, self.forward_step)
        gamma_powers = self.gamma ** steps
        mask = steps < (random_len).unsqueeze(1)
        info = data
        v_pi = torch.zeros((o.size(0), self.forward_step))
        v_pi_c = 0
        a = self.networks.policy.forward_all_policy(o, key_padding_mask=key_padding_mask)
        for step in range(self.forward_step):
            o, r, d, info = self.envmodel.forward(o, a[:, step, :], d, info)
            c = torch.clamp_min(info["constraint"], 0).sum(1)  # 若小于0，赋值为0，也就是满足约束则不惩罚
            v_pi[:, step] = r
            v_pi_c += c * (self.gamma ** step)
        weighted_rewards = v_pi * gamma_powers * mask
        v_pi_final = weighted_rewards.sum(dim=1)
        loss_reward = -v_pi_final.mean()
        loss_constraint = v_pi_c.mean()
        multiplier = torch.nn.functional.softplus(self.multiplier_param).item()
        loss_policy = loss_reward + multiplier * loss_constraint

        self.update_step += 1
        if self.update_step % self.multiplier_delay == 0:
            multiplier_loss = -self.multiplier_param * loss_constraint.item()
            self.multiplier_optim.zero_grad()
            multiplier_loss.backward()
            self.multiplier_optim.step()
        loss_info = {
            tb_tags["loss_actor"]: loss_policy.item(),
            tb_tags["loss_actor_reward"]: loss_reward.item(),
            tb_tags["loss_actor_constraint"]: loss_constraint.item(),
            "Loss/Lagrange multiplier-RL iter": multiplier,
        }
        return loss_policy, loss_info

