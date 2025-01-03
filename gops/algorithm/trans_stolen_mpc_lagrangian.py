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
from copy import deepcopy
from typing import Tuple
import torch
import torch.nn as nn
from torch.optim import Adam
import time
import torch.nn.functional as F
import warnings
from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.create_pkg.create_env_model import create_env_model
from gops.utils.common_utils import get_apprfunc_dict
from gops.utils.tensorboard_setup import tb_tags
from gops.algorithm.base import AlgorithmBase, ApprBase


class ApproxContainer(ApprBase):
    def __init__(self, **kwargs):
        """Approximate function container for FHADP."""
        """Contains one policy network."""
        super().__init__(**kwargs)
        policy_args = get_apprfunc_dict("policy", **kwargs)

        self.policy = create_apprfunc(**policy_args)
        self.policy_optimizer = Adam(
            self.policy.parameters(), lr=kwargs["policy_learning_rate"]
        )

    def create_action_distributions(self, logits):
        """create action distribution"""
        return self.policy.get_act_dist(logits)


class TRANSStolenMpcLagrangian(AlgorithmBase):
    """Approximate Dynamic Program Algorithm for Finity Horizon

    Paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4124940

    :param int forward_step: envmodel forward step.
    :param float gamma: discount factor.
    """

    def __init__(self,
                 multiplier: float = 1.0,
                 multiplier_lr: float = 1e-3,
                 multiplier_delay: int = 10,
                 index: int = 0,
                 **kwargs):
        super().__init__(index, **kwargs)
        self.networks = ApproxContainer(**kwargs)
        self.envmodel = create_env_model(**kwargs)
        self.forward_step = kwargs["pre_horizon"]
        self.gamma = 1.0
        self.tb_info = dict()
        self.max_trajectory = kwargs["max_trajectory"]
        self.state_dim = kwargs["state_dim"]
        self.ref_obs_dim = kwargs["ref_obs_dim"]
        self.batch_size = kwargs["replay_batch_size"]
        # inverse of softplus function
        self.multiplier_param = nn.Parameter(torch.tensor(
            math.log(math.exp(multiplier) - 1), dtype=torch.float32))
        self.multiplier_optim = Adam([self.multiplier_param], lr=multiplier_lr)
        self.multiplier_delay = multiplier_delay
        self.update_step = 0

    @property
    def adjustable_parameters(self):
        para_tuple = ("forward_step", "gamma")
        return para_tuple

    def local_update(self, data, iteration: int):
        self._compute_gradient(data)
        self.networks.policy_optimizer.step()
        return self.tb_info

    def get_remote_update_info(self, data: dict, iteration: int) -> Tuple[dict, dict]:
        self._compute_gradient(data)
        policy_grad = [p._grad for p in self.networks.policy.parameters()]
        update_info = dict()
        update_info["grad"] = policy_grad
        return self.tb_info, update_info

    def remote_update(self, update_info: dict):
        for p, grad in zip(self.networks.policy.parameters(), update_info["grad"]):
            p.grad = grad
        self.networks.policy_optimizer.step()

    def _compute_gradient(self, data):
        start_time = time.time()
        self.networks.policy.zero_grad()
        loss_policy, loss_info = self._compute_loss_policy(deepcopy(data))
        loss_policy.backward()
        # self.tb_info[tb_tags["loss_actor"]] = loss_info
        end_time = time.time()
        self.tb_info.update(loss_info)
        self.tb_info[tb_tags["alg_time"]] = (end_time - start_time) * 1000  # ms

        return

    # def pad_to_length(self, tensor, length):
    #     padding_size = length - tensor.size(0)
    #     if padding_size > 0:
    #         # 填充 (前后填充 0, 上下填充 0, 左右填充)
    #         padded_tensor = F.pad(tensor, (0, 0, 0, padding_size))
    #     else:
    #         padded_tensor = tensor[:length]
    #     return padded_tensor


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
            c = torch.clamp_min(info["constraint"], 0).sum(1)
            v_pi[:, step] = r
            v_pi_c += c * (self.gamma ** step)
        weighted_rewards = v_pi * gamma_powers * mask
        v_pi_final = weighted_rewards.sum(dim=1)
        loss_reward = -v_pi_final.mean()
        loss_constraint= v_pi_c.mean()
        multiplier = torch.nn.functional.softplus(self.multiplier_param).item()
        loss_policy = loss_reward + multiplier * loss_constraint

        self.update_step += 1
        if self.update_step % self.multiplier_delay == 0:
            multiplier_loss = -self.multiplier_param * loss_constraint.item()
            self.multiplier_optim.zero_grad()
            multiplier_loss.backward()
            self.multiplier_optim.step()
        loss_info = {
            tb_tags["loss_actor"]: loss_policy.item()
        }
        return loss_policy, loss_info

if __name__ == "__main__":
    print("11111")
