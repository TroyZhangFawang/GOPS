#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Approximate Dynamic Program Algorithm for Infinity Horizon (INFADP)
#  Reference: Li SE (2023) 
#             Reinforcement Learning for Sequential Decision and Optimal Control. Springer, Singapore.
#  Update: 2021-03-05, Wenxuan Wang: create infADP algorithm
#  Update: 2022-12-04, Jiaxin Gao: supplementary comment information

__all__ = ["INFADP"]

from copy import deepcopy
from typing import Tuple
from gops.algorithm.infadp import ApproxContainer, INFADP
import torch
from torch.optim import Adam
import time

from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.create_pkg.create_env_model import create_env_model
from gops.utils.common_utils import get_apprfunc_dict
from gops.utils.tensorboard_setup import tb_tags
from gops.algorithm.base import AlgorithmBase, ApprBase
from copy import deepcopy
from typing import Tuple
import torch
import torch.nn as nn
from torch.optim import Adam
import time, math
from gops.utils.tensorboard_setup import tb_tags


__all__ = ["INFADPLagrangian"]

class INFADPLagrangian(INFADP):
    """Approximate Dynamic Program Algorithm for Infinity Horizon
    Paper: https://link.springer.com/book/10.1007/978-981-19-7784-8

    :param int forward_step: envmodel forward step.
    :param float gamma: discount factor.
    :param float tau: param for soft update of target network.
    :param int pev_step: number of steps for policy evaluation.
    :param int pim_step: number of steps for policy improvement.
    """

    def __init__(
            self,
            *,
            multiplier: float = 35.0,
            multiplier_lr: float = 1e-3,
            multiplier_delay: int = 2,
            index: int = 0,
            **kwargs,
    ):
        super().__init__(
            index=index,
            **kwargs,
        )
        self.forward_step = 50
        # inverse of softplus function
        self.multiplier_param = nn.Parameter(torch.tensor(
            math.log(math.exp(multiplier) - 1), dtype=torch.float32))
        self.multiplier_optim = Adam([self.multiplier_param], lr=multiplier_lr)
        self.multiplier_delay = multiplier_delay
        self.update_step = 0

    @property
    def adjustable_parameters(self) -> Tuple[str]:
        return (
            *super().adjustable_parameters,
            "multiplier",
            "multiplier_lr",
            "multiplier_delay",
        )

    def local_update(self, data: dict, iteration: int) -> dict:
        update_list = self._compute_gradient(data, iteration)
        self._update(update_list)
        return self.tb_info


    def _compute_gradient(self, data, iteration):
        update_list = []

        start_time = time.time()

        if iteration % (self.pev_step + self.pim_step) < self.pev_step:
            self.networks.v.zero_grad()
            loss_v, v = self._compute_loss_v(data)
            loss_v.backward()
            self.tb_info[tb_tags["loss_critic"]] = loss_v.item()
            self.tb_info[tb_tags["critic_avg_value"]] = v.item()
            update_list.append("v")
        else:
            self.networks.policy.zero_grad()
            loss_policy,loss_info = self._compute_loss_policy(data)
            loss_policy.backward()
            self.tb_info[tb_tags["loss_actor"]] = loss_policy.item()
            self.tb_info.update(loss_info)
            update_list.append("policy")

        end_time = time.time()

        self.tb_info[tb_tags["alg_time"]] = (end_time - start_time) * 1000  # ms
        return update_list

    def _compute_loss_policy(self, data):
        o, a, r, o2, d = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )
        info_init = data
        v_pi = torch.zeros(1)
        v_pi_c = torch.zeros(1)
        for p in self.networks.v.parameters():
            p.requires_grad = False
        for step in range(self.forward_step):
            if step == 0:
                a = self.networks.policy(o)
                o2, r, d, info = self.envmodel.forward(o, a, d, info_init)
                c = torch.clamp_min(info["constraint"], 0).sum(1)
                v_pi = r
                v_pi_c = c
            else:
                o = o2
                a = self.networks.policy(o)
                o2, r, d, info = self.envmodel.forward(o, a, d, info)
                c = torch.clamp_min(info["constraint"], 0).sum(1)
                v_pi += self.gamma**step * r
                v_pi_c += c * (self.gamma ** step)
        v_pi += (~d) * self.gamma**self.forward_step * self.networks.v_target(o2)
        for p in self.networks.v.parameters():
            p.requires_grad = True
        loss_reward = -v_pi.mean()
        loss_constraint = v_pi_c.mean()
        multiplier = torch.nn.functional.softplus(self.multiplier_param).item()
        loss_policy = loss_reward + multiplier * loss_constraint

        self.update_step += 1
        if self.update_step % self.multiplier_delay == 0:
            multiplier_loss = -self.multiplier_param * loss_constraint.item()
            self.multiplier_optim.zero_grad()
            multiplier_loss.backward()
            self.multiplier_optim.step()
        # self.multiplier_param.data = torch.clamp(self.multiplier_param.data, max=50)
        loss_info = {
            tb_tags["loss_actor"]: loss_policy.item(),
            tb_tags["loss_actor_reward"]: loss_reward.item(),
            tb_tags["loss_actor_constraint"]: loss_constraint.item(),
            "Loss/Lagrange multiplier-RL iter": multiplier,
        }
        return loss_policy, loss_info


if __name__ == "__main__":
    print("11111")
