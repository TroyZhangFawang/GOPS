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

__all__ = ["RMPC3"]

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


class RMPC3(AlgorithmBase):
    """Approximate Dynamic Program Algorithm for Finity Horizon

    Paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4124940

    :param int forward_step: envmodel forward step.
    :param float gamma: discount factor.
    """

    def __init__(self, index=0, **kwargs):
        super().__init__(index, **kwargs)
        self.networks = ApproxContainer(**kwargs)
        self.envmodel = create_env_model(**kwargs)
        self.forward_step = kwargs["pre_horizon"]
        self.gamma = 1.0
        self.tb_info = dict()
        self.state_dim = kwargs["state_dim"]
        self.ref_obs_dim = kwargs["ref_obs_dim"]

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
        loss_policy = self._compute_loss_policy(deepcopy(data))

        loss_policy.backward()

        self.tb_info[tb_tags["loss_actor"]] = loss_policy.item()

        end_time = time.time()

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
        # random_number = torch.randint(1, self.forward_step+1, (1,)).item()
        # weights = torch.arange(1, self.forward_step + 1).float()  # 生成权重为1, 2, ..., forward_step
        weights = torch.ones(self.forward_step).float()  # 生成权重为1, 1, ..., 1
        # weights = torch.arange(20, 0, -1).float()  # 生成权重为1, 2, ..., forward_step
        random_len = torch.multinomial(weights, num_samples=1, replacement=True) + 1
        random_number = random_len.item()
        o_clip = o[:, :self.state_dim + random_number * self.ref_obs_dim]
        info = data
        v_pi = 0
        a = self.networks.policy.forward_all_policy(o_clip)
        for step in range(random_number):
            o, r, d, info = self.envmodel.forward(o, a[:, step, :], d, info)
            v_pi += r * (self.gamma ** step)
        loss_policy = -v_pi.mean()
        return loss_policy

if __name__ == "__main__":
    print("11111")
