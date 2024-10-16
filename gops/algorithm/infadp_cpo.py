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

__all__ = ["INFADPCpo"]

from copy import deepcopy
from typing import Tuple

import torch
from torch.optim import Adam
import time
from gops.utils.common_utils import get_activation_func
from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.create_pkg.create_env_model import create_env_model
from gops.utils.common_utils import get_apprfunc_dict
from gops.utils.tensorboard_setup import tb_tags
from gops.algorithm.base import AlgorithmBase, ApprBase
from gops.utils.act_distribution_cls import Action_Distribution
import torch.nn as nn

class ApproxContainer(ApprBase):
    def __init__(self, **kwargs):
        """Approximate function container for INFADP."""
        """Contains two policy and two action values."""

        super().__init__(**kwargs)

        v_args = get_apprfunc_dict("value", **kwargs)
        sv_args = get_apprfunc_dict("value", **kwargs)
        policy_args = get_apprfunc_dict("policy", **kwargs)

        self.v = create_apprfunc(**v_args)
        self.policy = create_apprfunc(**policy_args)
        self.sv = create_apprfunc(**sv_args)

        self.v_target = deepcopy(self.v)
        self.policy_target = deepcopy(self.policy)
        self.sv_target = deepcopy(self.sv)

        self.multiplier: nn.Module = Multiplier(**kwargs)

        for p in self.v_target.parameters():
            p.requires_grad = False
        for p in self.sv_target.parameters():
            p.requires_grad = False
        for p in self.policy_target.parameters():
            p.requires_grad = False

        self.policy_optimizer = Adam(
            self.policy.parameters(), lr=kwargs["policy_learning_rate"]
        )  #
        self.v_optimizer = Adam(self.v.parameters(), lr=kwargs["value_learning_rate"])
        self.sv_optimizer = Adam(self.sv.parameters(), lr=kwargs["value_learning_rate"])
        self.multiplier_optimizer = Adam(self.multiplier.parameters(), lr=1e-5)

        self.net_dict = {"v": self.v, 'sv': self.sv, "policy": self.policy, "multiplier": self.multiplier}
        self.target_net_dict = {"v": self.v_target, 'sv': self.sv_target, "policy": self.policy_target}
        self.optimizer_dict = {"v": self.v_optimizer, 'sv': self.sv_optimizer, "policy": self.policy_optimizer, "multiplier": self.multiplier_optimizer}

    # create action_distributions
    def create_action_distributions(self, logits):
        """create action distribution"""
        return self.policy.get_act_dist(logits)


class INFADPCpo(AlgorithmBase):
    """Approximate Dynamic Program Algorithm for Infinity Horizon
    Paper: https://link.springer.com/book/10.1007/978-981-19-7784-8

    :param int forward_step: envmodel forward step.
    :param float gamma: discount factor.
    :param float tau: param for soft update of target network.
    :param int pev_step: number of steps for policy evaluation.
    :param int pim_step: number of steps for policy improvement.
    """

    def __init__(self, index=0, **kwargs):
        super().__init__(index, **kwargs)
        self.networks = ApproxContainer(**kwargs)
        self.envmodel = create_env_model(**kwargs)
        self.gamma = 0.99
        self.tau = 0.005
        self.pev_step = 1
        self.pim_step = 1
        self.forward_step = 10
        self.tb_info = dict()
        self.update_step = 0
        self.multiplier_delay = 5
        self.mul_is_update = False
        self.safe_p = 0.01 #安全阈值，只针对机械臂
    @property
    def adjustable_parameters(self):
        para_tuple = (
            "gamma",
            "tau",
            "pev_step",
            "pim_step",
            "forward_step",
            "reward_scale",
        )
        return para_tuple

    def local_update(self, data: dict, iteration: int) -> dict:
        update_list = self._compute_gradient(data, iteration)
        self._update(update_list)
        return self.tb_info

    def get_remote_update_info(self, data: dict, iteration: int) -> Tuple[dict, dict]:
        update_list = self._compute_gradient(data, iteration)
        update_info = dict()
        for net_name in update_list:
            update_info[net_name] = [
                p.grad for p in self.networks.net_dict[net_name].parameters()
            ]
        return self.tb_info, update_info

    def remote_update(self, update_info: dict):
        for net_name, grads in update_info.items():
            for p, grad in zip(self.networks.net_dict[net_name].parameters(), grads):
                p.grad = grad
        self._update(list(update_info.keys()))

    def _update(self, update_list):
        self.update_step += 1
        tau = self.tau
        for net_name in update_list:
            if net_name == 'multiplier':
                for param in self.networks.multiplier.parameters():
                    param.grad = -param.grad  # 反转梯度

                self.networks.multiplier_optimizer.step()  # 使用反转后的梯度进行更新

                # 然后确保将梯度再次反转回来（如果后续还有其他依赖此梯度的计算）
                for param in self.networks.multiplier.parameters():
                    param.grad = -param.grad
            else:
                self.networks.optimizer_dict[net_name].step()

        with torch.no_grad():
            for net_name in update_list:
                if net_name == 'multiplier':
                    continue
                for p, p_targ in zip(
                    self.networks.net_dict[net_name].parameters(),
                    self.networks.target_net_dict[net_name].parameters(),
                ):
                    p_targ.data.mul_(1 - tau)
                    p_targ.data.add_(tau * p.data)

    def _compute_gradient(self, data, iteration):
        update_list = []

        start_time = time.time()

        if iteration % (self.pev_step + self.pim_step) < self.pev_step:
            self.networks.v.zero_grad()
            self.networks.sv.zero_grad()
            loss_v, loss_sv, v, sv, loss_info = self._compute_loss_v(data)
            loss_v.backward()
            loss_sv.backward()
            self.tb_info[tb_tags["loss_critic"]] = loss_v.item()
            self.tb_info[tb_tags["critic_avg_value"]] = v.item()
            self.tb_info.update(loss_info)
            update_list.append("v")
            update_list.append("sv")
        else:
            self.networks.policy.zero_grad()
            self.networks.multiplier.zero_grad()
            loss_policy, loss_info = self._compute_loss_policy(data)
            # 计算loss_policy的梯度
            loss_policy.backward()


            self.tb_info[tb_tags["loss_actor"]] = loss_policy.item()
            self.tb_info.update(loss_info)
            update_list.append("policy")
            update_list.append("multiplier")

        end_time = time.time()

        self.tb_info[tb_tags["alg_time"]] = (end_time - start_time) * 1000  # ms
        return update_list

    def _compute_loss_v(self, data):
        o, a, r, o2, d = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )
        v = self.networks.v(o)
        sv = self.networks.sv(o)
        info_init = data
        batch_size = o.shape[0]
        with torch.no_grad():
            v_pi_h = torch.zeros(batch_size)
            for step in range(self.forward_step):
                if step == 0:

                    a = self.networks.policy(o)
                    o2, r, d, info = self.envmodel.forward(o, a, d, info_init)
                    h = torch.clamp_min(info["constraint"], 0).sum(1)
                    v_pi_h = torch.max(v_pi_h, h)  # 保留最大值
                    backup = r

                else:
                    o = o2
                    a = self.networks.policy(o)
                    o2, r, d, info = self.envmodel.forward(o, a, d, info)
                    h = torch.clamp_min(info["constraint"], 0).sum(1)
                    v_pi_h = torch.max(v_pi_h, h)  # 保留最大值
                    backup += self.gamma**step * r

            backup += (
                (~d) * self.gamma**self.forward_step * self.networks.v_target(o2)
            )
            # v_pi_h = (1-self.gamma)*v_pi_h + self.gamma*(torch.max(v_pi_h, (~d) * self.gamma**self.forward_step * self.networks.sv_target(o2)))
            v_pi_h = (1-self.gamma)*v_pi_h + self.gamma*(torch.max(v_pi_h, (~d) * self.networks.sv_target(o2)))
        loss_sv = ((sv - v_pi_h)**2).mean()
        loss_v = ((v - backup) ** 2).mean()
        loss_info = {
            tb_tags["loss_critic"]: loss_v.item(),
            tb_tags["critic_avg_value"]: torch.mean(v).item(),
            "Loss_sv_iter" : loss_sv.item(),
            "avg_sv": torch.mean(sv).item(),
        }
        return loss_v, loss_sv, torch.mean(v), torch.mean(sv), loss_info

    def _compute_loss_policy(self, data):
        o, a, r, o2, d = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )
        o_ori = o
        batch_size = o.shape[0]
        info_init = data
        v_pi = torch.zeros(1)
        v_pi_h = torch.zeros(batch_size)
        for p in self.networks.v.parameters():
            p.requires_grad = False
        for p in self.networks.sv.parameters():
            p.requires_grad = False

        for step in range(self.forward_step):
            if step == 0:
                a = self.networks.policy(o)
                o2, r, d, info = self.envmodel.forward(o, a, d, info_init)
                h = torch.clamp_min(info["constraint"], 0).sum(1)
                v_pi = r
                v_pi_h = torch.max(v_pi_h, h) #保留最大值
            else:
                o = o2
                a = self.networks.policy(o)
                o2, r, d, info = self.envmodel.forward(o, a, d, info)
                h = torch.clamp_min(info["constraint"], 0).sum(1)
                v_pi += self.gamma**step * r
                v_pi_h = torch.max(v_pi_h, h) #保留最大值
        v_pi += (~d) * self.gamma**self.forward_step * self.networks.v_target(o2)
        # v_pi_h = torch.max(v_pi_h, (~d) * self.gamma**self.forward_step * self.networks.sv_target(o2))  # 保留最大值
        v_pi_h = torch.max(v_pi_h, (~d) * self.networks.sv_target(o2))  # 保留最大值
        lamada = self.networks.multiplier(o_ori)

        loss_reward = -v_pi

        loss_policy = loss_reward + lamada * (v_pi_h - self.safe_p)

        for p in self.networks.v.parameters():
            p.requires_grad = True
        for p in self.networks.sv.parameters():
            p.requires_grad = True

        loss_info = {
            tb_tags["loss_actor"]: torch.mean(loss_policy).item(),
            tb_tags["loss_actor_reward"]: torch.mean(loss_reward).item(),
            tb_tags["loss_actor_constraint"]: torch.mean(v_pi_h).item(),
            "Loss/Lagrange multiplier-RL iter": torch.mean(lamada).item(),
        }
        return loss_policy.mean(), loss_info



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
if __name__ == "__main__":
    print("11111")
