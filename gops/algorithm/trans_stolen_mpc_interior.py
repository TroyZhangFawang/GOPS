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

__all__ = ["TRANSStolenMpcInterior"]


from typing import Tuple
import torch
from gops.algorithm.trans_stolen_mpc import ApproxContainer, TRANSStolenMpc
from gops.utils.gops_typing import DataDict, InfoDict
from gops.utils.tensorboard_setup import tb_tags

EPSILON = 1e-8

class TRANSStolenMpcInterior(TRANSStolenMpc):
    """Approximate Dynamic Program Algorithm for Finity Horizon

    Paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4124940

    :param int forward_step: envmodel forward step.
    :param float gamma: discount factor.
    """

    def __init__(self,
                 *,
                 penalty: float = 1.0,
                 penalty_increase: float = 1.1,
                 penalty_delay: float = 100,
                 max_penalty: float = 1e3,
                 index=0, **kwargs):
        super().__init__(index=index, **kwargs)
        self.penalty = penalty
        self.penalty_increase = penalty_increase
        self.penalty_delay = penalty_delay
        self.max_penalty = max_penalty
        self.update_step = 0
        self.forward_step = kwargs["pre_horizon"]
        self.gamma = 1.0
        self.tb_info = dict()
        self.max_trajectory = kwargs["max_trajectory"]
        self.state_dim = kwargs["state_dim"]
        self.ref_obs_dim = kwargs["ref_obs_dim"]
        self.batch_size = kwargs["replay_batch_size"]

    @property
    def adjustable_parameters(self):
        # para_tuple = ("forward_step", "gamma")
        # return para_tuple
        return (
            *super().adjustable_parameters,
            "penalty",
            "penalty_increase",
            "penalty_delay",
        )
    #
    # def local_update(self, data, iteration: int):
    #     self._compute_gradient(data)
    #     self.networks.policy_optimizer.step()
    #     return self.tb_info

    # def get_remote_update_info(self, data: dict, iteration: int) -> Tuple[dict, dict]:
    #     self._compute_gradient(data)
    #     policy_grad = [p._grad for p in self.networks.policy.parameters()]
    #     update_info = dict()
    #     update_info["grad"] = policy_grad
    #     return self.tb_info, update_info

    # def remote_update(self, update_info: dict):
    #     for p, grad in zip(self.networks.policy.parameters(), update_info["grad"]):
    #         p.grad = grad
    #     self.networks.policy_optimizer.step()

    # def _compute_gradient(self, data):
    #     start_time = time.time()
    #     self.networks.policy.zero_grad()
    #     loss_policy = self._compute_loss_policy(deepcopy(data))
    #
    #     loss_policy.backward()
    #
    #     self.tb_info[tb_tags["loss_actor"]] = loss_policy.item()
    #
    #     end_time = time.time()
    #
    #     self.tb_info[tb_tags["alg_time"]] = (end_time - start_time) * 1000  # ms
    #
    #     return

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
        constraint = []
        v_pi_c_int = 0
        v_pi_c_ext = 0
        v_pi = torch.zeros((o.size(0), self.forward_step))
        a = self.networks.policy.forward_all_policy(o, key_padding_mask=key_padding_mask)
        for step in range(self.forward_step):
            o, r, d, info = self.envmodel.forward(o, a[:, step, :], d, info)
            v_pi[:, step] = r
            c = info["constraint"]
            constraint.append(c)
            c_int = (-torch.clamp_max(c, 0) + EPSILON).log().sum(1)
            c_ext = (torch.clamp_min(c, 0) ** 2).sum(1)
            v_pi_c_int += c_int * (self.gamma ** step)
            v_pi_c_ext += c_ext * (self.gamma ** step)
        weighted_rewards = v_pi * gamma_powers * mask
        v_pi_final = weighted_rewards.sum(dim=1)
        loss_reward = -v_pi_final.mean()
        constraint = torch.stack(constraint, dim=1)
        feasible = (constraint < 0).all(2).all(1)
        loss_constraint_int = (v_pi_c_int * feasible).mean()
        loss_constraint_ext = (v_pi_c_ext * ~feasible).mean()
        loss_policy = loss_reward + \
            1 / self.penalty * loss_constraint_int + \
            self.penalty * loss_constraint_ext
        loss_info = {
            tb_tags["loss_actor"]: loss_policy.item(),
            tb_tags["loss_actor_reward"]: loss_reward.item(),
            tb_tags["loss_actor_constraint"]: loss_constraint_ext.item(),
            "Loss/Penalty coefficient-RL iter": self.penalty,
            "Loss/Feasible ratio-RL iter": feasible.float().mean().item(),
        }
        return loss_policy, loss_info

if __name__ == "__main__":
    print("11111")
