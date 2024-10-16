


__all__ = [
    "ActionValue",
    "ActionValueDis",
    "ActionValueDistri",
    "StochaPolicyDis",
    "StateValue",
    "GRUPolicy",
    "GRUFullPolicy",
    "GRUFullPolicy2",
]


import numpy as np
import warnings
from gops.utils.common_utils import get_activation_func
from gops.utils.act_distribution_cls import Action_Distribution
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def generate_square_subsequent_mask(sz):
    """
    Generate a square mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    mask = torch.triu(torch.ones(sz, sz), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


class GRUPolicy(nn.Module, Action_Distribution):
    def __init__(self, **kwargs):
        super(GRUPolicy, self).__init__()
        act_dim = kwargs["act_dim"]
        self.pre_horizon = kwargs["pre_horizon"]
        self.state_dim = kwargs["state_dim"]
        self.ref_obs_dim = kwargs["ref_obs_dim"]
        self.hidden_dim = kwargs["hidden_dim"]
        self.num_layers = kwargs["num_layers"]
        input_dim = self.state_dim+self.ref_obs_dim
        self.bidirectional = kwargs["bidirectional"]
        # GRU 层
        self.gru = nn.GRU(input_dim, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=self.bidirectional)

        # 全连接层，用于从GRU输出到最终输出
        if self.bidirectional == True:
            self.fc = nn.Linear(2*self.hidden_dim, act_dim)
        else:
            self.fc = nn.Linear(self.hidden_dim, act_dim)
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]).float())
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]).float())
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        state = obs[:, :self.state_dim]
        trajectory = obs[:, self.state_dim:].reshape(obs.size(0), -1, self.ref_obs_dim)
        seq_length = trajectory.shape[1]
        state_expanded = state.unsqueeze(1).expand(-1, seq_length, -1)
        state_ref = torch.cat((state_expanded, trajectory), dim=2)
        # 初始化隐藏状态
        if self.bidirectional:
            h0 = torch.zeros(2*self.num_layers, state_ref.size(0), self.hidden_dim).to(state_ref.device)
        else:
            h0 = torch.zeros(self.num_layers, state_ref.size(0), self.hidden_dim).to(state_ref.device)
        # 前向传播GRU
        out, _ = self.gru(state_ref, h0)

        # 获取序列的最后一次输出作为结果
        out = out[:, -1, :]
        actions = self.fc(out)
        action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(actions) \
                 + (self.act_high_lim + self.act_low_lim) / 2
        return action
class GRUFullPolicy(nn.Module, Action_Distribution):
    def __init__(self, **kwargs):
        super(GRUFullPolicy, self).__init__()
        act_dim = kwargs["act_dim"]
        self.pre_horizon = kwargs["pre_horizon"]
        self.state_dim = kwargs["state_dim"]
        self.ref_obs_dim = kwargs["ref_obs_dim"]
        self.hidden_dim = kwargs["hidden_dim"]
        self.num_layers = kwargs["num_layers"]
        input_dim = self.state_dim+self.ref_obs_dim
        self.bidirectional = kwargs["bidirectional"]
        # GRU 层
        self.gru = nn.GRU(input_dim, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=self.bidirectional)

        # 全连接层，用于从GRU输出到最终输出
        if self.bidirectional == True:
            self.fc = nn.Linear(2*self.hidden_dim, act_dim)
        else:
            self.fc = nn.Linear(self.hidden_dim, act_dim)
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]).float())
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]).float())
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        return self.forward_all_policy(obs)[:, 0, :]

    def forward_all_policy(self, obs, lengths=None):
        state = obs[:, :self.state_dim]
        trajectory = obs[:, self.state_dim:].reshape(obs.size(0), -1, self.ref_obs_dim)
        seq_length = trajectory.shape[1]
        state_expanded = state.unsqueeze(1).expand(-1, seq_length, -1)
        state_ref = torch.cat((state_expanded, trajectory), dim=2)
        if lengths is not None:
            tgt = pack_padded_sequence(state_ref, lengths, batch_first=True, enforce_sorted=False)
        else:
            tgt = state_ref
        # 初始化隐藏状态
        if self.bidirectional:
            h0 = torch.zeros(2*self.num_layers, state_ref.size(0), self.hidden_dim).to(state_ref.device)
        else:
            h0 = torch.zeros(self.num_layers, state_ref.size(0), self.hidden_dim).to(state_ref.device)
        # 前向传播GRU
        out, _ = self.gru(tgt, h0)
        if lengths is not None:
            outputs, output_lengths = pad_packed_sequence(out, batch_first=True, total_length=self.pre_horizon)
        else:
            outputs = out
        # 获取序列的最后一次输出作为结果
        # out = out[:, -1, :]
        actions = self.fc(outputs)
        action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(actions) \
                 + (self.act_high_lim + self.act_low_lim) / 2
        return action

class GRUFullPolicy2(nn.Module, Action_Distribution):
    def __init__(self, **kwargs):
        super(GRUFullPolicy2, self).__init__()
        act_dim = kwargs["act_dim"]
        self.pre_horizon = kwargs["pre_horizon"]
        self.state_dim = kwargs["state_dim"]
        self.ref_obs_dim = kwargs["ref_obs_dim"]
        self.hidden_dim = kwargs["hidden_dim"]
        self.num_layers = kwargs["num_layers"]
        input_dim = self.state_dim+self.ref_obs_dim
        self.bidirectional = kwargs["bidirectional"]
        # GRU 层
        self.gru = nn.GRU(input_dim, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=self.bidirectional)

        # 全连接层，用于从GRU输出到最终输出
        if self.bidirectional == True:
            self.fc = nn.Linear(2*self.hidden_dim, act_dim)
        else:
            self.fc = nn.Linear(self.hidden_dim, act_dim)
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]).float())
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]).float())
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        return self.forward_all_policy(obs)[:, 0, :]

    def forward_all_policy(self, obs):
        state = obs[:, :self.state_dim]
        trajectory = obs[:, self.state_dim:].reshape(obs.size(0), -1, self.ref_obs_dim)
        seq_length = trajectory.shape[1]
        state_expanded = state.unsqueeze(1).expand(-1, seq_length, -1)
        state_ref = torch.cat((state_expanded, trajectory), dim=2)
        # 初始化隐藏状态
        if self.bidirectional:
            h0 = torch.zeros(2*self.num_layers, state_ref.size(0), self.hidden_dim).to(state_ref.device)
        else:
            h0 = torch.zeros(self.num_layers, state_ref.size(0), self.hidden_dim).to(state_ref.device)
        # 前向传播GRU
        out, _ = self.gru(state_ref, h0)
        # 获取序列的最后一次输出作为结果
        # out = out[:, -1, :]
        actions = self.fc(out)
        action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(actions) \
                 + (self.act_high_lim + self.act_low_lim) / 2
        return action

# Define MLP function
def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


# Count parameter number of MLP
def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])




class ActionValue(nn.Module, Action_Distribution):
    """
    Approximated function of action-value function.
    Input: observation, action.
    Output: action-value.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.q = mlp(
            [obs_dim + act_dim] + list(hidden_sizes) + [1],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)


class ActionValueDis(nn.Module, Action_Distribution):
    """
    Approximated function of action-value function for discrete action space.
    Input: observation.
    Output: action-value for all action.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_num = kwargs["act_num"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.q = mlp(
            [obs_dim] + list(hidden_sizes) + [act_num],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        return self.q(obs)


class ActionValueDistri(nn.Module):
    """
    Approximated function of distributed action-value function.
    Input: observation.
    Output: parameters of action-value distribution.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.q = mlp(
            [obs_dim + act_dim] + list(hidden_sizes) + [2],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        if "min_log_std"  in kwargs or "max_log_std" in kwargs:
            warnings.warn("min_log_std and max_log_std are deprecated in ActionValueDistri.")

    def forward(self, obs, act):
        logits = self.q(torch.cat([obs, act], dim=-1))
        value_mean, value_std = torch.chunk(logits, chunks=2, dim=-1)
        value_log_std = torch.nn.functional.softplus(value_std) 
        
        return torch.cat((value_mean, value_log_std), dim=-1)


class StochaPolicyDis(ActionValueDis, Action_Distribution):
    """
    Approximated function of stochastic policy for discrete action space.
    Input: observation.
    Output: parameters of action distribution.
    """

    pass


class StateValue(nn.Module, Action_Distribution):
    """
    Approximated function of state-value function.
    Input: observation, action.
    Output: state-value.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.v = mlp(
            [obs_dim] + list(hidden_sizes) + [1],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        v = self.v(obs)
        return torch.squeeze(v, -1)
