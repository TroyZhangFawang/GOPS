


__all__ = [
    "ActionValue",
    "ActionValueDis",
    "ActionValueDistri",
    "StochaPolicyDis",
    "StateValue",
    "TTTPolicy",
    "TTTPolicy2",
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
from gops.utils.ttt import TTTModel, TTTConfig
def generate_square_subsequent_mask(sz):
    """
    Generate a square mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    mask = torch.triu(torch.ones(sz, sz), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


class TTTPolicy(nn.Module, Action_Distribution):
    def __init__(self, **kwargs):
        super(TTTPolicy, self).__init__()
        act_dim = kwargs["act_dim"]
        d_model = kwargs["d_model"]
        nhead = kwargs["nhead"]
        num_decoder_layers = kwargs["num_decoder_layers"]
        self.pre_horizon = kwargs["pre_horizon"]
        self.max_trajectory = kwargs["max_trajectory"]
        self.state_dim = kwargs["state_dim"]
        self.ref_obs_dim = kwargs["ref_obs_dim"]
        self.dim_feedforward = kwargs["dim_feedforward"]
        self.state_embedding = nn.Linear(self.state_dim, d_model)
        self.trajectory_embedding = nn.Linear(self.ref_obs_dim, d_model)
        # self.pos_encoder = PositionalEncoding(d_model)
        self.configuration = TTTConfig()
        self.TTT = TTTModel(config=self.configuration)
        self.action_output = nn.Linear(d_model, act_dim)
        self.activation = F.gelu
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]).float())
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]).float())
        self.action_distribution_cls = kwargs["action_distribution_cls"]

        self.cache_initialized = False  # 增加一个标志位来检测缓存是否已初始化
    # def forward(self, obs):
    #     return self.get_all_action(obs)[:, -1, :]

    def forward(self, obs):
        state = obs[:, :self.state_dim]
        trajectory = obs[:, self.state_dim:].reshape(obs.size(0), -1, self.ref_obs_dim)
        state_emb = self.state_embedding(state).unsqueeze(1)  # [batch_size, 1, d_model]
        # state_emb = self.activation(state_emb)
        ref_emb = self.trajectory_embedding(trajectory)  # [ba8tch_size, seq_len, d_model]

        # ref_emb = self.activation(ref_emb)
        tgt = torch.cat((state_emb, ref_emb), dim=1)
        output = self.TTT(inputs_embeds=tgt)[0]
        output = output[:,-1,:]
        actions = self.action_output(output)
        action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(actions) \
                 + (self.act_high_lim + self.act_low_lim) / 2
        return action
class TTTPolicy2(nn.Module, Action_Distribution):
    def __init__(self, **kwargs):
        super(TTTPolicy2, self).__init__()
        act_dim = kwargs["act_dim"]
        d_model = kwargs["d_model"]
        nhead = kwargs["nhead"]
        num_decoder_layers = kwargs["num_decoder_layers"]
        self.pre_horizon = kwargs["pre_horizon"]
        self.max_trajectory = kwargs["max_trajectory"]
        self.state_dim = kwargs["state_dim"]
        self.ref_obs_dim = kwargs["ref_obs_dim"]
        self.dim_feedforward = kwargs["dim_feedforward"]
        self.state_ref_embedding = nn.Linear(self.state_dim+self.ref_obs_dim, d_model)
        # self.pos_encoder = PositionalEncoding(d_model)
        self.configuration = TTTConfig()
        self.TTT1 = TTTModel(config=self.configuration)
        self.TTT2 = TTTModel(config=self.configuration)
        self.action_output = nn.Linear(2*d_model, act_dim)
        self.activation = F.gelu
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]).float())
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]).float())
        self.action_distribution_cls = kwargs["action_distribution_cls"]

        self.cache_initialized = False  # 增加一个标志位来检测缓存是否已初始化

    def forward(self, obs):
        return self.forward_all_policy(obs)[:, 0, :]

    def forward_all_policy(self, obs, lengths=None):
        state = obs[:, :self.state_dim]
        trajectory = obs[:, self.state_dim:].reshape(obs.size(0), -1, self.ref_obs_dim)
        seq_length = trajectory.shape[1]
        state_expanded = state.unsqueeze(1).expand(-1, seq_length, -1)
        state_ref = torch.cat((state_expanded, trajectory), dim=2)
        tgt = self.state_ref_embedding(state_ref)
        tgt_reversed = torch.flip(tgt, [1])
        output1 = self.TTT1(inputs_embeds=tgt)[0]
        output2 = self.TTT2(inputs_embeds=tgt_reversed)[0]
        output2 = torch.flip(output2, [1])
        output = torch.cat((output1, output2), dim=2)
        # output = output[:,-1,:]
        actions = self.action_output(output)
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
