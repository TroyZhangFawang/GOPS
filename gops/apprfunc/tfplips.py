'''
TransformerPolicy is encoder and decoder
TransformerPolicy2 is decoder only
TransformerPolicy3 is self-attention and feedforward
the above three is output the only one action
TransformerPolicy4 outputs a sequence of actions,it only takes the first vector output by the attention layer
TransformerPolicy5 outputs a sequence of actions,taking a concat of all output vectors from the attention layer
TransformerPolicy6 :The state information is placed at the end, and the action is derived from the output of the pre-horizon one, with a reverse mask
456 is based on 3
'''


__all__ = [
    "ActionValue",
    "ActionValueDis",
    "ActionValueDistri",
    "StochaPolicyDis",
    "StateValue",
    "TP7lips"
]
"""
修改D:\\anaconda3\\envs\\gops\\Libsite-packagesltorch\\nn\\functional.py源码,实现lips连续，主要改动是
def l2_attention_with_softmax(q, k, scale_factor):
    参数:
    q (torch.Tensor): 查询张量，形状为 (B, Nt, E)。
    k (torch.Tensor): 键张量，形状为 (B, Nt, E)。
    scale_factor (float): 缩放因子，通常是 sqrt(D/H)。
    返回:
    torch.Tensor: 注意力权重矩阵，形状为 (B, Nt, Nt)。
    B, Nt, E = q.shape
    q_expanded = q.unsqueeze(2)  # (B, Nt, 1, E)
    k_expanded = k.unsqueeze(1)  # (B, 1, Nt, E)
    dist_squared = torch.sum((q_expanded - k_expanded) ** 2, dim=-1)  # (B, Nt, Nt)
    attn_output_weights = torch.exp(-dist_squared / scale_factor)
    attn_output_weights = F.softmax(attn_output_weights, dim=-1)
    return attn_output_weights
修改D:anaconda3\\envs\\gops\\Lib\\site-packagesltorch\\nn\\modules\\activation.py绑定WQ WK权重
"""

import numpy as np
import warnings
from gops.utils.common_utils import get_activation_func
from gops.utils.act_distribution_cls import Action_Distribution
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from gops.utils.ttt import TTTModel, TTTConfig

def generate_square_subsequent_mask(sz):
    """
    Generate a square mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    mask = torch.triu(torch.ones(sz, sz), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask
def generate_lower_triangle_mask(sz):
    """
    Generate a lower triangular mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    mask = torch.tril(torch.ones(sz, sz), diagonal=-1)
    mask = mask.masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
    return mask

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 方便计算
        pe = pe.unsqueeze(0)
        # 保存model
        self.register_buffer("pe", pe)

    def forward(self, x):
        # 将x和positional encoding相加。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

class TP7lips(nn.Module, Action_Distribution):
    def __init__(self, **kwargs):
        super(TP7lips, self).__init__()
        self.act_dim = kwargs["act_dim"]
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
        self.pos_encoder = PositionalEncoding(d_model)
        self.SelfAttention = MutiSelfAttention2(d_model, nhead, select_dim=None, num_layers=num_decoder_layers, dim_feedforward=self.dim_feedforward, batch_first=True)
        self.action_output = nn.Linear(d_model, self.act_dim)

        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]).float())
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]).float())
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        return self.forward_all_policy(obs)[:, 0, :]

    def forward_all_policy(self, obs, key_padding_mask=None):
        # state =
        state = obs[:, :self.state_dim]
        trajectory = obs[:, self.state_dim:].reshape(obs.size(0), -1, self.ref_obs_dim)
        state_emb = self.state_embedding(state).unsqueeze(1)  # [batch_size, 1, d_model]
        ref_emb = self.trajectory_embedding(trajectory)  # [batch_size, seq_len, d_model]
        tgt = torch.cat((state_emb, ref_emb), dim=1)
        # output = self.transformer(src, tgt, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        output = self.SelfAttention(tgt, key_padding_mask=key_padding_mask)
        output = output[:, 1:, :]
        actions = self.action_output(output)
        action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(actions) \
                 + (self.act_high_lim + self.act_low_lim) / 2
        # print("Check a: ", torch.isnan(action).any())
        return action
class MutiSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=1024, dropout=0.1, activation='gelu',  batch_first=True):
        super(MutiSelfAttention, self).__init__()
        self.layers = nn.ModuleList([
            SelfAttentionWithAddNorm(d_model, nhead, batch_first=batch_first)
            for _ in range(num_layers)
        ])
        self.norm1 = nn.LayerNorm(d_model)
        # self.norm = nn.LayerNorm(d_model)

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = F.relu if activation == 'relu' else F.gelu

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, tgt, attn_mask=None, key_padding_mask=None):
        output = tgt
        for layer in self.layers:
            output = layer(output, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        # tgt = self.norm(output)
        tgt = output[:, 0, :]
        tgt2 = self.linear1(tgt)
        tgt2 = self.dropout(self.activation(tgt2))
        tgt2 = self.linear2(tgt2)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm1(tgt)

        return tgt
class MutiSelfAttention2(nn.Module):
    def __init__(self, d_model, nhead, num_layers, select_dim=0, dim_feedforward=1024, dropout=0.1, activation='relu',  batch_first=True):
        super(MutiSelfAttention2, self).__init__()
        self.layers = nn.ModuleList([
            SelfAttentionWithAddNorm(d_model, nhead, batch_first=batch_first)
            for _ in range(num_layers)
        ])
        self.norm1 = nn.LayerNorm(d_model)
        # self.norm = nn.LayerNorm(d_model)
        self.select_dim = select_dim
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = F.relu if activation == 'relu' else F.gelu

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, tgt, attn_mask=None, key_padding_mask=None):
        output = tgt
        for layer in self.layers:
            output = layer(output, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        # tgt = self.norm(output)
        if self.select_dim is None:
            tgt = output
        else:
            tgt = output[:, self.select_dim, :]
        tgt2 = self.linear1(tgt)
        tgt2 = self.dropout(self.activation(tgt2))
        tgt2 = self.linear2(tgt2)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm1(tgt)
        return tgt

class SelfAttentionWithAddNorm(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, batch_first=True):
        super(SelfAttentionWithAddNorm, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, tgt, attn_mask=None, key_padding_mask=None):
        # Multi-Head Self-Attention
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        # tgt = tgt[:, 0, :] + self.dropout1(tgt2[:, 0, :])
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # Add & Norm
        return tgt


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
