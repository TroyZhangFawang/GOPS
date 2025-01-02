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
    "TransformerPolicy",
    "TransformerPolicy2",
    "TransformerPolicy3",
    "TransformerPolicy4",
    "TransformerPolicy5",
    "TransformerPolicy6",
    "TP7"
]


import numpy as np
import warnings
from gops.utils.common_utils import get_activation_func
from gops.utils.act_distribution_cls import Action_Distribution
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
        return x

class TransformerPolicy(nn.Module, Action_Distribution):
    def __init__(self, **kwargs):
        super(TransformerPolicy, self).__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        d_model = kwargs["d_model"]
        nhead = kwargs["nhead"]
        num_encoder_layers = kwargs["num_encoder_layers"]
        num_decoder_layers = kwargs["num_decoder_layers"]
        self.pre_horizon = kwargs["pre_horizon"]
        self.max_trajectory = kwargs["max_trajectory"]
        self.state_dim = kwargs["state_dim"]
        self.ref_obs_dim = kwargs["ref_obs_dim"]
        self.dim_feedforward = kwargs["dim_feedforward"]

        self.state_embedding = nn.Linear(self.state_dim, d_model)
        self.trajectory_embedding = nn.Linear(self.ref_obs_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward=self.dim_feedforward, batch_first=True)
        self.action_output = nn.Linear(d_model, act_dim)

        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]).float())
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]).float())
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        return self.get_all_action(obs)[:, -1, :]

    def get_all_action(self, obs):
        # state =
        state = obs[:, :self.state_dim]
        trajectory = obs[:, self.state_dim:].reshape(obs.size(0), -1, self.ref_obs_dim)
        src = self.state_embedding(state).unsqueeze(1)  # [batch_size, 1, d_model]
        tgt = self.trajectory_embedding(trajectory)  # [batch_size, seq_len, d_model]
        tgt = self.pos_encoder(tgt)

        # tgt_key_padding_mask = (trajectory == 0)

        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

        # output = self.transformer(src, tgt, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        output = self.transformer(src, tgt, tgt_mask=tgt_mask)
        actions = self.action_output(output)
        action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(actions) \
                 + (self.act_high_lim + self.act_low_lim) / 2
        # print("Check a: ", torch.isnan(action).any())
        return action
class TransformerPolicy2(nn.Module, Action_Distribution):
    def __init__(self, **kwargs):
        super(TransformerPolicy2, self).__init__()
        act_dim = kwargs["act_dim"]
        d_model = kwargs["d_model"]
        nhead = kwargs["nhead"]
        num_decoder_layers = kwargs["num_decoder_layers"]
        self.pre_horizon = kwargs["pre_horizon"]
        self.max_trajectory = kwargs["max_trajectory"]
        self.state_dim = kwargs["state_dim"]
        self.ref_obs_dim = kwargs["ref_obs_dim"]
        self.dim_feedforward = kwargs["dim_feedforward"]

        # self.state_ref_embedding = nn.Linear(self.state_dim+self.ref_obs_dim, d_model)
        self.state_embedding = nn.Linear(self.state_dim, d_model)
        self.trajectory_embedding = nn.Linear(self.ref_obs_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer_decoder = TransformerDecoder(d_model, nhead, num_decoder_layers, dim_feedforward=self.dim_feedforward, batch_first=True)
        self.action_output = nn.Linear(d_model, act_dim)

        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]).float())
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]).float())
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        return self.get_all_action(obs)[:, -1, :]

    def get_all_action(self, obs):
        # state =
        state = obs[:, :self.state_dim]
        trajectory = obs[:, self.state_dim:].reshape(obs.size(0), -1, self.ref_obs_dim)
        state_emb = self.state_embedding(state).unsqueeze(1)  # [batch_size, 1, d_model]
        ref_emb = self.trajectory_embedding(trajectory)  # [batch_size, seq_len, d_model]
        # seq_length = trajectory.shape[1]
        # state_expanded = state.unsqueeze(1).expand(-1, seq_length, -1)
        # state_ref = torch.cat((trajectory, state_expanded), dim=2)
        # tgt = self.state_ref_embedding(state_ref)  # [batch_size, seq_len, d_model]
        tgt = torch.cat((state_emb, ref_emb), dim=1)
        tgt = self.pos_encoder(tgt)

        # tgt_key_padding_mask = (trajectory == 0)

        tgt_mask = generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

        # output = self.transformer(src, tgt, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        output = self.transformer_decoder(tgt, tgt_mask=tgt_mask)[:, 1:, :]
        actions = self.action_output(output)
        action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(actions) \
                 + (self.act_high_lim + self.act_low_lim) / 2
        # print("Check a: ", torch.isnan(action).any())
        return action
class TransformerPolicy3(nn.Module, Action_Distribution):
    def __init__(self, **kwargs):
        super(TransformerPolicy3, self).__init__()
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
        self.pos_encoder = PositionalEncoding(d_model)
        self.SelfAttention = MutiSelfAttention(d_model, nhead, num_layers=num_decoder_layers, dim_feedforward=self.dim_feedforward, batch_first=True)
        # self.action_output = nn.Linear(d_model, act_dim)
        self.action_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),  # 第一层
            nn.GELU(),
            nn.Linear(d_model, d_model),  # 第二层
            nn.GELU(),
            nn.Linear(d_model, d_model),  # 第三层
            nn.GELU(),
            nn.Linear(d_model, act_dim)  # 输出层
        )
        self.activation = F.gelu
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]).float())
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]).float())
        self.action_distribution_cls = kwargs["action_distribution_cls"]
    # def forward(self, obs):
    #     return self.get_all_action(obs)[:, -1, :]

    def forward(self, obs):
        # state =
        state = obs[:, :self.state_dim]
        trajectory = obs[:, self.state_dim:].reshape(obs.size(0), -1, self.ref_obs_dim)
        state_emb = self.state_embedding(state).unsqueeze(1)  # [batch_size, 1, d_model]
        # state_emb = self.activation(state_emb)
        ref_emb = self.trajectory_embedding(trajectory)  # [batch_size, seq_len, d_model]
        # ref_emb = self.activation(ref_emb)
        tgt = torch.cat((state_emb, ref_emb), dim=1)
        tgt = self.pos_encoder(tgt)
        output = self.SelfAttention(tgt)
        actions = self.action_mlp(output)
        action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(actions) \
                 + (self.act_high_lim + self.act_low_lim) / 2
        return action

class TransformerPolicy4(nn.Module, Action_Distribution):
    def __init__(self, **kwargs):
        super(TransformerPolicy4, self).__init__()
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
        self.SelfAttention = MutiSelfAttention2(d_model, nhead, select_dim=0, num_layers=num_decoder_layers, dim_feedforward=self.dim_feedforward, batch_first=True)
        self.action_output = nn.Linear(d_model, self.act_dim * self.pre_horizon)

        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]).float())
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]).float())
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        return self.forward_all_policy(obs)[:, 0, :]

    def forward_all_policy(self, obs):
        # state =
        state = obs[:, :self.state_dim]
        trajectory = obs[:, self.state_dim:].reshape(obs.size(0), -1, self.ref_obs_dim)
        state_emb = self.state_embedding(state).unsqueeze(1)  # [batch_size, 1, d_model]
        ref_emb = self.trajectory_embedding(trajectory)  # [batch_size, seq_len, d_model]
        tgt = torch.cat((state_emb, ref_emb), dim=1)

        # output = self.transformer(src, tgt, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        output = self.SelfAttention(tgt)
        output = self.action_output(output)
        actions = output.reshape(obs.shape[0], self.pre_horizon, self.act_dim)
        action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(actions) \
                 + (self.act_high_lim + self.act_low_lim) / 2
        # print("Check a: ", torch.isnan(action).any())
        return action
class TransformerPolicy5(nn.Module, Action_Distribution):
    def __init__(self, **kwargs):
        super(TransformerPolicy5, self).__init__()
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
        self.action_output = nn.Linear((self.pre_horizon+1)*d_model, self.act_dim * self.pre_horizon)

        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]).float())
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]).float())
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        return self.forward_all_policy(obs)[:, 0, :]

    def forward_all_policy(self, obs):
        # state =
        state = obs[:, :self.state_dim]
        trajectory = obs[:, self.state_dim:].reshape(obs.size(0), -1, self.ref_obs_dim)
        state_emb = self.state_embedding(state).unsqueeze(1)  # [batch_size, 1, d_model]
        ref_emb = self.trajectory_embedding(trajectory)  # [batch_size, seq_len, d_model]
        tgt = torch.cat((state_emb, ref_emb), dim=1)

        # output = self.transformer(src, tgt, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        output = self.SelfAttention(tgt)
        output = output.view(obs.size(0), -1)
        output = self.action_output(output)
        actions = output.reshape(obs.shape[0], self.pre_horizon, self.act_dim)
        action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(actions) \
                 + (self.act_high_lim + self.act_low_lim) / 2
        # print("Check a: ", torch.isnan(action).any())
        return action
class TransformerPolicy6(nn.Module, Action_Distribution):
    def __init__(self, **kwargs):
        super(TransformerPolicy6, self).__init__()
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

    def forward_all_policy(self, obs):
        # state =
        state = obs[:, :self.state_dim]
        trajectory = obs[:, self.state_dim:].reshape(obs.size(0), -1, self.ref_obs_dim)
        state_emb = self.state_embedding(state).unsqueeze(1)  # [batch_size, 1, d_model]
        ref_emb = self.trajectory_embedding(trajectory)  # [batch_size, seq_len, d_model]
        tgt = torch.cat((ref_emb, state_emb), dim=1)
        tgt_mask = generate_lower_triangle_mask(tgt.size(1)).to(tgt.device)
        # output = self.transformer(src, tgt, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        output = self.SelfAttention(tgt, attn_mask=tgt_mask)
        output = output[:, :self.pre_horizon, :]
        actions = self.action_output(output)
        action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(actions) \
                 + (self.act_high_lim + self.act_low_lim) / 2
        # print("Check a: ", torch.isnan(action).any())
        return action
# class TP7(nn.Module, Action_Distribution):
#     def __init__(self, **kwargs):
#         super(TP7, self).__init__()
#         self.act_dim = kwargs["act_dim"]
#         d_model = kwargs["d_model"]
#         nhead = kwargs["nhead"]
#         num_decoder_layers = kwargs["num_decoder_layers"]
#         self.pre_horizon = kwargs["pre_horizon"]
#         self.max_trajectory = kwargs["max_trajectory"]
#         self.state_dim = kwargs["state_dim"]
#         self.ref_obs_dim = kwargs["ref_obs_dim"]
#         self.dim_feedforward = kwargs["dim_feedforward"]
#         self.pos_encoder = PositionalEncoding(d_model)
#         self.state_ref_embedding = nn.Linear(self.state_dim + self.ref_obs_dim, d_model)
#         self.pos_encoder = PositionalEncoding(d_model)
#         self.SelfAttention = MutiSelfAttention2(d_model, nhead, select_dim=None, num_layers=num_decoder_layers, dim_feedforward=self.dim_feedforward, batch_first=True)
#         self.action_output = nn.Linear(d_model, self.act_dim)
#
#         self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]).float())
#         self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]).float())
#         self.action_distribution_cls = kwargs["action_distribution_cls"]
#
#     def forward(self, obs):
#         return self.forward_all_policy(obs)[:, 0, :]
#
#     def forward_all_policy(self, obs, key_padding_mask=None):
#         state = obs[:, :self.state_dim]
#         trajectory = obs[:, self.state_dim:].reshape(obs.size(0), -1, self.ref_obs_dim)
#         seq_length = trajectory.shape[1]
#         state_expanded = state.unsqueeze(1).expand(-1, seq_length, -1)
#         state_ref = torch.cat((state_expanded, trajectory), dim=2)
#         tgt = self.state_ref_embedding(state_ref)
#         tgt = self.pos_encoder(tgt)
#         # output = self.transformer(src, tgt, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
#         output = self.SelfAttention(tgt, key_padding_mask=key_padding_mask)
#         actions = self.action_output(output)
#         action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(actions) \
#                  + (self.act_high_lim + self.act_low_lim) / 2
#         # print("Check a: ", torch.isnan(action).any())
#         return action

class TP7(nn.Module, Action_Distribution):
    def __init__(self, **kwargs):
        super(TP7, self).__init__()
        self.act_dim = kwargs["act_dim"]  # 动作维度，例如 2
        d_model = kwargs["d_model"]  # 模型维度
        nhead = kwargs["nhead"]  # 多头注意力的头数
        self.state_dim = kwargs["state_dim"]  # 状态维度
        self.ref_obs_dim = kwargs["ref_obs_dim"]  # 参考观测维度
        self.dim_feedforward = kwargs.get("dim_feedforward", 2048)  # 前馈网络维度

        # 输入编码部分，使用 MLP 和 GELU 激活函数
        self.input_mlp = nn.Sequential(
            nn.Linear(self.state_dim + self.ref_obs_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.pos_encoder = PositionalEncoding(d_model)

        # 自注意力层
        self.self_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        # 使用 MLP 进行动作输出，使用 GELU 激活函数
        self.action_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.act_dim)
        )

        # 动作空间的高低限制
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]).float())
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]).float())
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        return self.forward_all_policy(obs)[:, 0, :]

    def forward_all_policy(self, obs, key_padding_mask=None):
        # 提取状态和轨迹
        state = obs[:, :self.state_dim]  # (batch_size, state_dim)
        trajectory = obs[:, self.state_dim:].reshape(obs.size(0), -1,
                                                     self.ref_obs_dim)  # (batch_size, seq_length, ref_obs_dim)

        # 将状态扩展并与轨迹拼接
        seq_length = trajectory.size(1)
        state_expanded = state.unsqueeze(1).expand(-1, seq_length, -1)  # (batch_size, seq_length, state_dim)
        state_ref = torch.cat((state_expanded, trajectory), dim=2)  # (batch_size, seq_length, state_dim + ref_obs_dim)

        # 通过输入 MLP 进行编码
        tgt = self.input_mlp(state_ref)  # (batch_size, seq_length, d_model)
        tgt = self.pos_encoder(tgt)  # (batch_size, seq_length, d_model)

        # 经过自注意力层
        attn_output, _ = self.self_attention(tgt, tgt, tgt , key_padding_mask=key_padding_mask)  # (batch_size, seq_length, d_model)

        # 通过动作 MLP 进行输出
        actions = self.action_mlp(attn_output)  # (batch_size, seq_length, act_dim)

        # 经过激活函数
        actions = torch.tanh(actions)

        # 映射到动作空间
        action = (self.act_high_lim - self.act_low_lim) / 2 * actions \
                 + (self.act_high_lim + self.act_low_lim) / 2
        return action


# class TP7(nn.Module, Action_Distribution):
#     def __init__(self, **kwargs):
#         super(TP7, self).__init__()
#         self.act_dim = kwargs["act_dim"]
#         d_model = kwargs["d_model"]
#         nhead = kwargs["nhead"]
#         num_decoder_layers = kwargs["num_decoder_layers"]
#         self.pre_horizon = kwargs["pre_horizon"]
#         self.max_trajectory = kwargs["max_trajectory"]
#         self.state_dim = kwargs["state_dim"]
#         self.ref_obs_dim = kwargs["ref_obs_dim"]
#         self.dim_feedforward = kwargs["dim_feedforward"]
#         self.pos_encoder = PositionalEncoding(d_model)
#         self.state_embedding = nn.Linear(self.state_dim, d_model)
#         self.trajectory_embedding = nn.Linear(self.ref_obs_dim, d_model)
#         self.pos_encoder = PositionalEncoding(d_model)
#         self.SelfAttention = MutiSelfAttention2(d_model, nhead, select_dim=None, num_layers=num_decoder_layers, dim_feedforward=self.dim_feedforward, batch_first=True)
#         self.action_output = nn.Linear(d_model, self.act_dim)
#
#         self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]).float())
#         self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]).float())
#         self.action_distribution_cls = kwargs["action_distribution_cls"]
#
#     def forward(self, obs):
#         return self.forward_all_policy(obs)[:, 0, :]
#
#     def forward_all_policy(self, obs, key_padding_mask=None):
#         # state =
#         state = obs[:, :self.state_dim]
#         trajectory = obs[:, self.state_dim:].reshape(obs.size(0), -1, self.ref_obs_dim)
#         state_emb = self.state_embedding(state).unsqueeze(1)  # [batch_size, 1, d_model]
#         ref_emb = self.trajectory_embedding(trajectory)  # [batch_size, seq_len, d_model]
#         tgt = torch.cat((state_emb, ref_emb), dim=1)
#         tgt = self.pos_encoder(tgt)
#         # output = self.transformer(src, tgt, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
#         output = self.SelfAttention(tgt, key_padding_mask=key_padding_mask)
#         output = output[:, 1:, :]
#         actions = self.action_output(output)
#         action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(actions) \
#                  + (self.act_high_lim + self.act_low_lim) / 2
#         # print("Check a: ", torch.isnan(action).any())
#         return action


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

        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = F.relu if activation == 'relu' else F.gelu

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
        tgt2 = self.activation(tgt2)
        tgt2 = self.linear2(tgt2)
        tgt = tgt + self.activation(tgt2)
        tgt = self.norm1(tgt)
        return tgt

class SelfAttentionWithAddNorm(nn.Module):
    def __init__(self, d_model, nhead, batch_first=True):
        super(SelfAttentionWithAddNorm, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=batch_first)
        self.norm1 = nn.LayerNorm(d_model)

    def forward(self, tgt, attn_mask=None, key_padding_mask=None):
        # Multi-Head Self-Attention
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        tgt = tgt + tgt2
        tgt = self.norm1(tgt)
        # Add & Norm
        return tgt

class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', layer_norm_eps=1e-5,
                 batch_first=True):
        super(CustomTransformerDecoderLayer, self).__init__()
        # 第一个自注意力层（Masked Multi-Head Attention）
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        # 第二个自注意力层（替代 Cross-Attention）
        self.self_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = F.relu if activation == 'relu' else F.gelu
        # LayerNorm layers
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)


    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.self_attn2(tgt, tgt, tgt, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear1(tgt)
        tgt2 = self.dropout(self.activation(tgt2))
        tgt2 = self.linear2(tgt2)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1, activation='relu', layer_norm_eps=1e-5, batch_first=True):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            CustomTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first)
            for _ in range(num_layers)
        ])
        # self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None):
        output = tgt
        for layer in self.layers:
            output = layer(output, tgt_mask, tgt_key_padding_mask)
        # output = self.norm(output)
        return output


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
