#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Convolutional Neural NetworksAction (CNN)
#  Update: 2021-03-05, Wenjun Zou: create CNN function


__all__ = [
    "DetermPolicy",
    "FiniteHorizonPolicy",
    "StochaPolicy",
    "ActionValue",
    "ActionValueDis",
    "StateValue",
    "ActionValueDistri",
]

import torch
import warnings
import torch.nn as nn
import torchvision.models as models
from torch.distributions.categorical import Categorical as categorical
from torch.distributions import MultivariateNormal, Normal
from gops.utils.common_utils import get_activation_func
from gops.utils.act_distribution_cls import Action_Distribution
import torchvision.transforms as transforms
import gops.env.env_gym.recources.const as const
import copy
from PIL import Image
import numpy as np
import torch.nn.functional as F
def CNN(kernel_sizes, channels, strides, activation, input_channel):
    """Implementation of CNN.
    :param list kernel_sizes: list of kernel_size,
    :param list channels: list of channels,
    :param list strides: list of stride,
    :param activation: activation function,
    :param int input_channel: number of channels of input image.
    Return CNN.
    Input shape for CNN: (batch_size, channel_num, height, width).
    """
    layers = []
    for j in range(len(kernel_sizes)):
        act = activation
        if j == 0:
            layers += [
                nn.Conv2d(input_channel, channels[j], kernel_sizes[j], strides[j]),
                act(),
            ]
        else:
            layers += [
                nn.Conv2d(channels[j - 1], channels[j], kernel_sizes[j], strides[j]),
                act(),
            ]
    return nn.Sequential(*layers)


def tensor_transform(tensor):
    # 重塑tensor为(x, 3, 384, 800)
    x = tensor.shape[0]
    tensor = tensor.view(x, 3, 384, 800)

    # 归一化
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())

    # 转为灰度图
    tensor = 0.299 * tensor[:, 0, :, :] + 0.587 * tensor[:, 1, :, :] + 0.114 * tensor[:, 2, :, :]
    tensor = tensor.unsqueeze(1)  # 添加通道维度，变为(x, 1, 384, 800)

    # 调整大小
    tensor = F.interpolate(tensor, size=(120, 160), mode='bilinear', align_corners=False)

    # 标准化
    tensor = (tensor - 0.5) / 0.5

    return tensor


def image_transform(image):
    # 确保输入是tensor
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image)

    # 重塑tensor为(1, 3, 384, 800)
    image = image.view(image.size(0), 3, 384, 800)

    # 归一化
    image = (image - image.min()) / (image.max() - image.min())

    # 缩放到0-255范围
    image = image * 255

    # # 转换为uint8类型
    # image = image.to(torch.uint8)

    # 定义transform
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((120, 160)),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # 应用transform
    image = transform(image)

    return image
def numpy_to_pil(image_):
    image = copy.deepcopy(image_)

    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    image *= 255
    image = image.astype('uint8')

    im_obj = Image.fromarray(image)
    return im_obj

class ResBlock(nn.Module):
    def __init__(self, n_channels):
        super(ResBlock, self).__init__()
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(n_channels)

    def forward(self, x):
        out = nn.ReLU()(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        out = self.bn2(out)

        return out + x


class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImpalaBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.res1 = ResBlock(out_channels)
        self.res2 = ResBlock(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2)(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class ImpalaCNN(nn.Module):
    def __init__(self):
        super(ImpalaCNN, self).__init__()
        self.network = nn.Sequential(
            ImpalaBlock(in_channels=3, out_channels=16),
            ImpalaBlock(in_channels=16, out_channels=32),
            ImpalaBlock(in_channels=32, out_channels=64),
            nn.Flatten()
        )
        self.output_size = 13056

    def forward(self, x):
        return self.network(x)


class LargeImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LargeImpalaBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.res1 = ResBlock(out_channels)
        self.res2 = ResBlock(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2)(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class LargeImpalaCNN(nn.Module):
    def __init__(self):
        super(LargeImpalaCNN, self).__init__()
        self.network = nn.Sequential(
            LargeImpalaBlock(in_channels=3, out_channels=16),
            LargeImpalaBlock(in_channels=16, out_channels=32),
            LargeImpalaBlock(in_channels=32, out_channels=64),
            nn.Flatten()
        )

    def forward(self, x):
        return self.network(x)


# Define MLP function
def MLP(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class DetermPolicy(nn.Module, Action_Distribution):
    """
    Approximated function of deterministic policy.
    Input: observation.
    Output: action.
    """

    def __init__(self, **kwargs):
        super(DetermPolicy, self).__init__()
        act_dim = kwargs["act_dim"]
        obs_dim = kwargs["obs_dim"]
        conv_type = kwargs["conv_type"]
        act_high_lim = kwargs["act_high_lim"]
        act_low_lim = kwargs["act_low_lim"]
        self.register_buffer("act_high_lim", torch.from_numpy(act_high_lim))
        self.register_buffer("act_low_lim", torch.from_numpy(act_low_lim))
        self.hidden_activation = get_activation_func(kwargs["hidden_activation"])
        self.output_activation = get_activation_func(kwargs["output_activation"])
        self.action_distribution_cls = kwargs["action_distribution_cls"]
        if conv_type == "type_1":
            # CNN+MLP Parameters
            conv_kernel_sizes = [8, 4, 3]
            conv_channels = [32, 64, 64]
            conv_strides = [4, 2, 1]
            conv_activation = nn.ReLU
            conv_input_channel = obs_dim[0]
            mlp_hidden_layers = [512, 256]

            # Construct CNN+MLP
            self.conv = CNN(
                conv_kernel_sizes,
                conv_channels,
                conv_strides,
                conv_activation,
                conv_input_channel,
            )
            conv_num_dims = (
                self.conv(torch.ones(obs_dim).unsqueeze(0)).reshape(1, -1).shape[-1]
            )
            mlp_sizes = [conv_num_dims] + mlp_hidden_layers + [act_dim]

            self.mlp = MLP(mlp_sizes, self.hidden_activation, self.output_activation)

        elif conv_type == "type_2":
            # CNN+MLP Parameters
            conv_kernel_sizes = [4, 3, 3, 3, 3, 3]
            conv_channels = [8, 16, 32, 64, 128, 256]
            conv_strides = [2, 2, 2, 2, 1, 1]
            conv_activation = nn.ReLU
            conv_input_channel = obs_dim[0]
            mlp_hidden_layers = [256,256,256]

            # Construct CNN+MLP
            self.conv = CNN(
                conv_kernel_sizes,
                conv_channels,
                conv_strides,
                conv_activation,
                conv_input_channel,
            )
            conv_num_dims = (
                self.conv(torch.ones(obs_dim).unsqueeze(0)).reshape(1, -1).shape[-1]
            )
            # print(conv_num_dims)
            mlp_sizes = [conv_num_dims] + mlp_hidden_layers + [act_dim]

            self.mlp = MLP(mlp_sizes, self.hidden_activation, self.output_activation)
        else:
            raise NotImplementedError

    def forward(self, obs):
        # obs = obs.permute(0, 3, 1, 2)
        img = self.conv(obs)
        feature = img.view(img.size(0), -1)
        feature = self.mlp(feature)
        action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(feature) + (
            self.act_high_lim + self.act_low_lim
        ) / 2
        return action


class FiniteHorizonPolicy(nn.Module, Action_Distribution):
    def __init__(self, **kwargs):
        raise NotImplementedError


class StochaPolicy(nn.Module, Action_Distribution):
    """
    Approximated function of stochastic policy.
    Input: observation.
    Output: parameters of action distribution.
    """

    def __init__(self, **kwargs):
        super(StochaPolicy, self).__init__()
        act_dim = kwargs["act_dim"]
        obs_dim = kwargs["obs_dim"]
        self.conv_type = kwargs["conv_type"]
        act_high_lim = kwargs["act_high_lim"]
        act_low_lim = kwargs["act_low_lim"]
        self.register_buffer("act_high_lim", torch.from_numpy(act_high_lim))
        self.register_buffer("act_low_lim", torch.from_numpy(act_low_lim))
        self.hidden_activation = get_activation_func(kwargs["hidden_activation"])
        self.output_activation = get_activation_func(kwargs["output_activation"])
        self.min_log_std = kwargs["min_log_std"]
        self.max_log_std = kwargs["max_log_std"]
        self.action_distribution_cls = kwargs["action_distribution_cls"]
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((const.HEIGHT, const.WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        if self.conv_type == "type_1":
            # CNN+MLP Parameters
            conv_kernel_sizes = [8, 4, 3]
            conv_channels = [32, 64, 64]
            conv_strides = [4, 2, 1]
            conv_activation = nn.ReLU
            conv_input_channel = obs_dim[0]
            mlp_hidden_layers = [512, 256]

            # Construct CNN+MLP
            self.conv = CNN(
                conv_kernel_sizes,
                conv_channels,
                conv_strides,
                conv_activation,
                conv_input_channel,
            )
            conv_num_dims = (
                self.conv(torch.ones(obs_dim).unsqueeze(0)).reshape(1, -1).shape[-1]
            )
            policy_mlp_sizes = [conv_num_dims] + mlp_hidden_layers + [act_dim]
            self.mean = MLP(
                policy_mlp_sizes, self.hidden_activation, self.output_activation
            )
            self.log_std = MLP(
                policy_mlp_sizes, self.hidden_activation, self.output_activation
            )

        elif self.conv_type == "type_2":
            # CNN+MLP Parameters
            conv_kernel_sizes = [4, 3, 3, 3, 3, 3]
            conv_channels = [8, 16, 32, 64, 128, 256]
            conv_strides = [2, 2, 2, 2, 1, 1]
            conv_activation = nn.ReLU
            conv_input_channel = obs_dim[0]
            mlp_hidden_layers = [256, 256, 256]

            # Construct CNN+MLP
            self.conv = CNN(
                conv_kernel_sizes,
                conv_channels,
                conv_strides,
                conv_activation,
                conv_input_channel,
            )
            conv_num_dims = (
                self.conv(torch.ones(obs_dim).unsqueeze(0)).reshape(1, -1).shape[-1]
            )
            # print(conv_num_dims)
            policy_mlp_sizes = [conv_num_dims] + mlp_hidden_layers + [act_dim]
            self.mean = MLP(
                policy_mlp_sizes, self.hidden_activation, self.output_activation
            )
            self.log_std = MLP(
                policy_mlp_sizes, self.hidden_activation, self.output_activation
            )

        elif self.conv_type == "type_3":
            # MLP Parameters
            mlp_hidden_layers = [512, 256]

            # Construct CNN， output 512
            conv_kernel_sizes = [8, 4, 3]
            conv_channels = [32, 64, 64]
            conv_strides = [4, 2, 1]
            conv_activation = nn.ReLU
            m_camera_width = 96
            m_camera_height = 96
            conv_input_channel = 3
            n_traj_points = 10
            # Construct CNN+MLP
            self.conv = CNN(
                conv_kernel_sizes,
                conv_channels,
                conv_strides,
                conv_activation,
                conv_input_channel,
            )
            conv_num_dims = (
                self.conv(torch.ones((3, m_camera_width, m_camera_height)).unsqueeze(0)).reshape(1, -1).shape[-1]
            )
            # print(conv_num_dims)
            # Construct MLP， output 128
            policy_hidden_sizes = [10] + mlp_hidden_layers + [128]
            self.mlp = MLP(policy_hidden_sizes, self.hidden_activation, self.output_activation)

            policy_mlp_sizes = [conv_num_dims + 128] + mlp_hidden_layers + [n_traj_points*2]
            self.mean = MLP(
                policy_mlp_sizes, self.hidden_activation, self.output_activation
            )
            self.log_std = MLP(
                policy_mlp_sizes, self.hidden_activation, self.output_activation
            )

        elif self.conv_type == "type_4":
            n_traj_points = 10
            # MLP Parameters
            mlp_hidden_layers = [256,256]

            # Construct CNN， output 512
            self.conv = models.resnet18(pretrained=True)
            self.conv.fc = nn.Identity()  # Remove the last fully connected layer

            # print(conv_num_dims)
            # Construct MLP， output 128
            policy_hidden_sizes = [10]+mlp_hidden_layers+[128]
            self.mlp = MLP(policy_hidden_sizes, self.hidden_activation, self.output_activation)

            policy_mlp_sizes = [512+128] + mlp_hidden_layers + [n_traj_points*2]
            self.mean = MLP(
                policy_mlp_sizes, self.hidden_activation, self.output_activation
            )
            self.log_std = MLP(
                policy_mlp_sizes, self.hidden_activation, self.output_activation
            )
        elif self.conv_type == "type_5":

            n_traj_points = 10
            img_size = 13056
            # MLP Parameters
            mlp_hidden_layers = [512, 512]

            # Construct CNN， output 13056
            self.conv = nn.Sequential(
                LargeImpalaBlock(in_channels=1, out_channels=16),
                LargeImpalaBlock(in_channels=16, out_channels=32),
                LargeImpalaBlock(in_channels=32, out_channels=64),
                nn.Flatten())
            # print(conv_num_dims)
            # Construct MLP， output 128
            policy_hidden_sizes = [5] + mlp_hidden_layers + [128]
            self.mlp = MLP(policy_hidden_sizes, self.hidden_activation, self.output_activation)

            policy_mlp_sizes = [img_size + 128] + mlp_hidden_layers + [n_traj_points*2]
            self.mean = MLP(
                policy_mlp_sizes, self.hidden_activation, self.output_activation
            )
            self.log_std = MLP(
                policy_mlp_sizes, self.hidden_activation, self.output_activation
            )

        else:
            raise NotImplementedError

    def forward(self, obs):
        # -----zfw-20240704-----------
        if self.conv_type == "type_3":
            m_camera_width = 96
            m_camera_height = 96
            obs_img = obs[:, 0, :3 * m_camera_width * m_camera_height].reshape(
                (obs.size(0), 3, m_camera_width, m_camera_height))
            img = self.conv(obs_img)
            arr = self.mlp(obs[:, 0, 3 * m_camera_width * m_camera_height:])
            feature = torch.cat((img.view(img.size(0), -1), arr.view(arr.size(0), -1)), dim=1)
        elif self.conv_type == "type_4":
            m_camera_width = 224
            m_camera_height = 224
            obs_img = obs[0, :3*m_camera_width*m_camera_height].reshape((3, m_camera_width, m_camera_height))
            img = self.conv(obs_img)
            arr = self.mlp(obs[:, 0, 3*m_camera_width*m_camera_height:])
            feature = torch.cat((img, arr.reshape(arr.size(0), -1)), dim=1)
        # -----zfw-20240704-----------
        elif self.conv_type == "type_5":
            # # -----zfw-20240826-----------
            m_camera_width = 384
            m_camera_height = 800
            obs = obs.view(obs.size(0), 1, 3*m_camera_width*m_camera_height+5)
            obs_img = obs[:,0,  :3*m_camera_width*m_camera_height].reshape((obs.size(0), 3, m_camera_width, m_camera_height))  #Tensor(1, 3, 384, 800)
            # img_trans = obs_img[0].permute(1, 2, 0)  #Tensor(384, 800, 3)
            # img_trans = numpy_to_pil(img_trans.numpy())
            # img_trans = self.transform(img_trans).float()   #Tensor(3, 120, 160)
            #
            # img_trans = torch.unsqueeze(img_trans, 0)   #Tensor:(1, 3, 120, 160)
            img_trans = tensor_transform(obs_img)
            img = self.conv(img_trans)
            arr = self.mlp(obs[:, 0, 3 * m_camera_width * m_camera_height:])
            feature = torch.cat((img.view(img.size(0), -1), arr.view(arr.size(0), -1)), dim=1)
            # # -----zfw-20240826-----------
        else:
            img = self.conv(obs)
            feature = img.view(img.size(0), -1)
        action_mean = self.mean(feature)
        action_std = torch.clamp(
            self.log_std(feature), self.min_log_std, self.max_log_std
        ).exp()
        return torch.cat((action_mean, action_std), dim=-1)


class ActionValue(nn.Module, Action_Distribution):
    """
    Approximated function of action-value function.
    Input: observation, action.
    Output: action-value.
    """

    def __init__(self, **kwargs):
        super(ActionValue, self).__init__()
        act_dim = kwargs["act_dim"]
        obs_dim = kwargs["obs_dim"]
        conv_type = kwargs["conv_type"]
        self.hidden_activation = get_activation_func(kwargs["hidden_activation"])
        self.output_activation = get_activation_func(kwargs["output_activation"])
        self.action_distribution_cls = kwargs["action_distribution_cls"]
        if conv_type == "type_1":
            # CNN+MLP Parameters
            conv_kernel_sizes = [8, 4, 3]
            conv_channels = [32, 64, 64]
            conv_strides = [4, 2, 1]
            conv_activation = nn.ReLU
            conv_input_channel = obs_dim[0]
            mlp_hidden_layers = [512, 256]

            # Construct CNN+MLP
            self.conv = CNN(
                conv_kernel_sizes,
                conv_channels,
                conv_strides,
                conv_activation,
                conv_input_channel,
            )
            conv_num_dims = (
                self.conv(torch.ones(obs_dim).unsqueeze(0)).reshape(1, -1).shape[-1]
            )
            mlp_sizes = [conv_num_dims + act_dim] + mlp_hidden_layers + [1]

            self.mlp = MLP(mlp_sizes, self.hidden_activation, self.output_activation)

        elif conv_type == "type_2":
            # CNN+MLP Parameters
            conv_kernel_sizes = [4, 3, 3, 3, 3, 3]
            conv_channels = [8, 16, 32, 64, 128, 256]
            conv_strides = [2, 2, 2, 2, 1, 1]
            conv_activation = nn.ReLU
            conv_input_channel = obs_dim[0]
            mlp_hidden_layers = [256,256,256]

            # Construct CNN+MLP
            self.conv = CNN(
                conv_kernel_sizes,
                conv_channels,
                conv_strides,
                conv_activation,
                conv_input_channel,
            )
            conv_num_dims = (
                self.conv(torch.ones(obs_dim).unsqueeze(0)).reshape(1, -1).shape[-1]
            )
            mlp_sizes = [conv_num_dims + act_dim] + mlp_hidden_layers + [1]
            self.mlp = MLP(mlp_sizes, self.hidden_activation, self.output_activation)
        else:
            raise NotImplementedError

    def forward(self, obs, act):
        img = self.conv(obs)
        feature = torch.cat([img.view(img.size(0), -1), act], -1)
        return self.mlp(feature)


class ActionValueDis(nn.Module, Action_Distribution):
    """
    Approximated function of action-value function for discrete action space.
    Input: observation.
    Output: action-value for all action.
    """

    def __init__(self, **kwargs):
        super(ActionValueDis, self).__init__()
        act_num = kwargs["act_num"]
        obs_dim = kwargs["obs_dim"]
        conv_type = kwargs["conv_type"]
        self.hidden_activation = get_activation_func(kwargs["hidden_activation"])
        self.output_activation = get_activation_func(kwargs["output_activation"])
        self.action_distribution_cls = kwargs["action_distribution_cls"]
        if conv_type == "type_1":
            # CNN+MLP Parameters
            conv_kernel_sizes = [8, 4, 3]
            conv_channels = [32, 64, 64]
            conv_strides = [4, 2, 1]
            conv_activation = nn.ReLU
            conv_input_channel = obs_dim[0]
            mlp_hidden_layers = [512]

            # Construct CNN+MLP
            self.conv = CNN(
                conv_kernel_sizes,
                conv_channels,
                conv_strides,
                conv_activation,
                conv_input_channel,
            )
            conv_num_dims = (
                self.conv(torch.ones(obs_dim).unsqueeze(0)).reshape(1, -1).shape[-1]
            )
            mlp_sizes = [conv_num_dims] + mlp_hidden_layers + [act_num]
            self.mlp = MLP(mlp_sizes, self.hidden_activation, self.output_activation)

        elif conv_type == "type_2":
            # CNN+MLP Parameters
            conv_kernel_sizes = [4, 3, 3, 3, 3, 3]
            conv_channels = [8, 16, 32, 64, 128, 256]
            conv_strides = [2, 2, 2, 2, 1, 1]
            conv_activation = nn.ReLU
            conv_input_channel = obs_dim[0]
            mlp_hidden_layers = [256,256,256]

            # Construct CNN+MLP
            self.conv = CNN(
                conv_kernel_sizes,
                conv_channels,
                conv_strides,
                conv_activation,
                conv_input_channel,
            )
            conv_num_dims = (
                self.conv(torch.ones(obs_dim).unsqueeze(0)).reshape(1, -1).shape[-1]
            )
            mlp_sizes = [conv_num_dims] + mlp_hidden_layers + [act_num]
            self.mlp = MLP(mlp_sizes, self.hidden_activation, self.output_activation)

        else:
            raise NotImplementedError

    def forward(self, obs):
        img = self.conv(obs)
        feature = img.view(img.size(0), -1)
        act_value_dis = self.mlp(feature)
        return torch.squeeze(act_value_dis, -1)


class ActionValueDistri(nn.Module):
    """
    Approximated function of distributed action-value function.
    Input: observation.
    Output: parameters of action-value distribution.
    """

    def __init__(self, **kwargs):
        super(ActionValueDistri, self).__init__()
        act_dim = kwargs["act_dim"]
        obs_dim = kwargs["obs_dim"]
        self.conv_type = kwargs["conv_type"]
        self.hidden_activation = get_activation_func(kwargs["hidden_activation"])
        self.output_activation = get_activation_func(kwargs["output_activation"])
        self.action_distribution_cls = kwargs["action_distribution_cls"]
        if "min_log_std" in kwargs or "max_log_std" in kwargs:
            warnings.warn("min_log_std and max_log_std are deprecated in ActionValueDistri.")
        if self.conv_type == "type_1":
            # CNN+MLP Parameters
            conv_kernel_sizes = [8, 4, 3]
            conv_channels = [32, 64, 64]
            conv_strides = [4, 2, 1]
            conv_activation = nn.ReLU
            conv_input_channel = obs_dim[0]
            mlp_hidden_layers = [512, 256]

            # Construct CNN+MLP
            self.conv = CNN(
                conv_kernel_sizes,
                conv_channels,
                conv_strides,
                conv_activation,
                conv_input_channel,
            )
            conv_num_dims = (
                self.conv(torch.ones(obs_dim).unsqueeze(0)).reshape(1, -1).shape[-1]
            )
            mlp_sizes = [conv_num_dims + act_dim] + mlp_hidden_layers + [1]
            self.mean = MLP(mlp_sizes, self.hidden_activation, self.output_activation)
            self.log_std = MLP(
                mlp_sizes, self.hidden_activation, self.output_activation
            )

        elif self.conv_type == "type_2":
            # CNN+MLP Parameters
            conv_kernel_sizes = [4, 3, 3, 3, 3, 3]
            conv_channels = [8, 16, 32, 64, 128, 256]
            conv_strides = [2, 2, 2, 2, 1, 1]
            conv_activation = nn.ReLU
            conv_input_channel = obs_dim[0]
            mlp_hidden_layers = [256,256,256]

            # Construct CNN+MLP
            self.conv = CNN(
                conv_kernel_sizes,
                conv_channels,
                conv_strides,
                conv_activation,
                conv_input_channel,
            )
            conv_num_dims = (
                self.conv(torch.ones(obs_dim).unsqueeze(0)).reshape(1, -1).shape[-1]
            )
            mlp_sizes = [conv_num_dims + act_dim] + mlp_hidden_layers + [1]
            self.mean = MLP(mlp_sizes, self.hidden_activation, self.output_activation)
            self.log_std = MLP(
                mlp_sizes, self.hidden_activation, self.output_activation
            )

        elif self.conv_type == "type_3":
            # MLP Parameters
            mlp_hidden_layers = [512, 256]

            # Construct CNN， output 512
            conv_kernel_sizes = [8, 4, 3]
            conv_channels = [32, 64, 64]
            conv_strides = [4, 2, 1]
            conv_activation = nn.ReLU
            m_camera_width = 96
            m_camera_height = 96
            conv_input_channel = 3

            # Construct CNN+MLP
            self.conv = CNN(
                conv_kernel_sizes,
                conv_channels,
                conv_strides,
                conv_activation,
                conv_input_channel,
            )

            conv_num_dims = (
                self.conv(torch.ones((3, m_camera_width, m_camera_height)).unsqueeze(0)).reshape(1, -1).shape[-1]
            )
            # Construct MLP， output 128
            policy_hidden_sizes = [10] + mlp_hidden_layers + [128]
            self.mlp = MLP(policy_hidden_sizes, self.hidden_activation, self.output_activation)

            cat_hidden_sizes = [conv_num_dims + 128] + mlp_hidden_layers + [256]

            self.mlp_cat = MLP(cat_hidden_sizes, self.hidden_activation, self.output_activation)


            mlp_sizes = [256 + act_dim] + mlp_hidden_layers + [1]
            self.mean = MLP(mlp_sizes, self.hidden_activation, self.output_activation)
            self.log_std = MLP(
                mlp_sizes, self.hidden_activation, self.output_activation
            )

        elif self.conv_type == "type_4":
            # MLP Parameters
            mlp_hidden_layers = [256, 256]
            # Construct CNN， output 512
            self.conv = models.resnet18(pretrained=True)
            self.conv.fc = nn.Identity()  # Remove the last fully connected layer

            # Construct MLP， output 128
            policy_hidden_sizes = [10] + mlp_hidden_layers + [128]
            self.mlp = MLP(policy_hidden_sizes, self.hidden_activation, self.output_activation)

            cat_hidden_sizes = [512+128] + mlp_hidden_layers + [256]
            self.mlp_cat = MLP(cat_hidden_sizes, self.hidden_activation, self.output_activation)

            mlp_sizes = [256+act_dim] + mlp_hidden_layers + [1]
            self.mean = MLP(mlp_sizes, self.hidden_activation, self.output_activation)
            self.log_std = MLP(
                mlp_sizes, self.hidden_activation, self.output_activation
            )

        elif self.conv_type == "type_5":
            img_size = 13056
            # MLP Parameters
            mlp_hidden_layers = [512, 512]
            # Construct CNN， output 512
            self.conv = nn.Sequential(
                LargeImpalaBlock(in_channels=1, out_channels=16),
                LargeImpalaBlock(in_channels=16, out_channels=32),
                LargeImpalaBlock(in_channels=32, out_channels=64),
                nn.Flatten()
            )

            # Construct MLP， output 128
            policy_hidden_sizes = [5] + mlp_hidden_layers + [128]
            self.mlp = MLP(policy_hidden_sizes, self.hidden_activation, self.output_activation)
            mlp_sizes = [img_size + 128 + act_dim] + mlp_hidden_layers + [1]
            self.mean = MLP(mlp_sizes, self.hidden_activation, self.output_activation)
            self.log_std = MLP(
                mlp_sizes, self.hidden_activation, self.output_activation
            )

        else:
            raise NotImplementedError

    def forward(self, obs, act):
        # -----zfw-20240704-----------
        if self.conv_type == "type_3":
            m_camera_width = 96
            m_camera_height = 96
            obs_img = obs[:, 0, :3*m_camera_width*m_camera_height].reshape((obs.size(0), 3, m_camera_width, m_camera_height))
            img = self.conv(obs_img)
            arr = self.mlp(obs[:, 0, 3*m_camera_width*m_camera_height:].reshape((obs.size(0), -1)))
            feature_imgarr = self.mlp_cat(torch.cat((img.view(img.size(0), -1), arr.view(arr.size(0), -1)), dim=1))
            feature = torch.cat((feature_imgarr, act), dim=1)
        elif self.conv_type == "type_4":
            m_camera_width = 224
            m_camera_height = 224
            obs_img = obs[:, 0, :3*m_camera_width*m_camera_height].reshape((obs.size(0), 3, m_camera_width, m_camera_height))
            img = self.conv(obs_img)
            arr = self.mlp(obs[:, 0, 3*m_camera_width*m_camera_height:].reshape((obs.size(0), -1)))
            feature_imgarr = self.mlp_cat(torch.cat((img, arr), dim=1))
            feature = torch.cat((feature_imgarr, act), dim=1)
        # ----------------
        elif self.conv_type == "type_5":

            # -----zfw-20240826-----------
            m_camera_width = 384
            m_camera_height = 800
            obs = obs.view(obs.size(0), 1, 3 * m_camera_width * m_camera_height + 5)
            obs_img = obs[:, 0, :3 * m_camera_width * m_camera_height].reshape(
                (obs.size(0), 3, m_camera_width, m_camera_height))  # Tensor(1, 3, 384, 800)
            # img_trans = obs_img[0].permute(1, 2, 0)  #Tensor(384, 800, 3)
            # img_trans = numpy_to_pil(img_trans.numpy())
            # img_trans = self.transform(img_trans).float()   #Tensor(3, 120, 160)
            #
            # img_trans = torch.unsqueeze(img_trans, 0)   #Tensor:(1, 3, 120, 160)
            img_trans = tensor_transform(obs_img)
            img = self.conv(img_trans)
            arr = self.mlp(obs[:, 0, 3 * m_camera_width * m_camera_height:])
            feature = torch.cat((img.view(img.size(0), -1), arr.view(arr.size(0), -1), act), dim=1)

            # -----zfw-20240826-----------

        else:
            obs = obs.permute(0, 1, 3, 2)
            img = self.conv(obs)
            feature = torch.cat([img.view(img.size(0), -1), act], -1)
        value_mean = self.mean(feature)
        value_std = self.log_std(feature) # note: std, not log_std
        value_std = torch.nn.functional.softplus(value_std)  # avoid 0

        return torch.cat((value_mean, value_std), dim=-1)


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
        super(StateValue, self).__init__()
        obs_dim = kwargs["obs_dim"]
        conv_type = kwargs["conv_type"]
        self.hidden_activation = get_activation_func(kwargs["hidden_activation"])
        self.output_activation = get_activation_func(kwargs["output_activation"])
        self.action_distribution_cls = kwargs["action_distribution_cls"]
        if conv_type == "type_1":
            # CNN+MLP Parameters
            conv_kernel_sizes = [8, 4, 3]
            conv_channels = [32, 64, 64]
            conv_strides = [4, 2, 1]
            conv_activation = nn.ReLU
            conv_input_channel = obs_dim[0]
            mlp_hidden_layers = [512]

            # Construct CNN+MLP
            self.conv = CNN(
                conv_kernel_sizes,
                conv_channels,
                conv_strides,
                conv_activation,
                conv_input_channel,
            )
            conv_num_dims = (
                self.conv(torch.ones(obs_dim).unsqueeze(0)).reshape(1, -1).shape[-1]
            )
            mlp_sizes = [conv_num_dims] + mlp_hidden_layers + [1]
            self.mlp = MLP(mlp_sizes, self.hidden_activation, self.output_activation)

        elif conv_type == "type_2":
            # CNN+MLP Parameters
            conv_kernel_sizes = [4, 3, 3, 3, 3, 3]
            conv_channels = [8, 16, 32, 64, 128, 256]
            conv_strides = [2, 2, 2, 2, 1, 1]
            conv_activation = nn.ReLU
            conv_input_channel = obs_dim[0]
            mlp_hidden_layers = [256,256,256]

            # Construct CNN+MLP
            self.conv = CNN(
                conv_kernel_sizes,
                conv_channels,
                conv_strides,
                conv_activation,
                conv_input_channel,
            )
            conv_num_dims = (
                self.conv(torch.ones(obs_dim).unsqueeze(0)).reshape(1, -1).shape[-1]
            )
            mlp_sizes = [conv_num_dims] + mlp_hidden_layers + [1]
            self.mlp = MLP(mlp_sizes, self.hidden_activation, self.output_activation)
        else:
            raise NotImplementedError

    def forward(self, obs):
        img = self.conv(obs)
        feature = img.view(img.size(0), -1)
        v = self.mlp(feature)
        v = torch.squeeze(v, -1)
        return v
