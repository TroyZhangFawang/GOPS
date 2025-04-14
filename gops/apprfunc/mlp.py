#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Multilayer Perceptron (MLP)
#  Update: 2021-03-05, Wenjun Zou: create MLP function
#  Update: 2023-07-28, Jiaxin Gao: add FiniteHorizonFullPolicy function
#  Update: 2023-10-25, Wenxuan Wang: add DSAC-T algorithm


__all__ = [
    "DetermPolicy",
    "FiniteHorizonPolicy",
    "FiniteHorizonFullPolicy",
    "StochaPolicy",
    "ActionValue",
    "ActionValueDis",
    "ActionValueDistri",
    "StochaPolicyDis",
    "StateValue",
]

import numpy as np
import torch
import warnings
import torch.nn as nn
from gops.utils.common_utils import get_activation_func
from gops.utils.act_distribution_cls import Action_Distribution
from gops.utils.diffusion_helpers import (
    cosine_beta_schedule,
    linear_beta_schedule,
    vp_beta_schedule,
    extract,
    Losses,
    SinusoidalPosEmb,
)

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


# Deterministic policy
class DetermPolicy(nn.Module, Action_Distribution):
    """
    Approximated function of deterministic policy.
    Input: observation.
    Output: action.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]

        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(
            pi_sizes,
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]))
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]))
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(
            self.pi(obs)
        ) + (self.act_high_lim + self.act_low_lim) / 2
        return action


class FiniteHorizonPolicy(nn.Module, Action_Distribution):
    """
    Approximated function of deterministic policy for finite-horizon.
    Input: observation, time step.
    Output: action.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"] + 1
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]

        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(
            pi_sizes,
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]))
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]))
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs, virtual_t=1):
        virtual_t = virtual_t * torch.ones(
            size=[obs.shape[0], 1], dtype=torch.float32, device=obs.device
        )
        expand_obs = torch.cat((obs, virtual_t), 1)
        action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(
            self.pi(expand_obs)
        ) + (self.act_high_lim + self.act_low_lim) / 2
        return action


class FiniteHorizonFullPolicy(nn.Module, Action_Distribution):
    """
    Approximated function of deterministic policy for finite-horizon.
    Input: observation, time step.
    Output: action.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        self.act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.pre_horizon = kwargs["pre_horizon"]
        pi_sizes = [obs_dim] + list(hidden_sizes) + [self.act_dim * self.pre_horizon]

        self.pi = mlp(
            pi_sizes,
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]).float())
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]).float())
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        return self.forward_all_policy(obs)[:, 0, :]

    def forward_all_policy(self, obs):
        actions = self.pi(obs).reshape(obs.shape[0], self.pre_horizon, self.act_dim)
        action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(actions) \
                 + (self.act_high_lim + self.act_low_lim) / 2
        return action


# Stochastic Policy
class StochaPolicy(nn.Module, Action_Distribution):
    """
    Approximated function of stochastic policy.
    Input: observation.
    Output: parameters of action distribution.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.std_type = kwargs["std_type"]

        # mean and log_std are calculated by different MLP
        if self.std_type == "mlp_separated":
            pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
            self.mean = mlp(
                pi_sizes,
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
            )
            self.log_std = mlp(
                pi_sizes,
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
            )
        # mean and log_std are calculated by same MLP
        elif self.std_type == "mlp_shared":
            pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim * 2]
            self.policy = mlp(
                pi_sizes,
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
            )
        # mean is calculated by MLP, and log_std is learnable parameter
        elif self.std_type == "parameter":
            pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
            self.mean = mlp(
                pi_sizes,
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
            )
            self.log_std = nn.Parameter(-0.5*torch.ones(1, act_dim))

        self.min_log_std = kwargs["min_log_std"]
        self.max_log_std = kwargs["max_log_std"]
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]))
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]))
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        if self.std_type == "mlp_separated":
            action_mean = self.mean(obs)
            action_std = torch.clamp(
                self.log_std(obs), self.min_log_std, self.max_log_std
            ).exp()
        elif self.std_type == "mlp_shared":
            logits = self.policy(obs)
            action_mean, action_log_std = torch.chunk(
                logits, chunks=2, dim=-1
            )  # output the mean
            action_std = torch.clamp(
                action_log_std, self.min_log_std, self.max_log_std
            ).exp()
        elif self.std_type == "parameter":
            action_mean = self.mean(obs)
            action_log_std = self.log_std + torch.zeros_like(action_mean)
            action_std = torch.clamp(
                action_log_std, self.min_log_std, self.max_log_std
            ).exp()

        return torch.cat((action_mean, action_std), dim=-1)


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


# Diffusion Policy
class DiffusionMLP(nn.Module):
    """
    MLP Model
    """

    def __init__(self, state_dim, action_dim, hidden_dim, device, t_dim=16):
        super(DiffusionMLP, self).__init__()
        self.device = device
        self.t_dim = t_dim
        self.a_dim = action_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = state_dim + action_dim + t_dim
        self.mid_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )

        self.final_layer = nn.Linear(hidden_dim, action_dim)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, time, state, **kwargs):
        t = self.time_mlp(time)
        x = x.to(self.device)
        t = t.to(self.device)
        state = state.to(self.device)
        x = torch.cat([x, t, state], dim=1)
        x = self.mid_layer(x)

        return self.final_layer(x)

class DiffusionPolicy(nn.Module,Action_Distribution):
    """
    Approximated function of stochastic policy.
    Input: observation.
    Output: parameters of action distribution.
    """

    def __init__(
        self,
        beta_schedule="linear",
        loss_type="l2",
        clip_denoised=True,
        predict_epsilon=True,
        **kwargs
    ):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.std_type = kwargs["std_type"]
        self.device = torch.device(kwargs["device"])
        self.action_distribution_cls = kwargs["action_distribution_cls"]

        self.min_log_std = kwargs["min_log_std"]
        self.max_log_std = kwargs["max_log_std"]
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]))
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]))

        self.w = kwargs["policy_w"]
        self.T = kwargs["policy_T"]
        self.state_dim = kwargs["obs_dim"]
        self.action_dim = kwargs["act_dim"]
        self.max_action = kwargs["act_high_lim"][0]
        self.model = DiffusionMLP(obs_dim, act_dim, hidden_sizes, self.device).to(self.device)

        if beta_schedule == "linear":
            betas = linear_beta_schedule(self.T)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(self.T)
        elif beta_schedule == "vp":
            betas = vp_beta_schedule(self.T)
        betas = betas.to(self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, device=self.device), alphas_cumprod[:-1]])

        self.n_timesteps = self.T
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        self.loss_fn = Losses[loss_type]()

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        """
        if self.predict_epsilon, model output is (scaled) noise;
        otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, s, **kwargs):
        eps = self.model(x, t, s)
        x_recon = self.predict_start_from_noise(x, t=t, noise=eps)

        if self.clip_denoised:
            x_recon.clamp_(-self.max_action, self.max_action)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, x, t, s, **kwargs):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, s=s, **kwargs
        )
        noise = 0.5 * torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_sample_loop(
        self, state, shape, verbose=False, return_diffusion=False, **kwargs
    ):
        batch_size = shape[0]
        # x = torch.full(shape, 1.0, device=self.device, requires_grad=True)
        x = 0.5 * torch.randn(shape, device=self.device, requires_grad=True)

        if return_diffusion:
            diffusion = [x]

        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full(
                (batch_size,), i, device=self.device, dtype=torch.long
            )
            x = self.p_sample(x, timesteps, state, **kwargs)

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    def p_sample_approximate(self, state, action, **kwargs):
        # EDP sampling, one step to approximate the action
        batch_size = len(action)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=self.device).long()
        x_noisy = self.q_sample(x_start=action, t=t)
        if torch.allclose(
            kwargs["cemb"],
            torch.zeros_like(kwargs["cemb"]),
            rtol=1e-05,
            atol=1e-08,
            equal_nan=False,
        ):
            x_approx = self.predict_start_from_noise(
                x_t=x_noisy, t=t, noise=self.model(x_noisy, t, state)
            )
        else:
            cemb_shape = kwargs["cemb"].shape
            pred_eps_cond = self.model(x_noisy, t, state, **kwargs)
            kwargs["cemb"] = torch.zeros(cemb_shape, device=self.device)
            pred_eps_uncond = self.model(x_noisy, t, state, **kwargs)
            eps = pred_eps_uncond + self.w * (pred_eps_cond - pred_eps_uncond)
            x_approx = self.predict_start_from_noise(x_t=x_noisy, t=t, noise=eps)

        return x_approx

    def sample(self, state, *args, **kwargs):
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        if "edp" in kwargs and kwargs["edp"] == True:
            assert "action" in kwargs
            action = self.p_sample_approximate(
                state=state, shape=shape, *args, **kwargs
            )
        else:
            action = self.p_sample_loop(state, shape, *args, **kwargs)
        return action.clamp_(-1, 1)

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = 0.5 * torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, state, t, weights=1.0):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_recon = self.model(x_noisy, t, state)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss = self.loss_fn(x_recon, noise, weights)
        else:
            loss = self.loss_fn(x_recon, x_start, weights)

        return loss

    def loss(self, x, state, weights=1.0):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=self.device).long()
        return self.p_losses(x, state, t, weights)

    def forward(self, obs, **kwargs):
        obs = obs.to(self.device)
        return self.sample(obs, **kwargs)

class DiffusionPolicyeasy(nn.Module,Action_Distribution):
    """
    Approximated function of stochastic policy.
    Input: observation.
    Output: parameters of action distribution.
    """

    def __init__(
        self,
        beta_schedule="vp",
        loss_type="l2",
        clip_denoised=True,
        predict_epsilon=True,
        **kwargs
    ):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.std_type = kwargs["std_type"]
        self.device = torch.device(kwargs["device"])
        self.action_distribution_cls = kwargs["action_distribution_cls"]

        self.min_log_std = kwargs["min_log_std"]
        self.max_log_std = kwargs["max_log_std"]
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]))
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]))

        self.w = kwargs["policy_w"]
        self.T = kwargs["policy_T"]
        self.state_dim = kwargs["obs_dim"]
        self.action_dim = kwargs["act_dim"]
        self.max_action = kwargs["act_high_lim"][0]
        self.model = DiffusionMLP(obs_dim, act_dim, hidden_sizes, self.device).to(self.device)

        if beta_schedule == "linear":
            betas = linear_beta_schedule(self.T)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(self.T)
        elif beta_schedule == "vp":
            betas = vp_beta_schedule(self.T)
        betas = betas.to(self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, device=self.device), alphas_cumprod[:-1]])

        self.n_timesteps = self.T
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer("betas", betas)
        self.register_buffer("alphas" , alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        self.loss_fn = Losses[loss_type]()
    # ------------------------------------------ sampling ------------------------------------------#
    def sample(self, state, *args, **kwargs):
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        action = self.p_sample_loop(state, shape, *args, **kwargs)
        return action.clamp_(-1, 1)

    def p_sample_loop(
        self, state, shape, verbose=False, return_diffusion=False, **kwargs
    ):
        batch_size = shape[0]
        # x = torch.full(shape, -1.0, device=self.device, requires_grad=True)
        x = torch.zeros(shape, device=self.device, requires_grad=True)
        # x = 0.5 * torch.randn(shape, device=self.device, requires_grad=True)
        if return_diffusion:
            diffusion = [x]
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full(
                (batch_size,), i, device=self.device, dtype=torch.long
            )
            device = x.device
            eps = self.model(x, timesteps, state)
            x_0_pred = extract(self.sqrt_recip_alphas_cumprod, timesteps, x.shape) * x- extract(self.sqrt_recipm1_alphas_cumprod, timesteps, x.shape) * eps
            x_0_pred = x_0_pred.clamp_(-self.max_action, self.max_action)
            x = (
                extract(self.posterior_mean_coef1, timesteps, x.shape) * x_0_pred
                + extract(self.posterior_mean_coef2, timesteps, x.shape) * x
            )
            x = x.clamp(-1, 1)
            if return_diffusion:
                diffusion.append(x)
        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    def forward(self, obs, **kwargs):
        obs = obs.to(self.device)
        return self.sample(obs, **kwargs)

# class doublemlpDiffusionPolicy(nn.Module,Action_Distribution):
#     """
#     Approximated function of stochastic policy.
#     Input: observation.
#     Output: parameters of action distribution.
#     """
#
#     def __init__(
#         self,
#         beta_schedule="linear",
#         loss_type="l2",
#         clip_denoised=True,
#         predict_epsilon=True,
#         **kwargs
#     ):
#         super().__init__()
#         obs_dim = kwargs["obs_dim"]
#         act_dim = kwargs["act_dim"]
#         hidden_sizes = kwargs["hidden_sizes"]
#         self.std_type = kwargs["std_type"]
#         self.device = torch.device(kwargs["device"])
#         self.action_distribution_cls = kwargs["action_distribution_cls"]
#
#         self.min_log_std = kwargs["min_log_std"]
#         self.max_log_std = kwargs["max_log_std"]
#         self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]))
#         self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]))
#
#         self.w = kwargs["policy_w"]
#         self.T = kwargs["policy_T"]
#         self.state_dim = kwargs["obs_dim"]
#         self.action_dim = kwargs["act_dim"]
#         self.max_action = kwargs["act_high_lim"][0]
#         self.model = DiffusionMLP(obs_dim, act_dim, hidden_sizes, self.device).to(self.device)
#         #########################################################################################
#         pi_sizes = [obs_dim] + list([256, 256]) + [act_dim * 2]
#         self.policy = mlp(
#             pi_sizes,
#             get_activation_func("gelu"),
#             get_activation_func(kwargs["output_activation"]),
#         ).to(self.device)
#         #########################################################################################
#
#         if beta_schedule == "linear":
#             betas = linear_beta_schedule(self.T)
#         elif beta_schedule == "cosine":
#             betas = cosine_beta_schedule(self.T)
#         elif beta_schedule == "vp":
#             betas = vp_beta_schedule(self.T)
#         betas = betas.to(self.device)
#         alphas = 1.0 - betas
#         alphas_cumprod = torch.cumprod(alphas, axis=0)
#         alphas_cumprod_prev = torch.cat([torch.ones(1, device=self.device), alphas_cumprod[:-1]])
#
#         self.n_timesteps = self.T
#         self.clip_denoised = clip_denoised
#         self.predict_epsilon = predict_epsilon
#
#         self.register_buffer("betas", betas)
#         self.register_buffer("alphas" , alphas)
#         self.register_buffer("alphas_cumprod", alphas_cumprod)
#         self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
#
#         # calculations for diffusion q(x_t | x_{t-1}) and others
#         self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
#         self.register_buffer(
#             "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
#         )
#         self.register_buffer(
#             "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
#         )
#         self.register_buffer(
#             "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
#         )
#         self.register_buffer(
#             "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
#         )
#
#         # calculations for posterior q(x_{t-1} | x_t, x_0)
#         posterior_variance = (
#             betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
#         )
#         self.register_buffer("posterior_variance", posterior_variance)
#
#         ## log calculation clipped because the posterior variance
#         ## is 0 at the beginning of the diffusion chain
#         self.register_buffer(
#             "posterior_log_variance_clipped",
#             torch.log(torch.clamp(posterior_variance, min=1e-20)),
#         )
#         self.register_buffer(
#             "posterior_mean_coef1",
#             betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
#         )
#         self.register_buffer(
#             "posterior_mean_coef2",
#             (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
#         )
#
#         self.loss_fn = Losses[loss_type]()
#     # ------------------------------------------ sampling ------------------------------------------#
#     def sample(self, state, *args, **kwargs):
#         batch_size = state.shape[0]
#         shape = (batch_size, self.action_dim)
#         action = self.p_sample_loop(state, shape, *args, **kwargs)
#         return action.clamp_(-1, 1)
#
#     def p_sample_loop(
#         self, state, shape, verbose=False, return_diffusion=False, **kwargs
#     ):
#         batch_size = shape[0]
#         logits = self.policy(state)
#         action_mean, action_log_std = torch.chunk(logits, chunks=2, dim=-1)
#         # output the mean
#         action_std = torch.clamp(action_log_std, self.min_log_std, self.max_log_std).exp()
#         epsilon = torch.randn_like(action_mean)
#         x = action_mean + action_std * epsilon
#         # x = 0.5 * torch.randn(shape, device=self.device, requires_grad=True)
#         if return_diffusion:
#             diffusion = [x]
#         for i in reversed(range(0, self.n_timesteps)):
#             timesteps = torch.full(
#                 (batch_size,), i, device=self.device, dtype=torch.long
#             )
#             device = x.device
#             eps = self.model(x, timesteps, state)
#             x_0_pred = extract(self.sqrt_recip_alphas_cumprod, timesteps, x.shape) * x- extract(self.sqrt_recipm1_alphas_cumprod, timesteps, x.shape) * eps
#             x_0_pred = x_0_pred.clamp_(-self.max_action, self.max_action)
#             x = (
#                 extract(self.posterior_mean_coef1, timesteps, x.shape) * x_0_pred
#                 + extract(self.posterior_mean_coef2, timesteps, x.shape) * x
#             )
#             x = x.clamp(-1, 1)
#             if return_diffusion:
#                 diffusion.append(x)
#         if return_diffusion:
#             return x, torch.stack(diffusion, dim=1)
#         else:
#             return x
#
#     def forward(self, obs, **kwargs):
#         obs = obs.to(self.device)
#         return self.sample(obs, **kwargs)