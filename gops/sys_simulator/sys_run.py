#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: plot module for trained policy
#  Update: 2022-12-05, Congsheng Zhang: create plot module

import argparse
import datetime
import glob
import os
import gym
from typing import Any, Optional, Tuple
import matplotlib.pyplot as plt
# import gops.utils.planner_benchmark.visualize as vis
import numpy as np
import seaborn as sns
import torch
import pandas as pd
from gym import wrappers
from copy import copy, deepcopy
import time
from abc import abstractmethod
from gops.create_pkg.create_alg import create_approx_contrainer
from gops.create_pkg.create_env_model import create_env_model
from gops.create_pkg.create_env import create_env
from gops.env.env_gen_ocp.pyth_base import Env, State
from gops.utils.plot_evaluation import cm2inch
from gops.utils.common_utils import get_args_from_json, mp4togif
from gops.utils.gops_path import gops_path

default_cfg = dict()
default_cfg["fig_size"] = (12, 9)
default_cfg["dpi"] = 300
default_cfg["pad"] = 0.5

default_cfg["tick_size"] = 8
default_cfg["tick_label_font"] = "Times New Roman"
default_cfg["legend_font"] = {
    "family": "Times New Roman",
    "size": "8",
    "weight": "normal",
}
default_cfg["label_font"] = {
    "family": "Times New Roman",
    "size": "9",
    "weight": "normal",
}

default_cfg["img_fmt"] = "png"

class PolicyRunner:
    """Plot module for trained policy

    :param list log_policy_dir_list: directory of trained policy.
    :param list trained_policy_iteration_list: iteration of trained policy.
    :param bool save_render: save environment animation or not.
    :param list plot_range: customize plot range.
    :param bool is_init_info: customize initial information or not.
    :param dict init_info: initial information.
    :param list legend_list: legends of figures.
    :param bool use_opt: use optimal solution for comparison or not.
    :param Optional[str] load_opt_path: path to load optimal controller result.
    :param dict opt_args: arguments of optimal solution solver.
    :param bool save_opt: save optimal controller result or not.
    :param bool constrained_env: constrained environment or not.
    :param bool is_tracking: tracking problem or not.
    :param bool use_dist: use adversarial action or not.
    :param float dt: time interval between steps.
    :param str obs_noise_type: type of observation noise, "normal" or "uniform".
    :param list obs_noise_data: Mean and
        Standard deviation of Normal distribution or Upper
        and Lower bounds of Uniform distribution.
    :param str action_noise_type: type of action noise, "normal" or "uniform".
    :param list action_noise_data: Mean and
        Standard deviation of Normal distribution or Upper
        and Lower bounds of Uniform distribution.
    """

    def __init__(
        self,
        log_policy_dir_list: list,
        trained_policy_iteration_list: list,
        save_render: bool = False,
        plot_range: list = None,
        is_init_info: bool = False,
        init_info: dict = None,
        legend_list: list = None,
        use_opt: bool = False,
        load_opt_path: Optional[str] = None,
        opt_args: Optional[dict] = None,
        save_opt: bool = True,
        constrained_env: bool = False,
        is_tracking: bool = False,
        use_dist: bool = False,
        dt: float = None,
        obs_noise_type: str = None,
        obs_noise_data: list = None,
        action_noise_type: str = None,
        action_noise_data: list = None,
    ):
        self.log_policy_dir_list = [
            os.path.join(gops_path, d) for d in log_policy_dir_list
        ]
        self.trained_policy_iteration_list = trained_policy_iteration_list
        self.save_render = save_render
        self.args = None
        self.plot_range = plot_range
        if is_init_info:
            self.init_info = init_info
        else:
            self.init_info = {}
        self.legend_list = legend_list
        self.use_opt = use_opt
        if use_opt:
            assert load_opt_path is not None or opt_args is not None
            self.load_opt_path = load_opt_path
            self.opt_args = opt_args
            if isinstance(self.opt_args, dict) and \
                "use_MPC_for_general_env" not in self.opt_args.keys():
                self.opt_args["use_MPC_for_general_env"] = False
            self.save_opt = save_opt
        self.constrained_env = constrained_env
        self.use_dist = use_dist
        self.is_tracking = is_tracking
        self.dt = dt
        self.policy_num = len(self.log_policy_dir_list)
        if self.policy_num != len(self.trained_policy_iteration_list):
            raise RuntimeError(
                "The length of policy number is not equal to that of policy iteration"
            )
        self.obs_noise_type = obs_noise_type
        self.obs_noise_data = obs_noise_data
        self.action_noise_type = action_noise_type
        self.action_noise_data = action_noise_data
        self.ref_state_num = 0

        # data for plot
        self.args_list = []
        self.eval_list = []
        self.env_id_list = []
        self.algorithm_list = []
        self.tracking_list = []

        self.__load_all_args()
        self.env_id = self.get_n_verify_env_id()

        # save path
        path = os.path.join(os.path.dirname(__file__), "..", "..", "figures")
        path = os.path.abspath(path)

        algs_name = ""
        for item in self.algorithm_list:
            algs_name = algs_name + item + "-"
        self.save_path = os.path.join(
            path,
            algs_name + self.env_id,
            datetime.datetime.now().strftime("%y%m%d-%H%M%S"),
        )
        os.makedirs(self.save_path, exist_ok=True)

    def run_an_episode(
        self,
        env: Any,
        controller: Any,
        init_info: dict,
        is_opt: bool,
        render: bool = True,
    ) -> Tuple[dict, dict]:
        state_list = []
        action_list = []
        reward_list = []
        constrain_list = []
        obs_list = []
        step = 0
        step_list = []
        calctime_list = []
        info_list = [init_info]
        obs, info = env.reset(**init_info)
        state = env.state
        print("Initial robot state: ")
        print(self.__convert_format(np.asarray(state.robot_state)))
        # plot tracking
        state_with_ref_error = {}
        done = False
        info.update({"TimeLimit.truncated": False})
        while not (done or info["TimeLimit.truncated"]):
            print("step:", step + 1)
            state_list.append(state.robot_state)
            obs_list.append(obs)
            if is_opt:
                if isinstance(env.unwrapped, Env):
                    time_start = time.time()
                    action = controller(state)
                    calc_time = time.time()-time_start
                else:
                    time_start = time.time()
                    action = controller(obs, info)
                    calc_time = time.time()-time_start
            else:
                time_start = time.time()
                action = self.compute_action(obs, controller)
                action = self.__action_noise(action)
                calc_time = time.time() - time_start
            if self.use_dist:
                action = np.hstack((action, env.dist_func(step * env.tau)))
            if self.constrained_env:
                constrain_list.append(info["constraint"])
            if self.is_tracking:
                reference = get_reference_from_info(info)
                state_num = len(reference)
                self.ref_state_num = sum(x is not None for x in reference)
                if step == 0:
                    for i in range(state_num):
                        if reference[i] is not None:
                            state_with_ref_error["state-{}".format(i)] = []
                            state_with_ref_error["ref-{}".format(i)] = []
                            state_with_ref_error["state-{}-error".format(i)] = []

                robot_state = get_robot_state_from_info(info)
                for i in range(state_num):
                    if reference[i] is not None:
                        state_with_ref_error["state-{}".format(i)].append(robot_state[i])
                        state_with_ref_error["ref-{}".format(i)].append(reference[i])
                        state_with_ref_error["state-{}-error".format(i)].append(
                            reference[i] - robot_state[i]
                        )
            next_obs, reward, done, info = env.step(action)

            # save the real action (without scaling)
            action_list.append(info.get("raw_action", action))
            step_list.append(step)
            reward_list.append(reward)
            info_list.append(info)
            calctime_list.append(calc_time*1000)

            obs = next_obs
            state = env.state
            step = step + 1

            if "TimeLimit.truncated" not in info.keys():
                info["TimeLimit.truncated"] = False
            # Draw environment animation
            if render:
                env.render()

        eval_dict = {
            "reward_list": reward_list,
            "action_list": action_list,
            "state_list": state_list,
            "step_list": step_list,
            "obs_list": obs_list,
            "info_list": info_list,
            "calctime_list": calctime_list
        }
        if self.constrained_env:
            eval_dict.update(
                {"constrain_list": constrain_list,}
            )

        if self.is_tracking:
            tracking_dict = state_with_ref_error
        else:
            tracking_dict = {}

        return eval_dict, tracking_dict

    def compute_action(self, obs: np.ndarray, networks: Any) -> np.ndarray:
        batch_obs = torch.from_numpy(np.expand_dims(obs, axis=0).astype("float32"))
        logits = networks.policy(batch_obs)
        action_distribution = networks.create_action_distributions(logits)
        action = action_distribution.mode()
        action = action.detach().numpy()[0]
        return action

    def draw(self):
        fig_size = (
            default_cfg["fig_size"],
            default_cfg["fig_size"],
        )
        action_dim = self.eval_list[0]["action_list"][0].shape[0]
        state_dim = self.eval_list[0]["state_list"][0].shape[0]
        if self.constrained_env:
            constrain_dim = self.eval_list[0]["constrain_list"][0].shape[0]
        policy_num = len(self.algorithm_list)
        if self.use_opt:
            legend = ""
            policy_num += 1
            if self.opt_args["opt_controller_type"] == "OPT":
                legend = "OPT"
            elif self.opt_args["opt_controller_type"] == "MPC":
                legend = "MPC-" + str(self.opt_args["num_pred_step"])
                if (
                    "use_terminal_cost" not in self.opt_args.keys()
                    or self.opt_args["use_terminal_cost"] is False
                ):
                    legend += " (w/o TC)"
                else:
                    legend += " (w/ TC)"
            self.algorithm_list.append(legend)

        # Create initial list
        reward_list = []
        action_list = []
        state_list = []
        step_list = []
        state_ref_error_list = []
        constrain_list = []
        calctime_list = []
        # Put data into list
        for i in range(policy_num):
            reward_list.append(np.array(self.eval_list[i]["reward_list"]))
            action_list.append(np.array(self.eval_list[i]["action_list"]))
            state_list.append(np.array(self.eval_list[i]["state_list"]))
            step_list.append(np.array(self.eval_list[i]["step_list"]))
            calctime_list.append(np.array(self.eval_list[i]["calctime_list"]))
            if self.constrained_env:
                constrain_list.append(np.stack(self.eval_list[i]["constrain_list"]))
            if self.is_tracking:
                state_ref_error_list.append(self.tracking_list[i])

        if self.plot_range is None:
            pass
        elif len(self.plot_range) == 2:

            for i in range(policy_num):
                start_range = self.plot_range[0]
                end_range = min(self.plot_range[1], reward_list[i].shape[0])

                reward_list[i] = reward_list[i][start_range:end_range]
                action_list[i] = action_list[i][start_range:end_range]
                state_list[i] = state_list[i][start_range:end_range]
                step_list[i] = step_list[i][start_range:end_range]
                if self.constrained_env:
                    constrain_list[i] = constrain_list[i][start_range:end_range]
                if self.is_tracking:
                    for key, value in self.tracking_list[i].items():
                        self.tracking_list[i][key] = value[start_range:end_range]
        else:
            raise NotImplementedError("Figure range is wrong")

        if self.dt is None:
            x_label = "Time step"
        else:
            step_list = [s * self.dt for s in step_list]
            x_label = "Time (s)"

        # Plot reward
        path_reward_fmt = os.path.join(
            self.save_path, "Reward.{}".format(default_cfg["img_fmt"])
        )
        fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

        # save reward data to csv
        reward_data = pd.DataFrame(data=reward_list)
        reward_data.to_csv(os.path.join(self.save_path, "Reward.csv"), encoding="gbk")

        for i in range(policy_num):
            legend = (
                self.legend_list[i]
                if len(self.legend_list) == policy_num
                else self.algorithm_list[i]
            )
            sns.lineplot(x=step_list[i], y=reward_list[i], label="{}".format(legend))
        plt.tick_params(labelsize=default_cfg["tick_size"])
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
        plt.xlabel(x_label, default_cfg["label_font"])
        plt.ylabel("Reward", default_cfg["label_font"])
        plt.legend(loc="best", prop=default_cfg["legend_font"])
        fig.tight_layout(pad=default_cfg["pad"])
        plt.savefig(path_reward_fmt, format=default_cfg["img_fmt"], bbox_inches="tight")
        plt.close()

        # plot action
        for j in range(action_dim):
            path_action_fmt = os.path.join(
                self.save_path, "Action-{}.{}".format(j + 1, default_cfg["img_fmt"])
            )
            fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

            # save action data to csv
            action_data = pd.DataFrame(data=[a[:, j] for a in action_list])
            action_data.to_csv(
                os.path.join(self.save_path, "Action-{}.csv".format(j + 1)),
                encoding="gbk",
            )

            for i in range(policy_num):
                legend = (
                    self.legend_list[i]
                    if len(self.legend_list) == policy_num
                    else self.algorithm_list[i]
                )
                sns.lineplot(
                    x=step_list[i], y=action_list[i][:, j], label="{}".format(legend)
                )
            plt.tick_params(labelsize=default_cfg["tick_size"])
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
            plt.xlabel(x_label, default_cfg["label_font"])
            plt.ylabel("Action-{}".format(j + 1), default_cfg["label_font"])
            plt.legend(loc="best", prop=default_cfg["legend_font"])
            fig.tight_layout(pad=default_cfg["pad"])
            plt.savefig(
                path_action_fmt, format=default_cfg["img_fmt"], bbox_inches="tight"
            )
            plt.close()

        # plot state
        for j in range(state_dim):
            path_state_fmt = os.path.join(
                self.save_path, "State-{}.{}".format(j + 1, default_cfg["img_fmt"])
            )
            fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

            # save state data to csv
            state_data = pd.DataFrame(data=[s[:, j] for s in state_list])
            state_data.to_csv(
                os.path.join(self.save_path, "State-{}.csv".format(j + 1)),
                encoding="gbk",
            )

            for i in range(policy_num):
                legend = (
                    self.legend_list[i]
                    if len(self.legend_list) == policy_num
                    else self.algorithm_list[i]
                )
                sns.lineplot(
                    x=step_list[i], y=state_list[i][:, j], label="{}".format(legend)
                )
            plt.tick_params(labelsize=default_cfg["tick_size"])
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
            plt.xlabel(x_label, default_cfg["label_font"])
            plt.ylabel("State-{}".format(j + 1), default_cfg["label_font"])
            plt.legend(loc="best", prop=default_cfg["legend_font"])
            fig.tight_layout(pad=default_cfg["pad"])
            plt.savefig(
                path_state_fmt, format=default_cfg["img_fmt"], bbox_inches="tight"
            )
            plt.close()

        # plot tracking
        if self.is_tracking:
            # find index of the longest trajectory
            traj_lens = [len(r) for r in reward_list]
            longest_traj_index = np.argmax(traj_lens)

            for j in range(self.ref_state_num):

                # plot state and ref
                path_tracking_state_fmt = os.path.join(
                    self.save_path, "State-{}.{}".format(j + 1, default_cfg["img_fmt"])
                )
                fig, ax = plt.subplots(
                    figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"]
                )
                # save tracking state data to csv
                tracking_state_data = []
                for i in range(policy_num):
                    legend = (
                        self.legend_list[i]
                        if len(self.legend_list) == policy_num
                        else self.algorithm_list[i]
                    )
                    sns.lineplot(
                        x=step_list[i],
                        y=state_ref_error_list[i]["state-{}".format(j)],
                        label="{}".format(legend),
                    )
                    tracking_state_data.append(
                        state_ref_error_list[i]["state-{}".format(j)]
                    )
                sns.lineplot(
                    x=step_list[longest_traj_index],
                    y=state_ref_error_list[longest_traj_index]["ref-{}".format(j)],
                    label="ref",
                )
                tracking_state_data.append(state_ref_error_list[longest_traj_index]["ref-{}".format(j)])
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel(x_label, default_cfg["label_font"])
                plt.ylabel("State-{}".format(j + 1), default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(
                    path_tracking_state_fmt,
                    format=default_cfg["img_fmt"],
                    bbox_inches="tight",
                )
                plt.close()

                tracking_state_data = pd.DataFrame(data=tracking_state_data)
                tracking_state_data.to_csv(
                    os.path.join(self.save_path, "State-{}.csv".format(j + 1)),
                    encoding="gbk",
                )

                # plot state-ref error
                path_tracking_error_fmt = os.path.join(
                    self.save_path,
                    "Ref - State-{}.{}".format(j + 1, default_cfg["img_fmt"]),
                )
                fig, ax = plt.subplots(
                    figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"]
                )
                # save tracking error data to csv
                tracking_error_data = []
                for i in range(policy_num):
                    legend = (
                        self.legend_list[i]
                        if len(self.legend_list) == policy_num
                        else self.algorithm_list[i]
                    )
                    sns.lineplot(
                        x=step_list[i],
                        y=state_ref_error_list[i]["state-{}-error".format(j)],
                        label="{}".format(legend),
                    )
                    tracking_error_data.append(
                        state_ref_error_list[i]["state-{}-error".format(j)]
                    )
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel(x_label, default_cfg["label_font"])
                plt.ylabel("Ref $-$ State-{}".format(j + 1), default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(
                    path_tracking_error_fmt,
                    format=default_cfg["img_fmt"],
                    bbox_inches="tight",
                )
                plt.close()

                tracking_error_data = pd.DataFrame(data=tracking_error_data)
                tracking_error_data.to_csv(
                    os.path.join(self.save_path, "Ref - State-{}.csv".format(j + 1)),
                    encoding="gbk",
                )
        # plot calculation time
        path_state_fmt = os.path.join(
            self.save_path, "Calc time.{}".format(default_cfg["img_fmt"])
        )
        fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

        # save state data to csv
        state_data = pd.DataFrame(data=[s[:] for s in calctime_list])
        state_data.to_csv(
            os.path.join(self.save_path, "Calc time.csv"),
            encoding="gbk",
        )

        for i in range(policy_num):
            legend = (
                self.legend_list[i]
                if len(self.legend_list) == policy_num
                else self.algorithm_list[i]
            )
            sns.lineplot(
                x=step_list[i], y=calctime_list[i][:], label="{}".format(legend)
            )
        plt.tick_params(labelsize=default_cfg["tick_size"])
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
        plt.xlabel(x_label, default_cfg["label_font"])
        plt.ylabel("Calc Time [ms]", default_cfg["label_font"])
        plt.legend(loc="best", prop=default_cfg["legend_font"])
        fig.tight_layout(pad=default_cfg["pad"])
        plt.savefig(
            path_state_fmt, format=default_cfg["img_fmt"], bbox_inches="tight"
        )
        plt.close()
        # plot constraint value
        if self.constrained_env:
            for j in range(constrain_dim):
                path_constraint_fmt = os.path.join(
                    self.save_path,
                    "Constrain-{}.{}".format(j + 1, default_cfg["img_fmt"]),
                )
                fig, ax = plt.subplots(
                    figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"]
                )

                # save reward data to csv
                constrain_data = pd.DataFrame(data=[c[:, j] for c in constrain_list])
                constrain_data.to_csv(
                    os.path.join(self.save_path, "Constrain-{}.csv".format(j + 1)),
                    encoding="gbk",
                )

                for i in range(policy_num):
                    legend = (
                        self.legend_list[i]
                        if len(self.legend_list) == policy_num
                        else self.algorithm_list[i]
                    )
                    sns.lineplot(
                        x=step_list[i],
                        y=constrain_list[i][:, j],
                        label="{}".format(legend),
                    )
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel(x_label, default_cfg["label_font"])
                plt.ylabel("Constrain-{}".format(j + 1), default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(
                    path_constraint_fmt,
                    format=default_cfg["img_fmt"],
                    bbox_inches="tight",
                )
                plt.close()

        # plot error with opt
        if self.use_opt:
            # reward error
            path_reward_error_fmt = os.path.join(
                self.save_path, "Reward error.{}".format(default_cfg["img_fmt"])
            )
            fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

            # save reward error data to csv
            reward_error_list = []
            for r in reward_list:
                end = min(len(r), len(reward_list[-1]))
                reward_error_list.append(r[:end] - reward_list[-1][:end])
            reward_error_data = pd.DataFrame(data=reward_error_list)
            reward_error_data.to_csv(
                os.path.join(self.save_path, "Reward error.csv"), encoding="gbk"
            )

            for i in range(policy_num - 1):
                legend = (
                    self.legend_list[i]
                    if len(self.legend_list) == policy_num
                    else self.algorithm_list[i]
                )
                sns.lineplot(
                    x=step_list[i][:len(reward_error_list[i])],
                    y=reward_error_list[i], label="{}".format(legend)
                )
            plt.tick_params(labelsize=default_cfg["tick_size"])
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
            plt.xlabel(x_label, default_cfg["label_font"])
            plt.ylabel("Reward error", default_cfg["label_font"])
            plt.legend(loc="best", prop=default_cfg["legend_font"])
            fig.tight_layout(pad=default_cfg["pad"])
            plt.savefig(
                path_reward_error_fmt,
                format=default_cfg["img_fmt"],
                bbox_inches="tight",
            )
            plt.close()

            # action error
            action_error_list = []
            for a in action_list:
                end = min(len(a), len(action_list[-1]))
                action_error_list.append(a[:end] - action_list[-1][:end])
            for j in range(action_dim):
                path_action_error_fmt = os.path.join(
                    self.save_path,
                    "Action-{} error.{}".format(j + 1, default_cfg["img_fmt"]),
                )
                fig, ax = plt.subplots(
                    figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"]
                )
                for i in range(policy_num - 1):
                    legend = (
                        self.legend_list[i]
                        if len(self.legend_list) == policy_num
                        else self.algorithm_list[i]
                    )
                    sns.lineplot(
                        x=step_list[i][:len(action_error_list[i])],
                        y=action_error_list[i][:, j],
                        label="{}".format(legend),
                    )
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel(x_label, default_cfg["label_font"])
                plt.ylabel("Action-{} error".format(j + 1), default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(
                    path_action_error_fmt,
                    format=default_cfg["img_fmt"],
                    bbox_inches="tight",
                )
                plt.close()

                # save action error data to csv
                action_error_data = pd.DataFrame(data=[a[:, j] for a in action_error_list])
                action_error_data.to_csv(
                    os.path.join(self.save_path, "Action-{} error.csv".format(j + 1)),
                    encoding="gbk",
                )

            # state error
            state_error_list = []
            for s in state_list:
                end = min(len(s), len(state_list[-1]))
                state_error_list.append(s[:end] - state_list[-1][:end])
            for j in range(state_dim):
                path_state_error_fmt = os.path.join(
                    self.save_path,
                    "State-{} error.{}".format(j + 1, default_cfg["img_fmt"]),
                )
                fig, ax = plt.subplots(
                    figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"]
                )
                for i in range(policy_num - 1):
                    legend = (
                        self.legend_list[i]
                        if len(self.legend_list) == policy_num
                        else self.algorithm_list[i]
                    )
                    sns.lineplot(
                        x=step_list[i][:len(state_error_list[i])],
                        y=state_error_list[i][:, j],
                        label="{}".format(legend),
                    )
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel(x_label, default_cfg["label_font"])
                plt.ylabel("State-{} error".format(j + 1), default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(
                    path_state_error_fmt,
                    format=default_cfg["img_fmt"],
                    bbox_inches="tight",
                )
                plt.close()

                # save state data to csv
                state_error_data = pd.DataFrame(data=[s[:, j] for s in state_error_list])
                state_error_data.to_csv(
                    os.path.join(self.save_path, "State-{} error.csv".format(j + 1)),
                    encoding="gbk",
                )

            # compute relative error with opt
            error_result = {}
            for i in range(policy_num - 1):
                legend = (
                    self.legend_list[i]
                    if len(self.legend_list) == policy_num
                    else "Policy-{}".format(i + 1)
                )
                end = min(len(action_list[i]), len(action_list[-1]))
                error_result.update({legend: {}})
                # action error
                for j in range(action_dim):
                    action_error = {}
                    error_list = np.abs(
                        action_list[i][:end, j] - action_list[-1][:end, j]
                    ) / (
                        np.max(action_list[-1][:end, j])
                        - np.min(action_list[-1][:end, j])
                    )
                    action_error["Max_error"] = "{:.2f}%".format(max(error_list) * 100)
                    action_error["Mean_error"] = "{:.2f}%".format(
                        sum(error_list) / len(error_list) * 100
                    )
                    error_result[legend].update(
                        {"Action-{}".format(j + 1): action_error}
                    )
                # state error
                for j in range(state_dim):
                    state_error = {}
                    error_list = np.abs(
                        state_list[i][:end, j] - state_list[-1][:end, j]
                    ) / (
                        np.max(state_list[-1][:end, j])
                        - np.min(state_list[-1][:end, j])
                    )
                    state_error["Max_error"] = "{:.2f}%".format(max(error_list) * 100)
                    state_error["Mean_error"] = "{:.2f}%".format(
                        sum(error_list) / len(error_list) * 100
                    )
                    error_result[legend].update({"State-{}".format(j + 1): state_error})

            for i in range(self.policy_num):
                legend = (
                    self.legend_list[i]
                    if len(self.legend_list) == policy_num
                    else "Policy-{}".format(i + 1)
                )
                policy_result = pd.DataFrame(data=error_result[legend])
                policy_result.to_excel(os.path.join(self.save_path, "Error-result.xlsx"), legend)
            error_result_data = pd.DataFrame(data=error_result)
            pd.set_option("display.max_columns", None)
            pd.set_option("display.max_rows", None)
            for key, value in error_result_data.items():
                print("===========================================================")
                print("GOPS: Policy {}".format(key))
                for key, value in value.items():
                    print(key, value)

    @staticmethod
    def __load_args(log_policy_dir: str):
        json_path = os.path.join(log_policy_dir, "config.json")
        parser = argparse.ArgumentParser()
        args_dict = vars(parser.parse_args())
        args = get_args_from_json(json_path, args_dict)
        return args

    def __load_all_args(self):
        for i in range(self.policy_num):
            log_policy_dir = self.log_policy_dir_list[i]
            args = self.__load_args(log_policy_dir)
            args['vector_env_num'] = None
            args['gym2gymnasium'] = False
            self.args_list.append(args)
            env_id = args["env_id"]
            self.env_id_list.append(env_id)
            self.algorithm_list.append(args["algorithm"])

    def __load_env(self, use_opt: bool = False):
        if use_opt:
            env = create_env(**self.args)
        else:
            env_args = {
                **self.args,
                "obs_noise_type": self.obs_noise_type,
                "obs_noise_data": self.obs_noise_data,
                "action_noise_type": self.action_noise_type,
                "action_noise_data": self.action_noise_data,
            }
            env = create_env(**env_args)
        if self.save_render:
            video_path = os.path.join(self.save_path, "videos")
            if use_opt:
                name_prefix = "{}_video".format(self.opt_args["opt_controller_type"])
            else:
                name_prefix = "{}_video".format(self.args["algorithm"])
            env = wrappers.RecordVideo(env, video_path, name_prefix=name_prefix)
        self.args["action_high_limit"] = env.action_space.high
        self.args["action_low_limit"] = env.action_space.low
        return env

    def __load_policy(self, log_policy_dir: str, trained_policy_iteration: str):
        # Create policy
        networks = create_approx_contrainer(**self.args)

        # Load trained policy
        log_path = log_policy_dir + "/apprfunc/apprfunc_{}.pkl".format(
            trained_policy_iteration
        )
        networks.load_state_dict(torch.load(log_path))
        return networks

    def __convert_format(self, origin_data_list: list):
        data_list = copy(origin_data_list)
        for i in range(len(origin_data_list)):
            if isinstance(origin_data_list[i], list) or isinstance(
                origin_data_list[i], np.ndarray
            ):
                data_list[i] = self.__convert_format(origin_data_list[i])
            else:
                data_list[i] = "{:.2g}".format(origin_data_list[i])
        return data_list

    def __run_data(self):
        for i in range(self.policy_num):
            log_policy_dir = self.log_policy_dir_list[i]
            trained_policy_iteration = self.trained_policy_iteration_list[i]

            self.args = self.args_list[i]
            print("===========================================================")
            print("*** Begin to run policy {} ***".format(i + 1))
            env = self.__load_env()
            if hasattr(env, "set_mode"):
                env.set_mode("test")

            if hasattr(env, "train_space") and hasattr(env, "work_space"):
                print("Train space: ")
                print(self.__convert_format(env.train_space))
                print("Work space: ")
                print(self.__convert_format(env.work_space))
            networks = self.__load_policy(log_policy_dir, trained_policy_iteration)

            # Run policy
            eval_dict, tracking_dict = self.run_an_episode(
                env, networks, self.init_info, is_opt=False, render=False
            )
            print("Successfully run policy {}".format(i + 1))
            print("===========================================================\n")
            # mp4 to gif
            self.eval_list.append(eval_dict)
            self.tracking_list.append(tracking_dict)

        if self.use_opt:
            if self.load_opt_path is not None:
                eval_dict_opt = np.load(
                    os.path.join(self.load_opt_path, "eval_dict_opt.npy"),
                    allow_pickle=True).item()
                tracking_dict_opt = np.load(
                    os.path.join(self.load_opt_path, "tracking_dict_opt.npy"),
                    allow_pickle=True).item()
                print("Successfully load an optimal controller result!")
                print("===========================================================\n")
            else:
                self.args = self.args_list[self.policy_num - 1]
                print("GOPS: Use an optimal controller")
                env = self.__load_env(use_opt=True)
                print("The environment for opt")
                if hasattr(env, "set_mode"):
                    env.set_mode("test")

                assert (
                    self.opt_args is not None
                ), "Choose to use optimal controller, but the opt_args is None."

                if self.opt_args["opt_controller_type"] == "OPT":
                    assert (
                        env.has_optimal_controller
                    ), "The environment has no theoretical optimal controller."
                    opt_controller = env.control_policy
                elif self.opt_args["opt_controller_type"] == "MPC":
                    if self.opt_args["use_MPC_for_general_env"] == True:
                        self.args_list[self.policy_num - 1]["env"] = env
                        from gops.sys_simulator.opt_controller_for_gen_env import OptController
                    else:
                        from gops.sys_simulator.opt_controller import OptController
                    model = create_env_model(**self.args_list[self.policy_num - 1], mask_at_done=False)
                    opt_args = self.opt_args.copy()
                    opt_args.pop("opt_controller_type")
                    opt_args.pop("use_MPC_for_general_env")
                    opt_controller = OptController(model, **opt_args,)
                else:
                    raise ValueError(
                        "The optimal controller type should be either 'OPT' or 'MPC'."
                    )

                eval_dict_opt, tracking_dict_opt = self.run_an_episode(
                    env, opt_controller, self.init_info, is_opt=True, render=False
                )
                print("Successfully run an optimal controller!")
                print("===========================================================\n")

            if self.opt_args["opt_controller_type"] == "OPT":
                legend = "OPT"
            elif self.opt_args["opt_controller_type"] == "MPC":
                legend = "MPC-" + str(self.opt_args["num_pred_step"])
                if (
                    "use_terminal_cost" not in self.opt_args.keys()
                    or self.opt_args["use_terminal_cost"] == False
                ):
                    legend += " (w/o TC)"
                else:
                    legend += " (w/ TC)"
            self.legend_list.append(legend)

            if self.save_opt:
                np.save(os.path.join(self.save_path, "eval_dict_opt.npy"), eval_dict_opt)
                np.save(os.path.join(self.save_path, "tracking_dict_opt.npy"), tracking_dict_opt)

            self.eval_list.append(eval_dict_opt)
            if self.is_tracking:
                self.tracking_list.append(tracking_dict_opt)

    def __action_noise(self, action: np.ndarray) -> np.ndarray:
        if self.action_noise_type is None:
            return action
        elif self.action_noise_type == "normal":
            return action + np.random.normal(
                loc=self.action_noise_data[0], scale=self.action_noise_data[1]
            )
        elif self.action_noise_type == "uniform":
            return action + np.random.uniform(
                low=self.action_noise_data[0], high=self.action_noise_data[1]
            )

    def __save_mp4_as_gif(self):
        if self.save_render:
            videos_path = os.path.join(self.save_path, "videos")

            videos_list = [i for i in glob.glob(os.path.join(videos_path, "*.mp4"))]
            for v in videos_list:
                mp4togif(v)

    def get_n_verify_env_id(self):
        env_id = self.env_id_list[0]
        for i, eid in enumerate(self.env_id_list):
            assert (
                env_id == eid
            ), "GOPS: policy {} is not trained in the same environment".format(i)
        return env_id

    def run(self):
        self.__run_data()
        self.__save_mp4_as_gif()
        self.draw()

class PolicyRunner_Multiopt:
    """Plot module for trained policy

    :param list log_policy_dir_list: directory of trained policy.
    :param list trained_policy_iteration_list: iteration of trained policy.
    :param bool save_render: save environment animation or not.
    :param list plot_range: customize plot range.
    :param bool is_init_info: customize initial information or not.
    :param dict init_info: initial information.
    :param list legend_list: legends of figures.
    :param bool use_opt: use optimal solution for comparison or not.
    :param Optional[str] load_opt_path: path to load optimal controller result.
    :param dict opt_args: arguments of optimal solution solver.
    :param bool save_opt: save optimal controller result or not.
    :param bool constrained_env: constrained environment or not.
    :param bool is_tracking: tracking problem or not.
    :param bool use_dist: use adversarial action or not.
    :param float dt: time interval between steps.
    :param str obs_noise_type: type of observation noise, "normal" or "uniform".
    :param list obs_noise_data: Mean and
        Standard deviation of Normal distribution or Upper
        and Lower bounds of Uniform distribution.
    :param str action_noise_type: type of action noise, "normal" or "uniform".
    :param list action_noise_data: Mean and
        Standard deviation of Normal distribution or Upper
        and Lower bounds of Uniform distribution.
    """

    def __init__(
            self,
            log_policy_dir_list: list,
            trained_policy_iteration_list: list,
            save_render: bool = False,
            plot_range: list = None,
            is_init_info: bool = False,
            init_info: dict = None,
            legend_list: list = None,
            use_opt: bool = False,
            load_opt_path: Optional[str] = None,
            opt_args: Optional[dict] = None,
            save_opt: bool = True,
            multi_opt: bool = False,
            multi_opt_args: Optional[dict] = None,
            constrained_env: bool = False,
            is_tracking: bool = False,
            use_dist: bool = False,
            dt: float = None,
            obs_noise_type: str = None,
            obs_noise_data: list = None,
            action_noise_type: str = None,
            action_noise_data: list = None,
    ):
        self.log_policy_dir_list = [
            os.path.join(gops_path, d) for d in log_policy_dir_list
        ]
        self.trained_policy_iteration_list = trained_policy_iteration_list
        self.save_render = save_render
        self.args = None
        self.plot_range = plot_range
        if is_init_info:
            self.init_info = init_info
        else:
            self.init_info = {}
        self.legend_list = legend_list
        self.use_opt = use_opt
        if use_opt:
            assert load_opt_path is not None or opt_args is not None
            self.load_opt_path = load_opt_path
            self.opt_args = opt_args
            if isinstance(self.opt_args, dict) and \
                    "use_MPC_for_general_env" not in self.opt_args.keys():
                self.opt_args["use_MPC_for_general_env"] = False
            self.save_opt = save_opt
        self.multi_opt = multi_opt
        if multi_opt:
            self.multi_opt_args = multi_opt_args
        self.constrained_env = constrained_env
        self.use_dist = use_dist
        self.is_tracking = is_tracking
        self.dt = dt
        self.policy_num = len(self.log_policy_dir_list)
        if self.policy_num != len(self.trained_policy_iteration_list):
            raise RuntimeError(
                "The length of policy number is not equal to that of policy iteration"
            )
        self.obs_noise_type = obs_noise_type
        self.obs_noise_data = obs_noise_data
        self.action_noise_type = action_noise_type
        self.action_noise_data = action_noise_data
        self.ref_state_num = 0

        # data for plot
        self.args_list = []
        self.eval_list = []
        self.env_id_list = []
        self.algorithm_list = []
        self.tracking_list = []

        self.__load_all_args()
        self.env_id = self.get_n_verify_env_id()

        # save path
        path = os.path.join(os.path.dirname(__file__), "..", "..", "figures")
        path = os.path.abspath(path)

        algs_name = ""
        for item in self.algorithm_list:
            algs_name = algs_name + item + "-"
        self.save_path = os.path.join(
            path,
            algs_name + self.env_id,
            datetime.datetime.now().strftime("%y%m%d-%H%M%S"),
        )
        os.makedirs(self.save_path, exist_ok=True)

    def compute_action(self, obs: np.ndarray, networks: Any) -> np.ndarray:
        batch_obs = torch.from_numpy(np.expand_dims(obs, axis=0).astype("float32"))
        logits = networks.policy(batch_obs)
        action_distribution = networks.create_action_distributions(logits)
        action = action_distribution.mode()
        action = action.detach().numpy()[0]
        return action

    @staticmethod
    def __load_args(log_policy_dir: str):
        json_path = os.path.join(log_policy_dir, "config.json")
        parser = argparse.ArgumentParser()
        args_dict = vars(parser.parse_args())
        args = get_args_from_json(json_path, args_dict)
        return args

    def __load_all_args(self):
        for i in range(self.policy_num):
            log_policy_dir = self.log_policy_dir_list[i]
            args = self.__load_args(log_policy_dir)
            args['vector_env_num'] = None
            args['gym2gymnasium'] = False
            self.args_list.append(args)
            env_id = args["env_id"]
            self.env_id_list.append(env_id)
            self.algorithm_list.append(args["algorithm"])

    def __load_env(self, use_opt: bool = False):
        if use_opt:
            env = create_env(**self.args)
        else:
            env_args = {
                **self.args,
                "obs_noise_type": self.obs_noise_type,
                "obs_noise_data": self.obs_noise_data,
                "action_noise_type": self.action_noise_type,
                "action_noise_data": self.action_noise_data,
            }
            env = create_env(**env_args)
        if self.save_render:
            video_path = os.path.join(self.save_path, "videos")
            if use_opt:
                name_prefix = "{}_video".format(self.opt_args["opt_controller_type"])
            else:
                name_prefix = "{}_video".format(self.args["algorithm"])
            env = wrappers.RecordVideo(env, video_path, name_prefix=name_prefix)
        self.args["action_high_limit"] = env.action_space.high
        self.args["action_low_limit"] = env.action_space.low
        return env

    def __load_policy(self, log_policy_dir: str, trained_policy_iteration: str):
        # Create policy
        networks = create_approx_contrainer(**self.args)

        # Load trained policy
        log_path = log_policy_dir + "/apprfunc/apprfunc_{}.pkl".format(
            trained_policy_iteration
        )
        networks.load_state_dict(torch.load(log_path))
        return networks

    def __convert_format(self, origin_data_list: list):
        data_list = copy(origin_data_list)
        for i in range(len(origin_data_list)):
            if isinstance(origin_data_list[i], list) or isinstance(
                    origin_data_list[i], np.ndarray
            ):
                data_list[i] = self.__convert_format(origin_data_list[i])
            else:
                data_list[i] = "{:.2g}".format(origin_data_list[i])
        return data_list


    def __action_noise(self, action: np.ndarray) -> np.ndarray:
        if self.action_noise_type is None:
            return action
        elif self.action_noise_type == "normal":
            return action + np.random.normal(
                loc=self.action_noise_data[0], scale=self.action_noise_data[1]
            )
        elif self.action_noise_type == "uniform":
            return action + np.random.uniform(
                low=self.action_noise_data[0], high=self.action_noise_data[1]
            )

    def __save_mp4_as_gif(self):
        if self.save_render:
            videos_path = os.path.join(self.save_path, "videos")

            videos_list = [i for i in glob.glob(os.path.join(videos_path, "*.mp4"))]
            for v in videos_list:
                mp4togif(v)

    def get_n_verify_env_id(self):
        env_id = self.env_id_list[0]
        for i, eid in enumerate(self.env_id_list):
            assert (
                    env_id == eid
            ), "GOPS: policy {} is not trained in the same environment".format(i)
        return env_id

    def run(self):
        self.__run_data()
        self.__save_mp4_as_gif()
        self.draw()

    def __run_data(self):
        for i in range(self.policy_num):
            log_policy_dir = self.log_policy_dir_list[i]
            trained_policy_iteration = self.trained_policy_iteration_list[i]

            self.args = self.args_list[i]
            print("===========================================================")
            print("*** Begin to run policy {} ***".format(i + 1))
            env = self.__load_env()
            if hasattr(env, "set_mode"):
                env.set_mode("test")

            if hasattr(env, "train_space") and hasattr(env, "work_space"):
                print("Train space: ")
                print(self.__convert_format(env.train_space))
                print("Work space: ")
                print(self.__convert_format(env.work_space))
            networks = self.__load_policy(log_policy_dir, trained_policy_iteration)

            # Run policy
            eval_dict, tracking_dict = self.run_an_episode(
                env, networks, self.init_info, is_opt=False, render=False
            )
            print("Successfully run policy {}".format(i + 1))
            print("===========================================================\n")
            # mp4 to gif
            self.eval_list.append(eval_dict)
            self.tracking_list.append(tracking_dict)

        if self.use_opt:
            if self.multi_opt:
                for run_time in range(int(self.multi_opt_args["opt_run_times"])):
                    if self.load_opt_path is not None:
                        eval_dict_opt = np.load(
                            os.path.join(self.load_opt_path, "eval_dict_opt.npy"),
                            allow_pickle=True).item()
                        tracking_dict_opt = np.load(
                            os.path.join(self.load_opt_path, "tracking_dict_opt.npy"),
                            allow_pickle=True).item()
                        print("Successfully load an optimal controller result!")
                        print("===========================================================\n")
                    else:
                        self.args = self.args_list[self.policy_num - 1]
                        print("GOPS: Use an optimal controller")
                        env = self.__load_env(use_opt=True)
                        print("The environment for opt")
                        if hasattr(env, "set_mode"):
                            env.set_mode("test")

                        assert (
                                self.opt_args is not None
                        ), "Choose to use optimal controller, but the opt_args is None."

                        if self.opt_args["opt_controller_type"] == "OPT":
                            assert (
                                env.has_optimal_controller
                            ), "The environment has no theoretical optimal controller."
                            opt_controller = env.control_policy
                        elif self.opt_args["opt_controller_type"] == "MPC":
                            if self.opt_args["use_MPC_for_general_env"] == True:
                                self.args_list[self.policy_num - 1]["env"] = env
                                from gops.sys_simulator.opt_controller_for_gen_env import OptController
                            else:
                                from gops.sys_simulator.opt_controller import OptController
                            model = create_env_model(**self.args_list[self.policy_num - 1], mask_at_done=False)
                            model.update_cost_paras(self.multi_opt_args["cost_paras_list"][run_time])
                            print("The cost_paras for opt:", self.multi_opt_args["cost_paras_list"][run_time])
                            opt_args = self.opt_args.copy()
                            opt_args.pop("opt_controller_type")
                            opt_args.pop("use_MPC_for_general_env")
                            opt_controller = OptController(model, **opt_args, )
                        else:
                            raise ValueError(
                                "The optimal controller type should be either 'OPT' or 'MPC'."
                            )

                        eval_dict_opt, tracking_dict_opt = self.run_an_episode(
                            env, opt_controller, self.init_info, is_opt=True, render=False
                        )
                        print("Successfully run an optimal controller!")
                        print("===========================================================\n")

                    if self.opt_args["opt_controller_type"] == "OPT":
                        legend = "OPT"
                    elif self.opt_args["opt_controller_type"] == "MPC":
                        if self.multi_opt:
                            legend = "PDP-" + str(run_time)
                            if (
                                    "use_terminal_cost" not in self.opt_args.keys()
                                    or self.opt_args["use_terminal_cost"] == False
                            ):
                                legend += " (w/o TC)"
                            else:
                                legend += " (w/ TC)"
                        else:
                            legend = "MPC-" + str(self.opt_args["num_pred_step"])
                            if (
                                    "use_terminal_cost" not in self.opt_args.keys()
                                    or self.opt_args["use_terminal_cost"] == False
                            ):
                                legend += " (w/o TC)"
                            else:
                                legend += " (w/ TC)"
                    self.legend_list.append(legend)

                    if self.save_opt:
                        np.save(os.path.join(self.save_path, "eval_dict_opt.npy"), eval_dict_opt)
                        np.save(os.path.join(self.save_path, "tracking_dict_opt.npy"), tracking_dict_opt)

                    self.eval_list.append(eval_dict_opt)
                    if self.is_tracking:
                        self.tracking_list.append(tracking_dict_opt)

    def run_an_episode(
            self,
            env: Any,
            controller: Any,
            init_info: dict,
            is_opt: bool,
            render: bool = True,
    ) -> Tuple[dict, dict]:
        state_list = []
        action_list = []
        reward_list = []
        constrain_list = []
        obs_list = []
        step = 0
        step_list = []
        calctime_list = []
        info_list = [init_info]
        obs, info = env.reset(**init_info)
        state = env.state
        print("Initial robot state: ")
        print(self.__convert_format(np.asarray(state.robot_state)))
        # plot tracking
        state_with_ref_error = {}
        done = False
        info.update({"TimeLimit.truncated": False})
        while not (done or info["TimeLimit.truncated"]):
            print("step:", step + 1)
            state_list.append(state.robot_state)
            obs_list.append(obs)
            if is_opt:
                if isinstance(env.unwrapped, Env):
                    time_start = time.time()
                    action = controller(state)
                    calc_time = time.time()-time_start
                else:
                    time_start = time.time()
                    action = controller(obs, info)
                    calc_time = time.time() - time_start
            else:
                time_start = time.time()
                action = self.compute_action(obs, controller)
                action = self.__action_noise(action)
                calc_time = time.time() - time_start
            if self.use_dist:
                action = np.hstack((action, env.dist_func(step * env.tau)))
            if self.constrained_env:
                constrain_list.append(info["constraint"])
            if self.is_tracking:
                reference = get_reference_from_info(info)
                state_num = len(reference)
                self.ref_state_num = sum(x is not None for x in reference)
                if step == 0:
                    for i in range(state_num):
                        if reference[i] is not None:
                            state_with_ref_error["state-{}".format(i)] = []
                            state_with_ref_error["ref-{}".format(i)] = []
                            state_with_ref_error["state-{}-error".format(i)] = []

                robot_state = get_robot_state_from_info(info)
                for i in range(state_num):
                    if reference[i] is not None:
                        state_with_ref_error["state-{}".format(i)].append(robot_state[i])
                        state_with_ref_error["ref-{}".format(i)].append(reference[i])
                        state_with_ref_error["state-{}-error".format(i)].append(
                            reference[i] - robot_state[i]
                        )
            next_obs, reward, done, info = env.step(action)

            # save the real action (without scaling)
            action_list.append(info.get("raw_action", action))
            step_list.append(step)
            reward_list.append(reward)
            calctime_list.append(calc_time*1000)
            info_list.append(info)

            obs = next_obs
            state = env.state
            step = step + 1

            if "TimeLimit.truncated" not in info.keys():
                info["TimeLimit.truncated"] = False
            # Draw environment animation
            if render:
                env.render()

        eval_dict = {
            "reward_list": reward_list,
            "action_list": action_list,
            "state_list": state_list,
            "step_list": step_list,
            "obs_list": obs_list,
            "info_list": info_list,
            "calctime_list":calctime_list
        }
        if self.constrained_env:
            eval_dict.update(
                {"constrain_list": constrain_list, }
            )

        if self.is_tracking:
            tracking_dict = state_with_ref_error
        else:
            tracking_dict = {}

        return eval_dict, tracking_dict

    def draw(self):
        fig_size = (
            default_cfg["fig_size"],
            default_cfg["fig_size"],
        )
        action_dim = self.eval_list[0]["action_list"][0].shape[0]
        state_dim = self.eval_list[0]["state_list"][0].shape[0]
        if self.constrained_env:
            constrain_dim = self.eval_list[0]["constrain_list"][0].shape[0]
        policy_num = len(self.algorithm_list)
        if self.use_opt:
            legend = ""
            if self.multi_opt:
                policy_num += self.multi_opt_args["opt_run_times"]
            else:
                policy_num += 1
            if self.multi_opt:
                for run_time in range(int(self.multi_opt_args["opt_run_times"])):
                    if self.opt_args["opt_controller_type"] == "OPT":
                        legend = "OPT"
                    elif self.opt_args["opt_controller_type"] == "MPC":
                        legend = "MPC-" + str(run_time)
                        if (
                                "use_terminal_cost" not in self.opt_args.keys()
                                or self.opt_args["use_terminal_cost"] is False
                        ):
                            legend += " (w/o TC)"
                        else:
                            legend += " (w/ TC)"
                    self.algorithm_list.append(legend)
            else:
                if self.opt_args["opt_controller_type"] == "OPT":
                    legend = "OPT"
                elif self.opt_args["opt_controller_type"] == "MPC":
                    legend = "MPC-" + str(self.opt_args["num_pred_step"])
                    if (
                            "use_terminal_cost" not in self.opt_args.keys()
                            or self.opt_args["use_terminal_cost"] is False
                    ):
                        legend += " (w/o TC)"
                    else:
                        legend += " (w/ TC)"
                self.algorithm_list.append(legend)

        # Create initial list
        reward_list = []
        action_list = []
        state_list = []
        step_list = []
        calctime_list = []
        state_ref_error_list = []
        constrain_list = []
        # Put data into list
        for i in range(policy_num):
            reward_list.append(np.array(self.eval_list[i]["reward_list"]))
            action_list.append(np.array(self.eval_list[i]["action_list"]))
            state_list.append(np.array(self.eval_list[i]["state_list"]))
            step_list.append(np.array(self.eval_list[i]["step_list"]))
            calctime_list.append(np.array(self.eval_list[i]["calctime_list"]))
            if self.constrained_env:
                constrain_list.append(np.stack(self.eval_list[i]["constrain_list"]))
            if self.is_tracking:
                state_ref_error_list.append(self.tracking_list[i])

        if self.plot_range is None:
            pass
        elif len(self.plot_range) == 2:

            for i in range(policy_num):
                start_range = self.plot_range[0]
                end_range = min(self.plot_range[1], reward_list[i].shape[0])

                reward_list[i] = reward_list[i][start_range:end_range]
                action_list[i] = action_list[i][start_range:end_range]
                state_list[i] = state_list[i][start_range:end_range]
                step_list[i] = step_list[i][start_range:end_range]
                calctime_list[i] = calctime_list[i][start_range:end_range]
                if self.constrained_env:
                    constrain_list[i] = constrain_list[i][start_range:end_range]
                if self.is_tracking:
                    for key, value in self.tracking_list[i].items():
                        self.tracking_list[i][key] = value[start_range:end_range]
        else:
            raise NotImplementedError("Figure range is wrong")

        if self.dt is None:
            x_label = "Time step"
        else:
            step_list = [s * self.dt for s in step_list]
            x_label = "Time (s)"

        # Plot reward
        path_reward_fmt = os.path.join(
            self.save_path, "Reward.{}".format(default_cfg["img_fmt"])
        )
        fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

        # save reward data to csv
        reward_data = pd.DataFrame(data=reward_list)
        reward_data.to_csv(os.path.join(self.save_path, "Reward.csv"), encoding="gbk")

        for i in range(policy_num):
            legend = (
                self.legend_list[i]
                if len(self.legend_list) == policy_num
                else self.algorithm_list[i]
            )
            sns.lineplot(x=step_list[i], y=reward_list[i], label="{}".format(legend))
        plt.tick_params(labelsize=default_cfg["tick_size"])
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
        plt.xlabel(x_label, default_cfg["label_font"])
        plt.ylabel("Reward", default_cfg["label_font"])
        plt.legend(loc="best", prop=default_cfg["legend_font"])
        fig.tight_layout(pad=default_cfg["pad"])
        plt.savefig(path_reward_fmt, format=default_cfg["img_fmt"], bbox_inches="tight")
        plt.close()

        # plot action
        for j in range(action_dim):
            path_action_fmt = os.path.join(
                self.save_path, "Action-{}.{}".format(j + 1, default_cfg["img_fmt"])
            )
            fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

            # save action data to csv
            action_data = pd.DataFrame(data=[a[:, j] for a in action_list])
            action_data.to_csv(
                os.path.join(self.save_path, "Action-{}.csv".format(j + 1)),
                encoding="gbk",
            )

            for i in range(policy_num):
                legend = (
                    self.legend_list[i]
                    if len(self.legend_list) == policy_num
                    else self.algorithm_list[i]
                )
                sns.lineplot(
                    x=step_list[i], y=action_list[i][:, j], label="{}".format(legend)
                )
            plt.tick_params(labelsize=default_cfg["tick_size"])
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
            plt.xlabel(x_label, default_cfg["label_font"])
            plt.ylabel("Action-{}".format(j + 1), default_cfg["label_font"])
            plt.legend(loc="best", prop=default_cfg["legend_font"])
            fig.tight_layout(pad=default_cfg["pad"])
            plt.savefig(
                path_action_fmt, format=default_cfg["img_fmt"], bbox_inches="tight"
            )
            plt.close()

        # plot state
        for j in range(state_dim):
            path_state_fmt = os.path.join(
                self.save_path, "State-{}.{}".format(j + 1, default_cfg["img_fmt"])
            )
            fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

            # save state data to csv
            state_data = pd.DataFrame(data=[s[:, j] for s in state_list])
            state_data.to_csv(
                os.path.join(self.save_path, "State-{}.csv".format(j + 1)),
                encoding="gbk",
            )

            for i in range(policy_num):
                legend = (
                    self.legend_list[i]
                    if len(self.legend_list) == policy_num
                    else self.algorithm_list[i]
                )
                sns.lineplot(
                    x=step_list[i], y=state_list[i][:, j], label="{}".format(legend)
                )
            plt.tick_params(labelsize=default_cfg["tick_size"])
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
            plt.xlabel(x_label, default_cfg["label_font"])
            plt.ylabel("State-{}".format(j + 1), default_cfg["label_font"])
            plt.legend(loc="best", prop=default_cfg["legend_font"])
            fig.tight_layout(pad=default_cfg["pad"])
            plt.savefig(
                path_state_fmt, format=default_cfg["img_fmt"], bbox_inches="tight"
            )
            plt.close()

        # plot tracking
        if self.is_tracking:
            # find index of the longest trajectory
            traj_lens = [len(r) for r in reward_list]
            longest_traj_index = np.argmax(traj_lens)

            for j in range(self.ref_state_num):

                # plot state and ref
                path_tracking_state_fmt = os.path.join(
                    self.save_path, "Ref - State - {}.{}".format(j + 1, default_cfg["img_fmt"])
                )
                fig, ax = plt.subplots(
                    figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"]
                )
                # save tracking state data to csv
                tracking_state_data = []
                for i in range(policy_num):
                    legend = (
                        self.legend_list[i]
                        if len(self.legend_list) == policy_num
                        else self.algorithm_list[i]
                    )
                    sns.lineplot(
                        x=step_list[i],
                        y=state_ref_error_list[i]["state-{}".format(j)],
                        label="{}".format(legend),
                    )
                    tracking_state_data.append(
                        state_ref_error_list[i]["state-{}".format(j)]
                    )
                sns.lineplot(
                    x=step_list[longest_traj_index],
                    y=state_ref_error_list[longest_traj_index]["ref-{}".format(j)],
                    label="ref",
                )
                tracking_state_data.append(state_ref_error_list[longest_traj_index]["ref-{}".format(j)])
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel(x_label, default_cfg["label_font"])
                plt.ylabel("State-{}".format(j + 1), default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(
                    path_tracking_state_fmt,
                    format=default_cfg["img_fmt"],
                    bbox_inches="tight",
                )
                plt.close()

                tracking_state_data = pd.DataFrame(data=tracking_state_data)
                tracking_state_data.to_csv(
                    os.path.join(self.save_path, "State-{}.csv".format(j + 1)),
                    encoding="gbk",
                )

                # plot state-ref error
                path_tracking_error_fmt = os.path.join(
                    self.save_path,
                    "Ref - State - Error{}.{}".format(j + 1, default_cfg["img_fmt"]),
                )
                fig, ax = plt.subplots(
                    figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"]
                )
                # save tracking error data to csv
                tracking_error_data = []
                for i in range(policy_num):
                    legend = (
                        self.legend_list[i]
                        if len(self.legend_list) == policy_num
                        else self.algorithm_list[i]
                    )
                    sns.lineplot(
                        x=step_list[i],
                        y=state_ref_error_list[i]["state-{}-error".format(j)],
                        label="{}".format(legend),
                    )
                    tracking_error_data.append(
                        state_ref_error_list[i]["state-{}-error".format(j)]
                    )
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel(x_label, default_cfg["label_font"])
                plt.ylabel("Ref$-$State-Error{}".format(j + 1), default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(
                    path_tracking_error_fmt,
                    format=default_cfg["img_fmt"],
                    bbox_inches="tight",
                )
                plt.close()

                tracking_error_data = pd.DataFrame(data=tracking_error_data)
                tracking_error_data.to_csv(
                    os.path.join(self.save_path, "Ref-State-Error{}.csv".format(j + 1)),
                    encoding="gbk",
                )
        # plot calculation time
        path_state_fmt = os.path.join(
            self.save_path, "Calc time.{}".format(default_cfg["img_fmt"])
        )
        fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

        # save state data to csv
        state_data = pd.DataFrame(data=[s[:] for s in calctime_list])
        state_data.to_csv(
            os.path.join(self.save_path, "Calc time.csv"),
            encoding="gbk",
        )

        for i in range(policy_num):
            legend = (
                self.legend_list[i]
                if len(self.legend_list) == policy_num
                else self.algorithm_list[i]
            )
            sns.lineplot(
                x=step_list[i], y=calctime_list[i][:], label="{}".format(legend)
            )
        plt.tick_params(labelsize=default_cfg["tick_size"])
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
        plt.xlabel(x_label, default_cfg["label_font"])
        plt.ylabel("Calc Time [ms]", default_cfg["label_font"])
        plt.legend(loc="best", prop=default_cfg["legend_font"])
        fig.tight_layout(pad=default_cfg["pad"])
        plt.savefig(
            path_state_fmt, format=default_cfg["img_fmt"], bbox_inches="tight"
        )
        plt.close()
        # plot constraint value
        if self.constrained_env:
            for j in range(constrain_dim):
                path_constraint_fmt = os.path.join(
                    self.save_path,
                    "Constrain-{}.{}".format(j + 1, default_cfg["img_fmt"]),
                )
                fig, ax = plt.subplots(
                    figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"]
                )

                # save reward data to csv
                constrain_data = pd.DataFrame(data=[c[:, j] for c in constrain_list])
                constrain_data.to_csv(
                    os.path.join(self.save_path, "Constrain-{}.csv".format(j + 1)),
                    encoding="gbk",
                )

                for i in range(policy_num):
                    legend = (
                        self.legend_list[i]
                        if len(self.legend_list) == policy_num
                        else self.algorithm_list[i]
                    )
                    sns.lineplot(
                        x=step_list[i],
                        y=constrain_list[i][:, j],
                        label="{}".format(legend),
                    )
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel(x_label, default_cfg["label_font"])
                plt.ylabel("Constrain-{}".format(j + 1), default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(
                    path_constraint_fmt,
                    format=default_cfg["img_fmt"],
                    bbox_inches="tight",
                )
                plt.close()

        # plot error with opt
        if self.use_opt:
            # reward error
            path_reward_error_fmt = os.path.join(
                self.save_path, "Reward error.{}".format(default_cfg["img_fmt"])
            )
            fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

            # save reward error data to csv
            reward_error_list = []
            for r in reward_list:
                end = min(len(r), len(reward_list[-1]))
                reward_error_list.append(r[:end] - reward_list[-1][:end])
            reward_error_data = pd.DataFrame(data=reward_error_list)
            reward_error_data.to_csv(
                os.path.join(self.save_path, "Reward error.csv"), encoding="gbk"
            )

            for i in range(policy_num - 1):
                legend = (
                    self.legend_list[i]
                    if len(self.legend_list) == policy_num
                    else self.algorithm_list[i]
                )
                sns.lineplot(
                    x=step_list[i][:len(reward_error_list[i])],
                    y=reward_error_list[i], label="{}".format(legend)
                )
            plt.tick_params(labelsize=default_cfg["tick_size"])
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
            plt.xlabel(x_label, default_cfg["label_font"])
            plt.ylabel("Reward error", default_cfg["label_font"])
            plt.legend(loc="best", prop=default_cfg["legend_font"])
            fig.tight_layout(pad=default_cfg["pad"])
            plt.savefig(
                path_reward_error_fmt,
                format=default_cfg["img_fmt"],
                bbox_inches="tight",
            )
            plt.close()

            # action error
            action_error_list = []
            for a in action_list:
                end = min(len(a), len(action_list[-1]))
                action_error_list.append(a[:end] - action_list[-1][:end])
            for j in range(action_dim):
                path_action_error_fmt = os.path.join(
                    self.save_path,
                    "Action-{} error.{}".format(j + 1, default_cfg["img_fmt"]),
                )
                fig, ax = plt.subplots(
                    figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"]
                )
                for i in range(policy_num - 1):
                    legend = (
                        self.legend_list[i]
                        if len(self.legend_list) == policy_num
                        else self.algorithm_list[i]
                    )
                    sns.lineplot(
                        x=step_list[i][:len(action_error_list[i])],
                        y=action_error_list[i][:, j],
                        label="{}".format(legend),
                    )
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel(x_label, default_cfg["label_font"])
                plt.ylabel("Action-{} error".format(j + 1), default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(
                    path_action_error_fmt,
                    format=default_cfg["img_fmt"],
                    bbox_inches="tight",
                )
                plt.close()

                # save action error data to csv
                action_error_data = pd.DataFrame(data=[a[:, j] for a in action_error_list])
                action_error_data.to_csv(
                    os.path.join(self.save_path, "Action-{} error.csv".format(j + 1)),
                    encoding="gbk",
                )

            # state error
            state_error_list = []
            for s in state_list:
                end = min(len(s), len(state_list[-1]))
                state_error_list.append(s[:end] - state_list[-1][:end])
            for j in range(state_dim):
                path_state_error_fmt = os.path.join(
                    self.save_path,
                    "State-{} error.{}".format(j + 1, default_cfg["img_fmt"]),
                )
                fig, ax = plt.subplots(
                    figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"]
                )
                for i in range(policy_num - 1):
                    legend = (
                        self.legend_list[i]
                        if len(self.legend_list) == policy_num
                        else self.algorithm_list[i]
                    )
                    sns.lineplot(
                        x=step_list[i][:len(state_error_list[i])],
                        y=state_error_list[i][:, j],
                        label="{}".format(legend),
                    )
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel(x_label, default_cfg["label_font"])
                plt.ylabel("State-{} error".format(j + 1), default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(
                    path_state_error_fmt,
                    format=default_cfg["img_fmt"],
                    bbox_inches="tight",
                )
                plt.close()

                # save state data to csv
                state_error_data = pd.DataFrame(data=[s[:, j] for s in state_error_list])
                state_error_data.to_csv(
                    os.path.join(self.save_path, "State-{} error.csv".format(j + 1)),
                    encoding="gbk",
                )

            # compute relative error with opt
            error_result = {}
            for i in range(policy_num - 1):
                legend = (
                    self.legend_list[i]
                    if len(self.legend_list) == policy_num
                    else "Policy-{}".format(i + 1)
                )
                end = min(len(action_list[i]), len(action_list[-1]))
                error_result.update({legend: {}})
                # action error
                for j in range(action_dim):
                    action_error = {}
                    error_list = np.abs(
                        action_list[i][:end, j] - action_list[-1][:end, j]
                    ) / (
                                         np.max(action_list[-1][:end, j])
                                         - np.min(action_list[-1][:end, j])
                                 )
                    action_error["Max_error"] = "{:.2f}%".format(max(error_list) * 100)
                    action_error["Mean_error"] = "{:.2f}%".format(
                        sum(error_list) / len(error_list) * 100
                    )
                    error_result[legend].update(
                        {"Action-{}".format(j + 1): action_error}
                    )
                # state error
                for j in range(state_dim):
                    state_error = {}
                    error_list = np.abs(
                        state_list[i][:end, j] - state_list[-1][:end, j]
                    ) / (
                                         np.max(state_list[-1][:end, j])
                                         - np.min(state_list[-1][:end, j])
                                 )
                    state_error["Max_error"] = "{:.2f}%".format(max(error_list) * 100)
                    state_error["Mean_error"] = "{:.2f}%".format(
                        sum(error_list) / len(error_list) * 100
                    )
                    error_result[legend].update({"State-{}".format(j + 1): state_error})

            for i in range(self.policy_num):
                legend = (
                    self.legend_list[i]
                    if len(self.legend_list) == policy_num
                    else "Policy-{}".format(i + 1)
                )
                policy_result = pd.DataFrame(data=error_result[legend])
                policy_result.to_excel(os.path.join(self.save_path, "Error-result.xlsx"), legend)
            error_result_data = pd.DataFrame(data=error_result)
            pd.set_option("display.max_columns", None)
            pd.set_option("display.max_rows", None)
            for key, value in error_result_data.items():
                print("===========================================================")
                print("GOPS: Policy {}".format(key))
                for key, value in value.items():
                    print(key, value)

class OptRunner:
    """Plot module for trained policy

    :param list log_policy_dir_list: directory of trained policy.
    :param list trained_policy_iteration_list: iteration of trained policy.
    :param bool save_render: save environment animation or not.
    :param list plot_range: customize plot range.
    :param bool is_init_info: customize initial information or not.
    :param dict init_info: initial information.
    :param list legend_list: legends of figures.
    :param bool use_opt: use optimal solution for comparison or not.
    :param Optional[str] load_opt_path: path to load optimal controller result.
    :param dict opt_args: arguments of optimal solution solver.
    :param bool save_opt: save optimal controller result or not.
    :param bool constrained_env: constrained environment or not.
    :param bool is_tracking: tracking problem or not.
    :param bool use_dist: use adversarial action or not.
    :param float dt: time interval between steps.
    :param str obs_noise_type: type of observation noise, "normal" or "uniform".
    :param list obs_noise_data: Mean and
        Standard deviation of Normal distribution or Upper
        and Lower bounds of Uniform distribution.
    :param str action_noise_type: type of action noise, "normal" or "uniform".
    :param list action_noise_data: Mean and
        Standard deviation of Normal distribution or Upper
        and Lower bounds of Uniform distribution.
    """

    def __init__(
            self,
            log_policy_dir_list: list,
            env_id:str = None,
            save_render: bool = False,
            plot_range: list = None,
            is_init_info: bool = False,
            init_info: dict = None,
            legend_list: list = None,
            use_opt: bool = False,
            load_opt_path: Optional[str] = None,
            opt_args: Optional[dict] = None,
            save_opt: bool = True,
            multi_opt: bool = False,
            multi_opt_args: Optional[dict] = None,
            constrained_env: bool = False,
            is_tracking: bool = False,
            use_dist: bool = False,
            dt: float = None,
            obs_noise_type: str = None,
            obs_noise_data: list = None,
            action_noise_type: str = None,
            action_noise_data: list = None,
    ):
        self.log_policy_dir_list = [
            os.path.join(gops_path, d) for d in log_policy_dir_list
        ]
        self.save_render = save_render
        self.args = None
        self.plot_range = plot_range
        if is_init_info:
            self.init_info = init_info
        else:
            self.init_info = {}
        self.legend_list = legend_list
        self.use_opt = use_opt
        if use_opt:
            assert load_opt_path is not None or opt_args is not None
            self.load_opt_path = load_opt_path
            self.opt_args = opt_args
            if isinstance(self.opt_args, dict) and \
                    "use_MPC_for_general_env" not in self.opt_args.keys():
                self.opt_args["use_MPC_for_general_env"] = False
            self.save_opt = save_opt
        self.multi_opt = multi_opt
        if multi_opt:
            self.multi_opt_args = multi_opt_args
        self.constrained_env = constrained_env
        self.use_dist = use_dist
        self.is_tracking = is_tracking
        self.dt = dt
        self.policy_num = len(self.log_policy_dir_list)
        self.obs_noise_type = obs_noise_type
        self.obs_noise_data = obs_noise_data
        self.action_noise_type = action_noise_type
        self.action_noise_data = action_noise_data
        self.ref_state_num = 0

        # data for plot
        self.args_list = []
        self.eval_list = []
        self.env_id_list = []
        self.algorithm_list = []
        self.tracking_list = []

        self.__load_all_args()
        self.env_id = self.get_n_verify_env_id()

        # save path
        path = os.path.join(os.path.dirname(__file__), "..", "..", "figures")
        path = os.path.abspath(path)

        algs_name = "MPC-"
        self.save_path = os.path.join(
            path,
            algs_name + self.env_id,
            datetime.datetime.now().strftime("%y%m%d-%H%M%S"),
        )
        os.makedirs(self.save_path, exist_ok=True)


    @staticmethod
    def __load_args(log_policy_dir: str):
        json_path = os.path.join(log_policy_dir, "config.json")
        parser = argparse.ArgumentParser()
        args_dict = vars(parser.parse_args())
        args = get_args_from_json(json_path, args_dict)
        return args

    def __load_all_args(self):
        log_policy_dir = self.log_policy_dir_list[0]
        args = self.__load_args(log_policy_dir)
        args['vector_env_num'] = None
        args['gym2gymnasium'] = False
        self.args_list.append(args)
        env_id = args["env_id"]
        self.env_id_list.append(env_id)
        self.algorithm_list.append(args["algorithm"])

    def __load_env(self, use_opt: bool = False):
        if use_opt:
            env = create_env(**self.args)
        else:
            env_args = {
                **self.args,
                "obs_noise_type": self.obs_noise_type,
                "obs_noise_data": self.obs_noise_data,
                "action_noise_type": self.action_noise_type,
                "action_noise_data": self.action_noise_data,
            }
            env = create_env(**env_args)
        if self.save_render:
            video_path = os.path.join(self.save_path, "videos")
            if use_opt:
                name_prefix = "{}_video".format(self.opt_args["opt_controller_type"])
            else:
                name_prefix = "{}_video".format(self.args["algorithm"])
            env = wrappers.RecordVideo(env, video_path, name_prefix=name_prefix)
        # self.args["action_high_limit"] = self.args['action_high_limit']#env.action_space.high
        # self.args["action_low_limit"] = env.action_space.low
        return env

    def __load_policy(self, log_policy_dir: str, trained_policy_iteration: str):
        # Create policy
        networks = create_approx_contrainer(**self.args)

        # Load trained policy
        log_path = log_policy_dir + "/apprfunc/apprfunc_{}.pkl".format(
            trained_policy_iteration
        )
        networks.load_state_dict(torch.load(log_path))
        return networks

    def __convert_format(self, origin_data_list: list):
        data_list = copy(origin_data_list)
        for i in range(len(origin_data_list)):
            if isinstance(origin_data_list[i], list) or isinstance(
                    origin_data_list[i], np.ndarray
            ):
                data_list[i] = self.__convert_format(origin_data_list[i])
            else:
                data_list[i] = "{:.2g}".format(origin_data_list[i])
        return data_list

    def __action_noise(self, action: np.ndarray) -> np.ndarray:
        if self.action_noise_type is None:
            return action
        elif self.action_noise_type == "normal":
            return action + np.random.normal(
                loc=self.action_noise_data[0], scale=self.action_noise_data[1]
            )
        elif self.action_noise_type == "uniform":
            return action + np.random.uniform(
                low=self.action_noise_data[0], high=self.action_noise_data[1]
            )

    def __save_mp4_as_gif(self):
        if self.save_render:
            videos_path = os.path.join(self.save_path, "videos")

            videos_list = [i for i in glob.glob(os.path.join(videos_path, "*.mp4"))]
            for v in videos_list:
                mp4togif(v)

    def get_n_verify_env_id(self):
        env_id = self.env_id_list[0]
        for i, eid in enumerate(self.env_id_list):
            assert (
                    env_id == eid
            ), "GOPS: policy {} is not trained in the same environment".format(i)
        return env_id

    def run(self):
        self.__run_data()
        self.__save_mp4_as_gif()
        self.draw()

    def __run_data(self):
        if self.use_opt:
            if self.multi_opt:
                for run_time in range(int(self.multi_opt_args["opt_run_times"])):
                    if self.load_opt_path is not None:
                        eval_dict_opt = np.load(
                            os.path.join(self.load_opt_path, "eval_dict_opt.npy"),
                            allow_pickle=True).item()
                        tracking_dict_opt = np.load(
                            os.path.join(self.load_opt_path, "tracking_dict_opt.npy"),
                            allow_pickle=True).item()
                        print("Successfully load an optimal controller result!")
                        print("===========================================================\n")
                    else:
                        self.args = self.args_list[0]
                        print("GOPS: Use an optimal controller")
                        env = self.__load_env(use_opt=True)
                        print("The environment for opt")
                        if hasattr(env, "set_mode"):
                            env.set_mode("test")

                        assert (
                                self.opt_args is not None
                        ), "Choose to use optimal controller, but the opt_args is None."

                        if self.opt_args["opt_controller_type"] == "OPT":
                            assert (
                                env.has_optimal_controller
                            ), "The environment has no theoretical optimal controller."
                            opt_controller = env.control_policy
                        elif self.opt_args["opt_controller_type"] == "MPC":
                            if self.opt_args["use_MPC_for_general_env"] == True:
                                self.args_list[0]["env"] = env
                                from gops.sys_simulator.opt_controller_for_gen_env import OptController
                            else:
                                from gops.sys_simulator.opt_controller import OptController
                            model = create_env_model(**self.args_list[0], mask_at_done=False)
                            model.update_cost_paras(self.multi_opt_args["cost_paras_list"][run_time])
                            print("The cost_paras for opt:", self.multi_opt_args["cost_paras_list"][run_time])
                            opt_args = self.opt_args.copy()
                            opt_args.pop("opt_controller_type")
                            opt_args.pop("use_MPC_for_general_env")
                            opt_controller = OptController(model, **opt_args, )
                        else:
                            raise ValueError(
                                "The optimal controller type should be either 'OPT' or 'MPC'."
                            )

                        eval_dict_opt, tracking_dict_opt = self.run_an_episode(
                            env, opt_controller, self.init_info, is_opt=True, render=False
                        )
                        print("Successfully run an optimal controller!")
                        print("===========================================================\n")

                    if self.opt_args["opt_controller_type"] == "OPT":
                        legend = "OPT"
                    elif self.opt_args["opt_controller_type"] == "MPC":
                        if self.multi_opt:
                            legend = "MPC-" + str(run_time)
                            if (
                                    "use_terminal_cost" not in self.opt_args.keys()
                                    or self.opt_args["use_terminal_cost"] == False
                            ):
                                legend += " (w/o TC)"
                            else:
                                legend += " (w/ TC)"
                        else:
                            legend = "MPC-" + str(self.opt_args["num_pred_step"])
                            if (
                                    "use_terminal_cost" not in self.opt_args.keys()
                                    or self.opt_args["use_terminal_cost"] == False
                            ):
                                legend += " (w/o TC)"
                            else:
                                legend += " (w/ TC)"
                    self.legend_list.append(legend)

                    if self.save_opt:
                        np.save(os.path.join(self.save_path, "eval_dict_opt.npy"), eval_dict_opt)
                        np.save(os.path.join(self.save_path, "tracking_dict_opt.npy"), tracking_dict_opt)

                    self.eval_list.append(eval_dict_opt)
                    if self.is_tracking:
                        self.tracking_list.append(tracking_dict_opt)
                if self.opt_args["opt_controller_type"] == "OPT":
                    legend = "OPT"
                elif self.opt_args["opt_controller_type"] == "MPC":
                    legend = "MPC-" + str(self.opt_args["num_pred_step"])
                    if (
                            "use_terminal_cost" not in self.opt_args.keys()
                            or self.opt_args["use_terminal_cost"] == False
                    ):
                        legend += " (w/o TC)"
                    else:
                        legend += " (w/ TC)"
                self.legend_list.append(legend)

                if self.save_opt:
                    np.save(os.path.join(self.save_path, "eval_dict_opt.npy"), eval_dict_opt)
                    np.save(os.path.join(self.save_path, "tracking_dict_opt.npy"), tracking_dict_opt)

                self.eval_list.append(eval_dict_opt)
                if self.is_tracking:
                    self.tracking_list.append(tracking_dict_opt)
            else:
                if self.load_opt_path is not None:
                    eval_dict_opt = np.load(
                        os.path.join(self.load_opt_path, "eval_dict_opt.npy"),
                        allow_pickle=True).item()
                    tracking_dict_opt = np.load(
                        os.path.join(self.load_opt_path, "tracking_dict_opt.npy"),
                        allow_pickle=True).item()
                    print("Successfully load an optimal controller result!")
                    print("===========================================================\n")
                else:
                    self.args = self.args_list[0]
                    print("GOPS: Use an optimal controller")
                    env = self.__load_env(use_opt=True)
                    print("The environment for opt")
                    if hasattr(env, "set_mode"):
                        env.set_mode("test")

                    assert (
                            self.opt_args is not None
                    ), "Choose to use optimal controller, but the opt_args is None."

                    if self.opt_args["opt_controller_type"] == "OPT":
                        assert (
                            env.has_optimal_controller
                        ), "The environment has no theoretical optimal controller."
                        opt_controller = env.control_policy
                    elif self.opt_args["opt_controller_type"] == "MPC":
                        if self.opt_args["use_MPC_for_general_env"] == True:
                            self.args_list[0]["env"] = env
                            from gops.sys_simulator.opt_controller_for_gen_env import OptController
                        else:
                            from gops.sys_simulator.opt_controller import OptController
                        model = create_env_model(**self.args_list[0], mask_at_done=False)
                        opt_args = self.opt_args.copy()
                        opt_args.pop("opt_controller_type")
                        opt_args.pop("use_MPC_for_general_env")
                        opt_controller = OptController(model, **opt_args, )
                    else:
                        raise ValueError(
                            "The optimal controller type should be either 'OPT' or 'MPC'."
                        )

                    eval_dict_opt, tracking_dict_opt = self.run_an_episode(
                        env, opt_controller, self.init_info, is_opt=True, render=False
                    )
                    print("Successfully run an optimal controller!")
                    print("===========================================================\n")

                if self.opt_args["opt_controller_type"] == "OPT":
                    legend = "OPT"
                elif self.opt_args["opt_controller_type"] == "MPC":
                    legend = "MPC-" + str(self.opt_args["num_pred_step"])
                    if (
                            "use_terminal_cost" not in self.opt_args.keys()
                            or self.opt_args["use_terminal_cost"] == False
                    ):
                        legend += " (w/o TC)"
                    else:
                        legend += " (w/ TC)"
                self.legend_list.append(legend)

                if self.save_opt:
                    np.save(os.path.join(self.save_path, "eval_dict_opt.npy"), eval_dict_opt)
                    np.save(os.path.join(self.save_path, "tracking_dict_opt.npy"), tracking_dict_opt)

                self.eval_list.append(eval_dict_opt)
                if self.is_tracking:
                    self.tracking_list.append(tracking_dict_opt)

    def run_an_episode(
            self,
            env: Any,
            controller: Any,
            init_info: dict,
            is_opt: bool,
            render: bool = True,
    ) -> Tuple[dict, dict]:
        state_list = []
        action_list = []
        reward_list = []
        constrain_list = []
        obs_list = []
        step = 0
        step_list = []
        calctime_list = []
        info_list = [init_info]
        obs, info = env.reset(**init_info)
        state = env.state
        print("Initial robot state: ")
        print(self.__convert_format(np.asarray(state.robot_state)))
        # plot tracking
        state_with_ref_error = {}
        done = False
        info.update({"TimeLimit.truncated": False})
        while not (done or info["TimeLimit.truncated"]):
            print("step:", step + 1)
            state_list.append(state.robot_state)
            obs_list.append(obs)
            if is_opt:
                if isinstance(env.unwrapped, Env):
                    time_start = time.time()
                    action = controller(state)
                    calc_time = time.time()-time_start
                else:
                    time_start = time.time()
                    action = controller(obs, info)
                    calc_time = time.time()-time_start
            else:
                time_start = time.time()
                action = self.compute_action(obs, controller)
                action = self.__action_noise(action)
                calc_time = time.time() - time_start
            if self.use_dist:
                action = np.hstack((action, env.dist_func(step * env.tau)))
            if self.constrained_env:
                constrain_list.append(info["constraint"])
            if self.is_tracking:
                reference = get_reference_from_info(info)
                state_num = len(reference)
                self.ref_state_num = sum(x is not None for x in reference)
                if step == 0:
                    for i in range(state_num):
                        if reference[i] is not None:
                            state_with_ref_error["state-{}".format(i)] = []
                            state_with_ref_error["ref-{}".format(i)] = []
                            state_with_ref_error["state-{}-error".format(i)] = []

                robot_state = get_robot_state_from_info(info)
                for i in range(state_num):
                    if reference[i] is not None:
                        state_with_ref_error["state-{}".format(i)].append(robot_state[i])
                        state_with_ref_error["ref-{}".format(i)].append(reference[i])
                        state_with_ref_error["state-{}-error".format(i)].append(
                            reference[i] - robot_state[i]
                        )
            next_obs, reward, done, info = env.step(action)
            # save the real action (without scaling)
            # todo 若做model based planning, 需要把action的第0个存到action_list中
            action_list.append(info.get("raw_action", action[0, :]))#
            step_list.append(step)
            reward_list.append(reward)
            info_list.append(info)
            calctime_list.append(calc_time*1000)

            obs = next_obs
            state = env.state
            step = step + 1

            if "TimeLimit.truncated" not in info.keys():
                info["TimeLimit.truncated"] = False
            # Draw environment animation
            if render:
                env.render()

        eval_dict = {
            "reward_list": reward_list,
            "action_list": action_list,
            "state_list": state_list,
            "step_list": step_list,
            "obs_list": obs_list,
            "info_list": info_list,
            "calctime_list": calctime_list
        }
        if self.constrained_env:
            eval_dict.update(
                {"constrain_list": constrain_list, }
            )

        if self.is_tracking:
            tracking_dict = state_with_ref_error
        else:
            tracking_dict = {}

        return eval_dict, tracking_dict

    def draw(self):
        fig_size = (
            default_cfg["fig_size"],
            default_cfg["fig_size"],
        )
        action_dim = self.eval_list[0]["action_list"][0].shape[0]
        state_dim = self.eval_list[0]["state_list"][0].shape[0]
        if self.constrained_env:
            constrain_dim = self.eval_list[0]["constrain_list"][0].shape[0]

        if self.use_opt:
            legend = ""
            if self.multi_opt:
                policy_num = self.multi_opt_args["opt_run_times"]
            else:
                policy_num = 1
            if self.multi_opt:
                for run_time in range(int(self.multi_opt_args["opt_run_times"])):
                    if self.opt_args["opt_controller_type"] == "OPT":
                        legend = "OPT"
                    elif self.opt_args["opt_controller_type"] == "MPC":
                        legend = "MPC-" + str(run_time)
                        if (
                                "use_terminal_cost" not in self.opt_args.keys()
                                or self.opt_args["use_terminal_cost"] is False
                        ):
                            legend += " (w/o TC)"
                        else:
                            legend += " (w/ TC)"
                    self.algorithm_list.append(legend)
            else:
                if self.opt_args["opt_controller_type"] == "OPT":
                    legend = "OPT"
                elif self.opt_args["opt_controller_type"] == "MPC":
                    legend = "MPC-" + str(self.opt_args["num_pred_step"])
                    if (
                            "use_terminal_cost" not in self.opt_args.keys()
                            or self.opt_args["use_terminal_cost"] is False
                    ):
                        legend += " (w/o TC)"
                    else:
                        legend += " (w/ TC)"
                self.algorithm_list.append(legend)

        # Create initial list
        reward_list = []
        action_list = []
        state_list = []
        step_list = []
        state_ref_error_list = []
        constrain_list = []
        calctime_list = []
        # Put data into list
        for i in range(policy_num):
            reward_list.append(np.array(self.eval_list[i]["reward_list"]))
            action_list.append(np.array(self.eval_list[i]["action_list"]))
            state_list.append(np.array(self.eval_list[i]["state_list"]))
            step_list.append(np.array(self.eval_list[i]["step_list"]))
            calctime_list.append(np.array(self.eval_list[i]["calctime_list"]))
            if self.constrained_env:
                constrain_list.append(np.stack(self.eval_list[i]["constrain_list"]))
            if self.is_tracking:
                state_ref_error_list.append(self.tracking_list[i])

        if self.plot_range is None:
            pass
        elif len(self.plot_range) == 2:

            for i in range(policy_num):
                start_range = self.plot_range[0]
                end_range = min(self.plot_range[1], reward_list[i].shape[0])

                reward_list[i] = reward_list[i][start_range:end_range]
                action_list[i] = action_list[i][start_range:end_range]
                state_list[i] = state_list[i][start_range:end_range]
                step_list[i] = step_list[i][start_range:end_range]
                if self.constrained_env:
                    constrain_list[i] = constrain_list[i][start_range:end_range]
                if self.is_tracking:
                    for key, value in self.tracking_list[i].items():
                        self.tracking_list[i][key] = value[start_range:end_range]
        else:
            raise NotImplementedError("Figure range is wrong")

        if self.dt is None:
            x_label = "Time step"
        else:
            step_list = [s * self.dt for s in step_list]
            x_label = "Time (s)"

        # Plot reward
        path_reward_fmt = os.path.join(
            self.save_path, "Reward.{}".format(default_cfg["img_fmt"])
        )
        fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

        # save reward data to csv
        reward_data = pd.DataFrame(data=reward_list[0])
        reward_data.to_csv(os.path.join(self.save_path, "Reward.csv"), encoding="gbk")

        for i in range(policy_num):
            legend = (
                self.legend_list[i]
                if len(self.legend_list) == policy_num
                else self.algorithm_list[i]
            )
            sns.lineplot(x=step_list[i], y=reward_list[i], label="{}".format(legend))
        plt.tick_params(labelsize=default_cfg["tick_size"])
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
        plt.xlabel(x_label, default_cfg["label_font"])
        plt.ylabel("Reward", default_cfg["label_font"])
        plt.legend(loc="best", prop=default_cfg["legend_font"])
        fig.tight_layout(pad=default_cfg["pad"])
        plt.savefig(path_reward_fmt, format=default_cfg["img_fmt"], bbox_inches="tight")
        plt.close()

        # plot action
        for j in range(action_dim):
            path_action_fmt = os.path.join(
                self.save_path, "Action-{}.{}".format(j + 1, default_cfg["img_fmt"])
            )
            fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

            # save action data to csv
            action_data = pd.DataFrame(data=[a[:, j] for a in action_list])
            action_data.to_csv(
                os.path.join(self.save_path, "Action-{}.csv".format(j + 1)),
                encoding="gbk",
            )

            for i in range(policy_num):
                legend = (
                    self.legend_list[i]
                    if len(self.legend_list) == policy_num
                    else self.algorithm_list[i]
                )
                sns.lineplot(
                    x=step_list[i], y=action_list[i][:, j], label="{}".format(legend)
                )
            plt.tick_params(labelsize=default_cfg["tick_size"])
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
            plt.xlabel(x_label, default_cfg["label_font"])
            plt.ylabel("Action-{}".format(j + 1), default_cfg["label_font"])
            plt.legend(loc="best", prop=default_cfg["legend_font"])
            fig.tight_layout(pad=default_cfg["pad"])
            plt.savefig(
                path_action_fmt, format=default_cfg["img_fmt"], bbox_inches="tight"
            )
            plt.close()

        # plot state
        for j in range(state_dim):
            path_state_fmt = os.path.join(
                self.save_path, "State-{}.{}".format(j + 1, default_cfg["img_fmt"])
            )
            fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

            # save state data to csv
            state_data = pd.DataFrame(data=[s[:, j] for s in state_list])
            state_data.to_csv(
                os.path.join(self.save_path, "State-{}.csv".format(j + 1)),
                encoding="gbk",
            )

            for i in range(policy_num):
                legend = (
                    self.legend_list[i]
                    if len(self.legend_list) == policy_num
                    else self.algorithm_list[i]
                )
                sns.lineplot(
                    x=step_list[i], y=state_list[i][:, j], label="{}".format(legend)
                )
            plt.tick_params(labelsize=default_cfg["tick_size"])
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
            plt.xlabel(x_label, default_cfg["label_font"])
            plt.ylabel("State-{}".format(j + 1), default_cfg["label_font"])
            plt.legend(loc="best", prop=default_cfg["legend_font"])
            fig.tight_layout(pad=default_cfg["pad"])
            plt.savefig(
                path_state_fmt, format=default_cfg["img_fmt"], bbox_inches="tight"
            )
            plt.close()
        # plot state x-y
        path_state_fmt = os.path.join(
            self.save_path, "State-xy.{}".format(default_cfg["img_fmt"])
        )
        fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

        for i in range(policy_num):
            legend = (
                self.legend_list[i]
                if len(self.legend_list) == policy_num
                else self.algorithm_list[i]
            )
            sns.lineplot(
                x=state_list[i][:, 0], y=state_list[i][:, 1], label="{}".format(legend)
            )
        plt.tick_params(labelsize=default_cfg["tick_size"])
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
        plt.xlabel("x", default_cfg["label_font"])
        plt.ylabel("y", default_cfg["label_font"])
        plt.legend(loc="best", prop=default_cfg["legend_font"])
        fig.tight_layout(pad=default_cfg["pad"])
        plt.savefig(
            path_state_fmt, format=default_cfg["img_fmt"], bbox_inches="tight"
        )
        plt.close()
        # plot tracking
        if self.is_tracking:
            # find index of the longest trajectory
            traj_lens = [len(r) for r in reward_list]
            longest_traj_index = np.argmax(traj_lens)

            for j in range(self.ref_state_num):

                # plot state and ref
                path_tracking_state_fmt = os.path.join(
                    self.save_path, "Ref - State - {}.{}".format(j + 1, default_cfg["img_fmt"])
                )
                fig, ax = plt.subplots(
                    figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"]
                )
                # save tracking state data to csv
                tracking_state_data = []
                for i in range(policy_num):
                    legend = (
                        self.legend_list[i]
                        if len(self.legend_list) == policy_num
                        else self.algorithm_list[i]
                    )
                    sns.lineplot(
                        x=step_list[i],
                        y=state_ref_error_list[i]["state-{}".format(j)],
                        label="{}".format(legend),
                    )
                    tracking_state_data.append(
                        state_ref_error_list[i]["state-{}".format(j)]
                    )
                sns.lineplot(
                    x=step_list[longest_traj_index],
                    y=state_ref_error_list[longest_traj_index]["ref-{}".format(j)],
                    label="ref",
                )
                tracking_state_data.append(state_ref_error_list[longest_traj_index]["ref-{}".format(j)])
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel(x_label, default_cfg["label_font"])
                plt.ylabel("State-{}".format(j + 1), default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(
                    path_tracking_state_fmt,
                    format=default_cfg["img_fmt"],
                    bbox_inches="tight",
                )
                plt.close()

                tracking_state_data = pd.DataFrame(data=tracking_state_data)
                tracking_state_data.to_csv(
                    os.path.join(self.save_path, "State-{}.csv".format(j + 1)),
                    encoding="gbk",
                )

                # plot state-ref error
                path_tracking_error_fmt = os.path.join(
                    self.save_path,
                    "Ref - State - Error{}.{}".format(j + 1, default_cfg["img_fmt"]),
                )
                fig, ax = plt.subplots(
                    figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"]
                )
                # save tracking error data to csv
                tracking_error_data = []
                for i in range(policy_num):
                    legend = (
                        self.legend_list[i]
                        if len(self.legend_list) == policy_num
                        else self.algorithm_list[i]
                    )
                    sns.lineplot(
                        x=step_list[i],
                        y=state_ref_error_list[i]["state-{}-error".format(j)],
                        label="{}".format(legend),
                    )
                    tracking_error_data.append(
                        state_ref_error_list[i]["state-{}-error".format(j)]
                    )
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel(x_label, default_cfg["label_font"])
                plt.ylabel("Ref$-$State-Error{}".format(j + 1), default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(
                    path_tracking_error_fmt,
                    format=default_cfg["img_fmt"],
                    bbox_inches="tight",
                )
                plt.close()

                tracking_error_data = pd.DataFrame(data=tracking_error_data)
                tracking_error_data.to_csv(
                    os.path.join(self.save_path, "Ref-State-Error{}.csv".format(j + 1)),
                    encoding="gbk",
                )

        # plot calculation time
        path_state_fmt = os.path.join(
            self.save_path, "Calc time.{}".format(default_cfg["img_fmt"])
        )
        fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

        # save state data to csv
        state_data = pd.DataFrame(data=[s[:] for s in calctime_list])
        state_data.to_csv(
            os.path.join(self.save_path, "Calc time.csv"),
            encoding="gbk",
        )

        for i in range(policy_num):
            legend = (
                self.legend_list[i]
                if len(self.legend_list) == policy_num
                else self.algorithm_list[i]
            )
            sns.lineplot(
                x=step_list[i], y=calctime_list[i][:], label="{}".format(legend)
            )
        plt.tick_params(labelsize=default_cfg["tick_size"])
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
        plt.xlabel(x_label, default_cfg["label_font"])
        plt.ylabel("Calc Time [ms]", default_cfg["label_font"])
        plt.legend(loc="best", prop=default_cfg["legend_font"])
        fig.tight_layout(pad=default_cfg["pad"])
        plt.savefig(
            path_state_fmt, format=default_cfg["img_fmt"], bbox_inches="tight"
        )
        plt.close()


        # plot constraint value
        if self.constrained_env:
            for j in range(constrain_dim):
                path_constraint_fmt = os.path.join(
                    self.save_path,
                    "Constrain-{}.{}".format(j + 1, default_cfg["img_fmt"]),
                )
                fig, ax = plt.subplots(
                    figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"]
                )

                # save reward data to csv
                constrain_data = pd.DataFrame(data=[c[:, j] for c in constrain_list])
                constrain_data.to_csv(
                    os.path.join(self.save_path, "Constrain-{}.csv".format(j + 1)),
                    encoding="gbk",
                )

                for i in range(policy_num):
                    legend = (
                        self.legend_list[i]
                        if len(self.legend_list) == policy_num
                        else self.algorithm_list[i]
                    )
                    sns.lineplot(
                        x=step_list[i],
                        y=constrain_list[i][:, j],
                        label="{}".format(legend),
                    )
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel(x_label, default_cfg["label_font"])
                plt.ylabel("Constrain-{}".format(j + 1), default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(
                    path_constraint_fmt,
                    format=default_cfg["img_fmt"],
                    bbox_inches="tight",
                )
                plt.close()

        # plot error with opt
        if self.use_opt:
            # reward error
            path_reward_error_fmt = os.path.join(
                self.save_path, "Reward error.{}".format(default_cfg["img_fmt"])
            )
            fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

            # save reward error data to csv
            reward_error_list = []
            for r in reward_list:
                end = min(len(r), len(reward_list[-1]))
                reward_error_list.append(r[:end] - reward_list[-1][:end])
            reward_error_data = pd.DataFrame(data=reward_error_list)
            reward_error_data.to_csv(
                os.path.join(self.save_path, "Reward error.csv"), encoding="gbk"
            )

            for i in range(policy_num - 1):
                legend = (
                    self.legend_list[i]
                    if len(self.legend_list) == policy_num
                    else self.algorithm_list[i]
                )
                sns.lineplot(
                    x=step_list[i][:len(reward_error_list[i])],
                    y=reward_error_list[i], label="{}".format(legend)
                )
            plt.tick_params(labelsize=default_cfg["tick_size"])
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
            plt.xlabel(x_label, default_cfg["label_font"])
            plt.ylabel("Reward error", default_cfg["label_font"])
            plt.legend(loc="best", prop=default_cfg["legend_font"])
            fig.tight_layout(pad=default_cfg["pad"])
            plt.savefig(
                path_reward_error_fmt,
                format=default_cfg["img_fmt"],
                bbox_inches="tight",
            )
            plt.close()

            # action error
            action_error_list = []
            for a in action_list:
                end = min(len(a), len(action_list[-1]))
                action_error_list.append(a[:end] - action_list[-1][:end])
            for j in range(action_dim):
                path_action_error_fmt = os.path.join(
                    self.save_path,
                    "Action-{} error.{}".format(j + 1, default_cfg["img_fmt"]),
                )
                fig, ax = plt.subplots(
                    figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"]
                )
                for i in range(policy_num - 1):
                    legend = (
                        self.legend_list[i]
                        if len(self.legend_list) == policy_num
                        else self.algorithm_list[i]
                    )
                    sns.lineplot(
                        x=step_list[i][:len(action_error_list[i])],
                        y=action_error_list[i][:, j],
                        label="{}".format(legend),
                    )
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel(x_label, default_cfg["label_font"])
                plt.ylabel("Action-{} error".format(j + 1), default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(
                    path_action_error_fmt,
                    format=default_cfg["img_fmt"],
                    bbox_inches="tight",
                )
                plt.close()

                # save action error data to csv
                action_error_data = pd.DataFrame(data=[a[:, j] for a in action_error_list])
                action_error_data.to_csv(
                    os.path.join(self.save_path, "Action-{} error.csv".format(j + 1)),
                    encoding="gbk",
                )

            # state error
            state_error_list = []
            for s in state_list:
                end = min(len(s), len(state_list[-1]))
                state_error_list.append(s[:end] - state_list[-1][:end])
            for j in range(state_dim):
                path_state_error_fmt = os.path.join(
                    self.save_path,
                    "State-{} error.{}".format(j + 1, default_cfg["img_fmt"]),
                )
                fig, ax = plt.subplots(
                    figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"]
                )
                for i in range(policy_num - 1):
                    legend = (
                        self.legend_list[i]
                        if len(self.legend_list) == policy_num
                        else self.algorithm_list[i]
                    )
                    sns.lineplot(
                        x=step_list[i][:len(state_error_list[i])],
                        y=state_error_list[i][:, j],
                        label="{}".format(legend),
                    )
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel(x_label, default_cfg["label_font"])
                plt.ylabel("State-{} error".format(j + 1), default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(
                    path_state_error_fmt,
                    format=default_cfg["img_fmt"],
                    bbox_inches="tight",
                )
                plt.close()

                # save state data to csv
                state_error_data = pd.DataFrame(data=[s[:, j] for s in state_error_list])
                state_error_data.to_csv(
                    os.path.join(self.save_path, "State-{} error.csv".format(j + 1)),
                    encoding="gbk",
                )

            # compute relative error with opt
            error_result = {}
            for i in range(policy_num - 1):
                legend = (
                    self.legend_list[i]
                    if len(self.legend_list) == policy_num
                    else "Policy-{}".format(i + 1)
                )
                end = min(len(action_list[i]), len(action_list[-1]))
                error_result.update({legend: {}})
                # action error
                for j in range(action_dim):
                    action_error = {}
                    error_list = np.abs(
                        action_list[i][:end, j] - action_list[-1][:end, j]
                    ) / (
                                         np.max(action_list[-1][:end, j])
                                         - np.min(action_list[-1][:end, j])
                                 )
                    action_error["Max_error"] = "{:.2f}%".format(max(error_list) * 100)
                    action_error["Mean_error"] = "{:.2f}%".format(
                        sum(error_list) / len(error_list) * 100
                    )
                    error_result[legend].update(
                        {"Action-{}".format(j + 1): action_error}
                    )
                # state error
                for j in range(state_dim):
                    state_error = {}
                    error_list = np.abs(
                        state_list[i][:end, j] - state_list[-1][:end, j]
                    ) / (
                                         np.max(state_list[-1][:end, j])
                                         - np.min(state_list[-1][:end, j])
                                 )
                    state_error["Max_error"] = "{:.2f}%".format(max(error_list) * 100)
                    state_error["Mean_error"] = "{:.2f}%".format(
                        sum(error_list) / len(error_list) * 100
                    )
                    error_result[legend].update({"State-{}".format(j + 1): state_error})

            for i in range(self.policy_num):
                legend = (
                    self.legend_list[i]
                    if len(self.legend_list) == policy_num
                    else "Policy-{}".format(i + 1)
                )
                policy_result = pd.DataFrame(data=error_result[legend])
                policy_result.to_excel(os.path.join(self.save_path, "Error-result.xlsx"), legend)
            error_result_data = pd.DataFrame(data=error_result)
            pd.set_option("display.max_columns", None)
            pd.set_option("display.max_rows", None)
            for key, value in error_result_data.items():
                print("===========================================================")
                print("GOPS: Policy {}".format(key))
                for key, value in value.items():
                    print(key, value)

class CostLearningRunner:
    """Plot module for trained policy

    :param list log_policy_dir_list: directory of trained policy.
    :param list trained_policy_iteration_list: iteration of trained policy.
    :param bool save_render: save environment animation or not.
    :param list plot_range: customize plot range.
    :param bool is_init_info: customize initial information or not.
    :param dict init_info: initial information.
    :param list legend_list: legends of figures.
    :param bool use_opt: use optimal solution for comparison or not.
    :param Optional[str] load_opt_path: path to load optimal controller result.
    :param dict opt_args: arguments of optimal solution solver.
    :param bool save_opt: save optimal controller result or not.
    :param bool constrained_env: constrained environment or not.
    :param bool is_tracking: tracking problem or not.
    :param bool use_dist: use adversarial action or not.
    :param float dt: time interval between steps.
    :param str obs_noise_type: type of observation noise, "normal" or "uniform".
    :param list obs_noise_data: Mean and
        Standard deviation of Normal distribution or Upper
        and Lower bounds of Uniform distribution.
    :param str action_noise_type: type of action noise, "normal" or "uniform".
    :param list action_noise_data: Mean and
        Standard deviation of Normal distribution or Upper
        and Lower bounds of Uniform distribution.
    """

    def __init__(
            self,
            log_policy_dir_list: list,
            env_id:str = None,
            save_render: bool = False,
            plot_range: list = None,
            is_init_info: bool = False,
            init_info: dict = None,
            legend_list: list = None,
            use_opt: bool = False,
            load_opt_path: Optional[str] = None,
            opt_args: Optional[dict] = None,
            save_opt: bool = True,
            multi_opt: bool = False,
            multi_opt_args: Optional[dict] = None,
            constrained_env: bool = False,
            is_tracking: bool = False,
            use_dist: bool = False,
            dt: float = None,
            obs_noise_type: str = None,
            obs_noise_data: list = None,
            action_noise_type: str = None,
            action_noise_data: list = None,
    ):
        self.log_policy_dir_list = [
            os.path.join(gops_path, d) for d in log_policy_dir_list
        ]
        self.save_render = save_render
        self.args = None
        self.plot_range = plot_range
        if is_init_info:
            self.init_info = init_info
        else:
            self.init_info = {}
        self.legend_list = legend_list
        self.use_opt = use_opt
        if use_opt:
            assert load_opt_path is not None or opt_args is not None
            self.load_opt_path = load_opt_path
            self.opt_args = opt_args
            if isinstance(self.opt_args, dict) and \
                    "use_MPC_for_general_env" not in self.opt_args.keys():
                self.opt_args["use_MPC_for_general_env"] = False
            self.save_opt = save_opt
        self.multi_opt = multi_opt
        if multi_opt:
            self.multi_opt_args = multi_opt_args
        self.constrained_env = constrained_env
        self.use_dist = use_dist
        self.is_tracking = is_tracking
        self.dt = dt
        self.policy_num = len(self.log_policy_dir_list)
        self.obs_noise_type = obs_noise_type
        self.obs_noise_data = obs_noise_data
        self.action_noise_type = action_noise_type
        self.action_noise_data = action_noise_data
        self.ref_state_num = 0

        # data for plot
        self.args_list = []
        self.eval_list = []
        self.env_id_list = []
        self.algorithm_list = []
        self.tracking_list = []

        self.__load_all_args()
        self.env_id = self.get_n_verify_env_id()

        # save path
        path = os.path.join(os.path.dirname(__file__), "..", "..", "figures")
        path = os.path.abspath(path)

        algs_name = "MPC-"
        self.save_path = os.path.join(
            path,
            algs_name + self.env_id,
            datetime.datetime.now().strftime("%y%m%d-%H%M%S"),
        )
        os.makedirs(self.save_path, exist_ok=True)


    @staticmethod
    def __load_args(log_policy_dir: str):
        json_path = os.path.join(log_policy_dir, "config.json")
        parser = argparse.ArgumentParser()
        args_dict = vars(parser.parse_args())
        args = get_args_from_json(json_path, args_dict)
        return args

    def __load_all_args(self):
        log_policy_dir = self.log_policy_dir_list[0]
        args = self.__load_args(log_policy_dir)
        args['vector_env_num'] = None
        args['gym2gymnasium'] = False
        self.args_list.append(args)
        env_id = args["env_id"]
        self.env_id_list.append(env_id)
        self.algorithm_list.append(args["algorithm"])

    def __load_env(self, use_opt: bool = False):
        if use_opt:
            env = create_env(**self.args)
        else:
            env_args = {
                **self.args,
                "obs_noise_type": self.obs_noise_type,
                "obs_noise_data": self.obs_noise_data,
                "action_noise_type": self.action_noise_type,
                "action_noise_data": self.action_noise_data,
            }
            env = create_env(**env_args)
        if self.save_render:
            video_path = os.path.join(self.save_path, "videos")
            if use_opt:
                name_prefix = "{}_video".format(self.opt_args["opt_controller_type"])
            else:
                name_prefix = "{}_video".format(self.args["algorithm"])
            env = wrappers.RecordVideo(env, video_path, name_prefix=name_prefix)
        # self.args["action_high_limit"] = self.args['action_high_limit']#env.action_space.high
        # self.args["action_low_limit"] = env.action_space.low
        return env

    def __load_policy(self, log_policy_dir: str, trained_policy_iteration: str):
        # Create policy
        networks = create_approx_contrainer(**self.args)

        # Load trained policy
        log_path = log_policy_dir + "/apprfunc/apprfunc_{}.pkl".format(
            trained_policy_iteration
        )
        networks.load_state_dict(torch.load(log_path))
        return networks

    def __convert_format(self, origin_data_list: list):
        data_list = copy(origin_data_list)
        for i in range(len(origin_data_list)):
            if isinstance(origin_data_list[i], list) or isinstance(
                    origin_data_list[i], np.ndarray
            ):
                data_list[i] = self.__convert_format(origin_data_list[i])
            else:
                data_list[i] = "{:.2g}".format(origin_data_list[i])
        return data_list

    def __action_noise(self, action: np.ndarray) -> np.ndarray:
        if self.action_noise_type is None:
            return action
        elif self.action_noise_type == "normal":
            return action + np.random.normal(
                loc=self.action_noise_data[0], scale=self.action_noise_data[1]
            )
        elif self.action_noise_type == "uniform":
            return action + np.random.uniform(
                low=self.action_noise_data[0], high=self.action_noise_data[1]
            )

    def __save_mp4_as_gif(self):
        if self.save_render:
            videos_path = os.path.join(self.save_path, "videos")

            videos_list = [i for i in glob.glob(os.path.join(videos_path, "*.mp4"))]
            for v in videos_list:
                mp4togif(v)

    def get_n_verify_env_id(self):
        env_id = self.env_id_list[0]
        for i, eid in enumerate(self.env_id_list):
            assert (
                    env_id == eid
            ), "GOPS: policy {} is not trained in the same environment".format(i)
        return env_id

    def run(self):
        self.__run_data()
        self.__save_mp4_as_gif()
        self.draw()

    def __run_data(self):
        if self.use_opt:
            if self.multi_opt:
                for run_time in range(int(self.multi_opt_args["opt_run_times"])):
                    if self.load_opt_path is not None:
                        eval_dict_opt = np.load(
                            os.path.join(self.load_opt_path, "eval_dict_opt.npy"),
                            allow_pickle=True).item()
                        tracking_dict_opt = np.load(
                            os.path.join(self.load_opt_path, "tracking_dict_opt.npy"),
                            allow_pickle=True).item()
                        print("Successfully load an optimal controller result!")
                        print("===========================================================\n")
                    else:
                        self.args = self.args_list[0]
                        print("GOPS: Use an optimal controller")
                        env = self.__load_env(use_opt=True)
                        print("The environment for opt")
                        if hasattr(env, "set_mode"):
                            env.set_mode("test")

                        assert (
                                self.opt_args is not None
                        ), "Choose to use optimal controller, but the opt_args is None."

                        if self.opt_args["opt_controller_type"] == "OPT":
                            assert (
                                env.has_optimal_controller
                            ), "The environment has no theoretical optimal controller."
                            opt_controller = env.control_policy
                        elif self.opt_args["opt_controller_type"] == "MPC":
                            if self.opt_args["use_MPC_for_general_env"] == True:
                                self.args_list[0]["env"] = env
                                from gops.sys_simulator.opt_controller_for_gen_env import OptController
                            else:
                                from gops.sys_simulator.opt_controller import OptController
                            model = create_env_model(**self.args_list[0], mask_at_done=False)
                            model.update_cost_paras(self.multi_opt_args["cost_paras_list"][run_time])
                            print("The cost_paras for opt:", self.multi_opt_args["cost_paras_list"][run_time])
                            opt_args = self.opt_args.copy()
                            opt_args.pop("opt_controller_type")
                            opt_args.pop("use_MPC_for_general_env")
                            opt_controller = OptController(model, **opt_args, )
                        else:
                            raise ValueError(
                                "The optimal controller type should be either 'OPT' or 'MPC'."
                            )

                        eval_dict_opt, tracking_dict_opt = self.run_an_episode(
                            env, opt_controller, self.init_info, is_opt=True, render=False
                        )
                        print("Successfully run an optimal controller!")
                        print("===========================================================\n")

                    if self.opt_args["opt_controller_type"] == "OPT":
                        legend = "OPT"
                    elif self.opt_args["opt_controller_type"] == "MPC":
                        if self.multi_opt:
                            legend = "MPC-" + str(run_time)
                            if (
                                    "use_terminal_cost" not in self.opt_args.keys()
                                    or self.opt_args["use_terminal_cost"] == False
                            ):
                                legend += " (w/o TC)"
                            else:
                                legend += " (w/ TC)"
                        else:
                            legend = "MPC-" + str(self.opt_args["num_pred_step"])
                            if (
                                    "use_terminal_cost" not in self.opt_args.keys()
                                    or self.opt_args["use_terminal_cost"] == False
                            ):
                                legend += " (w/o TC)"
                            else:
                                legend += " (w/ TC)"
                    self.legend_list.append(legend)

                    if self.save_opt:
                        np.save(os.path.join(self.save_path, "eval_dict_opt.npy"), eval_dict_opt)
                        np.save(os.path.join(self.save_path, "tracking_dict_opt.npy"), tracking_dict_opt)

                    self.eval_list.append(eval_dict_opt)
                    if self.is_tracking:
                        self.tracking_list.append(tracking_dict_opt)
                if self.opt_args["opt_controller_type"] == "OPT":
                    legend = "OPT"
                elif self.opt_args["opt_controller_type"] == "MPC":
                    legend = "MPC-" + str(self.opt_args["num_pred_step"])
                    if (
                            "use_terminal_cost" not in self.opt_args.keys()
                            or self.opt_args["use_terminal_cost"] == False
                    ):
                        legend += " (w/o TC)"
                    else:
                        legend += " (w/ TC)"
                self.legend_list.append(legend)

                if self.save_opt:
                    np.save(os.path.join(self.save_path, "eval_dict_opt.npy"), eval_dict_opt)
                    np.save(os.path.join(self.save_path, "tracking_dict_opt.npy"), tracking_dict_opt)

                self.eval_list.append(eval_dict_opt)
                if self.is_tracking:
                    self.tracking_list.append(tracking_dict_opt)
            else:
                if self.load_opt_path is not None:
                    eval_dict_opt = np.load(
                        os.path.join(self.load_opt_path, "eval_dict_opt.npy"),
                        allow_pickle=True).item()
                    tracking_dict_opt = np.load(
                        os.path.join(self.load_opt_path, "tracking_dict_opt.npy"),
                        allow_pickle=True).item()
                    print("Successfully load an optimal controller result!")
                    print("===========================================================\n")
                else:
                    self.args = self.args_list[0]
                    print("GOPS: Use an optimal controller")
                    env = self.__load_env(use_opt=True)
                    print("The environment for opt")
                    if hasattr(env, "set_mode"):
                        env.set_mode("test")

                    assert (
                            self.opt_args is not None
                    ), "Choose to use optimal controller, but the opt_args is None."

                    if self.opt_args["opt_controller_type"] == "OPT":
                        assert (
                            env.has_optimal_controller
                        ), "The environment has no theoretical optimal controller."
                        opt_controller = env.control_policy
                    elif self.opt_args["opt_controller_type"] == "MPC":
                        if self.opt_args["use_MPC_for_general_env"] == True:
                            self.args_list[0]["env"] = env
                            from gops.sys_simulator.opt_controller_for_gen_env import OptController
                        else:
                            from gops.sys_simulator.opt_controller import OptController
                        model = create_env_model(**self.args_list[0], mask_at_done=False)
                        opt_args = self.opt_args.copy()
                        opt_args.pop("opt_controller_type")
                        opt_args.pop("use_MPC_for_general_env")
                        opt_controller = OptController(model, **opt_args, )
                    else:
                        raise ValueError(
                            "The optimal controller type should be either 'OPT' or 'MPC'."
                        )

                    eval_dict_opt, tracking_dict_opt = self.run_an_episode(
                        env, opt_controller, self.init_info, is_opt=True, render=False
                    )
                    print("Successfully run an optimal controller!")
                    print("===========================================================\n")

                if self.opt_args["opt_controller_type"] == "OPT":
                    legend = "OPT"
                elif self.opt_args["opt_controller_type"] == "MPC":
                    legend = "MPC-" + str(self.opt_args["num_pred_step"])
                    if (
                            "use_terminal_cost" not in self.opt_args.keys()
                            or self.opt_args["use_terminal_cost"] == False
                    ):
                        legend += " (w/o TC)"
                    else:
                        legend += " (w/ TC)"
                self.legend_list.append(legend)

                if self.save_opt:
                    np.save(os.path.join(self.save_path, "eval_dict_opt.npy"), eval_dict_opt)
                    np.save(os.path.join(self.save_path, "tracking_dict_opt.npy"), tracking_dict_opt)

                self.eval_list.append(eval_dict_opt)
                if self.is_tracking:
                    self.tracking_list.append(tracking_dict_opt)

    def run_an_episode(
            self,
            env: Any,
            controller: Any,
            init_info: dict,
            is_opt: bool,
            render: bool = True,
    ) -> Tuple[dict, dict]:
        state_list = []
        action_list = []
        reward_list = []
        constrain_list = []
        obs_list = []
        step = 0
        step_list = []
        info_list = [init_info]
        obs, info = env.reset(**init_info)
        state = env.state
        print("Initial robot state: ")
        print(self.__convert_format(np.asarray(state.robot_state)))
        # plot tracking
        state_with_ref_error = {}
        done = False
        info.update({"TimeLimit.truncated": False})
        while not (done or info["TimeLimit.truncated"]):
            print("step:", step + 1)
            state_list.append(state.robot_state)
            obs_list.append(obs)
            if is_opt:
                if isinstance(env.unwrapped, Env):
                    action = controller(state)
                else:
                    action = controller(obs, info)
            else:
                action = self.compute_action(obs, controller)
                action = self.__action_noise(action)
            if self.use_dist:
                action = np.hstack((action, env.dist_func(step * env.tau)))
            if self.constrained_env:
                constrain_list.append(info["constraint"])
            if self.is_tracking:
                reference = get_reference_from_info(info)
                state_num = len(reference)
                self.ref_state_num = sum(x is not None for x in reference)
                if step == 0:
                    for i in range(state_num):
                        if reference[i] is not None:
                            state_with_ref_error["state-{}".format(i)] = []
                            state_with_ref_error["ref-{}".format(i)] = []
                            state_with_ref_error["state-{}-error".format(i)] = []

                robot_state = get_robot_state_from_info(info)
                for i in range(state_num):
                    if reference[i] is not None:
                        state_with_ref_error["state-{}".format(i)].append(robot_state[i])
                        state_with_ref_error["ref-{}".format(i)].append(reference[i])
                        state_with_ref_error["state-{}-error".format(i)].append(
                            reference[i] - robot_state[i]
                        )
            next_obs, reward, done, info = env.step(action)
            # save the real action (without scaling)
            # todo 若做model based planning, 需要把action的第0个存到action_list中
            action_list.append(info.get("raw_action", action[0, :]))#
            step_list.append(step)
            reward_list.append(reward)
            info_list.append(info)

            obs = next_obs
            state = env.state
            step = step + 1

            if "TimeLimit.truncated" not in info.keys():
                info["TimeLimit.truncated"] = False
            # Draw environment animation
            if render:
                env.render()

        eval_dict = {
            "reward_list": reward_list,
            "action_list": action_list,
            "state_list": state_list,
            "step_list": step_list,
            "obs_list": obs_list,
            "info_list": info_list,
        }
        if self.constrained_env:
            eval_dict.update(
                {"constrain_list": constrain_list, }
            )

        if self.is_tracking:
            tracking_dict = state_with_ref_error
        else:
            tracking_dict = {}

        return eval_dict, tracking_dict

    def draw(self):
        fig_size = (
            default_cfg["fig_size"],
            default_cfg["fig_size"],
        )
        action_dim = self.eval_list[0]["action_list"][0].shape[0]
        state_dim = self.eval_list[0]["state_list"][0].shape[0]
        if self.constrained_env:
            constrain_dim = self.eval_list[0]["constrain_list"][0].shape[0]

        if self.use_opt:
            legend = ""
            if self.multi_opt:
                policy_num = self.multi_opt_args["opt_run_times"]
            else:
                policy_num = 1
            if self.multi_opt:
                for run_time in range(int(self.multi_opt_args["opt_run_times"])):
                    if self.opt_args["opt_controller_type"] == "OPT":
                        legend = "OPT"
                    elif self.opt_args["opt_controller_type"] == "MPC":
                        legend = "MPC-" + str(run_time)
                        if (
                                "use_terminal_cost" not in self.opt_args.keys()
                                or self.opt_args["use_terminal_cost"] is False
                        ):
                            legend += " (w/o TC)"
                        else:
                            legend += " (w/ TC)"
                    self.algorithm_list.append(legend)
            else:
                if self.opt_args["opt_controller_type"] == "OPT":
                    legend = "OPT"
                elif self.opt_args["opt_controller_type"] == "MPC":
                    legend = "MPC-" + str(self.opt_args["num_pred_step"])
                    if (
                            "use_terminal_cost" not in self.opt_args.keys()
                            or self.opt_args["use_terminal_cost"] is False
                    ):
                        legend += " (w/o TC)"
                    else:
                        legend += " (w/ TC)"
                self.algorithm_list.append(legend)

        # Create initial list
        reward_list = []
        action_list = []
        state_list = []
        step_list = []
        state_ref_error_list = []
        constrain_list = []
        # Put data into list
        for i in range(policy_num):
            reward_list.append(np.array(self.eval_list[i]["reward_list"]))
            action_list.append(np.array(self.eval_list[i]["action_list"]))
            state_list.append(np.array(self.eval_list[i]["state_list"]))
            step_list.append(np.array(self.eval_list[i]["step_list"]))
            if self.constrained_env:
                constrain_list.append(np.stack(self.eval_list[i]["constrain_list"]))
            if self.is_tracking:
                state_ref_error_list.append(self.tracking_list[i])

        if self.plot_range is None:
            pass
        elif len(self.plot_range) == 2:

            for i in range(policy_num):
                start_range = self.plot_range[0]
                end_range = min(self.plot_range[1], reward_list[i].shape[0])

                reward_list[i] = reward_list[i][start_range:end_range]
                action_list[i] = action_list[i][start_range:end_range]
                state_list[i] = state_list[i][start_range:end_range]
                step_list[i] = step_list[i][start_range:end_range]
                if self.constrained_env:
                    constrain_list[i] = constrain_list[i][start_range:end_range]
                if self.is_tracking:
                    for key, value in self.tracking_list[i].items():
                        self.tracking_list[i][key] = value[start_range:end_range]
        else:
            raise NotImplementedError("Figure range is wrong")

        if self.dt is None:
            x_label = "Time step"
        else:
            step_list = [s * self.dt for s in step_list]
            x_label = "Time (s)"

        # Plot reward
        path_reward_fmt = os.path.join(
            self.save_path, "Reward.{}".format(default_cfg["img_fmt"])
        )
        fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

        # save reward data to csv
        reward_data = pd.DataFrame(data=reward_list[0])
        reward_data.to_csv(os.path.join(self.save_path, "Reward.csv"), encoding="gbk")

        for i in range(policy_num):
            legend = (
                self.legend_list[i]
                if len(self.legend_list) == policy_num
                else self.algorithm_list[i]
            )
            sns.lineplot(x=step_list[i], y=reward_list[i], label="{}".format(legend))
        plt.tick_params(labelsize=default_cfg["tick_size"])
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
        plt.xlabel(x_label, default_cfg["label_font"])
        plt.ylabel("Reward", default_cfg["label_font"])
        plt.legend(loc="best", prop=default_cfg["legend_font"])
        fig.tight_layout(pad=default_cfg["pad"])
        plt.savefig(path_reward_fmt, format=default_cfg["img_fmt"], bbox_inches="tight")
        plt.close()

        # plot action
        for j in range(action_dim):
            path_action_fmt = os.path.join(
                self.save_path, "Action-{}.{}".format(j + 1, default_cfg["img_fmt"])
            )
            fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

            # save action data to csv
            action_data = pd.DataFrame(data=[a[:, j] for a in action_list])
            action_data.to_csv(
                os.path.join(self.save_path, "Action-{}.csv".format(j + 1)),
                encoding="gbk",
            )

            for i in range(policy_num):
                legend = (
                    self.legend_list[i]
                    if len(self.legend_list) == policy_num
                    else self.algorithm_list[i]
                )
                sns.lineplot(
                    x=step_list[i], y=action_list[i][:, j], label="{}".format(legend)
                )
            plt.tick_params(labelsize=default_cfg["tick_size"])
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
            plt.xlabel(x_label, default_cfg["label_font"])
            plt.ylabel("Action-{}".format(j + 1), default_cfg["label_font"])
            plt.legend(loc="best", prop=default_cfg["legend_font"])
            fig.tight_layout(pad=default_cfg["pad"])
            plt.savefig(
                path_action_fmt, format=default_cfg["img_fmt"], bbox_inches="tight"
            )
            plt.close()

        # plot state
        for j in range(state_dim):
            path_state_fmt = os.path.join(
                self.save_path, "State-{}.{}".format(j + 1, default_cfg["img_fmt"])
            )
            fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

            # save state data to csv
            state_data = pd.DataFrame(data=[s[:, j] for s in state_list])
            state_data.to_csv(
                os.path.join(self.save_path, "State-{}.csv".format(j + 1)),
                encoding="gbk",
            )

            for i in range(policy_num):
                legend = (
                    self.legend_list[i]
                    if len(self.legend_list) == policy_num
                    else self.algorithm_list[i]
                )
                sns.lineplot(
                    x=step_list[i], y=state_list[i][:, j], label="{}".format(legend)
                )
            plt.tick_params(labelsize=default_cfg["tick_size"])
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
            plt.xlabel(x_label, default_cfg["label_font"])
            plt.ylabel("State-{}".format(j + 1), default_cfg["label_font"])
            plt.legend(loc="best", prop=default_cfg["legend_font"])
            fig.tight_layout(pad=default_cfg["pad"])
            plt.savefig(
                path_state_fmt, format=default_cfg["img_fmt"], bbox_inches="tight"
            )
            plt.close()
        # plot state x-y
        path_state_fmt = os.path.join(
            self.save_path, "State-xy.{}".format(default_cfg["img_fmt"])
        )
        fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

        for i in range(policy_num):
            legend = (
                self.legend_list[i]
                if len(self.legend_list) == policy_num
                else self.algorithm_list[i]
            )
            sns.lineplot(
                x=state_list[i][:, 0], y=state_list[i][:, 1], label="{}".format(legend)
            )
        plt.tick_params(labelsize=default_cfg["tick_size"])
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
        plt.xlabel("x", default_cfg["label_font"])
        plt.ylabel("y", default_cfg["label_font"])
        plt.legend(loc="best", prop=default_cfg["legend_font"])
        fig.tight_layout(pad=default_cfg["pad"])
        plt.savefig(
            path_state_fmt, format=default_cfg["img_fmt"], bbox_inches="tight"
        )
        plt.close()
        # plot tracking
        if self.is_tracking:
            # find index of the longest trajectory
            traj_lens = [len(r) for r in reward_list]
            longest_traj_index = np.argmax(traj_lens)

            for j in range(self.ref_state_num):

                # plot state and ref
                path_tracking_state_fmt = os.path.join(
                    self.save_path, "Ref - State - {}.{}".format(j + 1, default_cfg["img_fmt"])
                )
                fig, ax = plt.subplots(
                    figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"]
                )
                # save tracking state data to csv
                tracking_state_data = []
                for i in range(policy_num):
                    legend = (
                        self.legend_list[i]
                        if len(self.legend_list) == policy_num
                        else self.algorithm_list[i]
                    )
                    sns.lineplot(
                        x=step_list[i],
                        y=state_ref_error_list[i]["state-{}".format(j)],
                        label="{}".format(legend),
                    )
                    tracking_state_data.append(
                        state_ref_error_list[i]["state-{}".format(j)]
                    )
                sns.lineplot(
                    x=step_list[longest_traj_index],
                    y=state_ref_error_list[longest_traj_index]["ref-{}".format(j)],
                    label="ref",
                )
                tracking_state_data.append(state_ref_error_list[longest_traj_index]["ref-{}".format(j)])
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel(x_label, default_cfg["label_font"])
                plt.ylabel("State-{}".format(j + 1), default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(
                    path_tracking_state_fmt,
                    format=default_cfg["img_fmt"],
                    bbox_inches="tight",
                )
                plt.close()

                tracking_state_data = pd.DataFrame(data=tracking_state_data)
                tracking_state_data.to_csv(
                    os.path.join(self.save_path, "State-{}.csv".format(j + 1)),
                    encoding="gbk",
                )

                # plot state-ref error
                path_tracking_error_fmt = os.path.join(
                    self.save_path,
                    "Ref - State - Error{}.{}".format(j + 1, default_cfg["img_fmt"]),
                )
                fig, ax = plt.subplots(
                    figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"]
                )
                # save tracking error data to csv
                tracking_error_data = []
                for i in range(policy_num):
                    legend = (
                        self.legend_list[i]
                        if len(self.legend_list) == policy_num
                        else self.algorithm_list[i]
                    )
                    sns.lineplot(
                        x=step_list[i],
                        y=state_ref_error_list[i]["state-{}-error".format(j)],
                        label="{}".format(legend),
                    )
                    tracking_error_data.append(
                        state_ref_error_list[i]["state-{}-error".format(j)]
                    )
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel(x_label, default_cfg["label_font"])
                plt.ylabel("Ref$-$State-Error{}".format(j + 1), default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(
                    path_tracking_error_fmt,
                    format=default_cfg["img_fmt"],
                    bbox_inches="tight",
                )
                plt.close()

                tracking_error_data = pd.DataFrame(data=tracking_error_data)
                tracking_error_data.to_csv(
                    os.path.join(self.save_path, "Ref-State-Error{}.csv".format(j + 1)),
                    encoding="gbk",
                )



        # plot constraint value
        if self.constrained_env:
            for j in range(constrain_dim):
                path_constraint_fmt = os.path.join(
                    self.save_path,
                    "Constrain-{}.{}".format(j + 1, default_cfg["img_fmt"]),
                )
                fig, ax = plt.subplots(
                    figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"]
                )

                # save reward data to csv
                constrain_data = pd.DataFrame(data=[c[:, j] for c in constrain_list])
                constrain_data.to_csv(
                    os.path.join(self.save_path, "Constrain-{}.csv".format(j + 1)),
                    encoding="gbk",
                )

                for i in range(policy_num):
                    legend = (
                        self.legend_list[i]
                        if len(self.legend_list) == policy_num
                        else self.algorithm_list[i]
                    )
                    sns.lineplot(
                        x=step_list[i],
                        y=constrain_list[i][:, j],
                        label="{}".format(legend),
                    )
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel(x_label, default_cfg["label_font"])
                plt.ylabel("Constrain-{}".format(j + 1), default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(
                    path_constraint_fmt,
                    format=default_cfg["img_fmt"],
                    bbox_inches="tight",
                )
                plt.close()

        # plot error with opt
        if self.use_opt:
            # reward error
            path_reward_error_fmt = os.path.join(
                self.save_path, "Reward error.{}".format(default_cfg["img_fmt"])
            )
            fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

            # save reward error data to csv
            reward_error_list = []
            for r in reward_list:
                end = min(len(r), len(reward_list[-1]))
                reward_error_list.append(r[:end] - reward_list[-1][:end])
            reward_error_data = pd.DataFrame(data=reward_error_list)
            reward_error_data.to_csv(
                os.path.join(self.save_path, "Reward error.csv"), encoding="gbk"
            )

            for i in range(policy_num - 1):
                legend = (
                    self.legend_list[i]
                    if len(self.legend_list) == policy_num
                    else self.algorithm_list[i]
                )
                sns.lineplot(
                    x=step_list[i][:len(reward_error_list[i])],
                    y=reward_error_list[i], label="{}".format(legend)
                )
            plt.tick_params(labelsize=default_cfg["tick_size"])
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
            plt.xlabel(x_label, default_cfg["label_font"])
            plt.ylabel("Reward error", default_cfg["label_font"])
            plt.legend(loc="best", prop=default_cfg["legend_font"])
            fig.tight_layout(pad=default_cfg["pad"])
            plt.savefig(
                path_reward_error_fmt,
                format=default_cfg["img_fmt"],
                bbox_inches="tight",
            )
            plt.close()

            # action error
            action_error_list = []
            for a in action_list:
                end = min(len(a), len(action_list[-1]))
                action_error_list.append(a[:end] - action_list[-1][:end])
            for j in range(action_dim):
                path_action_error_fmt = os.path.join(
                    self.save_path,
                    "Action-{} error.{}".format(j + 1, default_cfg["img_fmt"]),
                )
                fig, ax = plt.subplots(
                    figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"]
                )
                for i in range(policy_num - 1):
                    legend = (
                        self.legend_list[i]
                        if len(self.legend_list) == policy_num
                        else self.algorithm_list[i]
                    )
                    sns.lineplot(
                        x=step_list[i][:len(action_error_list[i])],
                        y=action_error_list[i][:, j],
                        label="{}".format(legend),
                    )
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel(x_label, default_cfg["label_font"])
                plt.ylabel("Action-{} error".format(j + 1), default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(
                    path_action_error_fmt,
                    format=default_cfg["img_fmt"],
                    bbox_inches="tight",
                )
                plt.close()

                # save action error data to csv
                action_error_data = pd.DataFrame(data=[a[:, j] for a in action_error_list])
                action_error_data.to_csv(
                    os.path.join(self.save_path, "Action-{} error.csv".format(j + 1)),
                    encoding="gbk",
                )

            # state error
            state_error_list = []
            for s in state_list:
                end = min(len(s), len(state_list[-1]))
                state_error_list.append(s[:end] - state_list[-1][:end])
            for j in range(state_dim):
                path_state_error_fmt = os.path.join(
                    self.save_path,
                    "State-{} error.{}".format(j + 1, default_cfg["img_fmt"]),
                )
                fig, ax = plt.subplots(
                    figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"]
                )
                for i in range(policy_num - 1):
                    legend = (
                        self.legend_list[i]
                        if len(self.legend_list) == policy_num
                        else self.algorithm_list[i]
                    )
                    sns.lineplot(
                        x=step_list[i][:len(state_error_list[i])],
                        y=state_error_list[i][:, j],
                        label="{}".format(legend),
                    )
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel(x_label, default_cfg["label_font"])
                plt.ylabel("State-{} error".format(j + 1), default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(
                    path_state_error_fmt,
                    format=default_cfg["img_fmt"],
                    bbox_inches="tight",
                )
                plt.close()

                # save state data to csv
                state_error_data = pd.DataFrame(data=[s[:, j] for s in state_error_list])
                state_error_data.to_csv(
                    os.path.join(self.save_path, "State-{} error.csv".format(j + 1)),
                    encoding="gbk",
                )

            # compute relative error with opt
            error_result = {}
            for i in range(policy_num - 1):
                legend = (
                    self.legend_list[i]
                    if len(self.legend_list) == policy_num
                    else "Policy-{}".format(i + 1)
                )
                end = min(len(action_list[i]), len(action_list[-1]))
                error_result.update({legend: {}})
                # action error
                for j in range(action_dim):
                    action_error = {}
                    error_list = np.abs(
                        action_list[i][:end, j] - action_list[-1][:end, j]
                    ) / (
                                         np.max(action_list[-1][:end, j])
                                         - np.min(action_list[-1][:end, j])
                                 )
                    action_error["Max_error"] = "{:.2f}%".format(max(error_list) * 100)
                    action_error["Mean_error"] = "{:.2f}%".format(
                        sum(error_list) / len(error_list) * 100
                    )
                    error_result[legend].update(
                        {"Action-{}".format(j + 1): action_error}
                    )
                # state error
                for j in range(state_dim):
                    state_error = {}
                    error_list = np.abs(
                        state_list[i][:end, j] - state_list[-1][:end, j]
                    ) / (
                                         np.max(state_list[-1][:end, j])
                                         - np.min(state_list[-1][:end, j])
                                 )
                    state_error["Max_error"] = "{:.2f}%".format(max(error_list) * 100)
                    state_error["Mean_error"] = "{:.2f}%".format(
                        sum(error_list) / len(error_list) * 100
                    )
                    error_result[legend].update({"State-{}".format(j + 1): state_error})

            for i in range(self.policy_num):
                legend = (
                    self.legend_list[i]
                    if len(self.legend_list) == policy_num
                    else "Policy-{}".format(i + 1)
                )
                policy_result = pd.DataFrame(data=error_result[legend])
                policy_result.to_excel(os.path.join(self.save_path, "Error-result.xlsx"), legend)
            error_result_data = pd.DataFrame(data=error_result)
            pd.set_option("display.max_columns", None)
            pd.set_option("display.max_rows", None)
            for key, value in error_result_data.items():
                print("===========================================================")
                print("GOPS: Policy {}".format(key))
                for key, value in value.items():
                    print(key, value)

class OptRunner_CoSimulation:
    """Plot module for trained policy

    :param list log_policy_dir_list: directory of trained policy.
    :param list trained_policy_iteration_list: iteration of trained policy.
    :param bool save_render: save environment animation or not.
    :param list plot_range: customize plot range.
    :param bool is_init_info: customize initial information or not.
    :param dict init_info: initial information.
    :param list legend_list: legends of figures.
    :param bool use_opt: use optimal solution for comparison or not.
    :param Optional[str] load_opt_path: path to load optimal controller result.
    :param dict opt_args: arguments of optimal solution solver.
    :param bool save_opt: save optimal controller result or not.
    :param bool constrained_env: constrained environment or not.
    :param bool is_tracking: tracking problem or not.
    :param bool use_dist: use adversarial action or not.
    :param float dt: time interval between steps.
    :param str obs_noise_type: type of observation noise, "normal" or "uniform".
    :param list obs_noise_data: Mean and
        Standard deviation of Normal distribution or Upper
        and Lower bounds of Uniform distribution.
    :param str action_noise_type: type of action noise, "normal" or "uniform".
    :param list action_noise_data: Mean and
        Standard deviation of Normal distribution or Upper
        and Lower bounds of Uniform distribution.
    """

    def __init__(
            self,
            log_policy_dir_list: list,
            env_id: str = None,
            save_render: bool = False,
            plot_range: list = None,
            is_init_info: bool = False,
            init_info: dict = None,
            legend_list: list = None,
            use_opt: bool = False,
            load_opt_path: Optional[str] = None,
            opt_args: Optional[dict] = None,
            save_opt: bool = True,
            multi_opt: bool = False,
            multi_opt_args: Optional[dict] = None,
            constrained_env: bool = False,
            is_tracking: bool = False,
            use_dist: bool = False,
            dt: float = None,
            obs_noise_type: str = None,
            obs_noise_data: list = None,
            action_noise_type: str = None,
            action_noise_data: list = None,
    ):
        self.log_policy_dir_list = [
            os.path.join(gops_path, d) for d in log_policy_dir_list
        ]
        self.save_render = save_render
        self.args = None
        self.plot_range = plot_range
        if is_init_info:
            self.init_info = init_info
        else:
            self.init_info = {}
        self.legend_list = legend_list
        self.use_opt = use_opt
        if use_opt:
            assert load_opt_path is not None or opt_args is not None
            self.load_opt_path = load_opt_path
            self.opt_args = opt_args
            if isinstance(self.opt_args, dict) and \
                    "use_MPC_for_general_env" not in self.opt_args.keys():
                self.opt_args["use_MPC_for_general_env"] = False
            self.save_opt = save_opt
        self.multi_opt = multi_opt
        if multi_opt:
            self.multi_opt_args = multi_opt_args
        self.constrained_env = constrained_env
        self.use_dist = use_dist
        self.is_tracking = is_tracking
        self.dt = dt
        self.policy_num = len(self.log_policy_dir_list)
        self.obs_noise_type = obs_noise_type
        self.obs_noise_data = obs_noise_data
        self.action_noise_type = action_noise_type
        self.action_noise_data = action_noise_data
        self.ref_state_num = 0

        # data for plot
        self.args_list = []
        self.eval_list = []
        self.env_id_list = []
        self.algorithm_list = []
        self.tracking_list = []

        self.__load_all_args()
        self.env_id = self.get_n_verify_env_id()

        # save path
        path = os.path.join(os.path.dirname(__file__), "..", "..", "figures")
        path = os.path.abspath(path)

        algs_name = "MPC-"
        self.save_path = os.path.join(
            path,
            algs_name + self.env_id,
            datetime.datetime.now().strftime("%y%m%d-%H%M%S")+"_carsim",
        )
        os.makedirs(self.save_path, exist_ok=True)

    @staticmethod
    def __load_args(log_policy_dir: str):
        json_path = os.path.join(log_policy_dir, "config.json")
        parser = argparse.ArgumentParser()
        args_dict = vars(parser.parse_args())
        args = get_args_from_json(json_path, args_dict)
        return args

    def __load_all_args(self):
        log_policy_dir = self.log_policy_dir_list[0]
        args = self.__load_args(log_policy_dir)
        args['vector_env_num'] = None
        args['gym2gymnasium'] = False
        self.args_list.append(args)
        env_id = args["env_id"]
        self.env_id_list.append(env_id)
        self.algorithm_list.append(args["algorithm"])

    def __load_env(self, use_opt: bool = False):
        if use_opt:
            env = create_env(**self.args)
        else:
            env_args = {
                **self.args,
                "obs_noise_type": self.obs_noise_type,
                "obs_noise_data": self.obs_noise_data,
                "action_noise_type": self.action_noise_type,
                "action_noise_data": self.action_noise_data,
            }
            env = create_env(**env_args)
        if self.save_render:
            video_path = os.path.join(self.save_path, "videos")
            if use_opt:
                name_prefix = "{}_video".format(self.opt_args["opt_controller_type"])
            else:
                name_prefix = "{}_video".format(self.args["algorithm"])
            env = wrappers.RecordVideo(env, video_path, name_prefix=name_prefix)
        # self.args["action_high_limit"] = self.args['action_high_limit']#env.action_space.high
        # self.args["action_low_limit"] = env.action_space.low
        return env

    def __load_policy(self, log_policy_dir: str, trained_policy_iteration: str):
        # Create policy
        networks = create_approx_contrainer(**self.args)

        # Load trained policy
        log_path = log_policy_dir + "/apprfunc/apprfunc_{}.pkl".format(
            trained_policy_iteration
        )
        networks.load_state_dict(torch.load(log_path))
        return networks

    def __convert_format(self, origin_data_list: list):
        data_list = copy(origin_data_list)
        for i in range(len(origin_data_list)):
            if isinstance(origin_data_list[i], list) or isinstance(
                    origin_data_list[i], np.ndarray
            ):
                data_list[i] = self.__convert_format(origin_data_list[i])
            else:
                data_list[i] = "{:.2g}".format(origin_data_list[i])
        return data_list

    def __action_noise(self, action: np.ndarray) -> np.ndarray:
        if self.action_noise_type is None:
            return action
        elif self.action_noise_type == "normal":
            return action + np.random.normal(
                loc=self.action_noise_data[0], scale=self.action_noise_data[1]
            )
        elif self.action_noise_type == "uniform":
            return action + np.random.uniform(
                low=self.action_noise_data[0], high=self.action_noise_data[1]
            )

    def __save_mp4_as_gif(self):
        if self.save_render:
            videos_path = os.path.join(self.save_path, "videos")

            videos_list = [i for i in glob.glob(os.path.join(videos_path, "*.mp4"))]
            for v in videos_list:
                mp4togif(v)

    def get_n_verify_env_id(self):
        env_id = self.env_id_list[0]
        for i, eid in enumerate(self.env_id_list):
            assert (
                    env_id == eid
            ), "GOPS: policy {} is not trained in the same environment".format(i)
        return env_id

    def run(self):
        self.__run_data()
        self.__save_mp4_as_gif()
        self.draw()

    def __run_data(self):
        if self.use_opt:
            if self.multi_opt:
                for run_time in range(int(self.multi_opt_args["opt_run_times"])):
                    if self.load_opt_path is not None:
                        eval_dict_opt = np.load(
                            os.path.join(self.load_opt_path, "eval_dict_opt.npy"),
                            allow_pickle=True).item()
                        tracking_dict_opt = np.load(
                            os.path.join(self.load_opt_path, "tracking_dict_opt.npy"),
                            allow_pickle=True).item()
                        print("Successfully load an optimal controller result!")
                        print("===========================================================\n")
                    else:
                        self.args = self.args_list[0]
                        print("GOPS: Use an optimal controller")
                        env = self.__load_env(use_opt=True)
                        print("The environment for opt")
                        if hasattr(env, "set_mode"):
                            env.set_mode("test")

                        assert (
                                self.opt_args is not None
                        ), "Choose to use optimal controller, but the opt_args is None."

                        if self.opt_args["opt_controller_type"] == "OPT":
                            assert (
                                env.has_optimal_controller
                            ), "The environment has no theoretical optimal controller."
                            opt_controller = env.control_policy
                        elif self.opt_args["opt_controller_type"] == "MPC":
                            if self.opt_args["use_MPC_for_general_env"] == True:
                                self.args_list[0]["env"] = env
                                from gops.sys_simulator.opt_controller_for_gen_env import OptController
                            else:
                                from gops.sys_simulator.opt_controller import OptController
                            model = create_env_model(**self.args_list[0], mask_at_done=False)
                            model.update_cost_paras(self.multi_opt_args["cost_paras_list"][run_time])
                            print("The cost_paras for opt:", self.multi_opt_args["cost_paras_list"][run_time])
                            opt_args = self.opt_args.copy()
                            opt_args.pop("opt_controller_type")
                            opt_args.pop("use_MPC_for_general_env")
                            opt_controller = OptController(model, **opt_args, )
                        else:
                            raise ValueError(
                                "The optimal controller type should be either 'OPT' or 'MPC'."
                            )

                        eval_dict_opt, tracking_dict_opt = self.run_an_episode(
                            env, opt_controller, self.init_info, is_opt=True, render=False
                        )
                        print("Successfully run an optimal controller!")
                        print("===========================================================\n")

                    if self.opt_args["opt_controller_type"] == "OPT":
                        legend = "OPT"
                    elif self.opt_args["opt_controller_type"] == "MPC":
                        if self.multi_opt:
                            legend = "MPC-" + str(run_time)
                            if (
                                    "use_terminal_cost" not in self.opt_args.keys()
                                    or self.opt_args["use_terminal_cost"] == False
                            ):
                                legend += " (w/o TC)"
                            else:
                                legend += " (w/ TC)"
                        else:
                            legend = "MPC-" + str(self.opt_args["num_pred_step"])
                            if (
                                    "use_terminal_cost" not in self.opt_args.keys()
                                    or self.opt_args["use_terminal_cost"] == False
                            ):
                                legend += " (w/o TC)"
                            else:
                                legend += " (w/ TC)"
                    self.legend_list.append(legend)

                    if self.save_opt:
                        np.save(os.path.join(self.save_path, "eval_dict_opt.npy"), eval_dict_opt)
                        np.save(os.path.join(self.save_path, "tracking_dict_opt.npy"), tracking_dict_opt)

                    self.eval_list.append(eval_dict_opt)
                    if self.is_tracking:
                        self.tracking_list.append(tracking_dict_opt)
                if self.opt_args["opt_controller_type"] == "OPT":
                    legend = "OPT"
                elif self.opt_args["opt_controller_type"] == "MPC":
                    legend = "MPC-" + str(self.opt_args["num_pred_step"])
                    if (
                            "use_terminal_cost" not in self.opt_args.keys()
                            or self.opt_args["use_terminal_cost"] == False
                    ):
                        legend += " (w/o TC)"
                    else:
                        legend += " (w/ TC)"
                self.legend_list.append(legend)

                if self.save_opt:
                    np.save(os.path.join(self.save_path, "eval_dict_opt.npy"), eval_dict_opt)
                    np.save(os.path.join(self.save_path, "tracking_dict_opt.npy"), tracking_dict_opt)

                self.eval_list.append(eval_dict_opt)
                if self.is_tracking:
                    self.tracking_list.append(tracking_dict_opt)
            else:
                if self.load_opt_path is not None:
                    eval_dict_opt = np.load(
                        os.path.join(self.load_opt_path, "eval_dict_opt.npy"),
                        allow_pickle=True).item()
                    tracking_dict_opt = np.load(
                        os.path.join(self.load_opt_path, "tracking_dict_opt.npy"),
                        allow_pickle=True).item()
                    print("Successfully load an optimal controller result!")
                    print("===========================================================\n")
                else:
                    self.args = self.args_list[0]
                    print("GOPS: Use an optimal controller")

                    env = self.__load_env(use_opt=True)

                    print("The cosimulation environment for opt")
                    if hasattr(env, "set_mode"):
                        env.set_mode("test")

                    assert (
                            self.opt_args is not None
                    ), "Choose to use optimal controller, but the opt_args is None."

                    if self.opt_args["opt_controller_type"] == "OPT":
                        assert (
                            env.has_optimal_controller
                        ), "The environment has no theoretical optimal controller."
                        opt_controller = env.control_policy
                    elif self.opt_args["opt_controller_type"] == "MPC":
                        if self.opt_args["use_MPC_for_general_env"] == True:
                            self.args_list[0]["env"] = env
                            from gops.sys_simulator.opt_controller_for_gen_env import OptController
                        else:
                            from gops.sys_simulator.opt_controller import OptController
                        model = create_env_model(**self.args_list[0], mask_at_done=False)
                        opt_args = self.opt_args.copy()
                        opt_args.pop("opt_controller_type")
                        opt_args.pop("use_MPC_for_general_env")
                        opt_controller = OptController(model, **opt_args, )
                    else:
                        raise ValueError(
                            "The optimal controller type should be either 'OPT' or 'MPC'."
                        )

                    eval_dict_opt, tracking_dict_opt = self.run_an_episode(
                        env, opt_controller, self.init_info, is_opt=True, render=False
                    )
                    print("Successfully run an optimal controller!")
                    print("===========================================================\n")

                if self.opt_args["opt_controller_type"] == "OPT":
                    legend = "OPT"
                elif self.opt_args["opt_controller_type"] == "MPC":
                    legend = "MPC-" + str(self.opt_args["num_pred_step"])
                    if (
                            "use_terminal_cost" not in self.opt_args.keys()
                            or self.opt_args["use_terminal_cost"] == False
                    ):
                        legend += " (w/o TC)"
                    else:
                        legend += " (w/ TC)"
                self.legend_list.append(legend)

                if self.save_opt:
                    np.save(os.path.join(self.save_path, "eval_dict_opt.npy"), eval_dict_opt)
                    np.save(os.path.join(self.save_path, "tracking_dict_opt.npy"), tracking_dict_opt)

                self.eval_list.append(eval_dict_opt)
                if self.is_tracking:
                    self.tracking_list.append(tracking_dict_opt)

    def run_an_episode(
            self,
            env: Any,
            controller: Any,
            init_info: dict,
            is_opt: bool,
            render: bool = True,
    ) -> Tuple[dict, dict]:
        state_list = []
        action_list = []
        reward_list = []
        constrain_list = []
        obs_list = []
        step = 0
        step_list = []
        calctime_list = []
        info_list = [init_info]
        env.load_carsim_env()

        obs, info = env.reset_carsim(**init_info)
        state = env.state
        print("Initial robot state: ")
        print(self.__convert_format(np.asarray(state.robot_state)))
        # plot tracking
        state_with_ref_error = {}
        done = False
        info.update({"TimeLimit.truncated": False})
        while not (done or info["TimeLimit.truncated"]):
            print("step:", step + 1)
            state_list.append(state.robot_state)
            obs_list.append(obs)
            if is_opt:
                if isinstance(env.unwrapped, Env):
                    time_start = time.time()
                    action = controller(state)
                    calc_time = time.time() - time_start
                else:
                    time_start = time.time()
                    action = controller(obs, info)
                    calc_time = time.time() - time_start
            else:
                time_start = time.time()
                action = self.compute_action(obs, controller)
                action = self.__action_noise(action)
                calc_time = time.time() - time_start
            if self.use_dist:
                action = np.hstack((action, env.dist_func(step * env.tau)))
            if self.constrained_env:
                constrain_list.append(info["constraint"])
            if self.is_tracking:
                reference = get_reference_from_info(info)
                state_num = len(reference)
                self.ref_state_num = sum(x is not None for x in reference)
                if step == 0:
                    for i in range(state_num):
                        if reference[i] is not None:
                            state_with_ref_error["state-{}".format(i)] = []
                            state_with_ref_error["ref-{}".format(i)] = []
                            state_with_ref_error["state-{}-error".format(i)] = []

                robot_state = get_robot_state_from_info(info)
                for i in range(state_num):
                    if reference[i] is not None:
                        state_with_ref_error["state-{}".format(i)].append(robot_state[i])
                        state_with_ref_error["ref-{}".format(i)].append(reference[i])
                        state_with_ref_error["state-{}-error".format(i)].append(
                            reference[i] - robot_state[i]
                        )

            next_obs, reward, done, info = env.step_carsim(action)
            # save the real action (without scaling)
            action_list.append(info.get("raw_action", action))
            step_list.append(step)
            reward_list.append(reward)
            info_list.append(info)
            calctime_list.append(calc_time*1000)

            obs = next_obs
            state = env.state
            step = step + 1

            if "TimeLimit.truncated" not in info.keys():
                info["TimeLimit.truncated"] = False
            # Draw environment animation
            if render:
                env.render()
        env.carsim_env.get_ternimated()
        eval_dict = {
            "reward_list": reward_list,
            "action_list": action_list,
            "state_list": state_list,
            "step_list": step_list,
            "obs_list": obs_list,
            "info_list": info_list,
            "calctime_list": calctime_list
        }
        if self.constrained_env:
            eval_dict.update(
                {"constrain_list": constrain_list, }
            )

        if self.is_tracking:
            tracking_dict = state_with_ref_error
        else:
            tracking_dict = {}

        return eval_dict, tracking_dict

    def draw(self):
        fig_size = (
            default_cfg["fig_size"],
            default_cfg["fig_size"],
        )
        action_dim = self.eval_list[0]["action_list"][0].shape[0]
        state_dim = self.eval_list[0]["state_list"][0].shape[0]
        if self.constrained_env:
            constrain_dim = self.eval_list[0]["constrain_list"][0].shape[0]

        if self.use_opt:
            legend = ""
            if self.multi_opt:
                policy_num = self.multi_opt_args["opt_run_times"]
            else:
                policy_num = 1
            if self.multi_opt:
                for run_time in range(int(self.multi_opt_args["opt_run_times"])):
                    if self.opt_args["opt_controller_type"] == "OPT":
                        legend = "OPT"
                    elif self.opt_args["opt_controller_type"] == "MPC":
                        legend = "MPC-" + str(run_time)
                        if (
                                "use_terminal_cost" not in self.opt_args.keys()
                                or self.opt_args["use_terminal_cost"] is False
                        ):
                            legend += " (w/o TC)"
                        else:
                            legend += " (w/ TC)"
                    self.algorithm_list.append(legend)
            else:
                if self.opt_args["opt_controller_type"] == "OPT":
                    legend = "OPT"
                elif self.opt_args["opt_controller_type"] == "MPC":
                    legend = "MPC-" + str(self.opt_args["num_pred_step"])
                    if (
                            "use_terminal_cost" not in self.opt_args.keys()
                            or self.opt_args["use_terminal_cost"] is False
                    ):
                        legend += " (w/o TC)"
                    else:
                        legend += " (w/ TC)"
                self.algorithm_list.append(legend)

        # Create initial list
        reward_list = []
        action_list = []
        state_list = []
        step_list = []
        state_ref_error_list = []
        constrain_list = []
        calctime_list = []
        # Put data into list
        for i in range(policy_num):
            reward_list.append(np.array(self.eval_list[i]["reward_list"]))
            action_list.append(np.array(self.eval_list[i]["action_list"]))
            state_list.append(np.array(self.eval_list[i]["state_list"]))
            step_list.append(np.array(self.eval_list[i]["step_list"]))
            calctime_list.append(np.array(self.eval_list[i]["calctime_list"]))
            if self.constrained_env:
                constrain_list.append(np.stack(self.eval_list[i]["constrain_list"]))
            if self.is_tracking:
                state_ref_error_list.append(self.tracking_list[i])

        if self.plot_range is None:
            pass
        elif len(self.plot_range) == 2:

            for i in range(policy_num):
                start_range = self.plot_range[0]
                end_range = min(self.plot_range[1], reward_list[i].shape[0])

                reward_list[i] = reward_list[i][start_range:end_range]
                action_list[i] = action_list[i][start_range:end_range]
                state_list[i] = state_list[i][start_range:end_range]
                step_list[i] = step_list[i][start_range:end_range]
                if self.constrained_env:
                    constrain_list[i] = constrain_list[i][start_range:end_range]
                if self.is_tracking:
                    for key, value in self.tracking_list[i].items():
                        self.tracking_list[i][key] = value[start_range:end_range]
        else:
            raise NotImplementedError("Figure range is wrong")

        if self.dt is None:
            x_label = "Time step"
        else:
            step_list = [s * self.dt for s in step_list]
            x_label = "Time (s)"

        # Plot reward
        path_reward_fmt = os.path.join(
            self.save_path, "Reward.{}".format(default_cfg["img_fmt"])
        )
        fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

        # save reward data to csv
        reward_data = pd.DataFrame(data=reward_list[0])
        reward_data.to_csv(os.path.join(self.save_path, "Reward.csv"), encoding="gbk")

        for i in range(policy_num):
            legend = (
                self.legend_list[i]
                if len(self.legend_list) == policy_num
                else self.algorithm_list[i]
            )
            sns.lineplot(x=step_list[i], y=reward_list[i], label="{}".format(legend))
        plt.tick_params(labelsize=default_cfg["tick_size"])
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
        plt.xlabel(x_label, default_cfg["label_font"])
        plt.ylabel("Reward", default_cfg["label_font"])
        plt.legend(loc="best", prop=default_cfg["legend_font"])
        fig.tight_layout(pad=default_cfg["pad"])
        plt.savefig(path_reward_fmt, format=default_cfg["img_fmt"], bbox_inches="tight")
        plt.close()

        # plot action
        for j in range(action_dim):
            path_action_fmt = os.path.join(
                self.save_path, "Action-{}.{}".format(j + 1, default_cfg["img_fmt"])
            )
            fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

            # save action data to csv
            action_data = pd.DataFrame(data=[a[:, j] for a in action_list])
            action_data.to_csv(
                os.path.join(self.save_path, "Action-{}.csv".format(j + 1)),
                encoding="gbk",
            )

            for i in range(policy_num):
                legend = (
                    self.legend_list[i]
                    if len(self.legend_list) == policy_num
                    else self.algorithm_list[i]
                )
                sns.lineplot(
                    x=step_list[i], y=action_list[i][:, j], label="{}".format(legend)
                )
            plt.tick_params(labelsize=default_cfg["tick_size"])
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
            plt.xlabel(x_label, default_cfg["label_font"])
            plt.ylabel("Action-{}".format(j + 1), default_cfg["label_font"])
            plt.legend(loc="best", prop=default_cfg["legend_font"])
            fig.tight_layout(pad=default_cfg["pad"])
            plt.savefig(
                path_action_fmt, format=default_cfg["img_fmt"], bbox_inches="tight"
            )
            plt.close()

        # plot state
        for j in range(state_dim):
            path_state_fmt = os.path.join(
                self.save_path, "State-{}.{}".format(j + 1, default_cfg["img_fmt"])
            )
            fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

            # save state data to csv
            state_data = pd.DataFrame(data=[s[:, j] for s in state_list])
            state_data.to_csv(
                os.path.join(self.save_path, "State-{}.csv".format(j + 1)),
                encoding="gbk",
            )

            for i in range(policy_num):
                legend = (
                    self.legend_list[i]
                    if len(self.legend_list) == policy_num
                    else self.algorithm_list[i]
                )
                sns.lineplot(
                    x=step_list[i], y=state_list[i][:, j], label="{}".format(legend)
                )
            plt.tick_params(labelsize=default_cfg["tick_size"])
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
            plt.xlabel(x_label, default_cfg["label_font"])
            plt.ylabel("State-{}".format(j + 1), default_cfg["label_font"])
            plt.legend(loc="best", prop=default_cfg["legend_font"])
            fig.tight_layout(pad=default_cfg["pad"])
            plt.savefig(
                path_state_fmt, format=default_cfg["img_fmt"], bbox_inches="tight"
            )
            plt.close()

        # plot tracking
        if self.is_tracking:
            # find index of the longest trajectory
            traj_lens = [len(r) for r in reward_list]
            longest_traj_index = np.argmax(traj_lens)

            for j in range(self.ref_state_num):

                # plot state and ref
                path_tracking_state_fmt = os.path.join(
                    self.save_path, "Ref - State - {}.{}".format(j + 1, default_cfg["img_fmt"])
                )
                fig, ax = plt.subplots(
                    figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"]
                )
                # save tracking state data to csv
                tracking_state_data = []
                for i in range(policy_num):
                    legend = (
                        self.legend_list[i]
                        if len(self.legend_list) == policy_num
                        else self.algorithm_list[i]
                    )
                    sns.lineplot(
                        x=step_list[i],
                        y=state_ref_error_list[i]["state-{}".format(j)],
                        label="{}".format(legend),
                    )
                    tracking_state_data.append(
                        state_ref_error_list[i]["state-{}".format(j)]
                    )
                sns.lineplot(
                    x=step_list[longest_traj_index],
                    y=state_ref_error_list[longest_traj_index]["ref-{}".format(j)],
                    label="ref",
                )
                tracking_state_data.append(state_ref_error_list[longest_traj_index]["ref-{}".format(j)])
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel(x_label, default_cfg["label_font"])
                plt.ylabel("State-{}".format(j + 1), default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(
                    path_tracking_state_fmt,
                    format=default_cfg["img_fmt"],
                    bbox_inches="tight",
                )
                plt.close()

                tracking_state_data = pd.DataFrame(data=tracking_state_data)
                tracking_state_data.to_csv(
                    os.path.join(self.save_path, "State-{}.csv".format(j + 1)),
                    encoding="gbk",
                )

                # plot state-ref error
                path_tracking_error_fmt = os.path.join(
                    self.save_path,
                    "Ref - State - Error{}.{}".format(j + 1, default_cfg["img_fmt"]),
                )
                fig, ax = plt.subplots(
                    figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"]
                )
                # save tracking error data to csv
                tracking_error_data = []
                for i in range(policy_num):
                    legend = (
                        self.legend_list[i]
                        if len(self.legend_list) == policy_num
                        else self.algorithm_list[i]
                    )
                    sns.lineplot(
                        x=step_list[i],
                        y=state_ref_error_list[i]["state-{}-error".format(j)],
                        label="{}".format(legend),
                    )
                    tracking_error_data.append(
                        state_ref_error_list[i]["state-{}-error".format(j)]
                    )
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel(x_label, default_cfg["label_font"])
                plt.ylabel("Ref$-$State-Error{}".format(j + 1), default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(
                    path_tracking_error_fmt,
                    format=default_cfg["img_fmt"],
                    bbox_inches="tight",
                )
                plt.close()

                tracking_error_data = pd.DataFrame(data=tracking_error_data)
                tracking_error_data.to_csv(
                    os.path.join(self.save_path, "Ref-State-Error{}.csv".format(j + 1)),
                    encoding="gbk",
                )
        # plot calculation time
        path_state_fmt = os.path.join(
            self.save_path, "Calc time.{}".format(default_cfg["img_fmt"])
        )
        fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

        # save state data to csv
        state_data = pd.DataFrame(data=[s[:] for s in calctime_list])
        state_data.to_csv(
            os.path.join(self.save_path, "Calc time.csv"),
            encoding="gbk",
        )

        for i in range(policy_num):
            legend = (
                self.legend_list[i]
                if len(self.legend_list) == policy_num
                else self.algorithm_list[i]
            )
            sns.lineplot(
                x=step_list[i], y=calctime_list[i][:], label="{}".format(legend)
            )
        plt.tick_params(labelsize=default_cfg["tick_size"])
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
        plt.xlabel(x_label, default_cfg["label_font"])
        plt.ylabel("Calc Time [ms]", default_cfg["label_font"])
        plt.legend(loc="best", prop=default_cfg["legend_font"])
        fig.tight_layout(pad=default_cfg["pad"])
        plt.savefig(
            path_state_fmt, format=default_cfg["img_fmt"], bbox_inches="tight"
        )
        plt.close()
        # plot constraint value
        if self.constrained_env:
            for j in range(constrain_dim):
                path_constraint_fmt = os.path.join(
                    self.save_path,
                    "Constrain-{}.{}".format(j + 1, default_cfg["img_fmt"]),
                )
                fig, ax = plt.subplots(
                    figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"]
                )

                # save reward data to csv
                constrain_data = pd.DataFrame(data=[c[:, j] for c in constrain_list])
                constrain_data.to_csv(
                    os.path.join(self.save_path, "Constrain-{}.csv".format(j + 1)),
                    encoding="gbk",
                )

                for i in range(policy_num):
                    legend = (
                        self.legend_list[i]
                        if len(self.legend_list) == policy_num
                        else self.algorithm_list[i]
                    )
                    sns.lineplot(
                        x=step_list[i],
                        y=constrain_list[i][:, j],
                        label="{}".format(legend),
                    )
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel(x_label, default_cfg["label_font"])
                plt.ylabel("Constrain-{}".format(j + 1), default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(
                    path_constraint_fmt,
                    format=default_cfg["img_fmt"],
                    bbox_inches="tight",
                )
                plt.close()

        # plot error with opt
        if self.use_opt:
            # reward error
            path_reward_error_fmt = os.path.join(
                self.save_path, "Reward error.{}".format(default_cfg["img_fmt"])
            )
            fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

            # save reward error data to csv
            reward_error_list = []
            for r in reward_list:
                end = min(len(r), len(reward_list[-1]))
                reward_error_list.append(r[:end] - reward_list[-1][:end])
            reward_error_data = pd.DataFrame(data=reward_error_list)
            reward_error_data.to_csv(
                os.path.join(self.save_path, "Reward error.csv"), encoding="gbk"
            )

            for i in range(policy_num - 1):
                legend = (
                    self.legend_list[i]
                    if len(self.legend_list) == policy_num
                    else self.algorithm_list[i]
                )
                sns.lineplot(
                    x=step_list[i][:len(reward_error_list[i])],
                    y=reward_error_list[i], label="{}".format(legend)
                )
            plt.tick_params(labelsize=default_cfg["tick_size"])
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
            plt.xlabel(x_label, default_cfg["label_font"])
            plt.ylabel("Reward error", default_cfg["label_font"])
            plt.legend(loc="best", prop=default_cfg["legend_font"])
            fig.tight_layout(pad=default_cfg["pad"])
            plt.savefig(
                path_reward_error_fmt,
                format=default_cfg["img_fmt"],
                bbox_inches="tight",
            )
            plt.close()

            # action error
            action_error_list = []
            for a in action_list:
                end = min(len(a), len(action_list[-1]))
                action_error_list.append(a[:end] - action_list[-1][:end])
            for j in range(action_dim):
                path_action_error_fmt = os.path.join(
                    self.save_path,
                    "Action-{} error.{}".format(j + 1, default_cfg["img_fmt"]),
                )
                fig, ax = plt.subplots(
                    figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"]
                )
                for i in range(policy_num - 1):
                    legend = (
                        self.legend_list[i]
                        if len(self.legend_list) == policy_num
                        else self.algorithm_list[i]
                    )
                    sns.lineplot(
                        x=step_list[i][:len(action_error_list[i])],
                        y=action_error_list[i][:, j],
                        label="{}".format(legend),
                    )
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel(x_label, default_cfg["label_font"])
                plt.ylabel("Action-{} error".format(j + 1), default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(
                    path_action_error_fmt,
                    format=default_cfg["img_fmt"],
                    bbox_inches="tight",
                )
                plt.close()

                # save action error data to csv
                action_error_data = pd.DataFrame(data=[a[:, j] for a in action_error_list])
                action_error_data.to_csv(
                    os.path.join(self.save_path, "Action-{} error.csv".format(j + 1)),
                    encoding="gbk",
                )

            # state error
            state_error_list = []
            for s in state_list:
                end = min(len(s), len(state_list[-1]))
                state_error_list.append(s[:end] - state_list[-1][:end])
            for j in range(state_dim):
                path_state_error_fmt = os.path.join(
                    self.save_path,
                    "State-{} error.{}".format(j + 1, default_cfg["img_fmt"]),
                )
                fig, ax = plt.subplots(
                    figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"]
                )
                for i in range(policy_num - 1):
                    legend = (
                        self.legend_list[i]
                        if len(self.legend_list) == policy_num
                        else self.algorithm_list[i]
                    )
                    sns.lineplot(
                        x=step_list[i][:len(state_error_list[i])],
                        y=state_error_list[i][:, j],
                        label="{}".format(legend),
                    )
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel(x_label, default_cfg["label_font"])
                plt.ylabel("State-{} error".format(j + 1), default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(
                    path_state_error_fmt,
                    format=default_cfg["img_fmt"],
                    bbox_inches="tight",
                )
                plt.close()

                # save state data to csv
                state_error_data = pd.DataFrame(data=[s[:, j] for s in state_error_list])
                state_error_data.to_csv(
                    os.path.join(self.save_path, "State-{} error.csv".format(j + 1)),
                    encoding="gbk",
                )

            # compute relative error with opt
            error_result = {}
            for i in range(policy_num - 1):
                legend = (
                    self.legend_list[i]
                    if len(self.legend_list) == policy_num
                    else "Policy-{}".format(i + 1)
                )
                end = min(len(action_list[i]), len(action_list[-1]))
                error_result.update({legend: {}})
                # action error
                for j in range(action_dim):
                    action_error = {}
                    error_list = np.abs(
                        action_list[i][:end, j] - action_list[-1][:end, j]
                    ) / (
                                         np.max(action_list[-1][:end, j])
                                         - np.min(action_list[-1][:end, j])
                                 )
                    action_error["Max_error"] = "{:.2f}%".format(max(error_list) * 100)
                    action_error["Mean_error"] = "{:.2f}%".format(
                        sum(error_list) / len(error_list) * 100
                    )
                    error_result[legend].update(
                        {"Action-{}".format(j + 1): action_error}
                    )
                # state error
                for j in range(state_dim):
                    state_error = {}
                    error_list = np.abs(
                        state_list[i][:end, j] - state_list[-1][:end, j]
                    ) / (
                                         np.max(state_list[-1][:end, j])
                                         - np.min(state_list[-1][:end, j])
                                 )
                    state_error["Max_error"] = "{:.2f}%".format(max(error_list) * 100)
                    state_error["Mean_error"] = "{:.2f}%".format(
                        sum(error_list) / len(error_list) * 100
                    )
                    error_result[legend].update({"State-{}".format(j + 1): state_error})

            for i in range(self.policy_num):
                legend = (
                    self.legend_list[i]
                    if len(self.legend_list) == policy_num
                    else "Policy-{}".format(i + 1)
                )
                policy_result = pd.DataFrame(data=error_result[legend])
                policy_result.to_excel(os.path.join(self.save_path, "Error-result.xlsx"), legend)
            error_result_data = pd.DataFrame(data=error_result)
            pd.set_option("display.max_columns", None)
            pd.set_option("display.max_rows", None)
            for key, value in error_result_data.items():
                print("===========================================================")
                print("GOPS: Policy {}".format(key))
                for key, value in value.items():
                    print(key, value)

class PolicyRunner_CoSimulation:
    """Plot module for trained policy

    :param list log_policy_dir_list: directory of trained policy.
    :param list trained_policy_iteration_list: iteration of trained policy.
    :param bool save_render: save environment animation or not.
    :param list plot_range: customize plot range.
    :param bool is_init_info: customize initial information or not.
    :param dict init_info: initial information.
    :param list legend_list: legends of figures.
    :param bool use_opt: use optimal solution for comparison or not.
    :param Optional[str] load_opt_path: path to load optimal controller result.
    :param dict opt_args: arguments of optimal solution solver.
    :param bool save_opt: save optimal controller result or not.
    :param bool constrained_env: constrained environment or not.
    :param bool is_tracking: tracking problem or not.
    :param bool use_dist: use adversarial action or not.
    :param float dt: time interval between steps.
    :param str obs_noise_type: type of observation noise, "normal" or "uniform".
    :param list obs_noise_data: Mean and
        Standard deviation of Normal distribution or Upper
        and Lower bounds of Uniform distribution.
    :param str action_noise_type: type of action noise, "normal" or "uniform".
    :param list action_noise_data: Mean and
        Standard deviation of Normal distribution or Upper
        and Lower bounds of Uniform distribution.
    """

    def __init__(
        self,
        log_policy_dir_list: list,
        trained_policy_iteration_list: list,
        save_render: bool = False,
        plot_range: list = None,
        is_init_info: bool = False,
        init_info: dict = None,
        legend_list: list = None,
        use_opt: bool = False,
        load_opt_path: Optional[str] = None,
        opt_args: Optional[dict] = None,
        save_opt: bool = True,
        constrained_env: bool = False,
        is_tracking: bool = False,
        use_dist: bool = False,
        dt: float = None,
        obs_noise_type: str = None,
        obs_noise_data: list = None,
        action_noise_type: str = None,
        action_noise_data: list = None,
    ):
        self.log_policy_dir_list = [
            os.path.join(gops_path, d) for d in log_policy_dir_list
        ]
        self.trained_policy_iteration_list = trained_policy_iteration_list
        self.save_render = save_render
        self.args = None
        self.plot_range = plot_range
        if is_init_info:
            self.init_info = init_info
        else:
            self.init_info = {}
        self.legend_list = legend_list
        self.use_opt = use_opt
        if use_opt:
            assert load_opt_path is not None or opt_args is not None
            self.load_opt_path = load_opt_path
            self.opt_args = opt_args
            if isinstance(self.opt_args, dict) and \
                "use_MPC_for_general_env" not in self.opt_args.keys():
                self.opt_args["use_MPC_for_general_env"] = False
            self.save_opt = save_opt
        self.constrained_env = constrained_env
        self.use_dist = use_dist
        self.is_tracking = is_tracking
        self.dt = dt
        self.policy_num = len(self.log_policy_dir_list)
        if self.policy_num != len(self.trained_policy_iteration_list):
            raise RuntimeError(
                "The length of policy number is not equal to that of policy iteration"
            )
        self.obs_noise_type = obs_noise_type
        self.obs_noise_data = obs_noise_data
        self.action_noise_type = action_noise_type
        self.action_noise_data = action_noise_data
        self.ref_state_num = 0

        # data for plot
        self.args_list = []
        self.eval_list = []
        self.env_id_list = []
        self.algorithm_list = []
        self.tracking_list = []

        self.__load_all_args()
        self.env_id = self.get_n_verify_env_id()

        # save path
        path = os.path.join(os.path.dirname(__file__), "..", "..", "figures")
        path = os.path.abspath(path)

        algs_name = ""
        for item in self.algorithm_list:
            algs_name = algs_name + item + "-"
        self.save_path = os.path.join(
            path,
            algs_name + self.env_id,
            datetime.datetime.now().strftime("%y%m%d-%H%M%S")+"_carsim",
        )
        os.makedirs(self.save_path, exist_ok=True)

    def run_an_episode(
        self,
        env: Any,
        controller: Any,
        init_info: dict,
        is_opt: bool,
        render: bool = True,
    ) -> Tuple[dict, dict]:
        state_list = []
        action_list = []
        reward_list = []
        constrain_list = []
        obs_list = []
        step = 0
        step_list = []
        calctime_list = []
        info_list = [init_info]
        env.load_carsim_env()
        obs, info = env.reset_carsim(**init_info)
        state = env.state
        print("Initial robot state: ")
        print(self.__convert_format(np.asarray(state.robot_state)))
        # plot tracking
        state_with_ref_error = {}
        done = False
        info.update({"TimeLimit.truncated": False})
        while not (done or info["TimeLimit.truncated"]):
            print("step:", step + 1)
            state_list.append(state.robot_state)
            obs_list.append(obs)
            if is_opt:
                if isinstance(env.unwrapped, Env):
                    time_start = time.time()
                    action = controller(state)
                    calc_time = time.time() - time_start
                else:
                    time_start = time.time()
                    action = controller(obs, info)
                    calc_time = time.time() - time_start
            else:
                time_start = time.time()
                action = self.compute_action(obs, controller)
                action = self.__action_noise(action)
                calc_time = time.time() - time_start
            if self.use_dist:
                action = np.hstack((action, env.dist_func(step * env.tau)))
            if self.constrained_env:
                constrain_list.append(info["constraint"])
            if self.is_tracking:
                reference = get_reference_from_info(info)
                state_num = len(reference)
                self.ref_state_num = sum(x is not None for x in reference)
                if step == 0:
                    for i in range(state_num):
                        if reference[i] is not None:
                            state_with_ref_error["state-{}".format(i)] = []
                            state_with_ref_error["ref-{}".format(i)] = []
                            state_with_ref_error["state-{}-error".format(i)] = []

                robot_state = get_robot_state_from_info(info)
                for i in range(state_num):
                    if reference[i] is not None:
                        state_with_ref_error["state-{}".format(i)].append(robot_state[i])
                        state_with_ref_error["ref-{}".format(i)].append(reference[i])
                        state_with_ref_error["state-{}-error".format(i)].append(
                            reference[i] - robot_state[i]
                        )
            next_obs, reward, done, info = env.step_carsim(action)

            # save the real action (without scaling)
            action_list.append(info.get("raw_action", action))
            step_list.append(step)
            reward_list.append(reward)
            info_list.append(info)
            calctime_list.append(calc_time*1000)

            obs = next_obs
            state = env.state
            step = step + 1

            if "TimeLimit.truncated" not in info.keys():
                info["TimeLimit.truncated"] = False
            # Draw environment animation
            if render:
                env.render()
        env.carsim_env.get_ternimated()
        eval_dict = {
            "reward_list": reward_list,
            "action_list": action_list,
            "state_list": state_list,
            "step_list": step_list,
            "obs_list": obs_list,
            "info_list": info_list,
            "calctime_list": calctime_list
        }
        if self.constrained_env:
            eval_dict.update(
                {"constrain_list": constrain_list,}
            )

        if self.is_tracking:
            tracking_dict = state_with_ref_error
        else:
            tracking_dict = {}

        return eval_dict, tracking_dict

    def compute_action(self, obs: np.ndarray, networks: Any) -> np.ndarray:
        batch_obs = torch.from_numpy(np.expand_dims(obs, axis=0).astype("float32"))
        logits = networks.policy(batch_obs)
        action_distribution = networks.create_action_distributions(logits)
        action = action_distribution.mode()
        action = action.detach().numpy()[0]
        return action

    def draw(self):
        fig_size = (
            default_cfg["fig_size"],
            default_cfg["fig_size"],
        )
        action_dim = self.eval_list[0]["action_list"][0].shape[0]
        state_dim = self.eval_list[0]["state_list"][0].shape[0]
        if self.constrained_env:
            constrain_dim = self.eval_list[0]["constrain_list"][0].shape[0]
        policy_num = len(self.algorithm_list)
        if self.use_opt:
            legend = ""
            policy_num += 1
            if self.opt_args["opt_controller_type"] == "OPT":
                legend = "OPT"
            elif self.opt_args["opt_controller_type"] == "MPC":
                legend = "MPC-" + str(self.opt_args["num_pred_step"])
                if (
                    "use_terminal_cost" not in self.opt_args.keys()
                    or self.opt_args["use_terminal_cost"] is False
                ):
                    legend += " (w/o TC)"
                else:
                    legend += " (w/ TC)"
            self.algorithm_list.append(legend)

        # Create initial list
        reward_list = []
        action_list = []
        state_list = []
        step_list = []
        state_ref_error_list = []
        constrain_list = []
        calctime_list = []
        # Put data into list
        for i in range(policy_num):
            reward_list.append(np.array(self.eval_list[i]["reward_list"]))
            action_list.append(np.array(self.eval_list[i]["action_list"]))
            state_list.append(np.array(self.eval_list[i]["state_list"]))
            step_list.append(np.array(self.eval_list[i]["step_list"]))
            calctime_list.append(np.array(self.eval_list[i]["calctime_list"]))
            if self.constrained_env:
                constrain_list.append(np.stack(self.eval_list[i]["constrain_list"]))
            if self.is_tracking:
                state_ref_error_list.append(self.tracking_list[i])

        if self.plot_range is None:
            pass
        elif len(self.plot_range) == 2:

            for i in range(policy_num):
                start_range = self.plot_range[0]
                end_range = min(self.plot_range[1], reward_list[i].shape[0])

                reward_list[i] = reward_list[i][start_range:end_range]
                action_list[i] = action_list[i][start_range:end_range]
                state_list[i] = state_list[i][start_range:end_range]
                step_list[i] = step_list[i][start_range:end_range]
                if self.constrained_env:
                    constrain_list[i] = constrain_list[i][start_range:end_range]
                if self.is_tracking:
                    for key, value in self.tracking_list[i].items():
                        self.tracking_list[i][key] = value[start_range:end_range]
        else:
            raise NotImplementedError("Figure range is wrong")

        if self.dt is None:
            x_label = "Time step"
        else:
            step_list = [s * self.dt for s in step_list]
            x_label = "Time (s)"

        # Plot reward
        path_reward_fmt = os.path.join(
            self.save_path, "Reward.{}".format(default_cfg["img_fmt"])
        )
        fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

        # save reward data to csv
        reward_data = pd.DataFrame(data=reward_list)
        reward_data.to_csv(os.path.join(self.save_path, "Reward.csv"), encoding="gbk")

        for i in range(policy_num):
            legend = (
                self.legend_list[i]
                if len(self.legend_list) == policy_num
                else self.algorithm_list[i]
            )
            sns.lineplot(x=step_list[i], y=reward_list[i], label="{}".format(legend))
        plt.tick_params(labelsize=default_cfg["tick_size"])
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
        plt.xlabel(x_label, default_cfg["label_font"])
        plt.ylabel("Reward", default_cfg["label_font"])
        plt.legend(loc="best", prop=default_cfg["legend_font"])
        fig.tight_layout(pad=default_cfg["pad"])
        plt.savefig(path_reward_fmt, format=default_cfg["img_fmt"], bbox_inches="tight")
        plt.close()

        # plot action
        for j in range(action_dim):
            path_action_fmt = os.path.join(
                self.save_path, "Action-{}.{}".format(j + 1, default_cfg["img_fmt"])
            )
            fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

            # save action data to csv
            action_data = pd.DataFrame(data=[a[:, j] for a in action_list])
            action_data.to_csv(
                os.path.join(self.save_path, "Action-{}.csv".format(j + 1)),
                encoding="gbk",
            )

            for i in range(policy_num):
                legend = (
                    self.legend_list[i]
                    if len(self.legend_list) == policy_num
                    else self.algorithm_list[i]
                )
                sns.lineplot(
                    x=step_list[i], y=action_list[i][:, j], label="{}".format(legend)
                )
            plt.tick_params(labelsize=default_cfg["tick_size"])
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
            plt.xlabel(x_label, default_cfg["label_font"])
            plt.ylabel("Action-{}".format(j + 1), default_cfg["label_font"])
            plt.legend(loc="best", prop=default_cfg["legend_font"])
            fig.tight_layout(pad=default_cfg["pad"])
            plt.savefig(
                path_action_fmt, format=default_cfg["img_fmt"], bbox_inches="tight"
            )
            plt.close()

        # plot state
        for j in range(state_dim):
            path_state_fmt = os.path.join(
                self.save_path, "State-{}.{}".format(j + 1, default_cfg["img_fmt"])
            )
            fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

            # save state data to csv
            state_data = pd.DataFrame(data=[s[:, j] for s in state_list])
            state_data.to_csv(
                os.path.join(self.save_path, "State-{}.csv".format(j + 1)),
                encoding="gbk",
            )

            for i in range(policy_num):
                legend = (
                    self.legend_list[i]
                    if len(self.legend_list) == policy_num
                    else self.algorithm_list[i]
                )
                sns.lineplot(
                    x=step_list[i], y=state_list[i][:, j], label="{}".format(legend)
                )
            plt.tick_params(labelsize=default_cfg["tick_size"])
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
            plt.xlabel(x_label, default_cfg["label_font"])
            plt.ylabel("State-{}".format(j + 1), default_cfg["label_font"])
            plt.legend(loc="best", prop=default_cfg["legend_font"])
            fig.tight_layout(pad=default_cfg["pad"])
            plt.savefig(
                path_state_fmt, format=default_cfg["img_fmt"], bbox_inches="tight"
            )
            plt.close()

        # plot tracking
        if self.is_tracking:
            # find index of the longest trajectory
            traj_lens = [len(r) for r in reward_list]
            longest_traj_index = np.argmax(traj_lens)

            for j in range(self.ref_state_num):

                # plot state and ref
                path_tracking_state_fmt = os.path.join(
                    self.save_path, "State-{}.{}".format(j + 1, default_cfg["img_fmt"])
                )
                fig, ax = plt.subplots(
                    figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"]
                )
                # save tracking state data to csv
                tracking_state_data = []
                for i in range(policy_num):
                    legend = (
                        self.legend_list[i]
                        if len(self.legend_list) == policy_num
                        else self.algorithm_list[i]
                    )
                    sns.lineplot(
                        x=step_list[i],
                        y=state_ref_error_list[i]["state-{}".format(j)],
                        label="{}".format(legend),
                    )
                    tracking_state_data.append(
                        state_ref_error_list[i]["state-{}".format(j)]
                    )
                sns.lineplot(
                    x=step_list[longest_traj_index],
                    y=state_ref_error_list[longest_traj_index]["ref-{}".format(j)],
                    label="ref",
                )
                tracking_state_data.append(state_ref_error_list[longest_traj_index]["ref-{}".format(j)])
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel(x_label, default_cfg["label_font"])
                plt.ylabel("State-{}".format(j + 1), default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(
                    path_tracking_state_fmt,
                    format=default_cfg["img_fmt"],
                    bbox_inches="tight",
                )
                plt.close()

                tracking_state_data = pd.DataFrame(data=tracking_state_data)
                tracking_state_data.to_csv(
                    os.path.join(self.save_path, "State-{}.csv".format(j + 1)),
                    encoding="gbk",
                )

                # plot state-ref error
                path_tracking_error_fmt = os.path.join(
                    self.save_path,
                    "Ref - State-{}.{}".format(j + 1, default_cfg["img_fmt"]),
                )
                fig, ax = plt.subplots(
                    figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"]
                )
                # save tracking error data to csv
                tracking_error_data = []
                for i in range(policy_num):
                    legend = (
                        self.legend_list[i]
                        if len(self.legend_list) == policy_num
                        else self.algorithm_list[i]
                    )
                    sns.lineplot(
                        x=step_list[i],
                        y=state_ref_error_list[i]["state-{}-error".format(j)],
                        label="{}".format(legend),
                    )
                    tracking_error_data.append(
                        state_ref_error_list[i]["state-{}-error".format(j)]
                    )
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel(x_label, default_cfg["label_font"])
                plt.ylabel("Ref $-$ State-{}".format(j + 1), default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(
                    path_tracking_error_fmt,
                    format=default_cfg["img_fmt"],
                    bbox_inches="tight",
                )
                plt.close()

                tracking_error_data = pd.DataFrame(data=tracking_error_data)
                tracking_error_data.to_csv(
                    os.path.join(self.save_path, "Ref - State-{}.csv".format(j + 1)),
                    encoding="gbk",
                )

        # plot calculation time
        path_state_fmt = os.path.join(
            self.save_path, "Calc time.{}".format(default_cfg["img_fmt"])
        )
        fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

        # save state data to csv
        state_data = pd.DataFrame(data=[s[:] for s in calctime_list])
        state_data.to_csv(
            os.path.join(self.save_path, "Calc time.csv"),
            encoding="gbk",
        )

        for i in range(policy_num):
            legend = (
                self.legend_list[i]
                if len(self.legend_list) == policy_num
                else self.algorithm_list[i]
            )
            sns.lineplot(
                x=step_list[i], y=calctime_list[i][:], label="{}".format(legend)
            )
        plt.tick_params(labelsize=default_cfg["tick_size"])
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
        plt.xlabel(x_label, default_cfg["label_font"])
        plt.ylabel("Calc Time [ms]", default_cfg["label_font"])
        plt.legend(loc="best", prop=default_cfg["legend_font"])
        fig.tight_layout(pad=default_cfg["pad"])
        plt.savefig(
            path_state_fmt, format=default_cfg["img_fmt"], bbox_inches="tight"
        )
        plt.close()
        # plot constraint value
        if self.constrained_env:
            for j in range(constrain_dim):
                path_constraint_fmt = os.path.join(
                    self.save_path,
                    "Constrain-{}.{}".format(j + 1, default_cfg["img_fmt"]),
                )
                fig, ax = plt.subplots(
                    figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"]
                )

                # save reward data to csv
                constrain_data = pd.DataFrame(data=[c[:, j] for c in constrain_list])
                constrain_data.to_csv(
                    os.path.join(self.save_path, "Constrain-{}.csv".format(j + 1)),
                    encoding="gbk",
                )

                for i in range(policy_num):
                    legend = (
                        self.legend_list[i]
                        if len(self.legend_list) == policy_num
                        else self.algorithm_list[i]
                    )
                    sns.lineplot(
                        x=step_list[i],
                        y=constrain_list[i][:, j],
                        label="{}".format(legend),
                    )
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel(x_label, default_cfg["label_font"])
                plt.ylabel("Constrain-{}".format(j + 1), default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(
                    path_constraint_fmt,
                    format=default_cfg["img_fmt"],
                    bbox_inches="tight",
                )
                plt.close()

        # plot error with opt
        if self.use_opt:
            # reward error
            path_reward_error_fmt = os.path.join(
                self.save_path, "Reward error.{}".format(default_cfg["img_fmt"])
            )
            fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

            # save reward error data to csv
            reward_error_list = []
            for r in reward_list:
                end = min(len(r), len(reward_list[-1]))
                reward_error_list.append(r[:end] - reward_list[-1][:end])
            reward_error_data = pd.DataFrame(data=reward_error_list)
            reward_error_data.to_csv(
                os.path.join(self.save_path, "Reward error.csv"), encoding="gbk"
            )

            for i in range(policy_num - 1):
                legend = (
                    self.legend_list[i]
                    if len(self.legend_list) == policy_num
                    else self.algorithm_list[i]
                )
                sns.lineplot(
                    x=step_list[i][:len(reward_error_list[i])],
                    y=reward_error_list[i], label="{}".format(legend)
                )
            plt.tick_params(labelsize=default_cfg["tick_size"])
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
            plt.xlabel(x_label, default_cfg["label_font"])
            plt.ylabel("Reward error", default_cfg["label_font"])
            plt.legend(loc="best", prop=default_cfg["legend_font"])
            fig.tight_layout(pad=default_cfg["pad"])
            plt.savefig(
                path_reward_error_fmt,
                format=default_cfg["img_fmt"],
                bbox_inches="tight",
            )
            plt.close()

            # action error
            action_error_list = []
            for a in action_list:
                end = min(len(a), len(action_list[-1]))
                action_error_list.append(a[:end] - action_list[-1][:end])
            for j in range(action_dim):
                path_action_error_fmt = os.path.join(
                    self.save_path,
                    "Action-{} error.{}".format(j + 1, default_cfg["img_fmt"]),
                )
                fig, ax = plt.subplots(
                    figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"]
                )
                for i in range(policy_num - 1):
                    legend = (
                        self.legend_list[i]
                        if len(self.legend_list) == policy_num
                        else self.algorithm_list[i]
                    )
                    sns.lineplot(
                        x=step_list[i][:len(action_error_list[i])],
                        y=action_error_list[i][:, j],
                        label="{}".format(legend),
                    )
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel(x_label, default_cfg["label_font"])
                plt.ylabel("Action-{} error".format(j + 1), default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(
                    path_action_error_fmt,
                    format=default_cfg["img_fmt"],
                    bbox_inches="tight",
                )
                plt.close()

                # save action error data to csv
                action_error_data = pd.DataFrame(data=[a[:, j] for a in action_error_list])
                action_error_data.to_csv(
                    os.path.join(self.save_path, "Action-{} error.csv".format(j + 1)),
                    encoding="gbk",
                )

            # state error
            state_error_list = []
            for s in state_list:
                end = min(len(s), len(state_list[-1]))
                state_error_list.append(s[:end] - state_list[-1][:end])
            for j in range(state_dim):
                path_state_error_fmt = os.path.join(
                    self.save_path,
                    "State-{} error.{}".format(j + 1, default_cfg["img_fmt"]),
                )
                fig, ax = plt.subplots(
                    figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"]
                )
                for i in range(policy_num - 1):
                    legend = (
                        self.legend_list[i]
                        if len(self.legend_list) == policy_num
                        else self.algorithm_list[i]
                    )
                    sns.lineplot(
                        x=step_list[i][:len(state_error_list[i])],
                        y=state_error_list[i][:, j],
                        label="{}".format(legend),
                    )
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel(x_label, default_cfg["label_font"])
                plt.ylabel("State-{} error".format(j + 1), default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(
                    path_state_error_fmt,
                    format=default_cfg["img_fmt"],
                    bbox_inches="tight",
                )
                plt.close()

                # save state data to csv
                state_error_data = pd.DataFrame(data=[s[:, j] for s in state_error_list])
                state_error_data.to_csv(
                    os.path.join(self.save_path, "State-{} error.csv".format(j + 1)),
                    encoding="gbk",
                )

            # compute relative error with opt
            error_result = {}
            for i in range(policy_num - 1):
                legend = (
                    self.legend_list[i]
                    if len(self.legend_list) == policy_num
                    else "Policy-{}".format(i + 1)
                )
                end = min(len(action_list[i]), len(action_list[-1]))
                error_result.update({legend: {}})
                # action error
                for j in range(action_dim):
                    action_error = {}
                    error_list = np.abs(
                        action_list[i][:end, j] - action_list[-1][:end, j]
                    ) / (
                        np.max(action_list[-1][:end, j])
                        - np.min(action_list[-1][:end, j])
                    )
                    action_error["Max_error"] = "{:.2f}%".format(max(error_list) * 100)
                    action_error["Mean_error"] = "{:.2f}%".format(
                        sum(error_list) / len(error_list) * 100
                    )
                    error_result[legend].update(
                        {"Action-{}".format(j + 1): action_error}
                    )
                # state error
                for j in range(state_dim):
                    state_error = {}
                    error_list = np.abs(
                        state_list[i][:end, j] - state_list[-1][:end, j]
                    ) / (
                        np.max(state_list[-1][:end, j])
                        - np.min(state_list[-1][:end, j])
                    )
                    state_error["Max_error"] = "{:.2f}%".format(max(error_list) * 100)
                    state_error["Mean_error"] = "{:.2f}%".format(
                        sum(error_list) / len(error_list) * 100
                    )
                    error_result[legend].update({"State-{}".format(j + 1): state_error})

            for i in range(self.policy_num):
                legend = (
                    self.legend_list[i]
                    if len(self.legend_list) == policy_num
                    else "Policy-{}".format(i + 1)
                )
                policy_result = pd.DataFrame(data=error_result[legend])
                policy_result.to_excel(os.path.join(self.save_path, "Error-result.xlsx"), legend)
            error_result_data = pd.DataFrame(data=error_result)
            pd.set_option("display.max_columns", None)
            pd.set_option("display.max_rows", None)
            for key, value in error_result_data.items():
                print("===========================================================")
                print("GOPS: Policy {}".format(key))
                for key, value in value.items():
                    print(key, value)

    @staticmethod
    def __load_args(log_policy_dir: str):
        json_path = os.path.join(log_policy_dir, "config.json")
        parser = argparse.ArgumentParser()
        args_dict = vars(parser.parse_args())
        args = get_args_from_json(json_path, args_dict)
        return args

    def __load_all_args(self):
        for i in range(self.policy_num):
            log_policy_dir = self.log_policy_dir_list[i]
            args = self.__load_args(log_policy_dir)
            args['vector_env_num'] = None
            args['gym2gymnasium'] = False
            self.args_list.append(args)
            env_id = args["env_id"]
            self.env_id_list.append(env_id)
            self.algorithm_list.append(args["algorithm"])

    def __load_env(self, use_opt: bool = False):
        if use_opt:
            env = create_env(**self.args)
        else:
            env_args = {
                **self.args,
                "obs_noise_type": self.obs_noise_type,
                "obs_noise_data": self.obs_noise_data,
                "action_noise_type": self.action_noise_type,
                "action_noise_data": self.action_noise_data,
            }
            env = create_env(**env_args)
        if self.save_render:
            video_path = os.path.join(self.save_path, "videos")
            if use_opt:
                name_prefix = "{}_video".format(self.opt_args["opt_controller_type"])
            else:
                name_prefix = "{}_video".format(self.args["algorithm"])
            env = wrappers.RecordVideo(env, video_path, name_prefix=name_prefix)
        self.args["action_high_limit"] = env.action_space.high
        self.args["action_low_limit"] = env.action_space.low
        return env

    def __load_policy(self, log_policy_dir: str, trained_policy_iteration: str):
        # Create policy
        networks = create_approx_contrainer(**self.args)

        # Load trained policy
        log_path = log_policy_dir + "/apprfunc/apprfunc_{}.pkl".format(
            trained_policy_iteration
        )
        networks.load_state_dict(torch.load(log_path))
        return networks

    def __convert_format(self, origin_data_list: list):
        data_list = copy(origin_data_list)
        for i in range(len(origin_data_list)):
            if isinstance(origin_data_list[i], list) or isinstance(
                origin_data_list[i], np.ndarray
            ):
                data_list[i] = self.__convert_format(origin_data_list[i])
            else:
                data_list[i] = "{:.2g}".format(origin_data_list[i])
        return data_list

    def __run_data(self):
        for i in range(self.policy_num):
            log_policy_dir = self.log_policy_dir_list[i]
            trained_policy_iteration = self.trained_policy_iteration_list[i]

            self.args = self.args_list[i]
            print("===========================================================")
            print("*** Begin to run policy {} ***".format(i + 1))
            env = self.__load_env()
            if hasattr(env, "set_mode"):
                env.set_mode("test")

            if hasattr(env, "train_space") and hasattr(env, "work_space"):
                print("Train space: ")
                print(self.__convert_format(env.train_space))
                print("Work space: ")
                print(self.__convert_format(env.work_space))
            networks = self.__load_policy(log_policy_dir, trained_policy_iteration)

            # Run policy
            eval_dict, tracking_dict = self.run_an_episode(
                env, networks, self.init_info, is_opt=False, render=False
            )
            print("Successfully run policy {}".format(i + 1))
            print("===========================================================\n")
            # mp4 to gif
            self.eval_list.append(eval_dict)
            self.tracking_list.append(tracking_dict)

        if self.use_opt:
            if self.load_opt_path is not None:
                eval_dict_opt = np.load(
                    os.path.join(self.load_opt_path, "eval_dict_opt.npy"),
                    allow_pickle=True).item()
                tracking_dict_opt = np.load(
                    os.path.join(self.load_opt_path, "tracking_dict_opt.npy"),
                    allow_pickle=True).item()
                print("Successfully load an optimal controller result!")
                print("===========================================================\n")
            else:
                self.args = self.args_list[self.policy_num - 1]
                print("GOPS: Use an optimal controller")
                env = self.__load_env(use_opt=True)
                print("The environment for opt")
                if hasattr(env, "set_mode"):
                    env.set_mode("test")

                assert (
                    self.opt_args is not None
                ), "Choose to use optimal controller, but the opt_args is None."

                if self.opt_args["opt_controller_type"] == "OPT":
                    assert (
                        env.has_optimal_controller
                    ), "The environment has no theoretical optimal controller."
                    opt_controller = env.control_policy
                elif self.opt_args["opt_controller_type"] == "MPC":
                    if self.opt_args["use_MPC_for_general_env"] == True:
                        self.args_list[self.policy_num - 1]["env"] = env
                        from gops.sys_simulator.opt_controller_for_gen_env import OptController
                    else:
                        from gops.sys_simulator.opt_controller import OptController
                    model = create_env_model(**self.args_list[self.policy_num - 1], mask_at_done=False)
                    opt_args = self.opt_args.copy()
                    opt_args.pop("opt_controller_type")
                    opt_args.pop("use_MPC_for_general_env")
                    opt_controller = OptController(model, **opt_args,)
                else:
                    raise ValueError(
                        "The optimal controller type should be either 'OPT' or 'MPC'."
                    )

                eval_dict_opt, tracking_dict_opt = self.run_an_episode(
                    env, opt_controller, self.init_info, is_opt=True, render=False
                )
                print("Successfully run an optimal controller!")
                print("===========================================================\n")

            if self.opt_args["opt_controller_type"] == "OPT":
                legend = "OPT"
            elif self.opt_args["opt_controller_type"] == "MPC":
                legend = "MPC-" + str(self.opt_args["num_pred_step"])
                if (
                    "use_terminal_cost" not in self.opt_args.keys()
                    or self.opt_args["use_terminal_cost"] == False
                ):
                    legend += " (w/o TC)"
                else:
                    legend += " (w/ TC)"
            self.legend_list.append(legend)

            if self.save_opt:
                np.save(os.path.join(self.save_path, "eval_dict_opt.npy"), eval_dict_opt)
                np.save(os.path.join(self.save_path, "tracking_dict_opt.npy"), tracking_dict_opt)

            self.eval_list.append(eval_dict_opt)
            if self.is_tracking:
                self.tracking_list.append(tracking_dict_opt)

    def __action_noise(self, action: np.ndarray) -> np.ndarray:
        if self.action_noise_type is None:
            return action
        elif self.action_noise_type == "normal":
            return action + np.random.normal(
                loc=self.action_noise_data[0], scale=self.action_noise_data[1]
            )
        elif self.action_noise_type == "uniform":
            return action + np.random.uniform(
                low=self.action_noise_data[0], high=self.action_noise_data[1]
            )

    def __save_mp4_as_gif(self):
        if self.save_render:
            videos_path = os.path.join(self.save_path, "videos")

            videos_list = [i for i in glob.glob(os.path.join(videos_path, "*.mp4"))]
            for v in videos_list:
                mp4togif(v)

    def get_n_verify_env_id(self):
        env_id = self.env_id_list[0]
        for i, eid in enumerate(self.env_id_list):
            assert (
                env_id == eid
            ), "GOPS: policy {} is not trained in the same environment".format(i)
        return env_id

    def run(self):
        self.__run_data()
        self.__save_mp4_as_gif()
        self.draw()

class PlanningBaseBenchmark:
    def __init__(self,
                 config=None):

        self.config = self.default_config()
        if not (config is None):
            self.config.update(config)

        self.env = None # wrapped environment
        # self.metrics = {}
        # self.initial_environment()
        # data for plot

    @classmethod
    def default_config(cls) -> dict:
        """
        :return: a configuration dict
        """
        return {
            "max_steps": 100,
            "random_seed": 666,
            "offscreen_rendering": False,
            "save_video": True,
            "video_root": './videos/',
            "video_name": 'benchmark.mp4'
        }

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def initial_environment(self):
        '''
        根据给定的config，初始化环境self.env
        '''
        pass

    @abstractmethod
    def test(self, spider_planner):
        '''
        给定一个planner，在设置好的环境里面开一遍，返回config中指定的metrics
        '''
        pass

    @abstractmethod
    def update_metrics(self, *args, **kwargs):
        pass

    @abstractmethod
    def visualize_plan(self, *args, **kwargs):
        '''
        把规划结果画在仿真器渲染的画面上
        '''
        pass

class PlanningRunner(PlanningBaseBenchmark):
    def __init__(self,
                 log_policy_dir_list: list,
                 trained_policy_iteration_list: list,
                 base_planner1: str = None,
                 base_planner2: str = None,
                 opt_args: Optional[dict] = None,
                 controller: str = None,
                 save_render: bool = False,
                 plot_range: list = None,
                 is_init_info: bool = False,
                 init_info: dict = None,
                 legend_list: list = None,
                 constrained_env: bool = False,
                 use_dist: bool = False,
                 dt: float = None,
                 load_opt_path: Optional[str] = None,
                 save_opt: bool = True,
                 is_tracking: bool = False,
                 obs_noise_type: str = None,
                 obs_noise_data: list = None,
                 action_noise_type: str = None,
                 action_noise_data: list = None,
                 render_args=None):
        super(PlanningRunner, self).__init__(render_args)
        self.base_planner1 = base_planner1
        self.base_planner2 = base_planner2
        self.controller_name = controller
        self.log_policy_dir_list = [
            os.path.join(gops_path, d) for d in log_policy_dir_list
        ]
        self.policy_num = len(self.log_policy_dir_list)
        self.trained_policy_iteration_list = trained_policy_iteration_list
        self.save_render = save_render
        self.render_args = render_args
        self.args = None
        self.plot_range = plot_range
        if is_init_info:
            self.init_info = init_info
        else:
            self.init_info = {}
        self.legend_list = legend_list
        self.constrained_env = constrained_env
        self.use_dist = use_dist
        self.dt = dt

        if base_planner1 == "MPCPlanner":
            assert load_opt_path is not None or opt_args is not None
            self.load_opt_path = load_opt_path
            self.opt_args = opt_args
            if isinstance(self.opt_args, dict) and \
                    "use_MPC_for_general_env" not in self.opt_args.keys():
                self.opt_args["use_MPC_for_general_env"] = False
            self.save_opt = save_opt

        self.constrained_env = constrained_env
        self.use_dist = use_dist
        self.is_tracking = is_tracking
        self.dt = dt
        self.policy_num = len(self.log_policy_dir_list)
        self.obs_noise_type = obs_noise_type
        self.obs_noise_data = obs_noise_data
        self.action_noise_type = action_noise_type
        self.action_noise_data = action_noise_data
        self.ref_state_num = 0
        # data for plot
        self.args_list = []
        self.eval_list = []
        self.env_id_list = []
        self.algorithm_list = []
        self.tracking_list = []

        self.__load_all_args()
        self.env_id = self.get_n_verify_env_id()
        # save path
        path = os.path.join(os.path.dirname(__file__), "..", "..", "figures")
        path = os.path.abspath(path)

        algs_name = ""
        for item in self.algorithm_list:
            algs_name = algs_name + item + "-"
        self.save_path = os.path.join(
            path,
            algs_name+"-""Multi_base_planner-" + self.env_id,
            datetime.datetime.now().strftime("%y%m%d-%H%M%S"),
        )
        os.makedirs(self.save_path, exist_ok=True)

        # 用于轨迹评价的
        self._evaluation = self.render_args["evaluation"]
        # 输入终点信息，判断轨迹是否走到目标点
        self._destination_x_range = (250, 260)
        self._destination_y_range = (-10, 10)
        if self.render_args["evaluation"]:
            self.init_metric_evaluators()

    @staticmethod
    def __load_args(log_policy_dir: str):
        json_path = os.path.join(log_policy_dir, "config.json")
        parser = argparse.ArgumentParser()
        args_dict = vars(parser.parse_args())
        args = get_args_from_json(json_path, args_dict)
        return args

    def __load_all_args(self):
        for i in range(self.policy_num):
            log_policy_dir = self.log_policy_dir_list[i]
            args = self.__load_args(log_policy_dir)
            args['vector_env_num'] = None
            args['gym2gymnasium'] = False
            self.args_list.append(args)
            env_id = args["env_id"]
            self.env_id_list.append(env_id)
            self.algorithm_list.append(args["algorithm"])

    def __load_env(self, use_opt: bool = False, is_base_planner: bool = False):
        if use_opt and is_base_planner:
            env = create_env(**self.args)
        else:
            env_args = {
                **self.args,
                "obs_noise_type": self.obs_noise_type,
                "obs_noise_data": self.obs_noise_data,
                "action_noise_type": self.action_noise_type,
                "action_noise_data": self.action_noise_data,
            }
            env = create_env(**env_args)
        if self.save_render:
            video_path = os.path.join(self.save_path, "videos")
            if use_opt and is_base_planner:
                name_prefix = "{}_video".format(self.base_planner1)
            elif not is_base_planner:
                name_prefix = "{}_video".format(self.args["algorithm"])
            else:
                name_prefix = "{}_video".format(self.base_planner2)
            env = wrappers.RecordVideo(env, video_path, name_prefix=name_prefix)
        # self.args["action_high_limit"] = self.args['action_high_limit']#env.action_space.high
        # self.args["action_low_limit"] = env.action_space.low
        return env

    def __load_policy(self, log_policy_dir: str, trained_policy_iteration: str):
        # Create policy
        networks = create_approx_contrainer(**self.args)

        # Load trained policy
        log_path = log_policy_dir + "/apprfunc/apprfunc_{}.pkl".format(
            trained_policy_iteration
        )
        networks.load_state_dict(torch.load(log_path))
        return networks

    def __convert_format(self, origin_data_list: list):
        data_list = copy(origin_data_list)
        for i in range(len(origin_data_list)):
            if isinstance(origin_data_list[i], list) or isinstance(
                    origin_data_list[i], np.ndarray
            ):
                data_list[i] = self.__convert_format(origin_data_list[i])
            else:
                data_list[i] = "{:.2g}".format(origin_data_list[i])
        return data_list

    def __action_noise(self, action: np.ndarray) -> np.ndarray:
        if self.action_noise_type is None:
            return action
        elif self.action_noise_type == "normal":
            return action + np.random.normal(
                loc=self.action_noise_data[0], scale=self.action_noise_data[1]
            )
        elif self.action_noise_type == "uniform":
            return action + np.random.uniform(
                low=self.action_noise_data[0], high=self.action_noise_data[1]
            )

    def __save_mp4_as_gif(self):
        if self.save_render:
            videos_path = os.path.join(self.save_path, "videos")

            videos_list = [i for i in glob.glob(os.path.join(videos_path, "*.mp4"))]
            for v in videos_list:
                mp4togif(v)

    def get_n_verify_env_id(self):
        env_id = self.env_id_list[0]
        for i, eid in enumerate(self.env_id_list):
            assert (
                    env_id == eid
            ), "GOPS: policy {} is not trained in the same environment".format(i)
        return env_id

    def init_metric_evaluators(self):
        from gops.utils.planner_benchmark.interface.metrics_collection import (MetricCombiner, CompletionMetric, CollisionRateMetric,
                                                         TTCMetric, SpeedMetric, JerkMetric, StuckMetric)
        self.metric_evaluators = MetricCombiner(
            CompletionMetric(self._destination_x_range, self._destination_y_range, self._max_duration),
            CollisionRateMetric(self.render_args["ego_veh_length"], self.render_args["ego_veh_width"]),
            TTCMetric(self.render_args["ego_veh_length"] / 2),
            SpeedMetric(),
            JerkMetric(delta_t=0.1),  # 这里要改成和planner的dt一致
            StuckMetric(0.2),
        )

    @property
    def metrics(self):
        if self.render_args["evaluation"]:
            return self.metric_evaluators.get_result()
        else:
            return {}

    def run(self):
        self.__run_data()
        self.__save_mp4_as_gif()
        self.draw()

    def __run_data(self):
        for i in range(self.policy_num):
            log_policy_dir = self.log_policy_dir_list[i]
            trained_policy_iteration = self.trained_policy_iteration_list[i]

            self.args = self.args_list[i]
            print("===========================================================")
            print("*** Begin to run policy {} ***".format(i + 1))
            env = self.__load_env()
            if hasattr(env, "set_mode"):
                env.set_mode("test")

            if hasattr(env, "train_space") and hasattr(env, "work_space"):
                print("Train space: ")
                print(self.__convert_format(env.train_space))
                print("Work space: ")
                print(self.__convert_format(env.work_space))
            networks = self.__load_policy(log_policy_dir, trained_policy_iteration)

            # Run policy
            eval_dict, tracking_dict = self.run_an_episode(
                env, networks,networks, self.init_info, is_opt=False, render=False, is_base_planner=False
            )
            print("Successfully run policy {}".format(i + 1))
            print("===========================================================\n")
            # mp4 to gif
            self.eval_list.append(eval_dict)
            self.tracking_list.append(tracking_dict)

        if self.base_planner1 == "MPCPlanner":
            # load main planner--MPCplanner
            if self.load_opt_path is not None:
                eval_dict_opt = np.load(
                    os.path.join(self.load_opt_path, "eval_dict_opt.npy"),
                    allow_pickle=True).item()
                tracking_dict_opt = np.load(
                    os.path.join(self.load_opt_path, "tracking_dict_opt.npy"),
                    allow_pickle=True).item()
                print("Successfully load an optimal controller result!")
                print("===========================================================\n")
            else:
                self.args = self.args_list[0]
                env = self.__load_env(use_opt=True, is_base_planner=True)
                print("The environment for planner")
                from gops.sys_simulator.opt_planner import OptPlanner
                model = create_env_model(**self.args_list[0], mask_at_done=False)
                opt_args = self.opt_args.copy()
                opt_args.pop("opt_controller_type")
                opt_args.pop("use_MPC_for_general_env")
                base_planner1 = OptPlanner(model, **opt_args, )
                print("MPCPlanner for planning")
                if self.opt_args["opt_controller_type"] == "OPT":
                    legend = "OPT"
                elif self.opt_args["opt_controller_type"] == "MPC":
                    legend = "MPCPlanner-" + str(self.opt_args["num_pred_step"])
                    if (
                            "use_terminal_cost" not in self.opt_args.keys()
                            or self.opt_args["use_terminal_cost"] == False
                    ):
                        legend += " (w/o TC)"
                    else:
                        legend += " (w/ TC)"
                self.legend_list.append(legend)
            controller = base_planner1
            eval_dict_opt, tracking_dict_opt = self.run_an_episode(
                env, base_planner1, controller, self.init_info, is_opt=True, render=False, is_base_planner=True
            )
            self.eval_list.append(eval_dict_opt)
            self.tracking_list.append(tracking_dict_opt)
            print("Successfully run main_planner")
            print("===========================================================\n")
            if self.save_opt:
                np.save(os.path.join(self.save_path, "eval_dict_opt.npy"), eval_dict_opt)
                np.save(os.path.join(self.save_path, "tracking_dict_opt.npy"), tracking_dict_opt)

            self.eval_list.append(eval_dict_opt)
            if self.is_tracking:
                self.tracking_list.append(tracking_dict_opt)
        if self.base_planner2 == "LatticePlanner" or self.base_planner2 == "BezierPlanner":
            base_planner_list = []
            env = self.__load_env(is_base_planner=True)
            # initialize the baseline planner
            print("GOPS: Loading Baseline Planner")
            if self.base_planner2 == "LatticePlanner":
                from gops.utils.planner_benchmark.planner_zoo import LatticePlanner
                base_planner = LatticePlanner({
                    "steps": self.args['pre_horizon'],
                    "dt": self.dt,
                    "end_s_candidates": (20, 40, 60),
                    "end_l_candidates": (-3.5, 0, 3.5),
                })
                legend = "LatticePlanner"
                self.legend_list.append(legend)
                print("LatticePlanner for planning")
            elif self.base_planner2 == "BezierPlanner":
                from gops.utils.planner_benchmark.planner_zoo import BezierPlanner
                base_planner = BezierPlanner({
                    "steps": self.args['pre_horizon'],
                    "dt": self.dt,
                    "end_s_candidates": (20, 40, 60),
                    "end_l_candidates": (-3.5, 0, 3.5),
                })
                legend = "BezierPlanner"
                self.legend_list.append(legend)
                print("BezierPlanner for planning")
            else:
                raise ValueError(
                    "Please specify the baseline  planner"
                )
            base_planner_list.append(base_planner)

            # initialize the controller
            if self.controller_name == "IDMController":
                from gops.utils.planner_benchmark.control.IDMController import IDMController
                controller = IDMController()
                print("IDMController for control")
            elif self.controller_name == "SimpleController":
                from gops.utils.planner_benchmark.control.SimpleController import SimpleController
                controller = SimpleController()
                print("SimpleController for control")
            elif self.controller_name == "MPCController":
                if self.main_planner == "MPCPlanner":
                    controller = base_planner1
                    print("MPC for plannning and control")
                else:
                    if self.load_opt_path is not None:
                        eval_dict_opt = np.load(
                            os.path.join(self.load_opt_path, "eval_dict_opt.npy"),
                            allow_pickle=True).item()
                        tracking_dict_opt = np.load(
                            os.path.join(self.load_opt_path, "tracking_dict_opt.npy"),
                            allow_pickle=True).item()
                        print("Successfully load an optimal controller result!")
                        print("===========================================================\n")
                    else:
                        self.args = self.args_list[0]
                        print("GOPS: Use an optimal controller")
                        env = self.__load_env(use_opt=True)
                        print("The environment for opt")
                        if hasattr(env, "set_mode"):
                            env.set_mode("test")

                        assert (
                                self.opt_args is not None
                        ), "Choose to use optimal controller, but the opt_args is None."

                        if self.opt_args["opt_controller_type"] == "OPT":
                            assert (
                                env.has_optimal_controller
                            ), "The environment has no theoretical optimal controller."
                            controller = env.control_policy
                        elif self.opt_args["opt_controller_type"] == "MPC":
                            if self.opt_args["use_MPC_for_general_env"] == True:
                                self.args_list[0]["env"] = env
                                from gops.sys_simulator.opt_controller_for_gen_env import OptController
                            else:
                                from gops.sys_simulator.opt_controller import OptController
                            model = create_env_model(**self.args_list[0], mask_at_done=False)
                            opt_args = self.opt_args.copy()
                            opt_args.pop("opt_controller_type")
                            opt_args.pop("use_MPC_for_general_env")
                            controller = OptController(model, **opt_args, )
                            print("MPCController for control")
                        else:
                            raise ValueError(
                                "The optimal controller type should be either 'OPT' or 'MPC'."
                            )
            else:
                raise ValueError(
                    "Please specify the controller"
                )
            for i in range(len(base_planner_list)):
                base_planner = base_planner_list[i]
                eval_dict, tracking_dict = self.run_an_episode(
                    env, base_planner, controller, self.init_info, is_opt=False, render=True, is_base_planner=True
                )
                self.eval_list.append(eval_dict)
                self.tracking_list.append(tracking_dict)
                print("Successfully run base_planner {}".format(i + 1))
                print("===========================================================\n")

    def run_an_episode(
            self,
            env: Any,
            planner: Any,
            controller: Any,
            init_info: dict,
            is_opt: bool,
            render: bool = True,
            is_base_planner=False,
    ) -> Tuple[dict, dict]:
        if is_opt==False and self.save_render:
            if self.render_args["evaluation"]:
                self.init_metric_evaluators()
        state_list = []
        action_list = []
        reward_list = []
        constrain_list = []
        obs_list = []
        step = 0
        step_list = []
        calctime_list = []
        info_list = [init_info]

        obs, info = env.reset(**init_info)
        state = env.state
        print("Initial robot state: ")
        print(self.__convert_format(np.asarray(state.robot_state)))
        # plot tracking
        state_with_ref_error = {}
        done = False
        info.update({"TimeLimit.truncated": False})
        # if not is_opt:
        planner.set_local_map(env.local_map)
        import matplotlib.pyplot as plt
        import gops.utils.planner_benchmark.visualize as vis
        if self.save_render:
            vis.figure(figsize=(15, 6), dpi=300)
            if is_opt == False and self.render_args["snapshot"] :
                video_name = type(planner).__name__ + '.mp4' if self.render_args["video_name"] is None else self.render_args[
                    "video_name"]
            else:
                video_name = type(planner).__name__ + '.mp4'
            videos_path = os.path.join(self.save_path, "videos")
            snapshot = vis.SnapShot(True, 20, record_video=self.render_args["save_video"],
                                    video_path=videos_path + '/' + video_name)
        while not (done or info["TimeLimit.truncated"]):
            # 地图信息更新
            if is_opt==False:
                if self.render_args["map_frequency"] == 0:
                    local_map = None
                else:
                    if step % self.render_args["map_frequency"] == 0:
                        local_map = deepcopy(env.local_map)
                    else:
                        local_map = None
            # local_map = deepcopy(env.local_map)
            print("step:", step + 1)
            state_list.append(state.robot_state)
            obs_list.append(obs)
            if is_opt and is_base_planner:
                if isinstance(env.unwrapped, Env):
                    time_start = time.time()
                    action = planner(state)
                    calc_time = time.time() - time_start
                else:
                    time_start = time.time()
                    action = planner(obs, info)
                    calc_time = time.time() - time_start
                if self.use_dist:
                    action = np.hstack((action, env.dist_func(step * env.tau)))
                if self.constrained_env:
                    constrain_list.append(info["constraint"])
                if self.is_tracking:
                    reference = get_reference_from_info(info)
                    state_num = len(reference)
                    self.ref_state_num = sum(x is not None for x in reference)
                    if step == 0:
                        for i in range(state_num):
                            if reference[i] is not None:
                                state_with_ref_error["state-{}".format(i)] = []
                                state_with_ref_error["ref-{}".format(i)] = []
                                state_with_ref_error["state-{}-error".format(i)] = []

                    robot_state = get_robot_state_from_info(info)
                    for i in range(state_num):
                        if reference[i] is not None:
                            state_with_ref_error["state-{}".format(i)].append(robot_state[i])
                            state_with_ref_error["ref-{}".format(i)].append(reference[i])
                            state_with_ref_error["state-{}-error".format(i)].append(
                                reference[i] - robot_state[i]
                            )
                # from gops.utils.planner_benchmark.elements.trajectory import FrenetTrajectory
                # traj = FrenetTrajectory(self.args['pre_horizon'], self.dt)
                # state = env.vehicle_dynamics.f_xu(state.robot_state, action[0, :], self.dt)
                # state_full = np.empty((self.args['pre_horizon'], env.state_dim))
                # state_full[0, :] = state.robot_state
                #
                # for i in range(1, self.args['pre_horizon']):
                #     state = env.vehicle_dynamics.f_xu(state, action[i, :], self.dt)
                #     state_full[i, :] = state
                # traj.x, traj.y, traj.heading, traj.v = state_full[:, 0].tolist(), state_full[:, 1].tolist(), state_full[:, 2].tolist(), state_full[:, 3].tolist()
            elif not is_opt and not is_base_planner:
                time_start = time.time()
                action = self.compute_action(obs, controller)
                action = self.__action_noise(action)
                calc_time = time.time() - time_start
            else:
                traj = planner.plan(deepcopy(env.ego_veh_state), deepcopy(env.obstaclesBox),
                                           local_map)  # , self.local_map
                if traj is None:
                    raise RuntimeError("DummyBenchmark receives no feasible trajectory!")
                # 评估
                if self.render_args["evaluation"]:
                    self.metric_evaluators.evaluate(env.ego_veh_state, env.obstaclesBox, local_map)
                if self.controller_name == "IDMController":
                    ref = np.stack((np.array(traj.x), np.array(traj.y)), axis=1)
                    current_pose = state.robot_state[:3]
                    current_speed = state.robot_state[3]
                    front_veh_speed, front_veh_dist = 0, 0
                    time_start = time.time()
                    action = controller.get_control(ref, front_veh_speed, front_veh_dist, traj.v[0], current_pose, current_speed)
                    calc_time = time.time() - time_start
                elif self.controller_name == "SimpleController":
                    ref = np.stack((np.array(traj.x), np.array(traj.y)), axis=1)
                    current_pose = state.robot_state[:3]
                    current_speed = state.robot_state[3]
                    time_start = time.time()
                    action_one = controller.get_control(ref, traj.v[0], current_pose, current_speed)
                    calc_time = time.time() - time_start
                    action = np.zeros((self.args["pre_horizon"], 2))+action_one
                # elif self.controller_name == "MPCController":
                #     if is_opt:
                #         if isinstance(env.unwrapped, Env):
                #             time_start = time.time()
                #             action = controller(state)
                #             calc_time = time.time() - time_start
                #         else:
                #             time_start = time.time()
                #             action = controller(obs, info)
                #             calc_time = time.time() - time_start
                #     else:
                #         time_start = time.time()
                #         action = self.compute_action(obs, controller)
                #         action = self.__action_noise(action)
                #         calc_time = time.time() - time_start
                #     if self.use_dist:
                #         action = np.hstack((action, env.dist_func(step * env.tau)))
                #     if self.constrained_env:
                #         constrain_list.append(info["constraint"])
                #     if self.is_tracking:
                #         reference = get_reference_from_info(info)
                #         state_num = len(reference)
                #         self.ref_state_num = sum(x is not None for x in reference)
                #         if step == 0:
                #             for i in range(state_num):
                #                 if reference[i] is not None:
                #                     state_with_ref_error["state-{}".format(i)] = []
                #                     state_with_ref_error["ref-{}".format(i)] = []
                #                     state_with_ref_error["state-{}-error".format(i)] = []
                #
                #         robot_state = get_robot_state_from_info(info)
                #         for i in range(state_num):
                #             if reference[i] is not None:
                #                 state_with_ref_error["state-{}".format(i)].append(robot_state[i])
                #                 state_with_ref_error["ref-{}".format(i)].append(reference[i])
                #                 state_with_ref_error["state-{}-error".format(i)].append(
                #                     reference[i] - robot_state[i]
                #                 )
                #     action = action[0, :]
            # env.conduct_trajectory(traj)
            next_obs, reward, done, info = env.step(action)
            if is_opt or (not is_opt and not is_base_planner):
                from gops.utils.planner_benchmark.elements.trajectory import FrenetTrajectory
                traj = FrenetTrajectory(self.args['pre_horizon'], self.dt)
                traj.x, traj.y, traj.heading, traj.v= env.state_full[:, 0],env.state_full[:, 1],env.state_full[:, 2],env.state_full[:, 3]
            if self.save_render:
                plt.cla()
                env.visualize(traj)
                plt.pause(0.001)
                if self.render_args["snapshot"]:
                    snapshot.snap(plt.gca())

            if is_opt or (not is_opt and not is_base_planner):
                action_list.append(info.get("raw_action", action[0, :]))#
            else:
                action_list.append(info.get("raw_action", action))  #
            step_list.append(step)
            reward_list.append(reward)
            info_list.append(info)
            calctime_list.append(calc_time*1000)
            obs = next_obs
            state = env.state
            step = step + 1

            if "TimeLimit.truncated" not in info.keys():
                info["TimeLimit.truncated"] = False
            # Draw environment animation

            if render:
                env.render()
        # if is_opt == False and self.save_render_baseplanner:
        #     plt.close()
        #     if self.render_args["snapshot"]:
        #         snapshot.print(3, 2, figsize=(15, 6))
        #         snapshot.save()
        #         plt.show()

        if self.save_render and self.render_args["snapshot"]:
            plt.close()
            snapshot.print(3, 2, figsize=(12, 9))
            path_snapshot = os.path.join(
                self.save_path, type(planner).__name__ + 'Shot.png'
            )
            plt.savefig(path_snapshot)
        eval_dict = {
            "reward_list": reward_list,
            "action_list": action_list,
            "state_list": state_list,
            "step_list": step_list,
            "obs_list": obs_list,
            "info_list": info_list,
            "calctime_list": calctime_list
        }
        if self.constrained_env:
            eval_dict.update(
                {"constrain_list": constrain_list, }
            )

        if self.is_tracking:
            tracking_dict = state_with_ref_error
        else:
            tracking_dict = {}

        return eval_dict, tracking_dict

    def compute_action(self, obs: np.ndarray, networks: Any) -> np.ndarray:
        batch_obs = torch.from_numpy(np.expand_dims(obs, axis=0).astype("float32"))
        logits = networks.policy(batch_obs)
        action_distribution = networks.create_action_distributions(logits)
        action = action_distribution.mode()
        action = action.detach().numpy()[0]
        return action

    @staticmethod
    def get_environment_presets(ego_length=5.0, ego_width=2.0, racetrack="curve"):
        from spider.interface.BaseInterface import DummyInterface
        return DummyInterface.get_environment_presets(ego_length, ego_width, racetrack)

    def update_metrics(self, *args, **kwargs):
        pass

    def draw(self):
        fig_size = (
            default_cfg["fig_size"],
            default_cfg["fig_size"],
        )
        action_dim = self.eval_list[0]["action_list"][0].shape[0]
        state_dim = self.eval_list[0]["state_list"][0].shape[0]
        if self.constrained_env:
            constrain_dim = self.eval_list[0]["constrain_list"][0].shape[0]
        policy_num = 2
        if self.base_planner1 == "MPCPlanner":
            if self.opt_args["opt_controller_type"] == "OPT":
                legend = "OPT"
            elif self.opt_args["opt_controller_type"] == "MPC":
                legend = "MPC-" + str(self.opt_args["num_pred_step"])
                if (
                        "use_terminal_cost" not in self.opt_args.keys()
                        or self.opt_args["use_terminal_cost"] is False
                ):
                    legend += " (w/o TC)"
                else:
                    legend += " (w/ TC)"

        if self.base_planner2 == "LatticePlanner":
            legend = self.base_planner2
        self.algorithm_list.append(legend)
        # Create initial list
        reward_list = []
        action_list = []
        state_list = []
        step_list = []
        state_ref_error_list = []
        constrain_list = []
        calctime_list = []
        # Put data into list
        for i in range(policy_num):
            reward_list.append(np.array(self.eval_list[i]["reward_list"]))
            action_list.append(np.array(self.eval_list[i]["action_list"]))
            state_list.append(np.array(self.eval_list[i]["state_list"]))
            step_list.append(np.array(self.eval_list[i]["step_list"]))
            calctime_list.append(np.array(self.eval_list[i]["calctime_list"]))
            if self.constrained_env:
                constrain_list.append(np.stack(self.eval_list[i]["constrain_list"]))
            if self.is_tracking:
                state_ref_error_list.append(self.tracking_list[i])

        if self.plot_range is None:
            pass
        elif len(self.plot_range) == 2:

            for i in range(policy_num):
                start_range = self.plot_range[0]
                end_range = min(self.plot_range[1], reward_list[i].shape[0])

                reward_list[i] = reward_list[i][start_range:end_range]
                action_list[i] = action_list[i][start_range:end_range]
                state_list[i] = state_list[i][start_range:end_range]
                step_list[i] = step_list[i][start_range:end_range]
                if self.constrained_env:
                    constrain_list[i] = constrain_list[i][start_range:end_range]
                if self.is_tracking:
                    for key, value in self.tracking_list[i].items():
                        self.tracking_list[i][key] = value[start_range:end_range]
        else:
            raise NotImplementedError("Figure range is wrong")

        if self.dt is None:
            x_label = "Time step"
        else:
            step_list = [s * self.dt for s in step_list]
            x_label = "Time (s)"

        # Plot reward
        path_reward_fmt = os.path.join(
            self.save_path, "Reward.{}".format(default_cfg["img_fmt"])
        )
        fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

        # save reward data to csv
        reward_data = pd.DataFrame(data=reward_list[0])
        reward_data.to_csv(os.path.join(self.save_path, "Reward.csv"), encoding="gbk")

        for i in range(policy_num):
            legend = (
                self.legend_list[i]
                if len(self.legend_list) == policy_num
                else self.algorithm_list[i]
            )
            sns.lineplot(x=step_list[i], y=reward_list[i], label="{}".format(legend))
        plt.tick_params(labelsize=default_cfg["tick_size"])
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
        plt.xlabel(x_label, default_cfg["label_font"])
        plt.ylabel("Reward", default_cfg["label_font"])
        plt.legend(loc="best", prop=default_cfg["legend_font"])
        fig.tight_layout(pad=default_cfg["pad"])
        plt.savefig(path_reward_fmt, format=default_cfg["img_fmt"], bbox_inches="tight")
        plt.close()

        # plot action
        for j in range(action_dim):
            path_action_fmt = os.path.join(
                self.save_path, "Action-{}.{}".format(j + 1, default_cfg["img_fmt"])
            )
            fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

            # save action data to csv
            action_data = pd.DataFrame(data=[a[:, j] for a in action_list])
            action_data.to_csv(
                os.path.join(self.save_path, "Action-{}.csv".format(j + 1)),
                encoding="gbk",
            )

            for i in range(policy_num):
                legend = (
                    self.legend_list[i]
                    if len(self.legend_list) == policy_num
                    else self.algorithm_list[i]
                )
                sns.lineplot(
                    x=step_list[i], y=action_list[i][:, j], label="{}".format(legend)
                )
            plt.tick_params(labelsize=default_cfg["tick_size"])
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
            plt.xlabel(x_label, default_cfg["label_font"])
            plt.ylabel("Action-{}".format(j + 1), default_cfg["label_font"])
            plt.legend(loc="best", prop=default_cfg["legend_font"])
            fig.tight_layout(pad=default_cfg["pad"])
            plt.savefig(
                path_action_fmt, format=default_cfg["img_fmt"], bbox_inches="tight"
            )
            plt.close()

        # plot state
        for j in range(state_dim):
            path_state_fmt = os.path.join(
                self.save_path, "State-{}.{}".format(j + 1, default_cfg["img_fmt"])
            )
            fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

            # save state data to csv
            state_data = pd.DataFrame(data=[s[:, j] for s in state_list])
            state_data.to_csv(
                os.path.join(self.save_path, "State-{}.csv".format(j + 1)),
                encoding="gbk",
            )

            for i in range(policy_num):
                legend = (
                    self.legend_list[i]
                    if len(self.legend_list) == policy_num
                    else self.algorithm_list[i]
                )
                sns.lineplot(
                    x=step_list[i], y=state_list[i][:, j], label="{}".format(legend)
                )
            plt.tick_params(labelsize=default_cfg["tick_size"])
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
            plt.xlabel(x_label, default_cfg["label_font"])
            plt.ylabel("State-{}".format(j + 1), default_cfg["label_font"])
            plt.legend(loc="best", prop=default_cfg["legend_font"])
            fig.tight_layout(pad=default_cfg["pad"])
            plt.savefig(
                path_state_fmt, format=default_cfg["img_fmt"], bbox_inches="tight"
            )
            plt.close()
        # plot state x-y
        path_state_fmt = os.path.join(
            self.save_path, "State-xy.{}".format(default_cfg["img_fmt"])
        )
        fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

        for i in range(policy_num):
            legend = (
                self.legend_list[i]
                if len(self.legend_list) == policy_num
                else self.algorithm_list[i]
            )
            sns.lineplot(
                x=state_list[i][:, 0], y=state_list[i][:, 1], label="{}".format(legend)
            )
        plt.tick_params(labelsize=default_cfg["tick_size"])
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
        plt.xlabel("x", default_cfg["label_font"])
        plt.ylabel("y", default_cfg["label_font"])
        plt.legend(loc="best", prop=default_cfg["legend_font"])
        fig.tight_layout(pad=default_cfg["pad"])
        plt.savefig(
            path_state_fmt, format=default_cfg["img_fmt"], bbox_inches="tight"
        )
        plt.close()
        # plot tracking
        if self.is_tracking:
            # find index of the longest trajectory
            traj_lens = [len(r) for r in reward_list]
            longest_traj_index = np.argmax(traj_lens)

            for j in range(self.ref_state_num):

                # plot state and ref
                path_tracking_state_fmt = os.path.join(
                    self.save_path, "Ref - State - {}.{}".format(j + 1, default_cfg["img_fmt"])
                )
                fig, ax = plt.subplots(
                    figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"]
                )
                # save tracking state data to csv
                tracking_state_data = []
                for i in range(policy_num):
                    legend = (
                        self.legend_list[i]
                        if len(self.legend_list) == policy_num
                        else self.algorithm_list[i]
                    )
                    sns.lineplot(
                        x=step_list[i],
                        y=state_ref_error_list[i]["state-{}".format(j)],
                        label="{}".format(legend),
                    )
                    tracking_state_data.append(
                        state_ref_error_list[i]["state-{}".format(j)]
                    )
                sns.lineplot(
                    x=step_list[longest_traj_index],
                    y=state_ref_error_list[longest_traj_index]["ref-{}".format(j)],
                    label="ref",
                )
                tracking_state_data.append(state_ref_error_list[longest_traj_index]["ref-{}".format(j)])
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel(x_label, default_cfg["label_font"])
                plt.ylabel("State-{}".format(j + 1), default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(
                    path_tracking_state_fmt,
                    format=default_cfg["img_fmt"],
                    bbox_inches="tight",
                )
                plt.close()

                tracking_state_data = pd.DataFrame(data=tracking_state_data)
                tracking_state_data.to_csv(
                    os.path.join(self.save_path, "State-{}.csv".format(j + 1)),
                    encoding="gbk",
                )

                # plot state-ref error
                path_tracking_error_fmt = os.path.join(
                    self.save_path,
                    "Ref - State - Error{}.{}".format(j + 1, default_cfg["img_fmt"]),
                )
                fig, ax = plt.subplots(
                    figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"]
                )
                # save tracking error data to csv
                tracking_error_data = []
                for i in range(policy_num):
                    legend = (
                        self.legend_list[i]
                        if len(self.legend_list) == policy_num
                        else self.algorithm_list[i]
                    )
                    sns.lineplot(
                        x=step_list[i],
                        y=state_ref_error_list[i]["state-{}-error".format(j)],
                        label="{}".format(legend),
                    )
                    tracking_error_data.append(
                        state_ref_error_list[i]["state-{}-error".format(j)]
                    )
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel(x_label, default_cfg["label_font"])
                plt.ylabel("Ref$-$State-Error{}".format(j + 1), default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(
                    path_tracking_error_fmt,
                    format=default_cfg["img_fmt"],
                    bbox_inches="tight",
                )
                plt.close()

                tracking_error_data = pd.DataFrame(data=tracking_error_data)
                tracking_error_data.to_csv(
                    os.path.join(self.save_path, "Ref-State-Error{}.csv".format(j + 1)),
                    encoding="gbk",
                )

        # plot calculation time
        path_state_fmt = os.path.join(
            self.save_path, "Calc time.{}".format(default_cfg["img_fmt"])
        )
        fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

        # save state data to csv
        state_data = pd.DataFrame(data=[s[:] for s in calctime_list])
        state_data.to_csv(
            os.path.join(self.save_path, "Calc time.csv"),
            encoding="gbk",
        )

        for i in range(policy_num):
            legend = (
                self.legend_list[i]
                if len(self.legend_list) == policy_num
                else self.algorithm_list[i]
            )
            sns.lineplot(
                x=step_list[i], y=calctime_list[i][:], label="{}".format(legend)
            )
        plt.tick_params(labelsize=default_cfg["tick_size"])
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
        plt.xlabel(x_label, default_cfg["label_font"])
        plt.ylabel("Calc Time [ms]", default_cfg["label_font"])
        plt.legend(loc="best", prop=default_cfg["legend_font"])
        fig.tight_layout(pad=default_cfg["pad"])
        plt.savefig(
            path_state_fmt, format=default_cfg["img_fmt"], bbox_inches="tight"
        )
        plt.close()


        # plot constraint value
        if self.constrained_env:
            for j in range(constrain_dim):
                path_constraint_fmt = os.path.join(
                    self.save_path,
                    "Constrain-{}.{}".format(j + 1, default_cfg["img_fmt"]),
                )
                fig, ax = plt.subplots(
                    figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"]
                )

                # save reward data to csv
                constrain_data = pd.DataFrame(data=[c[:, j] for c in constrain_list])
                constrain_data.to_csv(
                    os.path.join(self.save_path, "Constrain-{}.csv".format(j + 1)),
                    encoding="gbk",
                )

                for i in range(policy_num):
                    legend = (
                        self.legend_list[i]
                        if len(self.legend_list) == policy_num
                        else self.algorithm_list[i]
                    )
                    sns.lineplot(
                        x=step_list[i],
                        y=constrain_list[i][:, j],
                        label="{}".format(legend),
                    )
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel(x_label, default_cfg["label_font"])
                plt.ylabel("Constrain-{}".format(j + 1), default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(
                    path_constraint_fmt,
                    format=default_cfg["img_fmt"],
                    bbox_inches="tight",
                )
                plt.close()

        # plot error with opt
        if self.main_planner:
            # reward error
            path_reward_error_fmt = os.path.join(
                self.save_path, "Reward error.{}".format(default_cfg["img_fmt"])
            )
            fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

            # save reward error data to csv
            reward_error_list = []
            for r in reward_list:
                end = min(len(r), len(reward_list[-1]))
                reward_error_list.append(r[:end] - reward_list[-1][:end])
            reward_error_data = pd.DataFrame(data=reward_error_list)
            reward_error_data.to_csv(
                os.path.join(self.save_path, "Reward error.csv"), encoding="gbk"
            )

            for i in range(policy_num - 1):
                legend = (
                    self.legend_list[i]
                    if len(self.legend_list) == policy_num
                    else self.algorithm_list[i]
                )
                sns.lineplot(
                    x=step_list[i][:len(reward_error_list[i])],
                    y=reward_error_list[i], label="{}".format(legend)
                )
            plt.tick_params(labelsize=default_cfg["tick_size"])
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
            plt.xlabel(x_label, default_cfg["label_font"])
            plt.ylabel("Reward error", default_cfg["label_font"])
            plt.legend(loc="best", prop=default_cfg["legend_font"])
            fig.tight_layout(pad=default_cfg["pad"])
            plt.savefig(
                path_reward_error_fmt,
                format=default_cfg["img_fmt"],
                bbox_inches="tight",
            )
            plt.close()

            # action error
            action_error_list = []
            for a in action_list:
                end = min(len(a), len(action_list[-1]))
                action_error_list.append(a[:end] - action_list[-1][:end])
            for j in range(action_dim):
                path_action_error_fmt = os.path.join(
                    self.save_path,
                    "Action-{} error.{}".format(j + 1, default_cfg["img_fmt"]),
                )
                fig, ax = plt.subplots(
                    figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"]
                )
                for i in range(policy_num - 1):
                    legend = (
                        self.legend_list[i]
                        if len(self.legend_list) == policy_num
                        else self.algorithm_list[i]
                    )
                    sns.lineplot(
                        x=step_list[i][:len(action_error_list[i])],
                        y=action_error_list[i][:, j],
                        label="{}".format(legend),
                    )
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel(x_label, default_cfg["label_font"])
                plt.ylabel("Action-{} error".format(j + 1), default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(
                    path_action_error_fmt,
                    format=default_cfg["img_fmt"],
                    bbox_inches="tight",
                )
                plt.close()

                # save action error data to csv
                action_error_data = pd.DataFrame(data=[a[:, j] for a in action_error_list])
                action_error_data.to_csv(
                    os.path.join(self.save_path, "Action-{} error.csv".format(j + 1)),
                    encoding="gbk",
                )

            # state error
            state_error_list = []
            for s in state_list:
                end = min(len(s), len(state_list[-1]))
                state_error_list.append(s[:end] - state_list[-1][:end])
            for j in range(state_dim):
                path_state_error_fmt = os.path.join(
                    self.save_path,
                    "State-{} error.{}".format(j + 1, default_cfg["img_fmt"]),
                )
                fig, ax = plt.subplots(
                    figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"]
                )
                for i in range(policy_num - 1):
                    legend = (
                        self.legend_list[i]
                        if len(self.legend_list) == policy_num
                        else self.algorithm_list[i]
                    )
                    sns.lineplot(
                        x=step_list[i][:len(state_error_list[i])],
                        y=state_error_list[i][:, j],
                        label="{}".format(legend),
                    )
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel(x_label, default_cfg["label_font"])
                plt.ylabel("State-{} error".format(j + 1), default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(
                    path_state_error_fmt,
                    format=default_cfg["img_fmt"],
                    bbox_inches="tight",
                )
                plt.close()

                # save state data to csv
                state_error_data = pd.DataFrame(data=[s[:, j] for s in state_error_list])
                state_error_data.to_csv(
                    os.path.join(self.save_path, "State-{} error.csv".format(j + 1)),
                    encoding="gbk",
                )

            # compute relative error with opt
            error_result = {}
            for i in range(policy_num - 1):
                legend = (
                    self.legend_list[i]
                    if len(self.legend_list) == policy_num
                    else "Policy-{}".format(i + 1)
                )
                end = min(len(action_list[i]), len(action_list[-1]))
                error_result.update({legend: {}})
                # action error
                for j in range(action_dim):
                    action_error = {}
                    error_list = np.abs(
                        action_list[i][:end, j] - action_list[-1][:end, j]
                    ) / (
                                         np.max(action_list[-1][:end, j])
                                         - np.min(action_list[-1][:end, j])
                                 )
                    action_error["Max_error"] = "{:.2f}%".format(max(error_list) * 100)
                    action_error["Mean_error"] = "{:.2f}%".format(
                        sum(error_list) / len(error_list) * 100
                    )
                    error_result[legend].update(
                        {"Action-{}".format(j + 1): action_error}
                    )
                # state error
                for j in range(state_dim):
                    state_error = {}
                    error_list = np.abs(
                        state_list[i][:end, j] - state_list[-1][:end, j]
                    ) / (
                                         np.max(state_list[-1][:end, j])
                                         - np.min(state_list[-1][:end, j])
                                 )
                    state_error["Max_error"] = "{:.2f}%".format(max(error_list) * 100)
                    state_error["Mean_error"] = "{:.2f}%".format(
                        sum(error_list) / len(error_list) * 100
                    )
                    error_result[legend].update({"State-{}".format(j + 1): state_error})

            # for i in range(self.policy_num):
            #     legend = (
            #         self.legend_list[i]
            #         if len(self.legend_list) == policy_num
            #         else "Policy-{}".format(i + 1)
            #     )
            #     policy_result = pd.DataFrame(data=error_result[legend])
            #     policy_result.to_excel(os.path.join(self.save_path, "Error-result.xlsx"), legend)
            error_result_data = pd.DataFrame(data=error_result)
            pd.set_option("display.max_columns", None)
            pd.set_option("display.max_rows", None)
            for key, value in error_result_data.items():
                print("===========================================================")
                print("GOPS: Policy {}".format(key))
                for key, value in value.items():
                    print(key, value)

class PlanningMPCRunner(PlanningBaseBenchmark):
    def __init__(self,
                 log_policy_dir_list: list,
                 base_planner: str = None,
                 controller: str = None,
                 save_render: bool = False,
                 plot_range: list = None,
                 is_init_info: bool = False,
                 init_info: dict = None,
                 legend_list: list = None,
                 constrained_env: bool = False,
                 use_dist: bool = False,
                 dt: float = None,
                 main_planner: str = None,
                 load_opt_path: Optional[str] = None,
                 opt_args: Optional[dict] = None,
                 save_opt: bool = True,
                 is_tracking: bool = False,
                 obs_noise_type: str = None,
                 obs_noise_data: list = None,
                 action_noise_type: str = None,
                 action_noise_data: list = None,
                 render_args=None):
        super(PlanningMPCRunner, self).__init__(render_args)

        # from gops.utils.planner_benchmark.interface.BaseInterface import DummyInterface
        self.base_planner = base_planner
        self.controller_name = controller
        self.log_policy_dir_list = [
            os.path.join(gops_path, d) for d in log_policy_dir_list
        ]

        self.save_render = save_render
        self.render_args = render_args
        self.args = None
        self.plot_range = plot_range
        if is_init_info:
            self.init_info = init_info
        else:
            self.init_info = {}
        self.legend_list = legend_list
        self.constrained_env = constrained_env
        self.use_dist = use_dist
        self.dt = dt

        self.main_planner = main_planner
        if main_planner=="MPCPlanner":
            assert load_opt_path is not None or opt_args is not None
            self.load_opt_path = load_opt_path
            self.opt_args = opt_args
            if isinstance(self.opt_args, dict) and \
                    "use_MPC_for_general_env" not in self.opt_args.keys():
                self.opt_args["use_MPC_for_general_env"] = False
            self.save_opt = save_opt

        self.constrained_env = constrained_env
        self.use_dist = use_dist
        self.is_tracking = is_tracking
        self.dt = dt
        self.policy_num = len(self.log_policy_dir_list)
        self.obs_noise_type = obs_noise_type
        self.obs_noise_data = obs_noise_data
        self.action_noise_type = action_noise_type
        self.action_noise_data = action_noise_data
        self.ref_state_num = 0
        # data for plot
        self.args_list = []
        self.eval_list = []
        self.env_id_list = []
        self.algorithm_list = []
        self.tracking_list = []

        self.__load_all_args()
        self.env_id = self.get_n_verify_env_id()
        # save path
        path = os.path.join(os.path.dirname(__file__), "..", "..", "figures")
        path = os.path.abspath(path)

        self.save_path = os.path.join(
            path,
            main_planner+"-"+base_planner + self.env_id,
            datetime.datetime.now().strftime("%y%m%d-%H%M%S"),
        )
        os.makedirs(self.save_path, exist_ok=True)

        # 用于轨迹评价的
        self._evaluation = self.render_args["evaluation"]
        # 输入终点信息，判断轨迹是否走到目标点
        self._destination_x_range = (250, 260)
        self._destination_y_range = (-10, 10)
        if self.render_args["evaluation"]:
            self.init_metric_evaluators()
    @staticmethod
    def __load_args(log_policy_dir: str):
        json_path = os.path.join(log_policy_dir, "config.json")
        parser = argparse.ArgumentParser()
        args_dict = vars(parser.parse_args())
        args = get_args_from_json(json_path, args_dict)
        return args

    def __load_all_args(self):
        log_policy_dir = self.log_policy_dir_list[0]
        args = self.__load_args(log_policy_dir)
        args['vector_env_num'] = None
        args['gym2gymnasium'] = False
        self.args_list.append(args)
        env_id = args["env_id"]
        self.env_id_list.append(env_id)
        self.algorithm_list.append(args["algorithm"])

    def __load_env(self, use_opt: bool = False):
        if use_opt:
            env = create_env(**self.args)
        else:
            env_args = {
                **self.args,
                "obs_noise_type": self.obs_noise_type,
                "obs_noise_data": self.obs_noise_data,
                "action_noise_type": self.action_noise_type,
                "action_noise_data": self.action_noise_data,
            }
            env = create_env(**env_args)
        if self.save_render:
            video_path = os.path.join(self.save_path, "videos")
            if use_opt:
                name_prefix = "{}_video".format(self.opt_args["opt_controller_type"])
            else:
                name_prefix = "{}_video".format(self.render_args["planner_type"])
            env = wrappers.RecordVideo(env, video_path, name_prefix=name_prefix)
        # self.args["action_high_limit"] = self.args['action_high_limit']#env.action_space.high
        # self.args["action_low_limit"] = env.action_space.low
        return env

    def __convert_format(self, origin_data_list: list):
        data_list = copy(origin_data_list)
        for i in range(len(origin_data_list)):
            if isinstance(origin_data_list[i], list) or isinstance(
                    origin_data_list[i], np.ndarray
            ):
                data_list[i] = self.__convert_format(origin_data_list[i])
            else:
                data_list[i] = "{:.2g}".format(origin_data_list[i])
        return data_list

    def __save_mp4_as_gif(self):
        if self.save_render:
            videos_path = os.path.join(self.save_path, "videos")

            videos_list = [i for i in glob.glob(os.path.join(videos_path, "*.mp4"))]
            for v in videos_list:
                mp4togif(v)

    def get_n_verify_env_id(self):
        env_id = self.env_id_list[0]
        for i, eid in enumerate(self.env_id_list):
            assert (
                    env_id == eid
            ), "GOPS: policy {} is not trained in the same environment".format(i)
        return env_id

    def init_metric_evaluators(self):
        from gops.utils.planner_benchmark.interface.metrics_collection import (MetricCombiner, CompletionMetric, CollisionRateMetric,
                                                         TTCMetric, SpeedMetric, JerkMetric, StuckMetric)
        self.metric_evaluators = MetricCombiner(
            CompletionMetric(self._destination_x_range, self._destination_y_range, self._max_duration),
            CollisionRateMetric(self.render_args["ego_veh_length"], self.render_args["ego_veh_width"]),
            TTCMetric(self.render_args["ego_veh_length"] / 2),
            SpeedMetric(),
            JerkMetric(delta_t=0.1),  # 这里要改成和planner的dt一致
            StuckMetric(0.2),
        )

    @property
    def metrics(self):
        if self.render_args["evaluation"]:
            return self.metric_evaluators.get_result()
        else:
            return {}


    def run(self):
        self.__run_data()
        self.__save_mp4_as_gif()
        self.draw()


    def __run_data(self):
        if self.main_planner == "MPCPlanner":
            # load main planner--MPCplanner
            if self.load_opt_path is not None:
                eval_dict_opt = np.load(
                    os.path.join(self.load_opt_path, "eval_dict_opt.npy"),
                    allow_pickle=True).item()
                tracking_dict_opt = np.load(
                    os.path.join(self.load_opt_path, "tracking_dict_opt.npy"),
                    allow_pickle=True).item()
                print("Successfully load an optimal controller result!")
                print("===========================================================\n")
            else:
                self.args = self.args_list[0]
                env = self.__load_env(use_opt=True)
                print("The environment for planner")
                from gops.sys_simulator.opt_planner import OptPlanner
                model = create_env_model(**self.args_list[0], mask_at_done=False)
                opt_args = self.opt_args.copy()
                opt_args.pop("opt_controller_type")
                opt_args.pop("use_MPC_for_general_env")
                main_planner = OptPlanner(model, **opt_args, )
                print("MPCPlanner for planning")
                if self.opt_args["opt_controller_type"] == "OPT":
                    legend = "OPT"
                elif self.opt_args["opt_controller_type"] == "MPC":
                    legend = "MPCPlanner-" + str(self.opt_args["num_pred_step"])
                    if (
                            "use_terminal_cost" not in self.opt_args.keys()
                            or self.opt_args["use_terminal_cost"] == False
                    ):
                        legend += " (w/o TC)"
                    else:
                        legend += " (w/ TC)"
                self.legend_list.append(legend)
            controller = main_planner
            eval_dict_opt, tracking_dict_opt = self.run_an_episode(
                env, main_planner, controller, self.init_info, is_opt=True, render=False
            )
            self.eval_list.append(eval_dict_opt)
            self.tracking_list.append(tracking_dict_opt)
            print("Successfully run main_planner")
            print("===========================================================\n")
            if self.save_opt:
                np.save(os.path.join(self.save_path, "eval_dict_opt.npy"), eval_dict_opt)
                np.save(os.path.join(self.save_path, "tracking_dict_opt.npy"), tracking_dict_opt)

            self.eval_list.append(eval_dict_opt)
            if self.is_tracking:
                self.tracking_list.append(tracking_dict_opt)
        if self.base_planner == "LatticePlanner" or self.base_planner == "BezierPlanner":
            base_planner_list = []
            self.args = self.args_list[0]
            env = self.__load_env()

            # initialize the baseline planner
            print("GOPS: Loading Baseline Planner")
            if self.base_planner == "LatticePlanner":
                from gops.utils.planner_benchmark.planner_zoo import LatticePlanner
                base_planner = LatticePlanner({
                    "steps": self.args['pre_horizon'],
                    "dt": self.dt,
                    "end_s_candidates": (20, 40, 60),
                    "end_l_candidates": (-3.5, 0, 3.5),
                })
                legend = "LatticePlanner"
                self.legend_list.append(legend)
                print("LatticePlanner for planning")
            elif self.base_planner == "BezierPlanner":
                from gops.utils.planner_benchmark.planner_zoo import BezierPlanner
                base_planner = BezierPlanner({
                    "steps": self.args['pre_horizon'],
                    "dt": self.dt,
                    "end_s_candidates": (20, 40, 60),
                    "end_l_candidates": (-3.5, 0, 3.5),
                })
                legend = "BezierPlanner"
                self.legend_list.append(legend)
                print("BezierPlanner for planning")
            else:
                raise ValueError(
                    "Please specify the baseline  planner"
                )
            base_planner_list.append(base_planner)

            # initialize the controller
            if self.controller_name == "IDMController":
                from gops.utils.planner_benchmark.control.IDMController import IDMController
                controller = IDMController()
                print("IDMController for control")
            elif self.controller_name == "SimpleController":
                from gops.utils.planner_benchmark.control.SimpleController import SimpleController
                controller = SimpleController()
                print("SimpleController for control")
            elif self.controller_name == "MPCController":
                if self.main_planner == "MPCPlanner":
                    controller = main_planner
                    print("MPC for plannning and control")
                else:
                    if self.load_opt_path is not None:
                        eval_dict_opt = np.load(
                            os.path.join(self.load_opt_path, "eval_dict_opt.npy"),
                            allow_pickle=True).item()
                        tracking_dict_opt = np.load(
                            os.path.join(self.load_opt_path, "tracking_dict_opt.npy"),
                            allow_pickle=True).item()
                        print("Successfully load an optimal controller result!")
                        print("===========================================================\n")
                    else:
                        self.args = self.args_list[0]
                        print("GOPS: Use an optimal controller")
                        env = self.__load_env(use_opt=True)
                        print("The environment for opt")
                        if hasattr(env, "set_mode"):
                            env.set_mode("test")

                        assert (
                                self.opt_args is not None
                        ), "Choose to use optimal controller, but the opt_args is None."

                        if self.opt_args["opt_controller_type"] == "OPT":
                            assert (
                                env.has_optimal_controller
                            ), "The environment has no theoretical optimal controller."
                            controller = env.control_policy
                        elif self.opt_args["opt_controller_type"] == "MPC":
                            if self.opt_args["use_MPC_for_general_env"] == True:
                                self.args_list[0]["env"] = env
                                from gops.sys_simulator.opt_controller_for_gen_env import OptController
                            else:
                                from gops.sys_simulator.opt_controller import OptController
                            model = create_env_model(**self.args_list[0], mask_at_done=False)
                            opt_args = self.opt_args.copy()
                            opt_args.pop("opt_controller_type")
                            opt_args.pop("use_MPC_for_general_env")
                            controller = OptController(model, **opt_args, )
                            print("MPCController for control")
                        else:
                            raise ValueError(
                                "The optimal controller type should be either 'OPT' or 'MPC'."
                            )
            else:
                raise ValueError(
                    "Please specify the controller"
                )

            for i in range(len(base_planner_list)):
                base_planner = base_planner_list[i]
                eval_dict, tracking_dict = self.run_an_episode(
                    env, base_planner, controller, self.init_info, is_opt=False, render=True
                )
                self.eval_list.append(eval_dict)
                self.tracking_list.append(tracking_dict)
                print("Successfully run base_planner {}".format(i + 1))
                print("===========================================================\n")

    def run_an_episode(
            self,
            env: Any,
            planner: Any,
            controller: Any,
            init_info: dict,
            is_opt: bool,
            render: bool = True,
    ) -> Tuple[dict, dict]:
        if is_opt==False and self.save_render:
            if self.render_args["evaluation"]:
                self.init_metric_evaluators()
        state_list = []
        action_list = []
        reward_list = []
        constrain_list = []
        obs_list = []
        step = 0
        step_list = []
        calctime_list = []
        info_list = [init_info]

        obs, info = env.reset(**init_info)
        state = env.state
        print("Initial robot state: ")
        print(self.__convert_format(np.asarray(state.robot_state)))
        # plot tracking
        state_with_ref_error = {}
        done = False
        info.update({"TimeLimit.truncated": False})
        # if not is_opt:
        planner.set_local_map(env.local_map)
        import matplotlib.pyplot as plt
        import gops.utils.planner_benchmark.visualize as vis
        if self.save_render:
            vis.figure(figsize=(15, 6), dpi=300)
            if is_opt == False and self.render_args["snapshot"] :
                video_name = type(planner).__name__ + '.mp4' if self.render_args["video_name"] is None else self.render_args[
                    "video_name"]
            else:
                video_name = type(planner).__name__ + '.mp4'
            videos_path = os.path.join(self.save_path, "videos")
            snapshot = vis.SnapShot(True, 20, record_video=self.render_args["save_video"],
                                    video_path=videos_path + '/' + video_name)
        while not (done or info["TimeLimit.truncated"]):
            # 地图信息更新
            if is_opt==False:
                if self.render_args["map_frequency"] == 0:
                    local_map = None
                else:
                    if step % self.render_args["map_frequency"] == 0:
                        local_map = deepcopy(env.local_map)
                    else:
                        local_map = None
            # local_map = deepcopy(env.local_map)
            print("step:", step + 1)
            state_list.append(state.robot_state)
            obs_list.append(obs)
            if is_opt:
                if isinstance(env.unwrapped, Env):
                    time_start = time.time()
                    action = planner(state)
                    calc_time = time.time() - time_start
                else:
                    time_start = time.time()
                    action = planner(obs, info)
                    calc_time = time.time() - time_start
                if self.use_dist:
                    action = np.hstack((action, env.dist_func(step * env.tau)))
                if self.constrained_env:
                    constrain_list.append(info["constraint"])
                if self.is_tracking:
                    reference = get_reference_from_info(info)
                    state_num = len(reference)
                    self.ref_state_num = sum(x is not None for x in reference)
                    if step == 0:
                        for i in range(state_num):
                            if reference[i] is not None:
                                state_with_ref_error["state-{}".format(i)] = []
                                state_with_ref_error["ref-{}".format(i)] = []
                                state_with_ref_error["state-{}-error".format(i)] = []

                    robot_state = get_robot_state_from_info(info)
                    for i in range(state_num):
                        if reference[i] is not None:
                            state_with_ref_error["state-{}".format(i)].append(robot_state[i])
                            state_with_ref_error["ref-{}".format(i)].append(reference[i])
                            state_with_ref_error["state-{}-error".format(i)].append(
                                reference[i] - robot_state[i]
                            )
                # from gops.utils.planner_benchmark.elements.trajectory import FrenetTrajectory
                # traj = FrenetTrajectory(self.args['pre_horizon'], self.dt)
                # state = env.vehicle_dynamics.f_xu(state.robot_state, action[0, :], self.dt)
                # state_full = np.empty((self.args['pre_horizon'], env.state_dim))
                # state_full[0, :] = state.robot_state
                #
                # for i in range(1, self.args['pre_horizon']):
                #     state = env.vehicle_dynamics.f_xu(state, action[i, :], self.dt)
                #     state_full[i, :] = state
                # traj.x, traj.y, traj.heading, traj.v = state_full[:, 0].tolist(), state_full[:, 1].tolist(), state_full[:, 2].tolist(), state_full[:, 3].tolist()
            else:
                traj = planner.plan(deepcopy(env.ego_veh_state), deepcopy(env.obstaclesBox),
                                           local_map)  # , self.local_map
                if traj is None:
                    raise RuntimeError("DummyBenchmark receives no feasible trajectory!")
                # 评估
                if self.render_args["evaluation"]:
                    self.metric_evaluators.evaluate(env.ego_veh_state, env.obstaclesBox, local_map)
                if self.controller_name == "IDMController":
                    ref = np.stack((np.array(traj.x), np.array(traj.y)), axis=1)
                    current_pose = state.robot_state[:3]
                    current_speed = state.robot_state[3]
                    front_veh_speed, front_veh_dist = 0, 0
                    time_start = time.time()
                    action = controller.get_control(ref, front_veh_speed, front_veh_dist, traj.v[0], current_pose, current_speed)
                    calc_time = time.time() - time_start
                elif self.controller_name == "SimpleController":
                    ref = np.stack((np.array(traj.x), np.array(traj.y)), axis=1)
                    current_pose = state.robot_state[:3]
                    current_speed = state.robot_state[3]
                    time_start = time.time()
                    action_one = controller.get_control(ref, traj.v[0], current_pose, current_speed)
                    calc_time = time.time() - time_start
                    action = np.zeros((self.args["pre_horizon"], 2))+action_one
                # elif self.controller_name == "MPCController":
                #     if is_opt:
                #         if isinstance(env.unwrapped, Env):
                #             time_start = time.time()
                #             action = controller(state)
                #             calc_time = time.time() - time_start
                #         else:
                #             time_start = time.time()
                #             action = controller(obs, info)
                #             calc_time = time.time() - time_start
                #     else:
                #         time_start = time.time()
                #         action = self.compute_action(obs, controller)
                #         action = self.__action_noise(action)
                #         calc_time = time.time() - time_start
                #     if self.use_dist:
                #         action = np.hstack((action, env.dist_func(step * env.tau)))
                #     if self.constrained_env:
                #         constrain_list.append(info["constraint"])
                #     if self.is_tracking:
                #         reference = get_reference_from_info(info)
                #         state_num = len(reference)
                #         self.ref_state_num = sum(x is not None for x in reference)
                #         if step == 0:
                #             for i in range(state_num):
                #                 if reference[i] is not None:
                #                     state_with_ref_error["state-{}".format(i)] = []
                #                     state_with_ref_error["ref-{}".format(i)] = []
                #                     state_with_ref_error["state-{}-error".format(i)] = []
                #
                #         robot_state = get_robot_state_from_info(info)
                #         for i in range(state_num):
                #             if reference[i] is not None:
                #                 state_with_ref_error["state-{}".format(i)].append(robot_state[i])
                #                 state_with_ref_error["ref-{}".format(i)].append(reference[i])
                #                 state_with_ref_error["state-{}-error".format(i)].append(
                #                     reference[i] - robot_state[i]
                #                 )
                #     action = action[0, :]
            # env.conduct_trajectory(traj)
            next_obs, reward, done, info = env.step(action)
            if is_opt:
                from gops.utils.planner_benchmark.elements.trajectory import FrenetTrajectory
                traj = FrenetTrajectory(self.args['pre_horizon'], self.dt)
                traj.x, traj.y, traj.heading, traj.v= env.state_full[:, 0],env.state_full[:, 1],env.state_full[:, 2],env.state_full[:, 3]
            if self.save_render:
                plt.cla()
                env.visualize(traj)
                plt.pause(0.001)
                if self.render_args["snapshot"]:
                    snapshot.snap(plt.gca())

            if is_opt:
                action_list.append(info.get("raw_action", action[0, :]))#
            else:
                action_list.append(info.get("raw_action", action))  #
            step_list.append(step)
            reward_list.append(reward)
            info_list.append(info)
            calctime_list.append(calc_time*1000)
            obs = next_obs
            state = env.state
            step = step + 1

            if "TimeLimit.truncated" not in info.keys():
                info["TimeLimit.truncated"] = False
            # Draw environment animation

            if render:
                env.render()
        # if is_opt == False and self.save_render_baseplanner:
        #     plt.close()
        #     if self.render_args["snapshot"]:
        #         snapshot.print(3, 2, figsize=(15, 6))
        #         snapshot.save()
        #         plt.show()
        if self.save_render and self.render_args["snapshot"]:
            plt.close()
            snapshot.print(3, 2, figsize=(12, 9))
            path_snapshot = os.path.join(
                self.save_path, type(planner).__name__ + 'Shot.png'
            )
            plt.savefig(path_snapshot)
        eval_dict = {
            "reward_list": reward_list,
            "action_list": action_list,
            "state_list": state_list,
            "step_list": step_list,
            "obs_list": obs_list,
            "info_list": info_list,
            "calctime_list": calctime_list
        }
        if self.constrained_env:
            eval_dict.update(
                {"constrain_list": constrain_list, }
            )

        if self.is_tracking:
            tracking_dict = state_with_ref_error
        else:
            tracking_dict = {}

        return eval_dict, tracking_dict


    @staticmethod
    def get_environment_presets(ego_length=5.0, ego_width=2.0, racetrack="curve"):
        from spider.interface.BaseInterface import DummyInterface
        return DummyInterface.get_environment_presets(ego_length, ego_width, racetrack)

    def update_metrics(self, *args, **kwargs):
        pass

    def draw(self):
        fig_size = (
            default_cfg["fig_size"],
            default_cfg["fig_size"],
        )
        action_dim = self.eval_list[0]["action_list"][0].shape[0]
        state_dim = self.eval_list[0]["state_list"][0].shape[0]
        if self.constrained_env:
            constrain_dim = self.eval_list[0]["constrain_list"][0].shape[0]
        policy_num = 2
        if self.main_planner == "MPCPlanner":
            if self.opt_args["opt_controller_type"] == "OPT":
                legend = "OPT"
            elif self.opt_args["opt_controller_type"] == "MPC":
                legend = "MPC-" + str(self.opt_args["num_pred_step"])
                if (
                        "use_terminal_cost" not in self.opt_args.keys()
                        or self.opt_args["use_terminal_cost"] is False
                ):
                    legend += " (w/o TC)"
                else:
                    legend += " (w/ TC)"

        if self.base_planner == "LatticePlanner":
            legend = self.base_planner
        self.algorithm_list.append(legend)
        # Create initial list
        reward_list = []
        action_list = []
        state_list = []
        step_list = []
        state_ref_error_list = []
        constrain_list = []
        calctime_list = []
        # Put data into list
        for i in range(policy_num):
            reward_list.append(np.array(self.eval_list[i]["reward_list"]))
            action_list.append(np.array(self.eval_list[i]["action_list"]))
            state_list.append(np.array(self.eval_list[i]["state_list"]))
            step_list.append(np.array(self.eval_list[i]["step_list"]))
            calctime_list.append(np.array(self.eval_list[i]["calctime_list"]))
            if self.constrained_env:
                constrain_list.append(np.stack(self.eval_list[i]["constrain_list"]))
            if self.is_tracking:
                state_ref_error_list.append(self.tracking_list[i])

        if self.plot_range is None:
            pass
        elif len(self.plot_range) == 2:

            for i in range(policy_num):
                start_range = self.plot_range[0]
                end_range = min(self.plot_range[1], reward_list[i].shape[0])

                reward_list[i] = reward_list[i][start_range:end_range]
                action_list[i] = action_list[i][start_range:end_range]
                state_list[i] = state_list[i][start_range:end_range]
                step_list[i] = step_list[i][start_range:end_range]
                if self.constrained_env:
                    constrain_list[i] = constrain_list[i][start_range:end_range]
                if self.is_tracking:
                    for key, value in self.tracking_list[i].items():
                        self.tracking_list[i][key] = value[start_range:end_range]
        else:
            raise NotImplementedError("Figure range is wrong")

        if self.dt is None:
            x_label = "Time step"
        else:
            step_list = [s * self.dt for s in step_list]
            x_label = "Time (s)"

        # Plot reward
        path_reward_fmt = os.path.join(
            self.save_path, "Reward.{}".format(default_cfg["img_fmt"])
        )
        fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

        # save reward data to csv
        reward_data = pd.DataFrame(data=reward_list[0])
        reward_data.to_csv(os.path.join(self.save_path, "Reward.csv"), encoding="gbk")

        for i in range(policy_num):
            legend = (
                self.legend_list[i]
                if len(self.legend_list) == policy_num
                else self.algorithm_list[i]
            )
            sns.lineplot(x=step_list[i], y=reward_list[i], label="{}".format(legend))
        plt.tick_params(labelsize=default_cfg["tick_size"])
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
        plt.xlabel(x_label, default_cfg["label_font"])
        plt.ylabel("Reward", default_cfg["label_font"])
        plt.legend(loc="best", prop=default_cfg["legend_font"])
        fig.tight_layout(pad=default_cfg["pad"])
        plt.savefig(path_reward_fmt, format=default_cfg["img_fmt"], bbox_inches="tight")
        plt.close()

        # plot action
        for j in range(action_dim):
            path_action_fmt = os.path.join(
                self.save_path, "Action-{}.{}".format(j + 1, default_cfg["img_fmt"])
            )
            fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

            # save action data to csv
            action_data = pd.DataFrame(data=[a[:, j] for a in action_list])
            action_data.to_csv(
                os.path.join(self.save_path, "Action-{}.csv".format(j + 1)),
                encoding="gbk",
            )

            for i in range(policy_num):
                legend = (
                    self.legend_list[i]
                    if len(self.legend_list) == policy_num
                    else self.algorithm_list[i]
                )
                sns.lineplot(
                    x=step_list[i], y=action_list[i][:, j], label="{}".format(legend)
                )
            plt.tick_params(labelsize=default_cfg["tick_size"])
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
            plt.xlabel(x_label, default_cfg["label_font"])
            plt.ylabel("Action-{}".format(j + 1), default_cfg["label_font"])
            plt.legend(loc="best", prop=default_cfg["legend_font"])
            fig.tight_layout(pad=default_cfg["pad"])
            plt.savefig(
                path_action_fmt, format=default_cfg["img_fmt"], bbox_inches="tight"
            )
            plt.close()

        # plot state
        for j in range(state_dim):
            path_state_fmt = os.path.join(
                self.save_path, "State-{}.{}".format(j + 1, default_cfg["img_fmt"])
            )
            fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

            # save state data to csv
            state_data = pd.DataFrame(data=[s[:, j] for s in state_list])
            state_data.to_csv(
                os.path.join(self.save_path, "State-{}.csv".format(j + 1)),
                encoding="gbk",
            )

            for i in range(policy_num):
                legend = (
                    self.legend_list[i]
                    if len(self.legend_list) == policy_num
                    else self.algorithm_list[i]
                )
                sns.lineplot(
                    x=step_list[i], y=state_list[i][:, j], label="{}".format(legend)
                )
            plt.tick_params(labelsize=default_cfg["tick_size"])
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
            plt.xlabel(x_label, default_cfg["label_font"])
            plt.ylabel("State-{}".format(j + 1), default_cfg["label_font"])
            plt.legend(loc="best", prop=default_cfg["legend_font"])
            fig.tight_layout(pad=default_cfg["pad"])
            plt.savefig(
                path_state_fmt, format=default_cfg["img_fmt"], bbox_inches="tight"
            )
            plt.close()
        # plot state x-y
        path_state_fmt = os.path.join(
            self.save_path, "State-xy.{}".format(default_cfg["img_fmt"])
        )
        fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

        for i in range(policy_num):
            legend = (
                self.legend_list[i]
                if len(self.legend_list) == policy_num
                else self.algorithm_list[i]
            )
            sns.lineplot(
                x=state_list[i][:, 0], y=state_list[i][:, 1], label="{}".format(legend)
            )
        plt.tick_params(labelsize=default_cfg["tick_size"])
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
        plt.xlabel("x", default_cfg["label_font"])
        plt.ylabel("y", default_cfg["label_font"])
        plt.legend(loc="best", prop=default_cfg["legend_font"])
        fig.tight_layout(pad=default_cfg["pad"])
        plt.savefig(
            path_state_fmt, format=default_cfg["img_fmt"], bbox_inches="tight"
        )
        plt.close()
        # plot tracking
        if self.is_tracking:
            # find index of the longest trajectory
            traj_lens = [len(r) for r in reward_list]
            longest_traj_index = np.argmax(traj_lens)

            for j in range(self.ref_state_num):

                # plot state and ref
                path_tracking_state_fmt = os.path.join(
                    self.save_path, "Ref - State - {}.{}".format(j + 1, default_cfg["img_fmt"])
                )
                fig, ax = plt.subplots(
                    figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"]
                )
                # save tracking state data to csv
                tracking_state_data = []
                for i in range(policy_num):
                    legend = (
                        self.legend_list[i]
                        if len(self.legend_list) == policy_num
                        else self.algorithm_list[i]
                    )
                    sns.lineplot(
                        x=step_list[i],
                        y=state_ref_error_list[i]["state-{}".format(j)],
                        label="{}".format(legend),
                    )
                    tracking_state_data.append(
                        state_ref_error_list[i]["state-{}".format(j)]
                    )
                sns.lineplot(
                    x=step_list[longest_traj_index],
                    y=state_ref_error_list[longest_traj_index]["ref-{}".format(j)],
                    label="ref",
                )
                tracking_state_data.append(state_ref_error_list[longest_traj_index]["ref-{}".format(j)])
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel(x_label, default_cfg["label_font"])
                plt.ylabel("State-{}".format(j + 1), default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(
                    path_tracking_state_fmt,
                    format=default_cfg["img_fmt"],
                    bbox_inches="tight",
                )
                plt.close()

                tracking_state_data = pd.DataFrame(data=tracking_state_data)
                tracking_state_data.to_csv(
                    os.path.join(self.save_path, "State-{}.csv".format(j + 1)),
                    encoding="gbk",
                )

                # plot state-ref error
                path_tracking_error_fmt = os.path.join(
                    self.save_path,
                    "Ref - State - Error{}.{}".format(j + 1, default_cfg["img_fmt"]),
                )
                fig, ax = plt.subplots(
                    figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"]
                )
                # save tracking error data to csv
                tracking_error_data = []
                for i in range(policy_num):
                    legend = (
                        self.legend_list[i]
                        if len(self.legend_list) == policy_num
                        else self.algorithm_list[i]
                    )
                    sns.lineplot(
                        x=step_list[i],
                        y=state_ref_error_list[i]["state-{}-error".format(j)],
                        label="{}".format(legend),
                    )
                    tracking_error_data.append(
                        state_ref_error_list[i]["state-{}-error".format(j)]
                    )
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel(x_label, default_cfg["label_font"])
                plt.ylabel("Ref$-$State-Error{}".format(j + 1), default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(
                    path_tracking_error_fmt,
                    format=default_cfg["img_fmt"],
                    bbox_inches="tight",
                )
                plt.close()

                tracking_error_data = pd.DataFrame(data=tracking_error_data)
                tracking_error_data.to_csv(
                    os.path.join(self.save_path, "Ref-State-Error{}.csv".format(j + 1)),
                    encoding="gbk",
                )

        # plot calculation time
        path_state_fmt = os.path.join(
            self.save_path, "Calc time.{}".format(default_cfg["img_fmt"])
        )
        fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

        # save state data to csv
        state_data = pd.DataFrame(data=[s[:] for s in calctime_list])
        state_data.to_csv(
            os.path.join(self.save_path, "Calc time.csv"),
            encoding="gbk",
        )

        for i in range(policy_num):
            legend = (
                self.legend_list[i]
                if len(self.legend_list) == policy_num
                else self.algorithm_list[i]
            )
            sns.lineplot(
                x=step_list[i], y=calctime_list[i][:], label="{}".format(legend)
            )
        plt.tick_params(labelsize=default_cfg["tick_size"])
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
        plt.xlabel(x_label, default_cfg["label_font"])
        plt.ylabel("Calc Time [ms]", default_cfg["label_font"])
        plt.legend(loc="best", prop=default_cfg["legend_font"])
        fig.tight_layout(pad=default_cfg["pad"])
        plt.savefig(
            path_state_fmt, format=default_cfg["img_fmt"], bbox_inches="tight"
        )
        plt.close()


        # plot constraint value
        if self.constrained_env:
            for j in range(constrain_dim):
                path_constraint_fmt = os.path.join(
                    self.save_path,
                    "Constrain-{}.{}".format(j + 1, default_cfg["img_fmt"]),
                )
                fig, ax = plt.subplots(
                    figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"]
                )

                # save reward data to csv
                constrain_data = pd.DataFrame(data=[c[:, j] for c in constrain_list])
                constrain_data.to_csv(
                    os.path.join(self.save_path, "Constrain-{}.csv".format(j + 1)),
                    encoding="gbk",
                )

                for i in range(policy_num):
                    legend = (
                        self.legend_list[i]
                        if len(self.legend_list) == policy_num
                        else self.algorithm_list[i]
                    )
                    sns.lineplot(
                        x=step_list[i],
                        y=constrain_list[i][:, j],
                        label="{}".format(legend),
                    )
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel(x_label, default_cfg["label_font"])
                plt.ylabel("Constrain-{}".format(j + 1), default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(
                    path_constraint_fmt,
                    format=default_cfg["img_fmt"],
                    bbox_inches="tight",
                )
                plt.close()

        # plot error with opt
        if self.main_planner:
            # reward error
            path_reward_error_fmt = os.path.join(
                self.save_path, "Reward error.{}".format(default_cfg["img_fmt"])
            )
            fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

            # save reward error data to csv
            reward_error_list = []
            for r in reward_list:
                end = min(len(r), len(reward_list[-1]))
                reward_error_list.append(r[:end] - reward_list[-1][:end])
            reward_error_data = pd.DataFrame(data=reward_error_list)
            reward_error_data.to_csv(
                os.path.join(self.save_path, "Reward error.csv"), encoding="gbk"
            )

            for i in range(policy_num - 1):
                legend = (
                    self.legend_list[i]
                    if len(self.legend_list) == policy_num
                    else self.algorithm_list[i]
                )
                sns.lineplot(
                    x=step_list[i][:len(reward_error_list[i])],
                    y=reward_error_list[i], label="{}".format(legend)
                )
            plt.tick_params(labelsize=default_cfg["tick_size"])
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
            plt.xlabel(x_label, default_cfg["label_font"])
            plt.ylabel("Reward error", default_cfg["label_font"])
            plt.legend(loc="best", prop=default_cfg["legend_font"])
            fig.tight_layout(pad=default_cfg["pad"])
            plt.savefig(
                path_reward_error_fmt,
                format=default_cfg["img_fmt"],
                bbox_inches="tight",
            )
            plt.close()

            # action error
            action_error_list = []
            for a in action_list:
                end = min(len(a), len(action_list[-1]))
                action_error_list.append(a[:end] - action_list[-1][:end])
            for j in range(action_dim):
                path_action_error_fmt = os.path.join(
                    self.save_path,
                    "Action-{} error.{}".format(j + 1, default_cfg["img_fmt"]),
                )
                fig, ax = plt.subplots(
                    figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"]
                )
                for i in range(policy_num - 1):
                    legend = (
                        self.legend_list[i]
                        if len(self.legend_list) == policy_num
                        else self.algorithm_list[i]
                    )
                    sns.lineplot(
                        x=step_list[i][:len(action_error_list[i])],
                        y=action_error_list[i][:, j],
                        label="{}".format(legend),
                    )
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel(x_label, default_cfg["label_font"])
                plt.ylabel("Action-{} error".format(j + 1), default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(
                    path_action_error_fmt,
                    format=default_cfg["img_fmt"],
                    bbox_inches="tight",
                )
                plt.close()

                # save action error data to csv
                action_error_data = pd.DataFrame(data=[a[:, j] for a in action_error_list])
                action_error_data.to_csv(
                    os.path.join(self.save_path, "Action-{} error.csv".format(j + 1)),
                    encoding="gbk",
                )

            # state error
            state_error_list = []
            for s in state_list:
                end = min(len(s), len(state_list[-1]))
                state_error_list.append(s[:end] - state_list[-1][:end])
            for j in range(state_dim):
                path_state_error_fmt = os.path.join(
                    self.save_path,
                    "State-{} error.{}".format(j + 1, default_cfg["img_fmt"]),
                )
                fig, ax = plt.subplots(
                    figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"]
                )
                for i in range(policy_num - 1):
                    legend = (
                        self.legend_list[i]
                        if len(self.legend_list) == policy_num
                        else self.algorithm_list[i]
                    )
                    sns.lineplot(
                        x=step_list[i][:len(state_error_list[i])],
                        y=state_error_list[i][:, j],
                        label="{}".format(legend),
                    )
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel(x_label, default_cfg["label_font"])
                plt.ylabel("State-{} error".format(j + 1), default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(
                    path_state_error_fmt,
                    format=default_cfg["img_fmt"],
                    bbox_inches="tight",
                )
                plt.close()

                # save state data to csv
                state_error_data = pd.DataFrame(data=[s[:, j] for s in state_error_list])
                state_error_data.to_csv(
                    os.path.join(self.save_path, "State-{} error.csv".format(j + 1)),
                    encoding="gbk",
                )

            # compute relative error with opt
            error_result = {}
            for i in range(policy_num - 1):
                legend = (
                    self.legend_list[i]
                    if len(self.legend_list) == policy_num
                    else "Policy-{}".format(i + 1)
                )
                end = min(len(action_list[i]), len(action_list[-1]))
                error_result.update({legend: {}})
                # action error
                for j in range(action_dim):
                    action_error = {}
                    error_list = np.abs(
                        action_list[i][:end, j] - action_list[-1][:end, j]
                    ) / (
                                         np.max(action_list[-1][:end, j])
                                         - np.min(action_list[-1][:end, j])
                                 )
                    action_error["Max_error"] = "{:.2f}%".format(max(error_list) * 100)
                    action_error["Mean_error"] = "{:.2f}%".format(
                        sum(error_list) / len(error_list) * 100
                    )
                    error_result[legend].update(
                        {"Action-{}".format(j + 1): action_error}
                    )
                # state error
                for j in range(state_dim):
                    state_error = {}
                    error_list = np.abs(
                        state_list[i][:end, j] - state_list[-1][:end, j]
                    ) / (
                                         np.max(state_list[-1][:end, j])
                                         - np.min(state_list[-1][:end, j])
                                 )
                    state_error["Max_error"] = "{:.2f}%".format(max(error_list) * 100)
                    state_error["Mean_error"] = "{:.2f}%".format(
                        sum(error_list) / len(error_list) * 100
                    )
                    error_result[legend].update({"State-{}".format(j + 1): state_error})

            # for i in range(self.policy_num):
            #     legend = (
            #         self.legend_list[i]
            #         if len(self.legend_list) == policy_num
            #         else "Policy-{}".format(i + 1)
            #     )
            #     policy_result = pd.DataFrame(data=error_result[legend])
            #     policy_result.to_excel(os.path.join(self.save_path, "Error-result.xlsx"), legend)
            error_result_data = pd.DataFrame(data=error_result)
            pd.set_option("display.max_columns", None)
            pd.set_option("display.max_rows", None)
            for key, value in error_result_data.items():
                print("===========================================================")
                print("GOPS: Policy {}".format(key))
                for key, value in value.items():
                    print(key, value)

def get_robot_state_from_info(info: dict) -> np.ndarray:
    state = info["state"]
    if isinstance(state, State):
        return state.robot_state
    elif isinstance(state, np.ndarray):
        return state

def get_reference_from_info(info: dict) -> np.ndarray:
    state = info["state"]
    if isinstance(state, State):
        return state.context_state.reference[0]
    elif isinstance(state, np.ndarray):
        return info["ref"]
