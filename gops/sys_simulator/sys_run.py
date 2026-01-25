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
import os
import imageio
from PIL import Image
import glob
import datetime
import subprocess
from typing import Any, Optional, Tuple
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
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
# ================= 论文绘图风格配置 (Start) =================
# 1. 尝试加载宋体 (请确保路径正确，或改为系统路径)
try:
    # 假设字体文件在 ../gops/utils/ 下，根据你的目录结构调整
    # 如果找不到文件，会自动回退到 SimHei
    zhfont = fm.FontProperties(fname='./../gops/utils/SIMSUN.ttf', size=20)
except:
    zhfont = fm.FontProperties(family='SimHei', size=20)

# 2. 全局参数设置 (参考 energy_stability_analysis.py)
default_cfg = dict()
default_cfg["fig_size"] = (12, 9)
default_cfg["dpi"] = 300
default_cfg["tick_size"] = 14
default_cfg["label_size"] = 20
default_cfg["legend_size"] = 15
default_cfg["img_fmt"] = "png"
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
# ================= 论文绘图风格配置 (End) =================

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
        is_tracking: bool = True,
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
            self.env_id,
            algs_name,
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
                env, networks, self.init_info, is_opt=False, render=True
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
            self.env_id,
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
            self.env_id,
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
            self.env_id,
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
        self.draw_chinese()


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

    def draw_chinese(self):

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
            x_label = "时间步"
        else:
            step_list = [s * self.dt for s in step_list]
            x_label = r"时间 $\mathrm{(s)}$"

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
        # 设置刻度字体 (Times New Roman)
        ax.tick_params(labelsize=default_cfg["tick_size"])
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname('Times New Roman')

        plt.xlabel(x_label, fontproperties=zhfont, fontsize=default_cfg["label_size"])
        plt.ylabel("奖励", fontproperties=zhfont, fontsize=default_cfg["label_size"])
        plt.legend(loc="best", prop=zhfont, fontsize=default_cfg["legend_size"])
        fig.tight_layout(pad=default_cfg["pad"])
        plt.savefig(path_reward_fmt, format=default_cfg["img_fmt"], bbox_inches="tight")
        # plt.savefig(path_reward_fmt, format="pdf", bbox_inches="tight")
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
            ax.tick_params(labelsize=default_cfg["tick_size"])
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontname('Times New Roman')
            plt.xlabel(x_label, fontproperties=zhfont, fontsize=default_cfg["label_size"])
            plt.ylabel("控制量-{}".format(j + 1), fontproperties=zhfont, fontsize=default_cfg["label_size"])
            plt.legend(loc="best", prop=zhfont, fontsize=default_cfg["legend_size"])
            fig.tight_layout(pad=default_cfg["pad"])
            plt.savefig(
                path_action_fmt, format=default_cfg["img_fmt"], bbox_inches="tight"
            )
            # plt.savefig(path_action_fmt, format="pdf", bbox_inches="tight")
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
            ax.tick_params(labelsize=default_cfg["tick_size"])
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontname('Times New Roman')
            plt.xlabel(x_label, fontproperties=zhfont, fontsize=default_cfg["label_size"])
            plt.ylabel("State-{}".format(j + 1), fontproperties=zhfont, fontsize=default_cfg["label_size"])
            plt.legend(loc="best", prop=zhfont, fontsize=default_cfg["legend_size"])
            fig.tight_layout(pad=default_cfg["pad"])
            plt.savefig(
                path_state_fmt, format=default_cfg["img_fmt"], bbox_inches="tight"
            )
            # plt.savefig(path_state_fmt, format="pdf", bbox_inches="tight")
            plt.close()
        # plot state x-y
        path_traj_fmt = os.path.join(
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
        ax.tick_params(labelsize=default_cfg["tick_size"])
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname('Times New Roman')
        plt.xlabel(r"纵向位置 $p_x (\mathrm{m})$", fontproperties=zhfont, fontsize=default_cfg["label_size"])
        plt.ylabel(r"横向位置 $p_y (\mathrm{m})$", fontproperties=zhfont, fontsize=default_cfg["label_size"])
        plt.legend(loc="best", prop=zhfont, fontsize=default_cfg["legend_size"])
        fig.tight_layout(pad=default_cfg["pad"])
        plt.savefig(
            path_traj_fmt, format=default_cfg["img_fmt"], bbox_inches="tight"
        )
        # plt.savefig(path_traj_fmt, format="pdf", bbox_inches="tight")
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
                    label="全局轨迹",
                )
                tracking_state_data.append(state_ref_error_list[longest_traj_index]["ref-{}".format(j)])
                ax.tick_params(labelsize=default_cfg["tick_size"])
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontname('Times New Roman')

                plt.xlabel(x_label, fontproperties=zhfont, fontsize=default_cfg["label_size"])
                plt.ylabel("状态-{}".format(j + 1), fontproperties=zhfont, fontsize=default_cfg["label_size"])
                plt.legend(loc="best", prop=zhfont, fontsize=default_cfg["legend_size"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(
                    path_tracking_state_fmt,
                    format=default_cfg["img_fmt"],
                    bbox_inches="tight",
                )
                # plt.savefig(path_tracking_state_fmt, format="pdf", bbox_inches="tight")
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
                ax.tick_params(labelsize=default_cfg["tick_size"])
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontname('Times New Roman')
                plt.xlabel(x_label, fontproperties=zhfont, fontsize=default_cfg["label_size"])
                plt.ylabel("Ref$-$State-Error{}".format(j + 1), fontproperties=zhfont, fontsize=default_cfg["label_size"])
                plt.legend(loc="best", prop=zhfont, fontsize=default_cfg["legend_size"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(
                    path_tracking_error_fmt,
                    format=default_cfg["img_fmt"],
                    bbox_inches="tight",
                )
                # plt.savefig(path_tracking_error_fmt, format="pdf", bbox_inches="tight")
                plt.close()

                tracking_error_data = pd.DataFrame(data=tracking_error_data)
                tracking_error_data.to_csv(
                    os.path.join(self.save_path, "Ref-State-Error{}.csv".format(j + 1)),
                    encoding="gbk",
                )

        # plot calculation time
        path_calctime_fmt = os.path.join(
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
        ax.tick_params(labelsize=default_cfg["tick_size"])
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname('Times New Roman')

        plt.xlabel(x_label, fontproperties=zhfont, fontsize=default_cfg["label_size"])
        plt.ylabel(r"单步推理时间 ($\mathrm{ms}$)", fontproperties=zhfont, fontsize=default_cfg["label_size"])
        plt.legend(loc="best", prop=zhfont, fontsize=default_cfg["legend_size"])
        fig.tight_layout(pad=default_cfg["pad"])
        plt.savefig(
            path_calctime_fmt, format=default_cfg["img_fmt"], bbox_inches="tight"
        )
        # plt.savefig(path_calctime_fmt, format="pdf", bbox_inches="tight")
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
                ax.tick_params(labelsize=default_cfg["tick_size"])
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontname('Times New Roman')

                plt.xlabel(x_label, fontproperties=zhfont, fontsize=default_cfg["label_size"])
                plt.ylabel("Constrain-{}".format(j + 1), fontproperties=zhfont, fontsize=default_cfg["label_size"])
                plt.legend(loc="best", prop=zhfont, fontsize=default_cfg["legend_size"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(
                    path_constraint_fmt,
                    format=default_cfg["img_fmt"],
                    bbox_inches="tight",
                )
                # plt.savefig(path_constraint_fmt, format="pdf", bbox_inches="tight")
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
            ax.tick_params(labelsize=default_cfg["tick_size"])
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontname('Times New Roman')
            plt.xlabel(x_label, fontproperties=zhfont, fontsize=default_cfg["label_size"])
            plt.ylabel("Reward error", fontproperties=zhfont, fontsize=default_cfg["label_size"])
            plt.legend(loc="best", prop=zhfont, fontsize=default_cfg["legend_size"])
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
                ax.tick_params(labelsize=default_cfg["tick_size"])
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontname('Times New Roman')
                plt.xlabel(x_label, fontproperties=zhfont, fontsize=default_cfg["label_size"])
                plt.ylabel("Action-{} error".format(j + 1), fontproperties=zhfont, fontsize=default_cfg["label_size"])
                plt.legend(loc="best", prop=zhfont, fontsize=default_cfg["legend_size"])
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
                ax.tick_params(labelsize=default_cfg["tick_size"])
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontname('Times New Roman')
                plt.xlabel(x_label, fontproperties=zhfont, fontsize=default_cfg["label_size"])
                plt.ylabel("State-{} error".format(j + 1), fontproperties=zhfont, fontsize=default_cfg["label_size"])
                plt.legend(loc="best", prop=zhfont, fontsize=default_cfg["legend_size"])
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


class EnhancedPolicyRunner(PolicyRunner):
    """Enhanced PolicyRunner with video screenshot, GIF conversion and Trajectory Analysis"""

    def __init__(self,
                 save_render: bool = False,
                 plot_range: list = None,
                 is_init_info: bool = False,
                 init_info: dict = None,
                 legend_list: list = None,
                 load_opt_path: Optional[str] = None,
                 constrained_env: bool = False,
                 is_tracking: bool = True,
                 use_dist: bool = False,
                 dt: float = None,
                 obs_noise_type: str = None,
                 obs_noise_data: list = None,
                 action_noise_type: str = None,
                 action_noise_data: list = None,
                 fixed_seed: int = 2024,
                 is_base_planner: bool = False,
                 base_planner: str = None,
                 controller: str = "SimpleController",
                 *args, **kwargs):
        # Extract custom parameters first
        self.save_screenshots = kwargs.pop('save_screenshots', False)
        self.screenshot_interval = kwargs.pop('screenshot_interval', 10)
        self.convert_to_gif = kwargs.pop('convert_to_gif', False)
        self.gif_fps = kwargs.pop('gif_fps', 10)
        self.fixed_seed = fixed_seed
        super().__init__(*args, **kwargs)
        self.legend_list = legend_list
        self.is_tracking = is_tracking
        self.current_env = None  # 用于存储当前运行的环境实例

        self.env_storage = {}  # 存储 {算法名: env实例}
        self.frame_storage = {}  # 存储 {算法名: frame列表}
        self.current_alg_idx = 0  # 当前运行算法的索引
        self.is_base_planner = is_base_planner
        self.base_planner_name = base_planner
        self.controller_name = controller
        self.load_opt_path = load_opt_path
        # Create additional directories if needed
        old_path = self.save_path
        new_path = f"{old_path}_seed{self.fixed_seed}"
        self.save_path = new_path
        if os.path.exists(old_path):
            try:
                os.rename(old_path, new_path)
                print(f"Directory renamed: {old_path} -> {new_path}")
            except OSError as e:
                print(f"Warning: Could not rename directory ({e}). Creating new one.")
                os.makedirs(self.save_path, exist_ok=True)
        else:
            os.makedirs(self.save_path, exist_ok=True)
        if self.save_screenshots:
            self.screenshots_dir = os.path.join(self.save_path, "screenshots_png")
            os.makedirs(self.screenshots_dir, exist_ok=True)

        if self.convert_to_gif:
            self.gif_dir = os.path.join(self.save_path, "gifs")
            os.makedirs(self.gif_dir, exist_ok=True)

    def run_an_episode(self, env, controller, init_info, is_opt, render=True):
        """Override to capture screenshots and store env"""
        if self.legend_list and self.current_alg_idx < len(self.legend_list):
            current_alg_name = self.legend_list[self.current_alg_idx]
        else:
            current_alg_name = f"Algo_{self.current_alg_idx}"

            # 保存环境实例到字典
        self.env_storage[current_alg_name] = env
        current_seed = self.fixed_seed

        state_list = []
        action_list = []
        reward_list = []
        constrain_list = []
        obs_list = []
        step = 0
        step_list = []
        calctime_list = []
        info_list = [init_info]

        # For screenshot capture
        frame_images = []
        if hasattr(env, 'seed'):
            env.seed(current_seed)
        np.random.seed(current_seed)

        obs, info = env.reset(**init_info)
        state = env.state
        print("Initial robot state: ")
        try:
            print(self._PolicyRunner__convert_format(np.asarray(state.robot_state)))
        except AttributeError:
            print(np.asarray(state.robot_state))

        # plot tracking
        state_with_ref_error = {}
        done = False
        info.update({"TimeLimit.truncated": False})

        while not (done or info["TimeLimit.truncated"]):
            print("step:", step + 1)
            state_list.append(state.robot_state)
            obs_list.append(obs)

            if is_opt:
                try:
                    from gym import Env
                except ImportError:
                    Env = object

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
                try:
                    action = self._PolicyRunner__action_noise(action)
                except AttributeError:
                    pass
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

            action_list.append(info.get("raw_action", action))
            step_list.append(step)
            reward_list.append(reward)
            info_list.append(info)
            calctime_list.append(calc_time * 1000)
            obs = next_obs
            state = env.state

            # Capture screenshot
            if render and self.save_screenshots and step % self.screenshot_interval == 0:
                try:
                    frame = env.render(mode='rgb_array')
                    if frame is not None:
                        frame_images.append(frame)
                        screenshot_path = os.path.join(
                            self.screenshots_dir, f"frame_{current_alg_name}_{step * env.dt:.1f}s.pdf")
                        screenshot_path_png = os.path.join(
                            self.screenshots_dir, f"frame_{current_alg_name}_{step * env.dt:.1f}s.png")
                        plt.imsave(screenshot_path, frame)
                        plt.imsave(screenshot_path_png, frame)
                except Exception as e:
                    print(f"Failed to capture screenshot at step {step}: {e}")

            step = step + 1

            if "TimeLimit.truncated" not in info.keys():
                info["TimeLimit.truncated"] = False

            if render:
                try:
                    env.render(mode='human')
                except:
                    try:
                        env.render()
                    except:
                        pass
        if self.save_screenshots or self.convert_to_gif:
            self.frame_storage[current_alg_name] = frame_images

            # [修改 5] 索引递增，指向下一个算法
        self.current_alg_idx += 1

        # if self.convert_to_gif and frame_images:
        #     self._create_gif_from_frames(frame_images)
        #
        # self._convert_existing_videos_to_gif()

        eval_dict = {
            "reward_list": reward_list,
            "action_list": action_list,
            "state_list": state_list,
            "step_list": step_list,
            "obs_list": obs_list,
            "info_list": info_list,
            "calctime_list": calctime_list,
            "frame_images": frame_images if self.save_screenshots else None
        }

        if self.constrained_env:
            eval_dict.update({"constrain_list": constrain_list, })

        if self.is_tracking:
            tracking_dict = state_with_ref_error
        else:
            tracking_dict = {}

        return eval_dict, tracking_dict

    def _run_base_planner(self, render=True):
        from gops.utils.planner_benchmark.elements.map import RoutedLocalMap, Lane
        from gops.utils.planner_benchmark.elements.box import TrackingBoxList, TrackingBox
        from gops.utils.planner_benchmark.elements.vehicle import VehicleState
        """
        Runs baseline planners (LatticePlanner, etc.) with strict type checking and error reporting.
        """
        import traceback  # For detailed error logs

        if not self.base_planner_name:
            return

        print(f"=======================================")
        print(f"GOPS: Running Baseline Planner: {self.base_planner_name}")

        # 1. Environment Loading
        if not self.args_list:
            print("Warning: args_list is empty, using default configuration.")
            self.args = {"pre_horizon": 20}
        else:
            self.args = self.args_list[0]

        try:
            if not self.env_storage:
                raise ValueError("No environment stored in env_storage.")

            env_key = list(self.env_storage.keys())[0]
            env_wrapped = deepcopy(self.env_storage[env_key])

            # Unwrap environment to access raw attributes
            if hasattr(env_wrapped, 'unwrapped'):
                env = env_wrapped.unwrapped
            else:
                env = env_wrapped

            print(f"Using unwrapped env: {type(env)}")
        except Exception as e:
            print(f"Error preparing env: {e}")
            traceback.print_exc()
            return

        # -----------------------------------------------------------
        # A. Initialize Planner
        # Force strict types for config
        plan_steps = int(self.args.get('pre_horizon', 20))
        plan_dt = float(self.dt) if self.dt is not None else 0.1

        print(f"Planner Config: steps={plan_steps}, dt={plan_dt}")

        planner = None
        if self.base_planner_name == "LatticePlanner":
            from gops.utils.planner_benchmark.planner_zoo import LatticePlanner
            planner = LatticePlanner({
                "steps": plan_steps,
                "dt": plan_dt,
                "end_s_candidates": (20, 40, 60),
                "end_l_candidates": (-3.5, 0, 3.5),
            })
            print("LatticePlanner initialized.")

        elif self.base_planner_name == "BezierPlanner":
            from gops.utils.planner_benchmark.planner_zoo import BezierPlanner
            planner = BezierPlanner({
                "steps": plan_steps,
                "dt": plan_dt,
                "end_s_candidates": (20, 40, 60),
                "end_l_candidates": (-3.5, 0, 3.5),
            })
            print("BezierPlanner initialized.")
        else:
            raise ValueError(f"Unknown base planner: {self.base_planner_name}")

        # B. Initialize Controller
        controller = None
        if self.controller_name == "IDMController":
            from gops.utils.planner_benchmark.control.IDMController import IDMController
            controller = IDMController()
            print("IDMController initialized.")

        elif self.controller_name == "SimpleController":
            from gops.utils.planner_benchmark.control.SimpleController import SimpleController
            controller = SimpleController()
            print("SimpleController initialized.")

        elif self.controller_name == "MPCController":
            print("Initializing MPCController...")
            if self.opt_args is None:
                raise ValueError("Choose to use MPC controller, but opt_args is None.")
            from gops.utils.common_utils import create_env_model
            model = create_env_model(**self.args, mask_at_done=False)
            local_opt_args = self.opt_args.copy()
            if "opt_controller_type" in local_opt_args:
                local_opt_args.pop("opt_controller_type")
            if "use_MPC_for_general_env" in local_opt_args:
                local_opt_args.pop("use_MPC_for_general_env")

            if self.opt_args.get("use_MPC_for_general_env", False):
                from gops.sys_simulator.opt_controller_for_gen_env import OptController
                controller = OptController(model, **local_opt_args)
            else:
                from gops.sys_simulator.opt_controller import OptController
                controller = OptController(model, **local_opt_args)
            print("MPCController initialized.")
        else:
            raise ValueError(f"Unknown controller: {self.controller_name}")

        # -----------------------------------------------------------

        # 3. Run Episode
        current_seed = self.fixed_seed
        if hasattr(env, 'seed'): env.seed(current_seed)
        np.random.seed(current_seed)

        reset_res = env.reset(**self.init_info)
        if isinstance(reset_res, tuple):
            obs, info = reset_res
        else:
            obs = reset_res
            info = {}

        name = f"{self.base_planner_name}_{self.controller_name}"
        self.env_storage[name] = env
        frame_images = []
        step = 0
        done = False

        # Build LocalMap for Lattice Planner
        # We manually construct a map based on the reference trajectory
        local_map = RoutedLocalMap()
        xs, ys = [], []
        # Use floats to prevent type issues
        for t in range(0, 200):
            t_sec = float(t) * 0.5
            x = env.ref_traj.compute_x(t_sec, env.path_num, env.u_num)
            y = env.ref_traj.compute_y(t_sec, env.path_num, env.u_num)
            xs.append(float(x))
            ys.append(float(y))

        center_line = np.column_stack((xs, ys))

        for idx, lat_off in enumerate([-3.5, 0, 3.5]):
            lane_line = center_line.copy()
            lane_line[:, 1] += float(lat_off)
            local_map.lanes.append(Lane(int(idx), lane_line, width=3.5, speed_limit=15.0))

        planner.set_local_map(local_map)

        print(f"Start planning loop for {name}...")

        while not (done or info.get("TimeLimit.truncated", False)):
            # --- Data Sanitization & Adapter ---
            obstacles_box_list = []
            for i, obs in enumerate(env.obstacles):
                # Explicitly cast to float/int to avoid NoneType or numpy scalar issues
                obs_x = float(obs.x)
                obs_y = float(obs.y)
                obs_l = float(obs.l)
                obs_w = float(obs.w)
                obs_phi = float(obs.phi)
                obs_u = float(obs.u) if obs.u is not None else 0.0
                obs_id = int(obs.id)

                t_box = TrackingBox(
                    obb=(obs_x, obs_y, obs_l, obs_w, obs_phi, 0.5),
                    vx=obs_u, vy=0.0, id=obs_id, class_label=1
                )
                obstacles_box_list.append(t_box)
            obstacles_container = TrackingBoxList(obstacles_box_list)

            # Extract State
            raw_state = env.state
            if hasattr(raw_state, 'robot_state'):
                raw_state = raw_state.robot_state
            elif hasattr(raw_state, 'numpy'):
                raw_state = raw_state.numpy()

            try:
                current_state_np = np.array(raw_state).flatten()
            except:
                current_state_np = np.zeros(6)

            # Explicit float casting for VehicleState
            ego_x = float(current_state_np[0])
            ego_y = float(current_state_np[1])
            ego_yaw = float(current_state_np[2])
            ego_v = float(current_state_np[3])  # Longitudinal Speed
            ego_vy = float(current_state_np[4])  # Lateral Speed
            ego_yaw_rate = float(current_state_np[5])

            ego_veh_state = VehicleState.from_kine_states(
                x=ego_x, y=ego_y, yaw=ego_yaw,
                vx=ego_v, vy=ego_vy,
                length=float(env.veh_length), width=float(env.veh_width)
            )
            ego_veh_state.kinematics.yaw_rate = ego_yaw_rate

            # --- Planning ---
            try:
                traj = planner.plan(ego_veh_state, obstacles_container, local_map)
            except Exception as e:
                # Print FULL traceback to locate NoneType error
                print(f"Plan failed at step {step}: {e}")
                traceback.print_exc()
                traj = None

            # --- Control ---
            if traj is None:
                action_single = np.array([0.0, -3.0])
            else:
                if self.controller_name == "IDMController":
                    ref = np.stack((np.array(traj.x), np.array(traj.y)), axis=1)
                    current_pose = np.array([ego_x, ego_y, ego_yaw])
                    current_speed = ego_v

                    target_v_first = traj.v[0] if len(traj.v) > 0 else 0.0
                    action_single = controller.get_control(ref, 0, 100, target_v_first, current_pose, current_speed)

                elif self.controller_name == "SimpleController":
                    ref_path = np.stack((np.array(traj.x), np.array(traj.y)), axis=1)
                    target_v_first = traj.v[0] if len(traj.v) > 0 else 0.0

                    current_pose = np.array([ego_x, ego_y, ego_yaw])
                    current_speed = ego_v
                    action_single = controller.get_control(ref_path, target_v_first, current_pose, current_speed)

                elif self.controller_name == "MPCController":
                    action_single = controller(obs, info)
                else:
                    action_single = np.array([0.0, 0.0])

            # --- Step Env ---
            if traj is not None:
                # Inject planning trajectory for rendering
                try:
                    p_x = np.array(traj.x)
                    p_y = np.array(traj.y)
                    p_head = np.array(traj.heading)
                    p_v = np.array(traj.v)
                    min_len = min(len(p_x), len(p_y), len(p_head), len(p_v))
                    env.current_planning_traj = np.stack([
                        p_x[:min_len], p_y[:min_len], p_head[:min_len], p_v[:min_len]
                    ], axis=1)
                except Exception as e:
                    print(f"Error setting planning traj: {e}")

            # Manual Environment Step Update (Hack to bypass env internal planner)
            env.step_self += 1
            for o in env.obstacles:
                if o.type == "dynamic" and o.dynamic_data: o.dynamic_data.step()

            # Use calculated action to update dynamics
            if hasattr(env, 'ref_points'):
                road_info = env.ref_points[1, 4:]
            else:
                road_info = np.array([0.0, 0.0])

            next_state = env.vehicle_dynamics.f_xu(
                current_state_np,
                action_single,
                road_info,
                env.dt
            )
            env.state = next_state

            dist_traveled = ego_v * env.dt
            current_ref_v = max(env.ref_points[0, 3], 0.1)
            env.t += dist_traveled / current_ref_v

            env._update_ref_points()
            env.guide_trajectories = env._generate_guidance_prompts()

            done = env.judge_done()
            env._log_step_data()

            if hasattr(env, 'info'):
                info = env.info
            else:
                info = {}

            # Screenshot
            if render and self.save_screenshots and step % self.screenshot_interval == 0:
                try:
                    frame = env.render(mode='rgb_array')
                    if frame is not None:
                        frame_images.append(frame)
                        screenshot_path = os.path.join(
                            self.screenshots_dir, f"frame_{self.base_planner_name}_{step * env.dt:.1f}s.pdf")
                        screenshot_path_png = os.path.join(
                            self.screenshots_dir, f"frame_{self.base_planner_name}_{step * env.dt:.1f}s.png")
                        plt.imsave(screenshot_path, frame)
                        plt.imsave(screenshot_path_png, frame)
                except Exception as e:
                    print(f"Failed to capture screenshot at step {step}: {e}")
            step += 1
            if step >= env.max_episode_steps:
                done = True

        if self.save_screenshots or self.convert_to_gif:
            self.frame_storage[name] = frame_images

        print(f"Baseline {name} finished. Steps: {step}")

    def _plot_tracking_analysis(self, env, suffix=""):
        """
        绘制仿真结果：轨迹对比图 + 跟踪误差图
        使用 self.current_env.log_data，并保存到 self.save_path
        """
        # 1. 检查 Env 是否存在且包含数据
        env = self.current_env
        if env is None:
            print("Plotting Warning: Environment instance not captured.")
            return

        # 兼容 Wrapper
        if hasattr(env, 'unwrapped'):
            if hasattr(env.unwrapped, 'log_data'):
                env = env.unwrapped

        if not hasattr(env, 'log_data') or not env.log_data.get('actual_x'):
            print("Plotting Warning: Environment does not have 'log_data' or data is empty.")
            print("Please ensure Env step() method logs data into self.log_data.")
            return

        print("Plotting advanced tracking analysis...")
        data = env.log_data

        # 2. 数据对齐
        min_len = min(len(data['actual_x']), len(data['ref_x']), len(data['err_lat']))
        time_steps = np.arange(min_len) * env.dt

        # 截断数据
        act_x = np.array(data['actual_x'][:min_len])
        act_y = np.array(data['actual_y'][:min_len])
        act_u = np.array(data['actual_u'][:min_len])
        ref_x = np.array(data['ref_x'][:min_len])
        ref_y = np.array(data['ref_y'][:min_len])
        ref_u = np.array(data['ref_u'][:min_len])
        err_lat = np.array(data['err_lat'][:min_len])
        err_phi = np.array(data['err_phi'][:min_len])

        # 3. 创建画布 (使用文件顶部的 default_cfg)
        fig = plt.figure(figsize=cm2inch(18 * 2.54, 10 * 2.54), dpi=default_cfg["dpi"])
        gs = fig.add_gridspec(3, 2)

        # --- (A) 轨迹对比图 (左半边) ---
        ax_traj = fig.add_subplot(gs[:, 0])

        # 画障碍物
        if hasattr(env, 'obstacles'):
            for obs in env.obstacles:
                # 简单的判断类型
                is_dynamic = getattr(obs, 'type', 'static') == 'dynamic'
                can_cross = getattr(obs, 'can_cross', False)
                color = 'red' if is_dynamic else ('lime' if can_cross else 'gray')

                cx, cy = obs.x, obs.y
                l = getattr(obs, 'l', 2.0)
                w = getattr(obs, 'w', 2.0)
                radius = max(l, w) / 2.0

                circle = plt.Circle((cx, cy), radius, color=color, alpha=0.5, label='_nolegend_')
                ax_traj.add_patch(circle)
                if is_dynamic:
                    ax_traj.text(cx, cy, "Dyn", fontsize=8, color='darkred')

        # 画轨迹
        ax_traj.plot(ref_x, ref_y, 'b--', linewidth=1.5, label='规划目标 (Plan)')
        ax_traj.plot(act_x, act_y, 'k-', linewidth=2, label='实际轨迹 (Actual)')

        # 标注起终点
        ax_traj.plot(act_x[0], act_y[0], 'go', markersize=6, label='起点')
        ax_traj.plot(act_x[-1], act_y[-1], 'rx', markersize=6, label='终点')

        ax_traj.set_title("轨迹跟踪结果", fontproperties=zhfont, fontsize=12)
        ax_traj.set_xlabel(r"纵向位置 $X (\mathrm{m})$", fontproperties=zhfont, fontsize=10)
        ax_traj.set_ylabel(r"横向位置 $Y (\mathrm{m})$", fontproperties=zhfont, fontsize=10)
        ax_traj.legend(prop=zhfont, fontsize=8)
        ax_traj.axis('equal')
        ax_traj.grid(True, linestyle=':', alpha=0.6)

        # 设置刻度字体
        for label in ax_traj.get_xticklabels() + ax_traj.get_yticklabels():
            label.set_fontname('Times New Roman')

        # --- (B) 横向误差 (右上) ---
        ax_lat = fig.add_subplot(gs[0, 1])
        ax_lat.plot(time_steps, err_lat, 'r-')
        ax_lat.set_title("横向跟踪误差", fontproperties=zhfont, fontsize=12)
        ax_lat.set_ylabel(r"误差 $(\mathrm{m})$", fontproperties=zhfont, fontsize=10)
        ax_lat.grid(True, linestyle=':', alpha=0.6)
        # 标出最大误差
        max_lat = np.max(np.abs(err_lat))
        ax_lat.text(0.02, 0.9, f'Max Abs Err: {max_lat:.3f}m', transform=ax_lat.transAxes, fontsize=9)
        for label in ax_lat.get_xticklabels() + ax_lat.get_yticklabels():
            label.set_fontname('Times New Roman')

        # --- (C) 航向误差 (右中) ---
        ax_phi = fig.add_subplot(gs[1, 1])
        err_phi_deg = np.rad2deg(err_phi)
        ax_phi.plot(time_steps, err_phi_deg, 'g-')
        ax_phi.set_title("航向跟踪误差", fontproperties=zhfont, fontsize=12)
        ax_phi.set_ylabel(r"误差 $(\mathrm{deg})$", fontproperties=zhfont, fontsize=10)
        ax_phi.grid(True, linestyle=':', alpha=0.6)
        for label in ax_phi.get_xticklabels() + ax_phi.get_yticklabels():
            label.set_fontname('Times New Roman')

        # --- (D) 速度误差 (右下) ---
        ax_u = fig.add_subplot(gs[2, 1])
        ax_u.plot(time_steps, act_u, 'k-', label='实际')
        ax_u.plot(time_steps, ref_u, 'b--', label='规划')
        ax_u.set_title("速度跟踪", fontproperties=zhfont, fontsize=12)
        ax_u.set_ylabel(r"速度 $(\mathrm{m/s})$", fontproperties=zhfont, fontsize=10)
        ax_u.set_xlabel(r"时间 $(\mathrm{s})$", fontproperties=zhfont, fontsize=10)
        ax_u.legend(prop=zhfont, fontsize=8)
        ax_u.grid(True, linestyle=':', alpha=0.6)
        for label in ax_u.get_xticklabels() + ax_u.get_yticklabels():
            label.set_fontname('Times New Roman')

        plt.tight_layout()

        # 4. 保存图片
        save_path = os.path.join(self.save_path, f'tracking_analysis_{suffix}.png')
        print(f"Saving plot to: {save_path}")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

    def run(self):
        """Override run method to include post-processing and plotting"""
        try:
            self.current_alg_idx = 0
            self.env_storage = {}
            self.frame_storage = {}
            # 1. 运行仿真
            self._PolicyRunner__run_data()
            # 2. 运行 Baseline (Lattice)
            if self.is_base_planner:
                self._run_base_planner()
            else:
                print("Don't select any base planner, "
                      "you can choose LatticePlanner/BezierPlanner matching with "
                      "IDMController/SimpleController/MPCController")
            # 3. 转换视频
            self._convert_existing_videos_to_gif()

            # 3. 绘制默认图 (GOPS 基类方法)
            self.draw()

            for alg_name, env_instance in self.env_storage.items():
                print(f"Processing results for: {alg_name}")

                # 绘制分析图
                self._plot_tracking_analysis(env_instance, suffix=alg_name)

                # 生成 GIF
                if self.convert_to_gif and alg_name in self.frame_storage:
                    frames = self.frame_storage[alg_name]
                    self._create_gif_from_frames(frames, suffix=alg_name)

            # 5. 生成总结报告
            self._generate_summary_report()
            print("Simulation completed successfully!")

        except Exception as e:
            print(f"Error during simulation: {e}")
            import traceback
            traceback.print_exc()

            # 尝试备用方案
            print("\nTrying alternative approach...")

    def _create_gif_from_frames(self, frames, suffix=""):
        """Create GIF from captured frames with suffix"""
        if not frames:
            return
        try:
            # [修改] 文件名增加后缀
            gif_name = f"simulation_{suffix}.gif" if suffix else "simulation.gif"
            gif_path = os.path.join(self.gif_dir, gif_name)

            with imageio.get_writer(gif_path, mode='I', fps=self.gif_fps) as writer:
                for frame in frames:
                    writer.append_data(frame)
            print(f"Created GIF: {gif_path}")

            # Small GIF
            small_gif_name = f"simulation_small_{suffix}.gif" if suffix else "simulation_small.gif"
            small_gif_path = os.path.join(self.gif_dir, small_gif_name)
            self._create_optimized_gif(frames, small_gif_path)
        except Exception as e:
            print(f"Failed to create GIF for {suffix}: {e}")
    def _create_optimized_gif(self, frames, output_path, max_size=(640, 480)):
        """Create optimized GIF with reduced size"""
        try:
            resized_frames = []
            for frame in frames:
                img = Image.fromarray(frame)
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                resized_frames.append(np.array(img))
            with imageio.get_writer(output_path, mode='I', fps=self.gif_fps) as writer:
                for frame in resized_frames:
                    writer.append_data(frame)
            print(f"Created optimized GIF: {output_path}")
        except Exception as e:
            print(f"Failed to create optimized GIF: {e}")

    def _convert_existing_videos_to_gif(self):
        """Convert existing MP4 videos to GIF format"""
        if not hasattr(self, 'save_path'):
            return
        videos_path = os.path.join(self.save_path, "videos")
        if not os.path.exists(videos_path):
            return
        mp4_files = glob.glob(os.path.join(videos_path, "*.mp4"))
        for mp4_file in mp4_files:
            try:
                gif_file = mp4_file.replace('.mp4', '.gif')
                try:
                    cmd = [
                        'ffmpeg', '-i', mp4_file,
                        '-vf', 'fps=10,scale=640:-1:flags=lanczos',
                        '-y', gif_file
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        print(f"Converted {mp4_file} to GIF")
                except FileNotFoundError:
                    pass
            except Exception as e:
                print(f"Failed to convert {mp4_file} to GIF: {e}")

    def _generate_summary_report(self):
        """Generate a summary report with links to videos/GIFs and PLOTS"""
        try:
            report_path = os.path.join(self.save_path, "simulation_summary.md")
            with open(report_path, 'w') as f:
                f.write("# Simulation Summary Report\n\n")
                f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

                # Analysis Plot Section
                plot_file = 'tracking_analysis_chinese.png'
                if os.path.exists(os.path.join(self.save_path, plot_file)):
                    f.write("\n## 轨迹与控制分析\n")
                    f.write(f"![Analysis]({plot_file})\n")

                # Videos Section
                if os.path.exists(os.path.join(self.save_path, "videos")):
                    f.write("\n## Generated Videos\n")
                    videos = glob.glob(os.path.join(self.save_path, "videos", "*.mp4"))
                    for video in videos:
                        rel_path = os.path.relpath(video, self.save_path)
                        f.write(f"- [{os.path.basename(video)}]({rel_path})\n")

        except Exception as e:
            print(f"Failed to generate summary report: {e}")

    def _save_frames_as_video(self, frames, output_path):
        """将帧保存为视频文件"""
        try:
            if not frames:
                return

            # 使用OpenCV或imageio保存为视频
            try:
                import cv2
                height, width = frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video = cv2.VideoWriter(output_path, fourcc, 10.0, (width, height))

                for frame in frames:
                    # 转换颜色空间 RGB -> BGR
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    video.write(frame_bgr)

                video.release()
                print(f"Video saved to {output_path}")

            except ImportError:
                # 如果没有OpenCV，使用imageio
                with imageio.get_writer(output_path, mode='I', fps=10) as writer:
                    for frame in frames:
                        writer.append_data(frame)
                print(f"Video saved to {output_path} using imageio")

        except Exception as e:
            print(f"Failed to save video: {e}")

    def _generate_summary_report(self):
        """Generate a summary report with links to videos/GIFs and PLOTS"""
        try:
            report_path = os.path.join(self.save_path, "simulation_summary.md")
            with open(report_path, 'w', encoding='utf-8') as f:  # 建议指定 encoding
                f.write("# Simulation Summary Report\n\n")
                f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

                # [修改] 遍历所有算法生成报告
                # 如果没有 legend_list, 尝试从 env_storage 的 keys 获取
                algs = self.legend_list if self.legend_list else list(self.env_storage.keys())

                for alg_name in algs:
                    f.write(f"\n## Algorithm: {alg_name}\n")

                    # Analysis Plot Section
                    plot_file = f'tracking_analysis_{alg_name}.png'
                    if os.path.exists(os.path.join(self.save_path, plot_file)):
                        f.write(f"### 轨迹与控制分析 ({alg_name})\n")
                        f.write(f"![Analysis]({plot_file})\n")

                    # GIFs Section
                    if self.convert_to_gif:
                        gif_file = f"simulation_{alg_name}.gif"
                        if os.path.exists(os.path.join(self.gif_dir, gif_file)):
                            f.write(f"### Simulation GIF ({alg_name})\n")
                            # Markdown显示GIF，这里用相对路径
                            rel_gif_path = os.path.join("gifs", gif_file)
                            f.write(f"![GIF]({rel_gif_path})\n")

                # Videos Section (Common)
                if os.path.exists(os.path.join(self.save_path, "videos")):
                    f.write("\n## Generated Videos (MP4)\n")
                    videos = glob.glob(os.path.join(self.save_path, "videos", "*.mp4"))
                    for video in videos:
                        rel_path = os.path.relpath(video, self.save_path)
                        f.write(f"- [{os.path.basename(video)}]({rel_path})\n")

        except Exception as e:
            print(f"Failed to generate summary report: {e}")

class DisturbanceObserver:
    """非线性扰动观测器 (NDO) - 带低通滤波"""

    def __init__(self, vehicle_params: dict, gain=40.0, dt=0.01, lpf_ratio=0.15):
        self.p = vehicle_params
        self.L = gain
        self.dt = dt
        self.z = 0.0
        self.d_hat = 0.0

        # [新增] 滤波相关变量
        self.lpf_ratio = lpf_ratio  # 滤波系数 (0~1)，越小越平滑，滞后越大
        self.d_hat_filtered = 0.0  # 滤波后的观测值

        # 参数预提取
        self.lf = self.p['lf']
        self.lr = self.p['lr']
        self.Izz = self.p['Izz']
        self.Cf = self.p['k_alpha1']
        self.Cr = self.p['k_alpha3']

    def update(self, vx, vy, r, delta):
        if abs(vx) < 1.0:
            return 0.0

        # 1. 标称模型计算
        alpha_f = delta - (vy + self.lf * r) / vx
        alpha_r = - (vy - self.lr * r) / vx

        Fyf_total = 2 * self.Cf * alpha_f
        Fyr_total = 2 * self.Cr * alpha_r

        nominal_yaw_acc = (self.lf * Fyf_total - self.lr * Fyr_total) / self.Izz

        # 2. 观测器迭代
        z_dot = -self.L * (self.z + self.L * r + nominal_yaw_acc)
        self.z += z_dot * self.dt

        # 3. 计算原始扰动估计
        d_hat_raw = (self.z + self.L * r) * self.Izz

        # 4. [核心新增] 一阶低通滤波
        # y[k] = (1 - alpha) * y[k-1] + alpha * x[k]
        self.d_hat_filtered = (1 - self.lpf_ratio) * self.d_hat_filtered + self.lpf_ratio * d_hat_raw

        return self.d_hat_filtered

class PolicyRunner_CoSimulation_Disturbance_Compare:
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
            datetime.datetime.now().strftime("%y%m%d-%H%M%S") + "_carsim",
        )
        os.makedirs(self.save_path, exist_ok=True)

    def run_an_episode(
            self,
            env: Any,
            controller: Any,
            init_info: dict,
            is_opt: bool,
            render: bool = True,
            policy_index: int = 0  # 传入策略索引
    ) -> Tuple[dict, dict]:
        # 仅当运行第2个策略 (index=1) 时启用 NDO
        enable_ndo = (policy_index == 1)
        ndo = None
        veh_params = None
        # 用于差分注入的关键变量
        last_diff_torque = 0.0
        # 确保环境已加载 CarSim 且参数可获取
        if not hasattr(env, "vehicle_dynamics"):
            # 有些环境结构可能需要先 reset 才能初始化 vehicle_dynamics
            # 这里先尝试 reset, 下面代码本来就会 reset
            pass
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

        feed_mode = "NDO"
        Kp = 10.0
        Kd = 50.0
        # 在环境 Reset 后获取参数
        if enable_ndo and not is_opt:
            if hasattr(env, "vehicle_dynamics") and hasattr(env.vehicle_dynamics, "vehicle_params"):
                veh_params = env.vehicle_dynamics.vehicle_params
                print(f">>> Policy {policy_index + 1}: NDO Enabled. Params loaded from Env.")
                # 初始化 NDO
                lpf_ratio = 0.05
                ndo = DisturbanceObserver(veh_params, gain=100.0, dt=self.dt, lpf_ratio=lpf_ratio)
            else:
                print("Warning: 'vehicle_params' not found in env. NDO disabled.")
                enable_ndo = False
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
                # 获取当前真实物理状态
                current_real_phys = np.array(state.robot_state[8:13])
                target_phys = current_real_phys + action  # 策略的基础目标
                # --- 2. NDO 观测与前馈补偿 ---
                if enable_ndo and not is_opt:
                    if feed_mode == "NDO":
                        state_vec = np.array(state.robot_state)
                        vx = state_vec[3]
                        vy = state_vec[4]
                        r = state_vec[5]
                        steer_norm = action[-1]+state_vec[12]
                        delta_rad = steer_norm

                        # C. NDO 更新 (此时返回的是滤波后的值)
                        dist_yaw_moment = ndo.update(vx, vy, r, delta_rad)

                        # D. 计算总差动扭矩需求
                        Rw = veh_params['Rw']
                        lw = veh_params['lw']
                        raw_diff_torque_req = -dist_yaw_moment * 2 * Rw / lw

                        # [核心新增] 死区处理 (Dead Zone)
                        # 如果需要的补偿力矩小于 10 Nm (根据实际情况调整)，则忽略
                        dead_zone_threshold = 0.0
                        if abs(raw_diff_torque_req) < dead_zone_threshold:
                            raw_diff_torque_req = 0.0

                        max_wheel_torque = 298.0
                        safe_margin = 0.9
                        limit = max_wheel_torque * 2 * safe_margin
                        current_diff_torque_req = np.clip(raw_diff_torque_req, -limit, limit)

                        # E. 计算增量并注入
                        diff_torque_increment = current_diff_torque_req - last_diff_torque
                        last_diff_torque = current_diff_torque_req
                    elif feed_mode == "PD":
                        # 1. 获取误差信号
                        # State-2 是横向位置 (y)，我们假设参考线是 0
                        # State-5 是横向速度 (vy)
                        lat_error = state.robot_state[1]  # 或者是 state_vec[1]，请确认 y 的索引
                        lat_vel = state.robot_state[4]  # vy

                        # 如果你有参考轨迹，应该用 (y - y_ref)
                        # 这里假设跑直线，y_ref = 0

                        # 2. 计算回正力矩 (PD控制)
                        # 逻辑：车偏左 (y>0) -> 需要向右力矩 (M < 0) -> 符号取负
                        # 逻辑：车向左飞 (vy>0) -> 需要反向阻尼 -> 符号取负
                        comp_moment = - Kp * lat_error - Kd * lat_vel

                        # 3. 力矩分配 (转化为左右轮差动)
                        # M = (Fr - Fl) * lw/2
                        # Diff_Force = M / (lw/2)
                        # Diff_Torque = Diff_Force * Rw
                        Rw = veh_params['Rw']
                        lw = veh_params['lw']

                        diff_torque_req = comp_moment * 2 * Rw / lw

                        # 4. 限幅与安全
                        max_comp = 100.0  # 允许的最大额外救车力矩
                        diff_torque_increment = np.clip(diff_torque_req, -max_comp, max_comp)
                    else:
                        print("请选择补偿模式")
                        diff_torque_increment = 0
                    action[0] -= diff_torque_increment / 2.0
                    action[1] += diff_torque_increment / 2.0
                    action[2] -= diff_torque_increment / 2.0
                    action[3] += diff_torque_increment / 2.0
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
            calctime_list.append(calc_time * 1000)

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
            # 传入当前策略索引 i
            eval_dict, tracking_dict = self.run_an_episode(
                env, networks, self.init_info, is_opt=False, render=False, policy_index=i
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
