import os
import matplotlib
import seaborn as sns
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import Auxiliary_System as AuxiSys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import matplotlib as mpl
import torch
import argparse
import matplotlib.patches as mpatches
import pandas as pd
import matplotlib.animation as animation
import matplotlib.font_manager as fm
from matplotlib.legend_handler import HandlerPathCollection, HandlerLine2D
zhfont1 = fm.FontProperties(fname='./resources/SIMSUN.ttf')

default_cfg = dict()
default_cfg["fig_size"] = (12, 9)
default_cfg["dpi"] = 300
default_cfg["pad"] = 0.5

default_cfg["tick_size"] = 8
default_cfg["tick_label_font"] = "Times New Roman"
default_cfg["legend_font"] = {
    "family": "Times New Roman",
    "size": "12",
    "weight": "normal",
}
default_cfg["label_font"] = {
    "family": "Times New Roman", #zhfont1
    "size": "9",
    "weight": "normal",
}
default_cfg["img_fmt"] = "png"


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)

def read_csv(root_path, line_num):
    data_result = pd.DataFrame(pd.read_csv(root_path, header=None))
    step_list = np.array(data_result.iloc[0, 1:], dtype='float32')
    data_pool = np.zeros((line_num+1, len(step_list)))
    data_pool[0, :] = step_list
    for num in range(line_num):
        data_numi = np.array(data_result.iloc[num+1, 1:], dtype='float32')
        data_pool[num+1, :] = data_numi

    return data_pool

def plot_single(data_read, args):
    dt = args["time_step"]
    legend_list = args["legend_list"]
    line_num = args["line_num"]
    save_dir = args["figures_root"]+'/run_plot_self/'
    os.makedirs(save_dir, exist_ok=True)
    path_state_fmt = os.path.join(
        save_dir, args["csv_file_name"]+".{}".format(default_cfg["img_fmt"])
    )
    fig_size = (
        default_cfg["fig_size"],
        default_cfg["fig_size"],
    )
    fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

    for i in range(line_num):
        legend = (
            legend_list[i]
            if len(legend_list) == line_num
            else print("line_num != len of legend_list")
        )
        sns.lineplot(x=data_read[0, :]*dt, y=data_read[i+1, :]) #, label="{}".format(legend), linewidth=2

    plt.tick_params(labelsize=default_cfg["tick_size"])
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
    plt.xlabel(args["x_label"], default_cfg["label_font"])
    plt.ylabel(args["y_label"], default_cfg["label_font"])
    # plt.legend(loc="best", prop=default_cfg["legend_font"])
    fig.tight_layout(pad=default_cfg["pad"])
    plt.savefig(
        path_state_fmt, format=default_cfg["img_fmt"], bbox_inches="tight"
    )
    plt.close()





if __name__ == "__main__":
    # Parameters Setup
    parser = argparse.ArgumentParser()
    parser.add_argument("--time_step", type=float, default=0.01)
    parser.add_argument("--csv_file_name", type=str, default="State-8")
    parser.add_argument("--line_num", type=int, default=4)
    parser.add_argument("--x_label", type=str, default=r"Time $/\mathrm{s}$")
    parser.add_argument("--y_label", type=str, default=r"$\dot\varphi_2 /\mathrm{rad/s}$")
    parser.add_argument("--legend_list", type=list, default=
    ["FHADP", "Bilevel", "PDP", "MPC"]) #, "Ref"
    parser.add_argument("--figures_root", type=str,
                        default='../../figures/FHADP2-FHADP2-pyth_semitruckpu7dof/240401-100557/')
    # Get parameter dictionary
    args = vars(parser.parse_args())
    read_path = args["figures_root"]+args["csv_file_name"]+".csv"
    data_csv = read_csv(read_path, args["line_num"])
    plot_single(data_csv, args)