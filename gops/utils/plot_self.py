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
from tkinter import *
from matplotlib.legend_handler import HandlerPathCollection, HandlerLine2D
zhfont1 = fm.FontProperties(fname='./SIMSUN.ttf')
y_formatter = FormatStrFormatter('%1')

default_cfg = dict()
default_cfg["fig_size"] = (12, 9)
default_cfg["dpi"] = 300
default_cfg["pad"] = 0.5

default_cfg["tick_size"] = 20
default_cfg["tick_label_font"] = "Times New Roman"
default_cfg["legend_font"] = {
    "family": "Times New Roman",
    "size": "15",
    "weight": "normal",
}
default_cfg["label_font"] = {
    "size": "20",  # ch:30
    "weight": "normal",
"family": "Times New Roman",
}
default_cfg["img_fmt"] = "png"

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)

def read_csv_line1(root_path, line_num):
    data_result = pd.DataFrame(pd.read_csv(root_path, header=None))
    start_index = 1
    end_index = -1
    interval = 1
    step_list = np.array(data_result.iloc[start_index:end_index:interval, 0], dtype='float32')
    data_pool = np.zeros((line_num + 1, len(step_list)))
    data_pool[0, :] = step_list
    data_pool[1, :] = np.array(data_result.iloc[start_index:end_index:interval, 1], dtype='float32')
    return data_pool

def read_csv_line5(root_path, line_num):
    data_result = pd.DataFrame(pd.read_csv(root_path, header=None))
    start_index = 1
    end_index = -1
    interval = 1
    step_list = np.array(data_result.iloc[0, start_index:end_index:interval], dtype='float32')
    data_pool = np.zeros((line_num+1, len(step_list)))
    data_pool[0, :] = step_list

    data_pool[1, :] = np.array(data_result.iloc[5, start_index:end_index:interval], dtype='float32') # Ref
    data_pool[2, :] = np.array(data_result.iloc[3, start_index:end_index:interval], dtype='float32')  # MPC
    data_pool[3, :] = np.array(data_result.iloc[1, start_index:end_index:interval], dtype='float32')  # FHADP
    data_pool[4, :] = np.array(data_result.iloc[4, start_index:end_index:interval], dtype='float32')  # PDP
    data_pool[5, :] = np.array(data_result.iloc[2, start_index:end_index:interval], dtype='float32')  # Bilevel
    # for num in range(line_num):
    #     data_numi = np.array(data_result.iloc[num+1, 1:], dtype='float32')
    #     data_pool[num+1, :] = data_numi
    return data_pool

def read_csv_line4(root_path, line_num):
    data_result = pd.DataFrame(pd.read_csv(root_path, header=None))
    start_index = 1
    end_index = -1
    interval = 1
    step_list = np.array(data_result.iloc[0, start_index:end_index:interval], dtype='float32')

    data_pool = np.zeros((line_num+1, len(step_list)))
    data_pool[0, :] = step_list

    data_pool[1, :] = np.array(data_result.iloc[5, start_index:end_index:interval], dtype='float32')  # Ref
    data_pool[2, :] = np.array(data_result.iloc[3, start_index:end_index:interval], dtype='float32')  # MPC
    data_pool[3, :] = np.array(data_result.iloc[1, start_index:end_index:interval], dtype='float32')  # FHADP
    data_pool[4, :] = np.array(data_result.iloc[2, start_index:end_index:interval], dtype='float32')  # Bilevel
    # for num in range(line_num):
    #     data_numi = np.array(data_result.iloc[num+1, 1:], dtype='float32')
    #     data_pool[num+1, :] = data_numi
    return data_pool

def read_csv_line3(root_path, line_num):
    data_result = pd.DataFrame(pd.read_csv(root_path, header=None))
    start_index = 0
    end_index = -1
    interval = 20
    step_list = np.array(data_result.iloc[0, start_index:end_index:interval], dtype='float32')

    data_pool = np.zeros((line_num+1, len(step_list)))
    data_pool[0, :] = step_list

    data_pool[1, :] = np.array(data_result.iloc[3, start_index:end_index:interval], dtype='float32')/10  # MPC
    data_pool[2, :] = np.array(data_result.iloc[1, start_index:end_index:interval], dtype='float32')/10 # FHADP
    data_pool[3, :] = np.array(data_result.iloc[2, start_index:end_index:interval], dtype='float32')/10  # PDP
    # data_pool[4, :] = np.array(data_result.iloc[2, start_index:end_index:interval], dtype='float32')  # Bilevel
    # for num in range(line_num):
    #     data_numi = np.array(data_result.iloc[num+1, 1:], dtype='float32')
    #     data_pool[num+1, :] = data_numi
    return data_pool

def plot_Timevs_(data_read, args):
    dt = args["time_step"]
    legend_list = args["legend_list"]
    color_list = args["color_list"]
    line_num = args["line_num"]
    language = args["language"]
    save_dir = args["figures_root"]+'/run_plot_'+language+"/"
    os.makedirs(save_dir, exist_ok=True)
    path_state_fmt = os.path.join(
        save_dir, "Times-"+args["csv_file_name"]+".{}".format(default_cfg["img_fmt"])
    )
    path_state_fmtpdf = os.path.join(
        save_dir, "Times-"+args["csv_file_name"] + ".{}".format("pdf")
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
        color = (
            color_list[i]
            if len(color_list) == line_num
            else print("line_num != len of legend_list")
        )
        sns.lineplot(x=data_read[0, :]*dt, y=data_read[i+1, :], label="{}".format(legend), linewidth=2, color="{}".format(color)) #
        # plt.scatter(x=data_x[i + 1, :], y=data_[i + 1, :], label="{}".format(legend), s=2)
    # x = [0, 5, 10, 15, 20]
    # plt.yticks(range(0,50000,10000))
    plt.tick_params(labelsize=default_cfg["tick_size"])
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
    if args["language"] == "ch":
        plt.xlabel(args["x_label"], default_cfg["label_font"], fontproperties=zhfont1, fontsize=20) #
        plt.ylabel(args["y_label"], default_cfg["label_font"], fontproperties=zhfont1, fontsize=20) #
    else:
        plt.xlabel(args["x_label"], default_cfg["label_font"])
        plt.ylabel(args["y_label"], default_cfg["label_font"])
    plt.legend(loc="best", prop=default_cfg["legend_font"])
    fig.tight_layout(pad=default_cfg["pad"])
    plt.savefig(
        path_state_fmt, format=default_cfg["img_fmt"], bbox_inches="tight"
    )
    plt.savefig(
        path_state_fmtpdf, format="pdf", bbox_inches="tight"
    )

    plt.close()

def plot_stateXvs_(data_x, data_, args):
    legend_list = args["legend_list"]
    color_list = args["color_list"]
    line_num = args["line_num"]
    language = args["language"]
    save_dir = args["figures_root"]+'/run_plot_'+language+"/"
    os.makedirs(save_dir, exist_ok=True)
    path_state_fmt = os.path.join(
        save_dir, "stateX-"+args["csv_file_name"]+".{}".format(default_cfg["img_fmt"])
    )
    path_state_fmt_o = os.path.join(
        save_dir, "stateX-"+args["csv_file_name"]+".{}".format("pdf")
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
        color = (
            color_list[i]
            if len(color_list) == line_num
            else print("line_num != len of legend_list")
        )

        # sns.lineplot(x=data_x[i+1, :], y=data_[i+1, :], label="{}".format(legend), linewidth=2,color="{}".format(color))  #
        plt.scatter(x=data_x[i+1, :], y=data_[i+1, :], label="{}".format(legend), s=2, c="{}".format(color)) #, linewidths=0.1
    # x = [0, 5, 10, 15, 20] sns.lineplot
    # plt.xticks(range(0,8,1))
    plt.axis('equal')
    plt.tick_params(labelsize=default_cfg["tick_size"])
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
    if args["language"] == "ch":
        plt.xlabel(args["x_label"], default_cfg["label_font"], fontproperties=zhfont1, fontsize=20) #
        plt.ylabel(args["y_label"], default_cfg["label_font"], fontproperties=zhfont1, fontsize=20) #
    else:
        plt.xlabel(args["x_label"], default_cfg["label_font"])
        plt.ylabel(args["y_label"], default_cfg["label_font"])
    plt.legend(loc="best", prop=default_cfg["legend_font"])
    fig.tight_layout(pad=default_cfg["pad"])
    # plt.show()
    plt.savefig(
        path_state_fmt, format=default_cfg["img_fmt"], bbox_inches="tight"
    )
    plt.savefig(
        path_state_fmt_o, format="pdf", bbox_inches="tight"
    )
    plt.close()

def plot_upperloss(data, args):
    line_num = args["line_num"]
    save_dir = args["figures_root"] + '/run_plot_en/'
    os.makedirs(save_dir, exist_ok=True)
    path_state_fmt = os.path.join(
        save_dir, "stateX-" + args["csv_file_name"] + ".{}".format(default_cfg["img_fmt"])
    )
    path_state_fmt_o = os.path.join(
        save_dir, "stateX-" + args["csv_file_name"] + ".{}".format("pdf")
    )
    fig_size = (
        default_cfg["fig_size"],
        default_cfg["fig_size"],
    )
    fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])
    sns.lineplot(x=data[0, :], y=data[1, :], linewidth=2)
    plt.tick_params(labelsize=default_cfg["tick_size"])
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
    plt.xlabel(args["x_label"], default_cfg["label_font"])  #, fontproperties=zhfont1
    plt.ylabel(args["y_label"], default_cfg["label_font"])  #, fontproperties=zhfont1
    fig.tight_layout(pad=default_cfg["pad"])
    # plt.show()
    plt.savefig(
        path_state_fmt, format=default_cfg["img_fmt"], bbox_inches="tight"
    )
    plt.savefig(
        path_state_fmt_o, format="pdf", bbox_inches="tight"
    )
    plt.close()

if __name__ == "__main__":
    # Parameters Setup
    parser = argparse.ArgumentParser()
    parser.add_argument("--time_step", type=float, default=0.01)
    parser.add_argument("--csv_file_name", type=str, default="Calc time")
    parser.add_argument("--line_num", type=int, default=3)
    parser.add_argument("--language", type=str, default="ch")
    # parser.add_argument("--x_label", type=str, default=r"Pos $p_{\rm x,tl}\ /\mathrm{m}$")
    # parser.add_argument("--y_label", type=str, default=r"Pos $p_{\rm y,tl}\ /\mathrm{m}$")
    # parser.add_argument("--x_label", type=str, default=r"Time $/\mathrm{s}$")
    # parser.add_argument("--y_label", type=str, default=r"Lateral error $p_{\rm y,tt}^{\rm err}\ /\mathrm{m}$")
    # parser.add_argument("--y_label", type=str, default=r"Yaw $\phi_{\rm tl}\ /\mathrm{rad}$")
    # parser.add_argument("--y_label", type=str, default=r"Yaw error $\phi_{\rm tt}^{\rm err}\ /\mathrm{rad}$")
    # parser.add_argument("--y_label", type=str, default=r"Yawrate $\dot\phi_{\rm tt}\ /\mathrm{rad·s^{-1}}$")
    # parser.add_argument("--y_label", type=str, default=r"Roll $\varphi_{\rm tt}\ /\mathrm{rad}$") #
    # parser.add_argument("--y_label", type=str, default=r"Rollrate $\dot\varphi_{\rm tt}\ /\mathrm{rad·s^{-1}}$")
    # parser.add_argument("--y_label", type=str, default=r"Lateral speed $v_{\rm tt}\ /\mathrm{m·s^{-1}}$") # ·s^{-1}
    # parser.add_argument("--y_label", type=str, default=r"Steering Angle $\delta_{\rm tt}\ /\mathrm{rad}$")  #
    # parser.add_argument("--y_label", type=str, default=r"Calculation time $/\mathrm{ms}$")  #

    # parser.add_argument("--x_label", type=str, default=r"横向位置$p_{\rm x,tt}\ /\mathrm{m}$")
    # parser.add_argument("--y_label", type=str, default=r"纵向位置$p_{\rm y,tt}\ /\mathrm{m}$")
    parser.add_argument("--x_label", type=str, default=r"时间 $/\mathrm{s}$")
    # parser.add_argument("--y_label", type=str, default=r"纵向误差 $p_{\rm x,tl}^{\rm err}\ /\mathrm{m}$")
    # parser.add_argument("--y_label", type=str, default=r"横向误差$p_{\rm y,tl}^{\rm err}\ /\mathrm{m}$")
    # parser.add_argument("--y_label", type=str, default=r"横摆角$\phi_{\rm tt}\ /\mathrm{rad}$")
    # parser.add_argument("--y_label", type=str, default=r"横摆角误差$\phi_{\rm tl}^{\rm err}\ /\mathrm{rad}$")
    # parser.add_argument("--y_label", type=str, default=r"横摆角速度$\dot\phi_{\rm tt}/\mathrm{rad·s^{-1}}$")
    # parser.add_argument("--y_label", type=str, default=r"侧倾角$\varphi_{\rm tl}\ /\mathrm{rad}$") #
    # parser.add_argument("--y_label", type=str, default=r"侧倾角速度$\dot\varphi_{\rm tl}\ /\mathrm{rad·s^{-1}}$")
    # parser.add_argument("--y_label", type=str, default=r"横向速度$v_{\rm tt}\ /\mathrm{m·s^{-1}}$") # ·s^{-1}
    # parser.add_argument("--y_label", type=str, default=r"前轮转角$\delta_{\rm tt}\ /\mathrm{rad}$")  #
    parser.add_argument("--y_label", type=str, default=r"单步计算时间 $/\mathrm{ms}$")  #

    # parser.add_argument("--y_label", type=str, default=r"Upper loss $L$")
    # parser.add_argument("--x_label", type=str, default=r"Iteration $I_u$")

    parser.add_argument("--legend_list", type=list, default=
    ["MPC", "FHADP", "ABPO"])#, "PDP""Ref",
    parser.add_argument("--color_list", type=list, default=
    ["#8A2BE2", "lime", "magenta"]) #"#FA8072","b",
    parser.add_argument("--figures_root", type=str,
                        default='../../figures/FHADP2-FHADP2-pyth_semitruckpu7dof/240909-214411-dlc/240921-154439/')
    # parser.add_argument("--figures_root", type=str,default='../../results/pyth_semitruckpu7dof/FHADP2_240426-091408-upper_20-inner_50000/')
    # Get parameter dictionary
    args = vars(parser.parse_args())
    read_path_datax = args["figures_root"] + "State-1.csv"
    read_path = args["figures_root"]+args["csv_file_name"]+".csv"
    if args["line_num"]==5:
        data_csv = read_csv_line5(read_path, args["line_num"])
        read_datax = read_csv_line5(read_path_datax, args["line_num"])
    elif args["line_num"]==4:
        data_csv = read_csv_line4(read_path, args["line_num"])
        read_datax = read_csv_line4(read_path_datax, args["line_num"])
    elif args["line_num"]==3:
        data_csv = read_csv_line3(read_path, args["line_num"])
        read_datax = read_csv_line3(read_path_datax, args["line_num"])

    if args["x_label"] == r"Time $/\mathrm{s}$" or args["x_label"] == r"时间 $/\mathrm{s}$":
        plot_Timevs_(data_csv, args)
    elif args["x_label"] == r"State Pos X $p_{\rm x,tt}\ /\mathrm{m}$" or r"State Pos X $p_{\rm x,tl}\ /\mathrm{m}$"\
            or r"横向位置 X $p_{\rm x,tt}\ /\mathrm{m}$"or r"横向位置 X $p_{\rm x,tl}\ /\mathrm{m}$":
        plot_stateXvs_(read_datax, data_csv, args)
    else:
        print("please set the x label")

    # read_upper_loss = args["figures_root"] + "loss upper.csv"
    # data_csv = read_csv_line1(read_upper_loss, args["line_num"])
    # plot_upperloss(data_csv, args)
