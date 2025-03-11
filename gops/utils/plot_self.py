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
# font = FontProperties(fname="SimHei.ttf", size=15)
default_cfg = dict()
default_cfg["fig_size"] = (12, 9)
default_cfg["dpi"] = 300
default_cfg["pad"] = 0.5

default_cfg["tick_size"] = 20
default_cfg["tick_label_font"] = "Times New Roman"
default_cfg["legend_font"] = {
    "family": "Times New Roman",#, SimHei
    "size": "15",
    "weight": "normal",
}
default_cfg["label_font"] = {
    "size": "20",  # ch:30
    "weight": "normal",
"family": "Times New Roman",
}
default_cfg["img_fmt"] = "png"
# mpl.rcParams['font.sans-serif'] = ['SimSun']  # 指定宋体
# mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
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
    start_index = 2
    end_index = -1
    interval = 1
    step_list = np.array(data_result.iloc[0, start_index:end_index:interval], dtype='float32')

    data_pool = np.zeros((line_num+1, len(step_list)))
    data_pool[0, :] = step_list

    data_pool[1, :] = np.array(data_result.iloc[4, start_index:end_index:interval], dtype='float32')  # Ref
    data_pool[2, :] = np.array(data_result.iloc[3, start_index:end_index:interval], dtype='float32')  # MPC
    data_pool[3, :] = np.array(data_result.iloc[2, start_index:end_index:interval], dtype='float32')  # FHADP
    data_pool[4, :] = np.array(data_result.iloc[1, start_index:end_index:interval], dtype='float32')  # Bilevel
    # for num in range(line_num):
    #     data_numi = np.array(data_result.iloc[num+1, 1:], dtype='float32')
    #     data_pool[num+1, :] = data_numi
    return data_pool

def read_csv_line4_from2path(root_path1, root_path2,line_num):
    data_result = pd.DataFrame(pd.read_csv(root_path1, header=None))
    data_result2 = pd.DataFrame(pd.read_csv(root_path2, header=None))
    start_index = 1
    end_index = 1000
    interval = 1
    step_list = np.array(data_result.iloc[0, start_index:end_index:interval], dtype='float32')

    data_pool = np.zeros((line_num+1, len(step_list)))
    data_pool[0, :] = step_list
    data_pool[1, :] = np.array(data_result.iloc[3, start_index:end_index:interval], dtype='float32')  # Ref
    data_pool[2, :] = np.array(data_result2.iloc[1, start_index:end_index:interval], dtype='float32')  # MPC
    data_pool[3, :] = np.array(data_result.iloc[1, start_index:end_index:interval], dtype='float32')  # FHADP
    data_pool[4, :] = np.array(data_result.iloc[2, start_index:end_index:interval], dtype='float32')  # Bilevel
    # for num in range(line_num):
    #     data_numi = np.array(data_result.iloc[num+1, 1:], dtype='float32')
    #     data_pool[num+1, :] = data_numi
    # print(data_pool[2, :])
    return data_pool

def read_csv_line3(root_path, line_num):
    data_result = pd.DataFrame(pd.read_csv(root_path, header=None))
    start_index = 1
    end_index = 1000
    interval = 5
    step_list = np.array(data_result.iloc[0, start_index:end_index:interval], dtype='float32')

    data_pool = np.zeros((line_num+1, len(step_list)))
    data_pool[0, :] = step_list

    data_pool[1, :] = np.array(data_result.iloc[3, start_index:end_index:interval], dtype='float32')/10#3.14*180  # MPC
    data_pool[2, :] = np.array(data_result.iloc[2, start_index:end_index:interval], dtype='float32')/10#3.14*180 # FHADP
    data_pool[3, :] = np.array(data_result.iloc[1, start_index:end_index:interval], dtype='float32')/10#3.14*180  # ABPO
    # data_pool[4, :] = np.array(data_result.iloc[2, start_index:end_index:interval], dtype='float32')  # Bilevel
    # for num in range(line_num):
    #     data_numi = np.array(data_result.iloc[num+1, 1:], dtype='float32')
    #     data_pool[num+1, :] = data_numi
    return data_pool

def read_csv_line3_from2path(root_path, root_path2, line_num):
    data_result = pd.DataFrame(pd.read_csv(root_path, header=None))
    data_result2 = pd.DataFrame(pd.read_csv(root_path2, header=None))
    start_index = 1
    end_index = 1000
    interval = 5
    step_list = np.array(data_result.iloc[0, start_index:end_index:interval], dtype='float32')

    data_pool = np.zeros((line_num+1, len(step_list)))
    data_pool[0, :] = step_list

    data_pool[1, :] = np.array(data_result2.iloc[1, start_index:end_index:interval], dtype='float32')/10#3.14*180  # MPC
    data_pool[2, :] = np.array(data_result.iloc[2, start_index:end_index:interval], dtype='float32')#3.14*180 # FHADP
    data_pool[3, :] = np.array(data_result.iloc[1, start_index:end_index:interval], dtype='float32')#3.14*180  # ABPO
    # data_pool[4, :] = np.array(data_result.iloc[2, start_index:end_index:interval], dtype='float32')  # Bilevel
    # for num in range(line_num):
    #     data_numi = np.array(data_result.iloc[num+1, 1:], dtype='float32')
    #     data_pool[num+1, :] = data_numi
    return data_pool

def read_csv_line2(root_path, line_num):
    data_result = pd.DataFrame(pd.read_csv(root_path, header=None))
    start_index = 1
    end_index = -1
    interval = 1
    step_list = np.array(data_result.iloc[0, start_index:end_index:interval], dtype='float32')

    data_pool = np.zeros((line_num+1, len(step_list)))
    data_pool[0, :] = step_list

    data_pool[1, :] = np.array(data_result.iloc[1, start_index:end_index:interval], dtype='float32')/10#3.14*180  # MPC
    data_pool[2, :] = np.array(data_result.iloc[2, start_index:end_index:interval], dtype='float32')/10#3.14*180 # FHADP

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
        lw = 2
        if i == 1:
            lw=4
        sns.lineplot(x=data_read[0, :]*dt, y=data_read[i+1, :], linewidth=lw, color="{}".format(color), label="{}".format(legend)) #
        # plt.scatter(x=data_x[i + 1, :], y=data_[i + 1, :], label="{}".format(legend), s=2)
    # x = [0, 2.5, 5.0, 7.5]
    # plt.xticks(x)
    # plt.yticks(range(0,50000,10000))
    plt.tick_params(labelsize=default_cfg["tick_size"])
    # 使用 ax.tick_params 来设置刻度线方向
    ax.tick_params(axis='x', direction='in')  # x轴刻度线向内
    ax.tick_params(axis='y', direction='in')  # y轴刻度线向内
    plt.rcParams['font.sans-serif'] = ['SimSun']
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
    if args["language"] == "ch":
        plt.xlabel(args["x_label"], default_cfg["label_font"], fontproperties=zhfont1, fontsize=20) #
        plt.ylabel(args["y_label"], default_cfg["label_font"], fontproperties=zhfont1, fontsize=20) #
    else:
        plt.xlabel(args["x_label"], default_cfg["label_font"])
        plt.ylabel(args["y_label"], default_cfg["label_font"])
    plt.legend(loc="best", prop=default_cfg["legend_font"])#
    plt.legend(frameon=False) # 不显示图例框线
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
    # plt.axis('equal')
    # plt.legend(ncol=2)
    plt.tick_params(labelsize=default_cfg["tick_size"])
    # 使用 ax.tick_params 来设置刻度线方向
    ax.tick_params(axis='x', direction='in')  # x轴刻度线向内
    ax.tick_params(axis='y', direction='in')  # y轴刻度线向内
    plt.rcParams['font.sans-serif'] = ['SimSun']
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
    if args["language"] == "ch":
        plt.xlabel(args["x_label"], default_cfg["label_font"], fontproperties=zhfont1, fontsize=20) #
        plt.ylabel(args["y_label"], default_cfg["label_font"], fontproperties=zhfont1, fontsize=20) #
    else:
        plt.xlabel(args["x_label"], default_cfg["label_font"])
        plt.ylabel(args["y_label"], default_cfg["label_font"])
    plt.legend(loc="best", prop=default_cfg["legend_font"])
    plt.legend(frameon=False)
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
    save_dir = args["figures_root"] + '/run_plot_ch/'
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
    ax.tick_params(axis='x', direction='in')  # x轴刻度线向内
    ax.tick_params(axis='y', direction='in')  # y轴刻度线向内
    fig.tight_layout(pad=default_cfg["pad"])
    # plt.xlim(0, 19)
    # plt.show()
    plt.savefig(
        path_state_fmt, format=default_cfg["img_fmt"], bbox_inches="tight"
    )
    plt.savefig(
        path_state_fmt_o, format="pdf", bbox_inches="tight"
    )
    plt.close()

def TAR_with_error_bands():
    dir = '../../results/pyth_stabilitycontrol_cstr/'
    # data_result_1 = pd.DataFrame(pd.read_csv(dir + "/csv/Loss_loss_actor-20000.csv"))
    data_result_1 = pd.DataFrame(pd.read_csv(dir + "FHADP2Lagrangian_250108-085550/data/FHADP2Lagrangian_250108-085550.csv"))
    # data_result_2 = pd.DataFrame(pd.read_csv(dir + "FHADP2Lagrangian_250113-195414/data/FHADP2Lagrangian_250113-195414.csv"))
    # data_result_4 = pd.DataFrame(pd.read_csv(dir + "1209-123707-Np50-3la-50000it/csv/Loss_loss_actor-20000.csv"))
    # data_result_5 = pd.DataFrame(pd.read_csv(dir + "1210-205615-Np50-3la-50000it/csv/Loss_loss_actor-20000.csv"))
    data_result_6 = pd.DataFrame(pd.read_csv(dir + "TRANSStolenMpcLagrangian_250108-105229/data/TRANSStolenMpcLagrangian_250108-105229.csv"))
    # data_result_7 = pd.DataFrame(pd.read_csv(dir + "TRANSStolenMpcLagrangian_250118-093331/data/TRANSStolenMpcLagrangian_250118-093331.csv"))
    # data_result_8 = pd.DataFrame(pd.read_csv(dir + "1225-092251-Np100-5la-20000it-5_4e-2_2e-5/csv/Evaluation_1. TAR-RL iteration.csv"))
    # data_result_9 = pd.DataFrame(pd.read_csv(dir + "1225-231417-Np100-5la-20000it-5_4e-2_2e-5/csv/Evaluation_1. TAR-RL iteration.csv"))
    # data_result_10 = pd.DataFrame(pd.read_csv(dir + "1226-125715-Np100-5la-20000it-5_4e-2_2e-5/csv/Evaluation_1. TAR-RL iteration.csv"))
    # data_set1 = pd.concat([data_result_1, data_result_2], ignore_index=True)
    # data_set1 = data_set1.append(data_result_3, ignore_index=True)
    # data_set1 = data_set1.append(data_result_4, ignore_index=True)
    # data_set1 = data_set1.append(data_result_5, ignore_index=True)
    # data_set2 = pd.concat([data_result_6, data_result_7], ignore_index=True)
    # data_set2 = data_set2.append(data_result_8, ignore_index=True)
    # data_set2 = data_set2.append(data_result_9, ignore_index=True)
    # data_set2 = data_set2.append(data_result_10, ignore_index=True)
    plt.figure("-tar_with_error_bands")
    ax = plt.gca()
    sns.set_theme(style="darkgrid")
    line_50 = sns.lineplot(x="Step", y="Value", data=data_result_1, errorbar=('ci', 95), lw=2) #,hue="region", style="event"
    line_100 = sns.lineplot(x="Step", y="Value", data=data_result_6, errorbar=('ci', 50), lw=2)  # ,hue="region", style="event"

    plt.ylabel("Average return", fontsize=12)
    plt.xlabel("Iteration", fontsize=12)
    plt.tick_params(labelsize=12)
    # plt.grid(axis='both', ls='-.')
    # plt.legend(handles=[line_50, line_100], labels=['50', '100'], prop={'size': 7}, loc=2)
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.subplots_adjust(bottom=0.31)
    plt.rcParams['axes.unicode_minus'] = False
    plt.savefig(os.path.join(dir, "-loss_with_error_bands.pdf"))
    plt.show()

if __name__ == "__main__":
    # Parameters Setup
    parser = argparse.ArgumentParser()
    parser.add_argument("--time_step", type=float, default=0.01)
    parser.add_argument("--csv_file_name", type=str, default="Calc time")
    parser.add_argument("--line_num", type=int, default=3)
    parser.add_argument("--language", type=str, default="en")
    # parser.add_argument("--x_label", type=str, default=r"Pos $p_{\rm x}\ /\mathrm{m}$")
    # parser.add_argument("--y_label", type=str, default=r"Pos $p_{\rm y}\ /\mathrm{m}$")
    parser.add_argument("--x_label", type=str, default=r"Time $/\mathrm{s}$")
    # parser.add_argument("--y_label", type=str, default=r"Lateral error $p_y^{\rm err}\ /\mathrm{m}$")
    # parser.add_argument("--y_label", type=str, default=r"Yaw $\phi\ /\mathrm{rad}$")
    # parser.add_argument("--y_label", type=str, default=r"Yaw error $\phi^{\rm err}\ /\mathrm{rad}$")
    # parser.add_argument("--y_label", type=str, default=r"Velocity $v_x\ /\mathrm{m·s^{-1}}$")  # ·s^{-1}
    # parser.add_argument("--y_label", type=str, default=r"Velocity error $v_x^{\rm err}\ /\mathrm{m·s^{-1}}$")
    # parser.add_argument("--y_label", type=str, default=r"Velocity $v_y\ /\mathrm{m·s^{-1}}$")  # ·s^{-1}
    # parser.add_argument("--y_label", type=str, default=r"Yawrate $\dot\phi\ /\mathrm{rad·s^{-1}}$")
    # parser.add_argument("--y_label", type=str, default=r"Roll $\varphi\ /\mathrm{rad}$")#
    # parser.add_argument("--y_label", type=str, default=r"Rollrate $\dot\varphi\ /\mathrm{rad·s^{-1}}$")
    # parser.add_argument("--y_label", type=str, default=r"Torque $Q_{\rm rr}\ /\mathrm{N·m}$")  #
    # parser.add_argument("--y_label", type=str, default=r"Steering Angle $\delta\ /\mathrm{rad}$")  #
    parser.add_argument("--y_label", type=str, default=r"Computation time $/\mathrm{ms}$")  #
    # parser.add_argument("--y_label", type=str, default=r"Constraint $|\beta|$")  #
    # parser.add_argument("--y_label", type=str, default=r"Constraint $|\dot\phi|$")  #
    # parser.add_argument("--y_label", type=str, default=r"Roll index $I_{\rm rs}$")#

    # parser.add_argument("--x_label", type=str, default=r"$p_{\rm x,tt}\ /\mathrm{m}$")
    # parser.add_argument("--y_label", type=str, default=r"$p_{\rm y,tt}\ /\mathrm{m}$")
    # parser.add_argument("--x_label", type=str, default=r"时间 $/\mathrm{s}$")
    # parser.add_argument("--y_label", type=str, default=r" $u_{\rm tt}\ /\mathrm{(m·s^{-1})}$")
    # parser.add_argument("--y_label", type=str, default=r"速度误差$u_{\rm tt}^{\rm err}\ /\mathrm{(m·s^{-1})}$")
    # parser.add_argument("--y_label", type=str, default=r"横向误差$p_{\rm y,tl}^{\rm err}\ /\mathrm{m}$")
    # parser.add_argument("--y_label", type=str, default=r"$\phi_{\rm tt}\ /\degree$")
    # parser.add_argument("--y_label", type=str, default=r"横摆角误差$\phi_{\rm tt}^{\rm err}\ /\degree$")
    # parser.add_argument("--y_label", type=str, default=r"$\dot\phi_{\rm tt}/\mathrm{(rad·s^{-1})}$")
    # parser.add_argument("--y_label", type=str, default=r"$\varphi_{\rm tt}\ /\degree$") #
    # parser.add_argument("--y_label", type=str, default=r"$\dot\varphi_{\rm tt}\ /\mathrm{(rad·s^{-1})}$")
    # parser.add_argument("--y_label", type=str, default=r"$v_{\rm tt}\ /\mathrm{(m·s^{-1})}$") # ·s^{-1}
    # parser.add_argument("--y_label", type=str, default=r"$\delta_{\rm tt}\ /\degree$")  #
    # parser.add_argument("--y_label", type=str, default=r"$a_{x,\rm tt}\ /\mathrm{(m·s^{-2})}$")  #
    # parser.add_argument("--y_label", type=str, default=r"单步计算时间 $/\mathrm{ms}$")  #

    # parser.add_argument("--y_label", type=str, default=r"$J_\mathrm{L}$")
    # parser.add_argument("--x_label", type=str, default=r"$M$")

    parser.add_argument("--legend_list", type=list, default=
    ["OnMPC","CMMPC", "CTMPC"])#"参考状态","Ref",
    parser.add_argument("--color_list", type=list, default=
    ["#8A2BE2","lime", "magenta"]) #"#FA8072","b",
    parser.add_argument("--figures_root", type=str,
                        default='../../figures/TRANSStolenMpcLagrangian-FHADP2Lagrangian-pyth_stabilitycontrol_cstr/250308-111824_carsim/')
    # parser.add_argument("--figures_root", type=str,default='../../results/pyth_semitruckpu7dof/FHADP2_240426-091408-upper_20-inner_50000/')
    parser.add_argument("--figures_root2", type=str,
                        default='../../figures/MPC-pyth_stabilitycontrol_cstr/250308-132258/')
    # Get parameter dictionary
    args = vars(parser.parse_args())

    read_path_datax = args["figures_root"] + "State-1.csv"
    read_path_datax2 = args["figures_root2"] + "State-1.csv"
    read_path = args["figures_root"]+args["csv_file_name"]+".csv"
    read_path2 = args["figures_root2"] + args["csv_file_name"] + ".csv"
    if args["line_num"]==5:
        data_csv = read_csv_line5(read_path, args["line_num"])
        read_datax = read_csv_line5(read_path_datax, args["line_num"])

    elif args["line_num"]==4:
        # data_csv = read_csv_line4(read_path, args["line_num"])
        # read_datax = read_csv_line4(read_path_datax, args["line_num"])
        data_csv = read_csv_line4_from2path(read_path, read_path2, args["line_num"])
        read_datax = read_csv_line4_from2path(read_path_datax, read_path_datax2, args["line_num"])

    elif args["line_num"]==3:
        # data_csv = read_csv_line3(read_path, args["line_num"])
        # read_datax = read_csv_line3(read_path_datax, args["line_num"])
        data_csv = read_csv_line3_from2path(read_path, read_path2, args["line_num"])
        read_datax = read_csv_line3_from2path(read_path_datax,read_path_datax2, args["line_num"])

    elif args["line_num"]==2:
        data_csv = read_csv_line2(read_path, args["line_num"])
        read_datax = read_csv_line2(read_path_datax, args["line_num"])

    if args["x_label"] == r"Time $/\mathrm{s}$" or args["x_label"] == r"时间 $/\mathrm{s}$":
        plot_Timevs_(data_csv, args)
    elif args["x_label"] == r"State Pos X $p_{\rm x,tt}\ /\mathrm{m}$" or r"State Pos X $p_{\rm x,tl}\ /\mathrm{m}$"\
            or r"横向位置 X $p_{\rm x,tt}\ /\mathrm{m}$"or r"横向位置 X $p_{\rm x,tl}\ /\mathrm{m}$" or r"Pos $p_{\rm x}\ /\mathrm{m}$":
        plot_stateXvs_(read_datax, data_csv, args)
    else:
        print("please set the x label")

    # read_upper_loss = args["figures_root"] + "loss upper.csv"
    # data_csv = read_csv_line1(read_upper_loss, args["line_num"])
    # plot_upperloss(data_csv, args)
    # TAR_with_error_bands()
