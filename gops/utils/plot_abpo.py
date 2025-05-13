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
    "family": "Times New Roman",#,
    "size": "15",
    "weight": "normal",
}
default_cfg["label_font"] = {
    "size": "20",  # ch:30
    "weight": "normal",
"family": "Times New Roman",
}
default_cfg["img_fmt"] = "svg"
mpl.rcParams['font.sans-serif'] = ['SimSun']  # 指定宋体
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
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
    start_index = 2
    end_index = -1
    interval = 1
    step_list = np.array(data_result.iloc[0, start_index:end_index:interval], dtype='float32')
    data_pool = np.zeros((line_num+1, len(step_list)))
    data_pool[0, :] = step_list

    data_pool[1, :] = np.array(data_result.iloc[5, start_index:end_index:interval], dtype='float32') # Ref
    data_pool[2, :] = np.array(data_result.iloc[3, start_index:end_index:interval], dtype='float32')  # MPC
    data_pool[3, :] = np.array(data_result.iloc[4, start_index:end_index:interval], dtype='float32')  # FHADP
    data_pool[4, :] = np.array(data_result.iloc[1, start_index:end_index:interval], dtype='float32')  # PDP
    data_pool[5, :] = np.array(data_result.iloc[2, start_index:end_index:interval], dtype='float32')  # Bilevel
    # for num in range(line_num):
    #     data_numi = np.array(data_result.iloc[num+1, 1:], dtype='float32')
    #     data_pool[num+1, :] = data_numi
    return data_pool

def read_csv_line4(root_path, line_num):
    data_result = pd.DataFrame(pd.read_csv(root_path, header=None))
    start_index = 1
    end_index = 1199
    interval = 1
    step_list = np.array(data_result.iloc[0, start_index:end_index:interval], dtype='float32')

    data_pool = np.zeros((line_num+1, len(step_list)))
    data_pool[0, :] = step_list

    data_pool[1, :] = np.array(data_result.iloc[4, start_index:end_index:interval], dtype='float32')  # Ref
    data_pool[2, :] = np.array(data_result.iloc[3, start_index:end_index:interval], dtype='float32')  # MPC
    data_pool[3, :] = np.array(data_result.iloc[2, start_index:end_index:interval], dtype='float32')  # FHADP
    data_pool[4, :] = np.array(data_result.iloc[1, start_index:end_index:interval], dtype='float32')  # Bilevel
    # print(np.average(data_pool[1, :]),np.average(data_pool[2, :]), np.average(data_pool[3, :]), np.average(data_pool[4, :]))
    # for num in range(line_num):
    #     data_numi = np.array(data_result.iloc[num+1, 1:], dtype='float32')
    #     data_pool[num+1, :] = data_numi
    return data_pool

def read_csv_line3(root_path, line_num):
    data_result = pd.DataFrame(pd.read_csv(root_path, header=None))
    start_index = 1
    end_index = 1199
    interval = 1
    step_list = np.array(data_result.iloc[0, start_index:end_index:interval], dtype='float32')

    data_pool = np.zeros((line_num+1, len(step_list)))
    data_pool[0, :] = step_list

    data_pool[1, :] = np.array(data_result.iloc[3, start_index:end_index:interval], dtype='float32')#/3.14*180  # MPC
    data_pool[2, :] = np.array(data_result.iloc[2, start_index:end_index:interval], dtype='float32')#/3.14*180 # FHADP
    data_pool[3, :] = np.array(data_result.iloc[1, start_index:end_index:interval], dtype='float32')#/3.14*180  # ABPO
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
    save_dir = args["figures_root"]+'/run_plot_'+language+"sec/"
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

        sns.lineplot(x=data_read[0, :]*dt, y=data_read[i+1, :], linewidth=2, color="{}".format(color), label="{}".format(legend)) #
        # plt.scatter(x=data_x[i + 1, :], y=data_[i + 1, :], label="{}".format(legend), s=2)
    plt.xticks([0, 4,  8,  12])
    plt.xlim(0, 12)
    plt.yticks([25.6, 25.4, 25.2, 25.0])
    # plt.ylim(-40, 0)

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
        plt.legend(loc="best", prop=zhfont1)  # ,ncol=2
    else:
        plt.xlabel(args["x_label"], default_cfg["label_font"])
        plt.ylabel(args["y_label"], default_cfg["label_font"])
        plt.legend(loc="best", prop=default_cfg["legend_font"])  # ,ncol=2

    # plt.legend(frameon=False) # 不显示图例框线
    fig.tight_layout(pad=default_cfg["pad"])

    plt.savefig(
        path_state_fmt, format=default_cfg["img_fmt"], bbox_inches="tight"
    )
    # plt.savefig(
    #     path_state_fmtpdf, format="pdf", bbox_inches="tight"
    # )

    plt.close()

def plot_stateXvs_(data_x, data_, args):
    legend_list = args["legend_list"]
    color_list = args["color_list"]
    line_num = args["line_num"]
    language = args["language"]
    save_dir = args["figures_root"]+'/run_plot_'+language+"sec/"
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
    plt.xticks([0, 50, 100])
    plt.yticks([-200, -100, 0])
    # plt.axis('equal')
    # plt.legend(ncol=2)
    # plt.xlim(0, 100)
    # plt.ylim(-200, 0)
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
        plt.legend(loc="best", prop=zhfont1)
    else:
        plt.xlabel(args["x_label"], default_cfg["label_font"])
        plt.ylabel(args["y_label"], default_cfg["label_font"])
        plt.legend(loc="best", prop=default_cfg["legend_font"])
    # plt.legend(frameon=False)
    # fig.tight_layout(pad=default_cfg["pad"])
    # plt.show()
    plt.savefig(
        path_state_fmt, format=default_cfg["img_fmt"], bbox_inches="tight"
    )
    # plt.savefig(
    #     path_state_fmt_o, format="pdf", bbox_inches="tight"
    # )
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
    plt.xlim(0, 20)
    plt.ylim(None, 4)
    # plt.show()
    plt.savefig(
        path_state_fmt, format=default_cfg["img_fmt"], bbox_inches="tight"
    )
    plt.savefig(
        path_state_fmt_o, format="pdf", bbox_inches="tight"
    )
    plt.close()

def compute_Rmetrics(args):
    root_path = args["figures_root"] +"State-8.csv"
    root_path2 = args["figures_root"] + "State-12.csv"
    data_result = pd.DataFrame(pd.read_csv(root_path, header=None))
    data_result2 = pd.DataFrame(pd.read_csv(root_path2, header=None))
    start_index = 2
    end_index = -1
    interval = 1
    R_MPC = max(abs(np.array(data_result2.iloc[3, start_index:end_index:interval], dtype='float32')))/max(abs(np.array(data_result.iloc[3, start_index:end_index:interval], dtype='float32')))
    # R_MPC = remove_min_max(R_MPC.tolist())
    R_PUMPC = max(abs(np.array(data_result2.iloc[4, start_index:end_index:interval], dtype='float32'))) / max(abs(np.array(data_result.iloc[4, start_index:end_index:interval], dtype='float32')))
    # R_PUMPC = remove_min_max(R_PUMPC.tolist())
    R_FHADP = max(abs(np.array(data_result2.iloc[1, start_index:end_index:interval], dtype='float32')))/max(abs(np.array(data_result.iloc[1, start_index:end_index:interval], dtype='float32')))
    # R_FHADP = remove_min_max(R_FHADP.tolist())
    R_ABPO = max(abs(np.array(data_result2.iloc[2, start_index:end_index:interval], dtype='float32'))) / max(abs(np.array(data_result.iloc[2, start_index:end_index:interval], dtype='float32')))
    # R_ABPO = remove_min_max(R_ABPO.tolist())
    # print("absmaxR[MPC, PUMPC, FHADP, ABPO]:", max(abs(R_MPC)), max(abs(R_PUMPC)), max(abs(R_FHADP)), max(abs(R_ABPO)))
    print("avgR[MPC, PUMPC, FHADP, ABPO]:", R_MPC, R_PUMPC, R_FHADP, R_ABPO)

def compute_yoff_metrics(args):
    root_path = args["figures_root"] +"State-2.csv"
    root_path2 = args["figures_root"] + "State-5.csv"
    data_result = pd.DataFrame(pd.read_csv(root_path, header=None))
    data_result2 = pd.DataFrame(pd.read_csv(root_path2, header=None))
    start_index = 2
    end_index = -1
    interval = 1
    R_MPC = np.array(data_result2.iloc[3, start_index:end_index:interval], dtype='float32')-np.array(data_result.iloc[3, start_index:end_index:interval], dtype='float32')
    # R_MPC = remove_min_max(R_MPC.tolist())
    R_PUMPC = np.array(data_result2.iloc[4, start_index:end_index:interval], dtype='float32') -np.array(data_result.iloc[4, start_index:end_index:interval], dtype='float32')
    # R_PUMPC = remove_min_max(R_PUMPC.tolist())
    R_FHADP = np.array(data_result2.iloc[1, start_index:end_index:interval], dtype='float32')-np.array(data_result.iloc[1, start_index:end_index:interval], dtype='float32')
    # R_FHADP = remove_min_max(R_FHADP.tolist())
    R_ABPO = np.array(data_result2.iloc[2, start_index:end_index:interval], dtype='float32')-np.array(data_result.iloc[2, start_index:end_index:interval], dtype='float32')
    # R_ABPO = remove_min_max(R_ABPO.tolist())
    print("absmaxR[MPC, PUMPC, FHADP, ABPO]:", max(abs(R_MPC)), max(abs(R_PUMPC)), max(abs(R_FHADP)), max(abs(R_ABPO)))
    # print("avgR[MPC, PUMPC, FHADP, ABPO]:", R_MPC, R_PUMPC, R_FHADP, R_ABPO)

def compute_IR_metrics(args):
    root_path_varphi_tt = args["figures_root"] + "State-9.csv"
    root_path_varphidot_tt = args["figures_root"] +"State-10.csv"
    root_path_varphi_tl = args["figures_root"] + "State-13.csv"
    root_path_varphidot_tl = args["figures_root"] + "State-14.csv"
    data_result_varphi_tt = pd.DataFrame(pd.read_csv(root_path_varphi_tt, header=None))
    data_result_varphi_tl = pd.DataFrame(pd.read_csv(root_path_varphi_tl, header=None))
    data_result_varphidot_tt = pd.DataFrame(pd.read_csv(root_path_varphidot_tt, header=None))
    data_result_varphidot_tl = pd.DataFrame(pd.read_csv(root_path_varphidot_tl, header=None))


    start_index = 2
    end_index = -1
    interval = 1
    m1 = 5760.  # Total mass of the tractor [kg]
    m1s = 4455.  # Sprung mass of the tractor [kg]
    m2 = 20665  # Total mass of the semitrailer [kg]
    m2s = 20000  # Sprung mass of the semitrailer [kg]
    m1u = m1-m1s
    m2u = m2 - m2s
    h1s = 1.175  # Height of the CG of the sprung mass for the tractor [m]
    h2s = 2.125  # Height of the CG of the sprung mass for the semitrailer [m]
    kr1 = 9.1731e5  # roll stiffness of tire (front)[N/m]
    kr2 = 2.6023e6  # roll stiffness of tire (rear)[N/m]
    c1 = 1.2727e6  # Roll damping of the tractor's suspension [N-s/m]
    c2 = 4.1745e5  # Roll damping of the semitrailer's suspension [N-s/m]
    lw = 2.03
    g = 9.81
    h1r = 0.5
    h2r = 0.7
    h1u = 0.3
    h2u = 0.4
    C_varphi1 = 2 / (m1 * g * lw) * (kr1 * (1 + (m1s * h1r +m1u * h1u) /(m1s * h1s)) - (m1s * h1r +m1u * h1u) * g )
    C_varphi_dot1 = 2 * c1 / (m1 * g * lw ) * ((1 + (m1s * h1r + m1u * h1u) /(m1s * h1s)))

    C_varphi2 = 2 / (m2 * g * lw) * (kr2 * (1 + (m2s * h2r +m2u * h2u) /(m2s * h2s)) - (m2s * h2r +m2u * h2u) * g )
    C_varphi_dot2 = 2 * c2 / (m2 * g * lw ) * ((1 + (m2s * h2r + m2u * h2u) /(m2s * h2s)))


    R_MPC = max(max(C_varphi1*np.array(data_result_varphi_tt.iloc[3, start_index:end_index:interval], dtype='float32')+
                C_varphi_dot1*np.array(data_result_varphidot_tt.iloc[3, start_index:end_index:interval], dtype='float32')),
                max(C_varphi2*np.array(data_result_varphi_tl.iloc[3, start_index:end_index:interval], dtype='float32')+
                C_varphi_dot2*np.array(data_result_varphidot_tl.iloc[3, start_index:end_index:interval], dtype='float32')))

    R_PUMPC = max(max(C_varphi1*np.array(data_result_varphi_tt.iloc[4, start_index:end_index:interval], dtype='float32')+
                C_varphi_dot1*np.array(data_result_varphidot_tt.iloc[4, start_index:end_index:interval], dtype='float32')),
                max(C_varphi2*np.array(data_result_varphi_tl.iloc[4, start_index:end_index:interval], dtype='float32')+
                C_varphi_dot2*np.array(data_result_varphidot_tl.iloc[4, start_index:end_index:interval], dtype='float32')))
    R_FHADP = max(max(C_varphi1*np.array(data_result_varphi_tt.iloc[1, start_index:end_index:interval], dtype='float32')+
                C_varphi_dot1*np.array(data_result_varphidot_tt.iloc[1, start_index:end_index:interval], dtype='float32')),
                max(C_varphi2*np.array(data_result_varphi_tl.iloc[1, start_index:end_index:interval], dtype='float32')+
                C_varphi_dot2*np.array(data_result_varphidot_tl.iloc[1, start_index:end_index:interval], dtype='float32')))
    R_ABPO = max(max(C_varphi1*np.array(data_result_varphi_tt.iloc[2, start_index:end_index:interval], dtype='float32')+
                C_varphi_dot1*np.array(data_result_varphidot_tt.iloc[2, start_index:end_index:interval], dtype='float32')),
                max(C_varphi2*np.array(data_result_varphi_tl.iloc[2, start_index:end_index:interval], dtype='float32')+
                C_varphi_dot2*np.array(data_result_varphidot_tl.iloc[2, start_index:end_index:interval], dtype='float32')))
    # print("absmaxR[MPC, PUMPC, FHADP, ABPO]:", max(abs(R_MPC)), max(abs(R_PUMPC)), max(abs(R_FHADP)), max(abs(R_ABPO)))
    print("IR[MPC, PUMPC, FHADP, ABPO]:", R_MPC, R_PUMPC, R_FHADP, R_ABPO)

if __name__ == "__main__":
    # Parameters Setup
    parser = argparse.ArgumentParser()
    parser.add_argument("--time_step", type=float, default=0.01)
    parser.add_argument("--csv_file_name", type=str, default="State-4")
    parser.add_argument("--csv_file_name2", type=str, default="State-5")
    parser.add_argument("--line_num", type=int, default=4)
    parser.add_argument("--language", type=str, default="ch")
    # parser.add_argument("--x_label", type=str, default=r"Pos $p_{\rm x,tt}\ /\mathrm{m}$")
    # parser.add_argument("--y_label", type=str, default=r"Pos $p_{\rm y,tt}\ /\mathrm{m}$")
    # parser.add_argument("--x_label", type=str, default=r"Time $/\mathrm{s}$")
    # parser.add_argument("--y_label", type=str, default=r"Lateral error $p_{\rm y,tt}^{\rm err}\ /\mathrm{m}$")
    # parser.add_argument("--y_label", type=str, default=r"Yaw $\phi_{\rm tt}\ /\mathrm{rad}$")
    # parser.add_argument("--y_label", type=str, default=r"Yaw error $\phi_{\rm tl}^{\rm err}\ /\mathrm{rad}$")
    # parser.add_argument("--y_label", type=str, default=r"Yawrate $\dot\phi_{\rm tt}\ /\mathrm{rad·s^{-1}}$")
    # parser.add_argument("--y_label", type=str, default=r"Roll $\varphi_{\rm tt}\ /\mathrm{rad}$") #
    # parser.add_argument("--y_label", type=str, default=r"Roll rate $\dot\varphi_{\rm tl}\ /\mathrm{rad·s^{-1}}$")
    # parser.add_argument("--y_label", type=str, default=r"Lateral speed $v_{\rm tt}\ /\mathrm{m·s^{-1}}$") # ·s^{-1}
    # parser.add_argument("--y_label", type=str, default=r"Steering Angle $\delta_{\rm tt}\ /\mathrm{rad}$")  #
    # parser.add_argument("--y_label", type=str, default=r"Calculation time $/\mathrm{ms}$")  #

    # parser.add_argument("--x_label", type=str, default=r"$p_{\rm x,tl}\ /\mathrm{m}$")
    # parser.add_argument("--y_label", type=str, default=r"$p_{\rm y,tl}\ /\mathrm{m}$")
    parser.add_argument("--x_label", type=str, default=r"时间 $/\mathrm{s}$")
    parser.add_argument("--y_label", type=str, default=r" $u_{\rm tt}\ /\mathrm{(m·s^{-1})}$")
    # parser.add_argument("--y_label", type=str, default=r"速度误差$u_{\rm tt}^{\rm err}\ /\mathrm{(m·s^{-1})}$")
    # parser.add_argument("--y_label", type=str, default=r"横向误差$p_{\rm y,tt}^{\rm err}\ /\mathrm{m}$")
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
    ["参考状态","MPC方法", "FHADP方法", "Bi-level方法"])#
    # ["MPC", "PUMPC", "FHADP", "ABPO"])#"Ref",
    parser.add_argument("--color_list", type=list, default=
    ["b","#8A2BE2", "lime", "magenta"]) #,
    # ["#8A2BE2", "#FA8072", "lime", "magenta"]) #,"b",
    parser.add_argument("--figures_root", type=str,
                        default='../../figures/FHADP2-FHADP2-pyth_semitruckpu7dof/241223-184240-circle90/')
    # parser.add_argument("--figures_root", type=str,default='../../results/pyth_semitruckpu7doflateral/FHADP2_240426-091408-upper_20-inner_50000/')
    # Get parameter dictionary
    args = vars(parser.parse_args())

    read_path_datax = args["figures_root"] + "State-1.csv"
    read_path = args["figures_root"]+args["csv_file_name"]+".csv"
    read_path2 = args["figures_root"] + args["csv_file_name2"] + ".csv"
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
    # compute_Rmetrics(read_path, read_path2)
    # compute_yoff_metrics(args)
    # compute_IR_metrics(args)
