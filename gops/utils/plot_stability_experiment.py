import os
import matplotlib
import seaborn as sns
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
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
default_cfg["fig_size"] = (12, 4.5)
default_cfg["fig_size12-9"] = (12, 9)
default_cfg["ax_para"] = [0.22, 0.31, 0.75, 0.65]
default_cfg["ax_para2"] = [0.24, 0.15, 0.75, 0.82]
default_cfg["dpi"] = 300
default_cfg["pad"] = 0.5

default_cfg["tick_size"] = 20
default_cfg["tick_label_font"] = "Times New Roman"
default_cfg["legend_font"] = {
    "family": "Times New Roman",#, SimHei
    "size": "11",
    "weight": "normal",
}
default_cfg["label_font"] = {
    "size": "17",  # ch:30
    "weight": "normal",
"family": "Times New Roman",
}
default_cfg["img_fmt"] = "pdf"
# mpl.rcParams['font.sans-serif'] = ['SimSun']  # 指定宋体
# mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['xtick.direction']='in'
plt.rcParamss['ytick.direction']='in'
def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)

def read_data(root_path):
    data_result = pd.DataFrame(pd.read_csv(root_path, header=None))
    start_index = 1
    end_index = 3000
    interval = 5
    step_list = np.array(data_result.iloc[start_index:end_index:interval, 0], dtype='float32')
    ay = np.array(data_result.iloc[start_index:end_index:interval, 1], dtype='float32')
    ax = np.array(data_result.iloc[start_index:end_index:interval, 2], dtype='float32')
    Q_fl = np.array(data_result.iloc[start_index:end_index:interval, 3], dtype='float32')
    Q_fr = -np.array(data_result.iloc[start_index:end_index:interval, 4], dtype='float32')
    Q_rl = -np.array(data_result.iloc[start_index:end_index:interval, 5], dtype='float32')
    Q_rr = -np.array(data_result.iloc[start_index:end_index:interval, 6], dtype='float32')
    pos_z = np.array(data_result.iloc[start_index:end_index:interval, 7], dtype='float32')
    yaw = np.array(data_result.iloc[start_index:end_index:interval, 8], dtype='float32')
    vx = np.array(data_result.iloc[start_index:end_index:interval, 9], dtype='float32')
    pitch_rate = np.array(data_result.iloc[start_index:end_index:interval, 10], dtype='float32')
    roll_rate = np.array(data_result.iloc[start_index:end_index:interval, 11], dtype='float32')/180*3.14
    yaw_rate = np.array(data_result.iloc[start_index:end_index:interval, 12], dtype='float32')
    pos_y_err_ntp2 = np.array(data_result.iloc[start_index:end_index:interval, 13], dtype='float32')
    pos_y_err_nt = np.array(data_result.iloc[start_index:end_index:interval, 14], dtype='float32')
    pos_y_err_pre = np.array(data_result.iloc[start_index:end_index:interval, 15], dtype='float32')
    vx_err_pre = np.array(data_result.iloc[start_index:end_index:interval, 16], dtype='float32')
    vx_err_nt = np.array(data_result.iloc[start_index:end_index:interval, 17], dtype='float32')
    pitch = np.array(data_result.iloc[start_index:end_index:interval, 18], dtype='float32')
    pos_x = np.array(data_result.iloc[start_index:end_index:interval, 19], dtype='float32')
    pos_y = np.array(data_result.iloc[start_index:end_index:interval, 20], dtype='float32')
    ref_vx = np.array(data_result.iloc[start_index:end_index:interval, 21], dtype='float32')
    ref_vx_ntp2 = np.array(data_result.iloc[start_index:end_index:interval, 22], dtype='float32')
    ref_vx_nt = np.array(data_result.iloc[start_index:end_index:interval, 23], dtype='float32')
    ref_pos_x = np.array(data_result.iloc[start_index:end_index:interval, 24], dtype='float32')
    ref_pos_x_ntp2 = np.array(data_result.iloc[start_index:end_index:interval, 25], dtype='float32')
    ref_pos_x_nt = np.array(data_result.iloc[start_index:end_index:interval, 26], dtype='float32')
    ref_pos_y = np.array(data_result.iloc[start_index:end_index:interval, 27], dtype='float32')
    ref_pos_y_ntp2 = np.array(data_result.iloc[start_index:end_index:interval, 28], dtype='float32')
    ref_pos_y_nt = np.array(data_result.iloc[start_index:end_index:interval, 29], dtype='float32')
    ref_yaw = np.array(data_result.iloc[start_index:end_index:interval, 30], dtype='float32')
    ref_yaw_ntp2 = np.array(data_result.iloc[start_index:end_index:interval, 31], dtype='float32')
    ref_yaw_nt = np.array(data_result.iloc[start_index:end_index:interval, 32], dtype='float32')
    roll = np.array(data_result.iloc[start_index:end_index:interval, 33], dtype='float32')
    delta = np.array(data_result.iloc[start_index:end_index:interval, 34], dtype='float32') / 17
    des_brake = np.array(data_result.iloc[start_index:end_index:interval, 35], dtype='float32')
    des_deltaf = np.array(data_result.iloc[start_index:end_index:interval, 36], dtype='float32')
    des_delta = np.array(data_result.iloc[start_index:end_index:interval, 37], dtype='float32') / 17
    des_torque = np.array(data_result.iloc[start_index:end_index:interval, 38], dtype='float32') / 4
    time_calc = np.array(data_result.iloc[start_index:end_index:interval, 39], dtype='float32')*50
    yaw_err_pre = np.array(data_result.iloc[start_index:end_index:interval, 40], dtype='float32')
    vy = np.array(data_result.iloc[start_index:end_index:interval, 41], dtype='float32')

    state_pool = np.zeros((len(step_list), 14))
    state_pool[:, 0] = step_list
    state_pool[:, 1] = pos_x
    state_pool[:, 2] = pos_y
    state_pool[:, 3] = yaw
    state_pool[:, 4] = vx
    state_pool[:, 5] = vy
    state_pool[:, 6] = yaw_rate
    state_pool[:, 7] = roll
    state_pool[:, 8] = roll_rate

    state_pool[:, 9] = Q_fl
    state_pool[:, 10] = Q_fr
    state_pool[:, 11] = Q_rl
    state_pool[:, 12] = Q_rr
    state_pool[:, 13] = delta


    ref_pool = np.zeros((len(step_list), 8))
    ref_pool[:, 0] = ref_pos_x
    ref_pool[:, 1] = ref_pos_y
    ref_pool[:, 2] = ref_yaw
    ref_pool[:, 3] = des_torque
    ref_pool[:, 4] = des_delta
    ref_pool[:, 5] = ref_vx
    ref_pool[:, 6] = pos_z
    ref_pool[:, 7] = time_calc*100

    error_pool = np.zeros((len(step_list), 5))
    error_pool[:, 0] = pos_y_err_pre
    error_pool[:, 1] = yaw_err_pre
    error_pool[:, 2] = vx_err_pre
    error_pool[:, 3] = vx_err_nt
    error_pool[:, 4] = pos_y_err_nt
    return state_pool, ref_pool, error_pool

def read_data_new(root_path):
    data_result = pd.DataFrame(pd.read_csv(root_path, header=None))
    start_index = 21
    end_index = 1524

    interval = 1
    step_list = np.array(data_result.iloc[start_index:end_index:interval, 0], dtype='float32')
    ay = np.array(data_result.iloc[start_index:end_index:interval, 1], dtype='float32')
    ax = np.array(data_result.iloc[start_index:end_index:interval, 2], dtype='float32')
    brake_fb = np.array(data_result.iloc[start_index:end_index:interval, 3], dtype='float32')
    Q_fl = np.array(data_result.iloc[start_index:end_index:interval, 4], dtype='float32')
    Q_fr = -np.array(data_result.iloc[start_index:end_index:interval, 5], dtype='float32')
    Q_rl = -np.array(data_result.iloc[start_index:end_index:interval, 6], dtype='float32')
    Q_rr = -np.array(data_result.iloc[start_index:end_index:interval, 7], dtype='float32')
    pos_z = np.array(data_result.iloc[start_index:end_index:interval, 8], dtype='float32')
    yaw = np.array(data_result.iloc[start_index:end_index:interval, 9], dtype='float32')
    pos_y_err_pre = np.array(data_result.iloc[start_index:end_index:interval, 10], dtype='float32')
    pos_y_err_nt = np.array(data_result.iloc[start_index:end_index:interval, 11], dtype='float32')
    lat_slope = np.array(data_result.iloc[start_index:end_index:interval, 12], dtype='float32')
    longi_jerk = np.array(data_result.iloc[start_index:end_index:interval, 13], dtype='float32')
    longi_slope = np.array(data_result.iloc[start_index:end_index:interval, 14], dtype='float32')
    vx_err_nt = np.array(data_result.iloc[start_index:end_index:interval, 15], dtype='float32')
    mcu_v = np.array(data_result.iloc[start_index:end_index:interval, 16], dtype='float32')

    pos_y = np.array(data_result.iloc[start_index:end_index:interval, 17], dtype='float32')-np.array(data_result.iloc[start_index, 17], dtype='float32')
    pos_x = np.array(data_result.iloc[start_index:end_index:interval, 18], dtype='float32')-np.array(data_result.iloc[start_index, 18], dtype='float32')
    ref_vx = np.array(data_result.iloc[start_index:end_index:interval, 19], dtype='float32')
    ref_pos_y = np.array(data_result.iloc[start_index:end_index:interval, 20], dtype='float32')-np.array(data_result.iloc[start_index, 20], dtype='float32')
    ref_pos_x = np.array(data_result.iloc[start_index:end_index:interval, 21], dtype='float32')-np.array(data_result.iloc[start_index, 21], dtype='float32')
    ref_yaw = np.array(data_result.iloc[start_index:end_index:interval, 22], dtype='float32')
    roll = np.array(data_result.iloc[start_index:end_index:interval, 23], dtype='float32')
    roll_rate = np.array(data_result.iloc[start_index:end_index:interval, 24], dtype='float32') / 180 * 3.14

    delta = np.array(data_result.iloc[start_index:end_index:interval, 25], dtype='float32')
    des_brake = np.array(data_result.iloc[start_index:end_index:interval, 26], dtype='float32')
    des_deltaf = np.array(data_result.iloc[start_index:end_index:interval, 27], dtype='float32')
    des_delta = np.array(data_result.iloc[start_index:end_index:interval, 28], dtype='float32') / 16.5
    des_torque = np.array(data_result.iloc[start_index:end_index:interval, 29], dtype='float32') / 4
    time_calc = np.array(data_result.iloc[start_index:end_index:interval, 30], dtype='float32')
    vx = np.array(data_result.iloc[start_index:end_index:interval, 31], dtype='float32')
    vy = np.array(data_result.iloc[start_index:end_index:interval, 32], dtype='float32')
    yaw_rate = np.array(data_result.iloc[start_index:end_index:interval, 33], dtype='float32')
    yaw_err_nt = np.array(data_result.iloc[start_index:end_index:interval, 34], dtype='float32')
    # time_calc = np.array(data_result.iloc[start_index:end_index:interval, 39], dtype='float32')*50
    # yaw_err_pre = np.array(data_result.iloc[start_index:end_index:interval, 40], dtype='float32')
    # pitch_rate = np.array(data_result.iloc[start_index:end_index:interval, 10], dtype='float32')
    # pos_y_err_ntp2 = np.array(data_result.iloc[start_index:end_index:interval, 13], dtype='float32')
    # pos_y_err_nt = np.array(data_result.iloc[start_index:end_index:interval, 14], dtype='float32')
    # pos_y_err_pre = np.array(data_result.iloc[start_index:end_index:interval, 15], dtype='float32')
    # vx_err_pre = np.array(data_result.iloc[start_index:end_index:interval, 16], dtype='float32')

    # pitch = np.array(data_result.iloc[start_index:end_index:interval, 18], dtype='float32')

    state_pool = np.zeros((len(step_list), 16))
    state_pool[:, 0] = step_list
    state_pool[:, 1] = pos_x
    state_pool[:, 2] = pos_y
    state_pool[:, 3] = yaw
    state_pool[:, 4] = vx
    state_pool[:, 5] = vy
    state_pool[:, 6] = yaw_rate
    state_pool[:, 7] = roll
    state_pool[:, 8] = roll_rate

    state_pool[:, 9] = Q_fl
    state_pool[:, 10] = Q_fr
    state_pool[:, 11] = Q_rl
    state_pool[:, 12] = Q_rr
    state_pool[:, 13] = delta
    state_pool[:, 14] = brake_fb

    ref_pool = np.zeros((len(step_list), 10))
    ref_pool[:, 0] = ref_pos_x
    ref_pool[:, 1] = ref_pos_y
    ref_pool[:, 2] = ref_yaw
    ref_pool[:, 3] = des_torque
    ref_pool[:, 4] = des_delta
    ref_pool[:, 5] = ref_vx
    ref_pool[:, 6] = pos_z
    ref_pool[:, 7] = longi_slope
    ref_pool[:, 8] = lat_slope
    ref_pool[:, 9] = des_brake

    error_pool = np.zeros((len(step_list), 4))
    error_pool[:, 0] = pos_y_err_pre
    # error_pool[:, 1] = yaw_err_pre
    # error_pool[:, 2] = vx_err_pre
    error_pool[:, 1] = vx_err_nt
    error_pool[:, 2] = pos_y_err_nt
    error_pool[:, 3] = yaw_err_nt
    return state_pool, ref_pool, error_pool

def read_data_traj_record(root_path):
    data_result = pd.DataFrame(pd.read_csv(root_path, header=None))
    start_index = 1
    end_index = 3250

    interval = 1
    step_list = np.array(data_result.iloc[start_index:end_index:interval, 0], dtype='float32')
    ay = np.array(data_result.iloc[start_index:end_index:interval, 1], dtype='float32')
    ax = np.array(data_result.iloc[start_index:end_index:interval, 2], dtype='float32')
    ax_clac = np.array(data_result.iloc[start_index:end_index:interval, 3], dtype='float32')
    brake_fb = np.array(data_result.iloc[start_index:end_index:interval, 4], dtype='float32')
    Q_fl = np.array(data_result.iloc[start_index:end_index:interval, 5], dtype='float32')
    Q_fr = -np.array(data_result.iloc[start_index:end_index:interval, 6], dtype='float32')
    Q_rl = -np.array(data_result.iloc[start_index:end_index:interval, 7], dtype='float32')
    Q_rr = np.array(data_result.iloc[start_index:end_index:interval, 8], dtype='float32')
    engine_power = np.array(data_result.iloc[start_index:end_index:interval, 9], dtype='float32')
    engine_speed= np.array(data_result.iloc[start_index:end_index:interval, 10], dtype='float32')
    engine_torque= np.array(data_result.iloc[start_index:end_index:interval, 11], dtype='float32')
    pos_z = np.array(data_result.iloc[start_index:end_index:interval, 12], dtype='float32')
    yaw = np.array(data_result.iloc[start_index:end_index:interval, 13], dtype='float32')
    vx = np.array(data_result.iloc[start_index:end_index:interval, 14], dtype='float32')
    pitch_rate = np.array(data_result.iloc[start_index:end_index:interval, 15], dtype='float32')

    roll_rate = np.array(data_result.iloc[start_index:end_index:interval, 16], dtype='float32')
    yaw_rate = np.array(data_result.iloc[start_index:end_index:interval, 17], dtype='float32')
    lat_slope = np.array(data_result.iloc[start_index:end_index:interval, 18], dtype='float32')/180*3.14
    longi_slope = np.array(data_result.iloc[start_index:end_index:interval, 19], dtype='float32')/180*3.14


    motor_power_fl = np.array(data_result.iloc[start_index:end_index:interval, 20], dtype='float32')

    motor_power_fr = np.array(data_result.iloc[start_index:end_index:interval, 21], dtype='float32')
    motor_power_rl = np.array(data_result.iloc[start_index:end_index:interval, 22], dtype='float32')
    motor_power_rr = np.array(data_result.iloc[start_index:end_index:interval, 23], dtype='float32')
    pitch = np.array(data_result.iloc[start_index:end_index:interval, 24], dtype='float32')
    pos_x = np.array(data_result.iloc[start_index:end_index:interval, 25], dtype='float32')-np.array(data_result.iloc[start_index, 25], dtype='float32')
    pos_y = np.array(data_result.iloc[start_index:end_index:interval, 26], dtype='float32')-np.array(data_result.iloc[start_index, 26], dtype='float32')
    roll = np.array(data_result.iloc[start_index:end_index:interval, 27], dtype='float32')
    delta = np.array(data_result.iloc[start_index:end_index:interval, 28], dtype='float32') / 180 * 3.14/16.5
    vy = np.array(data_result.iloc[start_index:end_index:interval, 29], dtype='float32')
    state_pool = np.zeros((len(step_list), 16))
    state_pool[:, 0] = step_list
    state_pool[:, 1] = pos_x
    state_pool[:, 2] = pos_y
    state_pool[:, 3] = yaw
    state_pool[:, 4] = vx
    state_pool[:, 5] = vy
    state_pool[:, 6] = yaw_rate
    state_pool[:, 7] = roll
    state_pool[:, 8] = roll_rate

    state_pool[:, 9] = Q_fl
    state_pool[:, 10] = Q_fr
    state_pool[:, 11] = Q_rl
    state_pool[:, 12] = Q_rr
    state_pool[:, 13] = delta
    state_pool[:, 14] = brake_fb

    ref_pool = np.zeros((len(step_list), 10))
    ref_pool[:, 0] = pos_x
    ref_pool[:, 1] = pos_y
    ref_pool[:, 2] = yaw
    ref_pool[:, 3] = motor_power_fl
    ref_pool[:, 4] = motor_power_fr
    ref_pool[:, 5] = vx
    ref_pool[:, 6] = pos_z
    ref_pool[:, 7] = longi_slope
    ref_pool[:, 8] = lat_slope
    ref_pool[:, 9] = brake_fb

    error_pool = np.zeros((len(step_list), 4))
    error_pool[:, 0] = brake_fb
    # error_pool[:, 1] = yaw_err_pre
    # error_pool[:, 2] = vx_err_pre
    error_pool[:, 1] = brake_fb
    error_pool[:, 2] = brake_fb
    error_pool[:, 3] = brake_fb
    return state_pool, ref_pool, error_pool

def read_data_multi_method(root_path1, root_path2, root_path3):
    data_result = pd.DataFrame(pd.read_csv(root_path1, header=None))
    data_result2 = pd.DataFrame(pd.read_csv(root_path2, header=None))
    data_result3 = pd.DataFrame(pd.read_csv(root_path3, header=None))
    start_index = 1
    end_index = 1500
    interval = 1
    step_list = np.array(data_result.iloc[start_index:end_index:interval, 0], dtype='float32')
    ay = np.array(data_result.iloc[start_index:end_index:interval, 1], dtype='float32')
    ax = np.array(data_result.iloc[start_index:end_index:interval, 2], dtype='float32')
    brake_fb = np.array(data_result.iloc[start_index:end_index:interval, 3], dtype='float32')
    Q_fl = np.array(data_result.iloc[start_index:end_index:interval, 4], dtype='float32')
    Q_fr = -np.array(data_result.iloc[start_index:end_index:interval, 5], dtype='float32')
    Q_rl = -np.array(data_result.iloc[start_index:end_index:interval, 6], dtype='float32')
    Q_rr = -np.array(data_result.iloc[start_index:end_index:interval, 7], dtype='float32')
    pos_z = np.array(data_result.iloc[start_index:end_index:interval, 8], dtype='float32')
    yaw = np.array(data_result.iloc[start_index:end_index:interval, 9], dtype='float32')
    pos_y_err_pre = np.array(data_result.iloc[start_index:end_index:interval, 10], dtype='float32')
    pos_y_err_nt = np.array(data_result.iloc[start_index:end_index:interval, 11], dtype='float32')
    lat_slope = np.array(data_result.iloc[start_index:end_index:interval, 12], dtype='float32')
    longi_jerk = np.array(data_result.iloc[start_index:end_index:interval, 13], dtype='float32')
    longi_slope = np.array(data_result.iloc[start_index:end_index:interval, 14], dtype='float32')
    vx_err_nt = np.array(data_result.iloc[start_index:end_index:interval, 15], dtype='float32')
    mcu_v = np.array(data_result.iloc[start_index:end_index:interval, 16], dtype='float32')

    pos_y = np.array(data_result.iloc[start_index:end_index:interval, 17], dtype='float32')-np.array(data_result.iloc[start_index, 17], dtype='float32')
    pos_x = np.array(data_result.iloc[start_index:end_index:interval, 18], dtype='float32')-np.array(data_result.iloc[start_index, 18], dtype='float32')
    ref_vx = np.array(data_result.iloc[start_index:end_index:interval, 19], dtype='float32')
    ref_pos_y = np.array(data_result.iloc[start_index:end_index:interval, 20], dtype='float32')-np.array(data_result.iloc[start_index, 20], dtype='float32')
    ref_pos_x = np.array(data_result.iloc[start_index:end_index:interval, 21], dtype='float32')-np.array(data_result.iloc[start_index, 21], dtype='float32')
    ref_yaw = np.array(data_result.iloc[start_index:end_index:interval, 22], dtype='float32')
    roll = np.array(data_result.iloc[start_index:end_index:interval, 23], dtype='float32')
    roll_rate = np.array(data_result.iloc[start_index:end_index:interval, 24], dtype='float32') / 180 * 3.14

    delta = np.array(data_result.iloc[start_index:end_index:interval, 25], dtype='float32')
    des_brake = np.array(data_result.iloc[start_index:end_index:interval, 26], dtype='float32')
    des_deltaf = np.array(data_result.iloc[start_index:end_index:interval, 27], dtype='float32')
    des_delta = np.array(data_result.iloc[start_index:end_index:interval, 28], dtype='float32') / 16.5
    des_torque = np.array(data_result.iloc[start_index:end_index:interval, 29], dtype='float32') / 4
    time_calc = np.array(data_result.iloc[start_index:end_index:interval, 30], dtype='float32')
    vx = np.array(data_result.iloc[start_index:end_index:interval, 31], dtype='float32')
    vy = np.array(data_result.iloc[start_index:end_index:interval, 32], dtype='float32')
    yaw_rate = np.array(data_result.iloc[start_index:end_index:interval, 33], dtype='float32')
    yaw_err_nt = np.array(data_result.iloc[start_index:end_index:interval, 34], dtype='float32')

    step_list_2 = np.array(data_result2.iloc[start_index:end_index:interval, 0], dtype='float32')
    ay_2 = np.array(data_result2.iloc[start_index:end_index:interval, 1], dtype='float32')
    ax_2 = np.array(data_result2.iloc[start_index:end_index:interval, 2], dtype='float32')
    brake_fb_2 = np.array(data_result2.iloc[start_index:end_index:interval, 3], dtype='float32')
    Q_fl_2 = np.array(data_result2.iloc[start_index:end_index:interval, 4], dtype='float32')
    Q_fr_2 = -np.array(data_result2.iloc[start_index:end_index:interval, 5], dtype='float32')
    Q_rl_2 = -np.array(data_result2.iloc[start_index:end_index:interval, 6], dtype='float32')
    Q_rr_2 = -np.array(data_result2.iloc[start_index:end_index:interval, 7], dtype='float32')
    pos_z_2 = np.array(data_result2.iloc[start_index:end_index:interval, 8], dtype='float32')
    yaw_2 = np.array(data_result2.iloc[start_index:end_index:interval, 9], dtype='float32')
    pos_y_err_pre_2 = np.array(data_result2.iloc[start_index:end_index:interval, 10], dtype='float32')
    pos_y_err_nt_2 = np.array(data_result2.iloc[start_index:end_index:interval, 11], dtype='float32')
    lat_slope_2 = np.array(data_result2.iloc[start_index:end_index:interval, 12], dtype='float32')
    longi_jerk_2 = np.array(data_result2.iloc[start_index:end_index:interval, 13], dtype='float32')
    longi_slope_2 = np.array(data_result2.iloc[start_index:end_index:interval, 14], dtype='float32')
    vx_err_nt_2 = np.array(data_result2.iloc[start_index:end_index:interval, 15], dtype='float32')
    mcu_v_2 = np.array(data_result2.iloc[start_index:end_index:interval, 16], dtype='float32')

    pos_y_2 = np.array(data_result2.iloc[start_index:end_index:interval, 17], dtype='float32') - np.array(
        data_result2.iloc[start_index, 17], dtype='float32')
    pos_x_2 = np.array(data_result2.iloc[start_index:end_index:interval, 18], dtype='float32') - np.array(
        data_result2.iloc[start_index, 18], dtype='float32')
    ref_vx_2 = np.array(data_result2.iloc[start_index:end_index:interval, 19], dtype='float32')
    ref_pos_y_2 = np.array(data_result2.iloc[start_index:end_index:interval, 20], dtype='float32') - np.array(
        data_result2.iloc[start_index, 20], dtype='float32')
    ref_pos_x_2 = np.array(data_result2.iloc[start_index:end_index:interval, 21], dtype='float32') - np.array(
        data_result2.iloc[start_index, 21], dtype='float32')
    ref_yaw_2 = np.array(data_result2.iloc[start_index:end_index:interval, 22], dtype='float32')
    roll_2 = np.array(data_result2.iloc[start_index:end_index:interval, 23], dtype='float32')
    roll_rate_2 = np.array(data_result2.iloc[start_index:end_index:interval, 24], dtype='float32') / 180 * 3.14

    delta_2 = np.array(data_result2.iloc[start_index:end_index:interval, 25], dtype='float32')
    des_brake_2 = np.array(data_result2.iloc[start_index:end_index:interval, 26], dtype='float32')
    des_deltaf_2 = np.array(data_result2.iloc[start_index:end_index:interval, 27], dtype='float32')
    des_delta_2 = np.array(data_result2.iloc[start_index:end_index:interval, 28], dtype='float32') / 16.5
    des_torque_2 = np.array(data_result2.iloc[start_index:end_index:interval, 29], dtype='float32') / 4
    time_calc_2 = np.array(data_result2.iloc[start_index:end_index:interval, 30], dtype='float32')
    vx_2 = np.array(data_result2.iloc[start_index:end_index:interval, 31], dtype='float32')
    vy_2 = np.array(data_result2.iloc[start_index:end_index:interval, 32], dtype='float32')
    yaw_rate_2 = np.array(data_result2.iloc[start_index:end_index:interval, 33], dtype='float32')
    yaw_err_nt_2 = np.array(data_result2.iloc[start_index:end_index:interval, 34], dtype='float32')

    step_list_3 = np.array(data_result3.iloc[start_index:end_index:interval, 0], dtype='float32')
    ay_3 = np.array(data_result3.iloc[start_index:end_index:interval, 1], dtype='float32')
    ax_3 = np.array(data_result3.iloc[start_index:end_index:interval, 2], dtype='float32')
    brake_fb_3 = np.array(data_result3.iloc[start_index:end_index:interval, 3], dtype='float32')
    Q_fl_3 = np.array(data_result3.iloc[start_index:end_index:interval, 4], dtype='float32')
    Q_fr_3 = -np.array(data_result3.iloc[start_index:end_index:interval, 5], dtype='float32')
    Q_rl_3 = -np.array(data_result3.iloc[start_index:end_index:interval, 6], dtype='float32')
    Q_rr_3 = -np.array(data_result3.iloc[start_index:end_index:interval, 7], dtype='float32')
    pos_z_3 = np.array(data_result3.iloc[start_index:end_index:interval, 8], dtype='float32')
    yaw_3 = np.array(data_result3.iloc[start_index:end_index:interval, 9], dtype='float32')
    pos_y_err_pre_3 = np.array(data_result3.iloc[start_index:end_index:interval, 10], dtype='float32')
    pos_y_err_nt_3 = np.array(data_result3.iloc[start_index:end_index:interval, 11], dtype='float32')
    lat_slope_3 = np.array(data_result3.iloc[start_index:end_index:interval, 12], dtype='float32')
    longi_jerk_3 = np.array(data_result3.iloc[start_index:end_index:interval, 13], dtype='float32')
    longi_slope_3 = np.array(data_result3.iloc[start_index:end_index:interval, 14], dtype='float32')
    vx_err_nt_3 = np.array(data_result3.iloc[start_index:end_index:interval, 15], dtype='float32')
    mcu_v_3 = np.array(data_result3.iloc[start_index:end_index:interval, 16], dtype='float32')

    pos_y_3 = np.array(data_result3.iloc[start_index:end_index:interval, 17], dtype='float32') - np.array(
        data_result3.iloc[start_index, 17], dtype='float32')
    pos_x_3 = np.array(data_result3.iloc[start_index:end_index:interval, 18], dtype='float32') - np.array(
        data_result3.iloc[start_index, 18], dtype='float32')
    ref_vx_3 = np.array(data_result3.iloc[start_index:end_index:interval, 19], dtype='float32')
    ref_pos_y_3 = np.array(data_result3.iloc[start_index:end_index:interval, 20], dtype='float32') - np.array(
        data_result3.iloc[start_index, 20], dtype='float32')
    ref_pos_x_3 = np.array(data_result3.iloc[start_index:end_index:interval, 21], dtype='float32') - np.array(
        data_result3.iloc[start_index, 21], dtype='float32')
    ref_yaw_3 = np.array(data_result3.iloc[start_index:end_index:interval, 22], dtype='float32')
    roll_3 = np.array(data_result3.iloc[start_index:end_index:interval, 23], dtype='float32')
    roll_rate_3 = np.array(data_result3.iloc[start_index:end_index:interval, 24], dtype='float32') / 180 * 3.14

    delta_3 = np.array(data_result3.iloc[start_index:end_index:interval, 25], dtype='float32')
    des_brake_3 = np.array(data_result3.iloc[start_index:end_index:interval, 26], dtype='float32')
    des_deltaf_3 = np.array(data_result3.iloc[start_index:end_index:interval, 27], dtype='float32')
    des_delta_3 = np.array(data_result3.iloc[start_index:end_index:interval, 28], dtype='float32') / 16.5
    des_torque_3 = np.array(data_result3.iloc[start_index:end_index:interval, 29], dtype='float32') / 4
    time_calc_3 = np.array(data_result3.iloc[start_index:end_index:interval, 30], dtype='float32')
    vx_3 = np.array(data_result3.iloc[start_index:end_index:interval, 31], dtype='float32')
    vy_3 = np.array(data_result3.iloc[start_index:end_index:interval, 32], dtype='float32')
    yaw_rate_3 = np.array(data_result3.iloc[start_index:end_index:interval, 33], dtype='float32')
    yaw_err_nt_3 = np.array(data_result3.iloc[start_index:end_index:interval, 34], dtype='float32')

    state_pool = np.zeros((3, len(step_list), 16))
    state_pool[0, :, 0] = step_list
    state_pool[0, :, 1] = pos_x
    state_pool[0, :, 2] = pos_y
    state_pool[0, :, 3] = yaw
    state_pool[0, :, 4] = vx
    state_pool[0, :, 5] = vy
    state_pool[0, :, 6] = yaw_rate
    state_pool[0, :, 7] = roll
    state_pool[0, :, 8] = roll_rate

    state_pool[0, :, 9] = Q_fl
    state_pool[0, :, 10] = Q_fr
    state_pool[0, :, 11] = Q_rl
    state_pool[0, :, 12] = Q_rr
    state_pool[0, :, 13] = delta
    state_pool[0, :, 14] = brake_fb

    state_pool[1, :, 0] = step_list_2
    state_pool[1, :, 1] = pos_x_2
    state_pool[1, :, 2] = pos_y_2
    state_pool[1, :, 3] = yaw_2
    state_pool[1, :, 4] = vx_2
    state_pool[1, :, 5] = vy_2
    state_pool[1, :, 6] = yaw_rate_2
    state_pool[1, :, 7] = roll_2
    state_pool[1, :, 8] = roll_rate_2

    state_pool[1, :, 9] = Q_fl_2
    state_pool[1, :, 10] = Q_fr_2
    state_pool[1, :, 11] = Q_rl_2
    state_pool[1, :, 12] = Q_rr_2
    state_pool[1, :, 13] = delta_2
    state_pool[1, :, 14] = brake_fb_2

    state_pool[2, :, 0] = step_list_3
    state_pool[2, :, 1] = pos_x_3
    state_pool[2, :, 2] = pos_y_3
    state_pool[2, :, 3] = yaw_3
    state_pool[2, :, 4] = vx_3
    state_pool[2, :, 5] = vy_3
    state_pool[2, :, 6] = yaw_rate_3
    state_pool[2, :, 7] = roll_3
    state_pool[2, :, 8] = roll_rate_3

    state_pool[2, :, 9] = Q_fl_3
    state_pool[2, :, 10] = Q_fr_3
    state_pool[2, :, 11] = Q_rl_3
    state_pool[2, :, 12] = Q_rr_3
    state_pool[2, :, 13] = delta_3
    state_pool[2, :, 14] = brake_fb_3

    ref_pool = np.zeros((3, len(step_list), 10))
    ref_pool[0, :, 0] = ref_pos_x
    ref_pool[0, :, 1] = ref_pos_y
    ref_pool[0, :, 2] = ref_yaw
    ref_pool[0, :, 3] = des_torque
    ref_pool[0, :, 4] = des_delta
    ref_pool[0, :, 5] = ref_vx
    ref_pool[0, :, 6] = pos_z
    ref_pool[0, :, 7] = longi_slope
    ref_pool[0, :, 8] = lat_slope
    ref_pool[0, :, 9] = des_brake

    ref_pool[1, :, 0] = ref_pos_x_2
    ref_pool[1, :, 1] = ref_pos_y_2
    ref_pool[1, :, 2] = ref_yaw_2
    ref_pool[1, :, 3] = des_torque_2
    ref_pool[1, :, 4] = des_delta_2
    ref_pool[1, :, 5] = ref_vx_2
    ref_pool[1, :, 6] = pos_z_2
    ref_pool[1, :, 7] = longi_slope_2
    ref_pool[1, :, 8] = lat_slope_2
    ref_pool[1, :, 9] = des_brake_2

    ref_pool[2, :, 0] = ref_pos_x_3
    ref_pool[2, :, 1] = ref_pos_y_3
    ref_pool[2, :, 2] = ref_yaw_3
    ref_pool[2, :, 3] = des_torque_3
    ref_pool[2, :, 4] = des_delta_3
    ref_pool[2, :, 5] = ref_vx_3
    ref_pool[2, :, 6] = pos_z_3
    ref_pool[2, :, 7] = longi_slope_3
    ref_pool[2, :, 8] = lat_slope_3
    ref_pool[2, :, 9] = des_brake_3

    error_pool = np.zeros((3, len(step_list), 4))
    error_pool[0, :, 0] = pos_y_err_pre
    error_pool[0, :, 1] = vx_err_nt
    error_pool[0, :, 2] = pos_y_err_nt
    error_pool[0, :, 3] = yaw_err_nt

    error_pool[1, :, 0] = pos_y_err_pre_2
    error_pool[1, :, 1] = vx_err_nt_2
    error_pool[1, :, 2] = pos_y_err_nt_2
    error_pool[1, :, 3] = yaw_err_nt_2

    error_pool[2, :, 0] = pos_y_err_pre_3
    error_pool[2, :, 1] = vx_err_nt_3
    error_pool[2, :, 2] = pos_y_err_nt_3
    error_pool[2, :, 3] = yaw_err_nt_3
    return state_pool, ref_pool, error_pool

def read_globaltraj(root_path):
    data_result = pd.DataFrame(pd.read_csv(root_path, header=None))
    start_index = 1
    end_index = 3500
    interval = 30
    glo_x = np.array(data_result.iloc[start_index:end_index:interval, 0], dtype='float32')
    glo_y = -np.array(data_result.iloc[start_index:end_index:interval, 1], dtype='float32')
    glo_yaw = -np.array(data_result.iloc[start_index:end_index:interval, 2], dtype='float32')

    state_pool = np.zeros((len(glo_x), 3))
    state_pool[:, 0] = glo_x
    state_pool[:, 1] = glo_y
    state_pool[:, 2] = glo_yaw

    return state_pool

def plot_Timevs_(state,ref, args):
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

    # for i in range(line_num):
    #     legend = (
    #         legend_list[i]
    #         if len(legend_list) == line_num
    #         else print("line_num != len of legend_list")
    #     )
    #     color = (
    #         color_list[i]
    #         if len(color_list) == line_num
    #         else print("line_num != len of legend_list")
    #     )
    #     lw = 2
    #     if i == 1:
    #         lw=4
    #     sns.lineplot(x=data_read[0, :]*dt, y=data_read[i+1, :], linewidth=lw, color="{}".format(color), label="{}".format(legend)) #
        # plt.scatter(x=data_x[i + 1, :], y=data_[i + 1, :], label="{}".format(legend), s=2)
    # x = [0, 2.5, 5.0, 7.5]
    # plt.xticks(x)
    # plt.yticks(range(0,50000,10000))
    # l_glo = plt.scatter(x=state[1:, 0]*dt, y=state[1:, 2], label="glo", s=3)
    l_ref = sns.lineplot(x=state[1:, 0]*dt, y=ref[1:, 4], label="ref", linewidth=2)
    l_self = sns.lineplot(x=state[1:, 0]*dt, y=state[1:, 13], label="state", linewidth=2)

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
    # plt.savefig(
    #     path_state_fmtpdf, format="pdf", bbox_inches="tight"
    # )

    plt.close()

def plot_stateXvs_(state,ref, glo, args):
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

    # for i in range(line_num):
    #     legend = (
    #         legend_list[i]
    #         if len(legend_list) == line_num
    #         else print("line_num != len of legend_list")
    #     )
    #     color = (
    #         color_list[i]
    #         if len(color_list) == line_num
    #         else print("line_num != len of legend_list")
    #     )

    # sns.lineplot(x=data_x[i+1, :], y=data_[i+1, :], label="{}".format(legend), linewidth=2,color="{}".format(color))  #
    # l1 = plt.scatter(x=data.iloc[1:, 42], y=data.iloc[1:, 43], label="glo", s=2) #, linewidths=0.1


    l_glo = plt.scatter(glo[1:, 0], glo[1:, 1], label="glo", s=3)
    l_ref = plt.scatter(ref[1:, 0], ref[1:, 1], label="ref", s=2)
    l_self = plt.scatter(state[1:, 1], state[1:, 2], label="state", s=1)

    # l_ref = sns.lineplot(x=state[1:, 1], y=ref[1:, 4], label="ref", linewidth=2)
    # l_self = sns.lineplot(x=state[1:, 1], y=state[1:, 13], label="state", linewidth=2)
    # l2 = plt.scatter(x=data.iloc[1:, 23], y=data.iloc[1:, 26], label="ref", s=2)
    # l3 = plt.scatter(x=data.iloc[1:, 18], y=data.iloc[1:, 19], label="real", s=2)
    # x = [0, 5, 10, 15, 20] sns.lineplot
    # plt.xticks(range(0,8,1))
    # plt.axis('equal')
    # plt.legend(ncol=1)
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
    # plt.savefig(
    #     path_state_fmt_o, format="pdf", bbox_inches="tight"
    # )
    plt.close()

def plot_experiment(state_pool, ref_pool, error_pool):
    save_dir = args["figures_root"] + '/results_plot' + "/"
    linestyle_list = args["linestyle_list"]
    color_list = args["color_list"]
    os.makedirs(save_dir, exist_ok=True)
    fig_size = (
        default_cfg["fig_size"],
        default_cfg["fig_size"],
    )
    fig_size12_9 = (
        default_cfg["fig_size12-9"],
        default_cfg["fig_size12-9"],
    )

    ax_para = default_cfg["ax_para"]
    ax_para2 = default_cfg["ax_para2"]
    # --------------------------- plot ----------------------------------------
    fig1 = plt.figure('traj', figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])
    ax = fig1.add_axes(ax_para)  # [left, bottom, width, height]
    line_ref, = ax.plot(ref_pool[:, 0], ref_pool[:, 1], c=color_list[0], linestyle=linestyle_list[0],
                        linewidth=2)  #
    line_CTMPC, = ax.plot(state_pool[:, 2], state_pool[ :, 1], c=color_list[3], linestyle=linestyle_list[3],
                          linewidth=2)  #
    ax.set_xlabel(r"$p_{\rm x}\ /\mathrm{m}$", default_cfg["label_font"])
    ax.set_ylabel(r"$p_{\rm y}\ /\mathrm{m}$", default_cfg["label_font"])
    ax.tick_params(labelsize=default_cfg["tick_size"])
    ax.tick_params(axis='x', direction='in')  # x轴刻度线向内
    ax.tick_params(axis='y', direction='in')  # y轴刻度线向内
    ax.legend([line_ref, line_CTMPC], ['Ref', 'CTMPC'],
              loc='best', prop=default_cfg["legend_font"], frameon=False, ncol=2)
    plt.savefig(os.path.join(save_dir, "x-y.{}".format(default_cfg["img_fmt"])))

    fig2 = plt.figure('x-z', figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])
    ax = fig2.add_axes(ax_para)  # [left, bottom, width, height]
    line_z, = ax.plot(state_pool[:, 2], ref_pool[:, 6], color=color_list[0], linewidth=2)
    ax.set_ylabel(r"$p_{\rm z}\ /\mathrm{m}$", default_cfg["label_font"])
    ax.set_xlabel(r"$p_x /\mathrm{m}$", default_cfg["label_font"])
    ax.tick_params(labelsize=default_cfg["tick_size"])
    ax.tick_params(axis='x', direction='in')  # x轴刻度线向内
    ax.tick_params(axis='y', direction='in')  # y轴刻度线向内
    # fig2.tight_layout(pad=default_cfg["pad"])
    plt.savefig(os.path.join(save_dir, "x-z.{}".format(default_cfg["img_fmt"])))

    # t-ref yaw- yaw
    fig3 = plt.figure('t-heading', figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])
    ax = fig3.add_axes(ax_para)  # [left, bottom, width, height]
    line_ref, = ax.plot(state_pool[:, 0] * 0.02, ref_pool[:, 2], c=color_list[0], linestyle=linestyle_list[0],
                        linewidth=2)
    line_CTMPC, = ax.plot(state_pool[:, 0] * 0.02, state_pool[:, 3], c=color_list[3], linestyle=linestyle_list[3],
                          linewidth=2)
    # ax.legend([line_ref, line_CMMPC, line_CRMPC, line_CTMPC], ['Ref', 'CMMPC', 'CRMPC', 'CTMPC'],
    #             loc='best', prop=default_cfg["legend_font"], frameon=False, ncol=2)
    ax.set_ylabel(r"$\phi \ /\mathrm{rad}$", default_cfg["label_font"])
    ax.set_xlabel(r"Time $/\mathrm{s}$", default_cfg["label_font"])
    ax.tick_params(axis='x', direction='in')  # x轴刻度线向内
    ax.tick_params(axis='y', direction='in')  # y轴刻度线向内
    ax.tick_params(labelsize=default_cfg["tick_size"])
    plt.savefig(os.path.join(save_dir, "t-yaw.{}".format(default_cfg["img_fmt"])))


    fig4 = plt.figure('t-vx', figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])
    ax = fig4.add_axes(ax_para)  # [left, bottom, width, height]
    line_ref, = ax.plot(state_pool[:, 0] * 0.02, ref_pool[:, 5], c=color_list[0], linestyle=linestyle_list[0],
                        linewidth=2)
    line_CTMPC, = ax.plot(state_pool[:, 0] * 0.02, state_pool[:, 4], c=color_list[3],
                          linestyle=linestyle_list[3], linewidth=2)
    ax.set_ylabel(r"$v_x /\mathrm{(m·s^{-1})}$", default_cfg["label_font"])
    ax.set_xlabel(r"Time $/\mathrm{s}$", default_cfg["label_font"])
    ax.tick_params(labelsize=default_cfg["tick_size"])
    ax.tick_params(axis='x', direction='in')  # x轴刻度线向内
    ax.tick_params(axis='y', direction='in')  # y轴刻度线向内
    ax.legend([line_ref, line_CTMPC], ['Ref', 'CTMPC'],
              loc='best', prop=default_cfg["legend_font"], frameon=False, ncol=2)
    # fig4.tight_layout(pad=default_cfg["pad"])
    plt.savefig(os.path.join(save_dir, "t-vx.{}".format(default_cfg["img_fmt"])))

    # t-yerr-delta
    fig5 = plt.figure('t-yerr', figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])
    ax = fig5.add_axes(ax_para)  # [left, bottom, width, height]
    # line_posy_err, = ax21.plot(state_pool[:, 0]*0.02, error_pool[:, 2], color="magenta", linewidth=2)
    line_CTMPC, = plt.plot(state_pool[:, 0] * 0.02, error_pool[:, 2], c=color_list[3],
                           linestyle=linestyle_list[3], linewidth=2)

    ax.set_xlabel(r"Time $/\mathrm{s}$", default_cfg["label_font"])
    ax.tick_params(axis='x', direction='in')  # x轴刻度线向内
    ax.tick_params(axis='y', direction='in')  # y轴刻度线向内
    ax.set_ylabel(r"$y^\mathrm{err} /\mathrm{m}$", default_cfg["label_font"])

    plt.tick_params(labelsize=default_cfg["tick_size"])

    # fig2.suptitle('t-lateral', fontsize=20)
    # plt.tight_layout(pad=default_cfg["pad"])
    # plt.legend([line_posy_err, line_yaw_err], ['y_err', 'yaw_err'],
    #            loc='best', prop=default_cfg["legend_font"], frameon=False)
    plt.savefig(os.path.join(save_dir, "t-y_err.{}".format(default_cfg["img_fmt"])))

    # t-yawerr-delta
    fig6 = plt.figure('t-heading err', figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])
    ax = fig6.add_axes(ax_para)  # [left, bottom, width, height]
    # line_posy_err, = ax21.plot(state_pool[:, 0]*0.02, error_pool[:, 2], color="magenta", linewidth=2)

    line_CTMPC, = plt.plot(state_pool[:, 0] * 0.02, error_pool[:, 3], c=color_list[3],
                           linestyle=linestyle_list[3], linewidth=2)
    # plt.legend([line_CMMPC, line_CRMPC, line_CTMPC], ['CMMPC', 'CRMPC', 'CTMPC'],
    #             loc='best', prop=default_cfg["legend_font"], frameon=False, ncol=2)

    ax.set_xlabel(r"Time $/\mathrm{s}$", default_cfg["label_font"])
    ax.set_ylabel(r"$\phi^\mathrm{err}/\mathrm{rad}$", default_cfg["label_font"])
    ax.tick_params(axis='x', direction='in')  # x轴刻度线向内
    ax.tick_params(axis='y', direction='in')  # y轴刻度线向内
    plt.tick_params(labelsize=default_cfg["tick_size"])
    # fig2.suptitle('t-lateral', fontsize=20)
    # plt.tight_layout(pad=default_cfg["pad"])
    # plt.legend([line_posy_err, line_yaw_err], ['y_err', 'yaw_err'],
    #            loc='best', prop=default_cfg["legend_font"], frameon=False)
    plt.savefig(os.path.join(save_dir, "t-yaw_err.{}".format(default_cfg["img_fmt"])))

    fig7 = plt.figure('t-vx err', figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])
    ax = fig7.add_axes(ax_para)  # [left, bottom, width, height]
    line_CTMPC, = ax.plot(state_pool[:, 0] * 0.02, error_pool[:, 1], c=color_list[3],
                          linestyle=linestyle_list[3], linewidth=2)

    ax.set_ylabel(r"$v_x^\mathrm{err} /\mathrm{(m·s^{-1})}$", default_cfg["label_font"])
    ax.set_xlabel(r"Time $/\mathrm{s}$", default_cfg["label_font"])
    plt.tick_params(labelsize=default_cfg["tick_size"])
    ax.tick_params(axis='x', direction='in')  # x轴刻度线向内
    ax.tick_params(axis='y', direction='in')  # y轴刻度线向内
    plt.savefig(os.path.join(save_dir, "t-vx_err.{}".format(default_cfg["img_fmt"])))

    # line_car, = ax42.plot(state_pool[:, 0] * 0.02, state_pool[:, 5], color="magenta",
    #                      linewidth=2, label="vy")
    fig8 = plt.figure('t-vy', figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])
    ax = fig8.add_axes(ax_para)  # [left, bottom, width, height]
    line_CTMPC, = ax.plot(state_pool[:, 0] * 0.02, state_pool[:, 5], c=color_list[3],
                          linestyle=linestyle_list[3], linewidth=2)

    ax.set_ylabel(r"$v_y /\mathrm{(m·s^{-1})}$", default_cfg["label_font"])
    ax.set_xlabel(r"Time $/\mathrm{s}$", default_cfg["label_font"])
    # ax.set_facecolor('#E6E6E6')
    # ax.grid()
    # ax.set_aspect('equal')
    # fig.suptitle('x-y', fontsize=40)
    plt.tick_params(labelsize=default_cfg["tick_size"])
    ax.tick_params(axis='x', direction='in')  # x轴刻度线向内
    ax.tick_params(axis='y', direction='in')  # y轴刻度线向内
    # fig4.tight_layout(pad=default_cfg["pad"])
    plt.savefig(os.path.join(save_dir, "t-vy.{}".format(default_cfg["img_fmt"])))

    # t-roll, t-rollrate
    fig9 = plt.figure('t-varphi', figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])
    ax = fig9.add_axes(ax_para)  # [left, bottom, width, height]

    line_CTMPC, = ax.plot(state_pool[:, 0] * 0.02, state_pool[:, 7], c=color_list[3],
                          linestyle=linestyle_list[3], linewidth=2)
    ax.tick_params(labelsize=default_cfg["tick_size"])
    ax.set_ylabel(r"$\varphi /\mathrm{(rad)}$", default_cfg["label_font"])
    ax.set_xlabel(r"Time $/\mathrm{s}$", default_cfg["label_font"])
    ax.tick_params(axis='x', direction='in')  # x轴刻度线向内
    ax.tick_params(axis='y', direction='in')  # y轴刻度线向内
    plt.savefig(os.path.join(save_dir, "t-roll.{}".format(default_cfg["img_fmt"])))

    fig10 = plt.figure('t-varphidot', figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])
    ax = fig10.add_axes(ax_para)  # [left, bottom, width, height]
    line_CTMPC, = ax.plot(state_pool[:, 0] * 0.02, state_pool[:, 8], c=color_list[3],
                          linestyle=linestyle_list[3], linewidth=2)

    ax.set_ylabel(r"$\dot\varphi /\mathrm{(rad·s^{-1})}$", default_cfg["label_font"])
    ax.set_xlabel(r"Time $/\mathrm{s}$", default_cfg["label_font"])
    ax.tick_params(labelsize=default_cfg["tick_size"])
    ax.tick_params(axis='x', direction='in')  # x轴刻度线向内
    ax.tick_params(axis='y', direction='in')  # y轴刻度线向内
    # fig8.tight_layout(pad=default_cfg["pad"])
    plt.savefig(os.path.join(save_dir, "t-rollrate.{}".format(default_cfg["img_fmt"])))

    # t-torque
    # fig8, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])
    fig8 = plt.figure('torque', figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])
    ax = fig8.add_axes(ax_para)  # [left, bottom, width, height]
    line_fb, = ax.plot(state_pool[:, 0] * 0.02, state_pool[:, 9], c=color_list[0], linestyle=linestyle_list[0],
                          linewidth=2)  #
    line_CTMPC, = ax.plot(state_pool[:, 0] * 0.02, ref_pool[:, 3], c=color_list[3], linestyle=linestyle_list[3],
                          linewidth=2)  #
    # ax.legend([line_CMMPC, line_CRMPC, line_CTMPC], ['CMMPC', 'CRMPC', 'CTMPC'],
    #           loc=3, prop=default_cfg["legend_font"], frameon=False, ncol=3, bbox_to_anchor=(-0.02, -0.12))
    ax.set_ylabel(r"$Q_{\rm fl}/\mathrm{(N·m)}$", default_cfg["label_font"])
    ax.set_xlabel(r"Time $/\mathrm{s}$", default_cfg["label_font"])
    ax.tick_params(axis='x', direction='in')  # x轴刻度线向内
    ax.tick_params(axis='y', direction='in')  # y轴刻度线向内
    # ax.set_facecolor('#E6E6E6')
    # ax.grid()
    # ax.set_aspect('equal')
    # fig7.suptitle('t-control-des -fb', fontsize=20)
    ax.tick_params(labelsize=default_cfg["tick_size"])
    # fig8.tight_layout(pad=default_cfg["pad"])
    plt.savefig(os.path.join(save_dir, "torque_fl.{}".format(default_cfg["img_fmt"])))


    fig10 = plt.figure('delta_f', figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])
    ax = fig10.add_axes(ax_para)  # [left, bottom, width, height]

    line_fb, = ax.plot(state_pool[:, 0] * 0.02, state_pool[:, 13], c=color_list[0],
                          linestyle=linestyle_list[0],
                          linewidth=3)  #
    line_CTMPC, = ax.plot(state_pool[:, 0] * 0.02, ref_pool[:, 4],  c=color_list[3],
                          linestyle=linestyle_list[3],
                          linewidth=2)

    ax.set_ylabel(r"$\delta\ /\degree$", default_cfg["label_font"])
    ax.set_xlabel(r"Time $/\mathrm{s}$", default_cfg["label_font"])
    ax.legend([line_fb, line_CTMPC], ['Feedback', 'CTMPC'],
              loc=9, prop=default_cfg["legend_font"], frameon=False, ncol=2)
    ax.tick_params(axis='x', direction='in')  # x轴刻度线向内
    ax.tick_params(axis='y', direction='in')  # y轴刻度线向内
    # ax.set_facecolor('#E6E6E6')
    # ax.grid()
    # ax.set_aspect('equal')
    # fig7.suptitle('t-control-des -fb', fontsize=20)
    ax.tick_params(labelsize=default_cfg["tick_size"])
    # fig10.tight_layout(pad=default_cfg["pad"])
    plt.savefig(os.path.join(save_dir, "delta_f.{}".format(default_cfg["img_fmt"])))

    fig10 = plt.figure('time_calc', figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])
    ax = fig10.add_axes(ax_para)  # [left, bottom, width, height]

    line_CTMPC, = ax.plot(state_pool[:, 0] * 0.02, ref_pool[:, 7],  c=color_list[3],
                          linestyle=linestyle_list[3],
                          linewidth=2)

    ax.set_ylabel(r"Time$T_c\ /\mathrm{ms}$", default_cfg["label_font"])
    ax.set_xlabel(r"Time $/\mathrm{s}$", default_cfg["label_font"])
    ax.tick_params(axis='x', direction='in')  # x轴刻度线向内
    ax.tick_params(axis='y', direction='in')  # y轴刻度线向内
    # ax.set_facecolor('#E6E6E6')
    # ax.grid()
    # ax.set_aspect('equal')
    # fig7.suptitle('t-control-des -fb', fontsize=20)
    ax.tick_params(labelsize=default_cfg["tick_size"])
    # fig10.tight_layout(pad=default_cfg["pad"])
    plt.savefig(os.path.join(save_dir, "time_calc.{}".format(default_cfg["img_fmt"])))

    fig11 = plt.figure('slope', figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])
    ax = fig11.add_axes(ax_para)  # [left, bottom, width, height]

    line_longi, = ax.plot(state_pool[:, 2], ref_pool[:, 7], c=color_list[0],
                          linestyle=linestyle_list[0],
                          linewidth=2)
    line_lat, = ax.plot(state_pool[:, 2], ref_pool[:, 8]/10, c=color_list[3],
                          linestyle=linestyle_list[3],
                          linewidth=2)

    ax.set_ylabel(r"Slope$/\mathrm{rad}$", default_cfg["label_font"])
    ax.set_xlabel(r"$p_x /\mathrm{m}$", default_cfg["label_font"])
    ax.tick_params(axis='x', direction='in')  # x轴刻度线向内
    ax.tick_params(axis='y', direction='in')  # y轴刻度线向内
    ax.legend([line_longi, line_lat], [r'$\theta_{\rm r}$', r'$\varphi_{\rm r}$'],
              loc=2, prop=default_cfg["legend_font"], frameon=False, ncol=2, bbox_to_anchor=(-0.02, 1.10))
    # ax.set_facecolor('#E6E6E6')
    # ax.grid()
    # ax.set_aspect('equal')
    # fig7.suptitle('t-control-des -fb', fontsize=20)
    ax.tick_params(labelsize=default_cfg["tick_size"])
    # fig10.tight_layout(pad=default_cfg["pad"])
    plt.savefig(os.path.join(save_dir, "slope.{}".format(default_cfg["img_fmt"])))

def plot_experiment_multi_method(state_pool, ref_pool, error_pool):
    save_dir = args["figures_root"] + '/results_plot'+"/"
    linestyle_list = args["linestyle_list"]
    color_list = args["color_list"]
    os.makedirs(save_dir, exist_ok=True)
    fig_size = (
        default_cfg["fig_size"],
        default_cfg["fig_size"],
    )
    fig_size12_9 = (
        default_cfg["fig_size12-9"],
        default_cfg["fig_size12-9"],
    )

    ax_para = default_cfg["ax_para"]
    ax_para2 = default_cfg["ax_para2"]
    # --------------------------- plot ----------------------------------------
    fig1 = plt.figure('traj', figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])
    ax = fig1.add_axes(ax_para)  # [left, bottom, width, height]
    line_ref, = ax.plot(ref_pool[0, :, 0], ref_pool[0, :, 1], c=color_list[0],linestyle=linestyle_list[0], linewidth=2)#
    line_CMMPC, = ax.plot(state_pool[0, :, 1], state_pool[0, :, 2], c=color_list[1],linestyle=linestyle_list[1], linewidth=2)#
    line_CRMPC, = ax.plot(state_pool[1, :, 1], state_pool[1, :, 2], c=color_list[2],linestyle=linestyle_list[2], linewidth=2)#
    line_CTMPC, = ax.plot(state_pool[2, :, 1], state_pool[2, :, 2], c=color_list[3],linestyle=linestyle_list[3], linewidth=2)#
    ax.set_xlabel(r"$p_{\rm x}\ /\mathrm{m}$", default_cfg["label_font"])
    ax.set_ylabel(r"$p_{\rm y}\ /\mathrm{m}$", default_cfg["label_font"])
    ax.tick_params(labelsize=default_cfg["tick_size"])
    ax.tick_params(axis='x', direction='in')  # x轴刻度线向内
    ax.tick_params(axis='y', direction='in')  # y轴刻度线向内
    ax.legend([line_ref, line_CMMPC, line_CRMPC,line_CTMPC], ['Ref', 'CMMPC', 'CRMPC', 'CTMPC'],
               loc='best', prop=default_cfg["legend_font"], frameon=False, ncol=2)
    plt.savefig(os.path.join(save_dir, "x-y.{}".format(default_cfg["img_fmt"])))

    fig2 = plt.figure('x-z', figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])
    ax = fig2.add_axes(ax_para)  # [left, bottom, width, height]
    line_z, = ax.plot(state_pool[0, :, 1], ref_pool[0, :, 6], color=color_list[0], linewidth=2)
    ax.set_ylabel(r"$p_{\rm z}\ /\mathrm{m}$", default_cfg["label_font"])
    ax.set_xlabel(r"$p_x /\mathrm{m}$", default_cfg["label_font"])
    ax.tick_params(labelsize=default_cfg["tick_size"])
    ax.tick_params(axis='x', direction='in')  # x轴刻度线向内
    ax.tick_params(axis='y', direction='in')  # y轴刻度线向内
    # fig2.tight_layout(pad=default_cfg["pad"])
    plt.savefig(os.path.join(save_dir, "x-z.{}".format(default_cfg["img_fmt"])))

    # t-ref yaw- yaw
    fig3 = plt.figure('t-heading', figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])
    ax = fig3.add_axes(ax_para)  # [left, bottom, width, height]
    line_ref, = ax.plot(state_pool[0, :, 0] * 0.02, ref_pool[0, :, 2], c=color_list[0],linestyle=linestyle_list[0], linewidth=2)
    line_CMMPC, = ax.plot(state_pool[0, :, 0] * 0.02, state_pool[0, :, 3], c=color_list[1],linestyle=linestyle_list[1], linewidth=2)
    line_CRMPC, = ax.plot(state_pool[1, :, 0] * 0.02, state_pool[1, :, 3], c=color_list[2],linestyle=linestyle_list[2], linewidth=2)
    line_CTMPC, = ax.plot(state_pool[2, :, 0] * 0.02, state_pool[2, :, 3], c=color_list[3],linestyle=linestyle_list[3], linewidth=2)
    # ax.legend([line_ref, line_CMMPC, line_CRMPC, line_CTMPC], ['Ref', 'CMMPC', 'CRMPC', 'CTMPC'],
    #             loc='best', prop=default_cfg["legend_font"], frameon=False, ncol=2)
    ax.set_ylabel(r"$\phi \ /\mathrm{rad}$", default_cfg["label_font"])
    ax.set_xlabel(r"Time $/\mathrm{s}$", default_cfg["label_font"])
    ax.tick_params(axis='x', direction='in')  # x轴刻度线向内
    ax.tick_params(axis='y', direction='in')  # y轴刻度线向内
    ax.tick_params(labelsize=default_cfg["tick_size"])
    plt.savefig(os.path.join(save_dir, "t-yaw.{}".format(default_cfg["img_fmt"])))

    fig4 = plt.figure('t-vx', figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])
    ax = fig4.add_axes(ax_para)  # [left, bottom, width, height]
    line_ref, = ax.plot(state_pool[0, :, 0] * 0.02, ref_pool[0, :, 5], c=color_list[0],linestyle=linestyle_list[0], linewidth=2)
    line_CMMPC, = ax.plot(state_pool[0, :, 0] * 0.02, state_pool[0, :, 4], c=color_list[1],
                            linestyle=linestyle_list[1], linewidth=2)
    line_CRMPC, = ax.plot(state_pool[1, :, 0] * 0.02, state_pool[1, :, 4], c=color_list[2],
                            linestyle=linestyle_list[2], linewidth=2)
    line_CTMPC, = ax.plot(state_pool[2, :, 0] * 0.02, state_pool[2, :, 4], c=color_list[3],
                            linestyle=linestyle_list[3], linewidth=2)

    ax.set_ylabel(r"$v_x /\mathrm{(m·s^{-1})}$", default_cfg["label_font"])
    ax.set_xlabel(r"Time $/\mathrm{s}$", default_cfg["label_font"])
    ax.tick_params(labelsize=default_cfg["tick_size"])
    ax.tick_params(axis='x', direction='in')  # x轴刻度线向内
    ax.tick_params(axis='y', direction='in')  # y轴刻度线向内
    # fig4.tight_layout(pad=default_cfg["pad"])
    plt.savefig(os.path.join(save_dir, "t-vx.{}".format(default_cfg["img_fmt"])))

    # t-yerr-delta
    fig5 = plt.figure('t-yerr', figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])
    ax = fig5.add_axes(ax_para)  # [left, bottom, width, height]
    # line_posy_err, = ax21.plot(state_pool[:, 0]*0.02, error_pool[:, 2], color="magenta", linewidth=2)
    line_CMMPC, = plt.plot(state_pool[0, :, 0] * 0.02, error_pool[0, :, 2], c=color_list[1],
                            linestyle=linestyle_list[1], linewidth=2)
    line_CRMPC, = plt.plot(state_pool[1, :, 0] * 0.02, error_pool[1, :, 2], c=color_list[2],
                            linestyle=linestyle_list[2], linewidth=2)
    line_CTMPC, = plt.plot(state_pool[2, :, 0] * 0.02, error_pool[2, :, 2], c=color_list[3],
                            linestyle=linestyle_list[3], linewidth=2)
    # ax.legend([line_CMMPC, line_CRMPC, line_CTMPC], ['CMMPC', 'CRMPC', 'CTMPC'],
    #             loc='best', prop=default_cfg["legend_font"], frameon=False, ncol=2)
    ax.set_xlabel(r"Time $/\mathrm{s}$", default_cfg["label_font"])
    ax.tick_params(axis='x', direction='in')  # x轴刻度线向内
    ax.tick_params(axis='y', direction='in')  # y轴刻度线向内
    ax.set_ylabel(r"$y^\mathrm{err} /\mathrm{m}$", default_cfg["label_font"])

    plt.tick_params(labelsize=default_cfg["tick_size"])

    # fig2.suptitle('t-lateral', fontsize=20)
    # plt.tight_layout(pad=default_cfg["pad"])
    # plt.legend([line_posy_err, line_yaw_err], ['y_err', 'yaw_err'],
    #            loc='best', prop=default_cfg["legend_font"], frameon=False)
    plt.savefig(os.path.join(save_dir, "t-y_err.{}".format(default_cfg["img_fmt"])))

    # t-yawerr-delta
    fig6 = plt.figure('t-heading err', figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])
    ax = fig6.add_axes(ax_para)  # [left, bottom, width, height]
    # line_posy_err, = ax21.plot(state_pool[:, 0]*0.02, error_pool[:, 2], color="magenta", linewidth=2)

    # line_yaw_err, = ax22.plot(state_pool[:, 0]*0.02, error_pool[:, 3],
    #                     color="magenta", linewidth=2)
    line_CMMPC, = plt.plot(state_pool[0, :, 0] * 0.02, error_pool[0, :, 3], c=color_list[1],
                            linestyle=linestyle_list[1], linewidth=2)
    line_CRMPC, = plt.plot(state_pool[1, :, 0] * 0.02, error_pool[1, :, 3], c=color_list[2],
                            linestyle=linestyle_list[2], linewidth=2)
    line_CTMPC, = plt.plot(state_pool[2, :, 0] * 0.02, error_pool[2, :, 3], c=color_list[3],
                            linestyle=linestyle_list[3], linewidth=2)
    # plt.legend([line_CMMPC, line_CRMPC, line_CTMPC], ['CMMPC', 'CRMPC', 'CTMPC'],
    #             loc='best', prop=default_cfg["legend_font"], frameon=False, ncol=2)

    ax.set_xlabel(r"Time $/\mathrm{s}$", default_cfg["label_font"])
    ax.set_ylabel(r"$\phi^\mathrm{err}/\mathrm{rad}$", default_cfg["label_font"])
    ax.tick_params(axis='x', direction='in')  # x轴刻度线向内
    ax.tick_params(axis='y', direction='in')  # y轴刻度线向内
    plt.tick_params(labelsize=default_cfg["tick_size"])
    # fig2.suptitle('t-lateral', fontsize=20)
    # plt.tight_layout(pad=default_cfg["pad"])
    # plt.legend([line_posy_err, line_yaw_err], ['y_err', 'yaw_err'],
    #            loc='best', prop=default_cfg["legend_font"], frameon=False)
    plt.savefig(os.path.join(save_dir, "t-yaw_err.{}".format(default_cfg["img_fmt"])))


    #t-vy
    # t-yerr-yawerr-delta
    fig7 = plt.figure('t-vx err', figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])
    ax = fig7.add_axes(ax_para)  # [left, bottom, width, height]

    # line_vx_err, = ax41.plot(state_pool[:, 0] * 0.02, error_pool[:, 1],
    #                           color="magenta", linewidth=2)
    line_CMMPC, = ax.plot(state_pool[0, :, 0] * 0.02, error_pool[0, :, 1], c=color_list[1],
                           linestyle=linestyle_list[1], linewidth=2)
    line_CRMPC, = ax.plot(state_pool[1, :, 0] * 0.02, error_pool[1, :, 1], c=color_list[2],
                           linestyle=linestyle_list[2], linewidth=2)
    line_CTMPC, = ax.plot(state_pool[2, :, 0] * 0.02, error_pool[2, :, 1], c=color_list[3],
                           linestyle=linestyle_list[3], linewidth=2)
    ax.legend([line_CMMPC, line_CRMPC, line_CTMPC], ['CMMPC', 'CRMPC', 'CTMPC'],
               loc='best', prop=default_cfg["legend_font"], frameon=False, ncol=2)
    ax.set_ylabel(r"$v_x^\mathrm{err} /\mathrm{(m·s^{-1})}$", default_cfg["label_font"])
    ax.set_xlabel(r"Time $/\mathrm{s}$", default_cfg["label_font"])
    plt.tick_params(labelsize=default_cfg["tick_size"])
    ax.tick_params(axis='x', direction='in')  # x轴刻度线向内
    ax.tick_params(axis='y', direction='in')  # y轴刻度线向内
    plt.savefig(os.path.join(save_dir, "t-vx_err.{}".format(default_cfg["img_fmt"])))

    # line_car, = ax42.plot(state_pool[:, 0] * 0.02, state_pool[:, 5], color="magenta",
    #                      linewidth=2, label="vy")
    fig8 = plt.figure('t-vy', figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])
    ax = fig8.add_axes(ax_para)  # [left, bottom, width, height]
    line_CMMPC, = ax.plot(state_pool[0, :, 0] * 0.02, state_pool[0, :, 5], c=color_list[1],
                           linestyle=linestyle_list[1], linewidth=2)
    line_CRMPC, = ax.plot(state_pool[1, :, 0] * 0.02, state_pool[1, :, 5], c=color_list[2],
                           linestyle=linestyle_list[2], linewidth=2)
    line_CTMPC, = ax.plot(state_pool[2, :, 0] * 0.02, state_pool[2, :, 5], c=color_list[3],
                           linestyle=linestyle_list[3], linewidth=2)

    ax.set_ylabel(r"$v_y /\mathrm{(m·s^{-1})}$", default_cfg["label_font"])
    ax.set_xlabel(r"Time $/\mathrm{s}$", default_cfg["label_font"])
    # ax.set_facecolor('#E6E6E6')
    # ax.grid()
    # ax.set_aspect('equal')
    # fig.suptitle('x-y', fontsize=40)
    plt.tick_params(labelsize=default_cfg["tick_size"])
    ax.tick_params(axis='x', direction='in')  # x轴刻度线向内
    ax.tick_params(axis='y', direction='in')  # y轴刻度线向内
    # fig4.tight_layout(pad=default_cfg["pad"])
    plt.savefig(os.path.join(save_dir, "t-vy.{}".format(default_cfg["img_fmt"])))

    # t-roll, t-rollrate
    fig9 = plt.figure('t-varphi', figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])
    ax = fig9.add_axes(ax_para)  # [left, bottom, width, height]
    # line_car, = ax81.plot(state_pool[:, 0] * 0.02, state_pool[:, 7], color="magenta",
    #                       linewidth=2)
    line_CMMPC, = ax.plot(state_pool[0, :, 0] * 0.02, state_pool[0, :, 7], c=color_list[1],
                           linestyle=linestyle_list[1], linewidth=2)
    line_CRMPC, = ax.plot(state_pool[1, :, 0] * 0.02, state_pool[1, :, 7], c=color_list[2],
                           linestyle=linestyle_list[2], linewidth=2)
    line_CTMPC, = ax.plot(state_pool[2, :, 0] * 0.02, state_pool[2, :, 7], c=color_list[3],
                           linestyle=linestyle_list[3], linewidth=2)
    ax.tick_params(labelsize=default_cfg["tick_size"])
    ax.set_ylabel(r"$\varphi /\mathrm{(rad)}$", default_cfg["label_font"])
    ax.set_xlabel(r"Time $/\mathrm{s}$", default_cfg["label_font"])
    ax.tick_params(axis='x', direction='in')  # x轴刻度线向内
    ax.tick_params(axis='y', direction='in')  # y轴刻度线向内
    plt.savefig(os.path.join(save_dir, "t-roll.{}".format(default_cfg["img_fmt"])))

    # ax81.legend([line_car], ['Roll'], loc='best',
    #            prop=default_cfg["legend_font"], frameon=False)#prop=default_cfg["legend_font"], frameon=False)#

    # line_self, = ax82.plot(state_pool[:, 0] * 0.02, state_pool[:, 8], color="magenta",
    #                        linewidth=2, alpha=0.7)
    fig10 = plt.figure('t-varphidot', figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])
    ax = fig10.add_axes(ax_para)  # [left, bottom, width, height]
    line_CMMPC, = ax.plot(state_pool[0, :, 0] * 0.02, state_pool[0, :, 8], c=color_list[1],
                           linestyle=linestyle_list[1], linewidth=2)
    line_CRMPC, = ax.plot(state_pool[1, :, 0] * 0.02, state_pool[1, :, 8], c=color_list[2],
                           linestyle=linestyle_list[2], linewidth=2)
    line_CTMPC, = ax.plot(state_pool[2, :, 0] * 0.02, state_pool[2, :, 8], c=color_list[3],
                           linestyle=linestyle_list[3], linewidth=2)

    ax.set_ylabel(r"$\dot\varphi /\mathrm{(rad·s^{-1})}$", default_cfg["label_font"])
    ax.set_xlabel(r"Time $/\mathrm{s}$", default_cfg["label_font"])
    ax.tick_params(labelsize=default_cfg["tick_size"])
    ax.tick_params(axis='x', direction='in')  # x轴刻度线向内
    ax.tick_params(axis='y', direction='in')  # y轴刻度线向内
    # fig8.tight_layout(pad=default_cfg["pad"])
    plt.savefig(os.path.join(save_dir, "t-rollrate.{}".format(default_cfg["img_fmt"])))

    

    # t-torque
    # fig8, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])
    fig8 = plt.figure('torque', figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])
    ax = fig8.add_axes(ax_para)  # [left, bottom, width, height]
    # line_car, = ax71.plot(state_pool[:, 0] * 0.02, state_pool[:, 9], color="lime",
    #                      linewidth=2)
    # line_ref, = ax71.plot(state_pool[:, 0] * 0.02, ref_pool[:, 3], color="magenta",
    #                       linewidth=2, linestyle='--')

    # line_ref, = ax71.plot(state_pool[0, :, 0] * 0.02, ref_pool[0, :, 3], c=color_list[0], linestyle=linestyle_list[0],
    #                       linewidth=2)  #
    line_CMMPC, = ax.plot(state_pool[0, :, 0] * 0.02, ref_pool[0, :, 3], c=color_list[1], linestyle=linestyle_list[1],
                            linewidth=2)  #
    line_CRMPC, = ax.plot(state_pool[1, :, 0] * 0.02, ref_pool[1, :, 3], c=color_list[2], linestyle=linestyle_list[2],
                            linewidth=2)  #
    line_CTMPC, = ax.plot(state_pool[2, :, 0] * 0.02, ref_pool[2, :, 3], c=color_list[3], linestyle=linestyle_list[3],
                            linewidth=2)  #
    ax.legend([line_CMMPC, line_CRMPC, line_CTMPC], ['CMMPC', 'CRMPC', 'CTMPC'],
                loc=3, prop=default_cfg["legend_font"], frameon=False, ncol=3, bbox_to_anchor=(-0.02, -0.12))
    ax.set_ylabel(r"$Q_{\rm fl}/\mathrm{(N·m)}$", default_cfg["label_font"])
    ax.set_xlabel(r"Time $/\mathrm{s}$", default_cfg["label_font"])
    ax.tick_params(axis='x', direction='in')  # x轴刻度线向内
    ax.tick_params(axis='y', direction='in')  # y轴刻度线向内
    # ax.set_facecolor('#E6E6E6')
    # ax.grid()
    # ax.set_aspect('equal')
    # fig7.suptitle('t-control-des -fb', fontsize=20)
    ax.tick_params(labelsize=default_cfg["tick_size"])
    # fig8.tight_layout(pad=default_cfg["pad"])
    plt.savefig(os.path.join(save_dir, "torque_fl.{}".format(default_cfg["img_fmt"])))

    # t-torque
    # fig8, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])
    fig = plt.figure('torque_rr', figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])
    ax = fig.add_axes(ax_para)  # [left, bottom, width, height]
    # line_car, = ax71.plot(state_pool[:, 0] * 0.02, state_pool[:, 9], color="lime",
    #                      linewidth=2)
    # line_ref, = ax71.plot(state_pool[:, 0] * 0.02, ref_pool[:, 3], color="magenta",
    #                       linewidth=2, linestyle='--')

    # line_ref, = ax71.plot(state_pool[0, :, 0] * 0.02, ref_pool[0, :, 3], c=color_list[0], linestyle=linestyle_list[0],
    #                       linewidth=2)  #
    line_CMMPC, = ax.plot(state_pool[0, :, 0] * 0.02, state_pool[0, :, 12], c=color_list[1], linestyle=linestyle_list[1],
                          linewidth=2)  #
    line_CRMPC, = ax.plot(state_pool[1, :, 0] * 0.02, state_pool[1, :, 12], c=color_list[2], linestyle=linestyle_list[2],
                          linewidth=2)  #
    line_CTMPC, = ax.plot(state_pool[2, :, 0] * 0.02, state_pool[2, :, 12], c=color_list[3], linestyle=linestyle_list[3],
                          linewidth=2)  #
    # ax.legend([line_CMMPC, line_CRMPC, line_CTMPC], ['CMMPC', 'CRMPC', 'CTMPC'],
    #             loc=2, prop=default_cfg["legend_font"], frameon=False, ncol=3, bbox_to_anchor=(-0.03, 1.65))
    ax.set_ylabel(r"$Q_{\rm rr}/\mathrm{(N·m)}$", default_cfg["label_font"])
    ax.set_xlabel(r"Time $/\mathrm{s}$", default_cfg["label_font"])
    ax.tick_params(axis='x', direction='in')  # x轴刻度线向内
    ax.tick_params(axis='y', direction='in')  # y轴刻度线向内
    # ax.set_facecolor('#E6E6E6')
    # ax.grid()
    # ax.set_aspect('equal')
    # fig7.suptitle('t-control-des -fb', fontsize=20)
    ax.tick_params(labelsize=default_cfg["tick_size"])
    # fig8.tight_layout(pad=default_cfg["pad"])
    plt.savefig(os.path.join(save_dir, "torque_rr.{}".format(default_cfg["img_fmt"])))

    # t-des torque- toruqe, t-des delta- delta
    # fig9, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])
    fig9 = plt.figure('brake', figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])
    ax = fig9.add_axes(ax_para)  # [left, bottom, width, height]

    # line_car, = ax72.plot(state_pool[:, 0] * 0.02, state_pool[:, 14], color="lime",
    #                      linewidth=2)
    # line_ref, = ax72.plot(state_pool[:, 0] * 0.02, ref_pool[:, 9], color="magenta",
    #                       linewidth=2, linestyle='--')

    # line_ref, = ax72.plot(state_pool[0, :, 0] * 0.02, ref_pool[0, :, 9], c=color_list[0], linestyle=linestyle_list[0],
    #                       linewidth=2)  #
    line_CMMPC, = ax.plot(state_pool[0, :, 0] * 0.02, ref_pool[0, :, 9], c=color_list[1],
                            linestyle=linestyle_list[1],
                            linewidth=2)  #
    line_CRMPC, = ax.plot(state_pool[1, :, 0] * 0.02, ref_pool[1, :, 9], c=color_list[2],
                            linestyle=linestyle_list[2],
                            linewidth=2)  #
    line_CTMPC, = ax.plot(state_pool[2, :, 0] * 0.02, ref_pool[2, :, 9], c=color_list[3],
                            linestyle=linestyle_list[3],
                            linewidth=2)  #

    ax.set_ylabel(r"$P\ /\mathrm{MPa}$", default_cfg["label_font"])
    ax.set_xlabel(r"Time $/\mathrm{s}$", default_cfg["label_font"])
    # ax.set_facecolor('#E6E6E6')
    # ax.grid()
    # ax.set_aspect('equal')
    # fig7.suptitle('t-control-des -fb', fontsize=20)
    ax.tick_params(labelsize=default_cfg["tick_size"])
    ax.tick_params(axis='x', direction='in')  # x轴刻度线向内
    ax.tick_params(axis='y', direction='in')  # y轴刻度线向内
    # fig9.tight_layout(pad=default_cfg["pad"])
    plt.savefig(os.path.join(save_dir, "brake.{}".format(default_cfg["img_fmt"])))


    fig10 = plt.figure('delta_f', figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])
    ax = fig10.add_axes(ax_para)  # [left, bottom, width, height]

    # line_car, = ax.plot(state_pool[:, 0] * 0.02, state_pool[:, 13], color="lime",
    #                      linewidth=2)
    # line_ref, = ax.plot(state_pool[:, 0] * 0.02, ref_pool[:, 4], color="magenta",
    #                       linewidth=2, linestyle='--')

    # line_ref, = ax.plot(state_pool[0, :, 0] * 0.02, ref_pool[0, :, 4], c=color_list[0], linestyle=linestyle_list[0],
    #                       linewidth=2)  #
    line_CMMPC, = ax.plot(state_pool[0, :, 0] * 0.02, ref_pool[0, :, 4], c=color_list[1],
                            linestyle=linestyle_list[1],
                            linewidth=2)  #
    line_CRMPC, = ax.plot(state_pool[1, :, 0] * 0.02, ref_pool[1, :, 4], c=color_list[2],
                            linestyle=linestyle_list[2],
                            linewidth=2)  #
    line_CTMPC, = ax.plot(state_pool[2, :, 0] * 0.02, ref_pool[2, :, 4], c=color_list[3],
                            linestyle=linestyle_list[3],
                            linewidth=2)  #

    ax.set_ylabel(r"$\delta\ /\degree$", default_cfg["label_font"])
    ax.set_xlabel(r"Time $/\mathrm{s}$", default_cfg["label_font"])
    ax.tick_params(axis='x', direction='in')  # x轴刻度线向内
    ax.tick_params(axis='y', direction='in')  # y轴刻度线向内
    # ax.set_facecolor('#E6E6E6')
    # ax.grid()
    # ax.set_aspect('equal')
    # fig7.suptitle('t-control-des -fb', fontsize=20)
    ax.tick_params(labelsize=default_cfg["tick_size"])
    # fig10.tight_layout(pad=default_cfg["pad"])
    plt.savefig(os.path.join(save_dir, "delta_f.{}".format(default_cfg["img_fmt"])))

    # # t-y error
    # fig9, ax9 = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])
    # # line_yerr_pre, = ax9.plot(state_pool[:, 0] * 0.02, error_pool[:, 0], color="#8A2BE2",
    # #                       linewidth=2)
    # line_yerr_nt, = ax9.plot(state_pool[:, 0] * 0.02, error_pool[:, 2], color="magenta",
    #                       linewidth=2)
    # ax9.set_ylabel(r"$p_y^{\rm err}\ /\mathrm{m}$", default_cfg["label_font"])
    # ax9.set_xlabel(r"Time $/\mathrm{s}$", default_cfg["label_font"])
    # # plt.legend([line_yerr_pre, line_yerr_nt], ['pre', 'nearest'],
    # #
    # #            loc='best', prop=default_cfg["legend_font"], frameon=False)
    # # ax.set_facecolor('#E6E6E6')
    # # ax.grid()
    # # ax.set_aspect('equal')
    # # fig9.suptitle('t-control-des -fb', fontsize=20)
    # fig9.tight_layout(pad=default_cfg["pad"])
    # plt.tick_params(labelsize=default_cfg["tick_size"])
    # plt.savefig(os.path.join(save_dir, "t-y_err.{}".format(default_cfg["img_fmt"])))

    # x-z
    # fig10, ax10 = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])
    # line_z, = ax10.plot(state_pool[:, 1], ref_pool[:, 6], color="#8A2BE2",
    #                           linewidth=2)
    #
    # ax10.set_ylabel(r"Altitude$\ /\mathrm{m}$", default_cfg["label_font"])
    # ax10.set_xlabel(r"Pos $p_x /\mathrm{m}$", default_cfg["label_font"])
    # # ax.set_facecolor('#E6E6E6')
    # # ax.grid()
    # # ax.set_aspect('equal')
    # # fig10.suptitle('t-control-des -fb', fontsize=20)
    # plt.tick_params(labelsize=default_cfg["tick_size"])
    # fig10.tight_layout(pad=default_cfg["pad"])
    # plt.savefig(os.path.join(save_dir, "x-altitude.{}".format(default_cfg["img_fmt"])))
    # plt.show()

if __name__ == "__main__":
    # Parameters Setup
    parser = argparse.ArgumentParser()
    parser.add_argument("--time_step", type=float, default=0.02)
    parser.add_argument("--csv_file_name", type=str, default="undulating_traj_record-2025-09-06_1")#track_pre3m，20250321_suishi_mark
    parser.add_argument("--csv_file_name2", type=str, default="track_pre4m")
    parser.add_argument("--csv_file_name3", type=str, default="track_pre5m")
    parser.add_argument("--line_num", type=int, default=2)
    parser.add_argument("--language", type=str, default="en")
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
    parser.add_argument("--y_label", type=str, default=r"Torque $Q_{\rm rr}\ /\mathrm{N·m}$")  #
    # parser.add_argument("--y_label", type=str, default=r"Steering Angle $\delta\ /\mathrm{rad}$")  #
    # parser.add_argument("--y_label", type=str, default=r"Computation time $/\mathrm{ms}$")  #
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
    # parser.add_argument("--y_label", type=str, default=r"$\delta\ /\degree$")  #
    # parser.add_argument("--y_label", type=str, default=r"$a_{x,\rm tt}\ /\mathrm{(m·s^{-2})}$")  #
    # parser.add_argument("--y_label", type=str, default=r"单步计算时间 $/\mathrm{ms}$")  #

    # parser.add_argument("--y_label", type=str, default=r"$J_\mathrm{L}$")
    # parser.add_argument("--x_label", type=str, default=r"$M$")

    parser.add_argument("--legend_list", type=list, default=
    ["Ref", "CTMPC"])#"参考状态","Ref",
    parser.add_argument("--color_list", type=list, default=
    ["b", "lime", "#FA8072", "magenta"]) #"#FA8072"
    parser.add_argument("--linestyle_list", type=list, default=['-' ,'-.', ':', '-'])#
    #D:\1_Troy.Z\4_博士培养\4_论文写作与评审\2_论文写作\
    # parser.add_argument("--figures_root", type=str,
    #                     default='D:/1_Troy.Z/4_博士培养/4_论文写作与评审/2_论文写作/20_分布式模块化独立转向驱动稳定性控制/实车试验/0821dataRecord/qifu/')
    # parser.add_argument("--figures_root", type=str,default='D:/1_Troy.Z/4_博士培养/4_论文写作与评审/2_论文写作/20_分布式模块化独立转向驱动稳定性控制/实车试验/0321/')
    parser.add_argument("--figures_root", type=str,default='D:/1_Troy.Z/4_博士培养/4_论文写作与评审/2_论文写作/25_大论文/第三章/MixModeling/Examples/resource/')

    # Get parameter dictionary
    args = vars(parser.parse_args())

    # read_path_datax = args["figures_root"] + "State-1.csv"
    # read_path_datax2 = args["figures_root2"] + "State-1.csv"
    read_path = args["figures_root"]+args["csv_file_name"]+".csv"
    read_path2 = args["figures_root"] + args["csv_file_name2"] + ".csv"
    read_path3 = args["figures_root"] + args["csv_file_name3"] + ".csv"

    # read_path2 = args["figures_root"] + args["csv_file_name2"] + ".csv"
    # glo = read_globaltraj(read_path2)
    # state_pool, ref_pool, error_pool = read_data(read_path)
    state_pool, ref_pool, error_pool = read_data_traj_record(read_path)
    plot_experiment(state_pool, ref_pool, error_pool)
    # state_pool, ref_pool, error_pool = read_data_multi_method(read_path, read_path2, read_path3)
    # plot_experiment_multi_method(state_pool, ref_pool, error_pool)


