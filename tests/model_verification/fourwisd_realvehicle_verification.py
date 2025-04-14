import math

from gops.create_pkg.create_env import create_env
from gops.create_pkg.create_env_model import create_env_model
import gym
gym.logger.set_level(40)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.font_manager as fm
zhfont1 = fm.FontProperties(fname='../SIMSUN.ttf', size=14)
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
np.seterr(divide='ignore', invalid='ignore')

def stiffness_fitting(x, a, b, c, d):
    return c * np.sin(b * np.arctan(a * x - d * (a * x - np.arctan(a * x))))

def read_path(root_path):
    data_result = pd.DataFrame(pd.read_csv(root_path, header=None))
    state_1 = np.array(data_result.iloc[:, 0], dtype='float32') #x
    # state_2 = np.array(data_result.iloc[:, 1], dtype='float32')  #y
    state_traj = np.zeros((len(state_1), 1))
    state_traj[:, 0] = state_1
    # state_t[:, 1] = state_2
    return state_traj

def read_csv(root_path):
    data_result = pd.DataFrame(pd.read_csv(root_path, header=None))
    start_index = 20
    end_index = 200
    interval = 5
    step_list = np.array(data_result.iloc[start_index:end_index:interval, 0], dtype='float32')
    Q_fl = np.array(data_result.iloc[start_index:end_index:interval, 3], dtype='float32')
    Q_fr = -np.array(data_result.iloc[start_index:end_index:interval, 4], dtype='float32')
    Q_rl = -np.array(data_result.iloc[start_index:end_index:interval, 5], dtype='float32')
    Q_rr = -np.array(data_result.iloc[start_index:end_index:interval, 6], dtype='float32')
    delta = np.array(data_result.iloc[start_index:end_index:interval, 27], dtype='float32')/180*math.pi/ 17 # rad
    pos_x = np.array(data_result.iloc[start_index:end_index:interval, 17], dtype='float32')
    pos_y = np.array(data_result.iloc[start_index:end_index:interval, 18], dtype='float32')
    phi = np.array(data_result.iloc[start_index:end_index:interval, 7], dtype='float32')
    vx = np.array(data_result.iloc[start_index:end_index:interval, 8], dtype='float32')
    vy = np.array(data_result.iloc[start_index:end_index:interval, 33], dtype='float32')
    phidot = np.array(data_result.iloc[start_index:end_index:interval, 11], dtype='float32')/180*math.pi
    varphhi = np.array(data_result.iloc[start_index:end_index:interval, 26], dtype='float32')
    varphhi_dot = np.array(data_result.iloc[start_index:end_index:interval, 10], dtype='float32')/180*math.pi

    data_pool = np.zeros((len(step_list), 14))
    data_pool[:, 0] = step_list
    data_pool[:, 1] = pos_x
    data_pool[:, 2] = pos_y
    data_pool[:, 3] = phi
    data_pool[:, 4] = vx
    data_pool[:, 5] = vy
    data_pool[:, 6] = phidot
    data_pool[:, 7] = varphhi
    data_pool[:, 8] = varphhi_dot

    data_pool[:, 9] = Q_fl
    data_pool[:, 10] = Q_fr
    data_pool[:, 11] = Q_rl
    data_pool[:, 12] = Q_rr
    data_pool[:, 13] = delta
    return data_pool

def unit_transform_4wisd(state):
    state[0] = state[0]  # x
    state[1] = state[1]  # y
    state[2] = state[2] / 180 * np.pi  # yaw

    state[3] = state[3] / 3.6  # vx
    state[4] = state[4] / 3.6  # vy
    state[5] = state[5] / 180 * np.pi  # yaw_rate
    state[6] = state[6] / 180 * np.pi  # roll angle
    state[7] = state[7] / 180 * np.pi  # roll rate

    state[8] = state[8]  # kappa_1
    state[9] = state[9]  # kappa_2 rpm to rad/s
    state[10] = state[10] # kappa_3
    state[11] = state[11] # kappa_4
    # control feedback
    state[12] = state[12]  # drive torque on wheel 1
    state[13] = state[13]
    state[14] = state[14]
    state[15] = state[15]
    state[16] = state[16] / 180 * np.pi  # steering angle on wheel 1
    state[17] = state[17] / 180 * np.pi  # steering angle on wheel 2
    state[18] = state[18] / 180 * np.pi  # steering angle on wheel 3
    state[19] = state[19] / 180 * np.pi  # steering angle on wheel 4
    state[20] = state[20] / 180 * np.pi # beta
    state[21] = state[21] * 9.8 # acceleration
    state[30] = state[30] / 3.6  # vw L1
    state[31] = -state[31] / 180 * np.pi  # longitudinal slope of road
    state[32] = state[32] / 180 * np.pi  # lateral slope of road
    state[33] = -state[33] / 180 * np.pi  # longitudinal slope of road
    state[34] = state[34] / 180 * np.pi  # lateral slope of road
    state[35] = -state[35]/180*np.pi   # longitudinal slope of road
    state[36] = state[36]/180*np.pi   # lateral slope of road
    return state

def model_verification_4wisd(real_data,env_id):
    run_step = len(real_data[:, 0])
    delta_t = 0.02
    # state
    state_python = real_data[1, 1:9]
    model_self = create_env(env_id)
    
    step_sim = 0
    vx_self = []
    vx_car = []
    vy_self = []
    vy_car = []
    yawrate_self = []
    yawrate_car = []
    roll_self = []
    roll_car = []
    rollrate_self = []
    rollrate_car = []
    x_self = []
    x_car = []
    y_self = []
    y_car = []
    yaw_self = []
    yaw_car = []

    Qw1_self = []
    Qw1_car = []
    delta_w1_self = []
    delta_w1_car = []
    Qw2_self = []
    Qw2_car = []

    Qw3_self = []
    Qw3_car = []

    Qw4_self = []
    Qw4_car = []
    ax_car = []

    longi_slope = []
    lateral_slope = []

    for i in range(1, run_step):
        road_info = np.array([0, 0])
        steering_angle_rad = real_data[i, 13]
        control = np.array([real_data[i, 9], real_data[i, 10],
                            real_data[i, 11], real_data[i, 12],
                            steering_angle_rad])
        print("road_info", road_info)

        state_python = model_self.vehicle_dynamics.f_xu(state_python, control, delta_t, road_info)
        # state_python = model_self.vehicle_dynamics.f_xu(state_python, control, delta_t)

        # 4dof
        x_self.append(state_python[0])
        y_self.append(state_python[1])
        yaw_self.append(state_python[2])
        vx_self.append(state_python[3])
        vy_self.append(state_python[4])
        yawrate_self.append(state_python[5])
        roll_self.append(state_python[6])
        rollrate_self.append(state_python[7])

        Qw1_self.append(control[0])
        Qw3_self.append(control[2])
        Qw4_self.append(control[3])
        delta_w1_self.append(control[4])
        # -------------------------------
        x_car.append(real_data[i, 1])
        y_car.append(real_data[i, 2])
        yaw_car.append(real_data[i, 3])
        vx_car.append(real_data[i, 4])
        vy_car.append(real_data[i, 5])
        yawrate_car.append(real_data[i, 6])
        roll_car.append(real_data[i, 7])
        rollrate_car.append(real_data[i, 8])

        Qw1_car.append(real_data[i, 9])
        Qw3_car.append(real_data[i, 11])
        Qw4_car.append(real_data[i, 12])
        delta_w1_car.append(real_data[i, 13])

        longi_slope.append(road_info[0])
        lateral_slope.append(road_info[1])
        step_sim += 1
    print("run finished")
    data_result = pd.DataFrame(
        {'yaw_self': yaw_self, 'yaw_car': yaw_car,
         'yawrate_self': yawrate_self, 'yawrate_car': yawrate_car,
         'x_self': x_self, 'x_car': x_car,
         'Qw1_self': Qw1_self, 'Qw1_car': Qw1_car,
         'delta_w1_self': delta_w1_self, 'delta_w1_car': delta_w1_car,
         'roll_self': roll_self, 'roll_car': roll_car,
         'rollrate_self': rollrate_self, 'rollrate_car': rollrate_car,
         'y_self': y_self, 'y_car': y_car,
         'vx_self': vx_self, 'vx_car': vx_car,
         'vy_self': vy_self, 'vy_car': vy_car,
         'Qw3_self': Qw3_self, 'Qw3_car': Qw3_car,
         'longi_slope': longi_slope, 'lateral_slope': lateral_slope})
    picture_dir = "plot_4wisd_real/"
    os.makedirs(picture_dir, exist_ok=True)
    data_result.to_csv('./plot_4wisd_real/result_4wisd_real.csv', encoding='gbk')
    # '--------------------出图-----------------------'
    picture_dir = "plot_4wisd_real/"
    os.makedirs(picture_dir, exist_ok=True)
    # f9 = plt.figure("-kappa1", figsize=(8, 5))
    # ax = f9.add_axes([0.1, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    # l1, = plt.plot(np.arange(0, run_step, 1) * delta_t, kappa_1_self, lw=2, color="darkviolet")
    # l2, = plt.plot(np.arange(0, run_step, 1) * delta_t, kappa_1_car, lw=2, linestyle='--', color="deepskyblue")
    # plt.legend(handles=[l1, l2], labels=['4wisd', 'car'], prop={'size': 10}, loc=2,
    #            ncol=2)
    # plt.ylabel(r"$\kappa_1$ [-]", fontsize=14)
    # plt.xlabel("Times [s]", fontsize=14)
    # plt.tick_params(labelsize=12)
    # plt.subplots_adjust(bottom=0.31)
    # plt.grid(axis='both', ls='-.')
    # plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    # plt.savefig(os.path.join(picture_dir, "-kappa1.png"))

    # f21 = plt.figure("-accel", figsize=(8, 5))
    # ax = f21.add_axes([0.1, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    # l1, = plt.plot(np.arange(0, run_step, 1) * delta_t, ax_self, lw=2, color="darkviolet")
    # l2, = plt.plot(np.arange(0, run_step, 1) * delta_t, ax_car, lw=2, linestyle='--', color="deepskyblue")
    # plt.legend(handles=[l1, l2], labels=['4wisd', 'car'], prop={'size': 10}, loc=2,
    #            ncol=2)
    # plt.ylabel(r"$a_x$ [m/s^2]", fontsize=14)
    # plt.xlabel("Times [s]", fontsize=14)
    # plt.tick_params(labelsize=12)
    # plt.subplots_adjust(bottom=0.31)
    # plt.grid(axis='both', ls='-.')
    # plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    # plt.savefig(os.path.join(picture_dir, "-ax.png"))

    # f12 = plt.figure("-kappa4", figsize=(8, 5))
    # ax = f12.add_axes([0.1, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    # l1, = plt.plot(np.arange(0, run_step, 1) * delta_t, kappa_4_self, lw=2, color="darkviolet")
    # l2, = plt.plot(np.arange(0, run_step, 1) * delta_t, kappa_4_car, lw=2, linestyle='--', color="deepskyblue")
    # plt.legend(handles=[l1, l2], labels=['4wisd', 'car'], prop={'size': 10}, loc=2,
    #            ncol=2)
    # plt.ylabel(r"$\kappa_4$ [-]", fontsize=14)
    # plt.xlabel("Times [s]", fontsize=14)
    # plt.tick_params(labelsize=12)
    # plt.subplots_adjust(bottom=0.31)
    # plt.grid(axis='both', ls='-.')
    # plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    # plt.savefig(os.path.join(picture_dir, "-kappa4.png"))
    #
    # f12 = plt.figure("-kappa3", figsize=(8, 5))
    # ax = f12.add_axes([0.1, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    # l1, = plt.plot(np.arange(0, run_step, 1) * delta_t, kappa_3_self, lw=2, color="darkviolet")
    # l2, = plt.plot(np.arange(0, run_step, 1) * delta_t, kappa_3_car, lw=2, linestyle='--', color="deepskyblue")
    # plt.legend(handles=[l1, l2], labels=['4wisd', 'car'], prop={'size': 10}, loc=2,
    #            ncol=2)
    # plt.ylabel(r"$\kappa_3$ [-]", fontsize=14)
    # plt.xlabel("Times [s]", fontsize=14)
    # plt.tick_params(labelsize=12)
    # plt.subplots_adjust(bottom=0.31)
    # plt.grid(axis='both', ls='-.')
    # plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    # plt.savefig(os.path.join(picture_dir, "-kappa3.png"))
    #
    # f12 = plt.figure("-kappa2", figsize=(8, 5))
    # ax = f12.add_axes([0.1, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    # l1, = plt.plot(np.arange(0, run_step, 1) * delta_t, kappa_2_self, lw=2, color="darkviolet")
    # l2, = plt.plot(np.arange(0, run_step, 1) * delta_t, kappa_2_car, lw=2, linestyle='--', color="deepskyblue")
    # plt.legend(handles=[l1, l2], labels=['4wisd', 'car'], prop={'size': 10}, loc=2,
    #            ncol=2)
    # plt.ylabel(r"$\kappa_2$ [-]", fontsize=14)
    # plt.xlabel("Times [s]", fontsize=14)
    # plt.tick_params(labelsize=12)
    # plt.subplots_adjust(bottom=0.31)
    # plt.grid(axis='both', ls='-.')
    # plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    # plt.savefig(os.path.join(picture_dir, "-kappa2.png"))

    f0 = plt.figure("-vx", figsize=(8, 5))
    ax = f0.add_axes([0.1, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(1, run_step, 1)*delta_t, vx_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(1, run_step, 1)*delta_t, vx_car, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['4wisd', 'car'], prop={'size': 10}, loc=2,
               ncol=2)
    plt.ylabel(r"$v_x$ [m/s]", fontsize=14)
    plt.xlabel("Times [s]", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-vx.png"))
    
    f13 = plt.figure("-vy", figsize=(8, 5))
    ax = f13.add_axes([0.1, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(1, run_step, 1)*delta_t, vy_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(1, run_step, 1)*delta_t, vy_car, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['4wisd', 'car'], prop={'size': 10}, loc=2,
               ncol=2)
    plt.ylabel(r"$v_y$ [m/s]", fontsize=14)
    plt.xlabel("Times [s]", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-vy.png"))

    f2 = plt.figure("-yaw rate", figsize=(8, 5))
    ax = f2.add_axes([0.125, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(1, run_step, 1)*delta_t, yawrate_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(1, run_step, 1)*delta_t, yawrate_car, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['4wisd', 'car'], prop={'size': 10}, loc=2,
               ncol=2)
    plt.ylabel("yaw rate [rad/s]", fontsize=14)
    plt.xlabel("Times [s]", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-yaw rate.png"))

    f6 = plt.figure("-roll", figsize=(8, 5))
    ax = f6.add_axes([0.125, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(1, run_step, 1)*delta_t, roll_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(1, run_step, 1)*delta_t, roll_car, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['4wisd', 'car'], prop={'size': 10}, loc=2,
               ncol=2)
    plt.ylabel("roll [rad]", fontsize=14)
    plt.xlabel("Times [s]", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-roll.png"))

    f7 = plt.figure("-roll rate", figsize=(8, 5))
    ax = f7.add_axes([0.125, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(1, run_step, 1)*delta_t, rollrate_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(1, run_step, 1)*delta_t, rollrate_car, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['4wisd', 'car'], prop={'size': 10}, loc=2,
               ncol=2)
    plt.ylabel("roll rate [rad/s]", fontsize=14)
    plt.xlabel("Times [s]", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-roll rate.png"))

    f3 = plt.figure("-x", figsize=(8, 5))
    ax = f3.add_axes([0.09, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(1, run_step, 1)*delta_t, x_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(1, run_step, 1)*delta_t, x_car, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['4wisd', 'car'], prop={'size': 10}, loc=2,
               ncol=2)
    plt.ylabel("x [m]", fontsize=14)
    plt.xlabel("Times [s]", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-x.png"))

    f11 = plt.figure("-y", figsize=(8, 5))
    ax = f11.add_axes([0.1, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(1, run_step, 1)*delta_t, y_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(1, run_step, 1)*delta_t, y_car, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['4wisd', 'car'], prop={'size': 10}, loc=2,
               ncol=2)
    plt.ylabel("y [m]", fontsize=14)
    plt.xlabel("Times [s]", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-y.png"))

    f1 = plt.figure("-yaw", figsize=(8, 5))
    ax = f1.add_axes([0.125, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(1, run_step, 1)*delta_t, yaw_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(1, run_step, 1)*delta_t, yaw_car, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['4wisd', 'car'], prop={'size': 10}, loc=2, ncol=2)
    plt.ylabel("yaw [rad]", fontsize=14)
    plt.xlabel("Times [s]", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-yaw.png"))

    f4 = plt.figure("-Qw1", figsize=(8, 5))
    ax = f4.add_axes([0.1, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(1, run_step, 1)*delta_t, Qw1_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(1, run_step, 1)*delta_t, Qw1_car, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['4wisd', 'car'], prop={'size': 10}, loc=2,
               ncol=2)
    plt.ylabel(r"$Q_{w1}$ [rad]", fontsize=14)
    plt.xlabel("Times [s]", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-Qw1.png"))

    f5 = plt.figure("-delta_w1", figsize=(8, 5))
    ax = f5.add_axes([0.1, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(1, run_step, 1)*delta_t, delta_w1_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(1, run_step, 1)*delta_t, delta_w1_car, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['4wisd', 'car'], prop={'size': 10}, loc=2,
               ncol=2)
    plt.ylabel(r"$\delta_{w1}$ [rad]", fontsize=14)
    plt.xlabel("Times [s]", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-delta_w1.png"))

    # f8 = plt.figure("-delta_w2", figsize=(8, 5))
    # ax = f8.add_axes([0.1, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    # l1, = plt.plot(np.arange(0, run_step, 1)*delta_t, delta_w2_self, lw=2, color="darkviolet")
    # l2, = plt.plot(np.arange(0, run_step, 1)*delta_t, delta_w2_car, lw=2, linestyle='--', color="deepskyblue")
    # plt.legend(handles=[l1, l2], labels=['4wisd', 'car'], prop={'size': 10}, loc=2,
    #            ncol=2)
    # plt.ylabel(r"$\delta_{w2}$ [rad]", fontsize=14)
    # plt.xlabel("Times [s]", fontsize=14)
    # plt.tick_params(labelsize=12)
    # plt.subplots_adjust(bottom=0.31)
    # plt.grid(axis='both', ls='-.')
    # plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    # plt.savefig(os.path.join(picture_dir, "-delta_w2.png"))

    # f9 = plt.figure("-delta_w3", figsize=(8, 5))
    # ax = f9.add_axes([0.1, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    # l1, = plt.plot(np.arange(0, run_step, 1)*delta_t, delta_w3_self, lw=2, color="darkviolet")
    # l2, = plt.plot(np.arange(0, run_step, 1)*delta_t, delta_w3_car, lw=2, linestyle='--', color="deepskyblue")
    # plt.legend(handles=[l1, l2], labels=['4wisd', 'car'], prop={'size': 10}, loc=2,
    #            ncol=2)
    # plt.ylabel(r"$\delta_{w3}$ [rad]", fontsize=14)
    # plt.xlabel("Times [s]", fontsize=14)
    # plt.tick_params(labelsize=12)
    # plt.subplots_adjust(bottom=0.31)
    # plt.grid(axis='both', ls='-.')
    # plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    # plt.savefig(os.path.join(picture_dir, "-delta_w3.png"))

    # f10 = plt.figure("-delta_w4", figsize=(8, 5))
    # ax = f10.add_axes([0.1, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    # l1, = plt.plot(np.arange(0, run_step, 1)*delta_t, delta_w4_self, lw=2, color="darkviolet")
    # l2, = plt.plot(np.arange(0, run_step, 1)*delta_t, delta_w4_car, lw=2, linestyle='--', color="deepskyblue")
    # plt.legend(handles=[l1, l2], labels=['4wisd', 'car'], prop={'size': 10}, loc=2,
    #            ncol=2)
    # plt.ylabel(r"$\delta_{w4}$ [rad]", fontsize=14)
    # plt.xlabel("Times [s]", fontsize=14)
    # plt.tick_params(labelsize=12)
    # plt.subplots_adjust(bottom=0.31)
    # plt.grid(axis='both', ls='-.')
    # plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    # plt.savefig(os.path.join(picture_dir, "-delta_w4.png"))

    f14 = plt.figure("-Qw3", figsize=(8, 5))
    ax = f14.add_axes([0.1, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(1, run_step, 1)*delta_t, Qw3_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(1, run_step, 1)*delta_t, Qw3_car, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['4wisd', 'car'], prop={'size': 10}, loc=2,
               ncol=2)
    plt.ylabel(r"$Q_{w3}$ [Nm]", fontsize=14)
    plt.xlabel("Times [s]", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-Qw3.png"))

    f1 = plt.figure("-Qw4", figsize=(8, 5))
    ax = f1.add_axes([0.1, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(1, run_step, 1)*delta_t, Qw4_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(1, run_step, 1)*delta_t, Qw4_car, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['4wisd', 'car'], prop={'size': 10}, loc=2, ncol=2)
    plt.ylabel(r"$Q_{w4}$ [Nm]", fontsize=14)
    plt.xlabel("Times [s]", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-Qw4.png"))

    f33 = plt.figure("-road info", figsize=(8, 5))
    ax = f1.add_axes([0.1, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(1, run_step, 1) * delta_t, longi_slope, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(1, run_step, 1) * delta_t, lateral_slope, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['pitch road', 'roll road'], prop={'size': 10}, loc=2, ncol=2)
    plt.ylabel(r"angle [rad]", fontsize=14)
    plt.xlabel("Times [s]", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-road info.png"))

    # plt.show()

if __name__ == '__main__':
    root_path = "D:/1_Troy.Z/4_博士培养/4_论文写作与评审/2_论文写作/20_分布式模块化独立转向驱动稳定性控制/实车试验/0321/20250315_DLC.csv"
    real_data = read_csv(root_path)
    model_verification_4wisd(real_data, env_id='pyth_stabilitycontrol_real')

    # x = re[:, 0]
    # y = re[:, 1]
    # popt1, pcov1 = curve_fit(stiffness_fitting, x, y,maxfev = 10000)
    # print(popt1)


