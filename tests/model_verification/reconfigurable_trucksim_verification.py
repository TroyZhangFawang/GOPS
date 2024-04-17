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

def unit_transform_4A(state):
    state[0] = state[0] / 3.6  # vx
    state[1] = state[1] / 3.6  # vy
    state[2] = state[2] / 180 * np.pi  # yaw_rate
    state[3] = state[3] / 180 * np.pi  # roll angle
    state[4] = state[4] / 180 * np.pi  # roll rate
    state[5] = state[5]  # x
    state[6] = state[6]  # y
    state[7] = state[7] / 180 * np.pi  # yaw
    # control feedback
    state[8] = state[8]  # drive torque on wheel 1
    state[9] = state[9] / 180 * np.pi  # steering angle on wheel 1
    state[10] = state[10]
    state[11] = state[11] / 180 * np.pi  # steering angle on wheel 2
    state[12] = state[12]
    state[13] = state[13] / 180 * np.pi  # steering angle on wheel 3
    state[14] = state[14]
    state[15] = state[15] / 180 * np.pi  # steering angle on wheel 4
    state[16] = state[16]
    state[17] = state[17] / 180 * np.pi  # steering angle on wheel 5
    state[18] = state[18]
    state[19] = state[19] / 180 * np.pi  # steering angle on wheel 6
    state[20] = state[20]
    state[21] = state[21] / 180 * np.pi  # steering angle on wheel 7
    state[22] = state[22]
    state[23] = state[23] / 180 * np.pi  # steering angle on wheel 8
    return state

def unit_transform_2A(state):
    state[0] = state[0] / 3.6  # vx
    state[1] = state[1] / 3.6  # vy
    state[2] = state[2] / 180 * np.pi  # yaw_rate
    state[3] = state[3] / 180 * np.pi  # roll angle
    state[4] = state[4] / 180 * np.pi  # roll rate
    state[5] = state[5]  # x
    state[6] = state[6]  # y
    state[7] = state[7] / 180 * np.pi  # yaw
    # control feedback
    state[8] = state[8]  # drive torque on wheel 1
    state[9] = state[9] / 180 * np.pi  # steering angle on wheel 1
    state[10] = state[10]
    state[11] = state[11] / 180 * np.pi  # steering angle on wheel 2
    state[12] = state[12]
    state[13] = state[13] / 180 * np.pi  # steering angle on wheel 3
    state[14] = state[14]
    state[15] = state[15] / 180 * np.pi  # steering angle on wheel 4
    state[16] = state[16]
    return state

def model_compare_4A(env_id):
    run_step = 8000
    delta_t = 0.0005
    model_mechnical = gym.make(env_id, disable_env_checker=True)
    state = model_mechnical.reset()

    # state 
    state = unit_transform_4A(state)
    model_self = create_env(env_id)
    print(state[5], state[26:])
    # 4dof
    state_python = state[:8]

    step_sim = 0
    vx_self = []
    vx_trucksim = []
    vy_self = []
    vy_trucksim = []
    yawrate_self = []
    yawrate_trucksim = []
    roll_self = []
    roll_trucksim = []
    rollrate_self = []
    rollrate_trucksim = []
    x_self = []
    x_trucksim = []
    y_self = []
    y_trucksim = []
    yaw_self = []
    yaw_trucksim = []
        
    Qw1_self = []
    Qw1_trucksim = []
    delta_w1_self = []
    delta_w1_trucksim = []
    Qw2_self = []
    Qw2_trucksim = []
    delta_w2_self = []
    delta_w2_trucksim = []
    Qw3_self = []
    Qw3_trucksim = []
    delta_w3_self = []
    delta_w3_trucksim = []
    Qw4_self = []
    Qw4_trucksim = []
    delta_w4_self = []
    delta_w4_trucksim = []
    
    Qw5_self = []
    Qw5_trucksim = []

    delta_w5_self = []
    delta_w5_trucksim = []

    
    for i in range(run_step):
        drive_torque = -60
        # if i< 500 :
        #     steering_angle_degree = 0
        # elif i>500and i<1000 :
        #     steering_angle_degree = 3
        # elif i>1000and i<1500 :
        #     steering_angle_degree = -3
        # else:
        #     steering_angle_degree = -0
        steering_angle_degree = 3 * np.sin(np.pi * 2 / 20000 * i)
        steering_angle_rad = steering_angle_degree/180*3.14
        control_trucksim = np.array([drive_torque, steering_angle_degree,
                            drive_torque, steering_angle_degree,
                            drive_torque, steering_angle_degree,
                            drive_torque, steering_angle_degree,
                            drive_torque, 0,
                            drive_torque, 0,
                            drive_torque, 0,
                            drive_torque, 0])
        control = np.array([drive_torque, steering_angle_rad,
                            drive_torque, steering_angle_rad,
                            drive_torque, steering_angle_rad,
                            drive_torque, steering_angle_rad,
                            drive_torque, 0,
                            drive_torque, 0,
                            drive_torque, 0,
                            drive_torque, 0])

        state_python = model_self.vehicle_dynamics.f_xu(state_python, control, delta_t)

        state, _ = model_mechnical.step(control_trucksim)  # trucksim
        state = unit_transform_4A(state)

        # 4dof
        vx_self.append(state_python[0])
        vy_self.append(state_python[1])
        yawrate_self.append(state_python[2])
        roll_self.append(state_python[3])
        rollrate_self.append(state_python[4])
        x_self.append(state_python[5])
        y_self.append(state_python[6])
        yaw_self.append(state_python[7])

        Qw1_self.append(control[0])
        delta_w1_self.append(control[1])

        delta_w2_self.append(control[3])

        delta_w3_self.append(control[5])

        delta_w4_self.append(control[7])
        Qw5_self.append(control[8])
        delta_w5_self.append(control[9])

        #-------------------------------
        vx_trucksim.append(state[0])
        vy_trucksim.append(state[1])
        yawrate_trucksim.append(state[2])
        roll_trucksim.append(state[3])
        rollrate_trucksim.append(state[4])
        x_trucksim.append(state[5])
        y_trucksim.append(state[6])
        yaw_trucksim.append(state[7])
        Qw1_trucksim.append(state[8])
        delta_w1_trucksim.append(state[9])

        delta_w2_trucksim.append(state[11])
        delta_w3_trucksim.append(state[13])
        delta_w4_trucksim.append(state[15])
        Qw5_trucksim.append(state[16])
        delta_w5_trucksim.append(state[17])

        step_sim += 1
    print("run finished")
    data_result = pd.DataFrame(
        {'yaw_self': yaw_self, 'yaw_trucksim': yaw_trucksim,
         'yawrate_self': yawrate_self, 'yawrate_trucksim': yawrate_trucksim,
         'x_self': x_self, 'x_trucksim': x_trucksim,
         'Qw1_self': Qw1_self, 'Qw1_trucksim': Qw1_trucksim,
         'delta_w1_self': delta_w1_self, 'delta_w1_trucksim': delta_w1_trucksim,
         'roll_self': roll_self, 'roll_trucksim': roll_trucksim,
         'rollrate_self': rollrate_self, 'rollrate_trucksim': rollrate_trucksim,
         'delta_w2_self': delta_w2_self, 'delta_w2_trucksim': delta_w2_trucksim,
         'delta_w3_self': delta_w3_self, 'delta_w3_trucksim': delta_w3_trucksim,
         'delta_w4_self': delta_w4_self, 'delta_w4_trucksim': delta_w4_trucksim,
         'y_self': y_self, 'y_trucksim': y_trucksim,
         'vx_self': vx_self, 'vx_trucksim': vx_trucksim,
         'vy_self': vy_self, 'vy_trucksim': vy_trucksim,
         'Qw5_self': Qw5_self, 'Qw5_trucksim': Qw5_trucksim,})
    data_result.to_csv('./result_4A.csv', encoding='gbk')

    # '--------------------出图-----------------------'
    picture_dir ="./plot_4A/"
    os.makedirs(picture_dir, exist_ok=True)
    f12 = plt.figure("-vx", figsize=(8, 5))
    ax = f12.add_axes([0.1, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(0, run_step, 1)*delta_t, vx_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1)*delta_t, vx_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['6dof', 'trucksim'], prop={'size': 10}, loc=2,
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
    l1, = plt.plot(np.arange(0, run_step, 1)*delta_t, vy_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1)*delta_t, vy_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['6dof', 'trucksim'], prop={'size': 10}, loc=2,
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
    l1, = plt.plot(np.arange(0, run_step, 1)*delta_t, yawrate_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1)*delta_t, yawrate_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['6dof', 'trucksim'], prop={'size': 10}, loc=2,
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
    l1, = plt.plot(np.arange(0, run_step, 1)*delta_t, roll_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1)*delta_t, roll_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['6dof', 'trucksim'], prop={'size': 10}, loc=2,
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
    l1, = plt.plot(np.arange(0, run_step, 1)*delta_t, rollrate_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1)*delta_t, rollrate_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['6dof', 'trucksim'], prop={'size': 10}, loc=2,
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
    l1, = plt.plot(np.arange(0, run_step, 1)*delta_t, x_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1)*delta_t, x_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['6dof', 'trucksim'], prop={'size': 10}, loc=2,
               ncol=2)
    plt.ylabel("x [m]", fontsize=14)
    plt.xlabel("Times [s]", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-x.png"))

    f11 = plt.figure("-y", figsize=(8, 5))
    ax = f11.add_axes([0.09, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(0, run_step, 1)*delta_t, y_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1)*delta_t, y_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['6dof', 'trucksim'], prop={'size': 10}, loc=2,
               ncol=2)
    plt.ylabel("y [m]", fontsize=14)
    plt.xlabel("Times [s]", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-y.png"))

    f1 = plt.figure("-yaw", figsize=(8, 5))
    ax = f1.add_axes([0.1, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(0, run_step, 1)*delta_t, yaw_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1)*delta_t, yaw_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['6dof','trucksim'], prop={'size': 10}, loc=2, ncol=2)
    plt.ylabel("yaw [rad]", fontsize=14)
    plt.xlabel("Times [s]", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-yaw.png"))

    f4 = plt.figure("-Qw1", figsize=(8, 5))
    ax = f4.add_axes([0.1, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(0, run_step, 1)*delta_t, Qw1_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1)*delta_t, Qw1_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['6dof', 'trucksim'], prop={'size': 10}, loc=2,
               ncol=2)
    plt.ylabel(r"$Q_{w1}$ [Nm]", fontsize=14)
    plt.xlabel("Times [s]", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-Qw1.png"))

    f5 = plt.figure("-delta_w1", figsize=(8, 5))
    ax = f5.add_axes([0.1, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(0, run_step, 1)*delta_t, delta_w1_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1)*delta_t, delta_w1_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['6dof', 'trucksim'], prop={'size': 10}, loc=2,
               ncol=2)
    plt.ylabel(r"$\delta_{w1}$ [rad]", fontsize=14)
    plt.xlabel("Times [s]", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-delta_w1.png"))

    f8 = plt.figure("-delta_w2", figsize=(8, 5))
    ax = f8.add_axes([0.1, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(0, run_step, 1)*delta_t, delta_w2_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1)*delta_t, delta_w2_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['6dof', 'trucksim'], prop={'size': 10}, loc=2,
               ncol=2)
    plt.ylabel(r"$\delta_{w2}$ [rad]", fontsize=14)
    plt.xlabel("Times [s]", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-delta_w2.png"))

    f9= plt.figure("-delta_w3", figsize=(8, 5))
    ax = f9.add_axes([0.1, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(0, run_step, 1)*delta_t, delta_w3_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1)*delta_t, delta_w3_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['6dof', 'trucksim'], prop={'size': 10}, loc=2,
               ncol=2)
    plt.ylabel(r"$\delta_{w3}$ [rad]", fontsize=14)
    plt.xlabel("Times [s]", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-delta_w3.png"))

    f10 = plt.figure("-delta_w4", figsize=(8, 5))
    ax = f10.add_axes([0.1, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(0, run_step, 1)*delta_t, delta_w4_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1)*delta_t, delta_w4_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['6dof', 'trucksim'], prop={'size': 10}, loc=2,
               ncol=2)
    plt.ylabel(r"$\delta_{w4}$ [rad]", fontsize=14)
    plt.xlabel("Times [s]", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-delta_w4.png"))


    f14 = plt.figure("-Qw5", figsize=(8, 5))
    ax = f14.add_axes([0.1, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(0, run_step, 1)*delta_t, Qw5_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1)*delta_t, Qw5_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['6dof', 'trucksim'], prop={'size': 10}, loc=2,
               ncol=2)
    plt.ylabel(r"$Q_{w5}$ [Nm]", fontsize=14)
    plt.xlabel("Times [s]", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-Qw5.png"))

    f1 = plt.figure("-delta_w5", figsize=(8, 5))
    ax = f1.add_axes([0.1, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(0, run_step, 1)*delta_t, delta_w5_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1)*delta_t, delta_w5_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['6dof', 'trucksim'], prop={'size': 10}, loc=2, ncol=2)
    plt.ylabel(r"$\delta_{w5}$ [rad]", fontsize=14)
    plt.xlabel("Times [s]", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-delta_w5.png"))

    # plt.show()

def model_compare_2A(env_id):
    run_step = 8000
    delta_t = 0.0005
    model_mechnical = gym.make(env_id, disable_env_checker=True)
    state = model_mechnical.reset()

    # state
    state = unit_transform_2A(state)
    model_self = create_env(env_id)
    # 4dof
    state_python = state[:8]
    print(state[18:])
    step_sim = 0
    vx_self = []
    vx_trucksim = []
    vy_self = []
    vy_trucksim = []
    yawrate_self = []
    yawrate_trucksim = []
    roll_self = []
    roll_trucksim = []
    rollrate_self = []
    rollrate_trucksim = []
    x_self = []
    x_trucksim = []
    y_self = []
    y_trucksim = []
    yaw_self = []
    yaw_trucksim = []

    Qw1_self = []
    Qw1_trucksim = []
    delta_w1_self = []
    delta_w1_trucksim = []
    Qw2_self = []
    Qw2_trucksim = []
    delta_w2_self = []
    delta_w2_trucksim = []
    delta_w3_self = []
    delta_w3_trucksim = []
    delta_w4_self = []
    delta_w4_trucksim = []

    Qw3_self = []
    Qw3_trucksim = []

    Qw4_self = []
    Qw4_trucksim = []

    for i in range(run_step):
        drive_torque = -60
        # if i< 500 :
        #     steering_angle_degree = 0
        # elif i>500and i<1000 :
        #     steering_angle_degree = 3
        # elif i>1000and i<1500 :
        #     steering_angle_degree = -3
        # else:
        #     steering_angle_degree = 0
        steering_angle_degree = 3 * np.sin(np.pi * 2 / 20000 * i)
        steering_angle_rad = steering_angle_degree / 180 * 3.14
        control_trucksim = np.array([drive_torque, steering_angle_degree,
                                     drive_torque, steering_angle_degree,
                                     drive_torque, 0,
                                     drive_torque, 0])
        control = np.array([drive_torque, steering_angle_rad,
                            drive_torque, steering_angle_rad,
                            drive_torque, 0,
                            drive_torque, 0])

        state_python = model_self.vehicle_dynamics.f_xu(state_python, control, delta_t)

        state, _ = model_mechnical.step(control_trucksim)  # trucksim
        state = unit_transform_2A(state)

        # 4dof
        vx_self.append(state_python[0])
        vy_self.append(state_python[1])
        yawrate_self.append(state_python[2])
        roll_self.append(state_python[3])
        rollrate_self.append(state_python[4])
        x_self.append(state_python[5])
        y_self.append(state_python[6])
        yaw_self.append(state_python[7])

        Qw1_self.append(control[0])
        delta_w1_self.append(control[1])

        delta_w2_self.append(control[3])
        Qw3_self.append(control[4])

        delta_w3_self.append(control[5])
        Qw4_self.append(control[6])
        delta_w4_self.append(control[7])


        # -------------------------------
        vx_trucksim.append(state[0])
        vy_trucksim.append(state[1])
        yawrate_trucksim.append(state[2])
        roll_trucksim.append(state[3])
        rollrate_trucksim.append(state[4])
        x_trucksim.append(state[5])
        y_trucksim.append(state[6])
        yaw_trucksim.append(state[7])
        Qw1_trucksim.append(state[8])
        delta_w1_trucksim.append(state[9])

        delta_w2_trucksim.append(state[11])
        Qw3_trucksim.append(state[12])
        delta_w3_trucksim.append(state[13])
        Qw4_trucksim.append(state[14])
        delta_w4_trucksim.append(state[15])
        step_sim += 1
    print("run finished")
    data_result = pd.DataFrame(
        {'yaw_self': yaw_self, 'yaw_trucksim': yaw_trucksim,
         'yawrate_self': yawrate_self, 'yawrate_trucksim': yawrate_trucksim,
         'x_self': x_self, 'x_trucksim': x_trucksim,
         'Qw1_self': Qw1_self, 'Qw1_trucksim': Qw1_trucksim,
         'delta_w1_self': delta_w1_self, 'delta_w1_trucksim': delta_w1_trucksim,
         'roll_self': roll_self, 'roll_trucksim': roll_trucksim,
         'rollrate_self': rollrate_self, 'rollrate_trucksim': rollrate_trucksim,
         'delta_w2_self': delta_w2_self, 'delta_w2_trucksim': delta_w2_trucksim,
         'delta_w3_self': delta_w3_self, 'delta_w3_trucksim': delta_w3_trucksim,
         'delta_w4_self': delta_w4_self, 'delta_w4_trucksim': delta_w4_trucksim,
         'y_self': y_self, 'y_trucksim': y_trucksim,
         'vx_self': vx_self, 'vx_trucksim': vx_trucksim,
         'vy_self': vy_self, 'vy_trucksim': vy_trucksim,
         'Qw3_self': Qw3_self, 'Qw3_trucksim': Qw3_trucksim, })
    data_result.to_csv('./result_2A.csv', encoding='gbk')

    # '--------------------出图-----------------------'
    picture_dir = "./plot_2A/"
    os.makedirs(picture_dir, exist_ok=True)
    f12 = plt.figure("-vx", figsize=(8, 5))
    ax = f12.add_axes([0.1, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(0, run_step, 1)*delta_t, vx_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1)*delta_t, vx_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['6dof', 'trucksim'], prop={'size': 10}, loc=2,
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
    l1, = plt.plot(np.arange(0, run_step, 1)*delta_t, vy_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1)*delta_t, vy_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['6dof', 'trucksim'], prop={'size': 10}, loc=2,
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
    l1, = plt.plot(np.arange(0, run_step, 1)*delta_t, yawrate_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1)*delta_t, yawrate_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['6dof', 'trucksim'], prop={'size': 10}, loc=2,
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
    l1, = plt.plot(np.arange(0, run_step, 1)*delta_t, roll_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1)*delta_t, roll_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['6dof', 'trucksim'], prop={'size': 10}, loc=2,
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
    l1, = plt.plot(np.arange(0, run_step, 1)*delta_t, rollrate_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1)*delta_t, rollrate_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['6dof', 'trucksim'], prop={'size': 10}, loc=2,
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
    l1, = plt.plot(np.arange(0, run_step, 1)*delta_t, x_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1)*delta_t, x_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['6dof', 'trucksim'], prop={'size': 10}, loc=2,
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
    l1, = plt.plot(np.arange(0, run_step, 1)*delta_t, y_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1)*delta_t, y_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['6dof', 'trucksim'], prop={'size': 10}, loc=2,
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
    l1, = plt.plot(np.arange(0, run_step, 1)*delta_t, yaw_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1)*delta_t, yaw_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['6dof', 'trucksim'], prop={'size': 10}, loc=2, ncol=2)
    plt.ylabel("yaw [rad]", fontsize=14)
    plt.xlabel("Times [s]", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-yaw.png"))

    f4 = plt.figure("-Qw1", figsize=(8, 5))
    ax = f4.add_axes([0.1, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(0, run_step, 1)*delta_t, Qw1_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1)*delta_t, Qw1_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['6dof', 'trucksim'], prop={'size': 10}, loc=2,
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
    l1, = plt.plot(np.arange(0, run_step, 1)*delta_t, delta_w1_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1)*delta_t, delta_w1_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['6dof', 'trucksim'], prop={'size': 10}, loc=2,
               ncol=2)
    plt.ylabel(r"$\delta_{w1}$ [rad]", fontsize=14)
    plt.xlabel("Times [s]", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-delta_w1.png"))

    f8 = plt.figure("-delta_w2", figsize=(8, 5))
    ax = f8.add_axes([0.1, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(0, run_step, 1)*delta_t, delta_w2_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1)*delta_t, delta_w2_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['6dof', 'trucksim'], prop={'size': 10}, loc=2,
               ncol=2)
    plt.ylabel(r"$\delta_{w2}$ [rad]", fontsize=14)
    plt.xlabel("Times [s]", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-delta_w2.png"))

    f9 = plt.figure("-delta_w3", figsize=(8, 5))
    ax = f9.add_axes([0.1, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(0, run_step, 1)*delta_t, delta_w3_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1)*delta_t, delta_w3_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['6dof', 'trucksim'], prop={'size': 10}, loc=2,
               ncol=2)
    plt.ylabel(r"$\delta_{w3}$ [rad]", fontsize=14)
    plt.xlabel("Times [s]", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-delta_w3.png"))

    f10 = plt.figure("-delta_w4", figsize=(8, 5))
    ax = f10.add_axes([0.1, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(0, run_step, 1)*delta_t, delta_w4_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1)*delta_t, delta_w4_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['6dof', 'trucksim'], prop={'size': 10}, loc=2,
               ncol=2)
    plt.ylabel(r"$\delta_{w4}$ [rad]", fontsize=14)
    plt.xlabel("Times [s]", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-delta_w4.png"))

    f14 = plt.figure("-Qw3", figsize=(8, 5))
    ax = f14.add_axes([0.1, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(0, run_step, 1)*delta_t, Qw3_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1)*delta_t, Qw3_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['6dof', 'trucksim'], prop={'size': 10}, loc=2,
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
    l1, = plt.plot(np.arange(0, run_step, 1)*delta_t, Qw4_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1)*delta_t, Qw4_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['6dof', 'trucksim'], prop={'size': 10}, loc=2, ncol=2)
    plt.ylabel(r"$Q_{w4}$ [Nm]", fontsize=14)
    plt.xlabel("Times [s]", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-Qw4.png"))

    # plt.show()

if __name__ == '__main__':
    model_compare_4A(env_id='pyth_reconfigurablevehicle')

    # env = gym.make("pyth_reconfigurablevehicle")
    #
    # observation = env.reset()
    #
    #
    # for _ in range(100):
    #   print(_)
    #   # env.render()
    #   print(observation[24:])
    #   action =env.action_space.sample() # your agent here (this takes random actions)
    #   observation, reward = env.step(action) #, done, info
    #   # print(observation)
    #   # if done:
    #   #   observation = env.reset()
    # env.close()