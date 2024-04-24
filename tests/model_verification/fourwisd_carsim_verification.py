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
    state_2 = np.array(data_result.iloc[:, 1], dtype='float32')  #y
    state_traj = np.zeros((len(state_1), 2))
    state_traj[:, 0] = state_1
    state_traj[:, 1] = state_2
    return state_traj

def unit_transform_2A(state):
    state[0] = state[0]  # x
    state[1] = state[1]  # y
    state[2] = state[2] / 180 * np.pi  # yaw

    state[3] = state[3] / 3.6  # vx
    state[4] = state[4] / 3.6  # vy
    state[5] = state[5] / 180 * np.pi  # yaw_rate
    state[6] = state[6] / 180 * np.pi  # roll angle
    state[7] = state[7] / 180 * np.pi  # roll rate

    state[8] = state[8]  # kappa_1
    state[9] = state[9]  # kappa_2
    state[10] = state[10] # kappa_3
    state[11] = state[11] # kappa_4
    # control feedback
    state[12] = state[12]  # drive torque on wheel 1
    state[13] = state[13] / 180 * np.pi  # steering angle on wheel 1
    state[14] = state[14]
    state[15] = state[15] / 180 * np.pi  # steering angle on wheel 2
    state[16] = state[16]
    state[17] = state[17] / 180 * np.pi  # steering angle on wheel 3
    state[18] = state[18]
    state[19] = state[19] / 180 * np.pi  # steering angle on wheel 4
    state[20] = state[20]/ 180 * np.pi # beta
    return state


def model_compare_4wisd(env_id):
    run_step = 12000
    delta_t = 0.0005
    model_mechnical = gym.make(env_id, disable_env_checker=True)
    state = model_mechnical.reset()

    # state
    state = unit_transform_2A(state)
    model_self = create_env(env_id)
    # 4dof
    state_python = state[:12]
    print(state[21:])
    step_sim = 0
    vx_self = []
    vx_carsim = []
    vy_self = []
    vy_carsim = []
    yawrate_self = []
    yawrate_carsim = []
    roll_self = []
    roll_carsim = []
    rollrate_self = []
    rollrate_carsim = []
    x_self = []
    x_carsim = []
    y_self = []
    y_carsim = []
    yaw_self = []
    yaw_carsim = []
    kappa_1_self = []
    kappa_4_self = []
    kappa_1_carsim = []
    kappa_4_carsim = []
    
    
    Qw1_self = []
    Qw1_carsim = []
    delta_w1_self = []
    delta_w1_carsim = []
    Qw2_self = []
    Qw2_carsim = []
    delta_w2_self = []
    delta_w2_carsim = []
    delta_w3_self = []
    delta_w3_carsim = []
    delta_w4_self = []
    delta_w4_carsim = []

    Qw3_self = []
    Qw3_carsim = []

    Qw4_self = []
    Qw4_carsim = []
    
    

    for i in range(run_step):
        drive_torque = 80
        # if i< 4000 :
        #     steering_angle_degree = 0
        # # elif i>500and i<1000 :
        # #     steering_angle_degree = 3
        # # elif i>1000and i<1500 :
        # #     steering_angle_degree = -3
        # else:
        #     steering_angle_degree = 3
        steering_angle_degree = 3 * np.sin(np.pi * 2 / 5000 * i)
        steering_angle_rad = steering_angle_degree / 180 * 3.14
        control_carsim = np.array([drive_torque, steering_angle_degree,
                                     drive_torque, steering_angle_degree,
                                     drive_torque, 0,
                                     drive_torque, 0])
        control = np.array([drive_torque, steering_angle_rad,
                            drive_torque, steering_angle_rad,
                            drive_torque, 0,
                            drive_torque, 0])

        state_python = model_self.vehicle_dynamics.f_xu(state_python, control, delta_t)

        state, _ = model_mechnical.step(control_carsim)  # carsim
        state = unit_transform_2A(state)
        
        # 4dof
        x_self.append(state_python[0])
        y_self.append(state_python[1])
        yaw_self.append(state_python[2])
        vx_self.append(state_python[3])
        vy_self.append(state_python[4])
        yawrate_self.append(state_python[5])
        roll_self.append(state_python[6])
        rollrate_self.append(state_python[7])

        kappa_1_self.append(state_python[8])
        kappa_4_self.append(state_python[11])
        Qw1_self.append(control[0])
        delta_w1_self.append(control[1])

        delta_w2_self.append(control[3])
        
        Qw3_self.append(control[4])
        delta_w3_self.append(control[5])
        
        Qw4_self.append(control[6])
        delta_w4_self.append(control[7])


        # -------------------------------
        x_carsim.append(state[0])
        y_carsim.append(state[1])
        yaw_carsim.append(state[2])
        vx_carsim.append(state[3])
        vy_carsim.append(state[4])
        yawrate_carsim.append(state[5])
        roll_carsim.append(state[6])
        rollrate_carsim.append(state[7])

        kappa_1_carsim.append(state[8])
        kappa_4_carsim.append(state[11])
        
        
        Qw1_carsim.append(state[12])
        delta_w1_carsim.append(state[13])

        delta_w2_carsim.append(state[15])
        Qw3_carsim.append(state[16])
        delta_w3_carsim.append(state[17])
        Qw4_carsim.append(state[18])
        delta_w4_carsim.append(state[19])
        step_sim += 1
    print("run finished")
    data_result = pd.DataFrame(
        {'yaw_self': yaw_self, 'yaw_carsim': yaw_carsim,
         'yawrate_self': yawrate_self, 'yawrate_carsim': yawrate_carsim,
         'x_self': x_self, 'x_carsim': x_carsim,
         'Qw1_self': Qw1_self, 'Qw1_carsim': Qw1_carsim,
         'delta_w1_self': delta_w1_self, 'delta_w1_carsim': delta_w1_carsim,
         'roll_self': roll_self, 'roll_carsim': roll_carsim,
         'rollrate_self': rollrate_self, 'rollrate_carsim': rollrate_carsim,
         'delta_w2_self': delta_w2_self, 'delta_w2_carsim': delta_w2_carsim,
         'delta_w3_self': delta_w3_self, 'delta_w3_carsim': delta_w3_carsim,
         'delta_w4_self': delta_w4_self, 'delta_w4_carsim': delta_w4_carsim,
         'y_self': y_self, 'y_carsim': y_carsim,
         'vx_self': vx_self, 'vx_carsim': vx_carsim,
         'vy_self': vy_self, 'vy_carsim': vy_carsim,
         'Qw3_self': Qw3_self, 'Qw3_carsim': Qw3_carsim, })
    data_result.to_csv('./result_4wisd.csv', encoding='gbk')

    # '--------------------出图-----------------------'
    picture_dir = "./plot_4wisd/"
    os.makedirs(picture_dir, exist_ok=True)
    f9 = plt.figure("-kappa1", figsize=(8, 5))
    ax = f9.add_axes([0.1, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(0, run_step, 1) * delta_t, kappa_1_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1) * delta_t, kappa_1_carsim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['4wisd', 'carsim'], prop={'size': 10}, loc=2,
               ncol=2)
    plt.ylabel(r"$\kappa_1$ [-]", fontsize=14)
    plt.xlabel("Times [s]", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-kappa1.png"))

    f12 = plt.figure("-kappa4", figsize=(8, 5))
    ax = f12.add_axes([0.1, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(0, run_step, 1) * delta_t, kappa_4_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1) * delta_t, kappa_4_carsim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['4wisd', 'carsim'], prop={'size': 10}, loc=2,
               ncol=2)
    plt.ylabel(r"$\kappa_4$ [-]", fontsize=14)
    plt.xlabel("Times [s]", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-kappa4.png"))
    
    
    f0 = plt.figure("-vx", figsize=(8, 5))
    ax = f0.add_axes([0.1, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(0, run_step, 1)*delta_t, vx_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1)*delta_t, vx_carsim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['4wisd', 'carsim'], prop={'size': 10}, loc=2,
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
    l2, = plt.plot(np.arange(0, run_step, 1)*delta_t, vy_carsim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['4wisd', 'carsim'], prop={'size': 10}, loc=2,
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
    l2, = plt.plot(np.arange(0, run_step, 1)*delta_t, yawrate_carsim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['4wisd', 'carsim'], prop={'size': 10}, loc=2,
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
    l2, = plt.plot(np.arange(0, run_step, 1)*delta_t, roll_carsim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['4wisd', 'carsim'], prop={'size': 10}, loc=2,
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
    l2, = plt.plot(np.arange(0, run_step, 1)*delta_t, rollrate_carsim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['4wisd', 'carsim'], prop={'size': 10}, loc=2,
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
    l2, = plt.plot(np.arange(0, run_step, 1)*delta_t, x_carsim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['4wisd', 'carsim'], prop={'size': 10}, loc=2,
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
    l2, = plt.plot(np.arange(0, run_step, 1)*delta_t, y_carsim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['4wisd', 'carsim'], prop={'size': 10}, loc=2,
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
    l2, = plt.plot(np.arange(0, run_step, 1)*delta_t, yaw_carsim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['4wisd', 'carsim'], prop={'size': 10}, loc=2, ncol=2)
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
    l2, = plt.plot(np.arange(0, run_step, 1)*delta_t, Qw1_carsim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['4wisd', 'carsim'], prop={'size': 10}, loc=2,
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
    l2, = plt.plot(np.arange(0, run_step, 1)*delta_t, delta_w1_carsim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['4wisd', 'carsim'], prop={'size': 10}, loc=2,
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
    l2, = plt.plot(np.arange(0, run_step, 1)*delta_t, delta_w2_carsim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['4wisd', 'carsim'], prop={'size': 10}, loc=2,
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
    l2, = plt.plot(np.arange(0, run_step, 1)*delta_t, delta_w3_carsim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['4wisd', 'carsim'], prop={'size': 10}, loc=2,
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
    l2, = plt.plot(np.arange(0, run_step, 1)*delta_t, delta_w4_carsim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['4wisd', 'carsim'], prop={'size': 10}, loc=2,
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
    l2, = plt.plot(np.arange(0, run_step, 1)*delta_t, Qw3_carsim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['4wisd', 'carsim'], prop={'size': 10}, loc=2,
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
    l2, = plt.plot(np.arange(0, run_step, 1)*delta_t, Qw4_carsim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['4wisd', 'carsim'], prop={'size': 10}, loc=2, ncol=2)
    plt.ylabel(r"$Q_{w4}$ [Nm]", fontsize=14)
    plt.xlabel("Times [s]", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-Qw4.png"))

    # plt.show()

if __name__ == '__main__':
    model_compare_4wisd(env_id='pyth_holisticcontrol')
    # root_path = "C:/Users/Troy.Z/Desktop/lat.csv"
    # re = read_path(root_path)
    # x = re[:, 0]
    # y = re[:, 1]
    # popt1, pcov1 = curve_fit(stiffness_fitting, x, y,maxfev = 10000)
    # print(popt1)


