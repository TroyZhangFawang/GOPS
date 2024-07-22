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

def unit_transform(state):

    state[2] = state[2] / 180 * np.pi  # yaw
    state[3] = state[3] / 3.6  # vx1

    state[6] = state[6] / 180 * np.pi  # yaw2
    state[7] = state[7] / 3.6  # vx2
    state[8] = state[8] / 3.6  # vy1
    state[9] = state[9] / 180 * np.pi  # gamma1
    state[10] = state[10] / 180 * np.pi  # varphi1
    state[11] = state[11] / 180 * np.pi  # varphi1dot
    state[12] = state[12] / 3.6  # vy2
    state[13] = state[13] / 180 * np.pi  # gamma2
    state[14] = state[14] / 180 * np.pi  # varphi2
    state[15] = state[15] / 180 * np.pi  # varphi2dot

    state[19] = state[19]/25 / 180 * np.pi  # steering[rad]
    state[20] = state[20] / 180 * np.pi  # beta1
    state[21] = state[21] / 180 * np.pi  # beta2
    state[22] = state[22] / 180 * np.pi  # angle hitch
    state[23] = state[23] / 180 * np.pi  # angle rate hitch
    state[24] = state[24] * 9.8  # ax
    return state

def model_compare(env_id):
    run_step = 2000
    delta_t = 0.0005
    model_mechnical = gym.make(env_id, disable_env_checker=True)
    state = model_mechnical.reset()
    print(state[25:])
    # state 
    state = unit_transform(state)
    model_self = create_env(env_id)
    model_self.Load_engine_data()
    # 7dof
    state_python = state[:16]

    step_sim = 0
    step_MPC = 0
    phi1_self = []
    phi1_trucksim = []
    phi1dot_self = []
    phi1dot_trucksim = []
    ax_self = []
    ax_trucksim = []
    psi1_self = []
    psi1_trucksim = []
    psi1dot_self = []
    psi1dot_trucksim = []
    phi2_self = []
    phi2_trucksim = []
    phi2dot_self = []
    phi2dot_trucksim = []
    beta2_self = []
    beta2_trucksim = []
    psi2_self = []
    psi2_trucksim = []
    psi2dot_self = []
    psi2dot_trucksim = []
    y1_self = []
    y1_trucksim = []
    u1_self = []
    u1_trucksim = []
    v1_self = []
    v1_trucksim = []
    y2_self = []
    y2_trucksim = []
    steering_self = []
    steering_trucksim = []
    
    for i in range(run_step):
        # if i< 500 :
        #     steering_angle_degree = 0
        # # # elif i>500and i<1000 :
        # # #     steering_angle_degree = 3
        # # # elif i>1000and i<1500 :
        # # #     steering_angle_degree = -3
        # else:
        steering_angle_degree = 10 * np.sin(np.pi * 2 / 20000 * i)
        accel = 1.4 #* np.sin(np.pi * 2 / 20000 * i)
        steering_angle_rad = steering_angle_degree/180*3.14
        ig = state[17]
        EngSpd, TransSpd = model_self.Calc_EngSpd(state_python[3], ig)
        EngTorque = model_self.Calc_EngTe(state_python[3], accel, ig)
        throttle = model_self.Calc_Thr(EngSpd, EngTorque)

        state_python = model_self.vehicle_dynamics.f_xu(state_python, np.array([accel, steering_angle_rad]), delta_t)
        control_trucksim = np.array([throttle, steering_angle_degree*25])
        state, _ = model_mechnical.step(control_trucksim)  # trucksim
        state = unit_transform(state)

        steering_self.append(steering_angle_rad)
        y1_self.append(state_python[1])
        psi1_self.append(state_python[2])
        u1_self.append(state_python[3])

        y2_self.append(state_python[5])
        psi2_self.append(state_python[6])

        v1_self.append(state_python[8])
        psi1dot_self.append(state_python[9])
        phi1_self.append(state_python[10])
        phi1dot_self.append(state_python[11])
        psi2dot_self.append(state_python[13])
        phi2_self.append(state_python[14])
        phi2dot_self.append(state_python[15])
        ax_self.append(accel)
        
        steering_trucksim.append(state[19])
        y1_trucksim.append(state[1])
        psi1_trucksim.append(state[2])
        u1_trucksim.append(state[3])

        y2_trucksim.append(state[5])
        psi2_trucksim.append(state[6])

        v1_trucksim.append(state[8])
        psi1dot_trucksim.append(state[9])
        phi1_trucksim.append(state[10])
        phi1dot_trucksim.append(state[11])
        psi2dot_trucksim.append(state[13])
        phi2_trucksim.append(state[14])
        phi2dot_trucksim.append(state[15])
        ax_trucksim.append(state[24])


        step_MPC += 1
    data_result = pd.DataFrame(
        {'phi1_self': phi1_self, 'phi1_trucksim': phi1_trucksim,
         'phi1dot_self': phi1dot_self, 'phi1dot_trucksim': phi1dot_trucksim,
         'psi1_self': psi1_self, 'psi1_trucksim': psi1_trucksim,
         'psi1dot_self': psi1dot_self, 'psi1dot_trucksim': psi1dot_trucksim,
         'phi2_self': phi2_self, 'phi2_trucksim': phi2_trucksim,
         'phi2dot_self': phi2dot_self, 'phi2dot_trucksim': phi2dot_trucksim,
         'psi2_self': psi2_self, 'psi2_trucksim': psi2_trucksim,
         'psi2dot_self': psi2dot_self, 'psi2dot_trucksim': psi2dot_trucksim,
         'y1_self': y1_self, 'y1_trucksim': y1_trucksim,
         'u1_self': u1_self, 'u1_trucksim': u1_trucksim,
         'v1_self': v1_self, 'v1_trucksim': v1_trucksim,
         'y2_self': y2_self, 'y2_trucksim': y2_trucksim, })
    data_result.to_csv('./result_tractor_trailer8dof.csv', encoding='gbk')

    # '--------------------出图-----------------------'
    picture_dir = "./"+str(env_id)
    os.makedirs(picture_dir, exist_ok=True)
    f1 = plt.figure("-phi1", figsize=(8, 5))
    ax = f1.add_axes([0.085, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(0, run_step, 1), phi1_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1), phi1_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['8dof', 'trucksim'], prop={'size': 10}, loc=2, ncol=2)
    plt.ylabel("phi1 [rad]", fontsize=14)
    plt.xlabel("step", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-phi1.png"))

    f1 = plt.figure("-str", figsize=(8, 5))
    ax = f1.add_axes([0.085, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(0, run_step, 1), steering_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1), steering_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['8dof', 'trucksim'], prop={'size': 10}, loc=2, ncol=2)
    plt.ylabel("str [rad]", fontsize=14)
    plt.xlabel("step", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-str.png"))

    f2 = plt.figure("-phi1dot", figsize=(8, 5))
    ax = f2.add_axes([0.085, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(0, run_step, 1), phi1dot_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1), phi1dot_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['8dof', 'trucksim'], prop={'size': 10}, loc=2,
               ncol=2)
    plt.ylabel("phi1dot [rad/s]", fontsize=14)
    plt.xlabel("step", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-phi1dot.png"))

    f3 = plt.figure("-ax", figsize=(8, 5))
    ax = f3.add_axes([0.09, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(0, run_step, 1), ax_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1), ax_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['8dof', 'trucksim'], prop={'size': 10}, loc=2,
               ncol=2)
    plt.ylabel("ax [m/s^2]", fontsize=14)
    plt.xlabel("step", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-ax.png"))

    f4 = plt.figure("-psi1", figsize=(8, 5))
    ax = f4.add_axes([0.085, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(0, run_step, 1), psi1_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1), psi1_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['8dof', 'trucksim'], prop={'size': 10}, loc=2,
               ncol=2)
    plt.ylabel("psi1 [rad]", fontsize=14)
    plt.xlabel("step", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-psi1.png"))

    f5 = plt.figure("-psi1dot", figsize=(8, 5))
    ax = f5.add_axes([0.1, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(0, run_step, 1), psi1dot_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1), psi1dot_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['8dof', 'trucksim'], prop={'size': 10}, loc=2,
               ncol=2)
    plt.ylabel("psi1dot [rad/s]", fontsize=14)
    plt.xlabel("step", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-psi1dot.png"))

    f6 = plt.figure("-phi2", figsize=(8, 5))
    ax = f6.add_axes([0.085, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(0, run_step, 1), phi2_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1), phi2_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['8dof', 'trucksim'], prop={'size': 10}, loc=2,
               ncol=2)
    plt.ylabel("phi2 [rad]", fontsize=14)
    plt.xlabel("step", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-phi2.png"))

    f7 = plt.figure("-phi2dot", figsize=(8, 5))
    ax = f7.add_axes([0.085, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(0, run_step, 1), phi2dot_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1), phi2dot_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['8dof', 'trucksim'], prop={'size': 10}, loc=2,
               ncol=2)
    plt.ylabel("phi2dot [rad/s]", fontsize=14)
    plt.xlabel("step", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-phi2dot.png"))

    # f8 = plt.figure("-beta2", figsize=(8, 5))
    # ax = f8.add_axes([0.09, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    # l1, = plt.plot(np.arange(0, run_step, 1), beta2_self, lw=2, color="darkviolet")
    # l2, = plt.plot(np.arange(0, run_step, 1), beta2_trucksim, lw=2, linestyle='--', color="deepskyblue")
    # plt.legend(handles=[l1, l2], labels=['8dof', 'trucksim'], prop={'size': 10}, loc=2,
    #            ncol=2)
    # plt.ylabel("beta2 [rad]", fontsize=14)
    # plt.xlabel("step", fontsize=14)
    # plt.tick_params(labelsize=12)
    # plt.subplots_adjust(bottom=0.31)
    # plt.grid(axis='both', ls='-.')
    # plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    # plt.savefig(os.path.join(picture_dir, "-beta2.png"))

    f9 = plt.figure("-psi2", figsize=(8, 5))
    ax = f9.add_axes([0.085, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(0, run_step, 1), psi2_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1), psi2_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['8dof', 'trucksim'], prop={'size': 10}, loc=2,
               ncol=2)
    plt.ylabel("psi2 [rad]", fontsize=14)
    plt.xlabel("step", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-psi2.png"))

    f10 = plt.figure("-psi2dot", figsize=(8, 5))
    ax = f10.add_axes([0.1, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(0, run_step, 1), psi2dot_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1), psi2dot_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['8dof', 'trucksim'], prop={'size': 10}, loc=2,
               ncol=2)
    plt.ylabel("psi2dot [rad/s]", fontsize=14)
    plt.xlabel("step", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-psi2dot.png"))

    f11 = plt.figure("-y1", figsize=(8, 5))
    ax = f11.add_axes([0.1, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(0, run_step, 1), y1_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1), y1_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['8dof', 'trucksim'], prop={'size': 10}, loc=2,
               ncol=2)
    plt.ylabel("y1 [m]", fontsize=14)
    plt.xlabel("step", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-y1.png"))

    f12 = plt.figure("-u1", figsize=(8, 5))
    ax = f12.add_axes([0.1, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(0, run_step, 1), u1_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1), u1_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['8dof', 'trucksim'], prop={'size': 10}, loc=2,
               ncol=2)
    plt.ylabel("u1 [m/s]", fontsize=14)
    plt.xlabel("step", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-u1.png"))

    f13 = plt.figure("-v1", figsize=(8, 5))
    ax = f13.add_axes([0.1, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(0, run_step, 1), v1_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1), v1_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['8dof', 'trucksim'], prop={'size': 10}, loc=2,
               ncol=2)
    plt.ylabel("v1 [m/s]", fontsize=14)
    plt.xlabel("step", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-v1.png"))

    f14 = plt.figure("-y2", figsize=(8, 5))
    ax = f14.add_axes([0.1, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(0, run_step, 1), y2_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1), y2_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['8dof', 'trucksim'], prop={'size': 10}, loc=2,
               ncol=2)
    plt.ylabel("y2 [m]", fontsize=14)
    plt.xlabel("step", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-y2.png"))

    # plt.show()


if __name__ == '__main__':
    model_compare(env_id='pyth_semitruck8dof')
