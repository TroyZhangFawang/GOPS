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
    state[0] = state[0] / 180 * np.pi  # beta1
    state[1] = state[1] / 180 * np.pi  # omega1
    state[2] = state[2] / 180 * np.pi  # phi1
    state[3] = state[3] / 180 * np.pi  # phi1dot
    state[4] = state[4] / 180 * np.pi  # beta2
    state[5] = state[5] / 180 * np.pi  # omega2
    state[6] = state[6] / 180 * np.pi  # phi2
    state[7] = state[7] / 180 * np.pi  # phi2dot
    state[8] = state[8] / 180 * np.pi  # varphi1
    state[9] = state[9] / 180 * np.pi  # varphi2
    state[10] = state[10] / 3.6  # vy1
    state[15] = state[15] / 3.6  # vy2
    state[16] = state[16] / 3.6  # vx1
    state[17] = state[17] / 3.6  # vx2

    state[18] = state[18] / 180 * np.pi  # angle hitch
    state[19] = state[19] / 180 * np.pi  # angle rate hitch
    state[20] = state[20] / 180 * np.pi /25  # steering angle
    return state

def model_compare(env_id):
    run_step = 5000
    delta_t = 0.001
    model_mechnical = gym.make(env_id, disable_env_checker=True)
    state = model_mechnical.reset()
    print(state[22:])
    # state 
    state = unit_transform(state)
    model_self = create_env(env_id)
    # 7dof
    state_python = state[:15]

    step_sim = 0
    step_MPC = 0
    phi1_self = []
    phi1_trucksim = []
    phi1dot_self = []
    phi1dot_trucksim = []
    beta1_self = []
    beta1_trucksim = []
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
        if i< 500 :
            steering_angle_degree = 0
        elif i>500and i<1000 :
            steering_angle_degree = 3
        elif i>1000and i<1500 :
            steering_angle_degree = -3
        else:
            steering_angle_degree = -0
        # steering_angle_degree = 2 * np.sin(np.pi * 2 / 20000 * i)
        steering_angle_rad = steering_angle_degree/180*3.14

        state_python = model_self.vehicle_dynamics.f_xu(state_python, np.array([steering_angle_rad]), delta_t)
        control_trucksim = np.array([steering_angle_degree*25])
        state, _ = model_mechnical.step(control_trucksim)  # trucksim
        state = unit_transform(state)

        # print(state[11])
        steering_self.append(steering_angle_rad)
        beta1_self.append(state_python[0])
        psi1dot_self.append(state_python[1])
        phi1_self.append(state_python[2])
        phi1dot_self.append(state_python[3])
        beta2_self.append(state_python[4])
        psi2dot_self.append(state_python[5])
        phi2_self.append(state_python[6])
        phi2dot_self.append(state_python[7])
        psi1_self.append(state_python[8])
        psi2_self.append(state_python[9])
        v1_self.append(state_python[10])
        y1_self.append(state_python[11])
        y2_self.append(state_python[12])
        u1_self.append(model_self.vehicle_dynamics.v_x)

        steering_trucksim.append(state[20])
        beta1_trucksim.append(state[0])
        psi1dot_trucksim.append(state[1])
        phi1_trucksim.append(state[2])
        phi1dot_trucksim.append(state[3])
        beta2_trucksim.append(state[4])
        psi2dot_trucksim.append(state[5])
        phi2_trucksim.append(state[6])
        phi2dot_trucksim.append(state[7])
        psi1_trucksim.append(state[8])
        psi2_trucksim.append(state[9])
        v1_trucksim.append(state[10])
        y1_trucksim.append(state[11])
        y2_trucksim.append(state[12])
        u1_trucksim.append(state[16])

        step_MPC += 1
    data_result = pd.DataFrame(
        {'phi1_self': phi1_self, 'phi1_trucksim': phi1_trucksim,
         'phi1dot_self': phi1dot_self, 'phi1dot_trucksim': phi1dot_trucksim,
         'beta1_self': beta1_self, 'beta1_trucksim': beta1_trucksim,
         'psi1_self': psi1_self, 'psi1_trucksim': psi1_trucksim,
         'psi1dot_self': psi1dot_self, 'psi1dot_trucksim': psi1dot_trucksim,
         'phi2_self': phi2_self, 'phi2_trucksim': phi2_trucksim,
         'phi2dot_self': phi2dot_self, 'phi2dot_trucksim': phi2dot_trucksim,
         'beta2_self': beta2_self, 'beta2_trucksim': beta2_trucksim,
         'psi2_self': psi2_self, 'psi2_trucksim': psi2_trucksim,
         'psi2dot_self': psi2dot_self, 'psi2dot_trucksim': psi2dot_trucksim,
         'y1_self': y1_self, 'y1_trucksim': y1_trucksim,
         'u1_self': u1_self, 'u1_trucksim': u1_trucksim,
         'v1_self': v1_self, 'v1_trucksim': v1_trucksim,
         'y2_self': y2_self, 'y2_trucksim': y2_trucksim, })
    data_result.to_csv('./result_tractor_trailer.csv', encoding='gbk')

    # '--------------------出图-----------------------'
    picture_dir = "./"+str(env_id)
    os.makedirs(picture_dir, exist_ok=True)
    f1 = plt.figure("-phi1", figsize=(8, 5))
    ax = f1.add_axes([0.085, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(0, run_step, 1), phi1_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1), phi1_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['6dof', 'trucksim'], prop={'size': 10}, loc=2, ncol=2)
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
    plt.legend(handles=[l1, l2], labels=['6dof', 'trucksim'], prop={'size': 10}, loc=2, ncol=2)
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
    plt.legend(handles=[l1, l2], labels=['6dof', 'trucksim'], prop={'size': 10}, loc=2,
               ncol=2)
    plt.ylabel("phi1dot [rad/s]", fontsize=14)
    plt.xlabel("step", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-phi1dot.png"))

    f3 = plt.figure("-beta1", figsize=(8, 5))
    ax = f3.add_axes([0.09, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(0, run_step, 1), beta1_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1), beta1_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['6dof', 'trucksim'], prop={'size': 10}, loc=2,
               ncol=2)
    plt.ylabel("beta1 [rad]", fontsize=14)
    plt.xlabel("step", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-beta1.png"))

    f4 = plt.figure("-psi1", figsize=(8, 5))
    ax = f4.add_axes([0.085, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(0, run_step, 1), psi1_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1), psi1_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['6dof', 'trucksim'], prop={'size': 10}, loc=2,
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
    plt.legend(handles=[l1, l2], labels=['6dof', 'trucksim'], prop={'size': 10}, loc=2,
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
    plt.legend(handles=[l1, l2], labels=['6dof', 'trucksim'], prop={'size': 10}, loc=2,
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
    plt.legend(handles=[l1, l2], labels=['6dof', 'trucksim'], prop={'size': 10}, loc=2,
               ncol=2)
    plt.ylabel("phi2dot [rad/s]", fontsize=14)
    plt.xlabel("step", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-phi2dot.png"))

    f8 = plt.figure("-beta2", figsize=(8, 5))
    ax = f8.add_axes([0.09, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(0, run_step, 1), beta2_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1), beta2_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['6dof', 'trucksim'], prop={'size': 10}, loc=2,
               ncol=2)
    plt.ylabel("beta2 [rad]", fontsize=14)
    plt.xlabel("step", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.subplots_adjust(bottom=0.31)
    plt.grid(axis='both', ls='-.')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.savefig(os.path.join(picture_dir, "-beta2.png"))

    f9 = plt.figure("-psi2", figsize=(8, 5))
    ax = f9.add_axes([0.085, 0.11, 0.87, 0.86])  # [left, bottom, width, height]
    l1, = plt.plot(np.arange(0, run_step, 1), psi2_self, lw=2, color="darkviolet")
    l2, = plt.plot(np.arange(0, run_step, 1), psi2_trucksim, lw=2, linestyle='--', color="deepskyblue")
    plt.legend(handles=[l1, l2], labels=['6dof', 'trucksim'], prop={'size': 10}, loc=2,
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
    plt.legend(handles=[l1, l2], labels=['6dof', 'trucksim'], prop={'size': 10}, loc=2,
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
    plt.legend(handles=[l1, l2], labels=['6dof', 'trucksim'], prop={'size': 10}, loc=2,
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
    plt.legend(handles=[l1, l2], labels=['6dof', 'trucksim'], prop={'size': 10}, loc=2,
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
    plt.legend(handles=[l1, l2], labels=['6dof', 'trucksim'], prop={'size': 10}, loc=2,
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
    plt.legend(handles=[l1, l2], labels=['6dof', 'trucksim'], prop={'size': 10}, loc=2,
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
    model_compare(env_id='pyth_semitruck7dof')
