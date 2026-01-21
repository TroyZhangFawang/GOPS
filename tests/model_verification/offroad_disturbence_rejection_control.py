import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import pandas as pd
import os
import gym
import copy
import warnings

# 忽略部分无关紧要的警告
warnings.filterwarnings("ignore")
SAVE_DIR = "./offroad_disturbance_rejection"
# 请确保此文件存在，包含列 'x' 和 'y' (或第一列x第二列y)
REF_FILE_PATH = os.path.join(SAVE_DIR, "reference_trajectory.csv")
os.makedirs(SAVE_DIR, exist_ok=True)


class ReferencePath:
    """
    改进版：基于最近点搜索 (Nearest Point Search)
    """

    def __init__(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Error: Reference file not found: {file_path}")

        print(f"Loading reference trajectory from: {file_path}")
        df = pd.read_csv(file_path)

        # 读取数据
        if 'x' in df.columns and 'y' in df.columns:
            self.path_x = df['x'].values
            self.path_y = df['y'].values
        else:
            self.path_x = df.iloc[:, 0].values
            self.path_y = df.iloc[:, 1].values

        # 预计算航向角 (便于后续查询)
        dx = np.gradient(self.path_x)
        dy = np.gradient(self.path_y)
        self.path_psi = np.arctan2(dy, dx)

        # 记录路径长度
        self.length = len(self.path_x)
        # 上一次匹配的索引 (用于加速搜索)
        self.last_idx = 0

    def get_reference_state(self, x_veh, y_veh):
        """
        输入车辆当前 (x, y)
        返回:
        ref_x, ref_y: 最近点的坐标
        ref_psi: 最近点的切线方向
        lat_error: 带符号的横向误差 (左正右负，或反之，取决于坐标系)
        """
        # 1. 全局搜索：计算车辆到所有轨迹点的距离
        dx = self.path_x - x_veh
        dy = self.path_y - y_veh
        dist_sq = dx ** 2 + dy ** 2

        # 2. 找到最近点的索引
        min_idx = np.argmin(dist_sq)

        ref_x = self.path_x[min_idx]
        ref_y = self.path_y[min_idx]
        ref_psi = self.path_psi[min_idx]

        # 3. 计算横向误差的数值 (距离)
        lat_error_abs = np.sqrt(dist_sq[min_idx])

        # 4. 判断误差符号 (关键步骤！)
        # 构建路径切线向量
        path_vec_x = np.cos(ref_psi)
        path_vec_y = np.sin(ref_psi)

        # 构建 车辆->路径点 的向量 (注意方向: P_veh - P_ref)
        err_vec_x = x_veh - ref_x
        err_vec_y = y_veh - ref_y

        # 使用 2D 叉乘判断左右关系
        # Cross = V_path_x * V_err_y - V_path_y * V_err_x
        # 如果 > 0，说明车在路径左侧； < 0 在右侧
        cross_prod = path_vec_x * err_vec_y - path_vec_y * err_vec_x

        if cross_prod > 0:
            lat_error = lat_error_abs  # 车在左边
        else:
            lat_error = -lat_error_abs  # 车在右边

        return ref_x, ref_y, ref_psi, lat_error
# ================= 1. 全局绘图风格配置 (完全参考您提供的模板) =================
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['axes.unicode_minus'] = False

# 1.2 配色与线型配置
SINGLE_LINE_COLOR = '#D62728'
SINGLE_LINE_STYLE = '-'
MULTI_COLORS = ['blue', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b']
MULTI_STYLES = ['-', '--', '-.', ':']
# 尝试加载中文字体
try:
    zhfont1 = fm.FontProperties(fname='../../gops/utils/SIMSUN.ttf', size=16)
except:
    zhfont1 = fm.FontProperties(family='SimHei', size=16)

default_cfg = {
    "fig_size": (12, 9),
    "dpi": 300,
    "tick_size": 18,
    "label_size": 18,
    "legend_size": 14,
    "line_width": 2.0,
    "img_fmt": "png"
}

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)

def get_font_prop(size):
    prop = copy.copy(zhfont1)
    prop.set_size(size)
    return prop

# ================= 2. 核心算法类 =================
class VehicleDynamics:
    def __init__(self):
        self.m = 2060.0
        self.g = 9.81
        self.lw = 1.8
        self.lf = 1.4442
        self.lr = 1.5558
        self.Izz = 3524.9
        self.Cf = 110000.0
        self.Cr = 110000.0
        self.L = self.lf + self.lr

class PID:
    def __init__(self, kp, ki, kd, cmd_max, cmd_min):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.ep = self.ei = self.ed = 0.0
        self.cmd_max, self.cmd_min = cmd_max, cmd_min

    def get_cmd(self, error, dt=0.01):
        self.ed = (error - self.ep) / dt
        self.ei += error * dt
        # 积分抗饱和
        limit = max(abs(self.cmd_max), abs(self.cmd_min))
        if self.ki != 0:
            self.ei = np.clip(self.ei, -limit / self.ki, limit / self.ki)

        self.ep = error
        cmd = self.kp * self.ep + self.ki * self.ei + self.kd * self.ed
        return np.clip(cmd, self.cmd_min, self.cmd_max)

class DisturbanceObserver:
    """非线性扰动观测器 (NDO)"""

    def __init__(self, veh_params, gain=50.0, dt=0.01):
        self.p = veh_params
        self.L = gain
        self.dt = dt
        self.z = 0.0
        self.d_hat = 0.0

    def update(self, vx, vy, r, delta):
        if abs(vx) < 1.0: return 0.0

        # 标称模型计算
        alpha_f = delta - (vy + self.p.lf * r) / vx
        alpha_r = - (vy - self.p.lr * r) / vx
        Fyf = self.p.Cf * alpha_f
        Fyr = self.p.Cr * alpha_r
        nominal_yaw_acc = (self.p.lf * Fyf - self.p.lr * Fyr) / self.p.Izz

        # 观测器迭代
        z_dot = -self.L * (self.z + self.L * r + nominal_yaw_acc)
        self.z += z_dot * self.dt

        d_hat_acc = self.z + self.L * r
        self.d_hat = d_hat_acc * self.p.Izz  # N*m

        return self.d_hat

# ================= 3. 仿真与绘图函数 =================
def run_single_simulation_(mode_name, enable_ndo, env_id='pyth_stabilitycontrol', run_step=1000, dt=0.01):
    print(f"[{mode_name}] Starting Simulation (NDO={enable_ndo})...")
    # 1. 加载轨迹
    try:
        ref_path = ReferencePath(REF_FILE_PATH)
    except Exception as e:
        print(e)
        return
    try:
        env = gym.make(env_id, disable_env_checker=True)
    except Exception as e:
        print(f"Error loading Gym env: {e}")
        return

    state, _ = env.reset()
    veh_dyn = VehicleDynamics()

    # --- PID参数修正 ---
    # 纵向PID
    lon_pid = PID(kp=800, ki=2.0, kd=10, cmd_max=1500, cmd_min=-1500)
    # 横向PID：增加 ki=0.1 以消除稳态误差
    lat_pid = PID(kp=10.5, ki=0.5, kd=0.2, cmd_max=15.0, cmd_min=-15.0)

    ndo = DisturbanceObserver(veh_dyn, gain=60.0, dt=dt)

    data_records = []
    target_vx = 60 / 3.6

    for i in range(run_step):
        # 状态解包
        x, y, psi = state[0], state[1], state[2]
        vx, vy, r = state[3], state[4], state[5]
        phi = state[6]  # 侧倾角 rad
        beta = state[17]  # 质心侧偏角 rad

        # --- 1. 获取期望轨迹 ---
        ref_x, ref_y, psi_ref, current_lat_error = ref_path.get_reference_state(x, y)
        err_vec_x = x - ref_x
        err_vec_y = y - ref_y
        # --- 2. 基础控制 ---
        drive_torque = lon_pid.get_cmd(target_vx - vx, dt)

        # 计算跟踪误差
        lat_error = np.sqrt(err_vec_x ** 2 + err_vec_y ** 2)
        cross_prod = err_vec_x * np.sin(psi_ref) - err_vec_y * np.cos(psi_ref)
        if cross_prod > 0:
            lat_error = -lat_error  # 车辆在左侧，误差定义为负(视坐标系定义而定)
        else:
            lat_error = lat_error

        # 计算航向误差
        head_error = psi_ref - psi
        # 归一化到 [-pi, pi]
        while head_error > np.pi: head_error -= 2 * np.pi
        while head_error < -np.pi: head_error += 2 * np.pi

        # 混合误差输入

        k_lat = 1.0  # 侧向误差权重
        k_phi = 2.5  # 航向误差权重 (航向对于稳定性更重要)

        total_error = -k_lat * current_lat_error + k_phi * head_error
        steer_cmd_deg = lat_pid.get_cmd(total_error, dt)

        # --- 3. NDO 观测 ---
        delta_rad = steer_cmd_deg * np.pi / 180.0
        dist_yaw_moment = ndo.update(vx, vy, r, delta_rad)

        # --- 4. 前馈补偿 ---
        diff_torque = 0.0
        if enable_ndo:
            Rw = 0.368
            # 计算完全抵消所需的差动扭矩
            T_diff_req = -dist_yaw_moment * 2 * Rw / veh_dyn.lw
            T_diff_req = np.clip(T_diff_req, -2000, 2000)
            diff_torque = T_diff_req

        # --- 5. 执行分配 ---
        T_fl = drive_torque - diff_torque / 2.0
        T_fr = drive_torque + diff_torque / 2.0
        T_rl = drive_torque - diff_torque / 2.0
        T_rr = drive_torque + diff_torque / 2.0

        action = np.array([T_fl, T_fr, T_rl, T_rr, steer_cmd_deg])

        try:
            next_state, _, done, _ = env.step(action)
        except:
            break
        print(current_lat_error)
        data_records.append({
            "time": i * dt,
            "x": x, "y": y,
            "lat_err": current_lat_error,  # 记录下来画图
            "y_ref": ref_y,  # 记录期望Y
            "vx": vx, "r": r, "phi": phi, "beta": beta,
            "dist_est": dist_yaw_moment,
            "diff_torque": diff_torque,
            "lat_err": lat_error
        })

        state = next_state
        if done: break

    env.close()

    df = pd.DataFrame(data_records)
    filename = f"data_{mode_name.lower()}.csv"
    filepath = os.path.join(SAVE_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"Data saved: {filepath}")


def run_single_simulation(mode_name, enable_ndo, env_id='pyth_stabilitycontrol', run_step=1000, dt=0.01):
    print(f"[{mode_name}] Starting Simulation (NDO={enable_ndo})...")

    # 1. 加载轨迹
    try:
        ref_path = ReferencePath(REF_FILE_PATH)
    except Exception as e:
        print(e)
        return

    try:
        env = gym.make(env_id, disable_env_checker=True)
    except Exception as e:
        print(f"Error loading Gym env: {e}")
        return

    state, _ = env.reset()
    veh_dyn = VehicleDynamics()

    # --- 控制器参数调优 (Aggressive Tuning) ---
    # 纵向PID
    lon_pid = PID(kp=800, ki=2.0, kd=10, cmd_max=1500, cmd_min=-1500)

    # 横向PID (大幅增强)
    # Kp: 15.0 (原4.0) -> 快速响应误差
    # Ki: 0.5 (原0.05) -> 消除弯道稳态误差
    # Kd: 2.0 (原0.5)  -> 增加阻尼，防止震荡
    # 限幅: 40度 (原20度) -> 允许更大的转向角
    lat_pid = PID(kp=15.0, ki=0.5, kd=5.0, cmd_max=40.0, cmd_min=-40.0)

    ndo = DisturbanceObserver(veh_dyn, gain=60.0, dt=dt)

    data_records = []
    target_vx = 60 / 3.6

    # 车辆参数 (用于前馈计算)
    L = veh_dyn.L  # 轴距
    K_us = 0.0015  # 不足转向系数 (估算值)

    for i in range(run_step):
        # 状态解包
        x, y, psi = state[0], state[1], state[2]
        vx, vy, r = state[3], state[4], state[5]
        phi = state[6]
        beta = state[17]

        # --- 1. 获取期望轨迹与曲率 ---
        # 我们需要 ReferencePath 类增加一个返回曲率的功能，或者在这里数值差分计算
        # 简单起见，这里复用 ReferencePath 的逻辑，并手动计算曲率前馈
        ref_x, ref_y, ref_psi, lat_error = ref_path.get_reference_state(x, y)
        print(lat_error)
        # 计算期望曲率 kappa (根据正弦波公式 y = 1.5 sin(2pi x / 100))
        # y' = 1.5 * (2pi/100) * cos(...)
        # y'' = -1.5 * (2pi/100)^2 * sin(...)
        # kappa = y'' / (1 + y'^2)^1.5
        k = 2 * np.pi / 100
        y_prime = 1.5 * k * np.cos(k * ref_x)
        y_double_prime = -1.5 * k ** 2 * np.sin(k * ref_x)
        ref_kappa = y_double_prime / ((1 + y_prime ** 2) ** 1.5)

        # --- 2. 前馈转向 (Feedforward Steering) ---
        # 阿克曼角 + 动力学修正
        ackermann_angle_rad = ref_kappa * L
        steer_ff_rad = ackermann_angle_rad * (1 + K_us * vx ** 2)
        steer_ff_deg = steer_ff_rad * 180 / np.pi

        # --- 3. 基础控制 (PID 反馈) ---
        drive_torque = lon_pid.get_cmd(target_vx - vx, dt)

        # 计算航向误差
        head_error = ref_psi - psi
        while head_error > np.pi: head_error -= 2 * np.pi
        while head_error < -np.pi: head_error += 2 * np.pi

        k_lat = 4.0  # 侧向误差权重
        k_phi = 4.0  # 航向误差权重

        fb_error = -k_lat * lat_error + k_phi * head_error
        steer_fb_deg = lat_pid.get_cmd(fb_error, dt)
        # 总转向指令
        steer_cmd_deg = steer_ff_deg + steer_fb_deg

        # --- 4. NDO 观测 ---
        delta_rad = steer_cmd_deg * np.pi / 180.0
        dist_yaw_moment = ndo.update(vx, vy, r, delta_rad)

        # --- 5. 前馈补偿 ---
        diff_torque = 0.0
        if enable_ndo:
            Rw = 0.35
            T_diff_req = -dist_yaw_moment * 2 * Rw / veh_dyn.lw
            # 适当放宽限幅，因为纠正弯道中的扰动需要更大力矩
            T_diff_req = np.clip(T_diff_req, -2500, 2500)
            diff_torque = T_diff_req

        # --- 6. 执行 ---
        T_fl = drive_torque - diff_torque / 2.0
        T_fr = drive_torque + diff_torque / 2.0
        T_rl = drive_torque - diff_torque / 2.0
        T_rr = drive_torque + diff_torque / 2.0

        action = np.array([T_fl, T_fr, T_rl, T_rr, steer_cmd_deg])

        try:
            next_state, _, done, _ = env.step(action)
        except:
            break

        data_records.append({
            "time": i * dt,
            "x": x, "y": y,
            "y_ref": ref_y,  # 记录期望Y
            "vx": vx, "r": r, "phi": phi, "beta": beta,
            "dist_est": dist_yaw_moment,
            "diff_torque": diff_torque,
            "lat_err": lat_error  # 记录真实的侧向误差
        })

        state = next_state
        if done: break

    env.close()

    df = pd.DataFrame(data_records)
    filename = f"data_{mode_name.lower()}.csv"
    filepath = os.path.join(SAVE_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"Data saved: {filepath}")

# ================= 4. 美化绘图函数 (核心修改) =================
def plot_comparison_chart(time_seq, data_base, data_prop, label_y, title, filename, save_dir):
    """
    通用对比绘图函数：绘制 Baseline (虚线) vs Proposed (实线)
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=cm2inch(default_cfg["fig_size"]), dpi=default_cfg["dpi"], constrained_layout=True)

    # 绘制 Baseline (PID Only)
    ax.plot(time_seq, data_base,
            color='gray', linestyle='--', linewidth=default_cfg["line_width"] * 0.8,
            label='CTMPC', alpha=0.8)

    # 绘制 Proposed (PID + NDO)
    ax.plot(time_seq, data_prop,
            color=SINGLE_LINE_COLOR, linestyle='-', linewidth=default_cfg["line_width"],
            label='CTMPC+NDO-DRC')

    # 轴设置
    ax.tick_params(axis='both', which='major', labelsize=default_cfg["tick_size"], direction='in', width=1.0, length=6)
    ax.grid(False)  # 无网格
    ax.set_xlim(left=0, right=time_seq.max())

    # 标签
    ax.set_xlabel(r"时间 $t\ (\mathrm{s})$", fontproperties=zhfont1, fontsize=default_cfg["label_size"])
    ax.set_ylabel(label_y, fontproperties=zhfont1, fontsize=default_cfg["label_size"])
    # ax.set_title(title, fontproperties=zhfont1, fontsize=default_cfg["label_size"])

    # 图例
    ax.legend(loc="best", prop=get_font_prop(default_cfg["legend_size"]), frameon=False)

    save_path = os.path.join(save_dir, filename + "." + default_cfg["img_fmt"])
    plt.savefig(save_path, format=default_cfg["img_fmt"])
    plt.close()
    print(f"Figure saved: {save_path}")

def plot_trajectory_comparison(x_base, y_base, x_prop, y_prop, save_dir):
    """ 轨迹对比图 """
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=cm2inch(default_cfg["fig_size"]), dpi=default_cfg["dpi"], constrained_layout=True)

    ax.plot(x_base, y_base, color='gray', linestyle='--', linewidth=default_cfg["line_width"], label='Baseline')
    ax.plot(x_prop, y_prop, color=SINGLE_LINE_COLOR, linestyle='-', linewidth=default_cfg["line_width"],
            label='Proposed')

    ax.tick_params(axis='both', which='major', labelsize=default_cfg["tick_size"], direction='in')
    ax.set_xlabel(r"纵向位置 $x\ (\mathrm{m})$", fontproperties=zhfont1, fontsize=default_cfg["label_size"])
    ax.set_ylabel(r"横向位置 $y\ (\mathrm{m})$", fontproperties=zhfont1, fontsize=default_cfg["label_size"])
    ax.axis('equal')
    ax.set_xlim(left=0)

    # 局部放大图 (可选，展示撞击瞬间的偏差)
    # axins = ax.inset_axes([0.6, 0.6, 0.3, 0.3])
    # ...

    ax.legend(loc="best", prop=get_font_prop(default_cfg["legend_size"]), frameon=False)

    plt.savefig(os.path.join(save_dir, "Comp_Trajectory.png"), format="png")
    plt.close()

def plot_single_curve(time_seq, data, label_y, color, filename, save_dir):
    """ 绘制单条曲线 (用于展示NDO观测值) """
    fig, ax = plt.subplots(figsize=cm2inch(default_cfg["fig_size"]), dpi=default_cfg["dpi"], constrained_layout=True)
    ax.plot(time_seq, data, color=color, linewidth=default_cfg["line_width"])
    ax.set_xlabel(r"时间 $t\ (\mathrm{s})$", fontproperties=zhfont1, fontsize=default_cfg["label_size"])
    ax.set_ylabel(label_y, fontproperties=zhfont1, fontsize=default_cfg["label_size"])
    ax.set_xlim(left=0, right=time_seq.max())
    ax.tick_params(axis='both', which='major', labelsize=default_cfg["tick_size"], direction='in')
    plt.savefig(os.path.join(save_dir, filename + ".png"), format="png")
    plt.close()

def process_plotting():
    print("Generating Comparison Plots...")

    try:
        df_base = pd.read_csv(os.path.join(SAVE_DIR, "data_baseline.csv"))
        df_prop = pd.read_csv(os.path.join(SAVE_DIR, "data_proposed.csv"))
    except FileNotFoundError:
        print("Error: CSV files not found. Run simulations first.")
        return

    # 对齐数据长度
    min_len = min(len(df_base), len(df_prop))
    df_base = df_base.iloc[:min_len]
    df_prop = df_prop.iloc[:min_len]
    t = df_base["time"].values

    # 1. 侧向位移对比 (验证PID积分项和前馈效果)
    plot_comparison_chart(t, df_base["y"], df_prop["y"],
                          r"侧向位移 $y\ (\mathrm{m})$", "Lateral Error Comparison", "Comp_Lateral_Error", SAVE_DIR)

    # 2. 横摆角速度对比
    plot_comparison_chart(t, df_base["r"] , df_prop["r"] ,
                          r"横摆角速度 $\dot{\phi}\ (\mathrm{rad/s})$", "Yaw Rate Comparison", "Comp_Yaw_Rate", SAVE_DIR)

    # 3. 侧倾角对比
    plot_comparison_chart(t, df_base["phi"] , df_prop["phi"] ,
                          r"侧倾角 $\varphi\ (\mathrm{rad})$", "Roll Angle Comparison", "Comp_Roll_Angle", SAVE_DIR)

    # 4. 质心侧偏角对比
    plot_comparison_chart(t, df_base["beta"] , df_prop["beta"] ,
                          r"质心侧偏角 $\beta\ (\mathrm{rad})$", "Sideslip Comparison", "Comp_Sideslip", SAVE_DIR)

    # 5. 轨迹对比
    plot_trajectory_comparison(df_base["x"], df_base["y"], df_prop["x"], df_prop["y"], SAVE_DIR)

    # 6. NDO 观测值展示 (仅展示 Proposed)
    plot_single_curve(t, df_prop["dist_est"], r"观测扰动 $M_{dist}\ (\mathrm{N\cdot m})$", 'blue', "NDO_Est", SAVE_DIR)

    # 7. 补偿力矩展示
    plot_single_curve(t, df_prop["diff_torque"], r"前馈力矩 $\Delta T\ (\mathrm{N\cdot m})$", 'green', "Feedforward_Torque",
                      SAVE_DIR)


# ================= 5. 主入口 =================

if __name__ == '__main__':
    # ================= 0. 运行模式配置 =================
    # 步骤1: 改为 'RUN_BASELINE', 在CarSim点Send, 运行脚本
    # 步骤2: 改为 'RUN_PROPOSED', 在CarSim点Send(重置), 运行脚本
    # 步骤3: 改为 'PLOT_ONLY', 运行脚本出图
    EXECUTION_MODE = 'RUN_BASELINE'

    if EXECUTION_MODE == 'RUN_BASELINE':
        run_single_simulation("BASELINE", enable_ndo=False)
        print("\n>>> 请重置 CarSim 求解器，然后修改代码为 RUN_PROPOSED")

    elif EXECUTION_MODE == 'RUN_PROPOSED':
        run_single_simulation("PROPOSED", enable_ndo=True)
        print("\n>>> 仿真完成，请修改代码为 PLOT_ONLY 生成对比图")

    elif EXECUTION_MODE == 'PLOT_ONLY':
        process_plotting()

    else:
        print("Invalid Mode")