import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import seaborn as sns
import pandas as pd
import os
import gym
import matplotlib.patches as patches
import copy  # 必须导入 copy 模块

# ================= 1. 全局配置与美化 =================

# 1.1 字体与Latex设置
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['axes.unicode_minus'] = False

# 尝试加载中文字体
try:
    zhfont1 = fm.FontProperties(fname='../../gops/utils/SIMSUN.ttf', size=16)
except:
    zhfont1 = fm.FontProperties(family='SimHei', size=16)

# 1.2 配色与线型配置
SINGLE_LINE_COLOR = '#D62728'
SINGLE_LINE_STYLE = '-'
MULTI_COLORS = ['blue', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b']
MULTI_STYLES = ['-', '--', '-.', ':']

default_cfg = {
    "fig_size": (12, 9),
    "dpi": 300,
    "tick_size": 18,
    "label_size": 18,
    "legend_size": 12,  # 统一图例字体大小，您可以修改此值 (例如 10 或 12)
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
    """ 创建指定大小的字体属性副本，解决prop覆盖fontsize的问题 """
    prop = copy.copy(zhfont1)
    prop.set_size(size)
    return prop


def plot_line_chart(time_seq, data_seq, label, title_y, file_name, save_root, y_limit=None):
    # Plot line chart with smart Y-axis and 0-aligned X-axis
    os.makedirs(save_root, exist_ok=True)
    t = np.array(time_seq)
    fig, ax = plt.subplots(figsize=cm2inch(default_cfg["fig_size"]), dpi=default_cfg["dpi"], constrained_layout=True)

    is_multi_line = False
    if isinstance(data_seq, list) and len(data_seq) > 0 and isinstance(data_seq[0], (np.ndarray, list)):
        is_multi_line = True
        all_data = np.concatenate(data_seq)
        for i, data in enumerate(data_seq):
            lbl = label[i] if isinstance(label, list) else label
            c = MULTI_COLORS[i % len(MULTI_COLORS)]
            s = MULTI_STYLES[i % len(MULTI_STYLES)]
            sns.lineplot(x=t, y=data, linewidth=default_cfg["line_width"], color=c, linestyle=s, label=lbl, ax=ax)
    else:
        if isinstance(data_seq, list): data_seq = np.array(data_seq)
        all_data = data_seq
        sns.lineplot(x=t, y=data_seq, linewidth=default_cfg["line_width"], color=SINGLE_LINE_COLOR,
                     linestyle=SINGLE_LINE_STYLE, label=None, ax=ax)

    # Axis settings
    ax.tick_params(axis='both', which='major', labelsize=default_cfg["tick_size"], direction='in', width=1.0, length=6)
    ax.grid(False)

    # X-axis: Start from 0
    ax.set_xlim(left=0, right=t.max())

    # # Y-axis Smart Setting
    # if y_limit:
    #     ax.set_ylim(y_limit)
    # else:
    #     d_min, d_max = np.min(all_data), np.max(all_data)
    #     if np.isclose(d_min, d_max):
    #         # Handle constant or near-constant data
    #         margin = 1.0 if d_max == 0 else abs(d_max) * 0.2
    #         ax.set_ylim(d_min - margin, d_max + margin)
    #     else:
    #         # Dynamic range
    #         if d_min >= 0: ax.set_ylim(bottom=0)  # If data is all positive, anchor to 0
    #         ax.margins(x=0, y=0.1)

    ax.set_xlabel(r"时间 $t\ (\mathrm{s})$", fontproperties=zhfont1, fontsize=default_cfg["label_size"])
    ax.set_ylabel(title_y, fontproperties=zhfont1, fontsize=default_cfg["label_size"])

    if is_multi_line and ax.get_legend():
        ax.legend(loc="best", prop=get_font_prop(default_cfg["legend_size"]), frameon=False)

    save_path = os.path.join(save_root, file_name + "." + default_cfg["img_fmt"])
    plt.savefig(save_path, format=default_cfg["img_fmt"])
    plt.close()
    print(f"Figure saved: {save_path}")


def plot_trajectory(x_seq, y_seq, file_name, save_root):
    """ 轨迹图 (无网格) """
    os.makedirs(save_root, exist_ok=True)
    fig, ax = plt.subplots(figsize=cm2inch(default_cfg["fig_size"]), dpi=default_cfg["dpi"], constrained_layout=True)
    ax.plot(x_seq, y_seq, linewidth=default_cfg["line_width"], color=SINGLE_LINE_COLOR, linestyle=SINGLE_LINE_STYLE)

    ax.tick_params(axis='both', which='major', labelsize=default_cfg["tick_size"], direction='in')
    ax.grid(False)

    ax.set_xlabel(r"纵向位置 $x\ (\mathrm{m})$", fontproperties=zhfont1, fontsize=default_cfg["label_size"])
    ax.set_ylabel(r"横向位置 $y\ (\mathrm{m})$", fontproperties=zhfont1, fontsize=default_cfg["label_size"])
    ax.axis('equal')
    # X-axis: Start from 0
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    save_path = os.path.join(save_root, file_name + "." + default_cfg["img_fmt"])
    plt.savefig(save_path, format=default_cfg["img_fmt"])
    plt.close()
    print(f"Trajectory saved: {save_path}")


def plot_friction_circle(df, file_name, save_root):
    """ 绘制摩擦圆 (统一图例，无网格) """
    os.makedirs(save_root, exist_ok=True)
    fig, ax = plt.subplots(figsize=cm2inch(default_cfg["fig_size"]), dpi=default_cfg["dpi"], constrained_layout=True)

    # 绘制单位圆
    circle = patches.Circle((0, 0), 1.0, fill=False, edgecolor='black', linestyle='--', linewidth=1.5)
    ax.add_patch(circle)

    # 映射中文标签到 DataFrame 列后缀
    wheels_map = [
        ('左前轮', 'fl'),
        ('右前轮', 'fr'),
        ('左后轮', 'rl'),
        ('右后轮', 'rr')
    ]
    colors = ['r', 'g', 'b', 'orange']

    # 降采样
    step = max(1, len(df) // 500)

    for i, (label_cn, suffix) in enumerate(wheels_map):
        # 确保列名存在
        key_fx = f"Fx_{suffix}"
        key_fy = f"Fy_{suffix}"
        key_fz = f"Fz_{suffix}"

        if key_fx not in df.columns:
            print(f"Warning: Column {key_fx} not found. Skipping {label_cn}.")
            continue

        fx = df[key_fx].values[::step]
        fy = df[key_fy].values[::step]
        fz = df[key_fz].values[::step]
        mu = df["mu"].values[::step]

        # 归一化: F / (mu * Fz)
        norm_fx = fx / (mu * fz + 1e-5)
        norm_fy = fy / (mu * fz + 1e-5)

        ax.scatter(norm_fx, norm_fy, s=15, alpha=0.6, color=colors[i], label=label_cn)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')

    ax.set_xlabel(r"$F_x / (\mu F_z)$", fontproperties=zhfont1, fontsize=default_cfg["label_size"])
    ax.set_ylabel(r"$F_y / (\mu F_z)$", fontproperties=zhfont1, fontsize=default_cfg["label_size"])

    # 图例设置: 使用 get_font_prop 确保大小生效
    ax.legend(loc='best', prop=get_font_prop(default_cfg["legend_size"]), frameon=False, ncol=2)

    ax.tick_params(axis='both', which='major', labelsize=default_cfg["tick_size"], direction='in', width=1.0, length=6)
    ax.grid(False)

    save_path = os.path.join(save_root, file_name + "." + default_cfg["img_fmt"])
    plt.savefig(save_path, format=default_cfg["img_fmt"])
    plt.close()
    print(f"Friction Circle saved: {save_path}")


def plot_phase_portrait(x_data, y_data, x_label, y_label, file_name, save_root):
    # Phase Portrait Plot
    os.makedirs(save_root, exist_ok=True)
    fig, ax = plt.subplots(figsize=cm2inch(default_cfg["fig_size"]), dpi=default_cfg["dpi"], constrained_layout=True)

    # Plot Trajectory
    ax.plot(x_data, y_data, linewidth=default_cfg["line_width"], color='purple', linestyle='-')

    # Mark Start and End
    ax.scatter(x_data[0], y_data[0], color='green', s=60, marker='o', label='起点', zorder=5)
    ax.scatter(x_data[-1], y_data[-1], color='red', s=60, marker='x', label='终点', zorder=5)

    ax.tick_params(axis='both', which='major', labelsize=default_cfg["tick_size"], direction='in')
    ax.grid(False)

    ax.set_xlabel(x_label, fontproperties=zhfont1, fontsize=default_cfg["label_size"])
    ax.set_ylabel(y_label, fontproperties=zhfont1, fontsize=default_cfg["label_size"])

    # Zero lines
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')

    ax.legend(loc='best', prop=get_font_prop(default_cfg["legend_size"]), frameon=False)

    save_path = os.path.join(save_root, file_name + "." + default_cfg["img_fmt"])
    plt.savefig(save_path, format=default_cfg["img_fmt"])
    plt.close()
    print(f"Phase Portrait saved: {save_path}")

# ================= 2. 核心计算逻辑 (保持不变) =================

class VehicleDynamics:
    def __init__(self):
        self.m = 2060.0
        self.ms = 1836.2
        self.mu_mass = (self.m - self.ms) / 4.0
        self.g = 9.81
        self.lw = 1.8
        self.lf = 1.4442
        self.lr = 1.5558
        self.L = self.lf + self.lr
        self.hs = 0.75
        self.hr = 0.5
        self.hu = 0.25
        self.Izz = 3524.9
        self.Ixx = 846.6
        self.K_phi = 28000.0
        self.C_phi = 9000.0

    def calc_I_xy(self, Fx, Fy, Fz, mu):
        gamma_list = []
        for i in range(4):
            if Fz[i] <= 10.0:
                gamma_list.append(0.0)
                continue
            F_total = np.sqrt(Fx[i] ** 2 + Fy[i] ** 2)
            capacity = mu * Fz[i]
            gamma = F_total / (capacity + 1e-5)
            gamma_list.append(gamma)
        return min(max(gamma_list), 1.0)

    def calc_I_phi(self, vx, r, r_dot, delta, mu, Fz_total):
        K_stability = 0.002
        r_des = (vx * delta) / (self.L * (1 + K_stability * vx ** 2) + 1e-5)
        E_phi = 0.5 * self.Izz * r ** 2
        E_phi_des = 0.5 * self.Izz * r_des ** 2
        delta_E = E_phi - E_phi_des
        safe_vx = max(abs(vx), 1.0)
        max_lat_acc = mu * self.g
        max_yaw_rate = max_lat_acc / safe_vx
        E_norm = 0.5 * self.Izz * max_yaw_rate ** 2
        E_dot = self.Izz * r * r_dot
        P_max = (mu * Fz_total * self.lw / 2) * abs(r) + 1e-1
        kappa_coef = 0.5
        term1 = abs(delta_E) / (E_norm + 1e-5)
        term2 = kappa_coef * np.sign(delta_E) * (E_dot / P_max)
        I_phi = term1 + term2
        return np.clip(I_phi, 0.0, 1.0)

    def calc_I_rs(self, phi, phi_dot, slope_lat, slope_lon):
        phi_rel = phi - slope_lat
        mass_ratio = (self.ms * self.hr + self.mu_mass * self.hu) / (self.ms * self.hs)
        cos_theta = np.cos(slope_lon)
        cos_phi = np.cos(slope_lat)
        denom = self.m * self.g * self.lw * cos_phi * cos_theta + 1e-5
        term1 = (2 * self.K_phi) / denom * (1 + mass_ratio)
        term2 = (2 * (self.ms * self.hr + self.mu_mass * self.hu) * self.g * cos_phi) / denom
        I_phi_coef = term1 - term2
        I_dot_phi_coef = (2 * self.C_phi) / denom * (1 + mass_ratio)
        I_rs_raw = I_phi_coef * phi_rel + I_dot_phi_coef * phi_dot
        return np.clip(I_rs_raw, -1.0, 1.0)

    def calc_I_a(self, I_xy, I_rs, I_phi):
        ws, wr, wy = 1.0, 1.0, 0.5
        lam = 0.5
        risk_slip = I_xy
        risk_roll = abs(I_rs)
        risk_yaw = I_phi
        val = (ws * risk_slip) ** 2 + \
              (wr * risk_roll) ** 2 + \
              (wy * risk_yaw) ** 2 + \
              lam * (risk_slip * risk_roll)
        return np.sqrt(val)


class PID:
    def __init__(self, kp, ki, kd, cmd_max, cmd_min):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.ep = self.ei = self.ed = 0.0
        self.cmd_max, self.cmd_min = cmd_max, cmd_min

    def get_cmd(self, error):
        self.ed = error - self.ep
        self.ei += error
        self.ep = error
        cmd = self.kp * self.ep + self.ki * self.ei + self.kd * self.ed
        return np.clip(cmd, self.cmd_min, self.cmd_max)


# ================= 3. 仿真与绘图流程 =================

def run_simulation_carsim(env_id, run_step=30000, delta_t=0.01):
    print(f"Connecting to CarSim Env: {env_id}...")
    longi_controller = PID(kp=400, ki=0.5, kd=10, cmd_max=1000, cmd_min=-1000)

    try:
        model_mechnical = gym.make(env_id, disable_env_checker=True)
    except:
        print("Error: Cannot load gym environment. Ensure 'pyth_stabilitycontrol' is installed.")
        return pd.DataFrame()

    veh_dyn = VehicleDynamics()
    state, _ = model_mechnical.reset()

    data_records = []
    refer_v = 80/3.6
    last_r = state[5]

    for i in range(run_step):
        current_vx = state[3]
        drive_torque = longi_controller.get_cmd(refer_v - current_vx)

        steering_angle_degree = 0
        # if 200 < i < 1000:
        #     steering_angle_degree = 60 * np.sin(2 * np.pi * (i - 200) / 800)

        control_carsim = np.array([drive_torque, drive_torque, drive_torque, drive_torque, steering_angle_degree])
        next_state, _, _, _ = model_mechnical.step(control_carsim)

        x, y, psi = next_state[0], next_state[1], next_state[2]
        vx, vy, r = next_state[3], next_state[4], next_state[5]
        phi, phi_dot = next_state[6], next_state[7]
        delta_f = next_state[12] / 18.0
        beta = next_state[17]
        slope_lon, slope_lat = next_state[19], next_state[20]

        Fx = next_state[21:25]
        Fy = next_state[25:29]
        Fz = next_state[29:33]
        mu = next_state[33]

        r_dot = (r - last_r) / delta_t
        last_r = r

        I_xy = veh_dyn.calc_I_xy(Fx, Fy, Fz, mu)
        I_phi = veh_dyn.calc_I_phi(vx, r, r_dot, delta_f, mu, sum(Fz))
        I_rs = veh_dyn.calc_I_rs(phi, phi_dot, slope_lat, slope_lon)
        I_a = veh_dyn.calc_I_a(I_xy, I_rs, I_phi)

        record = {
            "time": i * delta_t,
            "vx": vx, "vy": vy, "r": r, "phi": phi, "psi": psi, "phi_dot": phi_dot, "beta": beta,
            "x": x, "y": y,
            "I_xy": I_xy, "I_phi": I_phi, "I_rs": I_rs, "I_a": I_a,
            "slope_lon": slope_lon, "slope_lat": slope_lat,
            "mu": mu,
            "Fx_fl": Fx[0], "Fy_fl": Fy[0], "Fz_fl": Fz[0],
            "Fx_fr": Fx[1], "Fy_fr": Fy[1], "Fz_fr": Fz[1],
            "Fx_rl": Fx[2], "Fy_rl": Fy[2], "Fz_rl": Fz[2],
            "Fx_rr": Fx[3], "Fy_rr": Fy[3], "Fz_rr": Fz[3],
        }
        data_records.append(record)
        state = next_state

    df = pd.DataFrame(data_records)
    print("Simulation Finished.")
    return df


def save_data_to_csv(df, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, "simulation_data.csv")
    df.to_csv(file_path, index=False)
    print(f"Data saved to: {file_path}")


def load_data_from_csv(save_dir):
    file_path = os.path.join(save_dir, "simulation_data.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find data file: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Data loaded from: {file_path}")
    return df


def plot_all_figures(df, save_dir):
    """ 生成所有图表 """
    time_seq = df["time"].values

    # 1. 轨迹图
    plot_trajectory(df["x"].values, df["y"].values, "State_Trajectory", save_dir)
    # 2. Slopes (New) - in Degrees
    plot_line_chart(time_seq, [df["slope_lon"].values, df["slope_lat"].values],
                    [r"纵坡 $\theta_\mathrm{r}$", r"横坡 $\varphi_\mathrm{r}$"],
                    r"坡度 $(\mathrm{rad})$", "Env_Slopes", save_dir)
    # 2. 状态图 (Label Latex 修正)
    plot_line_chart(time_seq, df["vx"].values, "Longitudinal Speed",
                    r"纵向速度 $v_x\ (\mathrm{m/s})$", "State_Vx", save_dir)

    # 新增: 角度图 (侧倾角 phi & 横摆角 psi)
    plot_line_chart(time_seq, [df["phi"].values, df["psi"].values],
                    [r"侧倾角 $\varphi$", r"横摆角 $\phi$"],
                    r"角度 $(\mathrm{rad})$", "State_Angles", save_dir)

    # 状态：横摆角速度 (r) & 侧倾角速度 (phi_dot)
    plot_line_chart(time_seq, [df["r"].values, df["phi_dot"].values],
                    [r"横摆角速度 $\dot{\phi}$", r"侧倾角速度 $\dot{\varphi}$"],
                    r"角速度 $(\mathrm{rad/s})$", "State_AngleRates", save_dir)

    # 3. 风险指标 (Label Latex 正体修正, Y轴范围微调)
    # (a) I_xy
    plot_line_chart(time_seq, df["I_xy"].values,
                    r"Tire Force Utilization",
                    r"平面预警 $I_{xy}$", "Risk_I_xy", save_dir, y_limit=(-0.1, 1.1))

    # (b) I_phi
    plot_line_chart(time_seq, df["I_phi"].values,
                    r"Yaw Incoordination",
                    r"横摆预警 $I_{\phi}$", "Risk_I_phi", save_dir, y_limit=(-0.1, 1.1))

    # (c) I_rs
    plot_line_chart(time_seq, df["I_rs"].values,
                    r"Rollover Index",
                    r"侧倾预警 $I_{\mathrm{rs}}$", "Risk_I_rs", save_dir, y_limit=(-1.1, 1.1))

    # (d) I_a
    plot_line_chart(time_seq, df["I_a"].values,
                    r"Comprehensive Risk",
                    r"综合预警 $I_{\mathrm{a}}$", "Risk_I_a", save_dir)

    # 4. 摩擦圆图
    plot_friction_circle(df, "Friction_Circle", save_dir)

    # 4. Phase Portrait (New) - Beta vs Yaw Rate
    # Classic plot to show instability
    plot_phase_portrait(df["beta"].values, df["r"].values,
                        r"质心侧偏角 $\beta\ (\mathrm{rad})$", r"横摆角速度 $\dot{\phi}\ (\mathrm{rad/s})$",
                        "PhasePortrait_Beta_YawRate", save_dir)

if __name__ == '__main__':
    # 切换数据源: 'CARSIM' (需要环境) 或 'CSV' (读取已有数据)
    DATA_SOURCE = 'CARSIM'
    save_dir = "./offroad_results_fishhook"
    if DATA_SOURCE == 'CARSIM':
        df_result = run_simulation_carsim(env_id='pyth_stabilitycontrol')
        if not df_result.empty:
            save_data_to_csv(df_result, save_dir)
            plot_all_figures(df_result, save_dir)

    elif DATA_SOURCE == 'CSV':
        try:
            df_result = load_data_from_csv(save_dir)
            plot_all_figures(df_result, save_dir)
        except Exception as e:
            print(f"Error: {e}")
            print("请先运行 CARSIM 模式生成数据，或将含有 Fx/Fy/Fz 的 CSV 文件放入目录。")