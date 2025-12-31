import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
import os
import matplotlib as mpl
import matplotlib.font_manager as fm
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.ticker import ScalarFormatter
import copy  # 用于复制字典

zhfont1 = fm.FontProperties(fname='../gops/utils/SIMSUN.ttf')
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
default_cfg["img_fmt"] = "pdf"
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定宋体
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# ================= 1. Configuration and Parameters =================
vehicle_params = {
    'm': 2568.4, 'g': 9.8, 'Iz': 3524.9,
    'a': 1.33, 'b': 1.81, 'width': 1.725, 'h': 0.75,
    'ms': 2568.4 * 0.9, 'hs': 0.45,
    'Ixx': 1500.0, 'K_phi': 180000, 'C_phi': 8000
}
vehicle_params['L'] = vehicle_params['a'] + vehicle_params['b']

OUTPUT_DIR = 'dissipated_energy_surface_plots2'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def scientific_colorbar(fig, ax, mappable, label, label_fontsize=15, shrink=True):
    """
    创建并格式化颜色条，使用科学计数法，并正确设置中文标签字体。
    """
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))  # 强制科学计数法

    # 1. 创建颜色条，不带字体参数
    if shrink:
        cbar = fig.colorbar(mappable, ax=ax, format=formatter, shrink=0.7, aspect=15, pad=0.1)
    else:
        cbar = fig.colorbar(mappable, ax=ax, format=formatter)

    # 2. 设置刻度字体格式 (使用 default_cfg["tick_size"])
    cbar.ax.yaxis.set_major_formatter(formatter)
    cbar.ax.tick_params(labelsize=default_cfg["tick_size"])

    # 3. 设置标签和字体
    cbar.set_label(label, fontproperties=zhfont1, fontsize=label_fontsize)
    return cbar

# [Note: The tire_pacejka and vehicle_dynamics functions from your previous response are used here.]
def tire_pacejka(alpha, Fz, mu):
    Cy, a1y, a2y = 1.3, -22.1, 1011
    a3y, a4y, a5y = 1078, 1.82, 0.208
    a6y, a7y, a8y = 0, -0.354, 0.707
    Fz_kN = Fz / 1000.0
    if Fz_kN <= 0: return 0.0
    alpha_deg = np.degrees(alpha)
    D = a1y * Fz_kN ** 2 + a2y * Fz_kN
    if D == 0: return 0.0
    BCD = a3y * np.sin(a4y * np.arctan(a5y * Fz_kN))
    B = BCD / (Cy * D)
    E = a6y * Fz_kN ** 2 + a7y * Fz_kN + a8y
    phi_mf = (1.25 - 0.25 * mu) * Cy * np.arctan(
        (2 - mu) * B * alpha_deg - E * ((2 - mu) * B * alpha_deg - np.arctan((2 - mu) * B * alpha_deg)))
    Fy = - (mu * D * np.sin(phi_mf))
    return Fy

def vehicle_dynamics(t, state, u_input, env_params, vp):
    v_y, r, roll, roll_rate = state
    v_x = env_params['v_x']
    delta_f = u_input['delta_f']
    mu = env_params['mu']
    theta_r = env_params['theta_r']
    phi_r = env_params['phi_r']
    m, g, Iz, Ixx = vp['m'], vp['g'], vp['Iz'], vp['Ixx']
    a, b, L, width = vp['a'], vp['b'], vp['L'], vp['width']
    ms, hs, K_phi, C_phi = vp['ms'], vp['hs'], vp['K_phi'], vp['C_phi']

    dFz = (K_phi * roll + C_phi * roll_rate) / width
    Fz_static_f = 0.5 * m * g * (b / L) * np.cos(theta_r) * np.cos(phi_r) - 0.5 * m * g * (vp['h'] / L) * np.sin(
        theta_r)
    Fz_static_r = 0.5 * m * g * (a / L) * np.cos(theta_r) * np.cos(phi_r) + 0.5 * m * g * (vp['h'] / L) * np.sin(
        theta_r)
    Fz_lat = 0.5 * m * g * (vp['h'] / width) * np.sin(phi_r)

    Fz_fl = Fz_static_f - dFz * (b / L) - Fz_lat
    Fz_fr = Fz_static_f + dFz * (b / L) + Fz_lat
    Fz_rl = Fz_static_r - dFz * (a / L) - Fz_lat
    Fz_rr = Fz_static_r + dFz * (a / L) + Fz_lat

    vx_safe = max(v_x, 0.1)
    alpha_f = np.arctan2(v_y + a * r, vx_safe) - delta_f
    alpha_r = np.arctan2(v_y - b * r, vx_safe)

    Fy_f = tire_pacejka(alpha_f, Fz_fl, mu) + tire_pacejka(alpha_f, Fz_fr, mu)
    Fy_r = tire_pacejka(alpha_r, Fz_rl, mu) + tire_pacejka(alpha_r, Fz_rr, mu)

    v_y_dot = (Fy_f * np.cos(delta_f) + Fy_r) / m - v_x * r + g * np.sin(phi_r) * np.cos(roll)
    r_dot = (a * Fy_f * np.cos(delta_f) - b * Fy_r) / Iz
    roll_moment = (Fy_f + Fy_r) * hs + ms * g * hs * np.sin(roll + phi_r) - K_phi * roll - C_phi * roll_rate
    roll_rate_dot = roll_moment / Ixx

    return [v_y_dot, r_dot, roll_rate, roll_rate_dot]

# ================= 4. Energy Calculation =================
def calculate_total_energy(state, v_x, vp):
    v_y, r, roll, roll_rate = state
    E_k = 0.5 * vp['m'] * (v_x ** 2 + v_y ** 2) + 0.5 * vp['Iz'] * r ** 2 + 0.5 * vp['Ixx'] * roll_rate ** 2
    E_p = 0.5 * vp['K_phi'] * roll ** 2
    return E_k + E_p

def get_dissipated_energy(state_0, env_params, u_input, vp, T_sim=2.0):
    v_x = env_params['v_x']
    E_0 = calculate_total_energy(state_0, v_x, vp)

    sol = solve_ivp(vehicle_dynamics, [0, T_sim], state_0,
                    args=(u_input, env_params, vp), method='RK45', rtol=1e-5, atol=1e-6)

    state_final = sol.y[:, -1]
    E_n = calculate_total_energy(state_final, v_x, vp)

    return E_0 - E_n

# ================= 5. Plotting Function (Surface Style) =================
# --- 2D Pairwise Plots (Vy, YawRate, RollRate) ---
# --- 2D Projection Plot (Requirement 3) ---
def plot_2d_projection(vx_mps, mu_val):
    print("正在生成 2D Projection Plot (Vy vs YawRate)...")
    vx = vx_mps
    N = 50
    vy_vals = np.linspace(-5, 5, N)
    r_vals = np.linspace(-1.5, 1.5, N)

    X, Y = np.meshgrid(vy_vals, r_vals)
    Z = np.zeros_like(X)  # Dissipated Energy U

    for i in range(N):
        for j in range(N):
            env = {'v_x': vx, 'mu': mu_val, 'theta_r': 0, 'phi_r': 0}
            Z[i, j] = get_dissipated_energy([X[i, j], Y[i, j], 0, 0], env, {'delta_f': 0.0}, vehicle_params)

    fig, ax = plt.subplots(figsize=(8, 6))
    mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    # Contour Plot
    cp = ax.contourf(X, Y, Z, levels=20, cmap='RdYlBu')

    # Colorbar with scientific notation (Requirement 1)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    cbar = fig.colorbar(cp, ax=ax, format=formatter)
    cbar.ax.yaxis.set_major_formatter(formatter)
    cbar.set_label(r'耗散能 $U\ (\mathrm{J})$', fontproperties=zhfont1, fontsize=15)
    plt.tick_params(labelsize=default_cfg["tick_size"])
    # 使用 ax.tick_params 来设置刻度线方向
    ax.tick_params(axis='x', direction='in')  # x轴刻度线向内
    ax.tick_params(axis='y', direction='in')  # y轴刻度线向内
    # ax.set_title(f'2D Dissipated Energy Projection ($\mu$={mu_val}, $V_x$={vx_kmh}km/h)')
    ax.set_xlabel(r'横向速度$v_y\ \mathrm{(m/s)}$', fontproperties=zhfont1, fontsize=20)
    ax.set_ylabel(r'横摆角速度$\dot\phi\ \mathrm{(rad/s)}$', fontproperties=zhfont1, fontsize=20)

    plt.savefig(f'{OUTPUT_DIR}/2d_projection_vy_r_vx_{vx_mps}mps.pdf', dpi=300)
    plt.close()

# --- 2D Projection Plot: Vy vs R (Requirement 3) ---
def plot_2d_vy_r_projection(vx_mps, mu_val):
    print("正在生成 2D Projection Plot (Vy vs YawRate)...")
    vx = vx_mps
    vx_kmh = vx * 3.6
    N = 50
    vy_vals = np.linspace(-3, 3, N)
    r_vals = np.linspace(-1.5, 1.5, N)

    X, Y = np.meshgrid(vy_vals, r_vals)
    Z = np.zeros_like(X)

    for i in range(N):
        for j in range(N):
            env = {'v_x': vx, 'mu': mu_val, 'theta_r': 0, 'phi_r': 0}
            Z[i, j] = get_dissipated_energy([X[i, j], Y[i, j], 0, 0], env, {'delta_f': 0.0}, vehicle_params)

    fig, ax = plt.subplots(figsize=(8, 6))
    cp = ax.contourf(X, Y, Z, levels=20, cmap='RdYlBu')

    # 刻度设置
    ax.tick_params(labelsize=default_cfg["tick_size"])
    ax.tick_params(axis='x', direction='in')
    ax.tick_params(axis='y', direction='in')

    # Colorbar with scientific notation
    scientific_colorbar(fig, ax, cp, r'耗散能 $U\ (\mathrm{J})$',shrink=False)

    # ax.set_title(f'2D 耗散能投影 ($\mu$={mu_val}, $V_x$={vx_kmh:.1f}km/h)', fontproperties=zhfont1, fontsize=20)
    ax.set_xlabel(r'横向速度$v_y\ \mathrm{(m/s)}$', fontproperties=zhfont1, fontsize=20)
    ax.set_ylabel(r'横摆角速度$\dot\phi\ \mathrm{(rad/s)}$', fontproperties=zhfont1, fontsize=20)

    plt.savefig(f'{OUTPUT_DIR}/2d_projection_vy_r_vx_{vx_mps}mps.pdf', dpi=300)
    plt.close()

# --- New Function: Plot Beta vs YawRate (User Request) ---
def plot_2d_beta_r_projection(vx_mps, mu_val):
    print("正在生成 2D Projection Plot (Beta vs YawRate)...")
    vx = vx_mps
    vx_kmh = vx * 3.6
    N = 50

    beta_vals = np.linspace(-0.5, 0.5, N)
    r_vals = np.linspace(-1.5, 1.5, N)

    X_beta, Y_r = np.meshgrid(beta_vals, r_vals)
    Z_U = np.zeros_like(X_beta)

    for i in range(N):
        for j in range(N):
            # 关键：通过小角度近似计算 Vy: vy = beta * vx
            vy_val = X_beta[i, j] * vx

            env = {'v_x': vx, 'mu': mu_val, 'theta_r': 0, 'phi_r': 0}
            Z_U[i, j] = get_dissipated_energy([vy_val, Y_r[i, j], 0, 0], env, {'delta_f': 0.0}, vehicle_params)

    fig, ax = plt.subplots(figsize=(8, 6))

    cp = ax.contourf(X_beta, Y_r, Z_U, levels=20, cmap='RdYlBu')

    # 刻度设置
    ax.tick_params(labelsize=default_cfg["tick_size"])
    ax.tick_params(axis='x', direction='in')
    ax.tick_params(axis='y', direction='in')

    # Colorbar with scientific notation
    scientific_colorbar(fig, ax, cp, r'耗散能 $U\ (\mathrm{J})$',shrink=False)

    # ax.set_title(f'2D 耗散能投影 ($\mu$={mu_val}, $V_x$={vx_kmh:.1f} km/h)', fontproperties=zhfont1, fontsize=20)
    ax.set_xlabel(r'质心侧偏角 $\beta\ (\mathrm{rad})$', fontproperties=zhfont1, fontsize=20)
    ax.set_ylabel(r'横摆角速度 $\dot\phi\ (\mathrm{rad/s})$', fontproperties=zhfont1, fontsize=20)

    plt.savefig(f'{OUTPUT_DIR}/2d_projection_beta_r_vx_{vx_mps}mps.pdf', dpi=300)
    plt.close()

def plot_surface_style(scenario_name, title_suffix, env_base, u_base, fixed_vx_kmh, fixed_state_vals):
    # fixed_state_vals: {'vy':0, 'rr':0} for a fixed value if not used as an axis
    # We will plot (Vx, Vy, YawRate) or (Vx, YawRate, RollRate) on a 3D surface.

    N = 40  # 提高分辨率

    # 确定轴变量和范围
    # 1. Longitudinal Speed (Vx)
    vx_vals = np.linspace(0, 25, N)  # m/s
    # 2. Lateral Speed (Vy)
    vy_vals = np.linspace(-5, 5, N)  # m/s
    # 3. Yaw Rate (R)
    r_vals = np.linspace(-1.5, 1.5, N)  # rad/s
    # 4. Roll Rate (RR)
    rr_vals = np.linspace(-1.0, 1.0, N)  # rad/s

    # --- A. Plot Vx (X) vs Vy (Y) vs YawRate (Color) ---
    # Need to choose 3 variables for X, Y, Color. The provided image uses 3 axes.
    # We will plot X=Vx, Y=Vy, Z=YawRate(R) and use Color for U.
    print(f"  Generating {scenario_name} (Vx, Vy, R)...")

    GVX, GVY, GR = np.meshgrid(vx_vals, vy_vals, r_vals)
    U_vals = np.zeros_like(GVX)

    # Pre-calculate U_vals
    for i in range(N):
        for j in range(N):
            for k in range(N):
                vx = GVX[i, j, k]
                vy = GVY[i, j, k]
                r = GR[i, j, k]

                curr_env = env_base.copy()
                curr_env['v_x'] = vx

                # State: [v_y, r, roll=0, roll_rate=0] (fixed roll state for simplicity)
                st0 = [vy, r, fixed_state_vals.get('roll_rate', 0), fixed_state_vals.get('roll_rate', 0)]
                U_vals[i, j, k] = get_dissipated_energy(st0, curr_env, u_base, vehicle_params)


    # Find maximum U value for color normalization (excluding outliers)
    U_flat = U_vals.flatten()
    U_max_clip = np.percentile(U_flat[U_flat > 0], 98)  # Clip top 2% outliers
    U_min_clip = np.percentile(U_flat, 2)  # Clip bottom 2% outliers (to define min color)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Prepare scatter data for plotting
    X_plot = GVX.flatten()   # Vx in m/s
    Y_plot = GVY.flatten()  # Vy in m/s
    Z_plot = GR.flatten()  # R in rad/s
    C_plot = np.clip(U_flat, U_min_clip, U_max_clip)  # Color clipped

    # Create normalized color map
    cmap = plt.cm.get_cmap('RdYlBu')
    norm = plt.Normalize(C_plot.min(), C_plot.max())
    colors = cmap(norm(C_plot))

    # Plot dense scatter points - This creates the 'volume' illusion
    sc = ax.scatter(X_plot, Y_plot, Z_plot, c=colors, s=10, alpha=0.5, marker='s')  # Use 's' for square markers
    # Colorbar with scientific notation
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(C_plot)
    scientific_colorbar(fig, ax, sm, '耗散能 $U\ (\mathrm{J})$')

    # Labels and Ticks
    # 刻度设置
    ax.tick_params(labelsize=default_cfg["tick_size"])
    ax.tick_params(axis='x', direction='in')
    ax.tick_params(axis='y', direction='in')
    ax.tick_params(axis='z', direction='in')

    # --- Final Polish ---
    ax.set_xlabel(r'纵向速度$v_x\ \mathrm{(m/s)}$', fontproperties=zhfont1, fontsize=20, labelpad=10)
    ax.set_ylabel(r'横向速度$v_y\ \mathrm{(m/s)}$', fontproperties=zhfont1, fontsize=20, labelpad=10)
    ax.set_zlabel(r'横摆角速度$\dot\phi\ \mathrm{(rad/s)}$', fontproperties=zhfont1, fontsize=20, labelpad=10)

    # ax.set_title(f'Dissipated Energy Distribution ({title_suffix})')

    # Create a separate colorbar using the normalized values
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])  # Only needed for the colorbar
    # fig.colorbar(sm, ax=ax, label=r'耗散能 $U\ (\mathrm{J})$', fontproperties=zhfont1, fontsize=20, labelpad=10)

    # Adjust view angle to match reference image (often azimuth=30, elevation=30)
    ax.view_init(elev=20, azim=45)

    plt.savefig(f'{OUTPUT_DIR}/3d_surface_{scenario_name}.pdf', dpi=300)
    plt.close()

# --- 3D Scatter Analysis (General Function) ---
def plot_3d_scatter_analysis(scenario_name, title_suffix, env_base, u_base):
    # Fixed V_x = 30 km/h (Used for Vy, R, RR analysis)
    vx = env_base['v_x']
    N = 40
    vy_vals = np.linspace(-5, 5, N)
    r_vals = np.linspace(-1.5, 1.5, N)
    rr_vals = np.linspace(-1.0, 1.0, N)

    GVY, GR, GRR = np.meshgrid(vy_vals, r_vals, rr_vals)
    U_list = []

    for i in range(N):
        for j in range(N):
            for k in range(N):
                vy = GVY[i, j, k]
                r = GR[i, j, k]
                rr = GRR[i, j, k]

                curr_env = env_base.copy()
                curr_env['v_x'] = vx

                st0 = [vy, r, 0, rr]
                U_list.append(get_dissipated_energy(st0, curr_env, u_base, vehicle_params))

    # Plotting
    U_flat = np.array(U_list)
    U_max_clip = np.percentile(U_flat[U_flat > 0], 98)
    U_min_clip = np.percentile(U_flat, 2)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    X_plot = GVY.flatten()
    Y_plot = GR.flatten()
    Z_plot = GRR.flatten()
    C_plot = np.clip(U_flat, U_min_clip, U_max_clip)

    cmap = plt.cm.get_cmap('RdYlBu')
    norm = plt.Normalize(C_plot.min(), C_plot.max())
    colors = cmap(norm(C_plot))

    sc = ax.scatter(X_plot, Y_plot, Z_plot, c=colors, s=10, alpha=0.5, marker='s')

    # Colorbar with scientific notation
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(C_plot)
    scientific_colorbar(fig, ax, sm, '耗散能 $U\ (\mathrm{J})$')

    # Labels and Ticks
    # 刻度设置
    ax.tick_params(labelsize=default_cfg["tick_size"])
    ax.tick_params(axis='x', direction='in')
    ax.tick_params(axis='y', direction='in')
    ax.tick_params(axis='z', direction='in')

    ax.set_xlabel(r'横向速度$v_y\ \mathrm{(m/s)}$', fontproperties=zhfont1, fontsize=20, labelpad=10)
    ax.set_ylabel(r'横摆角速度$\dot\phi\ \mathrm{(rad/s)}$', fontproperties=zhfont1, fontsize=20, labelpad=10)
    ax.set_zlabel(r'侧倾角速度$\dot\varphi\ \mathrm{(rad/s)}$', fontproperties=zhfont1, fontsize=20, labelpad=10)

    # ax.set_title(f'Dissipated Energy (Vx=30km/h, {title_suffix})', fontproperties=zhfont1, fontsize=16)
    ax.view_init(elev=20, azim=45)

    plt.savefig(f'{OUTPUT_DIR}/3d_vy_r_rr_{scenario_name}.pdf', dpi=300)
    plt.close()

# --- 3D Slope Analysis Plots (Requirement 2 & 1) ---
def plot_3d_scatter_slope_analysis(scenario_name, title_suffix, env_base, u_base):
    # Fixed V_x = 30 km/h
    vx = env_base['v_x']
    N = 40  # 提高分辨率
    # Axes: X=Vy, Y=R, Z=RR
    vy_vals = np.linspace(-5, 5, N)
    r_vals = np.linspace(-1.5, 1.5, N)
    rr_vals = np.linspace(-1.0, 1.0, N)

    GVY, GR, GRR = np.meshgrid(vy_vals, r_vals, rr_vals)
    U_list = []

    # Pre-calculate U_vals
    for i in range(N):
        for j in range(N):
            for k in range(N):
                vy = GVY[i, j, k]
                r = GR[i, j, k]
                rr = GRR[i, j, k]

                curr_env = env_base.copy()
                curr_env['v_x'] = vx

                # State: [v_y, r, roll=0, roll_rate=rr]
                st0 = [vy, r, 0, rr]
                U_list.append(get_dissipated_energy(st0, curr_env, u_base, vehicle_params))

    # --- Plotting ---
    U_flat = np.array(U_list)

    # Clipping for robust color range
    U_max_clip = np.percentile(U_flat[U_flat > 0], 98)
    U_min_clip = np.percentile(U_flat, 2)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    X_plot = GVY.flatten()
    Y_plot = GR.flatten()
    Z_plot = GRR.flatten()
    C_plot = np.clip(U_flat, U_min_clip, U_max_clip)

    # Plot dense scatter points
    cmap = plt.cm.get_cmap('RdYlBu')
    norm = plt.Normalize(C_plot.min(), C_plot.max())
    colors = cmap(norm(C_plot))

    sc = ax.scatter(X_plot, Y_plot, Z_plot, c=colors, s=10, alpha=0.5, marker='s')

    # Colorbar with scientific notation
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(C_plot)
    scientific_colorbar(fig, ax, sm, '耗散能 $U\ (\mathrm{J})$')

    # Labels and Ticks
    # 刻度设置
    ax.tick_params(labelsize=default_cfg["tick_size"])
    ax.tick_params(axis='x', direction='in')
    ax.tick_params(axis='y', direction='in')
    ax.tick_params(axis='z', direction='in')
    # Labels
    ax.set_xlabel(r'横向速度$v_y\ \mathrm{(m/s)}$', fontproperties=zhfont1, fontsize=20, labelpad=10)
    ax.set_ylabel(r'横摆角速度$\dot\phi\ \mathrm{(rad/s)}$', fontproperties=zhfont1, fontsize=20, labelpad=10)
    ax.set_zlabel(r'侧倾角速度$\dot\varphi\ \mathrm{(rad/s)}$', fontproperties=zhfont1, fontsize=20, labelpad=10)

    # ax.set_title(f'Dissipated Energy (Vx=30km/h, {title_suffix})')
    ax.view_init(elev=20, azim=45)

    plt.savefig(f'{OUTPUT_DIR}/3d_vy_r_rr_{scenario_name}.pdf', dpi=300)
    plt.close()

# === [NEW] Function for CoG Height Analysis ===
def plot_3d_scatter_cog_analysis(scenario_name, title_suffix, env_base, u_base, vp_override):
    """
    Plots the dissipated energy in 3D (Vy, R, RR) for a specific vehicle parameter set (vp_override).
    Used to analyze the effect of CoG height.
    """
    vx = env_base['v_x']
    N = 40  # Resolution
    vy_vals = np.linspace(-5, 5, N)
    r_vals = np.linspace(-1.5, 1.5, N)
    rr_vals = np.linspace(-1.0, 1.0, N)

    GVY, GR, GRR = np.meshgrid(vy_vals, r_vals, rr_vals)
    U_list = []

    # Pre-calculate U_vals
    for i in range(N):
        for j in range(N):
            for k in range(N):
                vy = GVY[i, j, k]
                r = GR[i, j, k]
                rr = GRR[i, j, k]

                curr_env = env_base.copy()
                curr_env['v_x'] = vx

                st0 = [vy, r, 0, rr]
                # Use vp_override here
                U_list.append(get_dissipated_energy(st0, curr_env, u_base, vp_override))

    # --- Plotting ---
    U_flat = np.array(U_list)
    U_max_clip = np.percentile(U_flat[U_flat > 0], 98)
    U_min_clip = np.percentile(U_flat, 2)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    X_plot = GVY.flatten()
    Y_plot = GR.flatten()
    Z_plot = GRR.flatten()
    C_plot = np.clip(U_flat, U_min_clip, U_max_clip)

    cmap = plt.cm.get_cmap('RdYlBu')
    norm = plt.Normalize(C_plot.min(), C_plot.max())
    colors = cmap(norm(C_plot))

    sc = ax.scatter(X_plot, Y_plot, Z_plot, c=colors, s=10, alpha=0.5, marker='s')

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(C_plot)
    scientific_colorbar(fig, ax, sm, '耗散能 $U\ (\mathrm{J})$')

    # Labels and Ticks
    ax.tick_params(labelsize=default_cfg["tick_size"])
    ax.tick_params(axis='x', direction='in')
    ax.tick_params(axis='y', direction='in')
    ax.tick_params(axis='z', direction='in')

    ax.set_xlabel(r'横向速度$v_y\ \mathrm{(m/s)}$', fontproperties=zhfont1, fontsize=20, labelpad=10)
    ax.set_ylabel(r'横摆角速度$\dot\phi\ \mathrm{(rad/s)}$', fontproperties=zhfont1, fontsize=20, labelpad=10)
    ax.set_zlabel(r'侧倾角速度$\dot\varphi\ \mathrm{(rad/s)}$', fontproperties=zhfont1, fontsize=20, labelpad=10)

    # ax.set_title(f'Dissipated Energy ({title_suffix})', fontproperties=zhfont1, fontsize=16)
    ax.view_init(elev=20, azim=45)

    plt.savefig(f'{OUTPUT_DIR}/3d_vy_r_rr_{scenario_name}.pdf', dpi=300)
    plt.close()

# ================= 6. Execution Run =================
if __name__ == '__main__':
    # --- Common parameters for all plots ---
    BASE_ENV = {'v_x': 36 / 3.6, 'mu': 0.6, 'theta_r': 0, 'phi_r': 0}
    VX_BASE = BASE_ENV['v_x']
    MU_BASE = BASE_ENV['mu']
    BASE_U = {'delta_f': 0.0}
    FIXED_ROLL = {'roll_rate': 0.0, 'roll': 0.0}
    # --- 1. 2D Projection Plot (Requirement 3) ---
    # plot_2d_projection(VX_BASE, MU_BASE)
    # plot_2d_beta_r_projection(VX_BASE, MU_BASE)
    # A. Base Comparison: Mu=0.8 vs Mu=0.2 (Closest to reference image style)
    # print("Generating A: Mu=0.8 Analysis...")
    # plot_surface_style('Mu_0.8_Reference_Style', 'High Adhesion $(\\mu=0.8)$',
    #                    {'v_x': VX_BASE, 'mu': 0.8, 'theta_r': 0, 'phi_r': 0}, BASE_U, 0, FIXED_ROLL)
    #
    # print("Generating B: Mu=0.2 Analysis...")
    # plot_surface_style('Mu_0.2_Reference_Style', 'Low Adhesion $(\\mu=0.2)$',
    #                    {'v_x': VX_BASE, 'mu': 0.2, 'theta_r': 0, 'phi_r': 0}, BASE_U, 0, FIXED_ROLL)
    #
    # print("Generating C: Mu=0.4 Analysis...")
    # plot_surface_style('Mu_0.4_Reference_Style', 'Low Adhesion $(\\mu=0.4)$',
    #                    {'v_x': VX_BASE, 'mu': 0.4, 'theta_r': 0, 'phi_r': 0}, BASE_U, 0, FIXED_ROLL)
    #
    # print("Generating D: Mu=0.6 Analysis...")
    # plot_surface_style('Mu_0.6_Reference_Style', 'Low Adhesion $(\\mu=0.2)$',
    #                    {'v_x': VX_BASE, 'mu': 0.6, 'theta_r': 0, 'phi_r': 0}, BASE_U, 0, FIXED_ROLL)

    # C. Slope Impact (Vx, Vy, YawRate) - Demonstrates shift and deformation
    # Use fixed roll rate for consistency
    # print("Generating C: Longitudinal Slope Analysis (20 deg)...")
    # plot_surface_style('LonSlope_20deg', 'Longitudinal Slope (20°)',
    #                    {'v_x': 0, 'mu': 0.6, 'theta_r': np.radians(20), 'phi_r': 0}, BASE_U, 0, FIXED_ROLL)
    #
    # print("Generating D: Lateral Slope Analysis (8 deg)...")
    # plot_surface_style('LatSlope_8deg', 'Lateral Slope (8°)',
    #                    {'v_x': 0, 'mu': 0.6, 'theta_r': 0, 'phi_r': np.radians(8)}, BASE_U, 0, FIXED_ROLL)

    # Case 1: Longitudinal Slope (0 deg, 0 deg) -> Flat
    # print("正在生成 3D scatter: Flat road (0, 0)...")
    # plot_3d_scatter_slope_analysis('Flat_Road_0_0', 'Flat Road (0°)',
    #                                {'v_x': VX_BASE, 'mu': MU_BASE, 'theta_r': 0, 'phi_r': 0}, BASE_U)
    #
    # # Case 2: Longitudinal Slope (20 deg, 0 deg)
    # print("正在生成 3D scatter: Longitudinal Slope (20, 0)...")
    # plot_3d_scatter_slope_analysis('LonSlope_20_0', 'Longitudinal Slope (20°)',
    #                                {'v_x': VX_BASE, 'mu': MU_BASE, 'theta_r': np.radians(20), 'phi_r': 0}, BASE_U)
    #
    # # Case 3: Lateral Slope (0 deg, 10 deg)
    # print("正在生成 3D scatter: Lateral Slope (0, 10)...")
    # plot_3d_scatter_slope_analysis('LatSlope_0_10', 'Lateral Slope (10°)',
    #                                {'v_x': VX_BASE, 'mu': MU_BASE, 'theta_r': 0, 'phi_r': np.radians(10)}, BASE_U)
    #
    # # Case 4: Combined Slope (20 deg, 10 deg) - Added for comprehensive analysis
    # print("正在生成 3D scatter: Combined Slope (20, 10)...")
    plot_3d_scatter_slope_analysis('CombinedSlope_20_10', 'Combined Slope (20° Lon, 10° Lat)',
                                   {'v_x': VX_BASE, 'mu': MU_BASE, 'theta_r': np.radians(20), 'phi_r': np.radians(10)},
                                   BASE_U)

    # 2. [NEW] Center of Gravity (CoG) Height Analysis
    print("\n--- Starting Center of Gravity (CoG) Height Analysis ---")

    # Define CoG heights to test.
    # Logic: Raise h, and assuming roll center (RC) height is fixed, hs (arm length) increases.
    # h_rc = original_h (0.75) - original_hs (0.45) = 0.30 m
    H_RC_FIXED = 0.20

    cog_variations = [
        {'label': 'Low_CoG', 'h': 0.35, 'title': 'Low CoG (h=0.35m)'},
        {'label': 'Baseline_CoG', 'h': 0.55, 'title': 'Baseline CoG (h=0.55m)'},
        {'label': 'High_CoG', 'h': 0.75, 'title': 'High CoG (h=0.75m)'},
    ]

    for var in cog_variations:
        h_new = var['h']
        hs_new = h_new - H_RC_FIXED

        # Create a deep copy of vehicle params to avoid modifying the global dict for other runs
        vp_mod = copy.deepcopy(vehicle_params)
        vp_mod['h'] = h_new
        vp_mod['hs'] = hs_new

        print(f"Generating 3D scatter for {var['title']} (h={h_new}, hs={hs_new:.2f})...")
        plot_3d_scatter_cog_analysis(
            f"CoG_{var['label']}",
            var['title'],
            BASE_ENV,
            BASE_U,
            vp_mod
        )

    print("\n所有图表生成完毕！请查看 'dissipated_energy_surface_plots' 目录。")