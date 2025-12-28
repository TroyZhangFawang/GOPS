import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
import os
import matplotlib as mpl
import matplotlib.font_manager as fm
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import pandas as pd
import copy  # 用于复制字典
import argparse
zhfont1 = fm.FontProperties(fname='../gops/utils/SIMSUN.ttf')
y_formatter = FormatStrFormatter('%1')
# font = FontProperties(fname="SimHei.ttf", size=15)
default_cfg = dict()
default_cfg["fig_size"] = (12, 9)
default_cfg["dpi"] = 300
default_cfg["pad"] = 0.5
default_cfg["ax_para"] = [0.17, 0.20, 0.80, 0.70]


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
# 字体设置：全局 Times New Roman，中文单独处理
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'stix'  # 公式字体风格更像 LaTeX

# SCI 经典配色 (Nature Publishing Group 风格)
# 顺序：红(朱红)、蓝(知更鸟蛋蓝)、绿(深海绿)、深蓝、橙
SCI_COLORS = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2']

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
    step_list = np.array(data_result.iloc[0, start_index:end_index:interval], dtype='float32')

    data_pool = np.zeros((line_num + 1, len(step_list)))
    data_pool[0, :] = step_list
    data_pool[1, :] = np.array(data_result.iloc[1, start_index:end_index:interval], dtype='float32')  # 3.14*180  # MPC
    return data_pool

def read_csv_line5(root_path, line_num):
    data_result = pd.DataFrame(pd.read_csv(root_path, header=None))
    start_index = 1
    end_index = -1
    interval = 5
    step_list = np.array(data_result.iloc[0, start_index:end_index:interval], dtype='float32')
    data_pool = np.zeros((line_num+1, len(step_list)))
    data_pool[0, :] = step_list

    data_pool[1, :] = np.array(data_result.iloc[1, start_index:end_index:interval], dtype='float32') # Ref
    data_pool[2, :] = np.array(data_result.iloc[3, start_index:end_index:interval], dtype='float32')  # MPC
    data_pool[3, :] = np.array(data_result.iloc[2, start_index:end_index:interval], dtype='float32')  # FHADP
    data_pool[4, :] = np.array(data_result.iloc[4, start_index:end_index:interval], dtype='float32')  # PDP
    data_pool[5, :] = np.array(data_result.iloc[5, start_index:end_index:interval], dtype='float32')  # Bilevel
    # for num in range(line_num):
    #     data_numi = np.array(data_result.iloc[num+1, 1:], dtype='float32')
    #     data_pool[num+1, :] = data_numi
    return data_pool

def read_csv_line5_from3path(root_path1, root_path2, root_path3, line_num):
    data_result = pd.DataFrame(pd.read_csv(root_path1, header=None))
    data_result2 = pd.DataFrame(pd.read_csv(root_path2, header=None))
    data_result3 = pd.DataFrame(pd.read_csv(root_path3, header=None))
    start_index = 1
    end_index = 2000
    interval = 1
    step_list = np.array(data_result.iloc[0, start_index:end_index:interval], dtype='float32')

    data_pool = np.zeros((line_num+1, len(step_list)))
    data_pool[0, :] = step_list
    data_pool[1, :] = np.array(data_result.iloc[3, start_index:end_index:interval], dtype='float32')  # Ref
    data_pool[2, :] = np.array(data_result2.iloc[1, start_index:end_index:interval], dtype='float32')  # OnMPC
    data_pool[3, :] = np.array(data_result.iloc[1, start_index:end_index:interval], dtype='float32')  # CMMPC
    data_pool[4, :] = np.array(data_result3.iloc[1, start_index:end_index:interval], dtype='float32')  # CRMPC
    data_pool[5, :] = np.array(data_result.iloc[2, start_index:end_index:interval], dtype='float32')  # CTMPC
    # for num in range(line_num):
    #     data_numi = np.array(data_result.iloc[num+1, 1:], dtype='float32')
    #     data_pool[num+1, :] = data_numi
    # print(data_pool[2, :])
    return data_pool

def read_csv_line4(root_path, line_num):
    data_result = pd.DataFrame(pd.read_csv(root_path, header=None))
    start_index = 2
    end_index = -1
    interval = 1
    step_list = np.array(data_result.iloc[0, start_index:end_index:interval], dtype='float32')

    data_pool = np.zeros((line_num+1, len(step_list)))
    data_pool[0, :] = step_list

    data_pool[1, :] = np.array(data_result.iloc[4, start_index:end_index:interval], dtype='float32')  # Ref
    data_pool[2, :] = np.array(data_result.iloc[3, start_index:end_index:interval], dtype='float32')  # MPC
    data_pool[3, :] = np.array(data_result.iloc[2, start_index:end_index:interval], dtype='float32')  # FHADP
    data_pool[4, :] = np.array(data_result.iloc[1, start_index:end_index:interval], dtype='float32')  # Bilevel
    # for num in range(line_num):
    #     data_numi = np.array(data_result.iloc[num+1, 1:], dtype='float32')
    #     data_pool[num+1, :] = data_numi
    return data_pool

def read_csv_line4_from3path(root_path1, root_path2, root_path3, line_num):
    data_result = pd.DataFrame(pd.read_csv(root_path1, header=None))
    data_result2 = pd.DataFrame(pd.read_csv(root_path2, header=None))
    data_result3 = pd.DataFrame(pd.read_csv(root_path3, header=None))
    start_index = 1
    end_index = 2000
    interval = 1
    step_list = np.array(data_result.iloc[0, start_index:end_index:interval], dtype='float32')

    data_pool = np.zeros((line_num+1, len(step_list)))
    data_pool[0, :] = step_list
    data_pool[1, :] = np.array(data_result2.iloc[1, start_index:end_index:interval], dtype='float32')#3.14*180  # MPC
    data_pool[2, :] = np.array(data_result.iloc[1, start_index:end_index:interval], dtype='float32')#3.14*180 # FHADP
    data_pool[3, :] = np.array(data_result3.iloc[1, start_index:end_index:interval], dtype='float32')  # Bilevel
    data_pool[4, :] = np.array(data_result.iloc[2, start_index:end_index:interval], dtype='float32')  # 3.14*180  # ABPO
    return data_pool

def read_csv_line3(root_path, line_num):
    data_result = pd.DataFrame(pd.read_csv(root_path, header=None))
    start_index = 1
    end_index = -1
    interval = 1
    step_list = np.array(data_result.iloc[0, start_index:end_index:interval], dtype='float32')

    data_pool = np.zeros((line_num+1, len(step_list)))
    data_pool[0, :] = step_list

    data_pool[1, :] = np.array(data_result.iloc[3, start_index:end_index:interval], dtype='float32')#3.14*180  # MPC
    data_pool[2, :] = np.array(data_result.iloc[1, start_index:end_index:interval], dtype='float32')#3.14*180 # FHADP
    data_pool[3, :] = np.array(data_result.iloc[2, start_index:end_index:interval], dtype='float32')#3.14*180  # ABPO
    # data_pool[4, :] = np.array(data_result.iloc[2, start_index:end_index:interval], dtype='float32')  # Bilevel
    # for num in range(line_num):
    #     data_numi = np.array(data_result.iloc[num+1, 1:], dtype='float32')
    #     data_pool[num+1, :] = data_numi
    return data_pool

def read_csv_line3_from3path(root_path, root_path2, line_num):
    data_result = pd.DataFrame(pd.read_csv(root_path, header=None))
    data_result2 = pd.DataFrame(pd.read_csv(root_path2, header=None))
    start_index = 1
    end_index = 1000
    interval = 1
    step_list = np.array(data_result.iloc[0, start_index:end_index:interval], dtype='float32')

    data_pool = np.zeros((line_num+1, len(step_list)))
    data_pool[0, :] = step_list

    data_pool[1, :] = np.array(data_result2.iloc[1, start_index:end_index:interval], dtype='float32')#3.14*180  # MPC
    data_pool[2, :] = np.array(data_result.iloc[1, start_index:end_index:interval], dtype='float32')#3.14*180 # FHADP
    data_pool[3, :] = np.array(data_result.iloc[2, start_index:end_index:interval], dtype='float32')#3.14*180  # ABPO
    # data_pool[4, :] = np.array(data_result.iloc[2, start_index:end_index:interval], dtype='float32')  # Bilevel
    # for num in range(line_num):
    #     data_numi = np.array(data_result.iloc[num+1, 1:], dtype='float32')
    #     data_pool[num+1, :] = data_numi
    return data_pool

def read_csv_line2(root_path, line_num):
    data_result = pd.DataFrame(pd.read_csv(root_path, header=None))
    start_index = 1
    end_index = -1
    interval = 1
    step_list = np.array(data_result.iloc[0, start_index:end_index:interval], dtype='float32')

    data_pool = np.zeros((line_num+1, len(step_list)))
    data_pool[0, :] = step_list

    data_pool[1, :] = np.array(data_result.iloc[1, start_index:end_index:interval], dtype='float32')#3.14*180  # MPC
    data_pool[2, :] = np.array(data_result.iloc[2, start_index:end_index:interval], dtype='float32')#3.14*180 # FHADP

    return data_pool


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

def plot_2d(data, args):
    dt = args["time_step"]
    legend_list = args["legend_list"]
    color_list = args["color_list"]
    linestyle_list = args["linestyle_list"]
    line_num = args["line_num"]
    save_dir = args["figures_root"] + '/plot_results'
    os.makedirs(save_dir, exist_ok=True)
    path_state_fmt = os.path.join(
        save_dir, "Times-" + args["csv_file_name"] + ".{}".format(default_cfg["img_fmt"])
    )
    path_state_fmtpdf = os.path.join(
        save_dir, "Times-" + args["csv_file_name"] + ".{}".format("pdf")
    )
    fig_size = (
        default_cfg["fig_size"],
        default_cfg["fig_size"],
    )

    fig, ax = plt.subplots(figsize=cm2inch(default_cfg["fig_size"]),
                           dpi=default_cfg["dpi"],
                           constrained_layout=True)
    # 绘图循环
    for i in range(line_num):
        # 安全获取样式，防止索引越界
        legend = legend_list[i] if i < len(legend_list) else f"Line {i}"
        color = color_list[i % len(color_list)]
        linestyle = linestyle_list[i % len(linestyle_list)]

        # 绘制
        sns.lineplot(
            x=data[0, :] * dt,
            y=data[i + 1, :],
            linewidth=default_cfg["line_width"],
            color=color,
            linestyle=linestyle,
            label=legend,
            ax=ax
        )

    ax.tick_params(labelsize=default_cfg["tick_size"])
    ax.tick_params(axis='x', direction='in')
    ax.tick_params(axis='y', direction='in')
    ax.set_xlabel(args["x_label"], default_cfg["label_font"], fontproperties=zhfont1, fontsize=20)  #
    ax.set_ylabel(args["y_label"], default_cfg["label_font"], fontproperties=zhfont1, fontsize=20)  #
    # === 图例优化 ===
    # frameon=False 去掉图例边框更简洁；ncol根据图例数量自动调整
    ncol = 2 if line_num > 3 else 1
    legend = ax.legend(loc="best", prop=zhfont1, fontsize=default_cfg["legend_size"],
                       frameon=True, edgecolor='black', fancybox=False, ncol=ncol)
    legend.get_frame().set_linewidth(0.8)  # 图例边框变细

    plt.savefig(
        path_state_fmt, format=default_cfg["img_fmt"], bbox_inches="tight"
    )
    plt.savefig(
        path_state_fmtpdf, format="pdf", bbox_inches="tight"
    )
    plt.close()

def plot_3d(scenario_name, title_suffix, env_base, u_base, fixed_vx_kmh, fixed_state_vals):
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


# ================= 6. Execution Run =================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--time_step", type=float, default=0.01)
    parser.add_argument("--csv_file_name", type=str, default="Reward")
    parser.add_argument("--line_num", type=int, default=1)

    parser.add_argument("--x_label", type=str, default=r"$p_{\rm x,tl}\ /\mathrm{m}$")
    parser.add_argument("--y_label", type=str, default=r"$p_{\rm y,tl}\ /\mathrm{m}$")
    parser.add_argument("--legend_list", type=list, default=
    ["A","B", "C", "D", "E"])
    parser.add_argument("--color_list", type=list, default=
    ["b","#8A2BE2", "#FA8072", "lime", "magenta"])
    parser.add_argument("--figures_root", type=str,
                        default='./../***/')
    # Get parameter dictionary
    args = vars(parser.parse_args())
    read_path_datax = args["figures_root"] + "State-1.csv"
    read_path_datax2 = args["figures_root2"] + "State-1.csv"
    read_path_datax3 = args["figures_root3"] + "State-1.csv"
    read_path = args["figures_root"]+args["csv_file_name"]+".csv"
    read_path2 = args["figures_root2"] + args["csv_file_name"] + ".csv"
    read_path3 = args["figures_root3"] + args["csv_file_name"] + ".csv"
    if args["line_num"]==5:
        data_csv = read_csv_line5(read_path, args["line_num"])#, read_path2, read_path3
        read_datax = read_csv_line5(read_path_datax, args["line_num"])#, read_path_datax2, read_path_datax3

    elif args["line_num"]==4:
        # data_csv = read_csv_line4(read_path, args["line_num"])
        # read_datax = read_csv_line4(read_path_datax, args["line_num"])
        data_csv = read_csv_line4_from3path(read_path, read_path2, read_path3, args["line_num"])
        read_datax = read_csv_line4_from3path(read_path_datax, read_path_datax2, read_path_datax3, args["line_num"])

    elif args["line_num"] == 3:
        data_csv = read_csv_line3(read_path, args["line_num"])
        read_datax = read_csv_line3(read_path_datax, args["line_num"])
        # data_csv = read_csv_line3_from3path(read_path, read_path2, args["line_num"])
        # read_datax = read_csv_line3_from3path(read_path_datax, read_path_datax2, args["line_num"])

    elif args["line_num"]==2:
        data_csv = read_csv_line2(read_path, args["line_num"])
        read_datax = read_csv_line2(read_path_datax, args["line_num"])
    elif args["line_num"]==1:
        data_csv = read_csv_line1(read_path, args["line_num"])
        read_datax = read_csv_line1(read_path_datax, args["line_num"])
    plot_2d(data_csv, args)