import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_figure8_trajectory(filename="figure8_trajectory.csv"):
    # --- 1. 参数设置 ---
    # 使用 Lissajous 曲线参数方程:
    # x = A * sin(t)
    # y = B * sin(2*t)

    # 尺寸设定 (宽度50m，高度25m)
    A = 50.0  # X轴振幅 (半宽)
    B = 25  # Y轴振幅 (半高)

    step_size = 0.1  # 目标间距 0.1m

    # --- 2. 生成高分辨率原始路径 (Raw Path) ---
    # 为了保证插值精度，先生成非常密集的点
    # t 从 0 到 2*pi (一个完整周期)
    num_raw_points = 10000
    t = np.linspace(0, 2 * np.pi, num_raw_points)

    # 原始坐标
    x_raw = A * np.sin(t)
    y_raw = B * np.sin(2 * t)

    # --- 3. 基于弧长进行重采样 (Resampling) ---
    # 计算原始路径上相邻点的距离
    dx = np.diff(x_raw)
    dy = np.diff(y_raw)
    dist_steps = np.sqrt(dx ** 2 + dy ** 2)

    # 计算累积距离 (S)
    # 在开头插入 0，使得 cum_dist 长度与 x_raw 一致
    cum_dist = np.insert(np.cumsum(dist_steps), 0, 0.0)
    total_length = cum_dist[-1]

    # 生成目标等间距的距离数组: [0, 0.1, 0.2, ..., L]
    target_dists = np.arange(0, total_length, step_size)

    # 使用线性插值获取对应距离下的 x, y 坐标
    # np.interp(x_new, x_old, y_old)
    x_new = np.interp(target_dists, cum_dist, x_raw)
    y_new = np.interp(target_dists, cum_dist, y_raw)

    # --- 4. 计算航向角 (Angle) ---
    # 计算差分 dx, dy
    # 最后一个点的角度沿用前一个点的角度，或者根据闭环特性计算与起点的角度

    new_dx = np.diff(x_new)
    new_dy = np.diff(y_new)

    # 计算切线角度
    angles = np.arctan2(new_dy, new_dx)

    # 补齐最后一个点的角度 (复制倒数第二个点的角度)
    # 或者为了闭环更平滑，可以计算最后一点指向第一点的角度
    angles = np.append(angles, angles[-1])

    # 角度归一化到 [-pi, pi] (虽然 arctan2 已经是这个范围，但为了保险)
    angles = (angles + np.pi) % (2 * np.pi) - np.pi

    # --- 5. 构建 DataFrame 并保存 ---
    df = pd.DataFrame({
        'x': x_new,
        'y': y_new,
        'angle': angles,
        'distance': target_dists
    })

    # 格式化数据，保留4位小数
    df = df.round(4)

    df.to_csv(filename, index=False)

    print(f"8字形轨迹已生成: {filename}")
    print(f"总点数: {len(df)}")
    print(f"总长度: {total_length:.2f}m")

    return df


if __name__ == "__main__":
    # 生成文件
    df = generate_figure8_trajectory("figure8_50x25_step0.1.csv")

    # --- 绘图验证 ---
    try:
        plt.figure(figsize=(10, 6))

        # 绘制路径点
        plt.plot(df['x'], df['y'], label='Trajectory', color='blue')

        # 标记起点 (绿色) 和 终点 (红色)
        plt.scatter(df['x'].iloc[0], df['y'].iloc[0], c='green', s=100, label='Start', zorder=5)
        plt.scatter(df['x'].iloc[-1], df['y'].iloc[-1], c='red', s=50, label='End', zorder=5)

        # 随机抽取一些点画箭头，验证角度是否正确
        step = 50  # 每隔50个点画一个箭头
        plt.quiver(df['x'][::step], df['y'][::step],
                   np.cos(df['angle'][::step]), np.sin(df['angle'][::step]),
                   color='orange', scale=20, width=0.003, label='Heading')

        plt.title("Figure-8 Trajectory (Lissajous Curve)")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.axis('equal')  # 保证比例一致
        plt.grid(True)
        plt.legend()
        plt.show()

    except ImportError:
        print("Matplotlib not found, skipping plot.")