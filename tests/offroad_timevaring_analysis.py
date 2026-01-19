import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建图形和子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.35, top=0.9)

# 初始参数
initial_speed = 70.0  # km/h
initial_friction = 0.7
initial_long_grade = 5.0  # 纵坡，度
initial_trans_grade = 2.0  # 横坡，度
initial_curve_radius = 50.0  # 曲线半径，米

# 车辆参数
vehicle_mass = 2000  # kg
cg_height = 0.5  # 重心高度，m
track_width = 1.8  # 轮距，m
wheelbase = 2.7  # 轴距，m

# 创建滑块控件
ax_speed = plt.axes([0.25, 0.25, 0.65, 0.03])
ax_friction = plt.axes([0.25, 0.20, 0.65, 0.03])
ax_long_grade = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_trans_grade = plt.axes([0.25, 0.10, 0.65, 0.03])
ax_radius = plt.axes([0.25, 0.05, 0.65, 0.03])

speed_slider = Slider(ax_speed, '速度 (km/h)', 0.0, 150.0, valinit=initial_speed)
friction_slider = Slider(ax_friction, '附着系数', 0.1, 1.2, valinit=initial_friction)
long_grade_slider = Slider(ax_long_grade, '纵坡 (°)', -20.0, 20.0, valinit=initial_long_grade)
trans_grade_slider = Slider(ax_trans_grade, '横坡 (°)', -10.0, 10.0, valinit=initial_trans_grade)
radius_slider = Slider(ax_radius, '曲线半径 (m)', 0.0, 200.0, valinit=initial_curve_radius)


# 计算稳定性边界函数 - 考虑横坡和纵坡
def calculate_stability_boundaries(speed, friction, long_grade, trans_grade, curve_radius):
    # 将速度从km/h转换为m/s
    v = speed * 1000 / 3600

    # 坡度角度转换为弧度
    theta_long = np.radians(long_grade)  # 纵坡角度
    theta_trans = np.radians(trans_grade)  # 横坡角度

    # 重力加速度
    g = 9.81

    # 计算坡度对垂直载荷的影响
    # 考虑纵坡和横坡的综合影响
    normal_force = vehicle_mass * g * np.cos(theta_long) * np.cos(theta_trans)

    # 计算最大侧向力（考虑横坡影响）
    # 横坡会影响左右轮的载荷分布
    max_lateral_force = friction * normal_force * np.cos(theta_trans)

    # 计算曲线行驶所需的侧向力
    required_lateral_force = vehicle_mass * v ** 2 / curve_radius

    # 计算纵向力极限（考虑纵坡影响）
    # 上坡时加速能力降低，下坡时制动能力降低
    long_force_limit = friction * normal_force * np.cos(theta_long)

    # 计算摩擦椭圆 - 表示纵向和侧向力的耦合关系
    ellipse_major = long_force_limit  # 纵向半轴
    ellipse_minor = max_lateral_force  # 侧向半轴

    # 判断是否失稳
    is_stable = required_lateral_force <= max_lateral_force

    # 计算侧翻阈值
    # 考虑横坡和曲线行驶的侧向加速度
    rollover_threshold = (track_width / 2 + cg_height * np.tan(theta_trans)) * g / (
                cg_height + track_width / 2 * np.tan(theta_trans))
    is_rollover_stable = v ** 2 / curve_radius <= rollover_threshold

    return {
        'v': v,
        'max_lateral_force': max_lateral_force,
        'required_lateral_force': required_lateral_force,
        'long_force_limit': long_force_limit,
        'ellipse_major': ellipse_major,
        'ellipse_minor': ellipse_minor,
        'is_stable': is_stable,
        'is_rollover_stable': is_rollover_stable,
        'normal_force': normal_force,
        'curve_radius': curve_radius,
        'rollover_threshold': rollover_threshold
    }


# 更新函数
def update(val):
    speed = speed_slider.val
    friction = friction_slider.val
    long_grade = long_grade_slider.val
    trans_grade = trans_grade_slider.val
    curve_radius = radius_slider.val

    # 计算稳定性边界
    stability = calculate_stability_boundaries(speed, friction, long_grade, trans_grade, curve_radius)

    # 清除当前图形
    ax1.clear()
    ax2.clear()

    # 设置第一个子图 - 摩擦椭圆
    ax1.set_title('摩擦椭圆 - 力边界', fontsize=14, fontweight='bold')
    ax1.set_xlabel('纵向力 (N)', fontsize=12)
    ax1.set_ylabel('侧向力 (N)', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # 设置坐标轴范围
    max_force = max(10000, stability['ellipse_major'] * 1.2, stability['ellipse_minor'] * 1.2)
    ax1.set_xlim(-max_force, max_force)
    ax1.set_ylim(0, max_force)

    # 绘制摩擦椭圆（稳定性边界）
    ellipse = patches.Ellipse(
        (0, 0),
        width=2 * stability['ellipse_major'],
        height=2 * stability['ellipse_minor'],
        fill=False,
        edgecolor='blue',
        linewidth=4,
        label='稳定性边界 (摩擦椭圆)'
    )
    ax1.add_patch(ellipse)

    # 绘制当前状态点
    curve_color = 'green' if stability['is_stable'] else 'red'
    ax1.plot(0, stability['required_lateral_force'], 'o', color=curve_color, markersize=10,
             label=f'当前状态: {"稳定" if stability["is_stable"] else "不稳定"}')

    # 绘制坐标轴
    ax1.axvline(0, color='black', linewidth=0.5)
    ax1.axhline(0, color='black', linewidth=0.5)

    # 添加参考线
    ax1.axhline(y=stability['max_lateral_force'], color='red', linestyle='--', alpha=0.7, linewidth=1, label='最大侧向力')

    # 添加图例
    ax1.legend(loc='lower right')

    # 设置第二个子图 - 侧翻边界
    ax2.set_title('侧翻边界分析', fontsize=14, fontweight='bold')
    ax2.set_xlabel(r'侧向加速度 (m/$\mathrm{s}^2$)', fontsize=12)
    ax2.set_ylabel('发生概率', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # 绘制侧翻边界
    lateral_acc = np.linspace(0, 15, 100)
    rollover_prob = 1 / (1 + np.exp(-5 * (lateral_acc - stability['rollover_threshold'])))

    ax2.plot(lateral_acc, rollover_prob, 'b-', linewidth=4, label='侧翻概率')
    ax2.axvline(x=stability['rollover_threshold'], color='r', linestyle='--', linewidth=4, label='侧翻阈值')

    current_lateral_acc = stability['required_lateral_force'] / vehicle_mass
    ax2.axvline(x=current_lateral_acc, color='g' if stability['is_rollover_stable'] else 'r',
                linestyle='-', linewidth=4, label='当前侧向加速度')

    ax2.set_ylim(0, 1.1)
    ax2.legend(loc='upper left')

    # 添加文本信息
    info_text = f"""
    速度: {speed:.1f} km/h ({stability['v']:.1f} m/s)
    附着系数: {friction:.2f}
    纵坡: {long_grade:.1f}°
    横坡: {trans_grade:.1f}°
    曲线半径: {curve_radius:.1f} m
    所需侧向力: {stability['required_lateral_force']:.0f} N
    最大可用侧向力: {stability['max_lateral_force']:.0f} N
    垂直载荷: {stability['normal_force']:.0f} N
    侧翻阈值: {stability['rollover_threshold']:.2f} m/s2
    当前侧向加速度: {current_lateral_acc:.2f} m/s2
    侧向稳定性: {'稳定' if stability['is_stable'] else '不稳定'}
    侧翻稳定性: {'稳定' if stability['is_rollover_stable'] else '不稳定'}
    """
    fig.text(0.02, 0.02, info_text, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)

    plt.draw()


# 注册更新函数
speed_slider.on_changed(update)
friction_slider.on_changed(update)
long_grade_slider.on_changed(update)
trans_grade_slider.on_changed(update)
radius_slider.on_changed(update)

# # 重置按钮
# reset_ax = plt.axes([0.8, 0.01, 0.1, 0.04])
# reset_button = Button(reset_ax, '重置', hovercolor='0.975')


def reset(event):
    speed_slider.reset()
    friction_slider.reset()
    long_grade_slider.reset()
    trans_grade_slider.reset()
    radius_slider.reset()


# reset_button.on_clicked(reset)

# 初始化图表
update(None)


# 创建动画函数
def animate(i):
    # 模拟参数变化
    if i < 10:
        speed_slider.set_val(40 + i * 10)
        friction_slider.set_val(0.8 - i * 0.05)
    elif i < 30:
        long_grade_slider.set_val(-10 + (i - 10) * 1.0)
        trans_grade_slider.set_val(-5 + (i - 10) * 0.5)
    else:
        speed_slider.set_val(70)
        friction_slider.set_val(0.7)
        radius_slider.set_val(100 - i * 5)
    return []


# 创建动画
ani = FuncAnimation(fig, animate, frames=50, interval=50, blit=True)

plt.show()

# 如果要保存动画，取消下面的注释
# ani.save('vehicle_stability.gif', writer='pillow', fps=20)