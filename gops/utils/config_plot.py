import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams

# ----------------- 论文绘图风格配置 (基于 energy_stability_analysis.py) -----------------
# 1. 字体配置
# 注意：请确保路径 '../gops/utils/SIMSUN.ttf' 是正确的，或者改为系统中的 'SimSun'
try:
    zhfont = fm.FontProperties(fname='../gops/utils/SIMSUN.ttf', size=14)
except:
    print("Warning: SIMSUN.ttf not found, using SimHei as fallback.")
    zhfont = fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf', size=14)  # Windows fallback

# 2. 全局参数配置
default_cfg = dict()
default_cfg["fig_size"] = (12, 9)  # 保持模板尺寸
default_cfg["dpi"] = 300
default_cfg["tick_size"] = 14  # 坐标轴刻度字号
default_cfg["label_size"] = 16  # 坐标轴标签字号
default_cfg["legend_size"] = 14  # 图例字号

# 设置英文字体为 Times New Roman
config = {
    "font.family": 'serif',
    "font.serif": ['Times New Roman'],
    "mathtext.fontset": 'stix',  # 公式字体风格
    "axes.unicode_minus": False  # 处理负号显示问题
}
rcParams.update(config)


def apply_paper_style(ax, title=None, x_label=None, y_label=None):
    """
    统一应用论文格式到当前的 ax
    """
    # 设置刻度字体 (Times New Roman)
    ax.tick_params(labelsize=default_cfg["tick_size"])
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Times New Roman')

    # 设置中文标签
    if x_label:
        ax.set_xlabel(x_label, fontproperties=zhfont, fontsize=default_cfg["label_size"])
    if y_label:
        ax.set_ylabel(y_label, fontproperties=zhfont, fontsize=default_cfg["label_size"])
    if title:
        ax.set_title(title, fontproperties=zhfont, fontsize=default_cfg["label_size"] + 2)

    # 设置图例字体 (如果有图例)
    if ax.get_legend():
        # 图例可能包含中文，也可能包含英文变量
        plt.setp(ax.get_legend().get_texts(), fontproperties=zhfont, fontsize=default_cfg["legend_size"])

    ax.grid(True, linestyle='--', alpha=0.6)