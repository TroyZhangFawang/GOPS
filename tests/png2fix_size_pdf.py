from PIL import Image
import numpy as np


def resize_to_aspect_ratio(image_path,save_path, target_aspect_ratio=12 / 9):
    """将图像调整到目标宽高比"""
    img = Image.open(image_path)
    width, height = img.size

    # 当前宽高比
    current_aspect = width / height

    if current_aspect > target_aspect_ratio:
        # 太宽了，需要裁剪宽度
        new_width = int(height * target_aspect_ratio)
        left = (width - new_width) // 2
        right = left + new_width
        cropped = img.crop((left, 0, right, height))
    else:
        # 太高了，需要裁剪高度
        new_height = int(width / target_aspect_ratio)
        top = (height - new_height) // 2
        bottom = top + new_height
        cropped = img.crop((0, top, width, bottom))

    # 调整到目标DPI
    target_width_px = 3600  # 12英寸 × 300DPI
    target_height_px = 2700  # 9英寸 × 300DPI

    resized = cropped.resize((target_width_px, target_height_px),
                             Image.Resampling.LANCZOS)

    # 保存为PDF（需要安装reportlab）
    resized.save(save_path, "PDF",
                 resolution=300,
                 save_all=True)
    return resized


# 使用
resize_to_aspect_ratio("D:/1_Troy.Z/4_博士培养/4_论文写作与评审/2_论文写作/25_大论文/作图/第五章/前馈扰动补偿/右侧凸起.png",
                       "D:/1_Troy.Z/4_博士培养/4_论文写作与评审/2_论文写作/25_大论文/作图/第五章/前馈扰动补偿/右侧凸起_fix.png", target_aspect_ratio=12 / 9)