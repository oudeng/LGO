#!/usr/bin/env python3
"""
将PDF图片文件缩放并居中放置到1200x1200像素的正方形画布上。
- 如果高>宽，以高为基准缩放到1200px，左右对称留白
- 如果宽>高，以宽为基准缩放到1200px，上下对称留白
- 保持原始宽高比不变

Usage：
# 默认方式（读取当前目录下的 Graphical_Abstract.pdf）
python utility_plots/resize_pdf_to_square.py

# 指定输入输出文件
python utility_plots/resize_pdf_to_square.py input.pdf output.pdf
"""

from pdf2image import convert_from_path
from PIL import Image
import sys
import os


def resize_pdf_to_square(input_pdf: str, output_pdf: str, target_size: int = 1200, dpi: int = 300):
    """
    将PDF转换为指定尺寸的正方形PDF，保持原始比例并居中。
    
    Args:
        input_pdf: 输入PDF文件路径
        output_pdf: 输出PDF文件路径
        target_size: 目标尺寸（像素），默认1200
        dpi: 渲染PDF时的DPI，默认300
    """
    # 1. 将PDF转换为图片（使用高DPI以保证质量）
    print(f"正在读取PDF文件: {input_pdf}")
    images = convert_from_path(input_pdf, dpi=dpi)
    
    if not images:
        raise ValueError("PDF文件为空或无法读取")
    
    # 取第一页（假设是单页PDF图片）
    original_image = images[0]
    orig_width, orig_height = original_image.size
    print(f"原始尺寸: {orig_width} x {orig_height} 像素")
    
    # 2. 计算缩放比例
    # 以较大的边为基准，缩放到target_size
    if orig_height >= orig_width:
        # 高度更大或相等，以高度为基准
        scale = target_size / orig_height
        new_height = target_size
        new_width = int(orig_width * scale)
        print(f"以高度为基准缩放，比例: {scale:.4f}")
    else:
        # 宽度更大，以宽度为基准
        scale = target_size / orig_width
        new_width = target_size
        new_height = int(orig_height * scale)
        print(f"以宽度为基准缩放，比例: {scale:.4f}")
    
    print(f"缩放后尺寸: {new_width} x {new_height} 像素")
    
    # 3. 缩放原始图片（使用高质量重采样）
    resized_image = original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # 4. 创建白色背景的正方形画布
    canvas = Image.new('RGB', (target_size, target_size), 'white')
    
    # 5. 计算居中位置
    paste_x = (target_size - new_width) // 2
    paste_y = (target_size - new_height) // 2
    print(f"居中位置: ({paste_x}, {paste_y})")
    
    # 6. 将缩放后的图片粘贴到画布中央
    canvas.paste(resized_image, (paste_x, paste_y))
    
    # 7. 保存为PDF
    canvas.save(output_pdf, 'PDF', resolution=300)
    print(f"已保存输出文件: {output_pdf}")
    print(f"输出尺寸: {target_size} x {target_size} 像素")


def main():
    # 默认输入输出文件名
    input_pdf = "utility_plots/Graphical_Abstract.pdf"
    output_pdf = "utility_plots/Graphical_Abstract_1200x1200.pdf"
    
    # 支持命令行参数
    if len(sys.argv) >= 2:
        input_pdf = sys.argv[1]
    if len(sys.argv) >= 3:
        output_pdf = sys.argv[2]
    
    # 检查输入文件是否存在
    if not os.path.exists(input_pdf):
        print(f"错误: 输入文件不存在: {input_pdf}")
        sys.exit(1)
    
    try:
        resize_pdf_to_square(input_pdf, output_pdf)
        print("\n处理完成！")
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()