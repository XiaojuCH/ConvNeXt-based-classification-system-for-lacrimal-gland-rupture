# show_resize.py
# -*- coding: utf-8 -*-
# 用途：读取一张原始图像，并将其缩放到多种尺寸，拼成一张对比图，方便直观比较。

import os
import argparse
from PIL import Image
import matplotlib.pyplot as plt

def parse_sizes(size_str):
    """
    将形如 "224x224,512x512,800x600" 的字符串解析为 [(224,224), (512,512), (800,600)]。
    如果只给出单个数字，比如 "256"，则解释为 (256,256)。
    """
    sizes = []
    for part in size_str.split(','):
        part = part.strip()
        if 'x' in part:
            w, h = part.split('x')
            sizes.append((int(w), int(h)))
        else:
            n = int(part)
            sizes.append((n, n))
    return sizes

def main(image_path, sizes, output_path):
    """
    - image_path: 原始图片文件路径
    - sizes: 形如 [(w1,h1), (w2,h2), ...] 的目标尺寸列表
    - output_path: 最终拼接图的保存路径（带文件名，例如 out.png）
    """

    # 1. 读取原始图像
    img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = img.size

    # 2. 准备所有待展示的图像列表：第一张是原图，后面是各个 resize 结果
    imgs = [img]
    titles = [f"原图 ({orig_w}×{orig_h})"]
    for (w, h) in sizes:
        resized = img.resize((w, h), resample=Image.BICUBIC)
        imgs.append(resized)
        titles.append(f"{w}×{h}")

    # 3. 拼图：一行 N+1 列，保证每张子图高度一致，宽度按宽高比自动缩放
    num = len(imgs)
    display_h = 3  # 每张子图在 matplotlib 中的显示高度（inch）
    aspect_ratios = [im.size[0] / im.size[1] for im in imgs]
    display_ws = [display_h * ar for ar in aspect_ratios]  # 每张子图的显示宽度（inch）

    fig = plt.figure(figsize=(sum(display_ws), display_h))
    for i, (im, title) in enumerate(zip(imgs, titles), start=1):
        ax = fig.add_subplot(1, num, i)
        ax.imshow(im)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    # 仅当 output_path 有父目录时才尝试创建
    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✅ 已保存比较图到：{output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="展示一张图在不同尺寸下的 resize 效果")
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="待展示的原始图像路径，如 image/images/11break.png"
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="224x224,512x512,800x800",
        help=(
            "逗号分隔的目标尺寸列表，格式如“224x224,512x512,800x800”；"
            "也可以只写单个数字“256”表示256×256。"
        )
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help="保存拼接后对比图的完整路径，例如 visualize/resize_comparison.png；"
             "如果留空，则不会创建任何目录，默认会保存到当前工作目录下，以 base name 形式命名。"
    )
    args = parser.parse_args()

    # 如果用户没有传 --output_path，则默认用一个简单文件名保存在当前目录
    out_path = args.output_path.strip()
    if not out_path:
        # 从 image_path 提取文件名并加后缀 "_resize_compare.png"
        base = os.path.splitext(os.path.basename(args.image_path))[0]
        out_path = f"{base}_resize_compare.png"

    size_list = parse_sizes(args.sizes)
    main(args.image_path, size_list, out_path)
