# show_patch_sampling.py
# -*- coding: utf-8 -*-
# @Author : ChatGPT
# Time    : 2025/06/05

import os
import argparse
import math
import cv2
import numpy as np

def draw_patches(image, center, radius, patch_size, K, output_path=None):
    """
    在输入图像上绘制采样圆环及对应的 K 个 patch 矩形，便于可视化 train.py 中的参数含义。

    参数：
        image (ndarray): BGR 格式的原始图像。
        center (tuple of int): 圆心坐标 (x0, y0)。
        radius (int): 圆环半径（像素）。
        patch_size (int): Patch 边长（正方形，单位像素）。
        K (int): 圆环上等距离采样 patch 的数量。
        output_path (str, optional): 如果给出，则将可视化结果保存到该路径。

    返回：
        vis (ndarray): 绘制好 patch/圆环的可视化图像（BGR）。
    """
    vis = image.copy()
    H, W = vis.shape[:2]
    x0, y0 = center

    # 1) 在图像上画出圆环中心和半径
    cv2.circle(vis, (x0, y0), 3, (0, 255, 0), -1)  # 绿色小圆点
    cv2.circle(vis, (x0, y0), radius, (0, 255, 0), 2)  # 绿色圆轮廓

    # 2) 计算并绘制 K 个 patch 的矩形框
    hs = patch_size // 2
    for i in range(K):
        angle = 2 * math.pi * i / K
        xi = int(x0 + radius * math.cos(angle))
        yi = int(y0 + radius * math.sin(angle))

        x1 = xi - hs
        y1 = yi - hs
        x2 = xi + hs
        y2 = yi + hs

        # 边界检查
        if x1 < 0:
            x1, x2 = 0, patch_size
        if y1 < 0:
            y1, y2 = 0, patch_size
        if x2 > W:
            x2 = W
            x1 = W - patch_size
        if y2 > H:
            y2 = H
            y1 = H - patch_size

        # 绘制矩形框（蓝色）
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.circle(vis, (xi, yi), 3, (0, 0, 255), -1)  # 红色中心点

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, vis)

    return vis


def parse_args():
    parser = argparse.ArgumentParser(
        description="可视化 train.py 中 Patch 采样参数：center、radius、patch_size、K"
    )
    parser.add_argument(
        "--image_path", type=str, default="22break.png",
        help="要展示的样本图像路径（BGR 格式），默认为 image/images/sample.png"
    )
    parser.add_argument(
        "--center", nargs=2, type=int, default=[630, 550],
        help="圆环中心坐标（x0 y0），默认为 320 240"
    )
    parser.add_argument(
        "--radius", type=int, default=400,
        help="圆环半径（像素），默认为 200"
    )
    parser.add_argument(
        "--patch_size", type=int, default=448,
        help="Patch 大小（正方形边长，像素），默认为 224"
    )
    parser.add_argument(
        "--K", type=int, default=8,
        help="圆环上采样 Patch 数量，默认为 8"
    )
    parser.add_argument(
        "--output_path", type=str, default="visualize/patch_layout.png",
        help="可视化结果保存路径，默认为 visualize/patch_layout.png"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 读取图像（确保使用 BGR）
    img = cv2.imread(args.image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像：{args.image_path}")

    center = (args.center[0], args.center[1])
    radius = args.radius
    patch_size = args.patch_size
    K = args.K

    vis = draw_patches(
        img,
        center,
        radius,
        patch_size,
        K,
        args.output_path
    )

    # 显示结果
    cv2.imshow("Patch Sampling Visualization", vis)
    print(f"已保存可视化结果到：{args.output_path}")
    print("按任意键关闭窗口……")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
