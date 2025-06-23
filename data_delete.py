# -*- coding: utf-8 -*-
# @Author :Xiaoju
# Time : 2025/6/23 上午9:57

#!/usr/bin/env python3
# remove_random_images.py
# 随机删除指定目录下图片数据集的 40%，保留原有文件路径结构

import os
import random
import argparse

# 支持的图片后缀名单
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}


def collect_image_paths(root_dir):
    """
    遍历 root_dir，收集所有图片文件的完整路径
    """
    paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in IMAGE_EXTS:
                paths.append(os.path.join(dirpath, fname))
    return paths


def remove_random(paths, fraction, seed=None):
    """
    从 paths 中随机选择 fraction 比例的文件进行删除
    """
    if seed is not None:
        random.seed(seed)
    n_remove = int(len(paths) * fraction)
    to_remove = random.sample(paths, n_remove)

    for fpath in to_remove:
        try:
            os.remove(fpath)
            print(f"Removed: {fpath}")
        except Exception as e:
            print(f"Failed to remove {fpath}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="随机删除图片数据集中的一定比例文件，保留原路径结构。"
    )
    parser.add_argument(
        'root_dir',
        help='数据集根目录，将递归遍历其中所有图片'
    )
    parser.add_argument(
        '--fraction', '-f',
        type=float,
        default=0.4,
        help='要删除的比例，默认为0.4（即删除40%%）'
    )
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=None,
        help='随机种子，可选'
    )

    args = parser.parse_args()

    if not os.path.isdir(args.root_dir):
        print(f"Error: {args.root_dir} 不是有效的目录")
        return

    all_paths = collect_image_paths(args.root_dir)
    print(f"Found {len(all_paths)} image files in {args.root_dir}.")
    remove_random(all_paths, args.fraction, args.seed)


if __name__ == '__main__':
    main()
