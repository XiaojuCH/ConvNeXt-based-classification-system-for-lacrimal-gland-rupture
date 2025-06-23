#!/usr/bin/env python3
# rename_images.py
# 将指定文件夹中的图片按从1开始的数字顺序重命名，保留文件扩展名

import os
import argparse

# 支持的图片后缀名单
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}


def get_image_files(folder):
    """
    获取 folder 中所有图片文件（不递归子目录），按文件名排序
    """
    files = []
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        if os.path.isfile(path):
            ext = os.path.splitext(fname)[1].lower()
            if ext in IMAGE_EXTS:
                files.append(fname)
    files.sort()
    return files


def rename_images(folder, start=1, padding=0):
    """
    将 folder 中的图片按从 start 开始的数字重命名。
    如果 padding > 0，则数字部分左侧填充零至固定宽度。
    """
    files = get_image_files(folder)
    total = len(files)

    # 如果未指定 padding，则根据总数自动计算宽度
    if padding <= 0:
        padding = len(str(total + start - 1))

    for idx, fname in enumerate(files, start):
        old_path = os.path.join(folder, fname)
        ext = os.path.splitext(fname)[1]
        new_name = f"{str(idx).zfill(padding)}{ext}"
        new_path = os.path.join(folder, new_name)

        # 如果目标文件已存在，先报错提示
        if os.path.exists(new_path):
            print(f"Skip: target {new_name} already exists.")
            continue

        os.rename(old_path, new_path)
        print(f"Renamed: {fname} -> {new_name}")


def main():
    parser = argparse.ArgumentParser(
        description="将指定文件夹中的图片按数字顺序重命名，从1开始。"
    )
    parser.add_argument(
        'folder', help='要重命名的图片所在文件夹'
    )
    parser.add_argument(
        '--start', '-s',
        type=int,
        default=1,
        help='起始数字，默认为1'
    )
    parser.add_argument(
        '--padding', '-p',
        type=int,
        default=0,
        help='数字部分宽度，默认为自动计算'
    )

    args = parser.parse_args()
    if not os.path.isdir(args.folder):
        print(f"Error: {args.folder} 不是有效的目录")
        return

    rename_images(args.folder, args.start, args.padding)


if __name__ == '__main__':
    main()
