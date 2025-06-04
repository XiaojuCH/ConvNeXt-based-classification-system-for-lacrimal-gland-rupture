# -*- coding: utf-8 -*-
# @Author :Xiaoju
# Time    : 2025/4/9

import os
import shutil
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
from torchvision import transforms
import torch.nn.functional as F

from models import ResNet18, ResNet50  # 已修改为可接受 in_channels=4
from dataset import get_dataloader  # 产出 (batch*K, 4,224,224)
from evaluator import getAUC, save_results  # 计算 AUC & 保存结果（CSV）

import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="torchvision\\.models\\._utils"
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="You are using `torch.load` with `weights_only=False`"
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="kornia\\.feature\\.lightglue"
)

# —— 导入混合精度 AMP 模块 —— #
from torch import amp


def weighted_cross_entropy(logits, targets, mask, ring_weight=5.0):
    """
    logits: 模型输出 (B, 2)
    targets: 真实标签 (B,)
    mask: 环形区域掩码 (B,) - 1表示环形区域，0表示背景
    ring_weight: 环形区域的惩罚权重
    """
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    weights = torch.ones_like(mask)
    weights[mask == 1] = ring_weight  # 环形区域错误的惩罚加重
    weighted_loss = ce_loss * weights
    return weighted_loss.mean()


def evaluate_with_patches(model, loader, device, output_root, result_name,
                          criterion=None, K=8, save_csv=True):
    """
    针对 Patch‐based 的验证/测试逻辑：
    - loader 会输出 (batch_size*K, 4, 224,224) 的 all_patches 以及 (batch_size,) 的 labels 和 [paths_list]。
    - 我们先按 patch 逐一跑过网络 → 得到 (batch_size*K, num_classes) 的 logits → softmax → (batch_size*K, num_classes) 的 probs。
    - 然后 reshape 为 (batch_size, K, num_classes)，对 K 维度取平均 → (batch_size, num_classes)。
    - 最后用 averaged probs 与原始 labels 计算 AUC/ACC，**仅在 save_csv=True 时**保存 CSV。
    """
    model.eval()
    ys, yps, losses = [], [], []

    with torch.no_grad():
        for all_patches, labels, paths, masks in loader:
            # all_patches: (N*K, 4, 224,224), labels: (N,), paths: list 长度 N
            all_patches = all_patches.to(device)  # (N*K,4,224,224)
            labels = labels.to(device)  # (N,)
            masks = masks.to(device)

            logits = model(all_patches)  # (N*K, num_classes)
            probs = torch.softmax(logits, dim=1).cpu().numpy()  # (N*K, num_classes)

            # 计算 loss：需要把 labels 重复 K 次
            if criterion is not None:
                N = labels.shape[0]
                K_ = probs.shape[0] // N
                repeated_labels = labels.unsqueeze(1).repeat(1, K_).view(-1)  # (N*K,)
                mask_flat = masks[:, 0, 0, 0]  # 假设mask是(B*K,1,1,1)
                loss_val = criterion(logits, repeated_labels, mask_flat).item()
                losses.append(loss_val)

            # reshape 为 (N, K, num_classes)
            N = labels.shape[0]
            num_classes = probs.shape[1]
            probs = probs.reshape(N, -1, num_classes)  # (N, K, num_classes)
            avg_probs = np.mean(probs, axis=1)  # (N, num_classes)

            ys.append(labels.cpu().numpy())
            yps.append(avg_probs)

    y_true = np.concatenate(ys, axis=0)  # (num_samples,)
    y_score = np.concatenate(yps, axis=0)  # (num_samples, num_classes)

    # 二分类时，用正类分数（索引 1）
    y_score_pos = y_score[:, 1]

    auc = getAUC(y_true, y_score_pos, task="binary")
    preds = (y_score_pos > 0.5).astype(int)
    acc = (preds == y_true).mean()

    avg_loss = np.mean(losses) if losses else None
    print(f'[{result_name}] AUC: {auc:.4f}  ACC: {acc:.4f}' +
          (f'  LOSS: {avg_loss:.4f}' if avg_loss is not None else ''))

    if save_csv:
        os.makedirs(output_root, exist_ok=True)
        save_results(y_true, y_score, os.path.join(output_root, result_name + ".csv"))

    return auc, acc, avg_loss


def main(input_root, output_root, num_epoch, model_name):
    # —— 1. 超参定义 —— #
    # 输入通道数改为 4（前三通道为灰度，第四通道为径向距离先验）
    in_channels, num_classes = 4, 2
    batch_size_train = 16
    batch_size_eval = 8
    lr, wd = 5e-4, 1e-4
    early_stop_patience = 15
    num_workers = 4

    # Patch‐based 相关参数（需根据实际对齐圆心/半径调整）
    patch_size = 224
    center = (320, 240)  # 圆心坐标 (x0,y0)——建议按给定数据集对齐后测得的固定值
    radius = 200  # 圆环半径（像素）
    K = 8  # 每张图切取的 Patch 数量

    # —— 2. 数据预处理与增强 —— #
    # 注意：这里只对前三通道（灰度复制到 3 通道）做增强，Dataset 内会自动拼第 4 通道
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ColorJitter(brightness=0.05, contrast=0.05),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    eval_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    # —— 3. 构造 DataLoader —— #
    train_loader = get_dataloader(
        root_dir=input_root,
        split='train',
        transform=train_tf,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers,
        patch_size=patch_size,
        center=center,
        radius=radius,
        train=True,
        K=K,
    )
    val_loader = get_dataloader(
        root_dir=input_root,
        split='val',
        transform=eval_tf,
        batch_size=batch_size_eval,
        shuffle=False,
        num_workers=num_workers,
        patch_size=patch_size,
        center=center,
        radius=radius,
        train=False,
        K=K,
    )
    test_loader = get_dataloader(
        root_dir=input_root,
        split='test',
        transform=eval_tf,
        batch_size=batch_size_eval,
        shuffle=False,
        num_workers=num_workers,
        patch_size=patch_size,
        center=center,
        radius=radius,
        train=False,
        K=K,
    )

    # —— 4. 模型、损失函数、优化器、调度器 —— #
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model_cls = ResNet50 if model_name.lower() == 'resnet50' else ResNet18
    model = model_cls(in_channels=in_channels, num_classes=num_classes).to(device)

    # 这里使用带类别权重的 Focal Loss（也可以换为 CrossEntropy + label_smoothing）
    cnt0 = len(os.listdir(os.path.join(input_root, 'train', 'normal')))
    cnt1 = len(os.listdir(os.path.join(input_root, 'train', 'break')))
    total = cnt0 + cnt1
    w0 = total / (2 * cnt0)
    w1 = total / (2 * cnt1)
    class_weights = torch.tensor([w0, w1]).to(device)

    def focal_loss(inputs, targets, alpha=None, gamma=2):
        logpt = -nn.functional.cross_entropy(inputs, targets, weight=alpha, reduction='none')
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** gamma) * logpt
        return loss.mean()

    criterion = lambda x, y, mask: weighted_cross_entropy(x, y, mask, ring_weight=5.0)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

    # —— ★ 混合精度 AMP —— #
    scaler = amp.GradScaler()  # 默认 device_type='cuda'

    # —— 5. 训练 + 验证循环 —— #
    run_best_auc = 0.0
    no_improve = 0

    for epoch in trange(1, num_epoch + 1, desc="Epochs"):
        # —— 5.1 训练模式 —— #
        model.train()
        train_loss_accum = 0.0

        for imgs, labels, paths, masks in train_loader:  # 添加 masks 信息
            # imgs: (batch_size*K, 4,224,224), labels: (batch_size,)
            imgs = imgs.to(device)
            labels = labels.to(device)
            masks = masks.to(device)  # 将 masks 移动到设备上

            optimizer.zero_grad()

            # —— 在 torch.amp.autocast 作用域内前向和计算 loss —— #
            with amp.autocast(device_type='cuda'):
                logits = model(imgs)  # (batch_size*K, 2)

                # 重复 labels 为 (batch*K,) 以匹配 logits
                N = labels.shape[0]
                repeated_labels = labels.unsqueeze(1).repeat(1, K).view(-1)  # (N*K,)
                mask_flat = masks[:, 0, 0, 0]  # 假设 mask 是 (B*K,1,1,1)
                loss = criterion(logits, repeated_labels, mask_flat)  # 传递 mask 参数

            # —— 用 scaler 进行反向传播和步进 —— #
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 按“图”统计 loss：loss 本身是平均每 patch，这里乘回 patch 数即可
            train_loss_accum += loss.item() * N

        avg_train_loss = train_loss_accum / len(train_loader.dataset)
        print(f"[Epoch {epoch}/{num_epoch}] Train Loss: {avg_train_loss:.10f}")

        # —— 5.2 验证模式（不保存 CSV） —— #
        print(f"[Epoch {epoch}/{num_epoch}] Validation:")
        val_auc, val_acc, val_loss = evaluate_with_patches(
            model, val_loader, device, output_root, f'val_epoch{epoch}',
            criterion, K=K, save_csv=False
        )
        scheduler.step(val_auc)

        # —— Early Stopping & 保存本次 run_best —— #
        if val_auc > run_best_auc:
            run_best_auc = val_auc
            save_path = os.path.join(output_root, 'checkpoints', 'run_best.pth')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({'net': model.state_dict(), 'epoch': epoch, 'val_auc': val_auc}, save_path)
            no_improve = 0
            print(f">>> New run-best at epoch {epoch}, AUC={val_auc:.4f} ACC={val_acc:.4f}")
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                print(f"No AUC improvement in {early_stop_patience} epochs. Stopping training.")
                break

    # —— 6. 测试阶段（保留 CSV） —— #
    print("\n===== Testing RUN-BEST model =====")
    ckpt = torch.load(os.path.join(output_root, 'checkpoints', 'run_best.pth'))
    model.load_state_dict(ckpt['net'])
    test_auc, test_acc, test_loss = evaluate_with_patches(
        model, test_loader, device, output_root, 'test_run_best',
        criterion, K=K, save_csv=True
    )
    print(f"[Test Result] AUC: {test_auc:.4f}  ACC: {test_acc:.4f}" +
          (f"  LOSS: {test_loss:.10f}" if test_loss is not None else ""))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root', required=True,
                        help="数据集根目录，应包含 train/val/test 三个子文件夹")
    parser.add_argument('--output_root', default='./output', help="结果保存目录")
    parser.add_argument('--num_epoch', type=int, default=20, help="训练轮数")
    parser.add_argument('--model', default='resnet50',
                        choices=['resnet18', 'resnet50'], help="网络型号")
    args = parser.parse_args()

    main(args.input_root, args.output_root, args.num_epoch, args.model)
