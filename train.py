# train.py
# -*- coding: utf-8 -*-
# @Author :Xiaoju
# Time    : 2025/4/9

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
from torchvision import transforms
from dataset import get_dataloader    # 上面已经改成统一 4×512×512
from evaluator import getAUC, save_results
import torch.utils.data as data
from PIL import Image
from confusion import (
    plot_confusion,
    save_classification_report,
    plot_multiclass_pr_curve,
    plot_loss_curve,
    visualize_misclassified
)
import warnings
from models import create_model  # 确保引入的是你修改后 forward 返回 (logits, attn_feat)


class FocalRingLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        return loss.mean()

def ring_regularization(attn_feat, radial_map, weight=0.1):
    """
    attn_feat: (B, C, H, W)  特征图
    radial_map: (B,1,H_orig,W_orig)  原始径向图
    """
    # upsample radial_map 到 attn_feat 尺寸
    _,C,H,W = attn_feat.shape
    rm = nn.functional.interpolate(radial_map, size=(H,W), mode='bilinear', align_corners=False)
    # 取一个通道平均响应
    mean_feat = attn_feat.mean(dim=1)  # (B, H, W)
    # 只保留环带 0.3~0.7 区间
    mask = ((rm[:,0] > 0.3) & (rm[:,0] < 0.7)).float()  # (B,H,W)
    # 惩罚模型在环带外响应，或鼓励环带上响应
    ring_resp = (mean_feat * mask).mean()
    return - weight * ring_resp

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

from torch import amp  # 混合精度 AMP
from torch.optim.lr_scheduler import LambdaLR


def evaluate(model, loader, device, output_root, name, criterion=None, save_csv=True):
    model.eval()
    ys, yps, losses = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)       # (B,4,512,512)
            labels = labels.to(device)   # (B,)

            logits = model(imgs)         # (B,2)
            probs = torch.softmax(logits, dim=1).cpu().numpy()  # (B,2)

            if criterion is not None:
                loss_val = criterion(logits, labels).item()
                losses.append(loss_val)

            ys.append(labels.cpu().numpy())
            yps.append(probs)

    y_true = np.concatenate(ys, axis=0)    # (N,)
    y_score = np.concatenate(yps, axis=0)  # (N,2)
    y_score_pos = y_score[:, 1]            # 取“断裂”类别的概率

    auc = getAUC(y_true, y_score_pos, task="binary")
    preds = (y_score_pos > 0.5).astype(int)
    acc = (preds == y_true).mean()

    avg_loss = np.mean(losses) if losses else None
    print(f"[{name}] AUC: {auc:.4f}  ACC: {acc:.4f}" +
          (f"  LOSS: {avg_loss:.4f}" if avg_loss is not None else ""))

    if save_csv:
        os.makedirs(output_root, exist_ok=True)
        save_results(y_true, y_score, os.path.join(output_root, name + ".csv"))

    return auc, acc, avg_loss


class WholeImageDataset(torch.utils.data.Dataset):
    """
    专门给画混淆矩阵 / 报告用的“整图”Dataset：
    同样把原图灰度化、resize到 512×512，再拼 4 通道。
    """
    def __init__(self, root_dir: str, split: str, transform=None):
        super().__init__()
        self.samples = []
        for cls_name, lb in [('normal', 0), ('break', 1)]:
            folder = os.path.join(root_dir, split, cls_name)
            if not os.path.isdir(folder):
                raise ValueError(f"目录不存在: {folder}")
            for fn in sorted(os.listdir(folder)):
                if fn.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.samples.append((os.path.join(folder, fn), lb))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, lb = self.samples[idx]
        img_gray = Image.open(path).convert('L')
        if img_gray.size == (640, 480):
            img_gray = img_gray.resize((1272, 920), Image.BICUBIC)
        img_gray = img_gray.resize((512, 512), Image.BICUBIC)

        if self.transform:
            img3 = self.transform(img_gray)  # (3,512,512)
        else:
            t = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3),
            ])
            img3 = t(img_gray)

        # 生成径向图 (1,512,512)
        cy, cx = 512/2.0, 512/2.0
        ys = torch.arange(0, 512).view(512, 1).expand(512, 512)
        xs = torch.arange(0, 512).view(1, 512).expand(512, 512)
        dist = torch.sqrt((xs - cx)**2 + (ys - cy)**2)
        radial = (dist / (512/2.0)).clamp(0, 1).unsqueeze(0).float()  # (1,640,640)

        img4 = torch.cat([img3, radial], dim=0)  # (4,640,640)
        return img4, lb


def _warmup_lambda(epoch):
    return min(1.0, (epoch + 1) / 5)

def main(input_root, output_root, num_epoch, model_name):
    # 1. 超参定义
    in_channels, num_classes = 4, 2
    batch_size = 16
    grad_accum_steps = 2  # 梯度累进步数，例如累积 2 个 batch 再更新
    lr = 5e-5
    wd = 5e-4
    early_stop_patience = 12
    num_workers = 4

    # 2. DataLoader（所有输入都已经在 dataset 里被 resize 到 512×512）
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10, fill=(128,)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    eval_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    train_loader = get_dataloader(
        root_dir=input_root,
        split='train',
        transform=train_tf,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = get_dataloader(
        root_dir=input_root,
        split='val',
        transform=eval_tf,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    test_loader = get_dataloader(
        root_dir=input_root,
        split='test',
        transform=eval_tf,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    # “整图”用来画混淆 / 报告
    full_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    train_loader_full = data.DataLoader(
        WholeImageDataset(input_root, 'train', transform=full_tf),
        batch_size=16, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    val_loader_full = data.DataLoader(
        WholeImageDataset(input_root, 'val', transform=full_tf),
        batch_size=16, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader_full = data.DataLoader(
        WholeImageDataset(input_root, 'test', transform=full_tf),
        batch_size=16, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    # 3. 模型、损失、优化器、调度器
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = create_model(
        model_name=model_name,      # 'resnet18' 或 'resnet50'
        in_channels=in_channels,
        num_classes=num_classes,
        pretrained=True,
        local_weights_dir='D:/Projects_/Tears_Check/pretrained'
    ).to(device)

    # criterion = FocalRingLoss(alpha=0.8, gamma=2.0).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.5]).to(device))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler_warm = LambdaLR(optimizer, lr_lambda=_warmup_lambda)

    # 在此之上再用 ReduceLROnPlateau
    scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5, threshold=1e-4
    )
    scaler = amp.GradScaler()

    train_losses, val_losses = [], []
    run_best_auc = 0.0
    early_cnt = 0
    for epoch in trange(1, num_epoch+1, desc="Epochs"):
        # —— 训练 —— #
        model.train()
        total_train_loss = 0.0
        optimizer.zero_grad()
        for step, (imgs, labels) in enumerate(train_loader, start=1):
            imgs  = imgs.to(device)   # (B,4,512,512)
            labels= labels.to(device)

            with amp.autocast(device_type='cuda'):
                out = model(imgs)
                logits, attn_feat = out if isinstance(out,(tuple,list)) else (out, None)
                loss = criterion(logits, labels) / grad_accum_steps
                if attn_feat is not None:
                    radial_map = imgs[:,3:4,:,:]
                    loss = loss + ring_regularization(attn_feat, radial_map, weight=0.1)

            scaler.scale(loss).backward()
            total_train_loss += loss.item() * imgs.size(0)
            if step % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler_warm.step()

        # 尾部更新
        if len(train_loader) % grad_accum_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()


        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.6f}")

        # —— 验证 —— #
        model.eval()
        total_val_loss = 0.0
        ys, yps = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs   = imgs.to(device)
                labels = labels.to(device)
                out    = model(imgs)
                logits,_ = out if isinstance(out,(tuple,list)) else (out,None)
                loss   = criterion(logits, labels)
                total_val_loss += loss.item()*imgs.size(0)
                probs  = torch.softmax(logits,dim=1).cpu().numpy()
                ys.append(labels.cpu().numpy())
                yps.append(probs)

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        y_true = np.concatenate(ys,axis=0)
        y_score= np.concatenate(yps,axis=0)
        val_auc= getAUC(y_true, y_score[:,1], task="binary")
        val_acc= ((y_score[:,1]>0.5).astype(int)==y_true).mean()
        print(f"[Epoch {epoch}] Val AUC: {val_auc:.4f}  ACC: {val_acc:.4f}  Val Loss: {avg_val_loss:.6f}")

        scheduler_plateau.step(val_auc)
        if val_auc > run_best_auc + 1e-5:
            run_best_auc = val_auc
            early_cnt = 0
            ckpt_path = os.path.join(output_root,'checkpoints','run_best.pth')
            os.makedirs(os.path.dirname(ckpt_path),exist_ok=True)
            torch.save({'net':model.state_dict(),'epoch':epoch,'val_auc':val_auc}, ckpt_path)
            print(f">>> New run‐best at epoch {epoch}: AUC={val_auc:.4f} ACC={val_acc:.4f}")
        else:
            early_cnt += 1
            if early_cnt >= early_stop_patience:
                print(f"No AUC improvement in {early_stop_patience} epochs. Early stopping.")
                break

    # 5. 测试阶段 (Final + Run-Best)
    print("\n===== Testing FINAL Model =====")
    test_auc, test_acc, test_loss = evaluate(
        model, test_loader, device, output_root, 'test_final',
        criterion, save_csv=True
    )
    # print(f"[Test Final] AUC: {test_auc:.4f}  ACC: {test_acc:.4f}")

    print("\n===== Testing RUN‐BEST Model =====")
    ckpt = torch.load(os.path.join(output_root, 'checkpoints', 'run_best.pth'))
    model.load_state_dict(ckpt['net'])
    test_auc_rb, test_acc_rb, test_loss_rb = evaluate(
        model, test_loader, device, output_root, 'test_run_best',
        criterion, save_csv=True
    )
    # print(f"[Test Run-Best] AUC: {test_auc_rb:.4f}  ACC: {test_acc_rb:.4f}")

    # 6. 生成可视化报告
    print("\n===== 生成可视化报告 =====")
    best_model = create_model(
        model_name=model_name,
        in_channels=in_channels,
        num_classes=num_classes,
        pretrained=False
    ).to(device)
    ckpt2 = torch.load(os.path.join(output_root, 'checkpoints', 'run_best.pth'), map_location=device)
    best_model.load_state_dict(ckpt2['net'])
    best_model.eval()

    best_thresh = 0.5
    class_names = ["Normal", "Break"]

    # 6.1 绘制混淆矩阵：Train / Val / Test
    plot_confusion(
        best_model,
        train_loader_full,
        device,
        os.path.join(output_root, "confusion_train.png"),
        class_names
    )
    plot_confusion(
        best_model,
        val_loader_full,
        device,
        os.path.join(output_root, "confusion_val.png"),
        class_names
    )
    plot_confusion(
        best_model,
        test_loader_full,
        device,
        os.path.join(output_root, "confusion_test.png"),
        class_names
    )

    # 6.2 保存分类报告：Train / Val / Test
    save_classification_report(
        best_model, train_loader_full, device,
        out_txt_path=os.path.join(output_root, "report_train_runbest.txt"),
        best_thresh=best_thresh, class_names=class_names
    )
    save_classification_report(
        best_model, val_loader_full, device,
        out_txt_path=os.path.join(output_root, "report_val_runbest.txt"),
        best_thresh=best_thresh, class_names=class_names
    )
    save_classification_report(
        best_model, test_loader_full, device,
        out_txt_path=os.path.join(output_root, "report_test_runbest.txt"),
        best_thresh=best_thresh, class_names=class_names
    )

    # 6.3 绘制 Loss 曲线
    plot_loss_curve(
        train_losses, val_losses,
        out_path=os.path.join(output_root, "loss_curve.png")
    )

    print("所有可视化已生成，保存在：", output_root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_root', required=True,
        help="数据集根目录，应包含 train/val/test 三个子文件夹"
    )
    parser.add_argument(
        '--output_root', default='./output', help="结果保存目录"
    )
    parser.add_argument(
        '--num_epoch', type=int, default=60, help="训练轮数"
    )
    parser.add_argument(
        '--model', default='convnext_tiny',
        choices=['resnet18', 'resnet50', 'convnext_tiny', 'convnext_small'], help="网络型号"
    )
    args = parser.parse_args()

    main(args.input_root, args.output_root, args.num_epoch, args.model)
