# dataset.py

import os
import math
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class EyeBreakDataset(Dataset):
    def __init__(self,
                 root_dir,         # e.g. 'data'
                 split,            # 'train' 或 'val' 或 'test'
                 transform=None,   # 只对前 3 通道灰度做增强
                 patch_size=224,
                 center=(320,240), # 原图已对齐后圆心
                 radius=200,       # 圆环半径
                 train=True,
                 K=8):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.patch_size = patch_size
        self.center = center
        self.radius = radius
        self.train = train
        self.K = K

        # 收集目录下所有图片路径与标签
        self.images = []
        for cls_name, label in [('normal', 0), ('break', 1)]:
            folder = os.path.join(root_dir, split, cls_name)
            if not os.path.isdir(folder):
                raise ValueError(f"目录不存在: {folder}")
            for fname in os.listdir(folder):
                if fname.lower().endswith(('.png','jpg','jpeg','bmp')):
                    self.images.append((os.path.join(folder, fname), label))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        img = Image.open(img_path).convert('L')  # 灰度
        img_np = np.array(img)                  # (H_orig, W_orig)

        patches = []
        masks = []  # 添加 masks 列表

        for i in range(self.K):
            # 训练时随机角度 + ±10px 半径抖动；验证/测试时等分角度 + 定半径
            if self.train:
                angle = np.random.uniform(0, 2*np.pi)
                r_off = np.random.uniform(self.radius-10, self.radius+10)
            else:
                angle = 2*np.pi * i / self.K
                r_off = self.radius

            x0, y0 = self.center
            xi = int(x0 + r_off * math.cos(angle))
            yi = int(y0 + r_off * math.sin(angle))
            hs = self.patch_size // 2

            x1 = xi - hs
            y1 = yi - hs
            x2 = x1 + self.patch_size
            y2 = y1 + self.patch_size

            # 边界检查
            H, W = img_np.shape
            if x1 < 0:
                x1, x2 = 0, self.patch_size
            if y1 < 0:
                y1, y2 = 0, self.patch_size
            if x2 > W:
                x2 = W
                x1 = W - self.patch_size
            if y2 > H:
                y2 = H
                y1 = H - self.patch_size

            patch_np = img_np[y1:y2, x1:x2]  # (224,224)
            patch_pil = Image.fromarray(patch_np)

            # ——— 前 3 通道：灰度重复或增强 ———
            if self.transform:
                patch_t_3 = self.transform(patch_pil)  # (3,224,224)
            else:
                to_tensor = transforms.Compose([
                    transforms.Grayscale(num_output_channels=3),
                    transforms.Resize((self.patch_size, self.patch_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5]*3, [0.5]*3),
                ])
                patch_t_3 = to_tensor(patch_pil)

            # ——— 第 4 通道：径向距离归一化（radial） ———
            # Patch 中心坐标 (cx, cy)：在 patch 内部坐标系下
            cx = xi - x1
            cy = yi - y1
            Hs = self.patch_size

            yy = np.arange(0, Hs)
            xx = np.arange(0, Hs)
            grid_y, grid_x = np.meshgrid(yy, xx, indexing='ij')
            dist = np.sqrt((grid_x - cx) ** 2 + (grid_y - cy) ** 2)
            # 使用高斯分布增强边缘关注度，σ控制聚焦强度
            sigma = self.radius * 0.1  # 可调整参数
            radial = np.exp(-0.5 * ((dist - self.radius) / sigma) ** 2)
            radial = (radial - radial.min()) / (radial.max() - radial.min())
            radial = np.clip(radial, 0, 1)
            radial_tensor = torch.from_numpy(radial).unsqueeze(0).float()  # (1,224,224)

            # 拼接成 4 通道
            patch_all = torch.cat([patch_t_3, radial_tensor], dim=0)  # (4,224,224)
            patches.append(patch_all)

            # 生成 mask
            mask = torch.ones((1, 1, 1)) if dist.min() <= self.radius + 10 and dist.max() >= self.radius - 10 else torch.zeros((1, 1, 1))
            masks.append(mask)

        patches = torch.stack(patches, dim=0)  # (K,4,224,224)
        masks = torch.stack(masks, dim=0)  # (K, 1, 1, 1)
        return patches, label, img_path, masks

def collate_fn_val(batch):
    """
    批量化处理：batch 中每个元素为 (patches, label, img_path, masks)。
    最终输出：
      all_patches: (batch_size*K, 4,224,224)
      labels:      (batch_size,)
      paths_list:  [img_path_1, img_path_2, …]
      masks:       (batch_size*K, 1, 1, 1)
    """
    patches_list, labels_list, paths_list, masks_list = [], [], [], []
    for all_patches, label, img_path, masks in batch:
        patches_list.append(all_patches)  # 每个 all_patches=(K,4,224,224)
        labels_list.append(label)
        paths_list.append(img_path)
        masks_list.append(masks)

    batch_patches = torch.cat(patches_list, dim=0)    # (batch*K,4,224,224)
    labels = torch.tensor(labels_list, dtype=torch.long)
    masks = torch.cat(masks_list, dim=0)  # 拼接 masks
    return batch_patches, labels, paths_list, masks

def get_dataloader(root_dir,
                   split,
                   transform,
                   batch_size,
                   shuffle,
                   num_workers,
                   patch_size,
                   center,
                   radius,
                   train,
                   K):
    """
    返回一个 DataLoader，批次包含 (batch*K,4,224,224) 的 patch 张量
    """
    dataset = EyeBreakDataset(
        root_dir=root_dir,
        split=split,
        transform=transform,
        patch_size=patch_size,
        center=center,
        radius=radius,
        train=train,
        K=K
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn_val
    )
    return loader


if __name__ == '__main__':
    # 测试示例：只是验证尺寸
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    loader = get_dataloader(
        root_dir='data',
        split='train',
        transform=train_transform,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        patch_size=224,
        center=(320,240),
        radius=200,
        train=True,
        K=4
    )
    patches, labels, paths, masks = next(iter(loader))
    print(f"patches shape: {patches.shape}, labels: {labels}, paths: {paths}, masks: {masks.shape}")