# dataset.py

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def calculate_ring_map(height, width, r0=0.5, sigma=0.1, cx=None, cy=None):
    """
    生成一个“环带响应图” (1, H, W)，
    在归一化半径 r0 附近有最高响应，宽度由 sigma 控制。
    r0: 环带中心半径占 (min(H,W)/2) 的比例，默认 0.5；
    sigma: 环带宽度，默认 0.1。
    """
    if cx is None: cx = (width - 1) / 2.0
    if cy is None: cy = (height - 1) / 2.0
    ys = torch.arange(0, height).view(height, 1).expand(height, width)
    xs = torch.arange(0, width).view(1, width).expand(height, width)
    dist = torch.sqrt((xs - cx)**2 + (ys - cy)**2)
    max_rad = min(height, width) / 2.0
    # 归一化到 [0,1]
    norm_dist = (dist / max_rad).clamp(0, 1)
    # 环带高斯响应
    ring = torch.exp(-0.5 * ((norm_dist - r0) / sigma)**2)
    # 归一化到 [0,1]
    ring = (ring - ring.min()) / (ring.max() - ring.min() + 1e-8)
    return ring.unsqueeze(0)  # (1, H, W)


class EyeBreakDataset(Dataset):
    """
    4 通道输入：
      - 前三通道：灰度复制 + 边缘抑制
      - 第四通道：环带高斯响应图
    """
    def __init__(self, root_dir: str, split: str, transform=None,
                 r0: float = 0.5, sigma: float = 0.1, edge_border: int = 30):
        super().__init__()
        self.transform = transform
        self.r0 = r0
        self.sigma = sigma
        self.edge_border = edge_border

        self.samples = []
        for cls_name, label in [('normal', 0), ('break', 1)]:
            folder = os.path.join(root_dir, split, cls_name)
            if not os.path.isdir(folder):
                raise ValueError(f"目录不存在: {folder}")
            for fn in sorted(os.listdir(folder)):
                if fn.lower().endswith(('.png','.jpg','.jpeg','.bmp')):
                    self.samples.append((os.path.join(folder, fn), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # 1. 读取灰度图并 resize 到 512×512
        img = Image.open(path).convert('L')
        if img.size == (640, 480):
            img = img.resize((1272, 920), Image.BICUBIC)
        img = img.resize((512, 512), Image.BICUBIC)

        # 2. 灰度复制到 3 通道 + ToTensor + Normalize
        if self.transform:
            img3 = self.transform(img)  # (3,512,512)
        else:
            to_tensor = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])
            img3 = to_tensor(img)

        # 3. 生成环带高斯响应图 (1,512,512)
        ring_map = calculate_ring_map(
            512, 512,
            r0=self.r0,
            sigma=self.sigma
        )  # (1,512,512)

        # 4. 边缘抑制掩码 (1,512,512)，减弱图像最边缘信息
        edge_mask = torch.ones_like(ring_map)
        b = self.edge_border
        edge_mask[:, :b, :] = 0.2
        edge_mask[:, -b:, :] = 0.2
        edge_mask[:, :, :b] = 0.2
        edge_mask[:, :, -b:] = 0.2

        # 5. 对前三通道做边缘抑制
        img3 = img3 * edge_mask

        # 6. 拼成 4 通道
        img4 = torch.cat([img3, ring_map], dim=0)  # (4,512,512)

        return img4, label


def collate_fn_val(batch):
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return imgs, labels


def get_dataloader(root_dir: str,
                   split: str,
                   transform,
                   batch_size: int,
                   shuffle: bool,
                   num_workers: int,
                   r0=0.375,  # 环带中心
                   sigma=0.065,  # 环带宽度
                   edge_border=30,
                   **kwargs):
    dataset = EyeBreakDataset(
        root_dir=root_dir,
        split=split,
        transform=transform,
        **kwargs
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn_val
    )


if __name__ == '__main__':
    # 简单测试
    small_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    loader = get_dataloader(
        root_dir='data',
        split='train',
        transform=small_tf,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        r0=0.375,            # 环带半径占最大半径的比例
        sigma=0.14,         # 环带宽度
        edge_border=30     # 边缘抑制宽度
    )
    imgs, labels = next(iter(loader))
    print("imgs.shape =", imgs.shape, ", labels =", labels)
    # 期望：imgs.shape = torch.Size([2, 4, 512, 512])
