# models.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from safetensors.torch import load_file
import math


class RingEnhancer(nn.Module):
    """
    环形特征增强：融合特征图与径向先验，生成局部环带权重
    """
    def __init__(self, in_channels, hidden_ratio=4):
        super().__init__()
        hidden = in_channels // hidden_ratio
        self.conv1 = nn.Conv2d(in_channels+1, hidden, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(hidden, in_channels, kernel_size=3, padding=1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()

    def forward(self, feat, radial):
        # feat: (B,C,H,W), radial: (B,1,H,W)
        x = torch.cat([feat, radial], dim=1)       # (B,C+1,H,W)
        x = self.act(self.conv1(x))               # (B,hidden,H,W)
        w = self.sig(self.conv2(x))               # (B,C,H,W)
        return feat * w                           # 加权输出


# —— 在文件顶部新增 SpatialAttention ——
class SpatialAttention(nn.Module):
    """ 简易 CBAM‐style 空间注意力 """
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # x: (B,C,H,W)
        # 按 channel 做 max & avg
        max_pool = x.max(dim=1, keepdim=True)[0]   # (B,1,H,W)
        avg_pool = x.mean(dim=1, keepdim=True)     # (B,1,H,W)
        y = torch.cat([max_pool, avg_pool], dim=1) # (B,2,H,W)
        y = self.conv(y)                           # (B,1,H,W)
        y = self.sigmoid(y)
        return x * y

# ─── 1. RingAttention（不变） ─────────────────────────────────────────────────────────
# models.py (RingAttention)

class RingAttention(nn.Module):

    def __init__(self, in_channels, reduction=16, ring_center=0.60, sigma=0.06):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
        self.center = ring_center
        self.sigma = sigma

    def forward(self, x, radial_map):
        # x: (B,C,H_feat,W_feat)
        # radial_map: (B,1,H_orig,W_orig)  归一化距离 [0,1]
        b, c, fh, fw = x.shape

        # 1) 通道注意力
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        # 2) resize 径向图
        rm = F.interpolate(radial_map, size=(fh, fw),
                           mode='bilinear', align_corners=False)  # (B,1,fh,fw)

        # 3) Gaussian 环带加权: peak 在 rm=center
        radial_weight = torch.exp(- ((rm - self.center) ** 2) / (2 * self.sigma ** 2))  # (B,1,fh,fw)

        # 4) 融合
        return x * y * radial_weight



# ─── 2. ECAAttention（Efficient Channel Attention，不变）───────────────────────────────
class ECAAttention(nn.Module):
    """
    Efficient Channel Attention (ECA)
    仅做全局平均池化 + 1D 卷积 + Sigmoid → 对每个通道加权
    """

    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        b, c, _, _ = x.size()
        # (B,1,C)
        y = self.avg_pool(x).view(b, 1, c)
        # (B,1,C)
        y = self.conv(y)
        # (B,C,1,1)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ─── 3. CoordinateAttention（可选保留 / 也可以移除）────────────────────────────────────
class CoordinateAttention(nn.Module):
    """
    坐标注意力（Capture long-range dependencies in one spatial direction）
    先做 H×1 与 1×W 的 Pooling，然后一层共享 conv1→ 再分支 conv_h + conv_w → sigmoid → 空间加权
    """

    def __init__(self, inp_channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp_channels // reduction)
        self.conv1 = nn.Conv2d(inp_channels, mip, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, inp_channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mip, inp_channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()
        x_h = self.pool_h(x)  # (b,c,h,1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (b,c,w,1)
        x_cat = torch.cat([x_h, x_w], dim=2)  # (b,c,h+w,1)
        x_cat = self.conv1(x_cat)  # (b,mip,h+w,1)
        x_cat = self.bn1(x_cat)
        x_cat = self.act(x_cat)
        x_h_split, x_w_split = torch.split(x_cat, [h, w], dim=2)
        x_w_split = x_w_split.permute(0, 1, 3, 2)  # (b,mip,1,w)
        a_h = torch.sigmoid(self.conv_h(x_h_split))  # (b,c,h,1)
        a_w = torch.sigmoid(self.conv_w(x_w_split))  # (b,c,1,w)
        out = x * a_h * a_w
        return out


# ─── 4. 本地加载 safetensors 权重（不变）────────────────────────────────────────────
def load_model_weights_from_local(model, path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"预训练权重文件未找到: {path}")
    state_dict = load_file(path)
    model_state_dict = model.state_dict()

    # 只加载键名 & 形状都匹配的参数
    filtered = {
        k: v
        for k, v in state_dict.items()
        if (k in model_state_dict and v.shape == model_state_dict[k].shape)
    }
    skipped = set(state_dict.keys()) - set(filtered.keys())
    for k in skipped:
        print(f"⚠️ 跳过不匹配权重: {k}")

    model.load_state_dict(filtered, strict=False)
    print(f">>> ✅ 成功加载本地权重: {path}")
    return model


# ─── 5. 自定义 ResNet / ConvNeXt / BaseFactory ─────────────────────────────────────────
class CustomResNet(nn.Module):
    """
    支持 ResNet18 / ResNet50 + RingAttention + ECAAttention
    """

    def __init__(self, model_name='resnet18', in_channels=4, num_classes=2,
                 pretrained=True, local_weights_dir='D:/Paper_/resnet/pretrained'):
        super().__init__()
        # 1) backbone：4 通道输入
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            in_chans=in_channels,
        )
        self.num_features = self.backbone.num_features
        # 去掉原本的分类 head
        self.backbone.reset_classifier(0)

        # 2) 加载 safetensors 权重
        if pretrained:
            weight_map = {
                'resnet18': 'resnet18.a1_in1k.safetensors',
                'resnet50': 'resnet50.a1_in1k.safetensors',
            }
            if model_name in weight_map:
                wpath = os.path.join(local_weights_dir, weight_map[model_name])
                self.backbone = load_model_weights_from_local(self.backbone, wpath)
            else:
                print(f"⚠️ 没有找到 {model_name} 的预训练权重，使用随机初始化")

        # 3) 注意力模块：Ring + ECA
        self.ring_attn = RingAttention(self.num_features,
                               ring_center=0.375,
                               sigma=0.065)
        self.ring_enhancer = RingEnhancer(self.num_features)  # ← 新增环形增强
        self.eca_attn = ECAAttention(self.num_features, k_size=3)

        # （可选）如果还想叠加 Coordinate Attention，就加下面这一行
        # self.coord_attn = CoordinateAttention(self.num_features)

        # 4) 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(self.num_features, num_classes)
        )

    def forward(self, x):
        raw_feat = self.backbone.forward_features(x)
        radial = x[:, 3:, :, :]
        if raw_feat.dim() != 4:
            B, C = raw_feat.shape
            Hf = Wf = int(math.sqrt(C))
            raw_feat = raw_feat.view(B, C, Hf, Wf)

        radial = F.interpolate(radial, size=raw_feat.shape[-2:], mode='bilinear', align_corners=False)
        enhanced = self.ring_enhancer(raw_feat, radial)
        attn = self.ring_attn(enhanced, radial)
        attn = self.eca_attn(attn)
        return self.classifier(attn)

class MILNet(nn.Module):
    """
    Patch‐level 特征 → Attention pooling → Image‐level 分类
    在 forward 中给每个 Patch 实时拼接径向第四通道。
    """
    def __init__(self, backbone_name='convnext_tiny', in_ch=1, num_classes=2, M=16, pretrained=True, local_weights_dir=None):
        super().__init__()
        # 1) backbone 去掉 head
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            in_chans=in_ch+1,   # 注意这里 in_ch+1
            num_classes=0,
            global_pool=''
        )
        if pretrained:
            weight_map = {
                'convnext_small': 'convnext_small.fb_in1k.safetensors',
                'convnext_tiny': 'convnext_tiny.fb_in1k.safetensors',
            }
            if backbone_name in weight_map:
                wpath = os.path.join(local_weights_dir, weight_map[backbone_name])
                self.backbone = load_model_weights_from_local(self.backbone, wpath)
            else:
                print(f"⚠️ 没有找到 {backbone_name} 的预训练权重，使用随机初始化")

        self.feat_dim = self.backbone.num_features
        self.M = M
        self.patch_clf = nn.Linear(self.feat_dim, 1)
        self.img_clf   = nn.Linear(self.feat_dim, num_classes)

    def _make_radial(self, B, H, W, device):
        """
        在 GPU 上即时生成 (B,1,H,W) 的 [0,1] 径向图
        """
        ys = torch.linspace(0, H-1, H, device=device).view(H,1).expand(H,W)
        xs = torch.linspace(0, W-1, W, device=device).view(1,W).expand(H,W)
        cx, cy = (W-1)/2, (H-1)/2
        dist = torch.sqrt((xs-cx)**2 + (ys-cy)**2)
        radial = (dist / (min(H,W)/2)).clamp(0,1)
        return radial.unsqueeze(0).unsqueeze(0).repeat(B,1,1,1)  # (B,1,H,W)

    def forward(self, x):
        """
        x: (B, M, 1, H, W)  # 来自 DataLoader
        """
        B, M, C, H, W = x.shape
        # 展平成 (B*M, 1, H, W)
        patches = x.view(B*M, 1, H, W)

        # GPU 上生成径向图
        radial = self._make_radial(B*M, H, W, patches.device)  # (B*M,1,H,W)

        # 拼接成 4D Patch： (B*M, 2, H, W)
        inp = torch.cat([patches, radial], dim=1)

        # backbone 提取 (B*M, D, h', w')
        f = self.backbone(inp)
        # 全局池化到 (B, M, D)
        f = F.adaptive_avg_pool2d(f, 1).view(B, M, -1)

        # Patch‐level 得分 & Attention
        ps = self.patch_clf(f).squeeze(-1)   # (B, M)
        α  = F.softmax(ps, dim=1)            # (B, M)

        # attention pooling → (B, D)
        img_feat = torch.einsum('bm,bmd->bd', α, f)
        logits   = self.img_clf(img_feat)    # (B, num_classes)
        return logits, ps

class CustomConvNeXt(nn.Module):
    """
    新增：ConvNeXt-Small + RingAttention + ECAAttention
    （ConvNeXt 默认输入 3 通道，这里把 in_chans=4 传进去）
    """

    def __init__(self, model_name='convnext_small', in_channels=4, num_classes=2,
                 pretrained=True, local_weights_dir='D:/Paper_/resnet/pretrained'):
        super().__init__()
        # 1) backbone：ConvNeXt-Small，in_chans=4
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            in_chans=in_channels
        )
        self.num_features = self.backbone.num_features
        # 去掉原分类 head
        self.backbone.reset_classifier(0)

        # 2) 如果需要加载 safetensors 权重（注意：ConvNeXt 的 safetensors 文件名自行配）
        if pretrained:
            # 你需要本地放一个 convnext_small/large 对应的 safetensors 文件。如果没有可以不加载：
            weight_map = {
                'convnext_small': 'convnext_small.fb_in1k.safetensors',
                'convnext_tiny': 'convnext_tiny.fb_in1k.safetensors',
                # 'convnext_base': 'convnext_base_1k_224.pth'   # 举例
            }
            if model_name in weight_map:
                wpath = os.path.join(local_weights_dir, weight_map[model_name])
                self.backbone = load_model_weights_from_local(self.backbone, wpath)
            else:
                print(f"⚠️ 没有找到 {model_name} 的预训练权重，使用随机初始化")

        # 3) 是否保留 Ring + ECA 注意力
        self.ring_attn = RingAttention(self.num_features,
                               ring_center=0.45,
                               sigma=0.055)
        self.ring_enhancer = RingEnhancer(self.num_features)  # ← 新增环形增强
        self.eca_attn = ECAAttention(self.num_features, k_size=3)
        self.spatial_attn = SpatialAttention(kernel_size=7)
        # self.coord_attn = CoordinateAttention(self.num_features)  # 如需叠加

        # 4) 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(self.num_features, num_classes)
        )

    def forward(self, x):
        raw_feat = self.backbone.forward_features(x)
        radial = x[:, 3:, :, :]
        if raw_feat.dim() != 4:
            B, C = raw_feat.shape
            Hf = Wf = int(math.sqrt(C))
            raw_feat = raw_feat.view(B, C, Hf, Wf)

        radial = F.interpolate(radial, size=raw_feat.shape[-2:], mode='bilinear', align_corners=False)
        enhanced = self.ring_enhancer(raw_feat, radial)
        attn = self.ring_attn(enhanced, radial)
        attn = self.eca_attn(attn)
        attn = self.spatial_attn(attn)
        return self.classifier(attn)


# ─── 6. Factory 函数 create_model ─────────────────────────────────────────────────────
def create_model(model_name='resnet18', in_channels=4, num_classes=2,
                 pretrained=True, local_weights_dir='D:/Paper_/resnet/pretrained'):
    """
    支持四种备选 backbone：
      - 'resnet18'
      - 'resnet50'
      - 'convnext_small'   (ConvNeXt-Small)

    """
    model_name = model_name.lower()
    if model_name in ['resnet18', 'resnet50']:
        return CustomResNet(
            model_name=model_name,
            in_channels=in_channels,
            num_classes=num_classes,
            pretrained=pretrained,
            local_weights_dir=local_weights_dir
        )
    elif model_name.startswith('convnext'):
        # 例如 model_name='convnext_small' 或 'convnext_base'
        return CustomConvNeXt(
            model_name=model_name,
            in_channels=in_channels,
            num_classes=num_classes,
            pretrained=pretrained,
            local_weights_dir=local_weights_dir
        )
    else:
        raise ValueError(f"不支持的模型: {model_name}")
