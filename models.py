# -*- coding: utf-8 -*-
# @Author :Xiaoju
# Time    : 2025/4/9

import torch
import torch.nn as nn
import torchvision.models as models


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


class CBAMBlock(nn.Module):
    def __init__(self, channels, ratio=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.channel_att = ChannelAttention(channels, ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.channel_att(x)
        out = out * self.spatial_att(out)
        return out


def attach_cbam(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Sequential):
            for block in child:
                if hasattr(block, 'conv2'):  # BasicBlock 或 Bottleneck
                    channels = block.conv2.out_channels
                    block.add_module('cbam', CBAMBlock(channels))
        else:
            attach_cbam(child)


class RingAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, radial_map):
        """
        x: 输入特征图 (B, C, H, W)
        radial_map: 径向距离图 (B, 1, H, W)
        """
        # 通道注意力
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        # 径向权重：离目标半径越近，权重越高
        radial_weight = torch.exp(-5 * torch.abs(radial_map - 0.5))  # 假设radial_map已归一化

        # 结合通道注意力和径向权重
        return x * y * radial_weight


# 修改models.py中的ResNet50和ResNet18类
class ResNet18(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=True)
        if in_channels != 3:
            self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 改为将fc定义为模型的属性
        self.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model.fc = nn.Identity()  # 禁用原model的fc层
        attach_cbam(self.model)
        self.ring_attn = RingAttention(self.model.layer4[-1].conv2.out_channels)

    def forward(self, x):
        # 直接使用4通道输入
        x = self.model(x)
        # 提取径向距离通道（第4通道）
        if x.dim() == 2:
            # 如果x是2维的，说明已经被展平，这里需要恢复维度
            # 假设最后一层卷积输出的通道数为最后一个卷积层的输出通道数
            channels = self.model.layer4[-1].conv2.out_channels
            # 计算高度和宽度
            height = width = int((x.size(1) // channels) ** 0.5)
            x = x.view(x.size(0), channels, height, width)

        radial_channel = x[:, 3:4, :, :]  # (B, 1, H, W)

        x = self.ring_attn(x, radial_channel)
        x = torch.flatten(x, 1)
        x = self.fc(x)  # 显式调用自定义的fc层
        return x


class ResNet50(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(pretrained=True)
        if in_channels != 3:
            self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 保存原始fc层的输入维度
        fc_in_features = self.model.fc.in_features

        # 禁用原model的fc层
        self.model.fc = nn.Identity()

        # 定义新的fc层作为模型的属性
        self.fc = nn.Linear(fc_in_features, num_classes)

        # 插入CBAM
        attach_cbam(self.model)
        self.ring_attn = RingAttention(self.model.layer4[-1].conv3.out_channels)

    def forward(self, x):
        # 直接使用4通道输入
        x = self.model(x)
        # 提取径向距离通道（第4通道）
        if x.dim() == 2:
            # 如果x是2维的，说明已经被展平，这里需要恢复维度
            # 假设最后一层卷积输出的通道数为最后一个卷积层的输出通道数
            channels = self.model.layer4[-1].conv3.out_channels
            # 计算高度和宽度
            height = width = int((x.size(1) // channels) ** 0.5)
            x = x.view(x.size(0), channels, height, width)

        radial_channel = x[:, 3:4, :, :]  # (B, 1, H, W)

        x = self.ring_attn(x, radial_channel)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x