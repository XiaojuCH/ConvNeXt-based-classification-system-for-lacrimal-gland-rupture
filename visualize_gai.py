# visualize.py
import os, math
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from models import create_model
from skimage import measure
from skimage.filters import frangi  # 添加血管增强滤波器
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

# 环带参数 - 调整为更关注环带区域
RING_CENTER = 0.375
RING_WIDTH  = 0.14
ANGLES = list(range(0, 360, 12))  # 更密集的角度采样
EDGE_LOW = 30  # 降低Canny阈值，检测更多边缘
EDGE_HIGH = 100  # 降低Canny阈值
NEIGHBORHOOD = 9  # 增加邻域采样范围


def load_model(checkpoint_path, model_name="convnext_tiny"):
    model = create_model(
        model_name=model_name,
        in_channels=4,
        num_classes=2,
        pretrained=False,
        local_weights_dir="output/checkpoints"
    )
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state = ckpt.get('net', ckpt)
    filtered = {k: v for k, v in state.items() if k in model.state_dict()}
    model.load_state_dict(filtered, strict=False)
    model.eval()
    return model


def full_image_cam(model, img4):
    # 1) 找到最后一层 conv
    backbone = model.backbone
    if hasattr(backbone, 'layer4'):  # ResNet
        target_layer = backbone.layer4[-1].conv3
    else:  # ConvNeXt
        stage = backbone.stages[-1]
        blk = stage.blocks[-1] if hasattr(stage, 'blocks') else stage[-1]
        if hasattr(blk, 'dwconv'):
            target_layer = blk.dwconv
        elif hasattr(blk, 'depthwise_conv'):
            target_layer = blk.depthwise_conv
        else:
            # 回退：遍历子模块，找第一个 groups>1 的 Conv2d
            target_layer = None
            for m in blk.modules():
                if isinstance(m, nn.Conv2d) and m.groups > 1:
                    target_layer = m
                    break
            if target_layer is None:
                raise RuntimeError("无法在 ConvNeXt block 中找到深度卷积层")

    # 2) 注册 hook
    fmap = []
    handle = target_layer.register_forward_hook(lambda m, i, o: fmap.append(o.detach()))

    # 3) 前向
    with torch.no_grad():
        logits = model(img4)
    handle.remove()

    # 4) 计算 probs
    probs = F.softmax(logits, dim=1)[0].cpu().numpy()  # [p_normal, p_break]

    # 5) 取 fmap → CAM
    feat = fmap[0][0].cpu()  # [C, Hf, Wf]
    fc_w = None
    for m in model.classifier.modules():
        if isinstance(m, nn.Linear):
            fc_w = m.weight.data.cpu()
    if fc_w is None:
        raise RuntimeError("在 classifier 找不到 Linear 层")

    # 增强环带区域的响应 - 更激进的调整
    ring_w = fc_w[1].clone()
    # 大幅降低中心区域的权重
    ring_w[:int(len(ring_w) * 0.4)] *= 0.2
    # 小幅降低边缘区域的权重
    ring_w[int(len(ring_w) * 0.8):] *= 0.6

    cam = (feat * ring_w[:, None, None]).sum(0).relu()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = F.interpolate(cam[None, None], size=(512, 512),
                        mode='bilinear', align_corners=False)[0, 0].numpy()

    return cam, probs


def visualize_full_cam(image_path, checkpoint_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(checkpoint_path).to(device)

    # ----- 1) 读原图 & 保存尺寸 -----
    orig_bgr = cv2.imread(image_path)
    if orig_bgr is None:
        print(f"Error: Cannot read image at {image_path}")
        return

    gray_full = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)
    H0, W0 = gray_full.shape

    # ----- 2) 环带检测优化 -----
    # 使用Frangi滤波器增强环带结构
    frangi_enhanced = frangi(gray_full, sigmas=range(1, 4, 1), black_ridges=False)
    frangi_enhanced = (frangi_enhanced * 255).astype(np.uint8)

    # 改进的边缘检测
    blurred = cv2.GaussianBlur(frangi_enhanced, (7, 7), 0)
    edges = cv2.Canny(blurred, EDGE_LOW, EDGE_HIGH)

    # 形态学操作增强环带连续性
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=1)

    # 生成径向蒙版
    xx = np.linspace(0, W0 - 1, W0)[None, :]
    yy = np.linspace(0, H0 - 1, H0)[:, None]
    dist0 = np.sqrt((xx - W0 / 2) ** 2 + (yy - H0 / 2) ** 2) / (min(H0, W0) / 2)
    ring_mask = ((dist0 > RING_CENTER - RING_WIDTH / 2) &
                 (dist0 < RING_CENTER + RING_WIDTH / 2)).astype(float)

    # 只保留环带上的边缘
    ring_edges = (edges * ring_mask).astype(np.uint8)

    # 找到最大的轮廓作为主环带
    contours, _ = cv2.findContours(ring_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        main_contour = max(contours, key=cv2.contourArea)

        # 计算主环带的中心点
        M = cv2.moments(main_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            # 更新中心点为环带中心，而不是图像中心
            center = (cx, cy)
        else:
            center = (W0 // 2, H0 // 2)
    else:
        center = (W0 // 2, H0 // 2)

    # ----- 3) Resize->512 for CAM -----
    pil = Image.fromarray(gray_full).resize((512, 512), Image.BICUBIC)
    tf3 = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    t3 = tf3(pil)
    ys = torch.linspace(0, 511, 512).view(512, 1).expand(512, 512)
    xs = torch.linspace(0, 511, 512).view(1, 512).expand(512, 512)
    cx, cy = center[0] * 512 / W0, center[1] * 512 / H0  # 使用检测到的中心点
    dist = torch.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    radial = (dist / (255)).clamp(0, 1).unsqueeze(0)
    img4 = torch.cat([t3, radial], dim=0)[None].to(device)

    # ----- 4) 计算 CAM & probs -----
    cam_small, probs = full_image_cam(model, img4)
    p_normal, p_break = probs

    # ----- 5) 上采样 CAM 回原始尺寸 -----
    cam = cv2.resize(cam_small, (W0, H0), interpolation=cv2.INTER_LINEAR)

    # ----- 6) 改进的多像素角度采样 -----
    angle_vals = []
    R0 = RING_CENTER * (min(H0, W0) / 2)
    ring_cam = cam * ring_mask

    for a in ANGLES:
        th = math.radians(a)
        cx0 = int(center[0] + R0 * math.cos(th))  # 使用检测到的中心点
        cy0 = int(center[1] + R0 * math.sin(th))

        # 在邻域内取加权平均
        patch_vals = []
        weights = []
        for dx in range(-NEIGHBORHOOD, NEIGHBORHOOD + 1):
            for dy in range(-NEIGHBORHOOD, NEIGHBORHOOD + 1):
                x = cx0 + dx
                y = cy0 + dy
                if 0 <= x < W0 and 0 <= y < H0:
                    weight = 1.0 - (dx ** 2 + dy ** 2) / (NEIGHBORHOOD ** 2 * 2)
                    patch_vals.append(ring_cam[y, x])
                    weights.append(weight)

        if weights:
            weighted_avg = np.average(patch_vals, weights=weights)
            angle_vals.append(weighted_avg)
        else:
            angle_vals.append(0)

    # ----- 7) 优化可视化 -----
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f"Tear Film Analysis - Break Probability: {p_break * 100:.1f}%",
                 fontsize=18, fontweight='bold')

    # a) 原图 + CAM + 环带轮廓
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
    ax1.imshow(cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB))

    # 增强环带区域的CAM显示
    ring_cam_vis = np.zeros_like(cam)
    ring_cam_vis = np.where(ring_mask > 0, cam, 0)
    ax1.imshow(ring_cam_vis, cmap='jet', alpha=0.7)

    # 绘制检测到的主环带轮廓
    if contours:
        for contour in contours:
            ax1.plot(contour[:, 0, 0], contour[:, 0, 1], 'c-', linewidth=2, alpha=0.8)

    # 绘制角度采样点
    for a in ANGLES:
        th = math.radians(a)
        cx0 = int(center[0] + R0 * math.cos(th))
        cy0 = int(center[1] + R0 * math.sin(th))
        ax1.plot(cx0, cy0, 'yo', markersize=6, alpha=0.8)

    ax1.set_title("Enhanced Ring Band CAM Visualization", fontsize=14)
    ax1.axis('off')

    # b) 环带区域CAM响应热力图
    ax2 = plt.subplot2grid((3, 3), (0, 2))
    ax2.imshow(ring_cam, cmap='hot', vmin=0, vmax=1)
    ax2.set_title("Ring Band CAM Heatmap", fontsize=14)
    ax2.axis('off')

    # c) 角度—CAM 曲线
    ax3 = plt.subplot2grid((3, 3), (1, 2))
    ax3.plot(ANGLES, angle_vals, 'g-o', linewidth=2.5, markersize=6)
    ax3.fill_between(ANGLES, angle_vals, 0, color='green', alpha=0.2)

    # 标记最高响应点
    max_idx = np.argmax(angle_vals)
    max_angle = ANGLES[max_idx]
    ax3.plot(max_angle, angle_vals[max_idx], 'ro', markersize=8)
    ax3.annotate(f'Max: {angle_vals[max_idx]:.2f}°',
                 (max_angle, angle_vals[max_idx]),
                 xytext=(max_angle + 10, angle_vals[max_idx] + 0.1),
                 arrowprops=dict(facecolor='red', shrink=0.05))

    ax3.set_ylim(0, 1)
    ax3.set_title("CAM Response Distribution Around Ring", fontsize=14)
    ax3.set_xlabel("Angle (°)", fontsize=12)
    ax3.set_ylabel("CAM Value", fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.5)

    # d) 分类概率
    ax4 = plt.subplot2grid((3, 3), (2, 0))
    colors = ['#4CAF50', '#F44336'] if p_break > 0.5 else ['#4CAF50', '#9E9E9E']
    ax4.barh(['Normal', 'Break'], [p_normal, p_break], color=colors, height=0.5)
    ax4.set_xlim(0, 1)
    ax4.set_title("Classification Probabilities", fontsize=14)
    ax4.invert_yaxis()
    ax4.tick_params(axis='y', labelsize=12)
    for i, v in enumerate([p_normal, p_break]):
        ax4.text(v + 0.02, i, f"{v * 100:.1f}%",
                 va='center', fontsize=14, fontweight='bold')

    # e) 环带边缘检测结果
    ax5 = plt.subplot2grid((3, 3), (2, 1))
    edge_vis = cv2.cvtColor(gray_full, cv2.COLOR_GRAY2RGB)
    edge_vis[edges > 0] = [0, 255, 255]  # 黄色边缘
    if contours:
        cv2.drawContours(edge_vis, [main_contour], -1, (0, 255, 0), 2)  # 主轮廓绿色
    ax5.imshow(edge_vis)
    ax5.set_title("Enhanced Ring Edge Detection", fontsize=14)
    ax5.axis('off')

    # f) 最高响应区域放大
    ax6 = plt.subplot2grid((3, 3), (2, 2))
    max_idx = np.argmax(angle_vals)
    max_angle = ANGLES[max_idx]
    th = math.radians(max_angle)
    cx0 = int(center[0] + R0 * math.cos(th))
    cy0 = int(center[1] + R0 * math.sin(th))

    # 提取并放大最高响应区域
    zoom_size = min(H0, W0) // 6  # 增大放大区域
    y1 = max(0, cy0 - zoom_size)
    y2 = min(H0, cy0 + zoom_size)
    x1 = max(0, cx0 - zoom_size)
    x2 = min(W0, cx0 + zoom_size)

    if y1 < y2 and x1 < x2:
        zoom_area = orig_bgr[y1:y2, x1:x2]
        zoom_cam = cam[y1:y2, x1:x2]

        # 叠加CAM热力图
        zoom_vis = cv2.cvtColor(zoom_area, cv2.COLOR_BGR2RGB)
        heatmap = cv2.applyColorMap((zoom_cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        zoom_vis = cv2.addWeighted(zoom_vis, 0.6, heatmap, 0.4, 0)

        # 标记中心点
        center_y, center_x = zoom_size, zoom_size
        cv2.circle(zoom_vis, (center_x, center_y), 5, (0, 0, 255), -1)

        ax6.imshow(zoom_vis)
        ax6.set_title(f"Max Response Area ({max_angle}°)", fontsize=14)
    else:
        ax6.text(0.5, 0.5, "Region out of bounds",
                 ha='center', va='center', fontsize=12, transform=ax6.transAxes)

    ax6.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("tear_film_analysis_enhanced.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    visualize_full_cam("222break.png", "output/checkpoints/run_best.pth")
    # visualize_full_cam("image/images/test/normal/100010 (44).png", "output/checkpoints/run_best.pth")