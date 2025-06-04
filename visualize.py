# visualize.py

import os
import math
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import warnings
import itertools
import pandas as pd
from PIL import Image
from torchvision import transforms
import torch.nn as nn

from sklearn.metrics import confusion_matrix
from models import ResNet18, ResNet50  # 导入你项目中的模型定义

# 允许重复加载 OpenMP 运行时，避免 OMP: Error #15
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore", category=FutureWarning)

# —— 过滤警告 ——#
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="torchvision\\.models\\._utils"
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="You are using torch.load with weights_only=False"
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="kornia\\.feature\\.lightglue"
)

# ---------------- 手动设置参数 ----------------
IMAGE_PATH      = "11break.png"            # 要可视化的泪膜图像文件
CHECKPOINT_PATH = "output/checkpoints/run_best.pth"
MODEL_NAME      = "resnet50"               # "resnet18" 或 "resnet50"
IN_CHANNELS     = 4                        # 修改为 4 通道
NUM_CLASSES     = 2                        # 二分类：normal vs break

# Patch 相关（自动按输入图像尺寸设置圆心、大小）
img_bgr = cv2.imread(IMAGE_PATH)
H, W = img_bgr.shape[:2]
CENTER     = (W // 2, H // 2)              # 圆环中心 (x0, y0)
RADIUS     = 1                             # 半径取 1 刚好裁到中心
PATCH_SIZE = int(H * 0.75)                 # Patch 大小 = 高度的 75%

# 测试结果 CSV 文件路径（由 train.py 在测试时保存）
TEST_CSV_PATH = "output/test_results.csv"
# --------------------------------------------


def load_model(model_name, checkpoint_path, in_channels=4, num_classes=2):
    """加载训练好的泪膜分类模型，只加载形状匹配的参数。"""
    if model_name.lower() == "resnet50":
        model = ResNet50(in_channels=in_channels, num_classes=num_classes)
    else:
        model = ResNet18(in_channels=in_channels, num_classes=num_classes)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get('net', checkpoint)
    model_dict = model.state_dict()
    # 只加载键名 & 大小都匹配的参数
    filtered = {
        k: v for k, v in state_dict.items()
        if k in model_dict and v.size() == model_dict[k].size()
    }
    model_dict.update(filtered)
    model.load_state_dict(model_dict)
    model.eval()
    print(f">>> Loaded weights from: {checkpoint_path} "
          f"(matched {len(filtered)}/{len(model_dict)} params)")
    return model


def extract_patch(image_np, center, radius, patch_size, angle=0.0):
    """
    从灰度 numpy 图像里，按给定角度抽取一个靠近“圆环”边缘的 Patch，
    返回 PIL.Image 格式的 Patch 以及坐标 (x1, y1, x2, y2)。
    """
    H_img, W_img = image_np.shape[:2]
    x0, y0 = center
    hs = patch_size // 2

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
    if x2 > W_img:
        x2 = W_img
        x1 = W_img - patch_size
    if y2 > H_img:
        y2 = H_img
        y1 = H_img - patch_size

    patch_np = image_np[y1:y2, x1:x2]
    patch_pil = Image.fromarray(patch_np)
    return patch_pil, (x1, y1, x2, y2)


def generate_gradcam(model, input_tensor, target_class=None):
    """
    使用 Grad-CAM 生成热力图：
    - 在 ResNet 的 layer4 最后一层卷积上注册前向 & 反向钩子，分别采集 fmap 和 gradients。
    - 前向得到 output 后，对 target_class 进行 backward，提取 fmap 和 grads → 计算通道权重 → 生成 CAM。
    返回：
      cam_map:      归一化到 [0,1] 的热力图 np.ndarray，
      logits:       原始模型 output logits (numpy)。
    """
    feature_maps = []
    gradients    = []

    # 选择目标卷积层
    if isinstance(model, ResNet50):
        block4 = model.model.layer4[-1]   # 最后一个 Bottleneck
        target_conv = block4.conv3        # 最后一个卷积
    elif isinstance(model, ResNet18):
        block4 = model.model.layer4[-1]   # 最后一个 BasicBlock
        target_conv = block4.conv2        # 最后一个卷积
    else:
        raise ValueError("只支持 ResNet18/ResNet50")

    # 前向钩子：保存 feature_maps
    def forward_hook(module, inp, outp):
        feature_maps.append(outp.detach().cpu())

    # 反向钩子：保存 gradients
    def backward_hook(module, grad_in, grad_out):
        # grad_out[0] 是 shape [1, C, h, w] 的梯度
        gradients.append(grad_out[0].detach().cpu())

    # 注册钩子
    fh = target_conv.register_forward_hook(forward_hook)
    bh = target_conv.register_backward_hook(backward_hook)
    print(f"✅ Hooked forward/backward on layer: {target_conv}")

    # 前向
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)  # [1, num_classes]

    # 选择 target_class
    logits = output.cpu()           # [1, num_classes]
    probs  = F.softmax(logits, dim=1)
    if target_class is None:
        target_class = torch.argmax(probs, dim=1).item()

    # 再次前向 & backward：要让 gradients 生效，必须是有梯度的前向
    # 先取消 no_grad，然后再做 backward
    fh.remove()
    bh.remove()
    feature_maps.clear()
    gradients.clear()

    # 重新注册钩子（这次不包裹 no_grad）
    fh = target_conv.register_forward_hook(forward_hook)
    bh = target_conv.register_backward_hook(backward_hook)

    model.zero_grad()
    output = model(input_tensor)    # 前向
    score  = output[0, target_class]
    score.backward()                # 反向，gradients 会被保存

    fh.remove()
    bh.remove()

    # 取出保存的 feature_maps[0] 和 gradients[0]
    fmap = feature_maps[0].squeeze(0)      # [C, h, w]
    grad = gradients[0].squeeze(0)         # [C, h, w]
    # 全局平均池化梯度：得到每个通道的权重
    weights = torch.mean(grad, dim=(1, 2)) # [C]

    # 计算 Grad-CAM：∑_c (weight_c * fmap_c)
    cam = torch.zeros(fmap.shape[1:], dtype=torch.float32)  # [h, w]
    for i, w in enumerate(weights):
        cam += w * fmap[i]

    # ReLU + 归一化到 [0,1]
    cam = F.relu(cam)
    cam -= cam.min()
    cam /= (cam.max() + 1e-8)

    return cam.numpy(), logits.detach().cpu().numpy()


def accumulate_contour_heatmap(frame_bgr, valid_motion_mask, decay=0.95, colormap=cv2.COLORMAP_JET):
    """
    根据二值掩码 valid_motion_mask，在 frame_bgr（BGR uint8）上生成
    “累积式轮廓热力图”并叠加：
      1. heatmap *= decay
      2. findContours → 根据 area 计算 value=area/20000 → drawContours
      3. GaussianBlur → 应用掩码 → normalize → applyColorMap → addWeighted
    """
    H_mask, W_mask = valid_motion_mask.shape[:2]
    heatmap = np.zeros((H_mask, W_mask), dtype=np.float32)

    # 1) 衰减
    heatmap *= decay

    # 2) 找轮廓并累积热力值
    _, thresh = cv2.threshold(valid_motion_mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        value = area / 20000.0
        cv2.drawContours(heatmap, [c], -1, float(value), thickness=cv2.FILLED)

    # 3) 高斯模糊
    heatmap_smooth = cv2.GaussianBlur(heatmap, (21, 21), 0)

    # 4) 应用掩码
    heatmap_smooth = cv2.bitwise_and(heatmap_smooth, heatmap_smooth, mask=valid_motion_mask)

    # 5) 归一化到 0-255
    cv2.normalize(heatmap_smooth, heatmap_smooth, 0, 255, cv2.NORM_MINMAX)
    heatmap_u8 = heatmap_smooth.astype(np.uint8)

    # 6) 伪彩色映射
    heatmap_colored = cv2.applyColorMap(heatmap_u8, colormap)

    # 7) 叠加
    alpha = 0.4
    overlay = cv2.addWeighted(frame_bgr, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay


def visualize_tearfim(image_path, model, center, radius, patch_size, device):
    """
    可视化泪膜裂纹分类：原图 + Patch 区域 + Grad-CAM 热力图 + 二分类概率柱状图
    """
    # 读取灰度图
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 抽取一个 Patch（angle=0）
    angle = 0.0
    patch_pil, (x1, y1, x2, y2) = extract_patch(img_gray, center, radius, patch_size, angle)
    patch_np = np.array(patch_pil)  # (patch_size, patch_size) 灰度

    # Patch→3通道 Tensor
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((patch_size, patch_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    patch_t_3 = transform(patch_pil)  # (3, patch_size, patch_size)

    # —— 第4通道：径向距离先验（高斯） ——
    x0, y0 = center
    xi = int(x0 + radius * math.cos(angle))
    yi = int(y0 + radius * math.sin(angle))
    cx = xi - x1
    cy = yi - y1
    Hs = patch_size

    yy = np.arange(0, Hs)
    xx = np.arange(0, Hs)
    grid_y, grid_x = np.meshgrid(yy, xx, indexing='ij')
    dist = np.sqrt((grid_x - cx) ** 2 + (grid_y - cy) ** 2)
    sigma = radius * 0.1
    radial = np.exp(-0.5 * ((dist - radius) / sigma) ** 2)
    radial = (radial - radial.min()) / (radial.max() - radial.min() + 1e-8)
    radial = np.clip(radial, 0, 1)
    radial_tensor = torch.from_numpy(radial).unsqueeze(0).float()  # (1, patch_size, patch_size)

    # 拼接 4 通道
    patch_all   = torch.cat([patch_t_3, radial_tensor], dim=0)  # (4, patch_size, patch_size)
    patch_tensor = patch_all.unsqueeze(0).to(device)            # (1,4, patch_size, patch_size)

    # 生成 Grad-CAM
    cam_map, logits = generate_gradcam(model, patch_tensor)

    # 计算概率
    probs = F.softmax(torch.tensor(logits), dim=1).numpy().flatten()
    class_names = ['Normal', 'Break']

    # 可视化布局
    fig = plt.figure(figsize=(16, 6))
    fig.suptitle("Tear Film Rupture Detection Visualization (Grad-CAM)", fontsize=18)

    # 1. 原图 + Patch 矩形
    ax1 = fig.add_subplot(1, 4, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rect_img = img_rgb.copy()
    cv2.rectangle(rect_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    ax1.imshow(rect_img)
    ax1.set_title("Original Image + Patch")
    ax1.axis('off')

    # 2. Patch 灰度
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.imshow(patch_np, cmap='gray')
    ax2.set_title(f"{patch_size}×{patch_size} Patch")
    ax2.axis('off')

    # 3. Grad-CAM 热力图叠加
    ax3 = fig.add_subplot(1, 4, 3)
    cam_resized = cv2.resize(cam_map, (patch_np.shape[1], patch_np.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    patch_color = cv2.cvtColor(patch_np, cv2.COLOR_GRAY2BGR)
    overlay_cam = cv2.addWeighted(patch_color, 0.7, heatmap, 0.3, 0)
    ax3.imshow(cv2.cvtColor(overlay_cam, cv2.COLOR_BGR2RGB))
    ax3.set_title("Grad-CAM on Patch")
    ax3.axis('off')

    # 4. 二分类概率柱状图
    ax4 = fig.add_subplot(1, 4, 4)
    bars = ax4.barh(class_names, probs, color='tomato')
    ax4.set_xlim(0, 1)
    ax4.set_title("Classification Probabilities")
    ax4.invert_yaxis()
    for i, bar in enumerate(bars):
        ax4.text(bar.get_width() + 0.02,
                 bar.get_y() + bar.get_height() / 2,
                 f'{probs[i] * 100:.1f}%',
                 va='center',
                 fontweight='bold')
    ax4.set_xlabel("Probability")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def generate_contour_heatmap(image_path, center, radius, patch_size):
    """
    单独生成“轮廓累积式热力图”叠加在 Patch 上：
    - 裁同样的 patch
    - 用简单阈值提取裂纹掩码 valid_motion_mask
    - 调用 accumulate_contour_heatmap 生成叠加结果
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    patch_pil, (x1, y1, x2, y2) = extract_patch(img_gray, center, radius, patch_size, angle=0.0)
    patch_np    = np.array(patch_pil)                     # 灰度
    patch_color = cv2.cvtColor(patch_np, cv2.COLOR_GRAY2BGR)  # BGR uint8

    # 用阈值提取“暗线”作为裂纹掩码
    _, valid_motion_mask = cv2.threshold(patch_np, 50, 255, cv2.THRESH_BINARY_INV)

    overlay_contour = accumulate_contour_heatmap(patch_color, valid_motion_mask,
                                                 decay=0.95, colormap=cv2.COLORMAP_JET)
    # 单独展示
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(overlay_contour, cv2.COLOR_BGR2RGB))
    plt.title("Contour-based Heatmap on Patch")
    plt.axis('off')
    plt.show()


def plot_confusion_matrix_counts(cm, classes,
                                 title="Confusion Matrix (Counts)",
                                 cmap=plt.cm.Blues):
    """
    绘制绝对个数混淆矩阵。
    """
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], "d"),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()


def visualize_confusion_matrix(test_csv_path, output_root):
    """
    从 test_results.csv 读取真值与预测概率，生成混淆矩阵热力图（绝对个数）。
    """
    if not os.path.exists(test_csv_path):
        print(f"Cannot find {test_csv_path}, skipping confusion matrix visualization.")
        return

    df = pd.read_csv(test_csv_path)
    if "true_0" in df.columns and "score_1" in df.columns:
        y_true     = df["true_0"].values
        y_score_pos = df["score_1"].values
    else:
        y_true     = df.iloc[:, 1].values
        y_score_pos = df.iloc[:, -1].values

    y_pred = (y_score_pos > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    classes = ["Normal", "Break"]

    plt.figure(figsize=(6, 5))
    plot_confusion_matrix_counts(cm, classes, title="Confusion Matrix (Counts)")
    os.makedirs(output_root, exist_ok=True)
    plt.savefig(os.path.join(output_root, "confusion_matrix_counts.png"))
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f">>> Loading and analyzing image: {IMAGE_PATH}")

    # 1) 可视化 CAM & 概率
    model = load_model(MODEL_NAME, CHECKPOINT_PATH, IN_CHANNELS, NUM_CLASSES).to(device)
    visualize_tearfim(IMAGE_PATH, model, CENTER, RADIUS, PATCH_SIZE, device)

    # 2) 单独可视化轮廓累积式热力图
    print(">>> Now generating contour‐based heatmap…")
    generate_contour_heatmap(IMAGE_PATH, CENTER, RADIUS, PATCH_SIZE)

    # # 3) 可视化测试集混淆矩阵（绝对个数）
    # print(">>> Visualizing confusion matrix from test results…")
    # visualize_confusion_matrix(TEST_CSV_PATH, os.path.dirname(TEST_CSV_PATH))
