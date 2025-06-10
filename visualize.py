# visualize.py
import itertools
import os
import torch
import torch.nn.functional as F
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import warnings
import pandas as pd
from PIL import Image
from torchvision import transforms
import torch.nn as nn

from sklearn.metrics import confusion_matrix
from models import create_model  # CustomResNet with RingAttention + CoordAttention

# 避免 OpenMP 重复加载运行时错误
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore", category=FutureWarning)
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

# 环形区域参数
RING_INNER = 0.8  # 内环半径比例
RING_OUTER = 1.2  # 外环半径比例

# 分析角度设置
ANGLES = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]  # 12个角度

# ---------------- 手动设置参数 ----------------

IMAGE_PATH = "11break.png"                 # 要可视化的单张泪膜图像
CHECKPOINT_PATH = "output/checkpoints/run_best.pth"
MODEL_NAME = "convnext_tiny"                    # "resnet18" 或 "resnet50"
IN_CHANNELS = 4                            # 4 通道输入
NUM_CLASSES = 2                            # 二分类：normal vs break

# 自动根据 IMAGE_PATH 读取尺寸
img_bgr = cv2.imread(IMAGE_PATH)
if img_bgr is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")
H, W = img_bgr.shape[:2]

# 将圆心大致设为图像中心偏下，以覆盖 Placido 环区域
CENTER = (int(W * 0.5), int(H * 0.55))
RADIUS = 1
PATCH_SIZE = int(H * 0.7)

# 测试结果 CSV 文件路径（由 train.py 保存）
TEST_CSV_PATH = "output/test_run_best.csv"

# --------------------------------------------

def load_model(model_name, checkpoint_path, in_channels=4, num_classes=2):
    model = create_model(
        model_name=model_name,
        in_channels=in_channels,
        num_classes=num_classes,
        pretrained=False
    )
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('net', ckpt)
    filtered = {
        k: v for k, v in state_dict.items()
        if k in model.state_dict() and v.size() == model.state_dict()[k].size()
    }
    model.load_state_dict(filtered, strict=False)
    model.eval()
    print(f">>> Loaded {model_name} weights: {len(filtered)}/{len(model.state_dict())} params")
    return model

def extract_patch(image_np, center, radius, patch_size, angle=0.0):
    """
    从灰度 numpy 图像中，按给定角度抽取一个靠近圆环边缘的 Patch。
    返回 PIL.Image 格式的 Patch 以及对应坐标 (x1, y1, x2, y2)。
    """
    H, W = image_np.shape[:2]
    x0, y0 = center
    half = patch_size // 2

    xi = int(x0 + radius * math.cos(angle))
    yi = int(y0 + radius * math.sin(angle))

    x1 = xi - half
    y1 = yi - half
    x2 = xi + half
    y2 = yi + half

    # 边界检查
    if x1 < 0:
        x1, x2 = 0, patch_size
    if y1 < 0:
        y1, y2 = 0, patch_size
    if x2 > W:
        x2 = W
        x1 = W - patch_size
    if y2 > H:
        y2 = H
        y1 = H - patch_size

    patch_np = image_np[y1:y2, x1:x2]
    patch_pil = Image.fromarray(patch_np)
    return patch_pil, (x1, y1, x2, y2)

def generate_cam(model, input_tensor, target_class=None):
    """
    生成 Class Activation Map (CAM) 并返回热力图和预测 logits。
    同时支持 ResNet 系列和 ConvNeXt-small。
    """
    feature_maps = []

    # 找到 backbone
    backbone = model.backbone

    # ResNet 系列：最后一层 conv 在 backbone.layer4[-1].conv3
    # ConvNeXt-small：最后一层 conv 在 backbone.stages[-1].blocks[-1].dwconv（或类似名称）
    if hasattr(backbone, 'layer4'):  # ResNet
        target_layer = backbone.layer4[-1].conv3
    else:  # ConvNeXt
        stage = backbone.stages[-1]
        # ConvNeXtStage 不可下标，用 .blocks 取
        if hasattr(stage, 'blocks'):
            block = stage.blocks[-1]
        else:
            # 兼容旧版 timm
            block = stage[-1]
        # 深度卷积层通常是 groups=in_channels
        # 优先取 known 属性
        if hasattr(block, 'dwconv'):
            target_layer = block.dwconv
        elif hasattr(block, 'depthwise_conv'):
            target_layer = block.depthwise_conv
        else:
            # 回退：找到第一个 DepthwiseConv2d
            for m in block.modules():
                if isinstance(m, nn.Conv2d) and m.groups > 1:
                    target_layer = m
                    break
            else:
                raise RuntimeError("无法在 ConvNeXt block 中找到深度卷积层")

    # 注册 hook
    def forward_hook(module, input, output):
        feature_maps.append(output.detach().cpu())

    handle = target_layer.register_forward_hook(forward_hook)
    print(f"✅ Hooked to layer: {target_layer}")

    # 前向
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
    handle.remove()

    probs = F.softmax(logits.cpu(), dim=1)
    if target_class is None:
        target_class = torch.argmax(probs, dim=1).item()

    # 找到 classifier 最后一层 Linear 的权重
    fc_weight = None
    for m in model.classifier.modules():
        if isinstance(m, nn.Linear):
            fc_weight = m.weight.detach().cpu()
    if fc_weight is None:
        raise RuntimeError("找不到 classifier 中的 Linear 层")

    # 取出 feature_map
    fmap = feature_maps[0].squeeze(0)  # [C, H, W]
    C, Hf, Wf = fmap.shape
    assert C == fc_weight.shape[1], "通道数不匹配"

    # 计算 CAM
    cam = torch.sum(fmap * fc_weight[target_class].view(-1, 1, 1), dim=0)
    cam = F.relu(cam)
    cam -= cam.min()
    cam /= (cam.max() + 1e-8)

    # 新增：标记环形区域
    yy = np.arange(0, Hf)
    xx = np.arange(0, Wf)
    grid_y, grid_x = np.meshgrid(yy, xx, indexing='ij')
    center_x, center_y = Wf // 2, Hf // 2
    dist = np.sqrt((grid_x - center_x) ** 2 + (grid_y - center_y) ** 2)
    ring_mask = (dist > RADIUS * RING_INNER) & (dist < RADIUS * RING_OUTER)
    cam[ring_mask] *= 1.5  # 增强环形区域响应
    cam = np.clip(cam, 0, 1)

    return cam.numpy(), logits.cpu().numpy()
def visualize_tearfim(image_path, model, center, radius, patch_size, device):
    """
    可视化泪膜裂纹分类：原图 + Patch 区域 + CAM 热力图 + 二分类概率柱状图
    """
    # 读取灰度图
    img_bgr = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 抽取 Patch（示例 angle=0，即右侧）
    angle = 0.0
    patch_pil, (x1, y1, x2, y2) = extract_patch(img_gray, center, radius, patch_size, angle)
    patch_np = np.array(patch_pil)  # (patch_size, patch_size)

    # Patch → 3 通道 Tensor
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((patch_size, patch_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    patch_t_3 = transform(patch_pil)  # (3, patch_size, patch_size)

    # 第 4 通道：径向距离 Gaussian
    # 计算 Patch 中心坐标 (cx, cy) 在 patch 内部位置
    x0, y0 = center
    xi = int(x0 + radius * math.cos(angle))
    yi = int(y0 + radius * math.sin(angle))
    cx = xi - x1
    cy = yi - y1
    Hs = patch_size

    yy = np.arange(0, Hs)
    xx = np.arange(0, Hs)
    grid_y, grid_x = np.meshgrid(yy, xx, indexing='ij')
    dist = np.sqrt((grid_x - cx)**2 + (grid_y - cy)**2)
    sigma = radius * 0.1
    radial = np.exp(-0.5 * ((dist - radius) / sigma)**2)
    radial = (radial - radial.min()) / (radial.max() - radial.min() + 1e-8)
    radial = np.clip(radial, 0, 1)
    radial_tensor = torch.from_numpy(radial).unsqueeze(0).float()  # (1, patch_size, patch_size)

    # 拼接成 4 通道
    patch_all = torch.cat([patch_t_3, radial_tensor], dim=0)  # (4, patch_size, patch_size)
    patch_tensor = patch_all.unsqueeze(0).to(device)         # (1, 4, patch_size, patch_size)

    # 生成 CAM
    cam_map, logits = generate_cam(model, patch_tensor)
    probs = F.softmax(torch.tensor(logits), dim=1).cpu().numpy().flatten()

    class_names = ['Normal', 'Break']

    # 可视化布局
    fig = plt.figure(figsize=(16, 6))
    fig.suptitle("Tear Film Rupture Detection Visualization", fontsize=18)

    # 1. 原图 + Patch 矩形框
    ax1 = fig.add_subplot(1, 4, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rect_img = img_rgb.copy()
    cv2.rectangle(rect_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    ax1.imshow(rect_img)
    ax1.set_title("Original Image + Patch")
    ax1.axis('off')

    # 2. Patch（灰度图）
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.imshow(patch_np, cmap='gray')
    ax2.set_title(f"{patch_size}×{patch_size} Patch")
    ax2.axis('off')

    # 3. CAM heatmap 覆盖在 Patch 上
    ax3 = fig.add_subplot(1, 4, 3)
    cam_resized = cv2.resize(cam_map, (patch_np.shape[1], patch_np.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    patch_color = cv2.cvtColor(patch_np, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(patch_color, 0.7, heatmap, 0.3, 0)
    ax3.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax3.set_title("CAM Heatmap on Patch")
    ax3.axis('off')

    # 4. 二分类概率柱状图
    ax4 = fig.add_subplot(1, 4, 4)
    bars = ax4.barh(class_names, probs, color='tomato')
    ax4.set_xlim(0, 1)
    ax4.set_title("Classification Probabilities")
    ax4.invert_yaxis()
    ax4.set_xlabel("Probability")
    for i, bar in enumerate(bars):
        ax4.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{probs[i] * 100:.1f}%",
            va='center',
            fontweight='bold'
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def visualize_confusion_matrix(test_csv_path, output_root):
    """
    从 test_results.csv 读取真值与预测概率，生成“绝对个数”混淆矩阵热力图。
    """
    if not os.path.exists(test_csv_path):
        print(f"Cannot find {test_csv_path}, skipping confusion matrix visualization.")
        return

    df = pd.read_csv(test_csv_path)
    # 二分类时假定：第一列是 id，第二列 true(label)，最后一列 score_positive
    if 'true_0' in df.columns and 'score_1' in df.columns:
        y_true = df['true_0'].values
        y_score_pos = df['score_1'].values
    else:
        y_true = df.iloc[:, 1].values
        y_score_pos = df.iloc[:, -1].values

    y_pred = (y_score_pos > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    classes = ['Normal', 'Break']

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Counts)")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j, i, format(cm[i, j], 'd'),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black"
        )

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    os.makedirs(output_root, exist_ok=True)
    save_path = os.path.join(output_root, "confusion_matrix_counts.png")
    plt.savefig(save_path)
    print(f">>> Confusion matrix saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(MODEL_NAME, CHECKPOINT_PATH, IN_CHANNELS, NUM_CLASSES).to(device)
    visualize_tearfim(IMAGE_PATH, model, CENTER, RADIUS, PATCH_SIZE, device)
    # visualize_confusion_matrix(TEST_CSV_PATH, os.path.dirname(TEST_CSV_PATH))