import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

import tensorflow as tf
from tensorflow import keras

from skimage.io import imread
from skimage.transform import resize
from skimage.feature import peak_local_max
from skimage.segmentation import watershed, find_boundaries
from skimage.measure import regionprops
from skimage.morphology import h_minima
from scipy import ndimage as ndi

# ======================
# 基本配置
# ======================
IMG_SIZE = 256
TRAIN_PATH = "../data-science-bowl-2018/stage1_train/"
RESULT_DIR = "result_混合变换加权形态"
MODEL_PATH = "unet_nuclei.keras"

os.makedirs(RESULT_DIR, exist_ok=True)

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# ======================
# 加载 UNet
# ======================
print("Loading UNet model...")
model = keras.models.load_model(MODEL_PATH)

def unet_foreground(img):
    """UNet 前景预测（返回 bool mask，与原图同尺寸）"""
    img_r = resize(
        img, (IMG_SIZE, IMG_SIZE),
        preserve_range=True, anti_aliasing=True
    ).astype(np.uint8)

    prob = model.predict(img_r[None, ...], verbose=0)[0, :, :, 0]
    binary = prob > 0.5

    binary = resize(
        binary, img.shape[:2],
        order=0, preserve_range=True
    ).astype(bool)

    return binary

# ======================
# 数据划分
# ======================
ids = sorted(next(os.walk(TRAIN_PATH))[1])
n = len(ids)

train_ids = ids[:int(0.7*n)]
val_ids   = ids[int(0.7*n):int(0.85*n)]
test_ids  = ids[int(0.85*n):]

# ======================
# 主循环
# ======================
print("Processing test set...")

relative_errors=[]

for id_ in tqdm(test_ids):
    path = os.path.join(TRAIN_PATH, id_)

    # ---------- 读图 ----------
    img_pil = Image.open(f"{path}/images/{id_}.png").convert("RGB")
    img = np.array(img_pil)
    img = img[:, :, :3]

    # GT mask（仅用于显示）
    gt_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    gt_count=0
    for m in os.listdir(path + "/masks/"):
        msk = imread(path + "/masks/" + m)
        msk = resize(msk, (IMG_SIZE, IMG_SIZE), preserve_range=True)
        msk = msk * random.randint(64, 191)
        gt_mask = np.maximum(gt_mask, msk)
        gt_count += 1

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # 转回 OpenCV BGR 格式
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)

    # Otsu 阈值
    _, binary_0 = cv2.threshold(gray, 0, 1,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 形态学清理
    kernel = np.ones((3, 3), np.uint8)
    # 1. 腐蚀去孤立点
    binary = cv2.erode(binary_0.astype(np.uint8), kernel, iterations=1)

    # 2. 膨胀恢复主体
    binary = cv2.dilate(binary, kernel, iterations=1)

    binary = binary.astype(bool)

    # 3. 连通域面积过滤
    labeled, num = ndi.label(binary)
    binary_clean = np.zeros_like(binary)

    for i in range(1, num + 1):
        region = (labeled == i)
        if region.sum() >= 50:
            binary_clean |= region

    binary = binary_clean

    # =====================================================
    # 🔑 UNet 前景方向校准（核心）
    # =====================================================
    binary_unet = unet_foreground(img)

    p_trad = binary.mean()
    p_unet = binary_unet.mean()

    if abs(p_trad - p_unet) > abs((1 - p_trad) - p_unet):
        binary = ~binary

    # ---------- 距离变换 ----------
    distance = ndi.distance_transform_edt(binary)
    labeled_cc, num_cc = ndi.label(binary)

    # ---------- Morphology-aware markers ----------
    markers = np.zeros_like(distance, dtype=np.int32)
    current_label = 1

    for cc in range(1, num_cc + 1):
        region = (labeled_cc == cc)
        if region.sum() < 30:
            continue

        props = regionprops(region.astype(np.uint8))[0]
        dist_region = distance * region

        # --- 形态 veto：不允许分裂 ---
        if props.solidity > 0.9 and props.eccentricity < 0.75:
            r, c = np.unravel_index(np.argmax(dist_region), dist_region.shape)
            markers[r, c] = current_label
            current_label += 1
            continue

        # --- 允许分裂 ---
        median_radius = np.median(dist_region[region])
        min_distance = int(np.clip(0.9 * median_radius, 4, 10))
        #0.9？
        coords = peak_local_max(
            dist_region,
            min_distance=min_distance,
            threshold_abs=0.15 * dist_region.max(),
            labels=region
        )

        for r, c in coords:
            markers[r, c] = current_label
            current_label += 1

    markers = ndi.binary_dilation(markers > 0, iterations=2)
    markers = ndi.label(markers)[0]

    # ---------- 梯度（基于 binary） ----------
    gx = ndi.sobel(binary.astype(float), axis=0)
    gy = ndi.sobel(binary.astype(float), axis=1)
    gradient = np.hypot(gx, gy)
    gradient /= (gradient.max() + 1e-8)

    edge_band = ndi.binary_dilation(binary, iterations=2) ^ \
                ndi.binary_erosion(binary, iterations=2)
    gradient *= edge_band

    # ---------- Elevation ----------
    alpha = 0.01
    elevation = alpha * gradient + (1 - alpha) * binary.astype(float)
    elevation /= (elevation.max() + 1e-8)
    elevation = h_minima(elevation, h=0.01)

    # ---------- Watershed ----------
    labels = watershed(
        elevation,
        markers,
        mask=binary
    )

    pred_count=len(regionprops(labels))
    if gt_count>0:
        rel_err=abs(pred_count-gt_count)/gt_count
        relative_errors.append(rel_err)

    # ---------- 可视化 ----------
    vis = img.copy()
    boundaries = find_boundaries(labels, mode="outer")
    vis[boundaries] = [255, 0, 0]

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(img); axs[0].set_title("Original")
    axs[1].imshow(binary, cmap="gray"); axs[1].set_title("Calibrated Binary")
    axs[2].imshow(elevation, cmap="magma"); axs[2].set_title("Elevation")
    axs[3].imshow(vis); axs[3].set_title(f"Pred {pred_count}/{gt_count}")

    for ax in axs:
        ax.axis("off")

    plt.savefig(
        os.path.join(RESULT_DIR, f"{id_}_vis.png"),
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()

    plt.figure(figsize=(4,4))
    plt.imshow(vis)
    plt.axis("off")
    plt.savefig(os.path.join(RESULT_DIR,f"{id_}_result.png"),dpi=150,bbox_inches="tight")
    plt.close()

mean_rel_error=np.mean(relative_errors)
print(f"\nAverage Relative Error on Test Set:{mean_rel_error:.4f}")
print("Done.")