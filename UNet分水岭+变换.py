import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from skimage.io import imread
from skimage.transform import resize
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries

# ======================
# 基本配置
# ======================
IMG_SIZE = 256
TRAIN_PATH = "../data-science-bowl-2018/stage1_train/"
RESULT_DIR = "result_UNet变换"
MODEL_PATH = "unet_nuclei.keras"

os.makedirs(RESULT_DIR, exist_ok=True)

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# ======================
# 读取所有 ID 并划分
# ======================
ids = sorted(next(os.walk(TRAIN_PATH))[1])
n_total = len(ids)

train_ids = ids[:int(0.7 * n_total)]
val_ids   = ids[int(0.7 * n_total):int(0.85 * n_total)]
test_ids  = ids[int(0.85 * n_total):]

# ======================
# 数据读取函数
# ======================
def load_data(id_list):
    X = np.zeros((len(id_list), IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    Y = np.zeros((len(id_list), IMG_SIZE, IMG_SIZE, 1), dtype=np.uint8)

    for i, id_ in tqdm(enumerate(id_list), total=len(id_list)):
        path = os.path.join(TRAIN_PATH, id_)

        img = imread(path + "/images/" + id_ + ".png")[:, :, :3]
        img = resize(img, (IMG_SIZE, IMG_SIZE), preserve_range=True)
        X[i] = img

        mask = np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.uint8)
        for m in os.listdir(path + "/masks/"):
            msk = imread(path + "/masks/" + m)
            msk = resize(msk, (IMG_SIZE, IMG_SIZE), preserve_range=True)
            mask = np.maximum(mask, np.expand_dims(msk, -1))

        Y[i] = (mask > 0).astype(np.uint8)

    return X, Y

# ======================
# UNet 模型
# ======================
def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
    return x

inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
x = layers.Rescaling(1./255)(inputs)

c1 = conv_block(x, 16)
p1 = layers.MaxPooling2D()(c1)

c2 = conv_block(p1, 32)
p2 = layers.MaxPooling2D()(c2)

c3 = conv_block(p2, 64)
p3 = layers.MaxPooling2D()(c3)

c4 = conv_block(p3, 128)
p4 = layers.MaxPooling2D()(c4)

c5 = conv_block(p4, 256)

u6 = layers.Conv2DTranspose(128, 2, strides=2, padding="same")(c5)
u6 = layers.concatenate([u6, c4])
c6 = conv_block(u6, 128)

u7 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(c6)
u7 = layers.concatenate([u7, c3])
c7 = conv_block(u7, 64)

u8 = layers.Conv2DTranspose(32, 2, strides=2, padding="same")(c7)
u8 = layers.concatenate([u8, c2])
c8 = conv_block(u8, 32)

u9 = layers.Conv2DTranspose(16, 2, strides=2, padding="same")(c8)
u9 = layers.concatenate([u9, c1])
c9 = conv_block(u9, 16)

outputs = layers.Conv2D(1, 1, activation="sigmoid")(c9)

model = keras.Model(inputs, outputs)
model.compile(optimizer="adam", loss="binary_crossentropy")

# ======================
# 训练 or 加载模型
# ======================
if os.path.exists(MODEL_PATH):
    print("Loading existing model...")
    model = keras.models.load_model(MODEL_PATH)
else:
    print("Loading training data...")
    X_train, Y_train = load_data(train_ids)

    print("Loading validation data...")
    X_val, Y_val = load_data(val_ids)
    model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=20,
        batch_size=8
    )
    model.save(MODEL_PATH)

# ======================
# Test 集：UNet + Watershed
# ======================
print("Processing test set...")

relative_errors=[]

for id_ in tqdm(test_ids):
    path = os.path.join(TRAIN_PATH, id_)

    img = imread(f"{path}/images/{id_}.png")[:, :, :3]
    img = resize(img, (IMG_SIZE, IMG_SIZE), preserve_range=True).astype(np.uint8)

    # GT mask
    gt_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    gt_count=0
    for m in os.listdir(path + "/masks/"):
        msk = imread(path + "/masks/" + m)
        msk = resize(msk, (IMG_SIZE, IMG_SIZE), preserve_range=True)
        msk=msk*random.randint(64,191)
        gt_mask = np.maximum(gt_mask, msk)
        gt_count += 1

    # UNet 预测
    pred_prob = model.predict(img[None, ...], verbose=0)[0, :, :, 0]

    # Watershed
    binary = pred_prob > 0.5
    distance = ndi.distance_transform_edt(binary)

    # 自适应 markers（关键改动）
    markers = np.zeros_like(distance, dtype=np.int32)
    current_label = 1

    # 先按连通区域划分候选核
    labeled_cc, num_cc = ndi.label(binary)

    for cc in range(1, num_cc + 1):
        region = (labeled_cc == cc)

        # 过滤极小区域（噪声）
        if region.sum() < 20:
            continue

        # 当前区域的距离图
        dist_region = distance * region

        # 自适应 min_distance（和核大小相关）
        local_radius = np.max(dist_region)
        min_dist = int(0.8 * local_radius)

        # 合理下限，防止过小
        min_dist = max(min_dist, 3)

        # 在该区域内找局部极大值
        coords = peak_local_max(
            dist_region,
            min_distance=min_dist,
            threshold_abs=0.2 * distance.max(),
            labels=binary
        )

        # 写入 markers
        for r, c in coords:
            markers[r, c] = current_label
            current_label += 1

    # Watershed（只在核区域内）
    labels = watershed(
        -distance,
        markers,
        mask=binary
    )
    pred_count = len(regionprops(labels))
    if gt_count > 0:
        rel_err = abs(pred_count - gt_count) / gt_count
        relative_errors.append(rel_err)

    # 可视化
    vis = img.copy()
    vis[labels == 0] = vis[labels == 0] * 0.3
    boundaries = find_boundaries(labels, mode="outer")
    vis[boundaries] = [255, 0, 0]

    fig, axs = plt.subplots(1, 4, figsize=(14, 4))
    axs[0].imshow(img); axs[0].set_title("Original")
    axs[1].imshow(gt_mask, cmap="gray"); axs[1].set_title("GT Mask")
    axs[2].imshow(pred_prob, cmap="jet"); axs[2].set_title("U-Net Prob")
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