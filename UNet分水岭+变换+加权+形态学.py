#best，问题在于binary图像里面有洞
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
from skimage.morphology import h_minima
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries

# 基本配置
A=0.9
IMG_SIZE = 256
TRAIN_PATH = "../data-science-bowl-2018/stage1_train/"
# RESULT_DIR = f"result_UNet变换加权形态_{A}_Binary"
RESULT_DIR = "result_UNet变换加权形态"
MODEL_PATH = "unet_nuclei.keras"

os.makedirs(RESULT_DIR, exist_ok=True)

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 图像降噪函数（这里有降噪是因为对于梯度图像，噪声较大）
def denoise_image(img):
    denoised = cv2.fastNlMeansDenoisingColored(
        img,
        None,
        h=10,# 亮度去噪强度
        hColor=10,# 颜色去噪强度
        templateWindowSize=7,
        searchWindowSize=21
    )

    # 轻度高斯平滑
    denoised = cv2.GaussianBlur(denoised, (3, 3), 0)

    return denoised

def suppress_background(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 结构元素大小约等于背景变化尺度
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (25, 25)
    )

    background = cv2.morphologyEx(
        gray, cv2.MORPH_OPEN, kernel
    )

    # 背景抑制
    suppressed = cv2.subtract(gray, background)

    # 归一化
    suppressed = suppressed.astype(np.float32)
    suppressed = (suppressed - suppressed.min()) / (
        suppressed.max() - suppressed.min() + 1e-8
    )

    return suppressed

def smooth_image(img_float):
    return cv2.GaussianBlur(img_float, (3, 3), 0.5)

# 读取所有 ID 并划分
ids = sorted(next(os.walk(TRAIN_PATH))[1])
n_total = len(ids)

train_ids = ids[:int(0.7 * n_total)]
val_ids   = ids[int(0.7 * n_total):int(0.85 * n_total)]
test_ids  = ids[int(0.85 * n_total):]

# 数据读取函数
def load_data(id_list):
    X = np.zeros((len(id_list), IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    Y = np.zeros((len(id_list), IMG_SIZE, IMG_SIZE, 1), dtype=np.uint8)

    for i, id_ in tqdm(enumerate(id_list), total=len(id_list)):
        path = os.path.join(TRAIN_PATH, id_)

        img = imread(path + "/images/" + id_ + ".png")[:, :, :3]
        img = resize(img, (IMG_SIZE, IMG_SIZE), preserve_range=True).astype(np.uint8)
        img = denoise_image(img)
        X[i] = img

        mask = np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.uint8)
        for m in os.listdir(path + "/masks/"):
            msk = imread(path + "/masks/" + m)
            msk = resize(msk, (IMG_SIZE, IMG_SIZE), preserve_range=True)
            mask = np.maximum(mask, np.expand_dims(msk, -1))

        Y[i] = (mask > 0).astype(np.uint8)

    return X, Y

# U-Net
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

# 训练/加载
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

# 分水岭算法开始
print("Processing test set...")

relative_errors=[]

for id_ in tqdm(test_ids):
    path = os.path.join(TRAIN_PATH, id_)

    img = imread(f"{path}/images/{id_}.png")[:, :, :3]
    img = resize(img, (IMG_SIZE, IMG_SIZE), preserve_range=True).astype(np.uint8)

    # GT mask（答案）
    gt_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    gt_count=0
    for m in os.listdir(path + "/masks/"):
        msk = imread(path + "/masks/" + m)
        msk = resize(msk, (IMG_SIZE, IMG_SIZE), preserve_range=True)
        msk = msk * random.randint(64, 191)
        gt_mask = np.maximum(gt_mask, msk)
        gt_count += 1

    # 先降噪
    img = denoise_image(img)

    # U-Net预测图
    pred_prob = model.predict(img[None, ...], verbose=0)[0, :, :, 0]

    # 生成Binary和Distance
    binary = pred_prob > 0.5
    distance = ndi.distance_transform_edt(binary)
    labeled_cc, num_cc = ndi.label(binary)

    # 改进分水岭+变换
    markers = np.zeros_like(distance, dtype=np.int32)
    current_label = 1

    for cc in range(1, num_cc + 1):
        region = (labeled_cc == cc)
        if region.sum() < 30:
            continue

        props = regionprops(region.astype(np.uint8))[0]

        dist_region = distance * region

        # 形态学处理
        if props.solidity > 0.9 and props.eccentricity < 0.75:
            r, c = np.unravel_index(np.argmax(dist_region), dist_region.shape)
            markers[r, c] = current_label
            current_label += 1
            continue

        # 允许分裂
        median_radius = np.median(dist_region[region])
        min_distance = int(np.clip(A * median_radius, 4, 10))

        coords = peak_local_max(
            dist_region,
            min_distance=min_distance,
            threshold_abs=0.15 * dist_region.max(),
            labels=region
        )

        # 写入 markers
        for r, c in coords:
            markers[r, c] = current_label
            current_label += 1
    markers = ndi.binary_dilation(markers > 0, iterations=2)
    markers = ndi.label(markers)[0]

    # 梯度
    gx = ndi.sobel(pred_prob, axis=0)
    gy = ndi.sobel(pred_prob, axis=1)
    gradient = np.hypot(gx, gy)
    gradient /= (gradient.max() + 1e-8)

    edge_band = ndi.binary_dilation(binary, iterations=2) ^ \
                ndi.binary_erosion(binary, iterations=2)
    gradient *= edge_band

    image_term = suppress_background(img)
    image_term = smooth_image(image_term)

    # 加权
    alpha = 0.01
    elevation = alpha * gradient + (1 - alpha) * image_term
    elevation /= (elevation.max() + 1e-8)
    elevation = h_minima(elevation, h=0.01)

    # 分水岭
    labels = watershed(
        elevation,
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
    axs[0].imshow(img); axs[0].set_title("Denoised Image")
    axs[1].imshow(gt_mask, cmap="gray"); axs[1].set_title("GT Mask")
    axs[2].imshow(elevation, cmap="magma"); axs[2].set_title("Elevation")
    # axs[2].imshow(binary, cmap="magma"); axs[2].set_title("Binary")
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