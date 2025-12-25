import os
import cv2
import numpy as np
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import opening, closing, square
from skimage.measure import find_contours
from skimage.draw import polygon_perimeter
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from skimage.morphology import opening, closing, footprint_rectangle
from scipy import ndimage as ndi

def denoise_image(img):
    denoised = cv2.fastNlMeansDenoisingColored(
        img, None, h=10, hColor=10,
        templateWindowSize=7, searchWindowSize=21
    )
    denoised = cv2.GaussianBlur(denoised, (3, 3), 0)
    return denoised

# -----------------------------
# 配置路径与随机种子
# -----------------------------
IMG_SIZE = 256
TRAIN_PATH = "../data-science-bowl-2018/stage1_train/"
RESULT_DIR = "result_Otsu"
os.makedirs(RESULT_DIR, exist_ok=True)

relative_errors=[]

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 获取样本ID列表
ids = sorted(next(os.walk(TRAIN_PATH))[1])
n = len(ids)
train_ids = ids[:int(0.7*n)]
val_ids   = ids[int(0.7*n):int(0.85*n)]
test_ids  = ids[int(0.85*n):]

# -----------------------------
# 批量处理
# -----------------------------
for id_ in tqdm(test_ids):
    path = os.path.join(TRAIN_PATH, id_)

    img = imread(path + "/images/" + id_ + ".png")[:, :, :3]
    img = resize(img, (IMG_SIZE, IMG_SIZE), preserve_range=True).astype(np.uint8)
    img = denoise_image(img)

    # GT mask（仅用于显示）
    gt_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    gt_count=0
    for m in os.listdir(path + "/masks/"):
        msk = imread(path + "/masks/" + m)
        msk = resize(msk, (IMG_SIZE, IMG_SIZE), preserve_range=True)
        msk = msk * random.randint(64, 191)
        gt_mask = np.maximum(gt_mask, msk)
        gt_count += 1
    # 2. 灰度 + 高斯平滑
    gray = rgb2gray(img)                
    blur = gaussian(gray, sigma=1)

    # 3. Otsu 阈值
    thresh = threshold_otsu(blur)
    binary = blur > thresh             
    # binary = ~binary
    # 4. 形态学清理
    
    selem = footprint_rectangle((3, 3))
    cleaned = opening(binary, selem)
    cleaned = closing(cleaned, selem)

    # 5. 找轮廓（skimage 返回的是浮点坐标）
    contours = find_contours(cleaned, level=0.5)
    _,pred_count = ndi.label(cleaned)
    if gt_count>0:
        rel_err=abs(pred_count-gt_count)/gt_count
        relative_errors.append(rel_err)
    # 6. 过滤轮廓（按面积）
    valid_contours = []
    for contour in contours:
        if contour.shape[0] < 20:
            continue

        # 轮廓面积（Shoelace formula）
        x = contour[:, 1]
        y = contour[:, 0]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

        if area > 20:
            valid_contours.append(contour)

    # 7. 在原图上画红色轮廓
    result = img.copy()

    for contour in valid_contours:
        rr, cc = polygon_perimeter(
            contour[:, 0],
            contour[:, 1],
            shape=result.shape
        )
        result[rr, cc] = [255, 0, 0]  # RGB 红色
    fig, axs = plt.subplots(1, 4 , figsize=(14, 4))
    axs[0].imshow(img); axs[0].set_title("Denoised Image")
    axs[1].imshow(gt_mask, cmap="gray"); axs[1].set_title("GT Mask")
    axs[2].imshow(binary, cmap="gray"); axs[2].set_title("binary")
    axs[3].imshow(result); axs[3].set_title(f"Pred {pred_count}/{gt_count}")

    for ax in axs:
        ax.axis("off")

    plt.savefig(
        os.path.join(RESULT_DIR, f"{id_}_vis.png"),
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()

    plt.figure(figsize=(4,4))
    plt.imshow(result)
    plt.axis("off")
    plt.savefig(os.path.join(RESULT_DIR,f"{id_}_otsu_result.png"),dpi=150,bbox_inches="tight")
    plt.close()
mean_rel_error=np.mean(relative_errors)
print(f"\nAverage Relative Error on Test Set:{mean_rel_error:.4f}")
print("Done.")