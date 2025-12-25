import os
import cv2
import numpy as np
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import find_contours
from skimage.draw import polygon_perimeter
from skimage.feature import peak_local_max
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.morphology import  footprint_rectangle,erosion, dilation
from skimage import color, filters,morphology,measure,segmentation
from skimage.measure import regionprops
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
RESULT_DIR = "result_改进梯度"
os.makedirs(RESULT_DIR, exist_ok=True)

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

relative_errors=[]

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

    thresh = threshold_otsu(blur)
    binary = blur > thresh  # skimage 返回 bool
    #binary = ~binary

    # -----------------------------
    # 2. 形态学清理
    # -----------------------------
    selem = footprint_rectangle((3,3))

    # 腐蚀去孤立点
    binary = erosion(binary, selem)

    # 膨胀恢复主体
    binary = dilation(binary, selem)

    # -----------------------------
    # 3. 连通域面积过滤
    # -----------------------------
    labeled, num = ndi.label(binary)
    binary_clean = np.zeros_like(binary, dtype=bool)

    for i in range(1, num+1):
        region = (labeled == i)
        if region.sum() >= 50:  # 面积阈值
            binary_clean |= region

    binary = binary_clean

    # -----------------------------
    # 4. 距离变换
    # -----------------------------
    distance = ndi.distance_transform_edt(binary)

    # -----------------------------
    # 5. 局部极大值作为分水岭种子
    # -----------------------------
    markers = np.zeros_like(distance, dtype=np.int32)
    current_label = 1

    labeled_cc, num_cc = ndi.label(binary)

    for cc in range(1, num_cc+1):
        region = (labeled_cc == cc)
        if region.sum() < 20:
            continue

        dist_region = distance * region
        local_radius = np.max(dist_region)

        min_dist = int(0.9 * local_radius)
        min_dist = max(min_dist, 3)

        coords = peak_local_max(dist_region, min_distance=min_dist, labels=region)

        for r, c in coords:
            markers[r, c] = current_label
            current_label += 1

    # -----------------------------
    # 6. Watershed 分割
    # -----------------------------
    gray_blur = gaussian(rgb2gray(img), sigma=1)
    gradient = filters.sobel(distance) 

    elevation = 0.6 *(-distance) + 0.4*gradient
    labels = segmentation.watershed(gradient, markers, mask=binary)

    pred_count=len(regionprops(labels))
    if gt_count>0:
        rel_err=abs(pred_count-gt_count)/gt_count
        relative_errors.append(rel_err)
    # 10. 提取分割边界
    boundaries = segmentation.find_boundaries(labels, mode="outer")
    result = img.copy()
    result[labels == 0] = (result[labels == 0] * 0.3).astype(result.dtype)
    boundaries = segmentation.find_boundaries(labels, mode="outer")
    result[boundaries] = [255, 0, 0]
    fig, axs = plt.subplots(1, 4, figsize=(14, 4))
    axs[0].imshow(img); axs[0].set_title("Denoised Image")
    axs[1].imshow(gt_mask, cmap="gray"); axs[1].set_title("GT Mask")
    axs[2].imshow(-distance, cmap="magma"); axs[2].set_title("elevation")
    axs[3].imshow(result); axs[3].set_title(f"Pred {pred_count}/{gt_count}")

    for ax in axs:
        ax.axis("off")

    plt.savefig(
        os.path.join(RESULT_DIR, f"{id_}_new_water_grad.png"),
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()

    plt.figure(figsize=(4,4))
    plt.imshow(result)
    plt.axis("off")
    plt.savefig(os.path.join(RESULT_DIR,f"{id_}_new_water_grad_result.png"),dpi=150,bbox_inches="tight")
    plt.close()
mean_rel_error=np.mean(relative_errors)
print(f"\nAverage Relative Error on Test Set:{mean_rel_error:.4f}")
print("Done.")