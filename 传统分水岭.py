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
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.morphology import opening, closing, footprint_rectangle
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
RESULT_DIR = "result_传统"
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

    # 2. Otsu 二值化
    thresh = filters.threshold_otsu(blur)
    binary = blur > thresh
    #binary = ~binary
    # 3. 开运算（去噪）
    selem = morphology.footprint_rectangle((3, 3))
    opening = morphology.opening(binary, selem)

    # 4. sure background（膨胀）
    sure_bg = morphology.dilation(opening, selem)

    # 5. 距离变换
    distance = ndi.distance_transform_edt(opening)

    # 6. sure foreground
    sure_fg = distance > 0.2 * distance.max()

    # 7. unknown 区域
    unknown = sure_bg & (~sure_fg)

    # 8. 连通区域标记
    markers = measure.label(sure_fg)
    markers = markers + 1
    markers[unknown] = 0

    # 9. Watershed
    labels = segmentation.watershed(
        -distance,        
        markers,
        mask=sure_bg
    )

    pred_count=len(regionprops(labels))
    if gt_count>0:
        rel_err=abs(pred_count-gt_count)/gt_count
        relative_errors.append(rel_err)
    # 10. 提取分割边界
    boundaries = segmentation.find_boundaries(labels, mode="outer")

    # 11. 画红色边界
    result = img.copy()
    result[boundaries] = [255, 0, 0]
    fig, axs = plt.subplots(1, 4, figsize=(14, 4))
    axs[0].imshow(img); axs[0].set_title("Denoised Image")
    axs[1].imshow(gt_mask, cmap="gray"); axs[1].set_title("GT Mask")
    axs[2].imshow(-distance, cmap="magma"); axs[2].set_title("distance")
    axs[3].imshow(result); axs[3].set_title(f"Pred {pred_count}/{gt_count}")

    for ax in axs:
        ax.axis("off")

    plt.savefig(
        os.path.join(RESULT_DIR, f"{id_}_water.png"),
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()

    plt.figure(figsize=(4,4))
    plt.imshow(result)
    plt.axis("off")
    plt.savefig(os.path.join(RESULT_DIR,f"{id_}_water_result.png"),dpi=150,bbox_inches="tight")
    plt.close()
mean_rel_error=np.mean(relative_errors)
print(f"\nAverage Relative Error on Test Set:{mean_rel_error:.4f}")
print("Done.")