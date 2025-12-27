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
from skimage.morphology import  footprint_rectangle,erosion, dilation,h_minima
from skimage import color, filters,morphology,measure,segmentation
from skimage.measure import regionprops
from scipy import ndimage as ndi
#去噪声
def denoise_image(img):
    denoised = cv2.fastNlMeansDenoisingColored(
        img, None, h=10, hColor=10,
        templateWindowSize=7, searchWindowSize=21
    )
    denoised = cv2.GaussianBlur(denoised, (3, 3), 0)
    return denoised

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
#用于计算分割错误
relative_errors=[]

#对测试集图片进行分割
for id_ in tqdm(test_ids):
    #读取图像
    path = os.path.join(TRAIN_PATH, id_)
    img = imread(path + "/images/" + id_ + ".png")[:, :, :3]
    img = resize(img, (IMG_SIZE, IMG_SIZE), preserve_range=True).astype(np.uint8)
    img = denoise_image(img)

    #GT mask作为答案参考(竞赛给出的答案)
    gt_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    gt_count=0
    for m in os.listdir(path + "/masks/"):
        msk = imread(path + "/masks/" + m)
        msk = resize(msk, (IMG_SIZE, IMG_SIZE), preserve_range=True)
        msk = msk * random.randint(64, 191)
        gt_mask = np.maximum(gt_mask, msk)
        gt_count += 1
    #将图像转为灰度图并进行高斯模糊
    gray = rgb2gray(img)                
    blur = gaussian(gray, sigma=1)
    #利用otsu算法自动选取阈值并进行二值化
    thresh = threshold_otsu(blur)
    binary = blur > thresh  # skimage 返回 bool
    #binary = ~binary
    #对二值化图像进行腐蚀和膨胀(去除噪声)
    selem = footprint_rectangle((3,3))
    binary = erosion(binary, selem)
    binary = dilation(binary, selem)
    #为每一个细胞区域分配一个标记
    labeled, num = ndi.label(binary)
    # 移除小区域噪声
    binary_clean = np.zeros_like(binary, dtype=bool)
    for i in range(1, num+1):
        region = (labeled == i)
        if region.sum() >= 50:
            binary_clean |= region

    binary = binary_clean
    #计算距离变换并选取标记点
    distance = ndi.distance_transform_edt(binary)
    markers = np.zeros_like(distance, dtype=np.int32)
    current_label = 1
    labeled_cc, num_cc = ndi.label(binary)
    #利用局部最大值选取细胞核心区域
    for cc in range(1, num_cc+1):
        region = (labeled_cc == cc)
        if region.sum() < 20:
            continue
        #取出该区域内的距离变换值
        dist_region = distance * region
        #估计该细胞的半径
        local_radius = np.max(dist_region)
        #设置两个细胞核心的最小距离
        min_dist = int(0.9 * local_radius)
        min_dist = max(min_dist, 3)
        #寻找局部最大值
        coords = peak_local_max(dist_region, min_distance=min_dist, labels=region)
        #将局部最大值位置作为注水点
        for r, c in coords:
            markers[r, c] = current_label
            current_label += 1
    #引入梯度
    gx = ndi.sobel(binary, axis=0)
    gy = ndi.sobel(binary, axis=1)
    gradient = np.hypot(gx, gy)
    gradient /= (gradient.max() + 1e-8)
    edge_band = ndi.binary_dilation(gray, iterations=2) ^ \
                ndi.binary_erosion(gray, iterations=2)
    gradient *= edge_band
    #混合梯度和二值图像
    alpha = 0.01
    elevation = alpha * gradient + (1 - alpha) * binary
    elevation /= (elevation.max() + 1e-8)
    elevation = h_minima(elevation, h=0.01)
    #应用改进分水岭算法进行分割
    labels = segmentation.watershed(
        elevation,      #对加权后的图像做分水岭
        markers,        #标记矩阵
        mask=binary    #限定分割范围在binary内
    )
    #计算预测细胞数量并计算相对误差
    pred_count=len(regionprops(labels))
    if gt_count>0:
        rel_err=abs(pred_count-gt_count)/gt_count
        relative_errors.append(rel_err)
    #获取分割边界
    boundaries = segmentation.find_boundaries(labels, mode="outer")
    #可视化
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
    #保存结果的单张图
    plt.figure(figsize=(4,4))
    plt.imshow(result)
    plt.axis("off")
    plt.savefig(os.path.join(RESULT_DIR,f"{id_}_new_water_grad_result.png"),dpi=150,bbox_inches="tight")
    plt.close()
mean_rel_error=np.mean(relative_errors)
print(f"\nAverage Relative Error on Test Set:{mean_rel_error:.4f}")
print("Done.")