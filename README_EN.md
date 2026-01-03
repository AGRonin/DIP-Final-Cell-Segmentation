# DIP-Final-Cell-Segmentation

This project is the final course assignment for the **Digital Image Processing** course (Fall Semester, Academic Year 2025–2026) at the **School of Artificial Intelligence, Beijing Normal University**.  
It focuses on **cell image segmentation based on the watershed algorithm and its improvements**.

---

## Project Overview

The Chinese title of this project is **基于分水岭算法的细胞图像分割及其改进**, as **cell image segmentation based on the watershed algorithm and its improvements** in English.  
In this work, we employ and improve the classical watershed algorithm by enhancing **foreground/background separation, marker (seed) selection, image selection, adaptive threshold selection, and morphological processing**. We compare the results with those of the **Otsu thresholding algorithm** and the **traditional watershed algorithm**.

Finally, on the **Data Science Bowl 2018** dataset, our proposed method —  
**U-Net foreground probability extraction + adaptive marker selection + adaptive threshold selection + morphological judgment + gradient–probability weighted image + watershed algorithm** —  
achieves an **average relative error of 0.1334 in cell count segmentation**.

Compared with the traditional watershed algorithm (which suffers from severe over-segmentation and under-segmentation), our method shows a significant improvement.

---

## Project Structure

```text
.
├── LICENSE
├── README.md
├── Otsu.py                              # Traditional threshold segmentation based on Otsu
├── 传统分水岭.py                        # Traditional watershed algorithm
├── 传统分水岭+梯度.py                  # Traditional watershed using gradient images
├── 改进分水岭.py                        # Improved adaptive watershed algorithm
├── 改进分水岭+梯度.py                  # Improved adaptive watershed using gradient images
├── UNet分水岭.py                        # Improved adaptive watershed with U-Net foreground markers
├── UNet分水岭+梯度.py                  # U-Net-based adaptive watershed using gradient images
├── UNet分水岭+变换.py                  # U-Net-based adaptive watershed with adaptive thresholds
├── UNet分水岭+变换+梯度.py              # U-Net-based adaptive watershed with adaptive thresholds and gradient images
├── UNet分水岭+变换+加权.py              # U-Net-based adaptive watershed using gradient–probability weighted images
├── UNet分水岭+变换+加权+形态学.py        # U-Net-based adaptive watershed with morphology and weighted images
├── 改进分水岭+变换+加权+形态学.py          # Improved adaptive watershed with morphology and weighted images
├── 改进分水岭+变换+形态学.py              # Improved adaptive watershed with morphology and adaptive thresholds
├── 混合分水岭+变换+加权+形态学.py         # Hybrid watershed (U-Net + binarization) with morphology and weighted images
├── unet_nuclei.keras                   # Trained U-Net model weights
├── 第六组数字图像处理PPT.pptx         # Final presentation slides
└── 数字图像处理论文                    # LaTeX source code of the paper
```

## Data Preparation

This project uses the Data Science Bowl 2018 stage1_train dataset.
Due to the large size of the dataset, it is not included in this repository.

Please download the dataset from Kaggle and place the extracted data-science-bowl-2018 folder at the same directory level as this project’s root directory. The directory structure should be as follows:

```text
..
├── DIP-Final-Cell-Segmentation           # This project directory
├── data-science-bowl-2018                # Extracted dataset directory
|   ├── stage1_train/
|   |   ├── <image_id>/
|   |   |    ├── images/<image_id>.png
|   |   |    └── masks/*.png
|   |   └── ...
|   └── ...
```

## Training and Execution

### Training

All U-Net related scripts retain the full training and validation logic.
If you wish to retrain the model, simply delete unet_nuclei.keras and run the corresponding script normally.

### Running

Taking UNet分水岭+变换+加权+形态学.py as an example, run:

```bash
python UNet分水岭+变换+加权+形态学.py
```

For each test image, two output images will be generated:

<image_id>_vis.png, a visualization of the complete pipeline, including the original image, ground-truth mask, the image used for watershed segmentation, and the final segmentation result.

<image_id>_result.png, the segmentation result only.

The result directory structure is as follows:

```text
result_*/
├── <image_id1>_vis.png
├── <image_id1>_result.png
├── <image_id2>_vis.png
...
```

### Quantitative Evaluation Metric

For each test image, this project outputs a comparison between the predicted number of cells and the ground-truth number of cells.
The average relative error at the test-set level is used as the quantitative evaluation metric.

### Environment and Dependencies

The recommended environment is:
```text
Python >= 3.9
TensorFlow >= 2.9
numpy
opencv-python
scikit-image
scipy
matplotlib
tqdm
```

Install dependencies using:

```bash
pip install tensorflow numpy opencv-python scikit-image scipy matplotlib tqdm
```


## Contributors and License

This project is the final assignment for the Digital Image Processing course (Fall Semester, Academic Year 2025–2026) at the School of Artificial Intelligence, Beijing Normal University, instructed by Prof. Jia Li. The group number is Group 6.

The project was jointly completed by AGRonin, evol-te, and zyc286.

This project is open-sourced under the MIT License.

Special note for students:
If you use this code directly for coursework, please comply with your institution’s academic integrity policy. Submitting this code as your own original work may constitute academic misconduct.

## Acknowledgements

We sincerely thank all classmates and instructors who contributed their time and effort to this project.
And thank you for reading all the way to the end.

