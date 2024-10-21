"""
@Time ： 2024/10/20 20:46
@Auth ： opprash
@File ：albu.py
@IDE ：PyCharm
"""
import cv2
import numpy as np
import os
from albumentations import (
    Compose,
    GaussianBlur,
    HorizontalFlip,
    Rotate,
    OpticalDistortion,
    GaussNoise,
    ToFloat,
    ISONoise,
)
from albumentations.pytorch import ToTensorV2

# 定义增强流水线
augment_pipeline = Compose([
    GaussNoise(p=0.5),  # 添加高斯噪声
    HorizontalFlip(p=0.5),  # 水平翻转
    Rotate(limit=30, p=0.5),  # 随机旋转
    OpticalDistortion(distort_limit=1, shift_limit=0.2, p=0.5),  # 光学畸变
    ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), always_apply=False, p=0.5),  # 模拟摄像头传感器噪音
    ToFloat(),  # 转换到浮点数
    ToTensorV2(),  # 转换成 PyTorch 张量
])

# 加载图像
image_path = 'F:\\data\\黑马\\albu\\11864.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将 BGR 转换为 RGB

# 应用增强
augmented_images = []
for _ in range(10):  # 对同一张图片应用多次增强
    augmented = augment_pipeline(image=image)
    augmented_image = augmented['image']
    augmented_images.append(augmented_image)

# 保存增强后的图像
output_dir = 'F:\\data\\黑马\\albu\\out'
os.makedirs(output_dir, exist_ok=True)
for idx, img in enumerate(augmented_images):
    output_path = os.path.join(output_dir, f'augmented_{idx}.jpg')
    cv2.imwrite(output_path, cv2.cvtColor(img.numpy().transpose(1, 2, 0), cv2.COLOR_RGB2BGR))

print(f"Augmented images saved to {output_dir}")


aug_labels = {
    "buildings":"buildings",
    "structures":"buildings",
    "edifices":"buildings",
    "constructs":"buildings",
    "woods":"forest",
    "forest":"forest",
    "jungle":"forest",
    "copse":"buildings",
    "glacier":"glacier",
    "glacial":"glacier",
    "ice_sheet ":"glacier",
    "mountain":"mountain",
    "hill":"buildings",
    "range":"buildings",
    "highlands":"mountain",
    "hillside":"mountain",
    "sea":"sea",
    "ocean":"sea",
    "seaway":"sea",
    "street":"street",
    "road":"street",
    "avenue":"street",
    "lane":"street",
    "way":"buildings",
}



