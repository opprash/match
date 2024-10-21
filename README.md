"""
@Time ： 2024/10/21 20:03
@Auth ： opprash
@File ：test_train.py
@IDE ：PyCharm
"""
import os
import clip
from PIL import Image
import torch

import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from tqdm import tqdm

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Load the dataset
#root = os.path.expanduser("~/.cache")
#train = CIFAR100(root, download=True, train=True, transform=preprocess)
#test = CIFAR100(root, download=True, train=False, transform=preprocess)


label_dict={
    "buildings":0,
"forest":1,
"glacier":2,
"mountain":3,
"sea":4,
"street":5,
}


label_dict_idx={
    0:"buildings",
1:"forest",
2:"glacier",
3:"mountain",
4:"sea",
5:"street",
}
import numpy as np
fr_train=open('./test_record.txt','r',encoding='utf8')
fr_test=open('./val_record.txt','r',encoding='utf8')
train=[]
test=[]
for each in fr_train:
    each = each.strip()
    each=each.split(',')
    train.append((each[0],label_dict[each[1]]))

for each in fr_test:
    each = each.strip()
    each=each.split(',')
    test.append((each[0],label_dict[each[1]]))

def get_features(dataset):
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataset):
            input_image_new = Image.open(images)
            image = preprocess(input_image_new).unsqueeze(0).to(device)
            features = model.encode_image(image.to(device))

            all_features.append(features)
            #print(features.shape)
            all_labels.append(torch.Tensor([labels]))

    #return all_features,np.ndarray(all_labels)
    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()


# Calculate the image features
train_features, train_labels = get_features(train)
test_features, test_labels = get_features(test)

# Perform logistic regression
classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
classifier.fit(train_features, train_labels)

# Evaluate using the logistic regression classifier
predictions = classifier.predict(test_features)
accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
print(f"Accuracy = {accuracy:.3f}")




# -*- coding: utf-8 -*-
import os
import sys
import cv2
import time
import random
import numpy as np
from glob import glob

np.set_printoptions(suppress=True)


def split_batch(num, batch_num):
    q, r = divmod(num, batch_num)
    split_list = [[batch_num * i, batch_num * (i + 1)] for i in range(q)]
    if split_list[-1][1] != num:
        split_list += [[q * batch_num, num]]
    return split_list


class CustomAlbumentations:

    def __init__(self, p=1.0):
        self.p = p
        self.transform = None
        prefix = "albumentations: "

        import albumentations as A

        # List of possible spatial transforms
        spatial_transforms = {
            "Affine",
            "BBoxSafeRandomCrop",
            "CenterCrop",
            "CoarseDropout",
            "Crop",
            "CropAndPad",
            "CropNonEmptyMaskIfExists",
            "D4",
            "ElasticTransform",
            "Flip",
            "GridDistortion",
            "GridDropout",
            "HorizontalFlip",
            "Lambda",
            "LongestMaxSize",
            "MaskDropout",
            "MixUp",
            "Morphological",
            "NoOp",
            "OpticalDistortion",
            "PadIfNeeded",
            "Perspective",
            "PiecewiseAffine",
            "PixelDropout",
            "RandomCrop",
            "RandomCropFromBorders",
            "RandomGridShuffle",
            "RandomResizedCrop",
            "RandomRotate90",
            "RandomScale",
            "RandomSizedBBoxSafeCrop",
            "RandomSizedCrop",
            "Resize",
            "Rotate",
            "SafeRotate",
            "ShiftScaleRotate",
            "SmallestMaxSize",
            "Transpose",
            "VerticalFlip",
            "XYMasking",
        }  # from https://albumentations.ai/docs/getting_started/transforms_and_targets/#spatial-level-transforms

        # Transforms
        T = [
            A.OneOf([
                A.ToGray(p=1.0),  # 将图像转换为灰度图
                A.CLAHE(p=1.0),  # 应用CLAHE算法对图像进行直方图均衡化
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),  # 调整图像的亮度和对比度
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),  # 调整图像的色调、饱和度和明度
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),  # 调整图像的红、绿、蓝通道
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),  # 调整图像的对比度
                A.ToSepia(p=1.0),  # 添加棕褐色滤镜
                A.ImageCompression(quality_lower=75, p=1.0),  # 压缩图像的质量
                A.Equalize(p=1.0),  # 使图像的对比度变为均衡的
                A.Posterize(num_bits=4, p=1.0),  # 减少每个颜色通道的位数
                # A.Solarize(threshold=128, p=1.0),  # 反转所有高于阈值的像素值
            ], p=1.0),
            # A.OneOf([
            #     A.RandomShadow(p=1.0),  # 在图像上添加阴影
            #     A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.5, brightness_coeff=1.5,  p=1.0),  # 在图像上添加雪花
            #     A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.5, brightness_coeff=2.5,  p=1.0),  # 在图像上添加雪花
            #     A.RandomRain(p=1.0),  # 在图像上添加雨滴
            #     A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.5, alpha_coef=0.1, p=1.0),  # 在图像上添加雾
            #     # A.RandomSunFlare(src_radius=100, src_color=(255, 255, 255), p=1.0),  # 在图像上添加太阳耀斑
            # ], p=0.1),
            A.OneOf([
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, p=1.0),  # 将图像乘以随机数或数字数组
                A.GaussNoise(var_limit=(10, 50), mean=0, p=1.0),  # 增加图像的高斯噪声
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),  # 增加图像的椒盐噪声
                A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=1)  # 使图像产生浮雕效果
            ], p=1.0),
            A.OneOf([
                A.Blur(blur_limit=(3, 5), p=1.0),  # 使用随机大小的内核模糊输入图像。
                A.MedianBlur(blur_limit=(3, 5), p=1.0),  # 中值滤波
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),  # 使用随机大小的高斯内核模糊输入图像。
                A.MotionBlur(blur_limit=5, p=1.0),  # 使用随机大小的内核将运动模糊应用于输入图像。
            ], p=1.0),
            A.OneOf([
                # A.Flip(p=1.0),  # 随机翻转图像
                A.HorizontalFlip(p=1.0),  # 随机水平翻转图像
            ], p=0.5),
            A.OneOf([
                A.Rotate(limit=5, p=1.0),  # 随机旋转图像
            ], p=0.5),
        ]
        # T = [
        #     A.Blur(p=0.01),
        #     A.MedianBlur(p=0.01),
        #     A.ToGray(p=0.01),
        #     A.CLAHE(p=0.01),
        #     A.RandomBrightnessContrast(p=0.0),
        #     A.RandomGamma(p=0.0),
        #     A.ImageCompression(quality_lower=75, p=0.0),
        # ]

        # Compose transforms
        self.transform = A.Compose(T)
        print(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))

    def __call__(self, image):
        if self.transform is None or random.random() > self.p:
            return image

        image = self.transform(image=image)["image"]  # transformed

        return image


def data_augment(file,i):
    print(f"{'-' * 10} {file} {'-' * 10}", flush=True)
    image = cv2.imread(file)

    start_time = time.time()

    yolo_image = aug_func(image)

    cv2.imwrite("./aug/"+str(i)+"_"+os.path.basename(file), yolo_image)

    print(f"file:{file} use time:{time.time() - start_time:.4f}s", flush=True)


if __name__ == '__main__':
    aug_func = CustomAlbumentations(p=1.0)
    st_time = time.time()
    files = sorted(glob(f"./rgb/*.*"))[:20]
    # files = sorted(glob(f"./xray/*.*"))[:20]
    index =0
    for file in files:
        for i in range(5):
            data_augment(file,i)
    print(f"total use time:{time.time() - st_time:.4f}")


