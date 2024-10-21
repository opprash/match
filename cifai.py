"""
@Time ： 2024/10/18 15:15
@Auth ： opprash
@File ：cifai.py
@IDE ：PyCharm
"""

import os
import pickle
import numpy as np
from PIL import Image


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def save_images(root_dir, data_batches, meta_data, train=True):
    """
    保存CIFAR-100数据集中的图片到对应的子文件夹中。

    :param root_dir: 保存图片的根目录
    :param data_batches: 包含数据批次的列表
    :param meta_data: 元数据字典，包含类别名称
    """
    # 创建每个类别的文件夹
    classes = [str(key.decode('utf-8')) for key in meta_data[b'fine_label_names']]
    for class_name in classes:
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

    # 遍历所有批次的数据
    for batch_file in data_batches:
        batch_data = unpickle(batch_file)
        images = batch_data[b'data']
        labels = batch_data[b'fine_labels']

        # 将图像从一维数组转为三维数组
        images = np.reshape(images, (-1, 3, 32, 32))
        images = np.transpose(images, (0, 2, 3, 1))  # 调整形状为 (height, width, channels)

        # 遍历图像并保存
        for idx, (image, label) in enumerate(zip(images, labels)):
            class_name = classes[label]
            class_dir = os.path.join(root_dir, class_name)
            img_path = os.path.join(class_dir, f"{'train' if train else 'test'}_{batch_file.split('/')[-1]}_{idx}.png")

            # 将numpy数组转换为PIL图像并保存
            img = Image.fromarray(image)
            img.save(img_path)


# 指定数据集的本地路径
data_dir = 'C:\\Users\\oppra\\Desktop\\智能客服2024\\个人\\黑马\\clip\\cifar-100-python'
meta_file = os.path.join(data_dir, 'meta')
train_files = [os.path.join(data_dir, 'train')]
test_files = [os.path.join(data_dir, 'test')]

# 加载元数据以获取类别名称
meta_data = unpickle(meta_file)

# 保存训练集图片
save_images('', train_files, meta_data, train=True)

# 保存测试集图片
save_images(os.path.join(data_dir, 'test1'), test_files, meta_data, train=False)
