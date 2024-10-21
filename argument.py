"""
@Time ： 2024/10/20 11:05
@Auth ： opprash
@File ：argument.py
@IDE ：PyCharm
"""
# 导入数据增强工具
import Augmentor

# 确定原始图像存储路径以及掩码文件存储路径
p = Augmentor.Pipeline("F:\\data\\黑马\\aug")
p.ground_truth("F:\\data\\黑马\\aug")

# 图像旋转： 按照概率0.8执行，最大左旋角度10，最大右旋角度10
p.rotate(probability=0.8, max_left_rotation=15, max_right_rotation=15)

# 图像左右互换： 按照概率0.5执行
p.flip_left_right(probability=0.5)

# 图像上下互换，按照概率0.5执行
p.flip_top_bottom(probability=0.5)

# 图像放大缩小： 按照概率0.8执行，面积为原始图0.85倍
p.zoom_random(probability=0.3, percentage_area=0.85)

# 随机擦除与遮挡
p.random_erasing(probability=1, rectangle_area=0.5)

# 最终扩充的数据样本数
p.sample(5)

def get_agument(path):
    # 确定原始图像存储路径以及掩码文件存储路径
    p = Augmentor.Pipeline("F:\\data\\黑马\\aug")
    #p.ground_truth("F:\\data\\黑马\\aug")

    # 图像旋转： 按照概率0.8执行，最大左旋角度10，最大右旋角度10
    p.rotate(probability=0.8, max_left_rotation=5, max_right_rotation=5)

    # # 图像左右互换： 按照概率0.5执行
    # p.flip_left_right(probability=0.5)
    #
    # # 图像上下互换，按照概率0.5执行
    # p.flip_top_bottom(probability=0.5)

    p.rotate90(probability=0.5)

    # 图像放大缩小： 按照概率0.8执行，面积为原始图0.85倍
    p.zoom_random(probability=0.3, percentage_area=0.85)

    # 随机擦除与遮挡
    p.random_erasing(probability=1, rectangle_area=0.5)

    # 最终扩充的数据样本数
    p.sample(2000)
