"""
@Time ： 2024/10/20 12:57
@Auth ： opprash
@File ：tss.py
@IDE ：PyCharm
"""
import os
from Augmentor import Pipeline
from tqdm import tqdm

def augment_images(input_directory, output_directory, num_samples):
    # 创建一个Pipeline对象，并指定输入目录
    p = Pipeline(input_directory)

    # 指定输出目录
    p.output_directory = output_directory

    # 可以添加多种不同的图像增强操作
    # 例如随机旋转
    p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)

    # 随机缩放
    p.scale(probability=0.5, min_factor=0.8, max_factor=1.2)



    # 随机翻转
    p.flip_left_right(probability=0.5)



    # 图像放大缩小： 按照概率0.8执行，面积为原始图0.85倍
    p.zoom_random(probability=0.3, percentage_area=0.85)

    # 随机擦除与遮挡
    p.random_erasing(probability=1, rectangle_area=0.5)


    # 生成指定数量的样本
    p.sample(num_samples)


image_extensions = ['.jpg', '.png', '.bmp', '.tif', '.tiff', '.jpeg']
def get_image_paths(folder_path):
    image_paths = []
    for root, directories, files in os.walk(folder_path):
        for file_name in files:
            file_extension = os.path.splitext(file_name)[1].lower()
            if file_extension in image_extensions:
                image_path = os.path.join(root, file_name)
                image_paths.append(image_path)
    return image_paths


import shutil
import os


def copy_image(source_path, destination_dir):
    # 检查目标目录是否存在，不存在则创建
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # 获取图片文件名
    image_filename = os.path.basename(source_path)

    # 构建目标路径
    destination_path = os.path.join(destination_dir, image_filename)

    # 复制图片
    shutil.copy2(source_path, destination_path)
    return destination_path


if __name__ == "__main__":
    origin = './val_final'
    input_dir = "./temp"
    output_dir = "./temp"
    image_paths = get_image_paths(origin)
    for each in tqdm(image_paths):
        print(each)
        destination_path = copy_image(each,input_dir)
        num_samples_per_image = 5

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        augment_images(input_dir, output_dir, num_samples_per_image)
        os.remove(destination_path)


{'test_test_1755.png': (['skunk', 'skunk', 'poppy', 'snail'], 'orchid'), 'test_test_4063.png': (['poppy', 'skunk'], 'orchid'), 'test_test_6919.png': (['plate', 'snail'], 'orchid'), 'test_test_7942.png': (['seal', 'porcupine', 'porcupine'], 'orchid'), 'test_test_519.png': (['poppy', 'snail', 'rose'], 'orchid'), 'test_test_2749.png': (['snail', 'road', 'road'], 'road'), 'test_test_4651.png': (['snake', 'snake'], 'road'), 'test_test_5136.png': (['seal', 'road'], 'road'), 'test_test_2289.png': (['sea', 'sea', 'sea'], 'road'), 'test_test_3571.png': (['snake', 'ray', 'road'], 'road'), 'test_test_5187.png': (['skyscraper', 'skyscraper', 'rocket'], 'rocket'), 'test_test_1685.png': (['skyscraper', 'rabbit', 'shrew'], 'rocket'), 'test_test_975.png': (['seal', 'rocket', 'seal', 'sea', 'skyscraper'], 'rocket'), 'test_test_8287.png': (['shrew', 'seal', 'rocket', 'snake'], 'rocket'), 'test_test_3743.png': (['sea'], 'rocket'), 'test_test_5801.png': (['squirrel'], 'squirrel'), 'test_test_9748.png': (['snake', 'snake', 'snake', 'snake', 'snake'], 'squirrel'), 'test_test_1883.png': (['snake', 'snake', 'squirrel', 'snake'], 'squirrel'), 'test_test_7840.png': (['squirrel', 'squirrel', 'snake', 'snake', 'snake'], 'squirrel'), 'test_test_5526.png': (['snail', 'seal', 'seal'], 'squirrel'), 'test_test_4119.png': (['rabbit', 'squirrel', 'snake'], 'rabbit'), 'test_test_5746.png': (['shrew', 'ray', 'raccoon', 'snake'], 'rabbit'), 'test_test_1632.png': (['ray'], 'rabbit'), 'test_test_7183.png': (['snake', 'raccoon', 'possum'], 'rabbit'), 'test_test_7711.png': (['snake', 'rabbit', 'snail'], 'rabbit'), 'test_test_5369.png': (['snake', 'shrew', 'snake'], 'snake'), 'test_test_5955.png': (['snake', 'snake'], 'snake'), 'test_test_5689.png': (['snake', 'snake'], 'snake'), 'test_test_699.png': (['snake', 'snake', 'snake', 'snake'], 'snake'), 'test_test_5426.png': (['shrew', 'seal', 'snake'], 'snake'), 'test_test_5922.png': (['snake', 'seal'], 'ray'), 'test_test_7696.png': (['snake', 'snake'], 'ray'), 'test_test_89.png': (['ray'], 'ray'), 'test_test_758.png': (['snake', 'snail', 'snail', 'snail'], 'ray'), 'test_test_5948.png': (['snake', 'snail'], 'ray'), 'test_test_8076.png': (['porcupine'], 'possum'), 'test_test_6292.png': (['snake', 'raccoon'], 'possum'), 'test_test_6648.png': (['snake', 'possum', 'possum'], 'possum'), 'test_test_5080.png': (['snake', 'snake', 'snake'], 'possum'), 'test_test_5226.png': (['snake'], 'possum'), 'test_test_8021.png': (['snake', 'seal', 'seal', 'seal'], 'plate'), 'test_test_1903.png': (['snail', 'snail', 'seal'], 'plate'), 'test_test_2916.png': (['snail'], 'plate'), 'test_test_7341.png': (['seal', 'seal', 'seal', 'snail'], 'plate'), 'test_test_9444.png': (['plate', 'seal', 'seal', 'seal', 'plate'], 'plate'), 'test_test_150.png': (['plate'], 'pickup_truck'), 'test_test_2321.png': (['snake', 'plate'], 'pickup_truck'), 'test_test_4588.png': (['snake', 'snail', 'snail'], 'pickup_truck'), 'test_test_2882.png': (['rocket', 'rocket'], 'pickup_truck'), 'test_test_5738.png': (['ray', 'road', 'plate'], 'pickup_truck'), 'test_test_3206.png': (['snail', 'poppy', 'snake'], 'poppy'), 'test_test_7951.png': (['poppy', 'poppy', 'poppy', 'poppy'], 'poppy'), 'test_test_5332.png': (['snail', 'poppy', 'snail'], 'poppy'), 'test_test_4045.png': (['sea', 'skyscraper', 'snail', 'sea'], 'poppy'), 'test_test_3307.png': (['rose', 'rose', 'rose', 'rose'], 'poppy'), 'test_test_3498.png': (['skunk', 'skunk', 'snake', 'porcupine'], 'shrew'), 'test_test_4177.png': (['snake', 'porcupine'], 'shrew'), 'test_test_2215.png': (['snake', 'shrew'], 'shrew'), 'test_test_1555.png': (['snake', 'seal', 'squirrel'], 'shrew'), 'test_test_9949.png': (['snail', 'spider', 'snake', 'snake', 'seal'], 'shrew'), 'test_test_9415.png': (['snake', 'snail', 'ray', 'snail', 'ray'], 'porcupine'), 'test_test_2276.png': (['raccoon', 'porcupine', 'seal'], 'porcupine'), 'test_test_4850.png': (['snake', 'snail'], 'porcupine'), 'test_test_3880.png': (['snail', 'shrew', 'raccoon', 'snail', 'possum'], 'porcupine'), 'test_test_7854.png': (['snake', 'porcupine', 'snake', 'squirrel', 'spider'], 'porcupine'), 'test_test_1601.png': (['snake'], 'pine_tree'), 'test_test_9583.png': (['porcupine', 'snail', 'road', 'rocket', 'porcupine'], 'pine_tree'), 'test_test_1395.png': (['snake', 'porcupine'], 'pine_tree'), 'test_test_5669.png': (['snake', 'snail', 'snake', 'road'], 'pine_tree'), 'test_test_9035.png': (['road', 'road'], 'pine_tree'), 'test_test_9798.png': (['squirrel', 'snail', 'skunk', 'snake', 'snake'], 'snail'), 'test_test_1112.png': (['snake', 'rabbit'], 'snail'), 'test_test_3711.png': (['snake'], 'snail'), 'test_test_8517.png': (['snail'], 'snail'), 'test_test_8320.png': (['snail', 'snail'], 'snail'), 'test_test_635.png': (['road', 'snake', 'snake'], 'plain'), 'test_test_4130.png': (['snake', 'snake', 'rocket'], 'plain'), 'test_test_9271.png': (['skyscraper', 'snake', 'ray', 'skyscraper', 'sea'], 'plain'), 'test_test_931.png': (['skyscraper', 'road', 'snake', 'road', 'sea'], 'plain'), 'test_test_9219.png': (['sea', 'plain', 'snake', 'snake', 'seal'], 'plain'), 'test_test_7914.png': (['seal', 'seal', 'seal'], 'skunk'), 'test_test_9341.png': (['snake', 'snake', 'skunk', 'porcupine', 'skunk'], 'skunk'), 'test_test_3889.png': (['snake', 'porcupine'], 'skunk'), 'test_test_4424.png': (['snake'], 'skunk'), 'test_test_7278.png': (['skunk', 'skunk', 'skunk', 'snail'], 'skunk'), 'test_test_9439.png': (['snake', 'skyscraper', 'snake', 'snake', 'snake'], 'palm_tree'), 'test_test_6319.png': (['snail', 'road', 'porcupine'], 'palm_tree'), 'test_test_5358.png': (['snake', 'ray', 'snail'], 'palm_tree'), 'test_test_3440.png': (['sea'], 'palm_tree'), 'test_test_5758.png': (['porcupine', 'sea'], 'palm_tree'), 'test_test_9751.png': (['snake', 'snake', 'seal', 'ray', 'snake'], 'otter'), 'test_test_5933.png': (['seal', 'seal', 'snail', 'ray'], 'otter'), 'test_test_8296.png': (['ray', 'ray'], 'otter'), 'test_test_6642.png': (['snake', 'skunk'], 'otter'), 'test_test_9223.png': (['snail', 'snail', 'porcupine', 'porcupine', 'porcupine'], 'otter'), 'test_test_4603.png': (['snail', 'snail', 'seal'], 'seal'), 'test_test_646.png': (['road'], 'seal'), 'test_test_7169.png': (['seal', 'porcupine', 'snail'], 'seal'), 'test_test_7355.png': (['ray', 'shark'], 'seal'), 'test_test_553.png': (['seal', 'shrew', 'snail', 'poppy'], 'seal'), 'test_test_6884.png': (['streetcar', 'snail'], 'streetcar'), 'test_test_6012.png': (['snail', 'streetcar', 'squirrel'], 'streetcar'), 'test_test_946.png': (['snail', 'road', 'snail', 'snail', 'snake'], 'streetcar'), 'test_test_2496.png': (['road'], 'streetcar'), 'test_test_2955.png': (['streetcar', 'ray', 'streetcar', 'snake'], 'streetcar'), 'test_test_3660.png': (['snake', 'skunk'], 'raccoon'), 'test_test_2265.png': (['seal', 'raccoon'], 'raccoon'), 'test_test_3772.png': (['snake'], 'raccoon'), 'test_test_845.png': (['porcupine', 'snake'], 'raccoon'), 'test_test_6666.png': (['raccoon', 'porcupine'], 'raccoon'), 'test_test_9620.png': (['sea', 'sea', 'snake', 'skyscraper', 'snake'], 'sea'), 'test_test_2135.png': (['sea', 'seal', 'sea'], 'sea'), 'test_test_5194.png': (['seal', 'sea', 'snake'], 'sea'), 'test_test_7100.png': (['snail', 'sea', 'sea', 'sea'], 'sea'), 'test_test_5153.png': (['sea', 'sea'], 'sea'), 'test_test_1412.png': (['seal', 'skyscraper'], 'skyscraper'), 'test_test_9183.png': (['snake', 'skyscraper', 'snake', 'skyscraper', 'ray', 'skyscraper', 'seal', 'snake', 'skyscraper', 'skyscraper', 'skyscraper', 'seal', 'ray', 'skyscraper', 'skyscraper', 'squirrel', 'seal', 'ray', 'seal', 'skyscraper', 'seal', 'skyscraper', 'ray', 'snake', 'squirrel', 'seal', 'skyscraper', 'snake', 'skyscraper', 'seal', 'ray', 'rabbit', 'skyscraper', 'skyscraper', 'plain', 'ray', 'skyscraper', 'skyscraper', 'seal', 'seal', 'snake', 'snake', 'ray', 'skyscraper', 'plate', 'skyscraper', 'skyscraper', 'snail', 'skyscraper', 'skyscraper', 'snake', 'skyscraper', 'snake', 'skyscraper', 'seal', 'skyscraper', 'seal', 'snake', 'seal', 'skyscraper', 'skyscraper', 'seal', 'skyscraper', 'seal', 'skyscraper', 'ray', 'skyscraper', 'seal', 'skyscraper', 'sea', 'skyscraper', 'skyscraper', 'ray', 'rabbit', 'ray', 'seal', 'skyscraper', 'seal', 'rabbit', 'skyscraper', 'skyscraper', 'skyscraper', 'seal', 'skyscraper', 'skyscraper', 'skyscraper', 'seal', 'skyscraper', 'ray', 'snake', 'seal', 'skyscraper', 'ray', 'snake', 'seal', 'skyscraper', 'ray', 'skyscraper', 'skyscraper', 'ray', 'shrew', 'skyscraper', 'skyscraper', 'seal', 'seal', 'skyscraper', 'seal', 'seal', 'seal', 'skyscraper', 'skyscraper', 'seal', 'squirrel', 'skyscraper', 'seal', 'skyscraper', 'skyscraper', 'ray', 'seal', 'ray', 'skyscraper', 'skyscraper', 'skyscraper', 'snake', 'snake', 'skyscraper', 'skyscraper', 'seal', 'seal', 'skyscraper', 'plain', 'skyscraper', 'skyscraper', 'skyscraper', 'skyscraper', 'seal', 'snake', 'seal', 'skyscraper', 'skyscraper', 'skyscraper', 'skyscraper', 'seal', 'skyscraper', 'seal', 'snake', 'skyscraper', 'snail', 'snake', 'seal', 'seal', 'skyscraper', 'skyscraper', 'skyscraper', 'seal', 'seal', 'seal', 'snake', 'seal', 'skyscraper', 'seal', 'skyscraper', 'ray', 'skyscraper', 'seal', 'seal', 'seal', 'skyscraper', 'ray', 'seal', 'ray', 'skyscraper', 'skyscraper', 'seal', 'ray', 'skyscraper', 'ray', 'seal', 'seal', 'snail', 'seal', 'seal', 'skyscraper', 'skyscraper', 'skyscraper', 'seal', 'skyscraper', 'snake', 'skyscraper', 'ray', 'sea', 'skyscraper', 'skyscraper', 'skyscraper', 'snake', 'snake', 'snake', 'ray', 'seal', 'seal', 'snake', 'snake', 'seal', 'snake', 'skyscraper', 'seal', 'seal', 'snake', 'skyscraper', 'ray', 'seal', 'ray', 'shrew', 'skyscraper', 'seal', 'ray', 'skyscraper', 'seal', 'skyscraper', 'snail', 'shrew', 'seal', 'skyscraper', 'snake', 'seal', 'seal', 'ray', 'snake', 'skyscraper', 'skyscraper', 'seal', 'skyscraper', 'seal', 'skyscraper', 'snake', 'seal', 'skyscraper', 'seal', 'seal', 'rabbit', 'skyscraper', 'skyscraper', 'skyscraper', 'skyscraper', 'skyscraper', 'skyscraper', 'skyscraper', 'snake', 'skyscraper', 'skyscraper', 'ray', 'snake', 'seal', 'ray', 'skyscraper', 'seal', 'skyscraper', 'seal', 'snail', 'skyscraper', 'skyscraper', 'snake', 'snake', 'seal', 'ray', 'seal', 'seal', 'skyscraper', 'seal', 'skyscraper', 'ray', 'sea', 'skyscraper', 'seal', 'seal', 'skyscraper', 'seal', 'seal', 'skyscraper', 'skyscraper', 'skyscraper', 'seal', 'seal', 'seal', 'seal', 'skyscraper', 'skyscraper', 'skyscraper', 'skyscraper', 'seal', 'ray', 'skyscraper', 'ray', 'skyscraper', 'skyscraper', 'skyscraper', 'skyscraper', 'ray', 'snake'], 'skyscraper'), 'test_test_3949.png': (['skyscraper', 'skyscraper', 'skyscraper', 'snake'], 'skyscraper'), 'test_test_5355.png': (['skyscraper', 'skyscraper', 'skyscraper'], 'skyscraper'), 'test_test_9819.png': (['skyscraper', 'seal', 'snake', 'skyscraper', 'seal'], 'skyscraper'), 'test_test_3574.png': (['snail', 'snake', 'snail'], 'spider'), 'test_test_3030.png': (['snake', 'snail', 'spider'], 'spider'), 'test_test_7497.png': (['sea'], 'spider'), 'test_test_3212.png': (['spider', 'snake', 'spider'], 'spider'), 'test_test_9188.png': (['snake', 'snake', 'snake', 'snake', 'snake'], 'spider'), 'test_test_9435.png': (['plate', 'snail', 'rose', 'snail', 'snail'], 'rose'), 'test_test_4557.png': (['snail', 'snake', 'rose'], 'rose'), 'test_test_8315.png': (['poppy'], 'rose'), 'test_test_3953.png': (['snail', 'rose', 'snail'], 'rose'), 'test_test_20.png': (['rose'], 'rose'), 'test_test_8061.png': (['snake', 'sea'], 'orange'), 'test_test_2752.png': (['snake', 'snail'], 'orange'), 'test_test_6641.png': ([], 'orange'), 'test_test_3059.png': (['snail', 'snail', 'snail', 'snail', 'snail'], 'orange'), 'test_test_8552.png': (['snail', 'snake', 'snail'], 'orange'), 'test_test_8507.png': (['skunk', 'snake'], 'shark'), 'test_test_474.png': (['seal', 'seal', 'seal'], 'shark'), 'test_test_3890.png': (['skunk', 'seal', 'rabbit'], 'shark'), 'test_test_9623.png': (['sea', 'sea', 'seal', 'seal', 'shark'], 'shark'), 'test_test_9913.png': (['shark', 'seal', 'shark', 'ray', 'snake'], 'shark'), 'test_test_6065.png': (['plain', 'ray', 'snail'], 'pear'), 'test_test_541.png': (['rose', 'skunk', 'porcupine'], 'pear'), 'test_test_828.png': (['snake', 'snail'], 'pear'), 'test_test_3958.png': (['snail', 'plain', 'snail', 'snail'], 'pear'), 'test_test_6536.png': (['snake', 'snake', 'snail'], 'pear')}
Traceback (most recent call last):