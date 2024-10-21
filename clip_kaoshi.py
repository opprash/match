"""
@Time ： 2024/10/18 11:28
@Auth ： opprash
@File ：clip_kaoshi.py
@IDE ：PyCharm
"""
import torch
import clip
from PIL import Image
import os

image_extensions = ['.jpg', '.png', '.bmp', '.tif', '.tiff', '.jpeg']
device = "cuda"


# 文本与图像相似性输出
def img_pair_txt(label_list, input_image):
    image = preprocess(input_image).unsqueeze(0).to(device)
    text = clip.tokenize(label_list).to(device)
    with torch.no_grad():
        # 提取图像特征
        image_features = model.encode_image(image)
        # 提取文本语言特征
        text_features = model.encode_text(text)
        # 计算相关性矩阵
        Logits_per_image, logits_per_text = model(image, text)
        probs = Logits_per_image.softmax(dim=-1).cpu().numpy()
    return probs


# 输出相似性最高的文本下标
def output_idx(probs):
    for idx in range(len(probs[0])):
        ida = max(probs[0])
        if probs[0][idx] == ida:
            return idx


# 读取文件夹中所有的图像路径并放到image_paths列表中
def get_image_paths(folder_path):
    image_paths = []
    for root, directories, files in os.walk(folder_path):
        for file_name in files:
            file_extension = os.path.splitext(file_name)[1].lower()
            if file_extension in image_extensions:
                image_path = os.path.join(root, file_name)
                image_paths.append(image_path)
    return image_paths


if __name__ == '__main__':
    # 文件夹路径
    folder_path = './new_final'
    # clip模型加载
    model, preprocess = clip.load("ViT-B/32", device=device)
    print("clip模型加载成功")
    # label_list = ["indoor", "apartment", "courtyard", "road", "restaurant", "corridor", "grocery", "farm", "farmland", "office",
    #             "warehouse", "intersection", "staircase", "street", "river_course", "refuse_room", "kitchen"]
    label_list = ['plain', 'shrew', 'road', 'skunk', 'raccoon', 'seal', 'sea', 'poppy', 'rose', 'plate', 'skyscraper',
                  'snail', 'possum', 'rabbit', 'streetcar', 'porcupine', 'squirrel', 'snake', 'spider', 'shark', 'ray',
                  'rocket']

    if os.path.isdir(folder_path):
        # 打开文件以写入模式('W')，如果文件不存在则创建它:如果存在，则会覆盖原有内容
        with open('predict.txt', 'w', encoding='utf-8') as file:
            image_paths = get_image_paths(folder_path)
            for image_path in image_paths:
                # print (image_path )
                input_image_new = Image.open(image_path)
                probs_new = img_pair_txt(label_list, input_image_new)
                idx = output_idx(probs_new)
                label = label_list[idx]
                print(image_path, ",", label.split(",")[0])
                # 不能修改输出格式，输出为图片路径，类别(与预给类别致)
                pic_label = image_path + "," + label.split(",")[0] + "\n"
                file.write(pic_label)
    else:
        print("文件夹路径错误")



import os

fr1 =open('./record_new.txt','r')
fr2 =open('./predict.txt','r')


datas = {}

for each in fr1:
    each=each.strip()
    m=each.split(',')
    file_name = os.path.basename(m[0])
    datas[file_name]=[m[1]]

for each in fr2:
    each = each.strip()
    m = each.split(',')
    file_name = os.path.basename(m[0])
    datas[file_name].append(m[1])

t=0
r=0
for k,v in datas.items():
    t+=1
    if v[0]==v[1]:
        r+=1

print(r/t)


