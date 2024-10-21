"""
@Time ： 2024/10/19 13:55
@Auth ： opprash
@File ：test_cifar.py
@IDE ：PyCharm
"""
"""
@Time ： 2024/10/18 11:28
@Auth ： opprash
@File ：clip_kaoshi.py
@IDE ：PyCharm
"""
import torch
import clip
import torch.nn.functional as F
from tqdm import tqdm
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
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        # 计算相关性矩阵
        Logits_per_image, logits_per_text = model(image, text)
        probs = Logits_per_image.softmax(dim=-1).cpu().numpy()
    return image_features,probs


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



def save_few_shot():

    all_feature_dict={}
    fr1 = open('./val_record.txt', 'r')

    datas = {}
    for each in fr1:
        each = each.strip()
        m = each.split(',')
        file_name = os.path.basename(m[0])
        datas[file_name] = m[1]
    val_images = get_image_paths('./val_final')
    shot_image_features = []
    shot_text_features = []
    for image_path in val_images:
        val_label = ''
        for k,v in datas.items():
            if k in image_path:
                val_label = v
                break
        input_image_new = Image.open(image_path)
        image = preprocess(input_image_new).unsqueeze(0).to(device)
        text = clip.tokenize(label_list).to(device)
        with torch.no_grad():
            # 提取图像特征
            image_features = model.encode_image(image)
            # 提取文本语言特征
            text_features = model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            #print(image_features.shape)
            #print(text_features.shape)
            if val_label in all_feature_dict.keys():
                all_feature_dict[val_label].append(image_features)
            else:
                all_feature_dict[val_label]=[image_features]
    # for i,j in all_feature_dict.items():
    #     print(i)
    #     print(len(j))
            #text_features /= text_features.norm(dim=-1, keepdim=True)
            #shot_text_features.append(text_features.tolist())
            #shot_image_features.append(image_features.tolist())

    # features = torch.Tensor(shot_text_features)
    # texts = torch.Tensor(shot_image_features)
    # #shot_image_features,shot_text_features =torch.cat(shot_image_features,dim=0), torch.cat(shot_text_features,dim=0)
    #
    # print(features.shape)
    # print(texts.shape)
    return all_feature_dict


def get_most_sim(query, shots):
    most_key = ''
    most_value = 0
    for k, v in shots.items():
        temp_value = 0
        for i in v:
            cosine_sim_direct = F.cosine_similarity(query, i)
            temp_value += float(cosine_sim_direct)

        if temp_value > most_value:
            most_key = k
            most_value = temp_value

    return most_key, most_value

if __name__ == '__main__':


    #文件夹路径
    folder_path = './test_final'
    # clip模型加载
    model, preprocess = clip.load("ViT-B/32", device=device)
    print("clip模型加载成功")
    # label_list = ["indoor", "apartment", "courtyard", "road", "restaurant", "corridor", "grocery", "farm", "farmland", "office",
    #             "warehouse", "intersection", "staircase", "street", "river_course", "refuse_room", "kitchen"]
    label_list = ['orchid', 'road', 'rocket', 'squirrel', 'rabbit', 'snake', 'ray', 'possum', 'plate', 'pickup_truck', 'poppy', 'shrew', 'porcupine', 'pine_tree', 'snail', 'plain', 'skunk', 'palm_tree', 'otter', 'seal', 'streetcar', 'raccoon', 'sea', 'skyscraper', 'spider', 'rose', 'orange', 'shark', 'pear']
    shot_dicts = save_few_shot()
    #label_list = ['plain', 'shrew', 'road', 'skunk', 'raccoon', 'seal', 'sea', 'poppy', 'rose', 'plate', 'skyscraper',
    #              'snail', 'possum', 'rabbit', 'streetcar', 'porcupine', 'squirrel', 'snake', 'spider', 'shark', 'ray',
    #              'rocket']

    if os.path.isdir(folder_path):
        # 打开文件以写入模式('W')，如果文件不存在则创建它:如果存在，则会覆盖原有内容
        with open('test_predict.txt', 'w', encoding='utf-8') as file:
            image_paths = get_image_paths(folder_path)
            for image_path in image_paths:
                # print (image_path )
                input_image_new = Image.open(image_path)
                image_features, probs_new = img_pair_txt(label_list, input_image_new)
                shot_key,shot_value = get_most_sim(image_features,shot_dicts)

                print(shot_key)
                idx = output_idx(probs_new)
                label = label_list[idx]
                print(image_path, ",", shot_key)
                # 不能修改输出格式，输出为图片路径，类别(与预给类别致)
                pic_label = image_path + "," + shot_key + "\n"
                # print(image_path, ",", label.split(",")[0])
                # # 不能修改输出格式，输出为图片路径，类别(与预给类别致)
                # pic_label = image_path + "," + label.split(",")[0] + "\n"
                file.write(pic_label)
    else:
        print("文件夹路径错误")
