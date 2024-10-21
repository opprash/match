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
