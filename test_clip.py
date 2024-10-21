"""
@Time ： 2024/10/18 11:24
@Auth ： opprash
@File ：test_clip.py
@IDE ：PyCharm
"""
import torch
import clip
from PIL import Image

device ="cuda" if torch.cuda.is_available()else "cpu"
model,preprocess= clip.load("ViT-B/32",device=device)#首次使用会默认下载clip模型
image = preprocess(Image.open("/data/wangyuxuan/gen model/CLIP/CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram","a dog","a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:",probs)

