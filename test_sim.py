"""
@Time ： 2024/10/19 15:49
@Auth ： opprash
@File ：test_sim.py
@IDE ：PyCharm
"""
import torch
import torch.nn.functional as F

# 假设你有两个Tensor
tensor_a = torch.tensor([[1.0, 2.0, 3.0]])
tensor_b = torch.tensor([[4.0, 5.0, 6.0]])

# 确保它们都是浮点数类型的Tensor
# tensor_a = tensor_a.float()
# tensor_b = tensor_b.float()

# 使用unsqueeze增加一个维度，使形状变为 (1, n)，以便可以批量处理
#tensor_a_unsq = tensor_a.unsqueeze(0)
#tensor_b_unsq = tensor_b.unsqueeze(0)

# 计算余弦相似度
#cosine_sim = F.cosine_similarity(tensor_a_unsq, tensor_b_unsq)

# 输出结果
#print("余弦相似度为:", cosine_sim.item())

# 如果你想直接在指定维度上计算，可以这样做：
cosine_sim_direct = F.cosine_similarity(tensor_a, tensor_b)
x= float(cosine_sim_direct)
print(type(cosine_sim_direct))
print(type(x))
print(x)
print("直接计算余弦相似度为:", cosine_sim_direct.item())


def get_most_sim(query,shots):
    most_key=''
    most_value =0
    for k,v in shots.items():
        temp_value=0
        for i in v:
            cosine_sim_direct = F.cosine_similarity(query, i)
            temp_value+=float(cosine_sim_direct)

        if temp_value>most_value:
            most_key=k
            most_value=temp_value

    return most_key,most_value




def find_most_frequent_element(lst):
    # 使用max函数配合lambda表达式来找出出现次数最多的元素
    return max(set(lst), key=lst.count)

a = ['a','a','b','b','b','c']
print(find_most_frequent_element(a))