"""
@Time ： 2024/10/21 17:08
@Auth ： opprash
@File ：zhipu.py
@IDE ：PyCharm
"""
import jsonlines
from tqdm import tqdm
from zhipuai import ZhipuAI



client = ZhipuAI(api_key="e702db0da03c95815f9b3e15345044b9.Ve6lDsVcQY0wKE5o")
final_result = []
response = client.chat.completions.create(
        model="glm-4",
        messages=[
            {
                "role": "user",
                "content": 'forest的同义词有哪些？'
            }
        ])

print(response.choices[0].message.content)
