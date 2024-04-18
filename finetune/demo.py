# from huggingface_hub import snapshot_download
# import os
from datasets import load_dataset, load_from_disk

# current_dir = os.getcwd()
# print(current_dir)
# # 下载模型
# snapshot_download(repo_id="THUDM/chatglm3-6b", local_dir="chatglm3-6b", local_dir_use_symlinks=False)

# 下载数据集
dataset = load_dataset("Amod/mental_health_counseling_conversations")
dataset.save_to_disk("./data") # 保存到该目录下
dataset = load_from_disk("./data") # 加载保存的数据集
print(len(dataset['train']))
print(dataset['train'])
print(dataset['train'][1])
# with open('./dataset/data.json', 'w', encoding='utf-8') as fp:
