from huggingface_hub import snapshot_download
# import os
import json
from datasets import load_dataset, load_from_disk

# current_dir = os.getcwd()
# print(current_dir)
# # 下载模型
snapshot_download(repo_id="Qwen/Qwen1.5-7B", local_dir="Qwen", local_dir_use_symlinks=False)

# 下载数据集
dataset = load_dataset("Amod/mental_health_counseling_conversations")
dataset.save_to_disk("./data") # 保存到该目录下
dataset = load_from_disk("./data") # 加载保存的数据集
print(dataset)
print(dataset['train'])
print(dataset['train'][1])

def write_jsonl(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            json_str = json.dumps(item,ensure_ascii=False)
            f.write(json_str + '\n')

def convert(dataList, output_filename):
    itemList = []
    for item in dataList:
        itemList.append({
            'from': item['Context'],
            'value': item['Response']
        })
    objList = [
        {
            "id": "identity_0",
            "conversations": itemList
        }
    ]
    write_jsonl(objList, output_filename)

convert(dataset['train'], 'train.jsonl')
# convert(dataset['dev'], 'dev.jsonl')
# convert(dataset['test'], 'test.jsonl')