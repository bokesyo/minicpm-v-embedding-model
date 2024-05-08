import json
import random
import os

random.seed(42)

# "2024-02-24-000651-model-base_with_padtoken-data-medi_jsonl_original-lr-5e-6-softm_temp-0.02-bsz4-ngpus8-nnodes1-inbatch-true/checkpoint-20000/"

# 文件路径
file1 = '/home/jeeves/medi-data-jsonl/train.jsonl'
# file2 = '/home/jeeves/dpo/ultrafeedback_train_prefs.medi.jsonl'
# file3 = '/home/jeeves/dpo/chatbot_arena.medi.jsonl'
file2 = '/home/jeeves/msmarco_hn_bge_en_15/train.jsonl'

mix_dir = '/home/jeeves/msmarco_hn_medi_merge'

os.makedirs(mix_dir, exist_ok=True)

def load_jsonl(file_path):
    """读取.jsonl文件并返回一个列表"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

# 合并三个文件的数据
# a, b = load_jsonl(file2), load_jsonl(file3)
a = load_jsonl(file2)

c = load_jsonl(file1)

# 删除所有 task_id 为 330 的字典
filtered_c = [data for data in c if data["task_id"] != 330 ]

combined_data = filtered_c + a

random.shuffle(combined_data)

# c_sampled = c[:10_0000]
# combined_data = c_sampled + a + b

# 保存合并后的数据到新的.jsonl文件
with open(os.path.join(mix_dir, 'train.jsonl'), 'w', encoding='utf-8') as file:
    for item in combined_data:
        file.write(json.dumps(item) + '\n')

print(f"Combined data saved to {mix_dir}")
