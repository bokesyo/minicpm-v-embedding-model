import json

# 文件路径
file1 = '/home/jeeves/medi-data-jsonl/medi-data.jsonl'
file2 = '/home/jeeves/dpo/ultrafeedback_train_prefs.medi.jsonl'
file3 = '/home/jeeves/dpo/chatbot_arena.medi.jsonl'
mix_dir = '/home/jeeves/medi-dpomix-chatbotarena-ultrafeedback-0224b/train.jsonl'

def load_jsonl(file_path):
    """读取.jsonl文件并返回一个列表"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

# 合并三个文件的数据
combined_data = load_jsonl(file1) + load_jsonl(file2) + load_jsonl(file3)

# 保存合并后的数据到新的.jsonl文件
with open(mix_dir, 'w', encoding='utf-8') as file:
    for item in combined_data:
        file.write(json.dumps(item) + '\n')

print(f"Combined data saved to {mix_dir}")
