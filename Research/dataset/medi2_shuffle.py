import json
import random

# 加载JSONL文件
def load_jsonl(input_path):
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# 保存为JSONL文件
def save_jsonl(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

# 主函数
def main():
    input_path = 'input.jsonl'  # 原始文件路径
    output_path = 'shuffled_output.jsonl'  # 输出文件路径
    
    # 加载数据
    data = load_jsonl(input_path)
    
    # 随机打乱数据
    random.shuffle(data)
    
    # 保存数据
    save_jsonl(data, output_path)
    
    print(f'Data shuffled and saved to {output_path}')

# 执行主函数
if __name__ == "__main__":
    main()
