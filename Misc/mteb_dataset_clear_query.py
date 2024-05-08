import json

# 输入和输出文件路径
input_file_path = '/home/jeeves/light_beir_eval/scidocs/corpus.raw.jsonl'
output_file_path = '/home/jeeves/light_beir_eval/scidocs/corpus.jsonl'

# 用于存储处理后的数据
processed_data = []

# 读取和处理输入文件
with open(input_file_path, 'r', encoding='utf-8') as input_file:
    for line in input_file:
        # 将每一行从JSON字符串转换为字典
        data = json.loads(line)
        
        # 如果存在'supporting_facts'字段，则删除它
        if 'metadata' in data:
            del data['metadata']
        
        # 将处理后的数据添加到列表中
        processed_data.append(data)

# 将处理后的数据写入到输出文件
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    for data in processed_data:
        # 将字典转换为JSON字符串，并写入到输出文件
        output_file.write(json.dumps(data) + '\n')

print(f"处理完成，输出文件已保存到：{output_file_path}")
