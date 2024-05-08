import json
import os

# 指定要读取的 .jsonl 文件路径
# input_file_path = '/home/jeeves/multihop_wiki_zh/data.jsonl'
# output_file_path = '/home/jeeves/multihop_wiki_zh/test2'

input_file_path = '/home/jeeves/book3.1000k/data.jsonl'
output_file_path = '/home/jeeves/book3.1000k/test2'

os.makedirs(output_file_path, exist_ok=True)


# 读取文件
with open(input_file_path, 'r', encoding='utf-8') as file:
    # 初始化一个列表来存储读取的 JSON 对象
    json_objects = []
    
    # 逐行读取
    for i in range(100):
        line = file.readline()
        if not line:  # 如果没有更多行，停止读取
            break
        json_object = json.loads(line)
        json_objects.append(json_object)

cnt = 0
for obj in json_objects:
    cnt += 1
    # 将读取的 JSON 对象保存到一个新文件
    save_path_i = os.path.join(output_file_path, f"{cnt}.json")
    with open(save_path_i, 'w', encoding='utf-8') as outfile:
        json.dump(obj, outfile, ensure_ascii=False, indent=4)
