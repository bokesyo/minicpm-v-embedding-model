

import json

json_path = "/home/jeeves/medi-data/medi-data.json"

f = open(json_path, 'r')
data = json.load(f)

# 将列表转换为JSONL并保存到文件
def list_of_dicts_to_jsonl(data, output_filename):
    with open(output_filename, 'w') as file:
        for item in data:
            json_str = json.dumps(item)
            file.write(json_str + '\n')

# 调用函数
output_filename = '/home/jeeves/medi-data-jsonl/medi-data.jsonl'
list_of_dicts_to_jsonl(data, output_filename)
