import json
import copy

# 读取JSON文件
json_path = '/home/jeeves/reshaped_output.json'
json_path_with_base64 = '/home/jeeves/reshaped_output_with_base64.json'
base64_ds_path = '/home/jeeves/visual_embedding_2_long_visual_dataset_jsonl_merged/data.jsonl'


with open(json_path, 'r') as f:
    original_json = json.load(f)

out_json = {}
# 读取JSONL文件并处理
cnt = 0
with open(base64_ds_path, 'r') as f:
    for line in f:
        cnt += 1
        if cnt % 1000 == 0:
            print(cnt)
        jsonl_item = json.loads(line.strip())
        if jsonl_item['id'] in original_json.keys():
            # 添加base64值到原始JSON中
            tmp_json = {}
            tmp_json["annotation"] = original_json[jsonl_item['id']]
            tmp_json["base64"] = jsonl_item["base64"]
            out_json[jsonl_item['id']] = tmp_json

# 保存更新后的JSON文件
with open(json_path_with_base64, 'w') as f:
    json.dump(out_json, f)

print("Updated JSON saved successfully!")
