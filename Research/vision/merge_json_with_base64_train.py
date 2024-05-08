import json
import copy

# 读取JSON文件
json_path = '/home/jeeves/train_data_w_o_image.jsonl'
json_path_with_base64 = '/home/jeeves/train_data_w_image.json'
base64_ds_path = '/home/jeeves/visual_embedding_1_dataset_jsonl_merged_latest/data.jsonl'

data = {}
with open(json_path, 'r') as f:
    for line in f:
        jsonified = json.loads(line)
        data[jsonified['id']] = jsonified


# out_json = {}
# 读取JSONL文件并处理
cnt = 0
with open(base64_ds_path, 'r') as f:
    for line in f:
        cnt += 1
        if cnt % 1000 == 0:
            print(cnt)
        jsonl_item = json.loads(line.strip())
        if jsonl_item['id'] in data.keys():
            data[jsonl_item['id']]["pos"][0]["image"] = jsonl_item["base64"]
            # tmp_json["base64"] = jsonl_item["base64"]
            # out_json[jsonl_item['id']] = tmp_json

# 保存更新后的JSON文件
# with open(json_path_with_base64, 'w') as f:
#     json.dump(out_json, f)
out_list = list(data.values())
print(len(out_list))
cnt = 0
with open(json_path_with_base64, "w") as f:
    for item in out_list:
        cnt += 1
        if cnt % 1000 == 0:
            print(cnt)
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
    # print("done")
print("Updated JSON saved successfully!")
