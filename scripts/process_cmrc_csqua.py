import json
import jsonlines

input_file = "../../dataset/m3e/train-zen-v1.0.json"
output_file = "../../dataset/our-zh_raw/csquad.jsonl"

with open(input_file, "r") as f:
    data = json.load(f)
data = data["data"]
new_data = []
for item in data:
    for par in item["paragraphs"]:
        for qa in par["qas"]:
            new_data.append({"query": qa["question"], "pos": par["context"]})
with jsonlines.open(output_file, "w") as f:
    f.write_all(new_data)