import jsonlines

input_file = "../../dataset/economic_information_daily_clean/data.jsonl"
output_file = "../../dataset/our-zh_raw/economic_information_daily_clean.jsonl"

with jsonlines.open(input_file) as reader:
    datas = list(reader)
new_datas = []
for item in datas:
    item = item["clean_content"]
    new_datas.append({"query": item["title"], "pos": item["text"]})
with jsonlines.open(output_file, "w") as f:
    f.write_all(new_datas)