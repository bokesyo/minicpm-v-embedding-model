import jsonlines

input_file = "../../dataset/10w_why/data.jsonl"
output_file = "../../dataset/our-zh_raw/10w_why.jsonl"

with jsonlines.open(input_file) as reader:
    datas = list(reader)
new_datas = []
for item in datas:
    item = eval(item["clean_content"])
    new_datas.append({"query": item["question"], "pos": item["answer"]})
with jsonlines.open(output_file, "w") as f:
    f.write_all(new_datas)