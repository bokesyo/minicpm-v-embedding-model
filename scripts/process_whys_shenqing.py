import jsonlines

datasets = ["whys","shenqing_news_20231109","shenqing_xingguang"]
for d in datasets:
    input_file = "../../dataset/{}/data.jsonl".format(d)
    output_file = "../../dataset/our-zh_raw/{}.jsonl".format(d)

    with jsonlines.open(input_file) as reader:
        datas = list(reader)
    new_datas = []
    for item in datas:
        new_datas.append({"query": item["title"], "pos": item["content"]})
    with jsonlines.open(output_file, "w") as f:
        f.write_all(new_datas)