import os
import json
# 读取一个文件夹中所有的jsonl文件到一个list
import re
import random

random.seed(42)

def parse_text(input_string):
    # 正则表达式匹配以```json开始，然后捕获[TEXT]部分，直到字符串以```结束
    pattern = r"```json\n(.*?)\n```"
    match = re.search(pattern, input_string, re.DOTALL)
    
    if not match:
        return None
    
    # 返回捕获的[TEXT]部分
    return match.group(1)


if __name__ == "__main__":
    path = "/Users/bokesyo/ys_data/vision_inverse_query_compare/inverse-query-qwenvl-2024-04-12-010826"
    data = []
    jsonl_list = os.listdir(path)
    for jsonl in jsonl_list:
        if jsonl.endswith('.jsonl'):
            with open(f"{path}/{jsonl}", "r") as f:
                lines = f.readlines()
                for line in lines:
                    line_dict = json.loads(line)
                    data.append(line_dict)
    print(len(data))
    
    parsed_data = []
    for idx in range(len(data)):
        if idx % 1000 == 0:
            print(idx)
        
        parsed_text = parse_text(data[idx]['inversed_queries'])
        data[idx]["parsed_text"] = parsed_text
        if parsed_text is not None:
            parsed_data.append(data[idx])
    
    n_err = 0
    n_key_err = 0 
    n_fewshot_repeat = 0
    valid_data = []
    for idx in range(len(parsed_data)):
        # if idx % 1000 == 0:
        #     print(idx)
        try:
            # print(parsed_data[idx]["parsed_text"])
            if "microscope" in parsed_data[idx]["parsed_text"]:
                n_fewshot_repeat += 1
                print(f"n_fewshot_repeat:{n_fewshot_repeat}")
                continue
            parsed_json = json.loads(parsed_data[idx]["parsed_text"])
            parsed_data[idx]["parsed_json"] = parsed_json
            # print('---')
            # print(parsed_json.keys())
            if not ("easy_query" in parsed_json.keys()):
                n_key_err += 1
                print(n_key_err)
                continue
            
            valid_data.append(parsed_data[idx])
                
        except:
            n_err += 1
            print(n_err)
    print(len(valid_data))

    for idx in range(len(valid_data)):
        n_key_err = 0
        try:
            this_data = valid_data[idx]
            this_data["query"] = [
                this_data["parsed_json"]["easy_query"], this_data["parsed_json"]["hard_query"], this_data["parsed_json"]["discussion"]
            ]
        except KeyError:
            n_key_err += 1
            print(f"n_key_err:{n_key_err}")
        # print(idx)
    
    output_data = []
    
    for idx in range(len(valid_data)):
        this_data = valid_data[idx]
        output_data.append(
            {
                "id": this_data["id"],
                "query": {
                    "instruction": "Represent this query for retrieving relative documents: ",
                    "text": random.choice(this_data["query"]),
                    "image": None,
                },
                "pos": [
                    {
                        "instruction": "",
                        "text": "",
                        "image": None,
                    }
                ],
                "neg": [], # NO negative samples
                
            }
        )
    
    with open("train_data_w_o_image.jsonl", "w") as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("done")