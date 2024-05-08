## simclue ##

from datasets import load_dataset
import jsonlines

###### ATTENTION: NEG and INSTRUCTION need edit!!!! #######

# QUERY_INSTRUCTION = "Represent the sentence for retrieving duplicate sentences;"
# DOC_INSTRUCTION = "Represent the sentence for retrieving duplicate sentences;"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
DOC_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
# read simclue
simclue_path = "../../dataset/simclue_public/train_rank.json"
simclue_output_path = "../../dataset/our-zh/simclue.jsonl"
simclue_data = []

with jsonlines.open(simclue_path) as reader:
    for obj in reader:
        simclue_data.append(obj)
    
keys_to_keep = ["query", "neg", "pos"]
keys_to_delete = [key for key in simclue_data[0].keys() if key not in keys_to_keep]

# add instruction
for item in simclue_data:
    item["pos"] = item["title"]
    item["neg"] = item["neg_title"]
    item["query"] = [QUERY_INSTRUCTION,item["query"]]
    item["neg"] = [DOC_INSTRUCTION,item["neg"]]
    item["pos"] = [DOC_INSTRUCTION,item["pos"]]
    for key in keys_to_delete:
        item.pop(key)
        
with jsonlines.open(simclue_output_path, mode='w') as writer:
    writer.write_all(simclue_data)