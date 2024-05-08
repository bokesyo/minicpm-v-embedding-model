import jsonlines
import numpy as np

input_file = "../../dataset/du_retrieve/dureader-retrieval-ranking/train.jsonl"
output_file = "../../dataset/our-zh/hard_negs/dhreader_retrieval.jsonl"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
DOC_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
np.random.seed(42)
with jsonlines.open(input_file) as reader:
    dataset = list(reader)
    
new_dataset = []
for data in dataset:
    pos = data["positive_passages"]
    if len(pos) == 0:
        continue
    pos = np.random.choice(pos, 1)[0]["text"]
    
    neg = data["negative_passages"]
    if len(neg) == 0:
        continue
        
    negs = [i["text"] for i in np.random.choice(neg, 10)]
    new_data = {"query" : [QUERY_INSTRUCTION, data["query"]], "pos" : [DOC_INSTRUCTION, pos], "neg" : [DOC_INSTRUCTION, *negs]}
    new_dataset.append(new_data)

with jsonlines.open(output_file, mode='w') as writer:
    for data in new_dataset:
        writer.write(data)