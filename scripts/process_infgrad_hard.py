from datasets import load_dataset
import jsonlines
import pandas as pd
output_file = "../../dataset/our-zh/retrieval_data_llm.jsonl"
dataset = load_dataset('infgrad/retrieval_data_llm',trust_remote_code=True)
dataset = dataset['train']
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
DOC_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
dataset = dataset.rename_columns({ "Query":"query","Positive Document" : "pos","Hard Negative Document":"neg"})

dataset = dataset.map(lambda x: {"query" : [QUERY_INSTRUCTION, x["query"]], "pos" : [DOC_INSTRUCTION, x["pos"]], "neg" : [DOC_INSTRUCTION, x["neg"]]})
with jsonlines.open(output_file, mode='w') as writer:
    for data in dataset:
        writer.write(data)