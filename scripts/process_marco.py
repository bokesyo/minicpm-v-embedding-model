from datasets import load_dataset
import jsonlines
import pandas as pd
from copy import deepcopy
output_file = "../../dataset/our-zh/hard_negs/marco_chinese.jsonl"
dataset = load_dataset('unicamp-dl/mmarco', 'chinese',trust_remote_code=True)
dataset = dataset['train']
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
DOC_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
dataset = dataset.rename_columns({ "positive" : "pos","negative":"neg"})

# make dataset to df
dataset_pd = dataset.to_pandas()
# only remain the unique queries
# dataset_pd.groupby("query")
dataset_pd.sort_values("query",inplace=True)
dataset_pd.drop_duplicates(subset="query", keep="first", inplace=True)
dataset_pd.reset_index(inplace=True)
undroped_pd = dataset.to_pandas()
# undroped_pd.groupby("query")
undroped_pd.sort_values("query",inplace=True)
undroped_pd.reset_index(inplace=True)
unde_i = 0


new_datas = []
for i in range(len(dataset_pd)):
    query = dataset_pd.loc[i]["query"]
    negs = []
    
    unde_query = undroped_pd.loc[unde_i]["query"]
    while query == unde_query:
        negs.append(undroped_pd.loc[unde_i]["neg"])
        unde_i += 1
        # print(i,unde_i)
        if unde_i == len(undroped_pd):
            break
        unde_query = undroped_pd.loc[unde_i]["query"]
    # dataset_pd.loc[i]["neg"] = negs[:10]
    x =  dataset_pd.loc[i]
    negs = negs[:10]
    new_datas.append({"query" : [QUERY_INSTRUCTION, x["query"]], "pos" : [DOC_INSTRUCTION, x["pos"]], "neg" : [DOC_INSTRUCTION, *deepcopy(negs)]})
    # print(new_datas[0])

with jsonlines.open(output_file, mode='w') as writer:
    for data in new_datas:
        writer.write(data)