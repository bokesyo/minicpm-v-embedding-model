from datasets import load_from_disk
from string import Template
import numpy as np
import jsonlines
from tqdm import tqdm
from joblib import Parallel, delayed
from rank_bm25 import BM25Okapi
import jieba
import os

np.random.seed(42)
###### ATTENTION: NEG and INSTRUCTION need edit!!!! #######

QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
DOC_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

QUERY_INSTRUCTION_FOR_EMBEDMODEL = "为这个句子生成表示以用于检索相关文章："

dataset_names = [
    "alpaca_gpt4",
    # "belle_2m",
    "chatmed_consult",
    "csl",
    "dureader_robust",
    # "firefly",
    # "miracl",
    "mlqa",
    "webqa",
    "xlsum",
    "zhihu_kol"
]
dataset_paths_template = Template("../../dataset/m3e/$dataset_name.dataset")
dataset_output_paths_template = Template("../../dataset/our-zh/hard_negs_bm25/$dataset_name.jsonl")

keys_to_keep = ["query", "pos"]

dataset_spilt = {
    "csl":"csl",
}


for d in dataset_names:
    print("================{}==============".format(d))
    if os.path.exists(dataset_output_paths_template.substitute(dataset_name=d)):
        continue
    dataset_path = dataset_paths_template.substitute(dataset_name=d)
    dataset = load_from_disk(dataset_path)
    dataset = dataset["train"]
    print("length:",len(dataset))
    # cut only 100
    # dataset = dataset[dataset_spilt.get(d,"train")]
    dataset = dataset.rename_columns({"text" : "query", "text_pos" : "pos"})
    # dataset = dataset.add_column("neg",[""]*len(dataset))
    keys_to_delete = [k for k in dataset.column_names if k not in keys_to_keep]
    dataset = dataset.remove_columns(keys_to_delete)
    
    # find the hard negative sample from the dataset
    queries = [x["query"] for x in dataset]#[:100]
    docs = [x["pos"] for x in dataset]#[:100]
    negs =[None] * len(dataset)
    
    jiebad_docs = [list(jieba.cut(x)) for x in docs]
    bm25 = BM25Okapi(jiebad_docs)
    
    def process_query(i):
        jiebad_query = list(jieba.cut(queries[i]))
        doc_scores = bm25.get_scores(jiebad_query)
        negs[i] = [docs[j] for j in np.argsort(doc_scores)[::-1][25:35]]
        return
    
    Parallel(n_jobs=-1,backend='threading')(delayed(process_query)(i) for i in tqdm(range(len(dataset)))) #,backend='threading'
    
    # for query in tqdm(queries):
    #     jiebad_query = list(jieba.cut(query))
    #     doc_scores = bm25.get_scores(jiebad_query)
    #     negs.append([docs[i] for i in np.argsort(doc_scores)[25:35]])   
    
    
    dataset = dataset.add_column("neg",negs)
    dataset = dataset.map(lambda x: {"query" : [QUERY_INSTRUCTION, x["query"]], "pos" : [DOC_INSTRUCTION, x["pos"]], "neg" : [DOC_INSTRUCTION, *x["neg"]]})
    with jsonlines.open(dataset_output_paths_template.substitute(dataset_name=d), mode="w") as f:
        for x in dataset:#[:100]:
            f.write(x)