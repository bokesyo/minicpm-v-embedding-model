from datasets import load_from_disk
from string import Template
from FlagEmbedding import FlagModel
import numpy as np
import jsonlines
from tqdm import tqdm
# from joblib import Parallel, delayed
import faiss
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
dataset_output_paths_template = Template("../../dataset/our-zh/hard_negs/$dataset_name.jsonl")

model = FlagModel('BAAI/bge-base-zh-v1.5', 
                  query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                  use_fp16=True) 
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
    q_embeddings = model.encode_queries(queries)
    d_embeddings = model.encode(docs)
    # for each query, find the hard negative sample from 25-35
    # use numpy to find the hard negative sample
    # parrallel 
    negs = [None] * len(dataset)
    # def process_query(i):
    #     similarity_q = np.dot(q_embeddings[i], d_embeddings.T)
    #     negs[i] = docs[np.argsort(similarity_q)[::-1][np.random.randint(25, 36)]]
    #     # print(dataset[i])
    #     return
    
    # # for i in range(len(dataset)):
    # Parallel(n_jobs=-1,backend='threading')(delayed(process_query)(i) for i in tqdm(range(len(dataset))))
    # print(negs)
    
    # use Faiss gpu
    # Step 1: change data type
    
    q_embeddings = q_embeddings.astype("float32")
    d_embeddings = d_embeddings.astype("float32")
    
    # Step 2 : Instantiate a FAISS index
    dim = q_embeddings.shape[1]
    measure = faiss.METRIC_INNER_PRODUCT
    param = 'IVF{},PQ16'.format(len(dataset)//100)
    index = faiss.index_factory(dim, param, measure)
    index.train(d_embeddings)
    index.add(d_embeddings)
    print("index")
    
    # Step 5: Search the index
    index.nprobe = 10
    _,similarity_q_index = index.search(q_embeddings, 36)
    # negs_index = np.random.randint(25, 36, size=(d_embeddings.shape[0],))
    # negs = [docs[similarity_q_index[i][negs_index[i]]] for i in range(len(dataset))]
    negs = [[docs[similarity_q_index[i][j]]for j in range(25,35)] for i in range(len(dataset))]
    
    # negs = [docs[negs_index[i]] for i in range(len(dataset))]
    # negs = [docs[np.argsort(similarity_q[i])[::-1][np.random.randint(25, 36)]] for i in range(len(dataset))]
    
    
    
    dataset = dataset.add_column("neg",negs)
    dataset = dataset.map(lambda x: {"query" : [QUERY_INSTRUCTION, x["query"]], "pos" : [DOC_INSTRUCTION, x["pos"]], "neg" : [DOC_INSTRUCTION, *x["neg"]]})
    with jsonlines.open(dataset_output_paths_template.substitute(dataset_name=d), mode="w") as f:
        for x in dataset:#[:100]:
            f.write(x)