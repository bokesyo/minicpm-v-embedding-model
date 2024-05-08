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

dataset_paths = "../../dataset/our-zh_raw/"
dataset_output_paths = "../../dataset/our-zh/hard_negs/"

model = FlagModel('BAAI/bge-base-zh-v1.5', 
                  query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                  use_fp16=True) 
keys_to_keep = ["query", "pos"]

dataset_files = os.listdir(dataset_paths)

for d in dataset_files:
    if d.endswith(".jsonl"):
        d = d.replace(".jsonl", "")
    else:
        continue
    # if d.startswith('shenqing'):
    #     continue
    print("================{}==============".format(d))
    if os.path.exists(dataset_output_paths + d + ".jsonl"):
        print("already exist")
        continue
    
    with jsonlines.open(dataset_paths + d + ".jsonl", "r") as reader:
        dataset = list(reader)
    print("length:",len(dataset))

    
    # find the hard negative sample from the dataset
    queries = [x["query"] for x in dataset]
    docs = [x["pos"] for x in dataset]
    q_embeddings = model.encode_queries(queries)
    d_embeddings = model.encode(docs)
    negs = [None] * len(dataset)

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
    
    for data,neg in tqdm(zip(dataset,negs), total=len(dataset)):
        data["neg"] = [DOC_INSTRUCTION, *neg]
        data["query"] = [QUERY_INSTRUCTION, data["query"]]
        data["pos"] = [DOC_INSTRUCTION, data["pos"]]
                         
    with jsonlines.open(dataset_output_paths + d + ".jsonl", mode="w") as f:
        for x in dataset:#[:100]:
            f.write(x)