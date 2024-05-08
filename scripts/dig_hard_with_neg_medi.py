from FlagEmbedding import FlagModel
import numpy as np
import jsonlines
from tqdm import tqdm
# from joblib import Parallel, delayed
import faiss
import os
import itertools

np.random.seed(42)
###### ATTENTION: NEG and INSTRUCTION need edit!!!! #######


input_file = "../../dataset/medi2-data-jsonl/train.jsonl"
output_file = "../../dataset/medi2-data-jsonl/hard_negs/train.jsonl"

# input_file = "../../dataset/medi2-data-jsonl/train.jsonl"
# output_file = "../../dataset/medi2-data-jsonl/clean/train.jsonl"


model = FlagModel('BAAI/bge-base-en-v1.5', 
                  query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
                  use_fp16=True)

with jsonlines.open(input_file, "r") as reader:
    data = list(reader)
data.sort(key=lambda x:x["task_id"])

datasets = []
i,j = 0,0
while j < len(data):
    if data[i]["task_id"] != data[j]["task_id"]:
        temp = data[i:j]
        if len(temp) != 0:
            datasets.append(temp)
        i = j
    else:
        j += 1

temp = data[i:j]

if len(temp) != 0:
    datasets.append(temp)
new_datasets = []
for dataset in datasets:
    
    # find the hard negative sample from the dataset
    queries = [x["query"][1] for x in dataset]
    docs = [x["pos"][1] for x in dataset]
    q_embeddings = model.encode_queries(queries)
    d_embeddings = model.encode_corpus(docs)
    old_negs = [x["neg"][1:] for x in dataset]


    # use Faiss gpu
    # Step 1: change data type
    
    q_embeddings = q_embeddings.astype("float32")
    d_embeddings = d_embeddings.astype("float32")
    
    # Step 2 : Instantiate a FAISS index
    dim = q_embeddings.shape[1]
    measure = faiss.METRIC_INNER_PRODUCT
    if len(dataset) < 10000:
        param =  "Flat"
    else:
        param = 'IVF{},PQ16'.format(len(dataset)//100)
    index = faiss.index_factory(dim, param, measure)
    if len(dataset) >= 10000:
        index.train(d_embeddings)
    index.add(d_embeddings)
    print("index")
    
    # Step 5: Search the index
    index.nprobe = 10
    _,similarity_q_index = index.search(q_embeddings, 36)
    # negs_index = np.random.randint(25, 36, size=(d_embeddings.shape[0],))
    # negs = [docs[similarity_q_index[i][negs_index[i]]] for i in range(len(dataset))]
    negs = [(neg+[docs[similarity_q_index[i][j]]for j in range(2,35)])[:10] for neg,i in zip(old_negs,range(len(dataset)))]
    
    for data,neg in tqdm(zip(dataset,negs), total=len(dataset)):
        data["neg"] = [data["neg"][0], *neg]
        # data["query"] = [QUERY_INSTRUCTION, data["query"]]
        # data["pos"] = [DOC_INSTRUCTION, data["pos"]]
                         
datasets = list(itertools.chain.from_iterable(datasets))
                     
with jsonlines.open(output_file , mode="w") as f:
    for x in datasets:#[:100]:
        f.write(x)