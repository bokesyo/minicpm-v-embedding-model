from FlagEmbedding import FlagModel
import numpy as np
import jsonlines
from tqdm import tqdm
# from joblib import Parallel, delayed
# import faiss
import os
import itertools

np.random.seed(42)
###### ATTENTION: NEG and INSTRUCTION need edit!!!! #######


input_file = "../../dataset/medi2-data-jsonl/train.jsonl"
output_file = "../../dataset/medi2-data-jsonl/clean/train.jsonl"

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
    poss = [x["pos"][1] for x in dataset]
    negs = [x["neg"][1] for x in dataset]


    q_embeddings = model.encode_queries(queries)
    p_embeddings = model.encode_corpus(poss)
    n_embeddings = model.encode_corpus(negs)

    remove_indies = set()
    diffs = []

    for i,(q_e,p_e,n_e) in enumerate(zip(q_embeddings, p_embeddings, n_embeddings)):
        qp_sim = np.dot(q_e, p_e)/(np.linalg.norm(q_e)*np.linalg.norm(p_e)+1e-6)
        qn_sim = np.dot(q_e, n_e)/(np.linalg.norm(q_e)*np.linalg.norm(n_e)+1e-6)

        if qn_sim > qp_sim or abs(qp_sim - qn_sim) < 0.05:
            remove_indies.add(i)
        else:
            diffs.append(abs(qp_sim - qn_sim))

    dataset = [dataset[i] for i in range(len(dataset)) if i not in remove_indies]
    remove_indies = set()
    if len(diffs) != 0:
        for i in range(1,11):
            print("diff_back_{}percent: {}".format(i*10, np.percentile(diffs, i*10)))

        diff_back_10percent = np.percentile(diffs, 10)
        diff_back_20percent = np.percentile(diffs, 20)

        for i,diff in enumerate(diffs):
            if i < diff_back_20percent:
                if i <= diff_back_10percent:
                    if np.random.random() < 0.6:
                        remove_indies.add(i)
                elif np.random.random() < 0.25:
                    remove_indies.add(i)

    dataset = [dataset[i] for i in range(len(dataset)) if i not in remove_indies]
    new_datasets.append(dataset)
    
datasets = list(itertools.chain.from_iterable(new_datasets))
                     
with jsonlines.open(output_file , mode="w") as f:
    for x in datasets:#[:100]:
        f.write(x)