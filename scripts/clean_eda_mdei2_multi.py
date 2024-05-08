from FlagEmbedding import FlagModel
import numpy as np
import jsonlines
from tqdm import tqdm
from joblib import Parallel, delayed
# import faiss
import os
from itertools import chain

np.random.seed(42)
###### ATTENTION: NEG and INSTRUCTION need edit!!!! #######


input_file = "../../dataset/medi2-data-jsonl/hard_negs/train.jsonl"
output_file = "../../dataset/medi2-data-jsonl/hard_negs/split/"

# input_file = "../../dataset/medi2-data-jsonl/train.jsonl"
# output_file = "../../dataset/medi2-data-jsonl/clean/train.jsonl"


# model = FlagModel('BAAI/bge-base-en-v1.5', 
#                   query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
#                   use_fp16=True) 


    
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


# for dataset in datasets:
#     num_i = dataset[0]["task_id"]
#     print(num_i)
#     with jsonlines.open(output_file + str(num_i) + ".jsonl" , mode="w") as f:
#         for x in dataset:#[:100]:
#             f.write(x)

# del data    
    
# def clean_subdataset(dataset):
#     num_i = dataset[0]["task_id"]
#     # find the hard negative sample from the dataset
#     queries = [x["query"][1] for x in dataset]
#     poss = [x["pos"][1] for x in dataset]
#     negs = [x["neg"][1:11] for x in dataset]
#     negs = list(chain.from_iterable(i for i in negs))
#     assert len(negs) % 10 == 0
    
#     n_embeddings = model.encode_corpus(negs)
#     n_embeddings = np.array_split(n_embeddings,len(negs)//10,axis=0)
#     print(len(n_embeddings))
#     n_embeddings = map(list,n_embeddings)


#     q_embeddings = model.encode_queries(queries)
#     p_embeddings = model.encode_corpus(poss)

#     remove_indies = set()
#     diffs = []

#     for i,(q_e,p_e,n_es) in tqdm(enumerate(zip(q_embeddings, p_embeddings, n_embeddings))):
#         qp_sim = np.dot(q_e, p_e)/(np.linalg.norm(q_e)*np.linalg.norm(p_e))
#         l_nes = len(n_es)
#         for j in range(l_nes-1,-1,-1):
#             n_e = n_es[j]
#             qn_sim = np.dot(q_e, n_e)/(np.linalg.norm(q_e)*np.linalg.norm(n_e))
#             if qn_sim > qp_sim or abs(qp_sim - qn_sim) < 0.05:
#                 n_es.pop(j)
#                 dataset[i]["neg"].pop(j+1) # +1 for the instruction
#             else:
#                 diffs.append(abs(qp_sim - qn_sim))
#         if len(n_es) < 5:
#             remove_indies.add(i)
    
#     if len(diffs) != 0:
#         for i in range(1,11):
#             print("diff_back_{}percent: {}".format(i*10, np.percentile(diffs, i*10)))

#         diff_back_10percent = np.percentile(diffs, 10)
#         diff_back_20percent = np.percentile(diffs, 20)
#         print(np.mean([len(x["neg"])for x in dataset]))

#         for i,(q_e,p_e,n_es) in enumerate(zip(q_embeddings, p_embeddings, n_embeddings)):
#             qp_sim = np.dot(q_e, p_e)/(np.linalg.norm(q_e)*np.linalg.norm(p_e))
#             l_nes = len(n_es)
#             for j in range(l_nes-1,-1,-1):
#                 n_e = n_es[j]
#                 qn_sim = np.dot(q_e, n_e)/(np.linalg.norm(q_e)*np.linalg.norm(n_e))
#                 diff = abs(qp_sim - qn_sim) 
#                 if diff < diff_back_20percent:
#                     if diff <= diff_back_10percent:
#                         if np.random.random() < 0.6:
#                             n_es.pop(j)
#                             dataset[i]["neg"].pop(j+1)
#                     elif np.random.random() < 0.25:
#                         n_es.pop(j)
#                         dataset[i]["neg"].pop(j+1)
#             if len(n_es) < 5:
#                 remove_indies.add(i)

#     dataset = [dataset[i] for i in range(len(dataset)) if i not in remove_indies]
#     print(len(dataset))
#     print(np.mean([len(x["neg"])for x in dataset]))
#     with jsonlines.open(output_file + num_i + "jsonl" , mode="w") as f:
#         for x in datasets:#[:100]:
#             f.write(x)

# Parallel(n_jobs=2)(delayed(clean_subdataset)(d) for d in datasets)
#     new_datasets.append(dataset)
    
# datasets = list(chain.from_iterable(new_datasets))
                     
# with jsonlines.open(output_file , mode="w") as f:
#     for x in datasets:#[:100]:
#         f.write(x)