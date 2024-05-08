from FlagEmbedding import FlagModel
import numpy as np
import jsonlines
from tqdm import tqdm
# from joblib import Parallel, delayed
# import faiss
import os

np.random.seed(42)
###### ATTENTION: NEG and INSTRUCTION need edit!!!! #######


dataset_paths = "../../dataset/medi-data-jsonl/"
dataset_output_paths = "../../dataset/medi-data-jsonl/clean/"

model = FlagModel('BAAI/bge-base-en-v1.5', 
                  query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
                  use_fp16=True) 

dataset_files = os.listdir(dataset_paths)

for d in dataset_files:
    if d.endswith(".jsonl"):
        d = d.replace(".jsonl", "")
    else:
        continue
    # if not d.startswith('retrieval_data_llm'):
    #     continue
    print("================{}==============".format(d))
    if os.path.exists(dataset_output_paths + d + ".jsonl"):
        print("already exist")
        continue
    
    with jsonlines.open(dataset_paths + d + ".jsonl", "r") as reader:
        dataset = list(reader)
    print("length:",len(dataset))

    
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
        qp_sim = np.dot(q_e, p_e)/(np.linalg.norm(q_e)*np.linalg.norm(p_e))
        qn_sim = np.dot(q_e, n_e)/(np.linalg.norm(q_e)*np.linalg.norm(n_e))
        
        if qn_sim > qp_sim or abs(qp_sim - qn_sim) < 0.05:
            remove_indies.add(i)
        else:
            diffs.append(abs(qp_sim - qn_sim))
    
    dataset = [dataset[i] for i in range(len(dataset)) if i not in remove_indies]
    remove_indies = set()
    
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
                         
    with jsonlines.open(dataset_output_paths + d + ".jsonl", mode="w") as f:
        for x in dataset:#[:100]:
            f.write(x)