from FlagEmbedding import FlagModel
import numpy as np
import jsonlines
from tqdm import tqdm
from joblib import Parallel, delayed
# import faiss
from itertools import chain
import os

np.random.seed(42)
###### ATTENTION: NEG and INSTRUCTION need edit!!!! #######

QUERY_INSTRUCTION_FOR_EMBEDMODEL = "为这个句子生成表示以用于检索相关文章："

# dataset_paths = "../../dataset/our-zh/hard_negs/"
# dataset_output_paths = "../../dataset/our-zh/hard_negs/clean/"

dataset_paths = "../../dataset/medi2-data-jsonl/hard_negs/split/"
dataset_output_paths = "../../dataset/medi2-data-jsonl/hard_negs/clean/"



dataset_files = os.listdir(dataset_paths)

def clean_dataset(d):
# for d in dataset_files:
    if d.endswith(".jsonl"):
        d = d.replace(".jsonl", "")
    else:
        return
    # if not d.startswith('retrieval_data_llm'):
    #     continue
    print("================{}==============".format(d))
    if os.path.exists(dataset_output_paths + d + ".jsonl"):
        print("already exist")
        return
    if d.startswith("zhihu") or d.startswith("shenqing"):
        return
    
    # # debug
    # if not d.startswith("cmrc2018"):
    #     continue
 
    with jsonlines.open(dataset_paths + d + ".jsonl", "r") as reader:
        dataset = list(reader)
    print("length:",len(dataset))

    # model = FlagModel('BAAI/bge-base-zh-v1.5', 
    #               query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
    #               use_fp16=True) 
    model = FlagModel('BAAI/bge-base-en-v1.5', 
                  query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
                  use_fp16=True) 


    
    # find the hard negative sample from the dataset
    queries = [x["query"][1] for x in dataset]
    poss = [x["pos"][1] for x in dataset]
    negs = [x["neg"][1:11] for x in dataset]
    negs = list(chain.from_iterable(i for i in negs))
    assert len(negs) % 10 == 0
    
    n_embeddings = model.encode_corpus(negs)
    n_embeddings = np.array_split(n_embeddings,len(negs)//10,axis=0)
    print(len(n_embeddings))
    n_embeddings = map(list,n_embeddings)
    # n_embeddings = [n_embeddings[i:i+10].tolist() for i in range(0,len(n_embeddings),10)]
    q_embeddings = model.encode_queries(queries)
    p_embeddings = model.encode_corpus(poss)
    
    remove_indies = set()
    diffs = []


    for i,(q_e,p_e,n_es) in tqdm(enumerate(zip(q_embeddings, p_embeddings, n_embeddings))):
        qp_sim = np.dot(q_e, p_e)/(np.linalg.norm(q_e)*np.linalg.norm(p_e))
        l_nes = len(n_es)
        for j in range(l_nes-1,-1,-1):
            n_e = n_es[j]
            qn_sim = np.dot(q_e, n_e)/(np.linalg.norm(q_e)*np.linalg.norm(n_e))
            if qn_sim > qp_sim or abs(qp_sim - qn_sim) < 0.05:
                n_es.pop(j)
                dataset[i]["neg"].pop(j+1) # +1 for the instruction
            else:
                diffs.append(abs(qp_sim - qn_sim))
        if len(n_es) < 5:
            remove_indies.add(i)
    if len(diffs) != 0:
        for i in range(1,11):
            print("diff_back_{}percent: {}".format(i*10, np.percentile(diffs, i*10)))
            
        diff_back_10percent = np.percentile(diffs, 10)
        diff_back_20percent = np.percentile(diffs, 20)
    
        print(np.mean([len(x["neg"])for x in dataset]))
        
        for i,(q_e,p_e,n_es) in enumerate(zip(q_embeddings, p_embeddings, n_embeddings)):
            qp_sim = np.dot(q_e, p_e)/(np.linalg.norm(q_e)*np.linalg.norm(p_e))
            l_nes = len(n_es)
            for j in range(l_nes-1,-1,-1):
                n_e = n_es[j]
                qn_sim = np.dot(q_e, n_e)/(np.linalg.norm(q_e)*np.linalg.norm(n_e))
                diff = abs(qp_sim - qn_sim) 
                if diff < diff_back_20percent:
                    if diff <= diff_back_10percent:
                        if np.random.random() < 0.6:
                            n_es.pop(j)
                            dataset[i]["neg"].pop(j+1)
                    elif np.random.random() < 0.25:
                        n_es.pop(j)
                        dataset[i]["neg"].pop(j+1)
            if len(n_es) < 5:
                remove_indies.add(i)
                    
    dataset = [dataset[i] for i in range(len(dataset)) if i not in remove_indies]
    print(len(dataset))
    print(np.mean([len(x["neg"])for x in dataset]))
                         
    with jsonlines.open(dataset_output_paths + d + ".jsonl", mode="w") as f:
        for x in dataset:#[:100]:
            f.write(x)
    return

Parallel(n_jobs=6)(delayed(clean_dataset)(d) for d in dataset_files)