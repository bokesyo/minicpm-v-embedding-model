
import numpy as np
import jsonlines
from tqdm import tqdm
import os
from pyserini.search.lucene import LuceneSearcher
# from rank_bm25 import BM25Okapi
# from joblib import Parallel, delayed
# import jieba

###### ATTENTION: NEG and INSTRUCTION need edit!!!! #######

QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
DOC_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

dataset_paths = "../../dataset/our-zh/"
dataset_output_paths = "../../dataset/our-zh/hard_negs_bm25/"


dataset_files = os.listdir(dataset_paths)

for d in dataset_files:
    if d.endswith(".jsonl"):
        d = d.replace(".jsonl", "")
    else:
        continue
    if d.startswith('train'):
        continue
    print("================{}==============".format(d))
    if os.path.exists(dataset_output_paths + d + ".jsonl"):
        print("already exist")
        continue
    
    with jsonlines.open(dataset_paths + d + ".jsonl", "r") as reader:
        dataset = list(reader)
    print("length:",len(dataset))

    
    # find the hard negative sample from the dataset
    queries = [x["query"][1] for x in dataset]
    docs = [x["pos"][1] for x in dataset]
    negs = [None]*len(dataset)
    
    batch_size = 60
    need_remove = []
    for query_batch_idx in tqdm(range(len(dataset)//batch_size+1)):
        query_batch_idx_start = query_batch_idx * batch_size
        query_batch_idx_end = min((query_batch_idx+1) * batch_size, len(dataset))
        
        query_batch = queries[query_batch_idx_start:query_batch_idx_end]
        
        searcher = LuceneSearcher('../../dataset/our-zh/bm25_index/'+d)
        searcher.set_bm25(0.9, 0.4)
        searcher.set_language("zh")
        
        qids = list(range(query_batch_idx_start, query_batch_idx_end))
        qids =list(map(str,qids))
        # print(query_batch)
        batch_hits = searcher.batch_search(query_batch, qids,threads=batch_size,k=50)
        for i in range(query_batch_idx_start, query_batch_idx_end):
            try:
                negs[i] = [docs[int(j.docid)] for j in batch_hits[str(i)][25:35]]
                if len(negs[i]) != 10:
                    need_remove.append(i)
            except KeyError as e:
                print(e)
                need_remove.append(i)
    need_remove.sort(reverse=True)
    print(len(need_remove))
    for i in need_remove:
        dataset.pop(i)
        negs.pop(i)
        
    ####    rank_bm25   ####
    # jiebad_docs = [list(jieba.cut(x)) for x in docs]
    # bm25 = BM25Okapi(jiebad_docs)
    
    # def process_query(i):
    #     jiebad_query = list(jieba.cut(queries[i]))
    #     doc_scores = bm25.get_scores(jiebad_query)
    #     negs[i] = [docs[j] for j in np.argsort(doc_scores)[::-1][25:35]]
    #     return
    
    # Parallel(n_jobs=-1,backend='threading')(delayed(process_query)(i) for i in tqdm(range(len(dataset)))) #,backend='threading'
    # # for query in tqdm(queries):
    # #     jiebad_query = list(jieba.cut(query))
    # #     doc_scores = bm25.get_scores(jiebad_query)
    # #     negs.append([docs[i] for i in np.argsort(doc_scores)[25:35]])

    
    for data,neg in tqdm(zip(dataset,negs), total=len(dataset)):
        data["neg"] = [DOC_INSTRUCTION, *neg]
                         
    with jsonlines.open(dataset_output_paths + d + ".jsonl", mode="w") as f:
        for x in dataset:#[:100]:
            f.write(x) 