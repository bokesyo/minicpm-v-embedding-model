'''
This sample python shows how to evaluate BEIR dataset quickly using Mutliple GPU for evaluation (for large datasets).
To run this code, you need Python >= 3.7 (not 3.6)
Enabling multi-gpu evaluation has been thanks due to tremendous efforts of Noumane Tazi (https://github.com/NouamaneTazi)

IMPORTANT: The following code will not run with Python 3.6! 
1. Please install Python 3.7 using Anaconda (conda create -n myenv python=3.7)

You are good to go!

To run this code, you preferably need access to mutliple GPUs. Faster than running on single GPU.
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 examples/retrieval/evaluation/dense/evaluate_sbert_multi_gpu.py
'''

from beir.retrieval import models
from beir.datasets.data_loader_hf import HFDataLoader
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalParallelExactSearch as DRPES
import time

import logging
import os
import random
import argparse

import torch
from torch import distributed as dist

import json


parser = argparse.ArgumentParser(description='Mteb eval multi gpu')
parser.add_argument('--model_path', type=str, help='Path to the model')
parser.add_argument('--dataset_path', type=str, help='Path to the dataset')
parser.add_argument('--query_instruction', type=str, help='query instruction')
parser.add_argument('--corpus_instruction', type=str, help='corpus instruction')
parser.add_argument('--log_path', type=str, help='Path to the log dir')
parser.add_argument('--batch_size', type=int, help='Encoder batch size')
parser.add_argument('--max_query_len', type=int, help='max query token length')
parser.add_argument('--max_corpus_len', type=int, help='max corpus token length')
parser.add_argument('--pooling_method', type=str, help='pooling method', default='lasttoken')
parser.add_argument('--split', type=str, help='dataset split', default='test')

if __name__ == "__main__":
    args = parser.parse_args()
    
    os.makedirs(args.log_path, exist_ok=True)

    model_path = args.model_path
    dataset_path = args.dataset_path
    
    # Initialize torch.distributed
    dist.init_process_group("nccl")
    device_id = int(os.getenv("LOCAL_RANK", 0)) # multi-gpu
    torch.cuda.set_device(torch.cuda.device(device_id))

    # Enable logging only first rank=0
    rank = int(os.getenv("RANK", 0))
    if rank != 0:
        logging.basicConfig(level=logging.WARN)
    else:
        logging.basicConfig(level=logging.INFO)

    tick = time.time()

    # dataset = "nfcorpus"
    keep_in_memory = False
    streaming = True
    corpus_chunk_size = 50000
    batch_size = args.batch_size # sentence bert model batch size
    # model_name = "msmarco-distilbert-base-tas-b"
    
    ignore_identical_ids = False # we don't choose top 1, we choose top 30, faster!!!!

    # corpus, queries, qrels = HFDataLoader(hf_repo=f"BeIR/{dataset}", streaming=streaming, keep_in_memory=keep_in_memory).load(split="test")

    corpus, queries, qrels = HFDataLoader(data_folder=dataset_path, streaming=False).load(split=[args.split]) # this is only for MSMARCO
    
    #### Dense Retrieval using SBERT (Sentence-BERT) ####
    #### Provide any pretrained sentence-transformers model
    #### The model was fine-tuned using cosine-similarity.
    #### Complete list - https://www.sbert.net/docs/pretrained_models.html
    # beir_model = models.SentenceBERT(model_name)
    
    if args.query_instruction != "none":
        query_instruction_content = open(args.query_instruction, 'r').read()
    else:
        query_instruction_content = ""
    
    if args.corpus_instruction != "none": 
        corpus_instruction_content = open(args.corpus_instruction, 'r').read()
    else:
        corpus_instruction_content = ""
    
    beir_model = models.SentenceBERT(
        model_path, 
        pooling_method=args.pooling_method, 
        normalize=True, 
        query_instruction=query_instruction_content, 
        corpus_instruction=corpus_instruction_content, 
        max_query_len=args.max_query_len, 
        max_corpus_len=args.max_corpus_len
    )

    #### Start with Parallel search and evaluation
    model = DRPES(beir_model, batch_size=batch_size, corpus_chunk_size=corpus_chunk_size)
    retriever = EvaluateRetrieval(model, k_values=[1,3,5,10,30,100], score_function="dot")

    #### Retrieve dense results (format of results is identical to qrels)
    start_time = time.time()
    results = retriever.retrieve(corpus, queries, ignore_identical_ids=ignore_identical_ids) 
    end_time = time.time()
    print("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))

    #### Evaluate your retrieval using NDCG@k, MAP@K ...

    logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
    ndcg, _map, recall, precision = retriever.evaluate(
        qrels, results, retriever.k_values, ignore_identical_ids=ignore_identical_ids)

    mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
    recall_cap = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="r_cap")
    hole = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="hole")

    print("ndcg", ndcg)
    print("precision", precision)
    print("recall", recall)
    
    metric = {
        "ndcg": ndcg,
        "precision": precision,
        "recall": recall,
        "mrr": mrr
    }
    
    tock = time.time()
    print("--- Total time taken: {:.2f} seconds ---".format(tock - tick))

    #### Print top-k documents retrieved ####
    # top_k = 10

    # query_id, ranking_scores = random.choice(list(results.items()))
    # scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
    # query = queries.filter(lambda x: x['id']==query_id)[0]['text']
    # logging.info("Query : %s\n" % query)

    # for rank in range(top_k):
    #     doc_id = scores_sorted[rank][0]
    #     doc = corpus.filter(lambda x: x['id']==doc_id)[0]
    #     # Format: Rank x: ID [Title] Body
    #     logging.info("Rank %d: %s [%s] - %s\n" % (rank+1, doc_id, doc.get("title"), doc.get("text")))
    with open(os.path.join(args.log_path, 'metric.json'), 'w') as f:
        f.write(json.dumps(metric, ensure_ascii=False, indent=4))
    
    with open(os.path.join(args.log_path, 'results.json'), 'w') as f:
        f.write(json.dumps(results, ensure_ascii=False, indent=4))
        