
import numpy as np
import jsonlines
from tqdm import tqdm
# from joblib import Parallel, delayed
# import faiss
import os

np.random.seed(42)
###### ATTENTION: NEG and INSTRUCTION need edit!!!! #######

dataset_paths = "../../dataset/our-zh/"
dataset_output_paths = "../../dataset/our-zh/bm25_index_input/"


dataset_files = os.listdir(dataset_paths)

for d in dataset_files:
    if d.endswith(".jsonl"):
        d = d.replace(".jsonl", "")
    else:
        continue
    # if not d.startswith('retrieval_data_llm'):
    #     continue
    print("================{}==============".format(d))
    if os.path.exists(dataset_output_paths + d):
        print("already exist")
        continue
    
    with jsonlines.open(dataset_paths + d + ".jsonl", "r") as reader:
        dataset = list(reader)
    new_dataset = []
    for i,row in enumerate(dataset):
        new_dataset.append(
            {
                "id":str(i),
                "contents":row["pos"][1]
            }
        )
    with jsonlines.open(dataset_output_paths + d + ".jsonl", mode="w") as f:
        for x in new_dataset:#[:100]:
            f.write(x)
    
# python -m pyserini.index.lucene \
#   --collection JsonCollection \
#   --input /data/WorkSpace/openbmb/dataset/our-zh/bm25_index_input \
#   --index /data/WorkSpace/openbmb/dataset/our-zh/bm25_index \
#   --generator DefaultLuceneDocumentGenerator \
#   --threads 1 \
#   --language zh \
#   --storePositions --storeDocvectors --storeRaw