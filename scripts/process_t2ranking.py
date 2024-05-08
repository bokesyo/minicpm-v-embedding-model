
import random

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import jsonlines

args = {
    "sample_num":11,
    "batch_size":64,
    "max_index":36,
    "top1000":"data/train.mined.tsv",
    "min_index":25,
    "max_seq_len":332,
    # "q_max_seq_len":32,
    # "p_max_seq_len":300,
    "collection":"data/collection.tsv",
    "qrels":"data/qrels.retrieval.train.tsv",
    "query":"data/queries.train.tsv",
    
}
output_file = "../our-zh/t2ranking_10_neg.jsonl"


QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
DOC_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

class DualEncoderTrainDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.collection = pd.read_csv(args['collection'],sep="\t", quoting=3)
        self.collection.columns=['pid','para']
        self.collection = self.collection.fillna("NA")
        self.collection.index = self.collection.pid 
        self.collection.pop('pid')
        self.query = pd.read_csv(args['query'],sep="\t")
        self.query.columns = ['qid','text']
        self.query.index = self.query.qid
        self.query.pop('qid')
        self.top1000 = pd.read_csv(args['top1000'], sep="\t")
        if len(self.top1000.columns)==3:
            self.top1000.columns=['qid','pid','index']
        else:
            self.top1000.columns=['qid','pid','index','score']
        self.top1000 = list(self.top1000.groupby("qid"))
        self.len = len(self.top1000)
        self.min_index = args['min_index']
        self.max_index = args['max_index']
        qrels={}
        with open(args['qrels'],'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                qid,pid = line.split()
                qid=int(qid)
                pid=int(pid)
                x=qrels.get(qid,[])
                x.append(pid)
                qrels[qid]=x
        self.qrels = qrels
        self.sample_num = args['sample_num']-1   
        self.epoch = 0
        self.num_samples = len(self.top1000)

    def set_epoch(self, epoch):
        self.epoch = epoch
        print(self.epoch)
    
    def sample(self, qid, pids, sample_num):
        '''
        qid:int
        pids:list
        sample_num:int
        '''
        pids = [pid for pid in pids if pid not in self.qrels[qid]]
        pids = pids[self.args['min_index']:self.args['max_index']]
        if len(pids)<sample_num:
            pad_num = sample_num - len(pids)
            pids+=[random.randint(0, 2303643) for _ in range(pad_num)]  # 用random neg补充
        interval = len(pids)//sample_num
        offset = self.epoch%interval
        sample_pids = pids[offset::interval][:sample_num]
        return sample_pids

    def __getitem__(self, idx):
        cols = self.top1000[idx]
        qid = cols[0]
        pids = list(cols[1]['pid'])
        sample_neg_pids = self.sample(qid, pids, self.sample_num)
        pos_id = random.choice(self.qrels.get(qid))
        query = self.query.loc[qid]['text']
        data = self.collection.loc[pos_id]
        psgs = [data['para']]
        for neg_pid in sample_neg_pids:
            data = self.collection.loc[neg_pid]
            psgs.append(data['para'])
        return {"query":query,"pos":psgs[0],"neg":psgs[1:10]}

    # def _collate_fn(self, sample_list):
    #     qrys = []
    #     psgs = []
    #     for q, p in sample_list:
    #         qrys+=q 
    #         psgs+=p 
    #     q_records = self.tokenizer(qrys, padding=True, truncation=True, return_tensors="pt", max_length=self.args['q_max_seq_len'])
    #     p_records = self.tokenizer(psgs, padding=True, truncation=True, return_tensors="pt", max_length=self.args['p_max_seq_len'])
    #     return {"query_inputs":q_records, "passage_inputs":p_records}

    def __len__(self):
        return self.num_samples



train_dataset = DualEncoderTrainDataset(args)

dataloader = DataLoader(train_dataset,batch_size=1)
datas = []

for i, data in enumerate(tqdm(dataloader)):
    datas.append(data)

with jsonlines.open(output_file,mode='w') as writer:
    for data in datas:
        data = {"query":[QUERY_INSTRUCTION,data["query"][0]],"pos":[DOC_INSTRUCTION,data["pos"][0]],"neg":[DOC_INSTRUCTION,*data["neg"][0]]}
        writer.write(data)