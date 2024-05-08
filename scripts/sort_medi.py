import jsonlines
import itertools

# input_file = "../../dataset/our-zh/train.jsonl"
input_file = "/data/WorkSpace/openbmb/dataset/medi2-data-jsonl/hard_negs/clean_5negs.jsonl"
batch_size = 256
# output_file = "../../dataset/our-zh/train_batch_{}.jsonl".format(batch_size)
output_file = "/data/WorkSpace/openbmb/dataset/medi2-data-jsonl/hard_negs/clean_03negs.jsonl"

data = []
with jsonlines.open(input_file) as reader:
    for obj in reader:
        data.append(obj)
        
print(len(data))

# print(data[0])
# print(data[-1])
data.sort(key=lambda x:x["task_id"])

# print(data[0])
# print(data[-1])
import numpy as np
np.random.seed(42)
i,j = 0,0
set_num = 1

ends = []
datasets = []
# lens = []
retrieval_keywords = ["etriev","erank"] # ["retrieving","retrieval","rerank"]

neg_words = ["wrong","Wrong", "ncorrect","egative","islead","isguide","istake","ntrue","False","false","naccurate","nrelated","issimilar","rrelevant","unlike"]

while j < len(data):
    if data[i]["task_id"] != data[j]["task_id"]:
        temp = data[i:j]
        if (any(nword in data[i]["query"][0] for nword in neg_words)) or (any(nword in data[i]["pos"][0] for nword in neg_words)):
            temp = np.random.choice(temp,len(temp)//3,replace=False)
        # if (any(retrieval_keyword in data[i]["query"][0] for retrieval_keyword in retrieval_keywords)) or (any(retrieval_keyword in data[i]["pos"][0] for retrieval_keyword in retrieval_keywords)):# and ("duplicate" not in data[i]["pos"][0]):
        #     datasets.append(temp)
        # else:
        #     temp = np.random.choice(temp,len(temp)//5,replace=False)
        #     datasets.append(temp)
        datasets.append(temp)
        # lens.append(j-i)
        # np.random.shuffle(temp)
        # data[i:j] = temp
        i = j
        set_num += 1
    else:
        j += 1

# print(set_num,j)
# print(data[-1])
temp = data[i:j]
if (any(nword in data[i]["query"][0] for nword in neg_words)) or (any(nword in data[i]["pos"][0] for nword in neg_words)) and data[i]["task_id"]:
    temp = np.random.choice(temp,len(temp)//3,replace=False)
# if (any(retrieval_keyword in data[i]["query"][0] for retrieval_keyword in retrieval_keywords)) or (any(retrieval_keyword in data[i]["pos"][0] for retrieval_keyword in retrieval_keywords)):# and ("duplicate" not in data[i]["pos"][0]):
#     datasets.append(temp)
# else:
#     temp = np.random.choice(temp,len(temp)//5,replace=False)
#     datasets.append(temp)
datasets.append(temp)
    # if (any(retrieval_keyword in data[i]["query"][0] for retrieval_keyword in retrieval_keywords)) or (any(retrieval_keyword in data[i]["pos"][0] for retrieval_keyword in retrieval_keywords)):# and ("duplicate" not in data[i]["pos"][0]):
    #     datasets.append(temp)
    # else:
    #     temp = np.random.choice(temp,len(temp)//5,replace=False)
    #     datasets.append(temp)
# lens.append(j-i)
datasets = list(itertools.chain.from_iterable(datasets))
print(len(datasets))
with jsonlines.open(output_file,'w') as writer:
    for item in datasets:
        writer.write(item)
