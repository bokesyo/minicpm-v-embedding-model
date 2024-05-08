import jsonlines
import itertools

input_file = "../../dataset/our-zh/train.jsonl"
input_file = "/data/WorkSpace/openbmb/dataset/medi2-data-jsonl/hard_negs/clean_03negs.jsonl"
batch_size = 256
output_file = "../../dataset/our-zh/train_batch_{}.jsonl".format(batch_size)
output_file = "/data/WorkSpace/openbmb/dataset/medi2-data-jsonl/hard_negs/clean_03negs.jsonl"

data = []
with jsonlines.open(input_file) as reader:
    for obj in reader:
        data.append(obj)
        
# print(len(data))

# print(data[0])
# print(data[-1])
data.sort(key=lambda x:x["task_id"])
# print(data[0])
# print(data[-1])
import numpy as np
np.random.seed(42)
i,j = 0,0
set_num = 1

blockes = []
ends = []
datasets = []
# lens = []

while j < len(data):
    if data[i]["task_id"] != data[j]["task_id"]:
        temp = data[i:j]
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
datasets.append(temp)
# lens.append(j-i)

for d in datasets:
    np.random.shuffle(d)
    s = 0
    while s + batch_size < len(d):
        blockes.append(d[s:s+batch_size])
        s += batch_size
    
np.random.shuffle(blockes)
blockes = list(itertools.chain.from_iterable(blockes))
# blockes = sum(blockes,[])
# print(len(blockes))
# print(blockes[0],blockes[31],blockes[32])
# print(len(datasets))
# print(min(lens),max(lens))
# print(len([i  for i in lens if i < 32]))
with jsonlines.open(output_file,'w') as writer:
    for item in blockes:
        writer.write(item)
