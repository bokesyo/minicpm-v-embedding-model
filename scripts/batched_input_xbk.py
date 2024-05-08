import jsonlines
import itertools

BATCH_SIZE = 1024

input_file = "/home/jeeves/medi-data-jsonl/train.jsonl"
output_file = f"/home/jeeves/medi-data-jsonl/train_batch_{BATCH_SIZE}.jsonl"
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



cnt = 0
for d in datasets:
    cnt += 1
    if BATCH_SIZE > len(d):
        print(f"{cnt}: this dataset is too small")
        continue
    else:
        print(cnt)
    np.random.shuffle(d)
    s = 0
    while s + BATCH_SIZE < len(d):
        blockes.append(d[s:s+BATCH_SIZE])
        s += BATCH_SIZE
    
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
