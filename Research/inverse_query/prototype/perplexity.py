# import lmppl

# MODEL_PATH = "/home/jeeves/cpm_d-2b_with_pad_token"
# scorer = lmppl.LM(MODEL_PATH)

# text = [
#     'sentiment classification: I dropped my laptop on my knee, and someone stole my coffee. I am happy.',
#     'sentiment classification: I dropped my laptop on my knee, and someone stole my coffee. I am sad.'
# ]

# ppl = scorer.get_perplexity(text)

# print(list(zip(text, ppl)))

# >>> [
#   ('sentiment classification: I dropped my laptop on my knee, and someone stole my coffee. I am happy.', 136.64255272925908),
#   ('sentiment classification: I dropped my laptop on my knee, and someone stole my coffee. I am sad.', 139.2400838400971)
# ]
# print(f"prediction: {text[ppl.index(min(ppl))]}")
# >>> "prediction: sentiment classification: I dropped my laptop on my knee, and someone stole my coffee. I am happy."

# load a dataset from a json, List[str], and calculate their ppl using batch
# if __name__ == "__main__":
#     import json
#     from typing import List
#     from lmppl import LM

#     with open("path/to/json") as f:
#         data = json.load(f)

#     # data = ["I am happy", "I am sad"]
#     lm = LM("path/to/model")
    
#     # here manually split batch
#     BATCH_SIZE = 1000
#     ppls = []
#     n_batches = len(data) // BATCH_SIZE
#     for batch_idx in range(n_batches):
#         batch = data[batch_idx * BATCH_SIZE: min(len(data), (batch_idx + 1) * BATCH_SIZE)]
#         ppl = lm.get_perplexity(batch)
#         ppls.extend(ppl)
#         print(ppl)
    
#     # save the ppls to a file
#     with open("path/to/save/ppl.json", "w") as f:
#         json.dump(ppls, f)


import os
import argparse
from lmppl import LM
import json

# use argparse to run the script
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--save_path", type=str, default="ppl.json")

    args = parser.parse_args()
    lm = LM(args.model_path)
    # with open(args.data_path) as f:
    #     data = json.load(f)
    all_data = []
    all_data_lens = []
    # i need to load all json in this path
    for json_file in os.listdir(args.data_path):
        with open(json_file) as f:
            data = json.load(f)
            all_data.extend(data)
            all_data_lens.append(len(data))
    print(f"Total data: {len(all_data)}")

    ppls = []
    n_batches = len(data) // args.batch_size
    for batch_idx in range(n_batches):
        batch = data[batch_idx * args.batch_size: min(len(data), (batch_idx + 1) * args.batch_size)]
        ppl = lm.get_perplexity(batch)
        ppls.extend(ppl)
        print(ppl)
    
    # save the ppls to a file
    with open(args.save_path, "w") as f:
        json.dump(ppls, f)

# python Research/Dataset/reverse_query/perplexity.py --model_path /home/jeeves/cpm_d-2b_with_pad_token --data_path /home/jeeves/Downloads/queries.json --batch_size 1000 --save_path /home/jeeves/Downloads/ppl.json

# now write a new script to filter out the queries with high perplexity and save low ppl data to a new file
# import json
# import numpy as np

# if __name__ == "__main__":
#     with open("/home/jeeves/Downloads/queries.json") as f:
#         data = json.load(f)
    
#     with open("/home/jeeves/Downloads/ppl.json") as f:
#         ppls = json.load(f)
    
#     ppls = np.array(ppls)
#     low_ppl_idx = np.where(ppls < 100)[0]
#     low_ppl_data = [data[i] for i in low_ppl_idx]
    
#     with open("/home/jeeves/Downloads/low_ppl_queries.json", "w") as f:
#         json.dump(low_ppl_data, f)

# good, use argparse
import argparse
import json
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--ppl_path", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=100)
    parser.add_argument("--save_path", type=str, default="low_ppl_queries.json")
    args = parser.parse_args()

    with open(args.data_path) as f:
        data = json.load(f)
    
    with open(args.ppl_path) as f:
        ppls = json.load(f)
    
    ppls = np.array(ppls)
    low_ppl_idx = np.where(ppls < args.threshold)[0]
    low_ppl_data = [data[i] for i in low_ppl_idx]
    
    with open(args.save_path, "w") as f:
        json.dump(low_ppl_data, f)


# python Research/Dataset/reverse_query/filter_low_ppl.py --data_path /home/jeeves/Downloads/queries.json --ppl_path /home/jeeves/Downloads/ppl.json --threshold 100 --save_path /home/jeeves/Downloads/low_ppl_queries.json

