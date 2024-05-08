import json
import os
import random
random.seed(42)

# Assuming the JSONL files are located in a directory named "jsonl_directory"
jsonl_raw_path = "/home/jeeves/mine_marco_2/train_hn/train.raw.hn.jsonl"
output_directory = "/home/jeeves/msmarco_hn_bge_en_15"
output_file_path = os.path.join(output_directory, "train.jsonl")

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
CORPUS_INSTRUCTION = "Represent this passage for searching relevant queries: "

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Initialize a list to hold all modified records
all_records = []


file_path = os.path.join(jsonl_raw_path)
    
# Open and process each file
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # Load the JSON line
        record = json.loads(line)
        
        # Modify the record
        # Remove the outer list from 'pos' and 'neg', and add 'task_id'
        record['query'] = [QUERY_INSTRUCTION, record['query']]
        record['pos'] = [CORPUS_INSTRUCTION, record['pos'][0]]
        record['neg'] = [CORPUS_INSTRUCTION, record['neg'][0]]
        record['task_id'] = 2000
        all_records.append(record)

# Shuffle the list of all records
print("shuffling records...")
random.shuffle(all_records)

print("saving records...")
# Write the modified records to a new JSONL file
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    for record in all_records:
        # Convert the record to a JSON string and write it to the file
        json_line = json.dumps(record)
        output_file.write(json_line + '\n')

