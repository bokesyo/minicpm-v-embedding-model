import json
import os
import random
random.seed(42)

# Assuming the JSONL files are located in a directory named "jsonl_directory"
jsonl_directory = "/home/jeeves/MEDI2"
output_directory = "/home/jeeves/medi2_jsonl"
output_file_path = os.path.join(output_directory, "train.jsonl")

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# List all JSONL files in the directory
jsonl_files = [file for file in os.listdir(jsonl_directory) if file.endswith('.jsonl')]

# Initialize a list to hold all modified records
all_records = []

# Process each JSONL file
for file_index, jsonl_file in enumerate(jsonl_files):
    print(file_index, jsonl_file)
    file_path = os.path.join(jsonl_directory, jsonl_file)
    
    # Open and process each file
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Load the JSON line
            record = json.loads(line)
            
            # Modify the record
            # Remove the outer list from 'pos' and 'neg', and add 'task_id'
            record['pos'] = record['pos'][0]
            record['neg'] = record['neg'][0]
            record['task_id'] = 10000 + file_index
            
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

