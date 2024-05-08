import json

def process_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)
            
            # Extract the initial prompt as query
            query = data['prompt']
            
            # Extract chosen and rejected conversation contents
            pos_content = data['chosen'][1]['content']  # Assuming the second item contains the response
            neg_content = data['rejected'][1]['content']  # Assuming the second item contains the response
            
            # Construct the dictionary
            result_dict = {
                'query': ["Represent user's question for searching satisfying answer;", query],
                'pos': ["Represent this answer for searching satisfying answer to a query;", pos_content],
                'neg': ["Represent this answer for searching satisfying answer to a query;", neg_content],
                'task_id': 1002
            }
            
            # Write to output file
            json.dump(result_dict, outfile)
            outfile.write('\n')


# Example usage
input_file = '/home/jeeves/dpo/ultrafeedback.dev.jsonl'
output_file = '/home/jeeves/dpo/ultrafeedback.dev.post.jsonl'
process_jsonl(input_file, output_file)

