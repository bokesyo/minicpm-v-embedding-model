import json

def process_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            record = json.loads(line)
            
            # Skip if winner is 'tie'
            if record['winner'] == 'tie':
                continue
            
            # Extract conversations
            conversation_a = record['conversation_a'][1]['content']
            conversation_b = record['conversation_b'][1]['content']
            query = record['conversation_a'][0]['content']  # Or conversation_b[0]['content']
            
            # Assign pos and neg based on the winner
            if record['winner'] == 'model_a':
                pos, neg = conversation_a, conversation_b
            else:
                pos, neg = conversation_b, conversation_a
            
            # Construct the dictionary
            result_dict = {
                'query': ["Represent user's question for searching satisfying answer;", query],
                'pos': ["Represent this answer for searching satisfying answer to a query;", pos],
                'neg': ["Represent this answer for searching satisfying answer to a query;", neg],
                'task_id': 1001
            }
            
            # Write to output file
            json.dump(result_dict, outfile)
            outfile.write('\n')

# Example usage
input_file = '/home/jeeves/dpo/chatbot_arena.jsonl'
output_file = '/home/jeeves/dpo/chatbot_arena.medi.jsonl'
process_jsonl(input_file, output_file)
