import json
import os

BASE_PATH = '/home/jeeves/openmatch/Research/ArguAna/'
# 读取并存储queries.jsonl中的_id
queries_ids = set()
with open(os.path.join(BASE_PATH, 'queries.jsonl'), 'r') as f:
    for line in f:
        query = json.loads(line)
        queries_ids.add(query['_id'])

# 创建一个新的corpus.jsonl文件，过滤掉与queries.jsonl中_id重复的行
with open(os.path.join(BASE_PATH, 'corpus.jsonl'), 'r') as corpus_file, open(os.path.join(BASE_PATH, 'corpus_deduped.jsonl'), 'w') as new_corpus_file:
    for line in corpus_file:
        corpus_item = json.loads(line)
        if corpus_item['_id'] not in queries_ids:
            new_corpus_file.write(line)

# 如果一切顺利，新的无重复_id的corpus文件已经保存在new_corpus.jsonl中
