import pandas as pd
import jsonlines


corpus_file = '../../dataset/NCP/NCPPolicies_context_20200301.csv'
qrels_file = '../../dataset/NCP/NCPPolicies_train_20200301.csv'
# corpus_df = pd.read_csv(corpus_file, sep='\t')
qrels_df = pd.read_csv(qrels_file, sep='\t')
datas = []
# print(corpus_df)

# corpus_dic = corpus_df.set_index("docid")["text"].to_dict()
# \t in doc, cannot use pandas
# read csv by line
corpus_dic = {}
for line in open(corpus_file):
    line = line.strip()
    if line:
        line_key = line[:line.find('\t')]
        line_value = line[line.find('\t')+1:]
        corpus_dic[line_key] = line_value
for i in range(len(qrels_df)):
    doc_id = qrels_df.iloc[i, 1] # docid
    query = qrels_df.iloc[i, 2] # question
    doc = corpus_dic[doc_id]
    data = {
        'query': query,
        'pos': doc,
    }
    datas.append(data)
    
output_file = '../../dataset/our-zh_raw/NCPPolicies.jsonl'
with jsonlines.open(output_file, mode='w') as writer:
    for data in datas:
        writer.write(data)


