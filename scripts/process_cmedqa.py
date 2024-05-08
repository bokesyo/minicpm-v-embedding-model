import pandas as pd
import jsonlines


corpus_file = '../../dataset/cMedQA2/answer.csv'
query_file = '../../dataset/cMedQA2/question.csv'
qrels_file = '../../dataset/cMedQA2/train_candidates.txt'
corpus_df = pd.read_csv(corpus_file, sep=',')
query_df = pd.read_csv(query_file, sep=',')
qrels_df = pd.read_csv(qrels_file, sep=',')
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
DOC_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
datas = []
# print(corpus_df)
query_dic = query_df.set_index("question_id")["content"].to_dict()
corpus_dic = corpus_df.set_index("ans_id")["content"].to_dict()
undedup_df = pd.read_csv(qrels_file, sep=',')
qrels_df.drop_duplicates(subset="question_id", keep="first",inplace=True)
unde_i = 0
for i in range(len(qrels_df)):
    pos_id = qrels_df.iloc[i, 1] # docid
    neg_ids = []
    
    
    query_id = qrels_df.iloc[i, 0] # question
    unde_query_id = undedup_df.iloc[unde_i,0]
    while query_id == unde_query_id:
        neg_ids.append(undedup_df.iloc[unde_i,2])
        unde_i += 1
        if unde_i == len(undedup_df):
            break
        unde_query_id = undedup_df.iloc[unde_i,0]
    
    # print(i,unde_i)
    
    pos = corpus_dic[pos_id]
    negs = [corpus_dic[neg_id] for neg_id in neg_ids][:10]
    query = query_dic[query_id]
    data = {"query" : [QUERY_INSTRUCTION, query], "pos" : [DOC_INSTRUCTION, pos], "neg" : [DOC_INSTRUCTION, *negs]}
    datas.append(data)
    
output_file = '../../dataset/our-zh/hard_negs/cMedQA2.jsonl'
with jsonlines.open(output_file, mode='w') as writer:
    for data in datas:
        writer.write(data)


