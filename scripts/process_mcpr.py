import pandas as pd
import jsonlines

subsets = ['ecom','medical','video']
for subset in subsets:
    corpus_file = '../../dataset/Multi-CPR-main/data/{}/corpus.tsv'.format(subset)
    query_file = '../../dataset/Multi-CPR-main/data/{}/train.query.txt'.format(subset)
    qrels_file = '../../dataset/Multi-CPR-main/data/{}/qrels.train.tsv'.format(subset)

    if subset == 'medical':
        corpus_files = ['../../dataset/Multi-CPR-main/data/{}/corpus_split_'.format(subset)+ str(i) + ".tsv" for i in range(1,5)]
    else:
        corpus_files = [corpus_file]
    corpus_df = pd.read_csv(corpus_files[0], sep='\t', header=None)
    for corpus_file in corpus_files[1:]:
        # pd.read_csv(corpus_file, sep='\t', header=None)
        corpus_df = pd.concat([corpus_df,pd.read_csv(corpus_file, sep='\t', header=None)],axis=0)
    query_df = pd.read_csv(query_file, sep='\t', header=None)
    qrels_df = pd.read_csv(qrels_file, sep='\t', header=None)

    datas = []
    # querydf to dicf
    query_dic = query_df.set_index(0)[1].to_dict()
    
    # print(corpus_df)
    
    corpus_dic = corpus_df.set_index(0)[1].to_dict()

    for i in range(len(qrels_df)):
        query_id = qrels_df.iloc[i, 0]
        doc_id = qrels_df.iloc[i, 2]
        try:
            query = query_dic[query_id]
            doc = corpus_dic[doc_id]
            data = {
                'query': query,
                'pos': doc,
            }
            datas.append(data)
        except:
            continue
        
    output_file = '../../dataset/our-zh_raw/mcpr-{}.jsonl'.format(subset)
    with jsonlines.open(output_file, mode='w') as writer:
        for data in datas:
            writer.write(data)


