COLLECTION_DIR=/home/jeeves
PROCESSED_DIR=/home/jeeves
RESULT_DIR=/home/jeeves
PLM_DIR=/home/jeeves

python scripts/msmarco/build_hn.py  \
    --tokenizer_name $PLM_DIR/bert-base-uncased  \
    --hn_file $RESULT_DIR/msmarco-bert-base-epoch1-embedding-retrieval/train.trec  \
    --qrels $COLLECTION_DIR/marco/qrels.train.tsv  \
    --queries $COLLECTION_DIR/marco/train.query.txt  \
    --collection $COLLECTION_DIR/marco/corpus.tsv  \
    --save_to $PROCESSED_DIR/msmarco-bert-base-epoch1-embedding-retrieval  \
    --doc_template "<title><SEP><text>"

