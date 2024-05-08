PLM_DIR=/home/jeeves
COLLECTION_DIR=/home/jeeves
PROCESSED_DIR=/home/jeeves

python scripts/msmarco/build_train.py \
    --tokenizer_name $PLM_DIR/cpm_d-2b  \
    --negative_file $COLLECTION_DIR/marco/train.negatives.tsv  \
    --qrels $COLLECTION_DIR/marco/qrels.train.tsv  \
    --queries $COLLECTION_DIR/marco/train.query.txt  \
    --collection $COLLECTION_DIR/marco/corpus.tsv  \
    --save_to $PROCESSED_DIR/msmarco-cpm_d-2b-dataset/  \
    --doc_template "<title> <text>"

