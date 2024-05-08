EMBEDDING_DIR=/home/jeeves
CHECKPOINT_DIR=/home/jeeves
COLLECTION_DIR=/home/jeeves

CUDA_VISIBLE_DEVICES=0 python -m openmatch.driver.build_index  \
    --output_dir $EMBEDDING_DIR/msmarco/bert_epoch2/  \
    --model_name_or_path $CHECKPOINT_DIR/bert-base-embedding-epoch2-180000  \
    --per_device_eval_batch_size 512  \
    --corpus_path $COLLECTION_DIR/marco/corpus.tsv  \
    --doc_template "<title><SEP><text>"  \
    --doc_column_names id,title,text  \
    --q_max_len 32  \
    --p_max_len 128  \
    --fp16  \
    --dataloader_num_workers 1 \
    --max_inmem_docs 1000000
