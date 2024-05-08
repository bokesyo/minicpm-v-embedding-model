
EMBEDDING_DIR=/home/jeeves
CHECKPOINT_DIR=/home/jeeves
RESULT_DIR=/home/jeeves
COLLECTION_DIR=/home/jeeves

python -m openmatch.driver.retrieve  \
    --output_dir $EMBEDDING_DIR/msmarco-bert-base-epoch1-embedding/  \
    --model_name_or_path $CHECKPOINT_DIR/msmarco-bert-base-output/checkpoint-140000  \
    --per_device_eval_batch_size 256  \
    --query_path $COLLECTION_DIR/marco/train.query.txt  \
    --query_template "<text>"  \
    --query_column_names id,text  \
    --q_max_len 32  \
    --fp16  \
    --trec_save_path $RESULT_DIR/msmarco-bert-base-epoch1-embedding-retrieval/train.trec \
    --dataloader_num_workers 1 \
    --use_gpu


