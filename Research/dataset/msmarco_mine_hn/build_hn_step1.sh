WORKDIR=/home/jeeves/mine_marco_2

CUDA_VISIBLE_DEVICES=0 python -m openmatch.driver.build_index  \
    --output_dir "$WORKDIR/embeddings"  \
    --model_name_or_path "/home/jeeves/bge-base-en-v1.5" \
    --per_device_eval_batch_size 128  \
    --corpus_path /home/jeeves/marco/corpus.tsv  \
    --doc_template "<title> <text>"  \
    --doc_column_names id,title,text  \
    --q_max_len 512  \
    --p_max_len 512  \
    --fp16  \
    --dataloader_num_workers 1 \
    --max_inmem_docs 1000000 \
    --normalize true \

# --query_template "Represent this sentence for searching relevant passages: <text>" \
    