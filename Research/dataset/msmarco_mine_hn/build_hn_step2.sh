# python 3.8
# pip uninstall faiss-cpu
# pip install faiss-gpu

WORKDIR=/home/jeeves/mine_marco_2

CUDA_VISIBLE_DEVICES=0 python -m openmatch.driver.retrieve  \
    --output_dir $WORKDIR/embeddings/  \
    --model_name_or_path "/home/jeeves/bge-base-en-v1.5" \
    --per_device_eval_batch_size 128  \
    --query_path "/home/jeeves/marco/train.query.txt"  \
    --query_template "Represent this sentence for searching relevant passages: <text>"  \
    --query_column_names id,text  \
    --q_max_len 512  \
    --fp16  \
    --trec_save_path $WORKDIR/results/train.trec  \
    --dataloader_num_workers 1 \
    --use_gpu \
    --normalize true \
