torchrun --nproc_per_node 1 \
    -m training.run \
    --output_dir /home/jeeves/gritlm_test \
    --model_name_or_path /home/jeeves/cpm_d-2b_with_pad_token \
    --train_path /home/jeeves/medi2-data-jsonl/train.jsonl \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --dataloader_num_workers 1 \
    --use_mapping_dataset false \
    --query_instruction true \
    --corpus_instruction true \
    --q_max_len 512  \
    --p_max_len 512  \
    --normalized \
    --temperature 0.02 \
    --train_n_passages 2  \
    --negatives_cross_device \
    --mode embedding \
    --logging_steps 1 \
    --num_train_epochs 1 \
    --bf16 \
    --pooling_method lasttoken \
    --attn cccc \
    --save_steps 2500 \
    --gradient_checkpointing \
    --attn_implementation flash_attention_2

#     --max_steps 1253 \
    # --attn_implementation sdpa \