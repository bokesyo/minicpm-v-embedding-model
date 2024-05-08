PLM_DIR=/home/jeeves
COLLECTION_DIR=/home/jeeves
PROCESSED_DIR=/home/jeeves
LOG_DIR=/home/jeeves
CHECKPOINT_DIR=/home/jeeves

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 src/openmatch/driver/train_dr.py  \
    --output_dir $CHECKPOINT_DIR/msmarco-cpm-d-2b-output  \
    --overwrite_output_dir \
    --model_name_or_path $PLM_DIR/cpm_d-2b  \
    --do_train  \
    --save_steps 20000  \
    --train_path $PROCESSED_DIR/msmarco-cpm_d-2b-epoch1-data/train.jsonl  \
    --fp16  \
    --per_device_train_batch_size 2  \
    --train_n_passages 8  \
    --learning_rate 5e-6  \
    --q_max_len 32  \
    --p_max_len 128  \
    --num_train_epochs 1  \
    --logging_dir $LOG_DIR/msmarco-cpm-d-2b-base-log

# cat *.jsonl > train.jsonl
