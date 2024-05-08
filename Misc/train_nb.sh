WORLD_SIZE=1
RANK=0
GPUS_PER_NODE=2
MASTER_ENDPOINT=localhost
MASTER_PORT=23456
CHECKPOINT_DIR=/home/jeeves/checkpoints
MODEL_NAME=test-model
# MODEL_PATH=/home/jeeves/cpm_d-2b_with_pad_token
# DATASET_PATH=/home/jeeves/msmarco_cpmd_2b_tokens_tiny
MODEL_PATH=/home/jeeves/bert-base-uncased-small
DATASET_PATH=/home/jeeves/msmarco_prototype_tokens_tiny

LOG_DIR=/home/jeeves/logs

LR=5e-6
# LR=1e-4
SOFTMAX_TEMPERATURE=0.2
PER_DEV_BATCH_SIZE=2 # full-parameter, cpmd 2p4b
# PER_DEV_BATCH_SIZE=8 # lora, cpmd 2p4b
# LORA=true
LORA=false
LORA_R=32

TIMESTR=$(date "+%Y-%m-%d-%H%M%S")\
DEEPSPEED="ds_config.json"

torchrun \
    --nnodes=$WORLD_SIZE \
    --node_rank=$RANK \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ENDPOINT \
    --master_port=$MASTER_PORT \
    src/openmatch/driver/train_dr.py \
    --overwrite_output_dir \
    --output_dir "$CHECKPOINT_DIR/$MODEL_NAME" \
    --model_name_or_path $MODEL_PATH \
    --do_train  \
    --save_steps 5000  \
    --train_path "$DATASET_PATH/train.jsonl" \
    --bf16  \
    --per_device_train_batch_size $PER_DEV_BATCH_SIZE  \
    --train_n_passages 8  \
    --learning_rate $LR  \
    --q_max_len 32  \
    --p_max_len 256  \
    --num_train_epochs 1 \
    --logging_dir "$LOG_DIR/$TIMESTR-$LR-bsz-$PER_DEV_BATCH_SIZE-temp$SOFTMAX_TEMPERATURE" \
    --logging_steps 10 \
    --softmax_temperature $SOFTMAX_TEMPERATURE \
    --negatives_x_device \
    --inbatch_loss false \
    --lr_scheduler_type "constant" \
    --lora $LORA \
    --lora_r $LORA_R \
    --deepspeed $DEEPSPEED

--max_steps 100 \
# --num_train_epochs 1  \