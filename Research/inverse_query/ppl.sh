
WORLD_SIZE=1
RANK=0
GPUS_PER_NODE=1
PER_DEV_BATCH_SIZE=16

BASE_DIR=/home/jeeves/openmatch/Research/inverse_query

# source ~/vision/bin/activate
cd $BASE_DIR

DATASET_PATH="/home/jeeves/longdoc-pdf-chunks-limited-1024-2024-04-11-152440"
MODEL_PATH="/home/jeeves/cpm_d-2b_with_pad_token"
MODEL_PATH=""

torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    ppl.py \
    --data_dir $DATASET_PATH \
    --model_name_or_path $MODEL_PATH \
    --output_dir "$DATASET_PATH-ppl" \
    --per_device_eval_batch_size $PER_DEV_BATCH_SIZE \
    --dataloader_num_workers 1 \
    --fp16 \
    --overwrite_output_dir true \

