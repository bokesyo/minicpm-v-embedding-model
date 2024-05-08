
WORLD_SIZE=1
RANK=0
GPUS_PER_NODE=1
PER_DEV_BATCH_SIZE=16

BASE_DIR=/home/jeeves/openmatch/Research/vision

# source ~/vision/bin/activate
cd $BASE_DIR

PROMPT_PATH=$BASE_DIR/prompt_en_multi.txt
echo "prompt path = $PROMPT_PATH"

# PROMPT=$(cat $PROMPT_PATH)
# echo "Checking:: $PROMPT"

/home/jeeves/vision/bin/python -m torch.distributed.launch \
    --use-env \
    --nproc_per_node=$GPUS_PER_NODE \
    image_inverse_query_qwen.py \
    --data_dir "/home/jeeves/visual_embedding_1_dataset_jsonl_merged" \
    --model_name_or_path "/home/jeeves/Qwen-VL-Chat" \
    --output_dir "/home/jeeves/visual_embedding_1_dataset_reversed" \
    --prompt_path $PROMPT_PATH \
    --per_device_eval_batch_size $PER_DEV_BATCH_SIZE \
    --dataloader_num_workers 1 \
    --fp16 \
    --overwrite_output_dir true \

