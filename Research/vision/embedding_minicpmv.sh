
WORLD_SIZE=1
RANK=0
GPUS_PER_NODE=1
PER_DEV_BATCH_SIZE=4

BASE_DIR=/home/jeeves/openmatch/Research/vision

# PROMPT_PATH=$BASE_DIR/prompt_en_multi_minicpmv.txt
# PROMPT_PATH=$BASE_DIR/prompt_en_multi_fewshot.txt

MODEL_PATH=/home/jeeves/MiniCPM-V-2.0

# source ~/vision/bin/activate
cd $BASE_DIR

# echo "prompt path = $PROMPT_PATH"

# PROMPT=$(cat $PROMPT_PATH)
# echo "Checking:: $PROMPT"

torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    visual_embedding_test.py \
    --data_dir "/home/jeeves/visual_embedding_2_long_visual_dataset_jsonl_merged" \
    --model_name_or_path $MODEL_PATH \
    --per_device_eval_batch_size $PER_DEV_BATCH_SIZE \
    --dataloader_num_workers 1 \
    --overwrite_output_dir true \
    --output_dir "/home/jeeves/tmp" \
    # --prompt_path $PROMPT_PATH \
    # --bf16 \
    

# cluster: pip install timm==0.9.10