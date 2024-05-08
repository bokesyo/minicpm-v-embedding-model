# on each node, the script will only run once.
# on each node, the script will only run once.
MAX_SEQ_LEN=$1
PER_DEV_BATCH_SIZE=$2
SOFTMAX_TEMPERATURE=$3
EPOCH=$4
QUERY_INSTRUCTION=$5 # bool
CORPUS_INSTRUCTION=$6 # bool
DEEPSPEED=$7 # ds_config.json or ds_config_warmup_decay.json or false
LR=$8
MAPPING=$9 # stream data
CHECKPOINT_MODEL=${10}


echo "======== Hostname: ========="

echo "IP Addr: $(hostname -I)"
echo "Hostname: $(hostname)"
echo "RANK: $RANK"
echo "Master addr: $MASTER_ENDPOINT"
TIMESTR=$(date "+%Y-%m-%d-%H%M%S")
echo "Time: $TIMESTR"
echo "WORLD_SIZE: $WORLD_SIZE"

MASTER_PORT=23456
GPUS_PER_NODE=8
# WORLD_SIZE=1

lr_scheduler_type="constant" # no use if with deepspeed
IN_BATCH=true

LORA=false
LORA_R=32
MAX_Q_LEN=$MAX_SEQ_LEN
MAX_P_LEN=$MAX_SEQ_LEN
# DEEPSPEED=false
# DEEPSPEED="ds_config.json"
echo "DEEPSPEED=$DEEPSPEED"

echo "======== Hyperparameters: =========="
echo "Learning rate:  $LR"
echo "======== Config ==========="
echo $(python -c "import json; import os; config_string = open(os.environ['PLATFORM_CONFIG_PATH'], 'r').read(); config = json.loads(config_string); print(config)")
echo "======== Config Path: =========="
cp "$PLATFORM_CONFIG_PATH" /data/results/

MODEL_PATH=$(python -c "import json; import os; config_string = open(os.environ['PLATFORM_CONFIG_PATH'], 'r').read(); config = json.loads(config_string); print(list(config['model_map'].values() )[0] )")
DATASET_PATH=$(python -c "import json; import os; config_string = open(os.environ['PLATFORM_CONFIG_PATH'], 'r').read(); config = json.loads(config_string); print(list(config['dataset_map'].values())[0] )")
CHECKPOINT_DIR=$(python -c "import json; import os; config_string = open(os.environ['PLATFORM_CONFIG_PATH'], 'r').read(); config = json.loads(config_string); print(config['export_map']['checkpoints_dir'] )")
MODEL_GROUP_NAME=$(python -c "import json; import os; config_string = open(os.environ['PLATFORM_CONFIG_PATH'], 'r').read(); config = json.loads(config_string); print(list(config['model_map'].keys())[0] )")

MODEL_REAL_NAME=$(python -c "import json; import os; config_string = open(os.environ['PLATFORM_CONFIG_PATH'], 'r').read(); config = json.loads(config_string); print(list(config['model_map'].values())[0].split('/')[-2] )")

DATASET_REAL_NAME=$(python -c "import json; import os; config_string = open(os.environ['PLATFORM_CONFIG_PATH'], 'r').read(); config = json.loads(config_string); print(list(config['dataset_map'].values())[0].split('/')[-1] )")


# MODEL_OUTPUT_DIR=$(python -c "import json; import os; config_string = open(os.environ['PLATFORM_CONFIG_PATH'], 'r').read(); config = json.loads(config_string); print(config['export_map']['models_dir'] )")
RESULT_DIR=$(python -c "import json; import os; config_string = open(os.environ['PLATFORM_CONFIG_PATH'], 'r').read(); config = json.loads(config_string); print(config['export_map']['results_dir'] )")
LOG_DIR=$(python -c "import json; import os; config_string = open(os.environ['PLATFORM_CONFIG_PATH'], 'r').read(); config = json.loads(config_string); print(config['export_map']['logs_dir'] )")


# if [ -n "$CHECKPOINT_MODEL" ]; then # a checkpoint model
#   MODEL_PATH="$CHECKPOINT_DIR/$CHECKPOINT_MODEL"
#   MODEL_REAL_NAME="ckpt-inherit-$MODEL_REAL_NAME"
#   echo "from checkpoint $CHECKPOINT_MODEL, model path = $MODEL_PATH"
# else # no checkpoint model
#   echo "from scratch"
# fi


IDENTITY="$TIMESTR-model-$MODEL_REAL_NAME-data-$DATASET_REAL_NAME-lr-$LR-softm_temp-$SOFTMAX_TEMPERATURE-bsz$PER_DEV_BATCH_SIZE-ngpus$GPUS_PER_NODE-nnodes$WORLD_SIZE-inbatch-$IN_BATCH-nepoch-$EPOCH"

echo "IDENTITY=$IDENTITY"
EXPORT_DIR="$CHECKPOINT_DIR/$IDENTITY"

echo "======== Arguments: =========="
echo "EXPORT_DIR: $EXPORT_DIR"
echo "Model Path: $MODEL_PATH"
echo "Dataset Path: $DATASET_PATH"
echo "Checkpoint Path: $CHECKPOINT_DIR"
echo "Model REAL Name: $MODEL_REAL_NAME"
# echo "Model Output Dir: $MODEL_OUTPUT_DIR"
echo "Result Dir: $RESULT_DIR"
echo "Log Dir: $LOG_DIR"


echo "======== Installing openmatch: =========="

pip install -U transformers 
echo "transformers setup succeed!"

pip install -U accelerate
echo "accelerate setup succeed!"

pip install -U datasets
echo "datasets setup succeed!"
# add pad token
# added to model file

cd Library
cd GradCache
pip install -e .
cd ..
# Library


cd ..
# root

pip install .
echo "openmatch setup succeed!"



echo "======== Train begin: =========="

# --master_addr=$MASTER_ENDPOINT \
# --master_port=$MASTER_PORT \
cd /local/apps/openmatch/Research/gritlm/gritlm

# torchrun will launch nproc_per_node process with --rank --world_size --master_port
torchrun \
    --nnodes=$WORLD_SIZE \
    --node_rank=$RANK \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_id=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    -m training.run \
    --output_dir $EXPORT_DIR \
    --model_name_or_path $MODEL_PATH \
    --train_path "$DATASET_PATH/train.jsonl" \
    --learning_rate $LR \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --per_device_train_batch_size $PER_DEV_BATCH_SIZE \
    --gradient_accumulation_steps 1 \
    --dataloader_num_workers 1 \
    --use_mapping_dataset $MAPPING \
    --query_instruction $QUERY_INSTRUCTION \
    --corpus_instruction $CORPUS_INSTRUCTION \
    --q_max_len $MAX_SEQ_LEN  \
    --p_max_len $MAX_SEQ_LEN  \
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
    --attn_implementation flash_attention_2 \
    --logging_dir "$LOG_DIR/$IDENTITY" \
    --output_dir $EXPORT_DIR \




# torchrun --nproc_per_node 1 \
#     -m training.run \
#     --output_dir /home/jeeves/gritlm_test \
#     --model_name_or_path /home/jeeves/cpm_d-2b_with_pad_token \
#     --train_path /home/jeeves/medi2-data-jsonl/train.jsonl \
#     --learning_rate 2e-5 \
#     --lr_scheduler_type linear \
#     --warmup_ratio 0.03 \
#     --per_device_train_batch_size 32 \
#     --gradient_accumulation_steps 1 \
#     --dataloader_num_workers 1 \
#     --use_mapping_dataset false \
#     --query_instruction true \
#     --corpus_instruction true \
#     --q_max_len 512  \
#     --p_max_len 512  \
#     --normalized \
#     --temperature 0.02 \
#     --train_n_passages 2  \
#     --negatives_cross_device \
#     --mode embedding \
#     --logging_steps 1 \
#     --num_train_epochs 1 \
#     --bf16 \
#     --pooling_method mean \
#     --attn cccc \
#     --save_steps 2500 \
#     --gradient_checkpointing \
#     --attn_implementation flash_attention_2
