# on each node, the script will only run once.
MAX_SEQ_LEN=$1
PER_DEV_BATCH_SIZE=$2
SOFTMAX_TEMPERATURE=$3
EPOCH=$4
QUERY_INSTRUCTION=$5 # bool
CORPUS_INSTRUCTION=$6 # bool
DEEPSPEED=$7 # ds_config.json or ds_config_warmup_decay.json or false
LR=$8
CHECKPOINT_MODEL=$9

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

# SOFTMAX_TEMPERATURE=0.02
# PER_DEV_BATCH_SIZE=4
# TRAIN_STEPS=5000
# LR=5e-6

# LR=5e-6 # full-paramter learning rate
# LR=1e-4 # LoRA learning rate
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


if [ -n "$CHECKPOINT_MODEL" ]; then # a checkpoint model
  MODEL_PATH="$CHECKPOINT_DIR/$CHECKPOINT_MODEL"
  MODEL_REAL_NAME="ckpt-inherit-$MODEL_REAL_NAME"
  echo "from checkpoint $CHECKPOINT_MODEL, model path = $MODEL_PATH"
else # no checkpoint model
  echo "from scratch"
fi


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



echo "===== mkdir tensorboard ========"
# mkdir "$CHECKPOINT_DIR/tensorboard"
# cd $CHECKPOINT_DIR
# echo "ls:"
# ls
# echo "------"



echo "======== Installing openmatch: =========="

# cd cd /local/apps/openmatch/jeeves
# tar zxvf transformers-4.37.2.tar.gz
# cd transformers-4.37.2
# pip install -e .

# echo "transformers setup succeed!"

# pip install -U transformers 
pip install -U transformers[deepspeed] # 4.37.2
echo "transformers[deepspeed] setup succeed!"

pip install -U accelerate
echo "accelerate setup succeed!"

pip install -U datasets
echo "datasets setup succeed!"
# add pad token
# added to model file

cd GradCache-main
pip install -e .
cd ..

cd /local/apps/openmatch
pip install .
echo "openmatch setup succeed!"

# cd pytrec_eval-0.5
# pip install . # here do not use -e .
# echo "pytrec_eval setup succeed!"
# cd ..

# cd sentence-transformers-2.3.1
# pip install .
# echo "sentence-transformers setup succeed!"
# cd ..

# cd NouamaneTazi-beir-57cd308
# pip install .
# echo "beir setup succeed!"
# cd ..

# pip install evaluate
# echo "evaluate setup succeed!"

echo "======== Train begin: =========="

# --master_addr=$MASTER_ENDPOINT \
# --master_port=$MASTER_PORT \

# torchrun will launch nproc_per_node process with --rank --world_size --master_port
torchrun \
    --nnodes=$WORLD_SIZE \
    --node_rank=$RANK \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_id=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    src/openmatch/driver/train_dr.py \
    --overwrite_output_dir \
    --output_dir $EXPORT_DIR\
    --model_name_or_path $MODEL_PATH \
    --do_train  \
    --save_steps 2500  \
    --train_path "$DATASET_PATH/train.jsonl" \
    --bf16  \
    --per_device_train_batch_size $PER_DEV_BATCH_SIZE  \
    --train_n_passages 2  \
    --learning_rate $LR  \
    --q_max_len $MAX_Q_LEN  \
    --p_max_len $MAX_P_LEN  \
    --num_train_epochs $EPOCH  \
    --logging_dir "$LOG_DIR/$IDENTITY" \
    --negatives_x_device \
    --softmax_temperature $SOFTMAX_TEMPERATURE \
    --logging_steps 10 \
    --inbatch_loss $IN_BATCH \
    --lora $LORA \
    --lora_r $LORA_R \
    --deepspeed $DEEPSPEED \
    --gradient_checkpointing true \
    --dataloader_num_workers 1 \
    --use_mapping_dataset true \
    --save_safetensors false \
    --query_instruction $QUERY_INSTRUCTION \
    --corpus_instruction $CORPUS_INSTRUCTION \
    # --seed 42 \
    # --lr_scheduler_type $lr_scheduler_type \

# --max_steps $TRAIN_STEPS \
# negatives_x_device wouldn't cost much computation and time.

# --num_train_epochs 1  \