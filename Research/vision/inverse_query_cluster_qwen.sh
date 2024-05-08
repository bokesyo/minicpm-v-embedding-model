PROMPT_NAME=$1 # prompt_en_multi.txt

BASE_DIR=/local/apps/openmatch
BASE_DIR_THIS=$BASE_DIR/Research/vision

# VENV_NAME=vision
GPUS_PER_NODE=8
PER_DEV_BATCH_SIZE=8

echo "======== Hostname: ========="

echo "IP Addr: $(hostname -I)"
echo "Hostname: $(hostname)"
echo "RANK: $RANK"
echo "Master addr: $MASTER_ENDPOINT"
TIMESTR=$(date "+%Y-%m-%d-%H%M%S")
echo "Time: $TIMESTR"
echo "WORLD_SIZE: $WORLD_SIZE"


MODEL_PATH=$(python -c "import json; import os; config_string = open(os.environ['PLATFORM_CONFIG_PATH'], 'r').read(); config = json.loads(config_string); print(list(config['model_map'].values() )[0] )")
DATASET_PATH=$(python -c "import json; import os; config_string = open(os.environ['PLATFORM_CONFIG_PATH'], 'r').read(); config = json.loads(config_string); print(list(config['dataset_map'].values())[0] )")
CHECKPOINT_DIR=$(python -c "import json; import os; config_string = open(os.environ['PLATFORM_CONFIG_PATH'], 'r').read(); config = json.loads(config_string); print(config['export_map']['checkpoints_dir'] )")
MODEL_GROUP_NAME=$(python -c "import json; import os; config_string = open(os.environ['PLATFORM_CONFIG_PATH'], 'r').read(); config = json.loads(config_string); print(list(config['model_map'].keys())[0] )")
MODEL_REAL_NAME=$(python -c "import json; import os; config_string = open(os.environ['PLATFORM_CONFIG_PATH'], 'r').read(); config = json.loads(config_string); print(list(config['model_map'].values())[0].split('/')[-2] )")
DATASET_REAL_NAME=$(python -c "import json; import os; config_string = open(os.environ['PLATFORM_CONFIG_PATH'], 'r').read(); config = json.loads(config_string); print(list(config['dataset_map'].values())[0].split('/')[-1] )")

LOG_DIR=$(python -c "import json; import os; config_string = open(os.environ['PLATFORM_CONFIG_PATH'], 'r').read(); config = json.loads(config_string); print(config['export_map']['logs_dir'] )")

IDENTITY="inverse-query-qwenvl-$PROMPT_NAME-$TIMESTR"
EXPORT_DIR="$CHECKPOINT_DIR/$IDENTITY"



echo "======== Arguments: =========="
echo "EXPORT_DIR: $EXPORT_DIR"

echo "Model Path: $MODEL_PATH"
echo "Dataset Path: $DATASET_PATH"

echo "Checkpoint Path: $CHECKPOINT_DIR"
echo "Model REAL Name: $MODEL_REAL_NAME"

# echo "Model Output Dir: $MODEL_OUTPUT_DIR"
# echo "Result Dir: $RESULT_DIR"
echo "Log Dir: $LOG_DIR"


# make virtual environment
cd $BASE_DIR
# python -m venv $VENV_NAME
# PYTHON_PATH=$BASE_DIR/$VENV_NAME/bin

# install QWen-VL dependency
pip install -r $BASE_DIR_THIS/requirements.txt
echo "QWen-VL dependencies ok."

sudo pip uninstall torchvision -y
pip install $BASE_DIR_THIS/torchvision-0.14.1+cu117-cp310-cp310-linux_x86_64.whl

pip install -U accelerate
echo "accelerate setup succeed!"

pip install -U datasets
echo "datasets setup succeed!"

# install openmatch
pip install -e .
echo "openmatch ok."

# load prompt
PROMPT_PATH="$BASE_DIR_THIS/$PROMPT_NAME"

# PROMPT="$(cat $BASE_DIR_THIS/$PROMPT_PATH)"
# echo "Checking:: prompt=$PROMPT"

# launch multiple process with multi-gpus
torchrun \
    --nnodes=$WORLD_SIZE \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_id=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ENDPOINT:$MASTER_PORT \
    $BASE_DIR_THIS/image_inverse_query_qwen.py \
    --data_dir $DATASET_PATH \
    --model_name_or_path $MODEL_PATH \
    --output_dir $EXPORT_DIR \
    --prompt_path $PROMPT_PATH \
    --per_device_eval_batch_size $PER_DEV_BATCH_SIZE \
    --dataloader_num_workers 1 \
    --fp16 \
    --overwrite_output_dir true \

#     --use-env \


# no klara-hw
# torchrun \
    # --nnodes=$WORLD_SIZE \
    # --nproc_per_node=$GPUS_PER_NODE \
    # --rdzv_id=1 \
    # --rdzv_backend=c10d \
    # --rdzv_endpoint=$MASTER_ENDPOINT:$MASTER_PORT \

# klara-hw
# torchrun \
#     --nnodes=$WORLD_SIZE \
#     --nproc_per_node=$GPUS_PER_NODE \
#     --node_rank=$RANK \
#     --master_addr=$MASTER_ENDPOINT \
#      --master_port=$MASTER_PORT \
