
# on each node, the script will only run once.
MAX_Q_LEN=$1
MAX_P_LEN=$2
PER_DEV_BATCH_SIZE=$3
POOLING=${4}
ATTENTION=${5}
SUB_DATASET=${6} # ArguAna, fiqa
GPUS_PER_NODE=${7}
CHECKPOINT_MODEL=${8}

MASTER_PORT=23456

# 使用 IFS（内部字段分隔符）和 read 命令将字符串分割为数组
IFS=',' read -r -a SUB_DATASET_LIST <<< "$SUB_DATASET"


MODEL_PATH=$(python -c "import json; import os; config_string = open(os.environ['PLATFORM_CONFIG_PATH'], 'r').read(); config = json.loads(config_string); print(list(config['model_map'].values() )[0] )")
DATASET_PATH=$(python -c "import json; import os; config_string = open(os.environ['PLATFORM_CONFIG_PATH'], 'r').read(); config = json.loads(config_string); print(list(config['dataset_map'].values())[0] )")
CHECKPOINT_DIR=$(python -c "import json; import os; config_string = open(os.environ['PLATFORM_CONFIG_PATH'], 'r').read(); config = json.loads(config_string); print(config['export_map']['checkpoints_dir'] )")
MODEL_GROUP_NAME=$(python -c "import json; import os; config_string = open(os.environ['PLATFORM_CONFIG_PATH'], 'r').read(); config = json.loads(config_string); print(list(config['model_map'].keys())[0] )")
MODEL_REAL_NAME=$(python -c "import json; import os; config_string = open(os.environ['PLATFORM_CONFIG_PATH'], 'r').read(); config = json.loads(config_string); print(list(config['model_map'].values())[0].split('/')[-2] )")
DATASET_REAL_NAME=$(python -c "import json; import os; config_string = open(os.environ['PLATFORM_CONFIG_PATH'], 'r').read(); config = json.loads(config_string); print(list(config['dataset_map'].values())[0].split('/')[-1] )")
# MODEL_OUTPUT_DIR=$(python -c "import json; import os; config_string = open(os.environ['PLATFORM_CONFIG_PATH'], 'r').read(); config = json.loads(config_string); print(config['export_map']['models_dir'] )")
RESULT_DIR=$(python -c "import json; import os; config_string = open(os.environ['PLATFORM_CONFIG_PATH'], 'r').read(); config = json.loads(config_string); print(config['export_map']['results_dir'] )")
LOG_DIR=$(python -c "import json; import os; config_string = open(os.environ['PLATFORM_CONFIG_PATH'], 'r').read(); config = json.loads(config_string); print(config['export_map']['logs_dir'] )")
CHECKPOINT_MODEL_PATH="${CHECKPOINT_DIR}/${CHECKPOINT_MODEL}"


echo "EXPORT_DIR: $EXPORT_DIR"
# echo "Model Path: $MODEL_PATH"
echo "Model Path: $CHECKPOINT_MODEL_PATH"
echo "Dataset Path: $DATASET_PATH"
echo "Checkpoint Path: $CHECKPOINT_DIR"
# echo "Model REAL Name: $MODEL_REAL_NAME"
# echo "Model Output Dir: $MODEL_OUTPUT_DIR"
echo "Result Dir: $RESULT_DIR"
echo "Log Dir: $LOG_DIR"


pip install transformers==4.37.2
echo "transformers, deepspeed setup succeed!"
pip install -U accelerate
echo "accelerate setup succeed!"
pip install -U datasets
echo "datasets setup succeed!"
cd /local/apps/openmatch
cd Library/pytrec_eval
pip install . # here do not use -e .
echo "pytrec_eval setup succeed!"
cd -
pip install .

# pip install --upgrade packaging
# pip install --upgrade wheel
# # pip install flash-attn
# pip install -U flash-attn==2.1.0 --no-build-isolation
echo "openmatch setup succeed!"


# LOCAL_DATASET_PATH="/local/apps/openmatch/dataset_tmp"
# copy files
# echo "copying data"
# cp -r $DATASET_PATH $LOCAL_DATASET_PATH
# echo "copied data to $LOCAL_DATASET_PATH"


# step1: distributed parallel encode on one node(shared storage is needed for multiple nodes), multiple GPU encode sharded query -> embedding.query.rank.{process_rank} (single file by default, hack is needed for multiple file)
# step2: distributed parallel encode on one node(shared storage is needed for multiple nodes), multiple GPU encode sharded corpus -> embedding.query.rank.{process_rank}.{begin_id}-{end_id} (multiple by default,hack is needed for single file)
# step3: distributed parallel retrieval on one node (shared storage is needed for multiple nodes), multiple gpu retrieve its part of query, and corpus will share, but load batches by batches (embedding.query.rank.{process_rank}) and save trec file trec.rank.{process_rank}
# step 4: master collect trec file and calculate metrics


for SUB_DATASET in "${SUB_DATASET_LIST[@]}"
do
    THIS_DATASET_PATH="$DATASET_PATH/$SUB_DATASET"
    THIS_LOG_DIR="$RESULT_DIR/$SUB_DATASET"
    QUERY_TEMPLATE="<text>"
    CORPUS_TEMPLATE="<text>"

    torchrun --nnodes=$WORLD_SIZE \
        --node_rank=$RANK \
        --nproc_per_node=$GPUS_PER_NODE \
        --rdzv_id=1 \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        src/openmatch/driver/beir_eval_pipeline.py \
        --data_dir "$THIS_DATASET_PATH" \
        --model_name_or_path "$CHECKPOINT_MODEL_PATH" \
        --output_dir "$THIS_LOG_DIR" \
        --query_template "$QUERY_TEMPLATE" \
        --doc_template "$CORPUS_TEMPLATE" \
        --q_max_len $MAX_Q_LEN \
        --p_max_len $MAX_P_LEN  \
        --per_device_eval_batch_size $PER_DEV_BATCH_SIZE \
        --dataloader_num_workers 1 \
        --fp16 \
        --use_gpu \
        --overwrite_output_dir false \
        --max_inmem_docs 1000000 \
        --normalize true \
        --pooling "$POOLING" \
        --attention "$ATTENTION" \
        --attn_implementation "flash_attention_2" \
        --phase "encode" \

    torchrun --nnodes=$WORLD_SIZE \
        --node_rank=$RANK \
        --nproc_per_node=$GPUS_PER_NODE \
        --rdzv_id=1 \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        src/openmatch/driver/beir_eval_pipeline.py \
        --model_name_or_path "$CHECKPOINT_MODEL_PATH" \
        --data_dir "$THIS_DATASET_PATH" \
        --output_dir "$THIS_LOG_DIR" \
        --use_gpu \
        --phase "retrieve" \

done



