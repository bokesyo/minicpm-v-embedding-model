# export q_max_len=64
# export p_max_len=128
# export n_gpus=1
# export port=12345


# on each node, the script will only run once.
MAX_Q_LEN=512
MAX_P_LEN=512
PER_DEV_BATCH_SIZE=256
QUERY_INSTRUCTION=true
CORPUS_INSTRUCTION=false
POOLING="wmean"
ATTENTION="bidirectional"
CHECKPOINT_MODEL="/home/jeeves/cpm_d-2b_with_pad_token"

# MASTER_PORT=23456
GPUS_PER_NODE=1

# MODEL_PATH=$(python -c "import json; import os; config_string = open(os.environ['PLATFORM_CONFIG_PATH'], 'r').read(); config = json.loads(config_string); print(list(config['model_map'].values() )[0] )")
# DATASET_PATH=$(python -c "import json; import os; config_string = open(os.environ['PLATFORM_CONFIG_PATH'], 'r').read(); config = json.loads(config_string); print(list(config['dataset_map'].values())[0] )")
# CHECKPOINT_DIR=$(python -c "import json; import os; config_string = open(os.environ['PLATFORM_CONFIG_PATH'], 'r').read(); config = json.loads(config_string); print(config['export_map']['checkpoints_dir'] )")
# MODEL_GROUP_NAME=$(python -c "import json; import os; config_string = open(os.environ['PLATFORM_CONFIG_PATH'], 'r').read(); config = json.loads(config_string); print(list(config['model_map'].keys())[0] )")
# MODEL_REAL_NAME=$(python -c "import json; import os; config_string = open(os.environ['PLATFORM_CONFIG_PATH'], 'r').read(); config = json.loads(config_string); print(list(config['model_map'].values())[0].split('/')[-2] )")
# DATASET_REAL_NAME=$(python -c "import json; import os; config_string = open(os.environ['PLATFORM_CONFIG_PATH'], 'r').read(); config = json.loads(config_string); print(list(config['dataset_map'].values())[0].split('/')[-1] )")
# # MODEL_OUTPUT_DIR=$(python -c "import json; import os; config_string = open(os.environ['PLATFORM_CONFIG_PATH'], 'r').read(); config = json.loads(config_string); print(config['export_map']['models_dir'] )")
# RESULT_DIR=$(python -c "import json; import os; config_string = open(os.environ['PLATFORM_CONFIG_PATH'], 'r').read(); config = json.loads(config_string); print(config['export_map']['results_dir'] )")
# LOG_DIR=$(python -c "import json; import os; config_string = open(os.environ['PLATFORM_CONFIG_PATH'], 'r').read(); config = json.loads(config_string); print(config['export_map']['logs_dir'] )")
# CHECKPOINT_MODEL_PATH="${CHECKPOINT_DIR}/${CHECKPOINT_MODEL}"
# TEMP_DIR="/local/apps/openmatch/eval_temp"
# mkdir $TEMP_DIR

TEMP_DIR="/home/jeeves/tmp-4"

# echo "EXPORT_DIR: $EXPORT_DIR"
# # echo "Model Path: $MODEL_PATH"
# echo "Model Path: $CHECKPOINT_MODEL_PATH"
# echo "Dataset Path: $DATASET_PATH"
# echo "Checkpoint Path: $CHECKPOINT_DIR"
# # echo "Model REAL Name: $MODEL_REAL_NAME"
# # echo "Model Output Dir: $MODEL_OUTPUT_DIR"
# echo "Result Dir: $RESULT_DIR"
# echo "Log Dir: $LOG_DIR"

# echo "======== Installing openmatch: =========="
# pip install transformers==4.37.2
# echo "transformers, deepspeed setup succeed!"
# pip install -U accelerate
# echo "accelerate setup succeed!"
# pip install -U datasets
# echo "datasets setup succeed!"



# cd /local/apps/openmatch

# cd Library/pytrec_eval
# pip install . # here do not use -e .
# echo "pytrec_eval setup succeed!"
# cd -

# pip install .
# echo "openmatch setup succeed!"


# LOCAL_DATASET_PATH="/local/apps/openmatch/dataset_tmp"
# copy files
# echo "copying data"
# cp -r $DATASET_PATH $LOCAL_DATASET_PATH
# echo "copied data to $LOCAL_DATASET_PATH"

# step1: distributed parallel encode on one node(shared storage is needed for multiple nodes), multiple GPU encode sharded query -> embedding.query.rank.{process_rank} (single file by default, hack is needed for multiple file)
# step2: distributed parallel encode on one node(shared storage is needed for multiple nodes), multiple GPU encode sharded corpus -> embedding.query.rank.{process_rank}.{begin_id}-{end_id} (multiple by default,hack is needed for single file)
# step3: distributed parallel retrieval on one node (shared storage is needed for multiple nodes), multiple gpu retrieve its part of query, and corpus will share, but load batches by batches (embedding.query.rank.{process_rank}) and save trec file trec.rank.{process_rank}
# step 4: master collect trec file and calculate metrics

# SUB_DATASET="fiqa"
DATASET_PATH="/home/jeeves/light_beir_eval"
SUB_DATASET="ArguAna"
QUERY_TEMPLATE_PATH="./Eval_Instruction/${SUB_DATASET}.query.txt"
QUERY_TEMPLATE=$(cat $QUERY_TEMPLATE_PATH)
CORPUS_TEMPLATE="<title> <text>"

torchrun --nproc_per_node=$GPUS_PER_NODE \
    src/openmatch/driver/beir_eval_pipeline.py \
    --data_dir "$DATASET_PATH/$SUB_DATASET" \
    --model_name_or_path "$CHECKPOINT_MODEL" \
    --output_dir "$TEMP_DIR" \
    --query_template "$QUERY_TEMPLATE" \
    --doc_template "$CORPUS_TEMPLATE" \
    --q_max_len 512 \
    --p_max_len 512  \
    --per_device_eval_batch_size $PER_DEV_BATCH_SIZE \
    --dataloader_num_workers 1 \
    --fp16 \
    --use_gpu \
    --overwrite_output_dir false \
    --use_split_search \
    --max_inmem_docs 1000000 \
    --normalize true \
    --pooling "$POOLING" \
    --attention "$ATTENTION" \
    --attn_implementation "flash_attention_2" \
    --phase "encode" \
    --data_cache_dir "/home/jeeves/cache" \

torchrun --nproc_per_node=$GPUS_PER_NODE \
    src/openmatch/driver/beir_eval_pipeline.py \
    --data_dir "$DATASET_PATH/$SUB_DATASET" \
    --model_name_or_path "$CHECKPOINT_MODEL" \
    --output_dir "$TEMP_DIR" \
    --use_gpu \
    --use_split_search \
    --phase "retrieve" \

