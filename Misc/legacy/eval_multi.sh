
Q_MAX_SEQ_LEN=$1
P_MAX_SEQ_LEN=$2
PER_DEV_BATCH_SIZE=$3
ENABLE_QUERY_INSTRUCTION=$4
ENABLE_CORPUS_INSTRUCTION=$5
POOLING=$6
ATTENTION=$7
CHECKPOINT_MODEL=$8

# hdfs:///user/tc_agi/user/xubokai/light_beir_eval

echo "======== Config ==========="
echo $(python -c "import json; import os; config_string = open(os.environ['PLATFORM_CONFIG_PATH'], 'r').read(); config = json.loads(config_string); print(config)")

DATASET_PATH=$(python -c "import json; import os; config_string = open(os.environ['PLATFORM_CONFIG_PATH'], 'r').read(); config = json.loads(config_string); print(list(config['dataset_map'].values())[0] )")

DATASET_REAL_NAME=$(python -c "import json; import os; config_string = open(os.environ['PLATFORM_CONFIG_PATH'], 'r').read(); config = json.loads(config_string); print(list(config['dataset_map'].values())[0].split('/')[-1] )")

CHECKPOINT_DIR=$(python -c "import json; import os; config_string = open(os.environ['PLATFORM_CONFIG_PATH'], 'r').read(); config = json.loads(config_string); print(config['export_map']['checkpoints_dir'] )")

# one node is fine
# LOG_DIR=$(python -c "import json; import os; config_string = open(os.environ['PLATFORM_CONFIG_PATH'], 'r').read(); config = json.loads(config_string); print(config['export_map']['logs_dir'] )")

RESULT_DIR=$(python -c "import json; import os; config_string = open(os.environ['PLATFORM_CONFIG_PATH'], 'r').read(); config = json.loads(config_string); print(config['export_map']['results_dir'] )")

MODEL_PATH="$CHECKPOINT_DIR/$CHECKPOINT_MODEL"

DATASET_BASE_PATH=$DATASET_PATH

TIMESTR=$(date "+%Y-%m-%d-%H%M%S")
IDENTITY="eval-$DATASET_REAL_NAME-$TIMESTR"
echo "IDENTITY=$IDENTITY"

mkdir "$RESULT_DIR/$IDENTITY"

GPUS_PER_NODE=8




cd /local/apps/openmatch

pip install transformers==4.37.2
echo "transformers setup succeed!"

pip install -U accelerate
echo "accelerate setup succeed!"

pip install -U datasets
echo "datasets setup succeed!"


pip install -e .
echo "openmatch setup succeed!"

cd Library
cd pytrec_eval
pip install . # here do not use -e .
echo "pytrec_eval setup succeed!"
cd ..

cd sentence-transformers
pip install .
echo "sentence-transformers setup succeed!"
cd ..

cd beir
pip install .
echo "beir setup succeed!"
cd ..

cd ..

pip install evaluate
echo "evaluate setup succeed!"




# ---------------------------------------------------------------------------------

# MODEL_PATH="/home/jeeves/cpm_d-2b_with_pad_token"
# GPUS_PER_NODE=2
# PER_DEV_BATCH_SIZE=32
# MAX_SEQ_LEN=512
# DATASET_BASE_PATH="/home/jeeves/light_beir_eval"
# RESULT_DIR="/home/jeeves/light_log"
# TIMESTR=$(date "+%Y-%m-%d-%H%M%S")
# IDENTITY="eval-$TIMESTR"
# echo "IDENTITY=$IDENTITY"
# mkdir "$RESULT_DIR/$IDENTITY"

# ---------------------------------------------------------------------------------
#  
# "fiqa" "nfcorpus" "scidocs"  "scifact"

# 为每个数据集创建一个循环
for DATASET in "ArguAna" "fiqa" "nfcorpus" "scidocs" "scifact"
do
    THIS_DATASET_PATH=$DATASET_BASE_PATH/$DATASET
    THIS_LOG_DIR="$RESULT_DIR/$IDENTITY/$DATASET"
    
    # 为每个数据集创建一个日志目录
    mkdir -p $THIS_LOG_DIR

    # ref: https://github.com/xlang-ai/instructor-embedding/blob/main/evaluation/MTEB/mteb/abstasks/AbsTaskRetrieval.py

    if [ "$ENABLE_QUERY_INSTRUCTION" = "true" ]; then
        THIS_QUERY_INSTRUCTION="./Eval_Instruction/$DATASET.query.txt"
    else
        THIS_QUERY_INSTRUCTION="none"
    fi

    if [ "$ENABLE_CORPUS_INSTRUCTION" = "true" ]; then
        THIS_CORPUS_INSTRUCTION="./Eval_Instruction/$DATASET.corpus.txt"
    else
        THIS_CORPUS_INSTRUCTION="none"
    fi
    
    # 运行评估脚本
    torchrun --nproc_per_node=$GPUS_PER_NODE mteb_eval/evaluate_sbert_multi_gpu.py \
        --model_path $MODEL_PATH \
        --dataset_path $THIS_DATASET_PATH \
        --log_path $THIS_LOG_DIR \
        --batch_size $PER_DEV_BATCH_SIZE \
        --max_query_len $Q_MAX_SEQ_LEN \
        --max_corpus_len $P_MAX_SEQ_LEN \
        --query_instruction $THIS_QUERY_INSTRUCTION \
        --corpus_instruction $THIS_CORPUS_INSTRUCTION \
        --split "test" \
        --pooling_method $POOLING \
        --attention $ATTENTION \
        --attn_implementation "flash_attention_2" \

done

