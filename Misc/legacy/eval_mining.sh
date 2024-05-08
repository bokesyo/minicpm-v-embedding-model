GPUS_PER_NODE=8

SPLIT=$1

echo "WORLD_SIZE=$WORLD_SIZE"
echo "RANK=$RANK"
echo "GPUS_PER_NODE=$GPUS_PER_NODE"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"


cd /local/apps/openmatch

pip install -U transformers 
echo "transformers setup succeed!"

pip install -U accelerate
echo "accelerate setup succeed!"

pip install -U datasets
echo "datasets setup succeed!"

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

pip install .

echo "openmatch setup succeed!"

MODEL_PATH=$(python -c "import json; import os; config_string = open(os.environ['PLATFORM_CONFIG_PATH'], 'r').read(); config = json.loads(config_string); print(list(config['model_map'].values() )[0] )")
DATASET_PATH=$(python -c "import json; import os; config_string = open(os.environ['PLATFORM_CONFIG_PATH'], 'r').read(); config = json.loads(config_string); print(list(config['dataset_map'].values())[0] )")
RESULT_DIR=$(python -c "import json; import os; config_string = open(os.environ['PLATFORM_CONFIG_PATH'], 'r').read(); config = json.loads(config_string); print(config['export_map']['results_dir'] )")

TIMESTR=$(date "+%Y-%m-%d-%H%M%S")
IDENTITY="msmarco-mining-$TIMESTR"
echo "IDENTITY=$IDENTITY"

# GPUS_PER_NODE=8
LOG_DIR="$RESULT_DIR/$IDENTITY/"
mkdir $LOG_DIR

torchrun \
    --nnodes=$WORLD_SIZE \
    --node_rank=$RANK \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_id=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --nproc_per_node=$GPUS_PER_NODE mteb_eval/evaluate_sbert_multi_gpu_mining.py \
    --model_path $MODEL_PATH \
    --dataset_path $DATASET_PATH \
    --log_path $LOG_DIR \
    --batch_size 32 \
    --max_query_len 512 \
    --max_corpus_len 512 \
    --query_instruction "./Eval_Instruction/bge.query.txt" \
    --corpus_instruction "none" \
    --pooling_method "cls" \
    --split $SPLIT

