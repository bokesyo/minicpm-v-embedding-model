# cd /local/apps/openmatch

# pip install -U transformers 
# echo "transformers setup succeed!"

# pip install -U accelerate
# echo "accelerate setup succeed!"

# pip install -U datasets
# echo "datasets setup succeed!"

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


# MODEL_PATH=$(python -c "import json; import os; config_string = open(os.environ['PLATFORM_CONFIG_PATH'], 'r').read(); config = json.loads(config_string); print(list(config['model_map'].values() )[0] )")
# DATASET_PATH=$(python -c "import json; import os; config_string = open(os.environ['PLATFORM_CONFIG_PATH'], 'r').read(); config = json.loads(config_string); print(list(config['dataset_map'].values())[0] )")

MODEL_PATH=/home/jeeves/cpm_d-2b_with_pad_token
# DATASET_PATH=/home/jeeves/fever
DATASET_PATH=/home/jeeves/light_beir_eval/fiqa

TIMESTR=$(date "+%Y-%m-%d-%H%M%S")

mkdir /home/jeeves/mteb_light_log/

LOG_PATH=/home/jeeves/mteb_light_log/$TIMESTR

mkdir $LOG_PATH

GPUS_PER_NODE=1

THIS_QUERY_INSTRUCTION="./Eval_Instruction/fiqa.query.txt"
THIS_CORPUS_INSTRUCTION="none"

torchrun --nproc_per_node=$GPUS_PER_NODE mteb_eval/evaluate_sbert_multi_gpu.py \
    --model_path $MODEL_PATH \
    --dataset_path $DATASET_PATH \
    --log_path $LOG_PATH \
    --batch_size 4 \
    --max_query_len 32 \
    --max_corpus_len 32 \
    --query_instruction $THIS_QUERY_INSTRUCTION \
    --corpus_instruction $THIS_CORPUS_INSTRUCTION \
    --split "test" \
    --pooling_method "mean" \
    --attention "bidirectional" \
    --attn_implementation "flash_attention_2" \
    

# cpmd(2p4b) x 64 cause OOM 