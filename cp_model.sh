RESULT_DIR="/data/models/cpm_d_2b_embedding/2024-04-11-170338-model-base_with_padtoken-data-clean_medi1_5negs-lr-1e-5-softm_temp-0.02-bsz8-ngpus8-nnodes2-inbatch-true-nepoch-1-pooling-drop_wmean-attention-bidirectional-qinstruct-true-cinstruct-false/"
CHECKPOINT_DIR1=$(python -c "import json; import os; config_string = open(os.environ['PLATFORM_CONFIG_PATH'], 'r').read(); config = json.loads(config_string); print(config['export_map']['checkpoints_dir'] )")
CHECKPOINT_DIR2="/data/checkpoints/2024-04-11-170338-model-base_with_padtoken-data-clean_medi1_5negs-lr-1e-5-softm_temp-0.02-bsz8-ngpus8-nnodes2-inbatch-true-nepoch-1-pooling-drop_wmean-attention-bidirectional-qinstruct-true-cinstruct-false/*"
mkdir /data/models/cpm_d_2b_embedding
mkdir $RESULT_DIR
touch /data/checkpoints/2024-04-11-170338-model-base_with_padtoken-data-clean_medi1_5negs-lr-1e-5-softm_temp-0.02-bsz8-ngpus8-nnodes2-inbatch-true-nepoch-1-pooling-drop_wmean-attention-bidirectional-qinstruct-true-cinstruct-false/vocabs.txt
touch /data/checkpoints/2024-04-11-170338-model-base_with_padtoken-data-clean_medi1_5negs-lr-1e-5-softm_temp-0.02-bsz8-ngpus8-nnodes2-inbatch-true-nepoch-1-pooling-drop_wmean-attention-bidirectional-qinstruct-true-cinstruct-false/a.pt
cp $CHECKPOINT_DIR2 $RESULT_DIR
ls $RESULT_DIR
exit 0