cd /home/jeeves
hadoop fs -get hdfs:///user/tc_agi/user/xubokai/msmarco_prototype_tokens_tiny .
hadoop fs -get hdfs:///user/tc_agi/user/xubokai/msmarco_prototype_tokens_medium .
hadoop fs -get hdfs:///user/tc_agi/user/xubokai/msmarco_cpmd_2b_tokens_tiny .
hadoop fs -get hdfs:///user/tc_agi/user/xubokai/msmarco_cpmd_2b_tokens_medium .
# hadoop fs -get hdfs:///user/tc_agi/user/xubokai/msmarco_cpmd_2b_tokens .
hadoop fs -get hdfs:///user/tc_agi/user/xubokai/bert-base-uncased-small .
hadoop fs -get hdfs:///user/tc_agi/user/xubokai/cpm_d-2b_with_pad_token .


pip install -U transformers 
echo "transformers setup succeed!"

pip install -U accelerate
echo "accelerate setup succeed!"

pip install -U datasets
echo "datasets setup succeed!"

# add pad token
# added to model file

cd /home/jeeves/openmatch
pip install -e .
echo "openmatch setup succeed!"

