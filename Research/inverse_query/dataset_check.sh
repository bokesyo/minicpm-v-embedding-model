hadoop fs -cat hdfs:///user/tc_agi/byq/80b_deduped/mbalib_data_clean/data.jsonl | head -n10


hadoop fs -cat hdfs:///user/tc_agi/byq/80b_deduped/zhihu_qa/data.jsonl | head -n1000 > zhihu_qa.sample.jsonl


# 高质量问答数据
1. 



hadoop fs -cat hdfs:///user/tc_agi/byq/80b_deduped/ultra_chat_dialog/data.jsonl | head -n1000 > ultra_chat_dialog.sample.jsonl

hadoop fs -cat hdfs:///user/tc_agi/byq/80b_deduped/mt_book_dic/data.jsonl | head -n1000 > mt_book_dic.sample.jsonl


hadoop fs -cat hdfs:///user/tc_agi/byq/80b_deduped/peS2o/data.jsonl | head -n1000 > peS2o.sample.jsonl


hadoop fs -cat hdfs:///user/tc_agi/byq/80b_deduped/stack_overflow/data.jsonl | head -n1000 > stack_overflow.sample.jsonl

hadoop fs -cat hdfs:///user/tc_agi/byq/80b_deduped/baike_baidu_clean/data.jsonl | head -n1000 > baike_baidu_clean.sample.jsonl

hadoop fs -cat hdfs:///user/tc_agi/byq/80b_deduped/stack_exchange_qa/data.jsonl | head -n1000 > stack_exchange_qa.sample.jsonl

stack_overflow