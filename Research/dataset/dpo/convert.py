import pandas as pd

# 替换为你的Parquet文件路径
parquet_file_path = '/home/jeeves/Nectar/data/rlaif.parquet'
# 输出的JSONL文件路径
jsonl_file_path = '/home/jeeves/Nectar/data/rlaif.jsonl'

# 读取Parquet文件
df = pd.read_parquet(parquet_file_path)

# 打开一个文件用于写入
with open(jsonl_file_path, 'w') as f:
    # 遍历DataFrame的每一行
    for _, row in df.iterrows():
        # 将行数据转换为JSON格式，并写入文件
        f.write(row.to_json() + '\n')
