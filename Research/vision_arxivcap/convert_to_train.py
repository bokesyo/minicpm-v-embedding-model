import os
import json
import base64
import pandas as pd
from multiprocessing import Pool

# 指定Parquet文件所在的目录和进程数量
directory = '/data/arxivcap2000-whole/data'
num_processes = 8

def process_file(filename):
    # 构建输出文件名
    output_file = f'data_{filename}.jsonl'
    
    # 确定完整的文件路径
    file_path = os.path.join(directory, filename)
    
    # 读取Parquet文件
    df = pd.read_parquet(file_path)
    data = []

    # 处理DataFrame中的每一行
    for idx in range(len(df)):
        byte_string = df['caption_images'][idx][0]['cil_pairs'][0]['image']['bytes']
        caption = df['caption_images'][idx][0]['caption']
        
        # 将字节字符串转换为base64编码
        base64_string = base64.b64encode(byte_string).decode('utf-8')

        # 构建JSON对象
        json_object = {
            "query": {
                "text": caption,
                "image": None,
                "instruction": "Represent this description for retrieving exact illustration: "
            },
            "pos": [{
                "text": "",
                "image": base64_string,
                "instruction": ""
            }],
            "neg": []
        }
        data.append(json.dumps(json_object, ensure_ascii=False))

    # 写入到相应的文件中
    with open(output_file, 'w') as f:
        for item in data:
            f.write(item + '\n')
    
    return f"Data written to {output_file}"

# 获取所有Parquet文件
files = [f for f in os.listdir(directory) if f.endswith('.parquet')]

# 创建一个进程池并分配任务
if __name__ == '__main__':
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_file, files)
        for result in results:
            print(result)
