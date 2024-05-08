import json
import os

def merge_jsonl_and_create_metadata(root_dir, output_dir):
    """
    合并根目录下的所有子目录中的JSONL文件，并创建元数据描述文件。

    :param root_dir: 根目录路径。
    """
    metadata = {}
    current_line = 0

    # 创建合并后的JSONL文件和元数据文件的路径
    merged_jsonl_path = os.path.join(output_dir, 'data.jsonl')
    metadata_path = os.path.join(output_dir, 'metadata.json')

    # 打开合并后的JSONL文件以便写入
    with open(merged_jsonl_path, 'w') as merged_file:
        # 遍历根目录下的所有子目录
        for subdir in next(os.walk(root_dir))[1]:
            print(f"processing {subdir}")
            pdf_name = subdir  # 子目录名称
            subdir_path = os.path.join(root_dir, subdir)
            start_line = current_line

            # 遍历子目录下的所有JSONL文件
            for filename in os.listdir(subdir_path):
                if filename.endswith('.jsonl'):
                    with open(os.path.join(subdir_path, filename), 'r') as f:
                        for line in f:
                            merged_file.write(line)
                            current_line += 1

            # 更新元数据文件
            metadata[pdf_name] = {'start_line': start_line, 'end_line': current_line}

    # 写入元数据文件
    with open(metadata_path, 'w') as meta_file:
        json.dump(metadata, meta_file, indent=4)

# 使用示例
# root_dir = '/home/jeeves/visual_embedding_1_dataset_jsonl_split'  # 替换为你的根目录路径
# output_dir = '/home/jeeves/visual_embedding_1_dataset_jsonl_merged'

root_dir = "/home/jeeves/visual_embedding_2_long_visual_dataset_jsonl_split"
output_dir = "/home/jeeves/visual_embedding_2_long_visual_dataset_jsonl_merged"

os.makedirs(output_dir, exist_ok=True)

merge_jsonl_and_create_metadata(root_dir, output_dir)
