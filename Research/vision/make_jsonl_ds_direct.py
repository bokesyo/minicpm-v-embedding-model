import fitz  # PyMuPDF
import os
import base64
from multiprocessing import Process
from tqdm import tqdm
import json

def export_pages_to_jsonl(rank, pdf_path, start_page, end_page, output_dir, dpi=200):
    """
    处理指定范围内的PDF页面，将它们导出为Base64编码的JSONL文件。
    每个进程创建自己的JSONL文件。

    :param rank: 进程编号。
    :param pdf_path: PDF文件路径。
    :param start_page: 分配给此进程的起始页面索引。
    :param end_page: 分配给此进程的结束页面索引。
    :param output_dir: 保存Base64编码的输出目录。
    :param dpi: 图片的DPI。
    """
    doc = fitz.open(pdf_path)
    pdf_source = os.path.basename(pdf_path)
    
    output_file_path = os.path.join(output_dir, f"output_{rank}.jsonl")
    
    iter_ = range(start_page, end_page + 1)

    if rank == 0:
        iter_ = tqdm(iter_)

    for page_num in iter_:
        page = doc.load_page(page_num)
        
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        # 将Pixmap转换为bytes，然后进行Base64编码
        img_bytes = pix.tobytes("png")
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')

        # 构造JSONL行
        jsonl_line = json.dumps({
            "id": f"{pdf_source}_{page_num + 1}",
            "source": pdf_source,
            "page": str(page_num + 1),  # 页码从1开始
            "base64": img_base64
        })

        # 写入JSONL文件
        with open(output_file_path, 'a') as f:
            f.write(jsonl_line + '\n')
    
    doc.close()

def split_processing_to_jsonl(pdf_path, output_dir, dpi=200, processes=15):
    """
    将PDF页面导出为Base64编码的任务分割成多个进程并行处理。

    :param pdf_path: PDF文件路径。
    :param output_dir: 输出目录。
    :param dpi: 输出图片的DPI。
    :param processes: 并行处理的进程数。
    """
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()
    
    pages_per_process = total_pages // processes
    processes_list = []

    for i in range(processes):
        start_page = i * pages_per_process
        end_page = (i + 1) * pages_per_process - 1 if i < processes - 1 else total_pages - 1
        
        process = Process(target=export_pages_to_jsonl, args=(i, pdf_path, start_page, end_page, output_dir, dpi))
        processes_list.append(process)

    for process in processes_list:
        process.start()

    for process in processes_list:
        process.join()

# 示例用法
# pdf_path = "path/to/your/pdf_file.pdf"
# output_dir = "path/to/your/output_directory"
# dpi = 200
# processes = 4

dpi = 100
processes = 14
# directory_path = "/home/jeeves/visual_embedding_1"
# output_directory_path = '/home/jeeves/visual_embedding_1_dataset_jsonl_split'
directory_path = "/home/jeeves/visual_embedding_2_long_visual"
output_directory_path = "/home/jeeves/visual_embedding_2_long_visual_dataset_jsonl_split"


os.makedirs(output_directory_path, exist_ok=True)

pdf_files = [file for file in os.listdir(directory_path) if file.endswith('.pdf')]

print(pdf_files)

for pdf_file in pdf_files:
    pdf_abs_path = os.path.join(directory_path, pdf_file)
    pdf_name = pdf_file.split('.')[0]
    this_pdf_output_directory = os.path.join(output_directory_path, pdf_name)
    os.makedirs(this_pdf_output_directory, exist_ok=True)
    # split_processing(pdf_abs_path, this_pdf_output_directory, dpi, processes)
    split_processing_to_jsonl(pdf_abs_path, this_pdf_output_directory, dpi, processes)

    
