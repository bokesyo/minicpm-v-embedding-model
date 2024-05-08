import fitz  # pip install PyMuPDF
import os
from multiprocessing import Process
from tqdm import tqdm

def export_pages_to_png(rank, pdf_path, start_page, end_page, output_dir, dpi=300):
    """
    处理指定范围内的PDF页面，将它们导出为PNG图片。

    :param pdf_path: PDF文件路径。
    :param start_page: 分配给此进程的起始页面索引。
    :param end_page: 分配给此进程的结束页面索引。
    :param output_dir: 保存PNG图片的输出目录。
    :param dpi: 图片的DPI。
    """
    doc = fitz.open(pdf_path)
    
    iter_ = range(start_page, end_page + 1)

    if rank == 0:
        iter_ = tqdm(iter_)

    for page_num in iter_:
        page = doc.load_page(page_num)
        
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        output_path = os.path.join(output_dir, f"{page_num}.png")
        pix.save(output_path)
        
        # print(f"processed {page_num}")
    
    doc.close()

def split_processing(pdf_path, output_dir, dpi=300, processes=4):
    """
    将PDF页面导出为PNG图片的任务分割成多个进程并行处理。

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
        # 确保最后一个进程处理所有剩余的页面
        end_page = (i + 1) * pages_per_process - 1 if i < processes - 1 else total_pages - 1
        
        process = Process(target=export_pages_to_png, args=(i, pdf_path, start_page, end_page, output_dir, dpi))
        processes_list.append(process)

    for process in processes_list:
        process.start()

    for process in processes_list:
        process.join()

# 示例用法
# pdf_path = "path/to/your/pdf_file.pdf"
# pdf_path = "/home/jeeves/visual/Microbiology-WEB.pdf"  # PDF文件路径

# output_dir = "/home/jeeves/visual/test3"

# os.makedirs(output_dir, exist_ok=True)

dpi = 100
processes = 6

# split_processing(pdf_path, output_dir, dpi, processes)

# 假设的目录路径，这里用当前目录"."作为例子
directory_path = "/home/jeeves/visual_embedding_1"

output_directory_path = '/home/jeeves/visual_embedding_1_dataset'
os.makedirs(output_directory_path, exist_ok=True)

# 列出给定目录下所有的PDF文件
pdf_files = [file for file in os.listdir(directory_path) if file.endswith('.pdf')]

print(pdf_files)

for pdf_file in pdf_files:
    pdf_abs_path = os.path.join(directory_path, pdf_file)
    pdf_name = pdf_file.split('.')[0]
    this_pdf_output_directory = os.path.join(output_directory_path, pdf_name)
    os.makedirs(this_pdf_output_directory, exist_ok=True)
    split_processing(pdf_abs_path, this_pdf_output_directory, dpi, processes)
    
    # break
