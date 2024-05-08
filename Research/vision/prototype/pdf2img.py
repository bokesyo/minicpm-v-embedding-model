# pip install PyMuPDF

import fitz  # PyMuPDF
import os

def export_pdf_pages_to_png(
    pdf_path, start_page, end_page, output_dir, dpi=200):
    """
    将PDF文件中的一部分页面导出为PNG图片。

    :param pdf_path: PDF文件的路径。
    :param start_page: 起始页面（从0开始计数）。
    :param end_page: 结束页面（包含在内）。
    :param output_dir: 输出目录路径，用于保存PNG图片。
    """
    # 打开PDF文件
    doc = fitz.open(pdf_path)

    # 遍历指定的页面范围
    for page_num in range(start_page, end_page + 1):
        page = doc.load_page(page_num)
        
        # 渲染页面为PNG图片
        # pix = page.get_pixmap()
        # 通过设置DPI来调整图片质量
        zoom = dpi / 72  # PDF文件的默认DPI是72
        mat = fitz.Matrix(zoom, zoom)
        
        # 渲染页面为PNG图片
        pix = page.get_pixmap(matrix=mat)
        
        # 构建输出文件路径
        output_path = f"{output_dir}/page_{page_num + 1}.png"
        
        # 保存PNG图片
        pix.save(output_path)

    # 关闭文档
    doc.close()

# 示例用法
pdf_path = "/home/jeeves/visual/Microbiology-WEB.pdf"  # PDF文件路径
start_page = 50  # 起始页面（例如：第1页，从0开始计数）
end_page = 60    # 结束页面（例如：第5页，从0开始计数）
output_dir = "/home/jeeves/visual/test1"  # 输出目录路径

os.makedirs(output_dir, exist_ok=True)
export_pdf_pages_to_png(pdf_path, start_page, end_page, output_dir)
