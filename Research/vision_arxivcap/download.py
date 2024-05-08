import os
import requests
from concurrent.futures import ThreadPoolExecutor

def download_file(file_name):
    cnt = 0
    # file_name = url.split('/')[-1]
    # file_name = 
    temp_file_name = f"{file_name}.downloading"

    # Attempt to download the file
    try:
        # https://huggingface.co/datasets/MMInstruction/ArxivCap/resolve/main/data/arXiv_src_0001_001.parquet
        response = requests.get(f"https://huggingface.co/datasets/MMInstruction/ArxivCap/resolve/main/data/{file_name}", stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Write to a temporary file
        with open(temp_file_name, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Rename the temporary file to the final file name once download is complete
        os.rename(temp_file_name, file_name)
        
        cnt += 1
        print(f"{cnt}: Downloaded {file_name}")
    except requests.RequestException as e:
        print(f"Failed to download {url}: {str(e)}")
        # Clean up any partially downloaded file
        if os.path.exists(temp_file_name):
            os.remove(temp_file_name)

# Load URLs from the sampled file
with open('../sampled_parquet_index.txt', 'r', encoding='utf-8') as file:
    urls = [line.strip() for line in file if line.strip()]

# Using ThreadPoolExecutor to download files concurrently
with ThreadPoolExecutor(max_workers=10) as executor:
    executor.map(download_file, urls)
