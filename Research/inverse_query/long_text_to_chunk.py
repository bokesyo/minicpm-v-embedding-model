from transformers import AutoTokenizer
import json
import os

input_file_path = "/home/jeeves/book3.1000k/test2/30.json" # 2009 Mark Tessler _ A History of the Israeli_Palestinian Conflict[2ndED]_Rebal
output_dir = "/home/jeeves/book3.1000k/chunks_test2" # 242335

os.makedirs(output_dir, exist_ok=True)

MODEL_PATH = "/home/jeeves/Yi-34B-Chat"

with open(input_file_path, 'r') as f:
    file_text = f.read()
    text = json.loads(file_text)["text"]

# 假设文本内容
# text = "This is a sample text." * 10000  # 这仅是一个简化的例子，实际文本应该更长

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Tokenize 全文
print("begin to encode")
tokens = tokenizer.encode(text, add_special_tokens=False)
print("end to encode")

chunk_sizes = [512, 1024]
overlap_fraction = 0.1

result = {}
for chunk_size in chunk_sizes:
    step_size = int(chunk_size * (1 - overlap_fraction))  # Calculate step size based on overlap
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens) - chunk_size + step_size, step_size)]

    # Decode the token chunks back to text
    texts = [tokenizer.decode(chunk) for chunk in chunks]

    # Add the list of decoded texts to the result dictionary
    result[chunk_size] = texts


# 切分成 512 和 1024 token 的块
# chunks_512 = [tokens[i:i + 512] for i in range(0, len(tokens), 512)]
# chunks_1024 = [tokens[i:i + 1024] for i in range(0, len(tokens), 1024)]

# Decode 回文本
# texts_512 = [tokenizer.decode(chunk) for chunk in chunks_512]
# texts_1024 = [tokenizer.decode(chunk) for chunk in chunks_1024]

# 组合两种大小的块
# combined_texts = {
#     "512_tokens": texts_512,
#     "1024_tokens": texts_1024
# }

# 将结果保存到 JSON 文件
# with open(os.path.join(output_dir, "512.json"), 'w', encoding='utf-8') as f:
#     json.dump(texts_512, f, ensure_ascii=False, indent=4)
    
# with open(os.path.join(output_dir, "1024.json"), 'w', encoding='utf-8') as f:
#     json.dump(texts_1024, f, ensure_ascii=False, indent=4)

for chunk_size in chunk_sizes:
    with open(os.path.join(output_dir, f"{chunk_size}.json"), 'w', encoding='utf-8') as f:
        json.dump(result[chunk_size], f, ensure_ascii=False, indent=4)
