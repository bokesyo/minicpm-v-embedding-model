import torch
import torch.nn.functional as F

# 假定你使用的是CUDA GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 生成数据和标签
data = torch.randn(12000, 1024, device=device, requires_grad=True)
labels = torch.randint(0, 12000, (12000,), device=device)

# 重置显存计数器
torch.cuda.reset_peak_memory_stats(device)

# 计算点积
similarity_matrix = torch.matmul(data, data.T)

# 定义损失函数并计算损失
loss = F.cross_entropy(similarity_matrix, labels)

# 反向传播
loss.backward()

peak_memory_allocated = torch.cuda.max_memory_allocated(device)

print(f"Peak memory allocated: {peak_memory_allocated / 1024**2:.2f} MB")

# 计算张量的显存占用
num_elements = data.numel()  # 张量中元素的总数
element_size = data.element_size()  # 每个元素的大小（字节）
total_size_bytes = num_elements * element_size  # 总字节数
total_size_megabytes = total_size_bytes / (1024 ** 2)  # 转换为兆字节

print(f"Number of elements: {num_elements}")
print(f"Element size (bytes): {element_size}")
print(f"Total size (bytes): {total_size_bytes}")
print(f"Total size (MB): {total_size_megabytes:.2f} MB")