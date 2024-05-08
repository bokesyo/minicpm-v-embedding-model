import faiss
import numpy as np

d = 768          # 向量维度
nq = 50000       # 查询向量数量
num_shards = 10  # 分片数量
nb = 1000000      # 每个分片的数据库大小
k = 4            # 查找最近的4个邻居

# 生成查询向量
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.

ngpu = faiss.get_num_gpus()
# 初始化 GPU 资源
gpu_res = faiss.StandardGpuResources()

for i in range(num_shards):
    print(f"shard {i}")

    print("generating corpus...")
    # 为每个分片生成随机数据
    xb = np.random.random((nb, d)).astype('float32')
    xb[:, 0] += np.arange(nb) / 1000.

    print("building index...")
    # 建立索引
    index = faiss.IndexFlatL2(d)
    index.add(xb)
    
    print("moving index to gpu...")

    # 将索引转移到 GPU
    ngpu = faiss.get_num_gpus()
    gpu_resources = []
    for i in range(ngpu):
        res = faiss.StandardGpuResources()
        gpu_resources.append(res)
    co = faiss.GpuMultipleClonerOptions()
    co.shard = True
    co.usePrecomputed = False
    vres = faiss.GpuResourcesVector()
    vdev = faiss.Int32Vector()

    for i in range(0, ngpu):
        vdev.push_back(i)
        vres.push_back(gpu_resources[i])
    
    index = faiss.index_cpu_to_gpu_multiple(vres, vdev, index, co)
    index_on_gpu = True
    
    print("searching on gpu...")
    # 在 GPU 上执行搜索
    D, I = index.search(xq, k)

    # 处理结果
    # print(f"Results for shard {i}:")
    # print(I)
