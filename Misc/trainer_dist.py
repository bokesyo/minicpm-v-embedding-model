import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import h5py
import os
import argparse
from torch.utils.data.distributed import DistributedSampler

# 1. 定义三层MLP模型
class ThreeLayerMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ThreeLayerMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.fc(x)

# 2. 自定义Dataset类
class H5Dataset(Dataset):
    def __init__(self, file_name):
        self.file = h5py.File(file_name, 'r')
        self.dataset = self.file['dataset']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return torch.tensor(self.dataset[idx], dtype=torch.float)

    def close(self):
        self.file.close()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, epochs):
    setup(rank, world_size)

    # 3. DataLoader和模型初始化
    file_name = "data_pairs.h5"
    dataset = H5Dataset(file_name)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, sampler=sampler)

    input_size = 128  # 根据您的数据维度调整
    hidden_size = 50
    output_size = 10  # 输出大小，根据您的任务调整
    model = ThreeLayerMLP(input_size, hidden_size, output_size).to(rank)
    print("model ok")
    
    model = DDP(model, device_ids=[rank])
    print("model ddp ok")
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        for i, data in enumerate(dataloader):
            inputs = data[:, 0, :].to(rank)
            targets = data[:, 1, :].to(rank)  # 假设目标是数据的第二个元素

            optimizer.zero_grad()

            # 前向传播
            output = model(inputs)

            # 同步所有GPU上的表征
            gathered_output = [torch.zeros_like(output) for _ in range(world_size)]
            dist.all_gather(gathered_output, output)
            gathered_output = torch.cat(gathered_output, dim=0)

            # 计算损失
            scores = torch.matmul(output, gathered_output.T)
            targets = torch.arange(scores.size(0), device=rank)
            loss = criterion(scores, targets)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f'Rank {rank}, Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
    
    cleanup()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', default=1, type=int, metavar='N')
    parser.add_argument('--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('--epochs', default=5, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
    args = parser.parse_args()

    args.world_size = args.gpus * args.nodes                
    torch.multiprocessing.spawn(train, args=(args.world_size, args.epochs), nprocs=args.gpus, join=True)

if __name__ == "__main__":
    main()
