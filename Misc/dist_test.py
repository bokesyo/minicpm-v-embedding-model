import os
import sys

import torch.distributed as dist

if __name__ == "__main__":
    dist.init_process_group("nccl")
    
    rank_dist = dist.get_rank()
    world_size_dist = dist.get_world_size()

    rank_os = int(os.environ['RANK'])
    world_size_os = int(os.environ['WORLD_SIZE'])
    local_rank_os = int(os.environ['LOCAL_RANK'])


    print(f"rank_dist = {rank_dist}, world_size_dist={world_size_dist}")

    print(f"rank_os = {rank_os}, world_size_os = {world_size_os}, local_rank_os = {local_rank_os}")

    print(f"sys.argv = {sys.argv}")

