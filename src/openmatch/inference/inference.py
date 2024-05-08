import logging
import os
import sys
import gc
import glob
import pickle
from contextlib import nullcontext
from typing import Dict, List, Union

# import faiss
import numpy as np
import torch
from torch.cuda import amp
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer
import pytrec_eval
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser

from openmatch.arguments import DataArguments
from openmatch.arguments import InferenceArguments as EncodingArguments
from openmatch.arguments import ModelArguments
from openmatch.dataset import InferenceDataset, DRInferenceCollator
from openmatch.modeling import DRModelForInference
# from openmatch.retriever import Retriever
from openmatch.utils import save_as_trec


def distributed_parallel_embedding_inference(
    dataset: InferenceDataset,
    model: DRModelForInference,
    args: EncodingArguments,
    dataset_type: str = "corpus", # corpus or query
    split_save: bool = True, # whether to save the embeddings in separate files
):
    # Note: during evaluation, there's no point in wrapping the model
    # inside a DistributedDataParallel as we'll be under `no_grad` anyways.
    if dataset is None:
        raise ValueError("No dataset provided")
    dataloader = DataLoader(
        dataset, # this dataset can be sharded (data parallel)
        batch_size=args.per_device_eval_batch_size,
        collate_fn=DRInferenceCollator(),
        num_workers=args.dataloader_num_workers,
        pin_memory=args.dataloader_pin_memory,
        drop_last=False, # we don't want to drop the last batch, this is evaluation
    )

    os.makedirs(args.output_dir, exist_ok=True)
    
    encoded = []
    lookup_indices = []
    idx = 0
    prev_idx = 0
    for batch_ids, batch in tqdm(dataloader, disable=args.process_index > 0):
        lookup_indices.extend(batch_ids)
        idx += len(batch_ids)
        with amp.autocast() if args.fp16 else nullcontext():
            # print(f"{args.fp16}!!!")
            with torch.no_grad():
                for k, v in batch.items():
                    batch[k] = v.to(args.device)
                
                # print(batch['input_ids'].shape)
                
                if dataset_type == "corpus":
                    model_output: DROutput = model(passage=batch)
                    encoded_ = model_output.p_reps.cpu().detach().numpy()
                elif dataset_type == "query":
                    model_output: DROutput = model(query=batch)
                    encoded_ = model_output.q_reps.cpu().detach().numpy()
                else:
                    raise ValueError(f"dataset_type: {dataset_type} is not valid.")
                
                # logging.info(f"you got {encoded_.dtype} for embedding")
                
                encoded.append(encoded_)
        
        if len(lookup_indices) >= args.max_inmem_docs // args.world_size:
            if split_save:
                encoded = np.concatenate(encoded)
                with open(
                    os.path.join(
                        args.output_dir,
                        "embeddings.{}.rank.{}.{}-{}".format(
                            dataset_type,
                            args.process_index, 
                            prev_idx, idx
                        ),
                    ),
                    "wb",
                ) as f:
                    pickle.dump((encoded, lookup_indices), f, protocol=4)
                encoded = []
                lookup_indices = []
                prev_idx = idx
                gc.collect()

    # this is to handle the last batch
    if len(lookup_indices) > 0:
        if split_save:
            encoded = np.concatenate(encoded)
            with open(
                os.path.join(
                    args.output_dir,
                    "embeddings.{}.rank.{}.{}-{}".format(
                        dataset_type,
                        args.process_index, 
                        prev_idx, idx
                    ),
                ),
                "wb",
            ) as f:
                pickle.dump((encoded, lookup_indices), f, protocol=4)

    # if save to a whole file (each rank only save one file)
    if not split_save:
        encoded = np.concatenate(encoded)
        with open(
            os.path.join(
                args.output_dir,
                "embeddings.{}.rank.{}".format(
                    dataset_type,
                    args.process_index, 
                    # prev_idx, idx
                ),
            ),
            "wb",
        ) as f:
            pickle.dump((encoded, lookup_indices), f, protocol=4)
    
    del encoded
    del lookup_indices

    if args.world_size > 1:
        torch.distributed.barrier()

    return





# def qwen_vl_distributed_parallel_embedding_inference(
#     dataset: InferenceDataset,
#     model: torch.nn.Module,
#     args: None,
#     split_save: bool = True, # whether to save the embeddings in separate files
# ):
#     # Note: during evaluation, there's no point in wrapping the model
#     # inside a DistributedDataParallel as we'll be under `no_grad` anyways.
#     if dataset is None:
#         raise ValueError("No dataset provided")
#     dataloader = DataLoader(
#         dataset,
#         batch_size=args.per_device_eval_batch_size,
#         collate_fn=DRInferenceCollator(),
#         num_workers=args.dataloader_num_workers,
#         pin_memory=args.dataloader_pin_memory,
#         drop_last=False, # we don't want to drop the last batch, this is evaluation
#     )

#     os.makedirs(args.output_dir, exist_ok=True)
    
#     encoded = []
#     lookup_indices = []
#     idx = 0
#     prev_idx = 0
#     for batch_ids, batch in tqdm(dataloader, disable=args.process_index > 0):
#         lookup_indices.extend(batch_ids)
#         idx += len(batch_ids)
#         with amp.autocast() if args.fp16 else nullcontext():
#             # print(f"{args.fp16}!!!")
#             with torch.no_grad():
#                 for k, v in batch.items():
#                     batch[k] = v.to(args.device)
                
#                 # print(batch['input_ids'].shape)
                
#                 if dataset_type == "corpus":
#                     model_output: DROutput = model(passage=batch)
#                     encoded_ = model_output.p_reps.cpu().detach().numpy()
#                 elif dataset_type == "query":
#                     model_output: DROutput = model(query=batch)
#                     encoded_ = model_output.q_reps.cpu().detach().numpy()
#                 else:
#                     raise ValueError(f"dataset_type: {dataset_type} is not valid.")
                
#                 # logging.info(f"you got {encoded_.dtype} for embedding")
                
#                 encoded.append(encoded_)
        
#         if len(lookup_indices) >= args.max_inmem_docs // args.world_size:
#             if split_save:
#                 encoded = np.concatenate(encoded)
#                 with open(
#                     os.path.join(
#                         args.output_dir,
#                         "embeddings.{}.rank.{}.{}-{}".format(
#                             dataset_type,
#                             args.process_index, 
#                             prev_idx, idx
#                         ),
#                     ),
#                     "wb",
#                 ) as f:
#                     pickle.dump((encoded, lookup_indices), f, protocol=4)
#                 encoded = []
#                 lookup_indices = []
#                 prev_idx = idx
#                 gc.collect()

#     # this is to handle the last batch
#     if len(lookup_indices) > 0:
#         if split_save:
#             encoded = np.concatenate(encoded)
#             with open(
#                 os.path.join(
#                     args.output_dir,
#                     "embeddings.{}.rank.{}.{}-{}".format(
#                         dataset_type,
#                         args.process_index, 
#                         prev_idx, idx
#                     ),
#                 ),
#                 "wb",
#             ) as f:
#                 pickle.dump((encoded, lookup_indices), f, protocol=4)

#     # if save to a whole file (each rank only save one file)
#     if not split_save:
#         encoded = np.concatenate(encoded)
#         with open(
#             os.path.join(
#                 args.output_dir,
#                 "embeddings.{}.rank.{}".format(
#                     dataset_type,
#                     args.process_index, 
#                     # prev_idx, idx
#                 ),
#             ),
#             "wb",
#         ) as f:
#             pickle.dump((encoded, lookup_indices), f, protocol=4)
    
#     del encoded
#     del lookup_indices

#     if args.world_size > 1:
#         torch.distributed.barrier()

#     return
