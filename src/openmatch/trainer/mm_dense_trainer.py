# Adapted from Tevatron (https://github.com/texttron/tevatron)

import logging
import os
from itertools import repeat
from typing import Any, Dict, List, Optional, Tuple, Union
import shutil

import datasets
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler # for distributed training 

from transformers.file_utils import is_datasets_available
from transformers.trainer import TRAINING_ARGS_NAME, Trainer
from transformers.trainer_pt_utils import IterableDatasetShard

from ..loss import DistributedContrastiveLoss, SimpleContrastiveLoss

logger = logging.getLogger(__name__)

try:
    from grad_cache import GradCache
    _grad_cache_available = True
except ModuleNotFoundError:
    _grad_cache_available = False


# def reshape_input(data):
#     result = {}
#     for d in data:
#         for key, value in d.items():
#             data_list = result.get(key, [])
#             data_list.append(value)
#             result[key] = data_list
#     for key, value in result.items():
#         result[key] = torch.stack(value)
#     return result

def to_device(data, device):
    """
    Recursively move tensors in a nested list, tuple, or dictionary to the specified device.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(to_device(item, device) for item in data)
    else:
        return data  # 如果不是上述类型，则原样返回


class MMDRTrainer(Trainer):
    def __init__(
        self, 
        *args, 
        model_name_or_path,
        **kwargs,
        
    ):
        super(MMDRTrainer, self).__init__(*args, **kwargs)
        self._dist_loss_scale_factor = dist.get_world_size() if self.args.negatives_x_device else 1
        
        self.model_name_or_path = model_name_or_path
        # process_rank and world_size work well for multi nodes and multi gpu setting
        # for example, 4 nodes, each node has 8 gpus, then each node has 8 processes, each process control one gpu.
        # for node 2, gpu 3, the get_rank will return rank=2*8+3
        self.process_rank = dist.get_rank()
        # world_size = 4*8 = 32
        self.world_size = dist.get_world_size()
        
        self.metric_hook = {
            "accuracy": [],
        }
        
        # rank = int(os.getenv('RANK', '0'))
        # node_id = int(os.getenv('NODE_RANK', '0'))
        # gpu_id = torch.cuda.current_device()
        
        # print(f"---> rank={rank}, node_id={node_id}, gpu_id={gpu_id}")

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        
        # this state_dict is contructed by deepspeed or others, not the model we know.
        # because the parameter of each process is not the full parameter
        
        if state_dict is not None: # with deepspeed training
            # step1: remove prefix "lm_q."
            
            if self.model.base_model_arch == "SmartCPMModel": # Text embedding
                state_dict = {k.replace("lm_q", "model"): v for k, v in state_dict.items()}
            if self.model.base_model_arch == "MiniCPMVForMMEmbedding": # Multimodal embedding
                state_dict = {k.replace("lm_q", ""): v for k, v in state_dict.items()}
            elif "Mistral" in self.model.base_model_arch:
                state_dict = {k.replace("lm_q", "model"): v for k, v in state_dict.items()}
            elif "Bert" in self.model.base_model_arch:
                state_dict = {k.replace("lm_q", "bert"): v for k, v in state_dict.items()}
            elif "QWen" in self.model.base_model_arch:
                state_dict = {k.replace("lm_q", "transformer"): v for k, v in state_dict.items()}
            elif "Llama" in self.model.base_model_arch:
                state_dict = {k.replace("lm_q", "model"): v for k, v in state_dict.items()}
            else:
                raise NotImplementedError
            
            # step2: save this state_dict as model paramters
            super(MMDRTrainer, self)._save(output_dir=output_dir, state_dict=state_dict)
            
            # step3: copy config file (important for identifying model arch)
            config_file_path = os.path.join(self.model_name_or_path, 'config.json')
            shutil.copy(config_file_path, output_dir)
        
        else: # normal training 
            # step1: save model params
            self.model.lm_q.save_pretrained(output_dir)
            # step2: save tokenizer
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)
            # step3: save training args
            torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
    
    def _prepare_inputs(
        self, inputs: Tuple[Dict[str, Union[torch.Tensor, Any]], ...]
    ) -> Tuple[Dict[str, Union[torch.Tensor, Any]]]:
        # move to device
        query, passages = inputs
        query = to_device(query, self.args.device)
        passages = to_device(passages, self.args.device)
        
        return query, passages

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `self.train_dataset` does not implement `__len__`, a random sampler (adapted to
        distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            
            print(f"self.args.train_batch_size = {self.args.train_batch_size}")
            print(f"self.world_size = {self.world_size}")
            print(f"self.process_rank = {self.process_rank}")
            
            if self.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self.args.train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.world_size, # this world_size is global (multi gpu and multi nodes).
                    process_index=self.process_rank, # this process_rank is global (multi gpu and multi nodes).
                )
                
                # No worry! IterableDatasetShard will not duplicate for each process, though the dataset.process_fn will be called for each data point on each process!

            return DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
        else:
        
            # train_sampler = self._get_train_sampler() # what is this sampler? distributed? [exception: this does not work for distributed training]
            
            train_sampler = DistributedSampler(
                train_dataset, 
                # shuffle=False, # each epoch will shuffle once by default
                num_replicas=self.world_size, 
                rank=self.process_rank
            )
            
            return DataLoader(
                train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=train_sampler,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

    def log(self, log_dict): # overwrite Trainer.log [hacked]
        if len(self.metric_hook["accuracy"]) != 0:
            log_dict["accuracy"] = sum(self.metric_hook["accuracy"]) / len(self.metric_hook["accuracy"])
            self.metric_hook["accuracy"] = []
            
        # we need to restore the true global CrossEntropyLoss (global batch)
        # when logging
        if "loss" in log_dict:
            log_dict["loss"] = log_dict["loss"] / self.world_size
                
        super().log(log_dict)
    
    def compute_loss(self, model, inputs, return_outputs=False): # here inputs is from _prepare_data_for
        # print(f"self.args.distillation = {self.args.distillation}")
        # print(f"self.args.distil_mode = {self.args.distil_mode}")
        if self.args.distillation:
            raise NotImplementedError
            # print("distillation = True")
            # if self.args.distil_mode == "pairwise":
            #     query, positive, negative, score = inputs
            #     outputs = model(query=query, positive=positive, negative=negative, score=score)
            # else:  # listwise
            #     query, passage, score = inputs
            #     outputs = model(query=query, passage=passage, score=score)
        else:
            # print(inputs)
            query, passage = inputs
            # print(f"query = {query['input_ids'].shape}") # [batch_size, max_query_length]
            # print(f"passage = {passage['input_ids'].shape}") # [batch_size * n_passages(negative), max_passage_length]
            outputs = model(query=query, passage=passage)
            
            # hack metric logging
            self.metric_hook["accuracy"].append(outputs.accuracy.item())
            
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def training_step(self, *args):
        # logger.info(f"training_step invoked!")
        loss = super(MMDRTrainer, self).training_step(*args)
        return loss


# def split_dense_inputs(model_input: dict, chunk_size: int):
    
#     assert len(model_input) == 1
#     arg_key = list(model_input.keys())[0]
#     arg_val = model_input[arg_key]

#     keys = list(arg_val.keys())
#     chunked_tensors = [arg_val[k].split(chunk_size, dim=0) for k in keys]
#     chunked_arg_val = [dict(zip(kk, tt)) for kk, tt in zip(repeat(keys), zip(*chunked_tensors))]

#     return [{arg_key: c} for c in chunked_arg_val]


# def get_dense_rep(x):
#     if x.q_reps is None:
#         return x.p_reps
#     else:
#         return x.q_reps


# class GCDenseTrainer(DRTrainer):
#     pass
#     def __init__(self, *args, **kwargs):
#         logger.info("Initializing Gradient Cache Trainer")
#         if not _grad_cache_available:
#             raise ValueError(
#                 "Grad Cache package not available. You can obtain it from https://github.com/luyug/GradCache."
#             )
#         super(GCDenseTrainer, self).__init__(*args, **kwargs)

#         loss_fn_cls = (
#             DistributedContrastiveLoss if self.args.negatives_x_device else SimpleContrastiveLoss
#         )
#         loss_fn = loss_fn_cls()

#         self.gc = GradCache(
#             models=[self.model, self.model],
#             chunk_sizes=[self.args.gc_q_chunk_size, self.args.gc_p_chunk_size],
#             loss_fn=loss_fn,
#             split_input_fn=split_dense_inputs,
#             get_rep_fn=get_dense_rep,
#             fp16=self.args.fp16,
#             scaler=self.scaler,
#         )

#     def training_step(self, model, inputs) -> torch.Tensor:
#         model.train()
#         queries, passages = self._prepare_inputs(inputs)
#         queries, passages = {"query": queries}, {"passage": passages}

#         _distributed = self.args.local_rank > -1
#         self.gc.models = [model, model]
#         loss = self.gc(queries, passages, no_sync_except_last=_distributed)

#         return loss / self._dist_loss_scale_factor  # this is the real global CE loss value?


