# Adapted from Tevatron (https://github.com/texttron/tevatron)

import copy
import importlib
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import (
    AutoConfig,
    AutoModel,
    BatchEncoding,
    PreTrainedModel,
    T5EncoderModel,
)

from transformers.modeling_outputs import ModelOutput

from ..arguments import DataArguments
from ..arguments import DRTrainingArguments as TrainingArguments
from ..arguments import ModelArguments
from ..utils import mean_pooling
from .linear import LinearHead

logger = logging.getLogger(__name__)


# torch.set_printoptions(threshold=10_000)
# torch.set_printoptions(threshold=10_000)  # 设置阈值为10000，对于大多数情况下应该足够

# For decoder-only models
def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


@dataclass
class DROutput(ModelOutput):
    q_reps: Tensor = None
    p_reps: Tensor = None
    loss: Tensor = None
    scores: Tensor = None
    accuracy: Tensor = None # loss绝对值不可参考，因为loss与temperature有关


class DRModel(nn.Module):
    def __init__(
        self,
        lm_q: PreTrainedModel,
        # lm_p: PreTrainedModel,
        # tied: bool = True,
        feature: str = "last_hidden_state",
        pooling: str = "lasttoken",
        attention: str = "causal",
        head_q: nn.Module = None,
        head_p: nn.Module = None,
        normalize: bool = False,
        model_args: ModelArguments = None,
        data_args: DataArguments = None,
        train_args: TrainingArguments = None,
        base_model_arch: str = "Llama",
        # model_name_or_path: str = "",
    ):
        super().__init__()

        # self.tied = tied
        self.lm_q = lm_q
        # self.lm_p = lm_p
        self.head_q = head_q
        self.head_p = head_p

        self.feature = feature
        self.pooling = pooling
        self.normalize = normalize
        self.attention = attention

        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args
        
        self.base_model_arch = base_model_arch
        # self.model_name_or_path = model_name_or_path

        if train_args is not None:
            if train_args.distillation:
                self.loss_fn = (
                    nn.MSELoss() if train_args.distil_mode == "pairwise" else nn.KLDivLoss()
                )
            else:
                self.loss_fn = nn.CrossEntropyLoss(reduction="mean")

            if train_args.negatives_x_device:
                if not dist.is_initialized():
                    raise ValueError(
                        "Distributed training has not been initialized for representation all gather."
                    )
                
                # process_rank and world_size work well for multi nodes and multi gpu setting
                # for example, 4 nodes, each node has 8 gpus, then each node has 8 processes, each process control one gpu.
                # for node 2, gpu 3, the get_rank will return rank=2*8+3
                self.process_rank = dist.get_rank()
                
                print(f"process_rank = {self.process_rank}")
                
                # world_size = 4*8 = 32
                self.world_size = dist.get_world_size()
                
                print(f"world_size = {self.world_size}")
        else:
            # raise ValueError("Please specify train_args.")
            logger.info("train_args not specified.")
        
        return

    def _get_config_dict(self):
        config = {
            # "tied": self.tied,
            "plm_backbone": {
                "type": type(self.lm_q).__name__,
                "feature": self.feature,
            },
            "pooling": self.pooling,
            "linear_head": bool(self.head_q),
            "normalize": self.normalize,
        }
        return config

    def forward(
        self,
        query: Dict[str, Tensor] = None,
        passage: Dict[str, Tensor] = None,
    ):  

        if query is not None:
            _, q_reps = self.encode_query(query)
        else:
            q_reps = None
        
        if passage is not None:
            if self.train_args.passage_stop_grad: # 可以选择对passage的梯度停止，inspired by GritLM & Kaiming He's Siamese network
                with torch.no_grad():
                    logger.info("you are stopping gradient of passages")
                    _, p_reps = self.encode_passage(passage)
            else:
                _, p_reps = self.encode_passage(passage)
        else:
            p_reps = None

        return DROutput(q_reps=q_reps, p_reps=p_reps)


    def encode(self, items, model, head, is_q=False):
        if items is None:
            return None, None # for Inference
        
        # MistralModel
        
        # E5-Mistral implementation
        # decoder_only_model_arch = ["CPM", "Qwen", "Llama", "GPT", "Baichuan", "Mistral"]
        # logger.info(f"in encode self.base_model_arch = {self.base_model_arch}")
        # logger.info([arch in self.base_model_arch for arch in decoder_only_model_arch])
        # if True in [arch in self.base_model_arch for arch in decoder_only_model_arch]:
            # print("decoder-only model")
            # print(f"input_ids.shape = {items['input_ids'].shape}")
        
        # # use causal or bidirectional attention
        # if self.attention == "bidirectional":
        #     items["is_causal"] = False
        # elif self.attention == "causal":
        #     items["is_causal"] = True
        # else:
        #     raise ValueError(f"attention type {self.attention} is not valid")
        # # by default, is_causal is True
        
        # print(f"actual input = {items['input_ids'].shape}")
        
        # print(items)
        
        items_out = model(**items, return_dict=True)
            # print(f"items_out = {items_out}")
            # print(f"items_out.shape = {items_out.keys()}")
            # print(f"last_hidden_state.shape = {items_out['last_hidden_state'].shape}")
        
        hidden = getattr(items_out, self.feature) # usually "last_hidden_state", if no exist, error will happen
        
        if self.pooling == "lasttoken":
            attention_mask = getattr(items, "attention_mask")
            
            # print(f"attention_mask = {attention_mask}")
            
            reps = last_token_pool(
                last_hidden_states=hidden,
                attention_mask=attention_mask
            )
        elif self.pooling == "simple_lasttoken":
            reps = hidden[:, -1, :]
        elif self.pooling == "wmean":
            attention_mask = getattr(items, "attention_mask")
            attention_mask_ = attention_mask * attention_mask.cumsum(dim=1) # [0,1,1,1,0,0] -> [0,1,2,3,0,0]
            # print(attention_mask.shape)
            # print(attention_mask_.shape)
            # print(attention_mask_)
            s = torch.sum(hidden * attention_mask_.unsqueeze(-1).float(), dim=1)
            d = attention_mask_.sum(dim=1, keepdim=True).float()
            reps = s / d
        elif self.pooling == "drop_wmean":
            vector_dropout = nn.Dropout1d(0.3)
            attention_mask = getattr(items, "attention_mask")
            attention_mask_ = attention_mask * attention_mask.cumsum(dim=1) # [0,1,1,1,0,0] -> [0,1,2,3,0,0]
            # print(attention_mask.shape)
            # print(attention_mask_.shape)
            # print(attention_mask_)
            hidden_masked = hidden * attention_mask_.unsqueeze(-1).float()
            hidden_masked  = vector_dropout(hidden_masked)
            s = torch.sum(hidden_masked, dim=1)
            d = attention_mask_.sum(dim=1, keepdim=True).float()
            reps = s / d
        elif self.pooling == "drop_mean":
            vector_dropout = nn.Dropout1d(0.3)
            attention_mask = getattr(items, "attention_mask")
            # print(attention_mask.shape)
            # print(attention_mask_.shape)
            # print(attention_mask_)
            hidden_masked = hidden * attention_mask.unsqueeze(-1).float()
            hidden_masked  = vector_dropout(hidden_masked)
            s = torch.sum(hidden_masked, dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            reps = s / d
        elif self.pooling == "mean":
            attention_mask = getattr(items, "attention_mask")
            s = torch.sum(hidden * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            reps = s / d
        elif self.pooling == "cls":
            # Bert-style model, use cls
            reps = hidden[:, 0, :]
        else:
            raise ValueError("Unknown pooling type: {}".format(self.pooling))

        # # Another implementation
        # if True in [arch in type(model).__name__ for arch in decoder_only_model_arch]:
        #     items_out = model(**items, return_dict=True)
        #     if hasattr(items_out, "last_hidden_state"):
        #         attention_mask = items["attention_mask"]
        #         # attention_mask_sum = torch.sum(attention_mask, dim=1)
        #         # print(attention_mask_sum.dtype)
        #         # print(attention_mask.shape)
        #         hidden = items_out.last_hidden_state
        #         # reps = hidden[:, -1, :]
        #         seq_lengths = attention_mask.sum(dim=1) - 1
        #         last_token_indices = seq_lengths.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        #         # Gather the last valid token representations
        #         last_valid_tokens = hidden.gather(1, last_token_indices).squeeze(1)
        #         reps = last_valid_tokens
        #         # print(reps)
        #         # print(f"hidden = {hidden.shape}")
        #         # print(f"reps = {reps.shape}")
        #     else:
        #         raise NotImplementedError
        
        # elif "CLIP" in type(model).__name__:
        #     reps = hidden = items_out = (
        #         model.get_text_features(**items, return_dict=True)
        #         if is_q
        #         else model.get_image_features(**items, return_dict=True)
        #     )
        
        # else: # Bert, encoder model
        #     items_out = model(**items, return_dict=True)
        #     hidden = getattr(items_out, self.feature)
        #     if self.pooling == "first":
        #         reps = hidden[:, 0, :]
        #     elif self.pooling == "mean":
        #         reps = mean_pooling(hidden, items.attention_mask)
        #     elif self.pooling == "no":
        #         reps = hidden
        #     else:
        #         raise ValueError("Unknown pooling type: {}".format(self.pooling))
        # if head is not None:
        #     reps = head(reps)  # D * d
        
        assert self.normalize == True, "Normalize must be true"
        # if self.normalize:
        reps = F.normalize(reps, dim=1)
        
        return None, reps
        # return hidden, reps

    def encode_passage(self, psg):
        return self.encode(psg, self.lm_q, self.head_p)

    def encode_query(self, qry):
        return self.encode(qry, self.lm_q, self.head_q)

    @classmethod
    def build(
        cls,
        model_args: ModelArguments,
        model_name_or_path: str = None,
        data_args: DataArguments = None,
        train_args: TrainingArguments = None,
        **hf_kwargs,
    ):
        
        model_name_or_path = model_name_or_path or model_args.model_name_or_path
        
        # load local
        config = None
        head_q = head_p = None
        # if os.path.exists(os.path.join(model_name_or_path, "openmatch_config.json")):
        #     with open(os.path.join(model_name_or_path, "openmatch_config.json")) as f:
        #         config = json.load(f)

        # tied = not model_args.untie_encoder
        config_json = json.load(open(os.path.join(model_name_or_path, 'config.json')))
        
        # raise NotImplementedError
        # hacked
        # CPM model
        if True in ["SmartCPMForCausalLM" in arch for arch in config_json["architectures"]]: # in base model config.json
            # print(config_json["architectures"])
            logging.info("using CPM model, load modeling from openmatch.modeling.modeling_smartcpm")
            from openmatch.modeling.modeling_smartcpm.configuration_smartcpm import SmartCPMConfig
            config_cls = SmartCPMConfig
            from openmatch.modeling.modeling_smartcpm.modeling_smartcpm import SmartCPMModel # what we use in DRTrainer
            model_class = SmartCPMModel
        
        # MiniCPMV model
        elif True in ["MiniCPMV" in arch for arch in config_json["architectures"]]: # in base model config.json
            logging.info("using MiniCPMV model, load modeling from openmatch.modeling.modeling_minicpmv")
            from openmatch.modeling.modeling_minicpmv.configuration_minicpm import MiniCPMVConfig
            config_cls = MiniCPMVConfig
            # from openmatch.modeling.modeling_minicpmv.modeling_minicpmv_hf import MiniCPMVForMMEmbedding # what we use in DRTrainer
            from openmatch.modeling.modeling_minicpmv.modeling_minicpmv import MiniCPMVForMMEmbedding # what we use in DRTrainer
            model_class = MiniCPMVForMMEmbedding
        
        else: # other model
            logging.info("using AutoModel model")
            config_cls = AutoConfig
            model_class = AutoModel
            hf_kwargs["trust_remote_code"]=True
        
        logger.info(f"model class = {model_class}")
        
        config = config_cls.from_pretrained(model_name_or_path)
        
        # add attention pattern 
        if model_args.attention == "bidirectional":
            config.is_causal = False
        elif model_args.attention == "causal":
            # config.is_causal = True
            pass
        else:
            raise ValueError(f"attention type {model_args.attention} is not valid")
        
        # Create raw hf model
        lm_q = model_class.from_pretrained(
            model_name_or_path, 
            trust_remote_code=True,
            attn_implementation=model_args.attn_implementation, 
            config=config,
            **hf_kwargs
        )
        
        base_model_arch = type(lm_q).__name__ # in case LoRA will replace class name
        logger.info(f"base model type = {base_model_arch}")
        
        # Inject LoRA if use LoRA
        if model_args.lora:
            logger.info("Using LoRA")
            from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
            
            # FEATURE_EXTRACTION: Feature extraction. 
            # Provides the hidden states which can be used as embeddings or features for downstream tasks.
            # https://huggingface.co/docs/peft/package_reference/peft_types
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION, 
                inference_mode=False, 
                r=model_args.lora_r, lora_alpha=32, lora_dropout=0.1,
                target_modules=["q_proj", "v_proj"],
            )
            
            # Transform the model to LoRA model
            lm_q = get_peft_model(lm_q, peft_config)
            logger.info("trainable parameters")
            lm_q.print_trainable_parameters()
        
        # lm_p = copy.deepcopy(lm_q) if not tied else lm_q
        
        # Finally add linear head 
        if model_args.add_linear_head:
            # head_q = LinearHead(model_args.projection_in_dim, model_args.projection_out_dim)
            raise NotImplementedError
            # head_p = copy.deepcopy(head_q) if not tied else head_q

        model = cls(
            lm_q=lm_q,
            # lm_p=lm_p,
            # tied=tied,
            feature=model_args.feature,
            pooling=model_args.pooling,
            attention=model_args.attention,
            head_q=head_q,
            head_p=head_p,
            normalize=model_args.normalize,
            model_args=model_args,
            data_args=data_args,
            train_args=train_args,
            base_model_arch=base_model_arch,
            # model_name_or_path=model_name_or_path,
        )
        
        return model

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs):
        # logger.info(f"*** gradient_checkpointing_kwargs = {gradient_checkpointing_kwargs}")
        gradient_checkpointing_kwargs["use_reentrant"] = False # handle a bug with DDP
        # tied = not self.model_args.untie_encoder
        # logger.info(f"*** tied = {tied}")
        self.lm_q.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs) # this model should be transformers model
        # if not tied:
        #     self.lm_p.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs) # this model should be transformers model
        return
    
    # def save(self, output_dir: str): # not compatible with deepspeed!
    #     print("save model")
    #     if not self.tied:
    #         # os.makedirs(os.path.join(output_dir, "query_model"))
    #         # os.makedirs(os.path.join(output_dir, "passage_model"))
    #         # self.lm_q.save_pretrained(os.path.join(output_dir, "query_model"))
    #         # self.lm_p.save_pretrained(os.path.join(output_dir, "passage_model"))
    #         # if self.head_q is not None:
    #         #     self.head_q.save(os.path.join(output_dir, "query_head"))
    #         #     self.head_p.save(os.path.join(output_dir, "passage_head"))
    #         raise NotImplementedError
    #     else:
    #         self.lm_q.save_pretrained(output_dir) # work for peft model
    #         if self.head_q is not None:
    #             self.head_q.save(output_dir)

        with open(os.path.join(output_dir, "openmatch_config.json"), "w") as f:
            json.dump(self._get_config_dict(), f, indent=4)

    def dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)] # gpus across all nodes , counted
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors


class DRModelForInference(DRModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval()

    @torch.no_grad()
    def encode_passage(self, psg):
        return super(DRModelForInference, self).encode_passage(psg)

    @torch.no_grad()
    def encode_query(self, qry):
        return super(DRModelForInference, self).encode_query(qry)

    def forward(
        self,
        query: Dict[str, Tensor] = None,
        passage: Dict[str, Tensor] = None,
    ):
        q_hidden, q_reps = self.encode_query(query)

        
        p_hidden, p_reps = self.encode_passage(passage)
        return DROutput(q_reps=q_reps, p_reps=p_reps)

class DRModelForGDCache(DRModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(
        self,
        query: Dict[str, Tensor] = None,
        passage: Dict[str, Tensor] = None,
    ):
        q_hidden, q_reps = self.encode_query(query)
        p_hidden, p_reps = self.encode_passage(passage)
        return DROutput(q_reps=q_reps, p_reps=p_reps)