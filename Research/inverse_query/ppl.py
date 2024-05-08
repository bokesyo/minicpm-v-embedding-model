# This script is designed for Batched Inference (ppl) with PretrainedModel with streaming jsonl datasets.

# Input format: Jsonl, each line {"text": "xxxxxxx"}

import logging
import os
import sys
from pathlib import Path
import glob
import csv
import json
from tqdm import tqdm
import torch
import torch.distributed as dist
from dataclasses import dataclass, field
from transformers import AutoConfig, AutoTokenizer, AutoModel, PreTrainedTokenizer, AutoModelForCausalLM
from transformers import HfArgumentParser, TrainingArguments
from openmatch.dataset import InferenceDataset
from torch.utils.data import DataLoader
from contextlib import nullcontext
from torch.cuda import amp

import gc


logger = logging.getLogger(__name__)

@dataclass
class PPLArguments:
    model_name_or_path: str = field(
        default=None, 
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: str = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    data_dir: str = field(
        default=None,
        metadata={"help": "Input data dir"},
    )

# 将输入字典转换为目标格式的函数
def reshape_dict(input_dict):
    # 获取字典中第一个键的值的长度，假设所有列表长度一致
    length = len(next(iter(input_dict.values())))
    # 创建空列表，用于存储转换后的字典
    reshaped_list = []
    for i in range(length):
        # 对于每个索引，创建一个新字典，从每个键的列表中取出对应元素
        new_dict = {key: value[i] for key, value in input_dict.items()}
        reshaped_list.append(new_dict)
    return reshaped_list

# 检查文件大小
def is_file_size_over_limit(file_path, limit_mb):
    return os.path.exists(file_path) and os.path.getsize(file_path) > limit_mb * 1024 * 1024

def ppl(
    model,
    tokenizer,
    loss_fct,
    batch,
    args,
):
    
    CELOSS_MASK_TOKEN_LABEL_ID = torch.nn.CrossEntropyLoss().ignore_index
    
    tokenized = tokenizer(
        batch["text"], 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    
    # for k, v in tokenized.items():
    #     tokenized[k] = v.to(args.device)
    
    output = model(**{k: v.to(args.device) for k, v in tokenized.items()})
    logit = output['logits']
    
    print(f"logit: {logit.shape}")
    
    # if pad_token_initialized_diy:
    #     logit = logit[:, :, :-1] # remove the last element in logits (pad token)
    #     print(f"logit-pad_token_initialized_diy: {logit.shape}")
        
    label = tokenized['input_ids']
    print(f"label: {label.shape}")
    
    # mask out the pad_token_id in label by -100
    label[label == tokenizer.pad_token_id] = CELOSS_MASK_TOKEN_LABEL_ID
    
    # shift so that tokens < n predict n
    shift_logits = logit[..., :-1, :].contiguous()
    print(f"shift_logits: {shift_logits.shape}")
    
    # label move to device here 
    shift_label = label[:, 1:].contiguous().to(args.device)
    print(f"shift_label: {shift_label.shape}")
    
    # compute loss
    valid_length = (shift_label != CELOSS_MASK_TOKEN_LABEL_ID).sum(dim=-1)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_label.view(-1))
    print(f"loss: {loss.shape}")
    loss = loss.view(len(output['logits']), -1)
    print(f"loss -view : {loss.shape}")
    loss = torch.sum(loss, -1) / valid_length
    print(f"print -sum : {loss.shape}")

    # loss is ppl here
    ppl = loss.cpu().tolist()
    
    del loss
    del shift_label
    del shift_logits
    del label
    del logit
    del output
    del tokenized
    
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"ppl = {ppl}")

    batch["ppl"] = ppl
    del batch["text"] # delete image base64 string to save space. (temperal)
    blobs = reshape_dict(batch) # reshape the blob
    
    return blobs


def distributed_parallel_ppl_inference(
    dataset: InferenceDataset,
    model: object,
    tokenizer: PreTrainedTokenizer,
    data_output_dir: str,
    args: TrainingArguments
):
    # Note: during evaluation, there's no point in wrapping the model
    # inside a DistributedDataParallel as we'll be under `no_grad` anyways.
    if dataset is None:
        raise ValueError("No dataset provided")
    
    logger.info("initializing distributed dataloader")
    dataloader = DataLoader(
        dataset, # this dataset can be sharded (data parallel)
        batch_size=args.per_device_eval_batch_size,
        collate_fn=None, # for qwen_vl_chat inference, no need to collate
        num_workers=args.dataloader_num_workers,
        pin_memory=args.dataloader_pin_memory,
        drop_last=False, # we don't want to drop the last batch, this is evaluation
    )
    logger.info("distributed dataloader ok.")

    os.makedirs(data_output_dir, exist_ok=True)

    file_count = 1
    MAX_OUTPUT_SIZE_MB = 8 # 8MB
    
    # this is -100 if I am right
    CELOSS_MASK_TOKEN_LABEL_ID = torch.nn.CrossEntropyLoss().ignore_index
    logging.info(f"CELOSS_MASK_TOKEN_LABEL_ID = {CELOSS_MASK_TOKEN_LABEL_ID}")
    
    # loss function
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    
    # if no predefined pad token, set one
    # pad_token_initialized_diy = False 
    if tokenizer.pad_token is None: # if there is no padding token set, we DIY a pad token
        # tokenizer.add_special_tokens({'pad_token': "<unk>"})
        # model.resize_token_embeddings(len(self.tokenizer))
        # pad_token_initialized_diy = True
        raise ValueError("No padding token in tokenizer, please add one.")

    with amp.autocast() if args.fp16 else nullcontext():
        with torch.no_grad():
            for batch in tqdm(dataloader, disable=args.process_index > 0):
                
                blobs = ppl(model, tokenizer, loss_fct, batch, args)

                current_file = f"{data_output_dir}/{args.local_rank}-{file_count}.jsonl"
                if is_file_size_over_limit(current_file, MAX_OUTPUT_SIZE_MB):
                    file_count += 1
                    current_file = f"{data_output_dir}/{args.local_rank}-{file_count}.jsonl"
                    
                with open(current_file, "a", encoding='utf-8') as f:
                    for blob in blobs:
                        print(blob)
                        f.write(json.dumps(blob, ensure_ascii=False))
                        f.write('\n')
    
    if args.world_size > 1:
        torch.distributed.barrier()

    return


def main():
    parser = HfArgumentParser((PPLArguments, TrainingArguments))
    
    self_args, encoding_args = parser.parse_args_into_dataclasses()
    
    if os.path.exists(encoding_args.output_dir) and os.listdir(encoding_args.output_dir):
        if not encoding_args.overwrite_output_dir:
            logger.warning(
                f"Output directory ({encoding_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            )
        else:
            # remove all files in the output directory
            if encoding_args.local_process_index == 0:
                for file in os.listdir(encoding_args.output_dir):
                    os.remove(os.path.join(encoding_args.output_dir, file))

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if encoding_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed inference: %s, 16-bits inference: %s",
        encoding_args.local_rank,
        encoding_args.device,
        encoding_args.n_gpu,
        bool(encoding_args.local_rank != -1),
        encoding_args.fp16,
    )

    tokenizer_cls = AutoTokenizer
    tokenizer = tokenizer_cls.from_pretrained(
        self_args.model_name_or_path
    )
    model_cls = AutoModelForCausalLM

    logging.info("1 - load dataset begin..")
    
    doc_dataset = InferenceDataset.load(
        tokenizer=tokenizer,
        data_args=None,
        data_files=os.path.join(self_args.data_dir, "data.jsonl"),
        # full_tokenization=True,
        mode="raw",
        stream=True,
        batch_size=encoding_args.per_device_eval_batch_size,
        num_processes=encoding_args.world_size,
        process_index=encoding_args.process_index,
    )
    logging.info("1 - load dataset end..")

    logging.info("2 - load model begin..")
    model = model_cls.from_pretrained(
        self_args.model_name_or_path, 
        trust_remote_code=True,
        # config=config,
    )
    logging.info("2 - load model end..")

    # inference mode
    print(f"to encoding_args.device = {encoding_args.device}")
    model.to(encoding_args.device)
    model.eval()

    distributed_parallel_ppl_inference(
        dataset=doc_dataset,
        model=model,
        tokenizer=tokenizer,
        data_output_dir=encoding_args.output_dir,
        args=encoding_args,
    )

    if encoding_args.world_size > 1:
        torch.distributed.barrier()


if __name__ == "__main__":
    main()





