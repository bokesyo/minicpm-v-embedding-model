# Adapted from Tevatron (https://github.com/texttron/tevatron)

import logging
import os
import sys
import json

from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, set_seed

from openmatch.arguments import DataArguments
from openmatch.arguments import DRTrainingArguments as TrainingArguments
from openmatch.arguments import ModelArguments
# from openmatch.dataset import MappingDRTrainDataset, QPCollator, StreamDRTrainDataset
from openmatch.dataset import MappingMMDRTrainDataset, StreamMMDRTrainDataset, MMQPCollator
from openmatch.modeling import DRModel
from openmatch.trainer import DRTrainer as Trainer
# from openmatch.trainer import MMDRTrainer as Trainer
from openmatch.trainer import GCDenseTrainer
# from openmatch.utils import get_delta_model_class
import torch

logger = logging.getLogger(__name__)



def pad(orig_items, key, max_length=None, padding_value=0, padding_side="left"):
    items = []
    if isinstance(orig_items[0][key], list):
        assert isinstance(orig_items[0][key][0], torch.Tensor)
        for it in orig_items:
            for tr in it[key]:
                items.append({key: tr})
    else:
        assert isinstance(orig_items[0][key], torch.Tensor)
        items = orig_items

    batch_size = len(items)
    shape = items[0][key].shape
    dim = len(shape)
    assert dim <= 3
    if max_length is None:
        max_length = 0
    max_length = max(max_length, max(item[key].shape[-1] for item in items))
    min_length = min(item[key].shape[-1] for item in items)
    dtype = items[0][key].dtype

    if dim == 1:
        return torch.cat([item[key] for item in items], dim=0)
    elif dim == 2:
        if max_length == min_length:
            return torch.cat([item[key] for item in items], dim=0)
        tensor = torch.zeros((batch_size, max_length), dtype=dtype) + padding_value
    else:
        tensor = (
            torch.zeros((batch_size, max_length, shape[-1]), dtype=dtype)
            + padding_value
        )

    for i, item in enumerate(items):
        if dim == 2:
            if padding_side == "left":
                tensor[i, -len(item[key][0]) :] = item[key][0].clone()
            else:
                tensor[i, : len(item[key][0])] = item[key][0].clone()
        elif dim == 3:
            if padding_side == "left":
                tensor[i, -len(item[key][0]) :, :] = item[key][0].clone()
            else:
                tensor[i, : len(item[key][0]), :] = item[key][0].clone()

    return tensor

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    print(f"training_args.local_rank = {training_args.local_rank}")
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    num_labels = 1
    # config = AutoConfig.from_pretrained(
    #     model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    #     num_labels=num_labels,
    #     # cache_dir=model_args.cache_dir,
    #     trust_remote_code=True,
    # )
    
    # hacked config
    config_json = json.load(open(os.path.join(model_args.model_name_or_path, 'config.json')))
    if True in ["MiniCPMV" in arch for arch in config_json["architectures"]]: # in base model config.json
        logging.info("using MiniCPMV model tokenizer, load tokenizer from openmatch.modeling.modeling_minicpmv")
        from openmatch.modeling.modeling_minicpmv.modeling_minicpmv import LlamaTokenizerWrapper # what we use in DRTrainer
        # model_class = MiniCPMVForMMEmbedding
        tokenizer_cls = LlamaTokenizerWrapper
    else:
        tokenizer_cls = AutoTokenizer

    tokenizer = tokenizer_cls.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False, 
    )
    
    model = DRModel.build(
        model_args=model_args,
        data_args=data_args,
        train_args=training_args,
        cache_dir=model_args.cache_dir
    )

    # streaming or not
    # if data_args.dataset_class == "text":
    #     train_dataset_cls = (
    #         MappingDRTrainDataset if training_args.use_mapping_dataset else StreamDRTrainDataset
    #     )
    # elif data_args.dataset_class == "multimodal":
    train_dataset_cls = ( # for multimodal dense retrieval
        MappingMMDRTrainDataset if training_args.use_mapping_dataset else StreamMMDRTrainDataset
    )
    # else:
    #     raise NotImplementedError("dataset_class not supported.")
    
    train_dataset = train_dataset_cls(
        tokenizer,
        data_args,
        cache_dir=data_args.data_cache_dir or model_args.cache_dir,
    )
    
    logger.info(f"DataArgs: {data_args}")
    
    eval_dataset = (
        train_dataset_cls(
            tokenizer,
            data_args,
            is_eval=True,
            cache_dir=data_args.data_cache_dir or model_args.cache_dir,
        )
        if data_args.eval_path is not None
        else None
    )
    
    # Handle data collator (for handling batch truncation and padding)
    # data_collator = None

    # data_collator = MMQPCollator(
    #     tokenizer, max_q_len=1024, max_p_len=1024
    # )

    data_collator = MMQPCollator(tokenizer, max_q_len=data_args.q_max_len, max_p_len=data_args.p_max_len)
    
    # else:
    #     if (data_args.q_max_len != 0) or (data_args.p_max_len != 0):
    #         raise ValueError("--q_max_len and --p_max_len are not supported with collator none, please remove them.")
    # def identity(*args):
    #     return args
    # data_collator = identity
    
    trainer_cls = GCDenseTrainer if training_args.grad_cache else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        model_name_or_path=model_args.model_name_or_path
    )
    train_dataset.trainer = trainer

    trainer.train()
    trainer.save_model()
    if trainer.is_world_process_zero(): # should be LOCAL_RANK=0
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
