import logging
import os
import sys
from tqdm import tqdm
from contextlib import nullcontext
import pickle
import gc

import numpy as np
import torch
from torch.cuda import amp
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset

from transformers import AutoConfig, AutoProcessor, AutoTokenizer, HfArgumentParser

from openmatch.arguments import DataArguments
from openmatch.arguments import InferenceArguments as EncodingArguments
from openmatch.arguments import ModelArguments
from openmatch.dataset import InferenceDataset
from openmatch.modeling import DRModelForInference, DROutput
from openmatch.retriever import Retriever
from ..dataset import DRInferenceCollator
from openmatch.utils import get_delta_model_class


logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, EncodingArguments))
    # if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    #     model_args, data_args, encoding_args = parser.parse_json_file(
    #         json_file=os.path.abspath(sys.argv[1])
    #     )
    # else:    
    model_args, data_args, encoding_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    encoding_args: EncodingArguments

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if encoding_args.local_rank in [-1, 0] else logging.WARN,
    )
    
    dist.init_process_group(backend="nccl")
    assert dist.is_initialized()
    
    world_size = dist.get_world_size()
    local_rank = dist.get_rank()
    
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed inference: %s, 16-bits inference: %s",
        local_rank,
        encoding_args.device,
        encoding_args.n_gpu,
        bool(local_rank != -1),
        encoding_args.fp16,
    )
    
    logger.info("Encoding parameters %s", encoding_args)
    logger.info("MODEL parameters %s", model_args)

    # here we only load tokenizer and processor
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    
    try:
        processor = AutoProcessor.from_pretrained(
            model_args.processor_name
            if model_args.processor_name
            else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )
    except (ValueError, OSError):
        processor = None

    model = DRModelForInference.build(
        model_args=model_args,
        cache_dir=model_args.cache_dir,
    )

    # if model_args.param_efficient_method:
    #     model_class = get_delta_model_class(model_args.param_efficient_method)
    #     delta_model = model_class.from_finetuned(
    #         model_args.model_name_or_path + "/delta_model", model, local_files_only=True
    #     )
    #     logger.info("Using param efficient method: %s", model_args.param_efficient_method)

    corpus_dataset = InferenceDataset.load(
        tokenizer=tokenizer,
        processor=processor,
        data_args=data_args,
        is_query=False,
        stream=True,
        batch_size=encoding_args.per_device_eval_batch_size,
        num_processes=world_size,
        process_index=local_rank,
        cache_dir=model_args.cache_dir,
    )

    Retriever.build_embeddings(model, corpus_dataset, encoding_args)

    # corpus_dataloader = DataLoader(
    #     corpus_dataset,
    #     # Note that we do not support DataParallel here
    #     batch_size=encoding_args.per_device_eval_batch_size,
    #     collate_fn=DRInferenceCollator(),
    #     num_workers=encoding_args.dataloader_num_workers,
    #     pin_memory=encoding_args.dataloader_pin_memory,
    # )

    # os.makedirs(encoding_args.output_dir, exist_ok=True)
    
    # encoded = []
    # lookup_indices = []
    # idx = 0
    # prev_idx = 0
    # for batch_ids, batch in tqdm(corpus_dataloader, disable=local_rank > 0):
    #     lookup_indices.extend(batch_ids)
    #     idx += len(batch_ids)
    #     with amp.autocast() if encoding_args.fp16 else nullcontext():
    #         with torch.no_grad():
    #             for k, v in batch.items():
    #                 batch[k] = v.to(encoding_args.device)
    #             model_output: DROutput = model(passage=batch)
    #             encoded.append(model_output.p_reps.cpu().detach().numpy())
    #     if len(lookup_indices) >= encoding_args.max_inmem_docs // world_size:
    #         encoded = np.concatenate(encoded)
    #         with open(
    #             os.path.join(
    #                 encoding_args.output_dir,
    #                 "embeddings.corpus.rank.{}.{}-{}".format(
    #                     local_rank, prev_idx, idx
    #                 ),
    #             ),
    #             "wb",
    #         ) as f:
    #             pickle.dump((encoded, lookup_indices), f, protocol=4)
    #         encoded = []
    #         lookup_indices = []
    #         prev_idx = idx
    #         gc.collect() # clear cache
    
    # # the last batch
    # if len(lookup_indices) > 0:
    #     encoded = np.concatenate(encoded)
    #     with open(
    #         os.path.join(
    #             encoding_args.output_dir,
    #             "embeddings.corpus.rank.{}.{}-{}".format(
    #                 local_rank, prev_idx, idx # rank, and begin-end index
    #             ),
    #         ),
    #         "wb",
    #     ) as f:
    #         pickle.dump((encoded, lookup_indices), f, protocol=4)

    # del encoded
    # del lookup_indices

    # if world_size > 1:
    #     torch.distributed.barrier()


if __name__ == "__main__":
    main()
