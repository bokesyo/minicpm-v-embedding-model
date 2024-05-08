import logging
import os
import sys

from transformers import AutoConfig, AutoTokenizer, HfArgumentParser

from openmatch.arguments import DataArguments
from openmatch.arguments import InferenceArguments as EncodingArguments
from openmatch.arguments import ModelArguments
from openmatch.dataset import InferenceDataset
from openmatch.modeling import DRModelForInference
from openmatch.retriever import Retriever

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, EncodingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, encoding_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
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
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed inference: %s, 16-bits inference: %s",
        encoding_args.local_rank,
        encoding_args.device,
        encoding_args.n_gpu,
        bool(encoding_args.local_rank != -1),
        encoding_args.fp16,
    )
    logger.info("Encoding parameters %s", encoding_args)
    logger.info("MODEL parameters %s", model_args)

    num_labels = 1
    try:
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            cache_dir=model_args.cache_dir,
        )
    except OSError:
        config = None

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )

    model = DRModelForInference.build(
        model_args=model_args,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    corpus_dataset = InferenceDataset.load(
        tokenizer=tokenizer,
        data_args=data_args,
        mode="raw",
        is_query=False,
        stream=False,
        cache_dir=model_args.cache_dir,
    )

    retriever = Retriever.from_embeddings(model, encoding_args)
    print("Enter your query below. Press Ctrl+D to stop.")
    while True:
        try:
            query = input(">>> ")
        except EOFError:
            print("Exiting")
            break
        result = retriever.retrieve(
            query=query,
            tokenizer=tokenizer,
            doc_id_to_doc=corpus_dataset,
            topk=encoding_args.retrieve_depth,
        )
        result_list = sorted(result.items(), key=lambda x: x[1]["score"], reverse=True)
        for rank, (doc_id, info) in enumerate(result_list):
            print(f"{rank + 1}\t{doc_id}\t{info}")


if __name__ == "__main__":
    main()
