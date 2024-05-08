# Adapted from Tevatron (https://github.com/texttron/tevatron)

import glob
import logging
import os
import random
from typing import Callable, Dict, List, Union, Optional

from datasets import load_dataset
from torch.utils.data import Dataset, IterableDataset
from transformers import BatchEncoding, PreTrainedTokenizer
from transformers.trainer_pt_utils import IterableDatasetShard

from ..arguments import DataArguments, DRPretrainingDataArguments
from ..data_augmentation_strategy import Cropping, NullStrategy, SequentialStrategies
from ..trainer import DRTrainer

import torch.distributed as dist

import torch

import base64
from PIL import Image
from io import BytesIO
from torchvision import transforms
from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
import math

logger = logging.getLogger(__name__)


class TrainDatasetBase:
    """
    Abstract base class for all train datasets in Openmatch.\n
    This implants arguments and data preparation, but should be mostly used for identifying an OpenMatch Train Dataset.\n
    All future dataset ABCs would subclass this and `(Iterable)Dataset`.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_args: DataArguments,
        trainer: DRTrainer = None,
        is_eval: bool = False,
        shuffle_seed: int = None,
        cache_dir: str = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.q_max_len = data_args.q_max_len
        self.p_max_len = data_args.p_max_len
        self.trainer = trainer
        self.is_eval = is_eval
        self._prepare_data(data_args, shuffle_seed, cache_dir)

    def _prepare_data(self, data_args, shuffle_seed, cache_dir):
        if not self.is_eval:
            self.data_files = (
                [data_args.train_path]
                if data_args.train_dir is None
                else glob.glob(os.path.join(data_args.train_dir, "*.jsonl"))
            )
        else:
            self.data_files = [data_args.eval_path]

    def get_process_fn(self, epoch, hashed_seed):
        raise NotImplementedError


class StreamTrainDatasetMixin(IterableDataset):
    def _prepare_data(self, data_args, shuffle_seed, cache_dir):
        super()._prepare_data(data_args, shuffle_seed, cache_dir)
        self.dataset = load_dataset(
            "json", data_files=self.data_files, streaming=True, cache_dir=cache_dir
        )["train"]
        sample = list(self.dataset.take(1))[0]
        self.all_columns = sample.keys()

    def __len__(self):
        concat_filenames = " ".join(self.data_files)
        count = 0
        with os.popen("wc -l {}".format(concat_filenames)) as f:
            for line in f:
                lc, filename = line.strip().split()
                lc = int(lc)
                if filename != "total":
                    count += lc
        return count

    def __iter__(self):
        # rank = dist.get_rank()
        # print(f"rank = {rank}, Fetching once")
        
        # if not self.is_eval:
        #     epoch = int(self.trainer.state.epoch)
        #     _hashed_seed = hash(self.trainer.args.seed)
        #     self.dataset.set_epoch(epoch)
        #     return iter(
        #         self.dataset.map(
        #             self.get_process_fn(epoch, _hashed_seed), remove_columns=self.all_columns
        #         )
        #     )
        return iter(self.dataset.map(self.get_process_fn(0, None), remove_columns=self.all_columns))


class MappingTrainDatasetMixin(Dataset):
    def _prepare_data(self, data_args, shuffle_seed, cache_dir):
        super()._prepare_data(data_args, shuffle_seed, cache_dir)
        self.dataset = load_dataset(
            "json", data_files=self.data_files, streaming=False, cache_dir=cache_dir
        )["train"]
        
        # manually shuffle here
        # logger.info(f"shuffle_seed = {shuffle_seed}")
        # self.dataset = (
        #     self.dataset.shuffle(seed=shuffle_seed)
        #     if shuffle_seed is not None
        #     else self.dataset
        # )
        
        sample = self.dataset[0]
        self.all_columns = sample.keys()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        group = self.dataset[index]
        # if not self.is_eval:
        #     epoch = int(self.trainer.state.epoch)
        #     _hashed_seed = hash(index + self.trainer.args.seed)
        #     return self.get_process_fn(epoch, _hashed_seed)(group) # 负例采样
        return self.get_process_fn(0, None)(group) # 指定负例


class DRTrainDataset(TrainDatasetBase):
    def create_one_example(self, text: str, is_query=False):
        return self.tokenizer.encode_plus(
            text,
            truncation=True,
            max_length=self.data_args.q_max_len if is_query else self.data_args.p_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

    def get_process_fn(self, epoch, hashed_seed):
        def process_fn(example):
            """
            example format:
            {
                "query": ["query instruction", "query text"],
                "pos": ["passage instruction", "positve document 1"], # usually 1
                "neg": ["passage instruction", 
                    "negative document 1", 
                    "negative document 2", 
                    "negative document 3", 
                ...] # can be set by --train_n_passages
            }
            """
            
            query: str = " ".join(example["query"]) if self.data_args.query_instruction else example["query"][1] # with query instruction
            pos: str = " ".join(example["pos"]) if self.data_args.corpus_instruction else example["pos"][1]  # without passage instruction
            
            # TODO: this assertion could be eliminated
            assert self.data_args.train_n_passages >= 1 # we have hard negative
            
            negs: List[str] = [" ".join([example["neg"][0], example["neg"][i]]) if self.data_args.corpus_instruction else example["neg"][i] for i in range(1, self.data_args.train_n_passages)]  # without passage instruction
            
            # assert self.data_args.train_n_passages == 2 # MEDI and more data
            
            # print("query", query, "pos", pos)
            encoded_query = self.create_one_example(query, is_query=True)
            
            # print(encoded_query)
            
            # print("encoded_query", encoded_query)
            encoded_passages = [self.create_one_example(pos)]
            encoded_passages.extend([self.create_one_example(neg) for neg in negs])
            # print("encoded_passages", encoded_passages)
            # raise Exception
            # Avoid name conflict with query in the original dataset
            return {"query_": encoded_query, "passages": encoded_passages}

        return process_fn


def slice_image(
    image, max_slice_nums=9, scale_resolution=448, patch_size=14, never_split=False):
    original_size = image.size
    original_width, original_height = original_size
    log_ratio = math.log(original_width / original_height)
    ratio = original_width * original_height / (scale_resolution * scale_resolution)
    multiple = min(math.ceil(ratio), max_slice_nums)

    source_image = None
    best_grid = None
    patches = []

    if multiple <= 1 or never_split:
        # dont need to slice, upsample
        best_size = find_best_resize(
            original_size, scale_resolution, patch_size, allow_upscale=True
        )
        source_image = image.resize(best_size, Image.Resampling.BICUBIC)
    else:
        candidate_split_grids_nums = []
        for i in [multiple - 1, multiple, multiple + 1]:
            if i == 1 or i > max_slice_nums:
                continue
            candidate_split_grids_nums.append(i)

        # source image, down-sampling and ensure divided by patch_size
        best_resize = find_best_resize(original_size, scale_resolution, patch_size)
        source_image = image.copy().resize(best_resize, Image.Resampling.BICUBIC)
        candidate_grids = []

        # find best grid
        for split_grids_nums in candidate_split_grids_nums:
            m = 1
            while m <= split_grids_nums:
                if split_grids_nums % m == 0:
                    candidate_grids.append([m, split_grids_nums // m])
                m += 1

        best_grid = [1, 1]
        min_error = float("inf")
        for grid in candidate_grids:
            error = abs(log_ratio - math.log(grid[0] / grid[1]))
            if error < min_error:
                best_grid = grid
                min_error = error

        refine_size = get_refine_size(
            original_size, best_grid, scale_resolution, patch_size, allow_upscale=True
        )

        refine_image = image.resize(refine_size, Image.Resampling.BICUBIC)
        patches = split_to_patches(refine_image, best_grid)

    return source_image, patches, best_grid

def find_best_resize(original_size, scale_resolution, patch_size, allow_upscale=False):
    width, height = original_size
    if (width * height > scale_resolution * scale_resolution) or allow_upscale:
        r = width / height
        height = int(scale_resolution / math.sqrt(r))
        width = int(height * r)
    best_width = ensure_divide(width, patch_size)
    best_height = ensure_divide(height, patch_size)
    return (best_width, best_height)

def ensure_divide(length, patch_size):
    return max(round(length / patch_size) * patch_size, patch_size)

def get_refine_size(
    original_size, grid, scale_resolution, patch_size, allow_upscale=False):
    width, height = original_size
    grid_x, grid_y = grid

    refine_width = ensure_divide(width, grid_x)
    refine_height = ensure_divide(height, grid_y)

    grid_width = refine_width / grid_x
    grid_height = refine_height / grid_y

    best_grid_size = find_best_resize(
        (grid_width, grid_height),
        scale_resolution,
        patch_size,
        allow_upscale=allow_upscale,
    )

    refine_size = (best_grid_size[0] * grid_x, best_grid_size[1] * grid_y)

    return refine_size

def split_to_patches(image, grid):
    patches = []
    width, height = image.size
    grid_x = int(width / grid[0])
    grid_y = int(height / grid[1])

    for i in range(0, height, grid_y):
        images = []
        for j in range(0, width, grid_x):
            box = (j, i, j + grid_x, i + grid_y)
            patch = image.crop(box)
            images.append(patch)
        patches.append(images)

    return patches

def get_grid_placeholder(tokenizer, grid, query_num):
    image_placeholder = (
        tokenizer.im_start + tokenizer.unk_token * query_num + tokenizer.im_end
    )

    cols = grid[0]
    rows = grid[1]
    slices = []
    for i in range(rows):
        lines = []
        for j in range(cols):
            lines.append(image_placeholder)
        slices.append("".join(lines))
    slice_placeholder = tokenizer.slice_start + "\n".join(slices) + tokenizer.slice_end
    return slice_placeholder

# For multimodal Dense Retrieval Model
class MMDRTrainDataset(TrainDatasetBase):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD
            ),
        ]
    )
    
    def _convert_to_tensors(
        self, tokenizer, input_str, max_inp_length: Optional[int] = None):
        if tokenizer.add_bos_token:
            input_ids = tokenizer.encode(input_str)
        else:
            input_ids = [tokenizer.bos_id] + tokenizer.encode(input_str)
        if max_inp_length is not None:
            input_ids = input_ids[:max_inp_length]
        
        input_ids = torch.tensor(input_ids, dtype=torch.int32)

        image_start_tokens = torch.where(input_ids == tokenizer.im_start_id)[0]
        # 跳过 im_start
        image_start_tokens += 1
        image_end_tokens = torch.where(input_ids == tokenizer.im_end_id)[0]
        valid_image_nums = max(len(image_start_tokens), len(image_end_tokens))
        image_bound = torch.hstack(
            [
                image_start_tokens[:valid_image_nums].unsqueeze(-1),
                image_end_tokens[:valid_image_nums].unsqueeze(-1),
            ]
        )

        model_input = {}
        model_input["input_ids"] = input_ids.unsqueeze(0)
        model_input["image_bound"] = image_bound

        return model_input

    def get_slice_image_placeholder(self, image, tokenizer):
        query_num = 64 # self.config.query_num
        image_placeholder = (
            tokenizer.im_start
            + tokenizer.unk_token * query_num
            + tokenizer.im_end
        )

        slice_images = []

        source_image, patches, best_grid = slice_image(
            image,
            # self.config.max_slice_nums,
            9,
            # self.config.scale_resolution,
            448,
            # self.config.patch_size,
            14
        )

        slice_images.append(source_image)
        final_placeholder = image_placeholder

        if len(patches) > 0:
            for i in range(len(patches)):
                for j in range(len(patches[0])):
                    slice_images.append(patches[i][j])

            final_placeholder += get_grid_placeholder(
                tokenizer, best_grid, query_num
            )

        return slice_images, final_placeholder
    
    def fused_tokenize(
            self,
            input_texts=None, # List[str] one raw string for each data
            input_images=None, # List[PIL.Image] one raw image for each data
            tokenizer=None, 
            max_input_length: Optional[int] = 1024, 
            slice_mode=True
        ):
        
        assert input_texts is not None
        bs = len(input_texts)
        if input_images == None:
            input_images = [[] for i in range(bs)]
        assert bs == len(input_images)
        
        # Step1: insert placeholders into text strings, and slice image to multiple images
        input_texts_with_image_placeholder = []
        input_images_sliced = []
        for text, image in zip(input_texts, input_images):
            content = text
            if image is not None: # if there is image in this data
                if slice_mode:
                    images, final_placeholder = self.get_slice_image_placeholder(
                        image, tokenizer
                    ) # crop one image into multiple sub images -> List[Image]
                    content = final_placeholder + "\n" + content
                else:
                    images = [image] # only keep one image without cropping -> List[Image]
                    content = (
                        tokenizer.im_start
                        + tokenizer.unk_token * self.config.query_num
                        + tokenizer.im_end
                        + "\n"
                        + content
                    )
            else:
                images = []
            input_texts_with_image_placeholder.append(content)
            input_images_sliced.append(images)
        
        model_inputs = {
            "input_ids": [],
            "image_bound": []
        }

        for data in input_texts_with_image_placeholder:
            tokenized = self._convert_to_tensors(tokenizer, data, max_input_length)
            for key in model_inputs:
                model_inputs[key].append(tokenized[key])
        
        assert len(model_inputs["image_bound"]) > 0
        assert len(model_inputs["input_ids"]) > 0

        pixel_values = []
        for i in range(bs):
            image_transformed = []
            for img in input_images_sliced[i]:
                image_transformed.append(self.transform(img))
            if image_transformed:
                pixel_values.append(image_transformed)
            else:
                pixel_values.append([])
        model_inputs["pixel_values"] = pixel_values # inject image pixels to model_inputs
        
        return model_inputs # all keys are tensors
    
    def convert_base64string_to_image(self, base64_string):
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))
        image = image.convert("RGB")
        return image
    
    def get_process_fn(self, epoch, hashed_seed):
        def process_fn(example):
            
            # MM example format:
            # {
            #     "query": {
            #             "instruction": string,
            #             "image": None or base64_string,
            #             "text": string
            #         }
            #     ]
            #     "pos": [
            #         {
            #             "instruction": string,
            #             "image": None or base64_string,
            #             "text": string
            #         }, ...
            #     ]
            #     "neg": [
            #         {
            #             "instruction": string,
            #             "image": None or base64_string,
            #             "text": string
            #         }, ...
            #     ]
            # }
            
            # TODO: this assertion could be eliminated
            # assert self.data_args.train_n_passages > 1 # we have hard negative
            # assert self.data_args.train_n_passages <= (len(example["neg"]) + 1), "--train_n_passages should <= number of provided negative samples + number of positive samples "
            
            query = example["query"]
            pos = example["pos"][0]
            neg = example["neg"][0: self.data_args.train_n_passages - 1] # we can exactly have self.data_args.train_n_passages-1 negative passages
            
            # Step1: merge instructions into texts if requested
            if self.data_args.query_instruction:
                query["text"] = query["instruction"] + query["text"]
                # query["image"] = pos["image"] # test

            if self.data_args.corpus_instruction: # test
                pos["text"] = pos["instruction"] + pos["text"]
                for idx in range(len(neg)):
                    neg[idx]["text"] = neg[idx]["instruction"] + neg[idx]["text"]
            
            # Step2: convert base64_string to images
            if query["image"] is not None:
                query["image"] = self.convert_base64string_to_image(query["image"])
            
            if pos["image"] is not None:
                pos["image"] = self.convert_base64string_to_image(pos["image"])
            
            for idx in range(len(neg)):
                if neg[idx]["image"] is not None:
                    neg[idx]["image"] = self.convert_base64string_to_image(neg[idx]["image"])
            
            passages = [pos, *neg]
            # passages = [pos, pos] # just a test
            
            query_tokenized = self.fused_tokenize(
                input_texts=[query["text"]],
                input_images=[query["image"]],
                tokenizer=self.tokenizer,
                max_input_length=self.q_max_len,
                slice_mode=True,
            )
            
            passage_tokenized = self.fused_tokenize(
                input_texts=[p["text"] for p in passages],
                input_images=[p["image"] for p in passages],
                tokenizer=self.tokenizer,
                max_input_length=self.p_max_len,
                slice_mode=True,
            )
            
            return {"query_": query_tokenized, "passages": passage_tokenized}

        return process_fn


class StreamDRTrainDataset(StreamTrainDatasetMixin, DRTrainDataset):
    pass


class MappingDRTrainDataset(MappingTrainDatasetMixin, DRTrainDataset):
    pass


class StreamMMDRTrainDataset(StreamTrainDatasetMixin, MMDRTrainDataset):
    pass


class MappingMMDRTrainDataset(MappingTrainDatasetMixin, MMDRTrainDataset):
    pass


class DRPretrainDataset(TrainDatasetBase):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_args: DRPretrainingDataArguments,
        trainer: DRTrainer = None,
        is_eval: bool = False,
        shuffle_seed: int = None,
        cache_dir: str = None,
    ) -> None:
        super(DRPretrainDataset, self).__init__(
            tokenizer, data_args, trainer, is_eval, shuffle_seed, cache_dir
        )
        pretrain_strategies_str = (
            data_args.pretrain_strategies.split(",")
            if data_args.pretrain_strategies is not None
            else []
        )
        strategies = []
        for strategy_str in pretrain_strategies_str:
            if strategy_str == "null":
                strategies.append(NullStrategy())
                logger.info("Adding NullStrategy")
            elif strategy_str == "crop":
                strategies.append(
                    Cropping(
                        ratio_min=data_args.cropping_ratio_min,
                        ratio_max=data_args.cropping_ratio_max,
                    )
                )
                logger.info(
                    "Adding Cropping, ratio_min={}, ratio_max={}".format(
                        data_args.cropping_ratio_min, data_args.cropping_ratio_max
                    )
                )
            else:
                raise ValueError("Unknown pretraining strategy: {}".format(strategy_str))
        self.apply_strategy = SequentialStrategies(*strategies)

    def create_one_example(self, text_encoding: List[int], is_query=False) -> BatchEncoding:
        text_encoding = self.apply_strategy(text_encoding)
        item = self.tokenizer.encode_plus(
            text_encoding,
            truncation="only_first",
            max_length=self.data_args.q_max_len if is_query else self.data_args.p_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def get_process_fn(self, epoch, hashed_seed):
        def process_fn(example):
            content = example[self.data_args.pretrain_target_field]
            encoded_query = self.create_one_example(content, is_query=True)
            encoded_passages = [self.create_one_example(content)]

            return {"query_": encoded_query, "passages": encoded_passages}

        return process_fn


class StreamDRPretrainDataset(StreamTrainDatasetMixin, DRPretrainDataset):
    pass


class MappingDRPretrainDataset(MappingTrainDatasetMixin, DRPretrainDataset):
    pass


class RRTrainDataset(TrainDatasetBase):
    def create_one_example(self, qry_encoding, psg_encoding) -> BatchEncoding:
        if self.data_args.encode_as_text_pair:
            item = self.tokenizer.encode_plus(
                qry_encoding,
                psg_encoding,
                truncation="longest_first",
                max_length=self.data_args.q_max_len + self.data_args.p_max_len + 2,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=True,
            )
        else:
            item = self.tokenizer.encode_plus(
                qry_encoding + psg_encoding,
                truncation="longest_first",
                max_length=self.data_args.q_max_len + self.data_args.p_max_len + 2,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
        return item

    def get_process_fn(self, epoch, hashed_seed):
        def process_fn(example):
            qry = example["query"]
            group_positives = example["positives"]
            group_negatives = example["negatives"]

            if self.data_args.positive_passage_no_shuffle or hashed_seed is None:
                pos_psg = group_positives[0]
            else:
                pos_psg = group_positives[(hashed_seed + epoch) % len(group_positives)]
            encoded_pos_pair = self.create_one_example(qry, pos_psg)

            if hashed_seed is None:
                neg_psg = group_negatives[0]
            else:
                neg_psg = group_negatives[(hashed_seed + epoch) % len(group_negatives)]
            encoded_neg_pair = self.create_one_example(qry, neg_psg)
            return {"pos_pair": encoded_pos_pair, "neg_pair": encoded_neg_pair}

        return process_fn


class StreamRRTrainDataset(StreamTrainDatasetMixin, RRTrainDataset):
    pass


class MappingRRTrainDataset(MappingTrainDatasetMixin, RRTrainDataset):
    pass


class QGTrainDataset(TrainDatasetBase):
    def create_one_example(self, text_encoding: List[int], is_query=False) -> BatchEncoding:
        item = self.tokenizer.encode_plus(
            text_encoding,
            truncation="only_first",
            max_length=self.data_args.q_max_len if is_query else self.data_args.p_max_len,
            padding="max_length",
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        return item

    def get_process_fn(self, epoch, hashed_seed):
        def process_fn(example):
            qry = example["query"]
            group_positives = example["positives"]
            if self.data_args.positive_passage_no_shuffle or hashed_seed is None:
                pos_psg = group_positives[0]
            else:
                pos_psg = group_positives[(hashed_seed + epoch) % len(group_positives)]

            encoded_query = self.create_one_example(qry, is_query=True).input_ids
            encoded_query[encoded_query == self.tokenizer.pad_token_id] == -100
            encoded_psg = self.create_one_example(pos_psg)
            psg_input_ids, psg_attention_mask = encoded_psg.input_ids, encoded_psg.attention_mask
            return {
                "input_ids": psg_input_ids[0],
                "attention_mask": psg_attention_mask[0],
                "labels": encoded_query[0],
            }

        return process_fn


class StreamQGTrainDataset(StreamTrainDatasetMixin, QGTrainDataset):
    pass


class MappingQGTrainDataset(MappingTrainDatasetMixin, QGTrainDataset):
    pass


class CQGTrainDataset(TrainDatasetBase):
    def create_one_example(
        self,
        qry_encoding: List[int] = None,
        psg_encoding_pos: List[int] = None,
        psg_encoding_neg: List[int] = None,
    ) -> BatchEncoding:
        if qry_encoding is not None:
            return self.tokenizer.encode_plus(
                qry_encoding,
                truncation="only_first",
                max_length=self.data_args.q_max_len,
                padding="max_length",
                return_attention_mask=True,
                return_token_type_ids=False,
                return_tensors="pt",
            )
        return self.tokenizer.encode_plus(
            psg_encoding_pos + psg_encoding_neg,
            truncation="only_first",
            max_length=self.data_args.p_max_len * 2,
            padding="max_length",
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors="pt",
        )

    def get_process_fn(self, epoch, hashed_seed):
        def process_fn(example):
            qry = example["query"]
            group_positives = example["positives"]
            group_negatives = example["negatives"]
            if self.data_args.positive_passage_no_shuffle or hashed_seed is None:
                pos_psg = group_positives[0]
            else:
                pos_psg = group_positives[(hashed_seed + epoch) % len(group_positives)]

            if hashed_seed is None:
                neg_psg = group_negatives[0]
            else:
                neg_psg = group_negatives[(hashed_seed + epoch) % len(group_negatives)]

            encoded_query = self.create_one_example(qry_encoding=qry).input_ids
            encoded_query[encoded_query == self.tokenizer.pad_token_id] == -100
            encoded_psg_pair = self.create_one_example(
                psg_encoding_pos=pos_psg, psg_encoding_neg=neg_psg
            )
            psg_input_ids, psg_attention_mask = (
                encoded_psg_pair.input_ids,
                encoded_psg_pair.attention_mask,
            )
            return {
                "input_ids": psg_input_ids[0],
                "attention_mask": psg_attention_mask[0],
                "labels": encoded_query[0],
            }

        return process_fn


class StreamCQGTrainDataset(StreamTrainDatasetMixin, CQGTrainDataset):
    pass


class MappingCQGTrainDataset(MappingTrainDatasetMixin, CQGTrainDataset):
    pass


class PairwiseDistillationTrainDataset(TrainDatasetBase):
    def create_one_example(self, text_encoding: List[int], is_query=False) -> BatchEncoding:
        item = self.tokenizer.encode_plus(
            text_encoding,
            truncation="only_first",
            max_length=self.data_args.q_max_len if is_query else self.data_args.p_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def get_process_fn(self, epoch, hashed_seed):
        def process_fn(example):
            qry = self.create_one_example(example["query"], is_query=True)
            pos = self.create_one_example(example["positive"])
            neg = self.create_one_example(example["negative"])
            score = example["score"]
            return {"query_": qry, "positive_": pos, "negative_": neg, "score_": score}

        return process_fn


class StreamPairwiseDistillationTrainDataset(
    StreamTrainDatasetMixin, PairwiseDistillationTrainDataset
):
    pass


class MappingPairwiseDistillationTrainDataset(
    MappingTrainDatasetMixin, PairwiseDistillationTrainDataset
):
    pass


class ListwiseDistillationTrainDataset(TrainDatasetBase):
    def create_one_example(self, text_encoding: List[int], is_query=False) -> BatchEncoding:
        item = self.tokenizer.encode_plus(
            text_encoding,
            truncation="only_first",
            max_length=self.data_args.q_max_len if is_query else self.data_args.p_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def get_process_fn(self, epoch, hashed_seed):
        def process_fn(example):
            qry = example["query"]
            encoded_query = self.create_one_example(qry, is_query=True)
            encoded_passages = []
            passages = example["docs"]
            scores = example["scores"]
            passages_and_scores = list(zip(passages, scores))

            if len(passages) < self.data_args.train_n_passages:
                if hashed_seed is not None:
                    psgs = random.choices(passages_and_scores, k=self.data_args.train_n_passages)
                else:
                    psgs = [x for x in passages_and_scores]
                    psgs = psgs * 2
                    psgs = psgs[: self.data_args.train_n_passages]
            else:
                _offset = epoch * self.data_args.train_n_passages % len(passages)
                psgs = [x for x in passages_and_scores]
                if hashed_seed is not None:
                    random.Random(hashed_seed).shuffle(psgs)
                psgs = psgs * 2
                psgs = psgs[_offset : _offset + self.data_args.train_n_passages]

            for psg in psgs:
                encoded_passages.append(self.create_one_example(psg[0]))

            assert len(encoded_passages) == self.data_args.train_n_passages

            return {
                "query_": encoded_query,
                "passages": encoded_passages,
                "scores_": [x[1] for x in psgs],
            }  # Avoid name conflict with query in the original dataset

        return process_fn


class StreamListwiseDistillationTrainDataset(
    StreamTrainDatasetMixin, ListwiseDistillationTrainDataset
):
    pass


class MappingListwiseDistillationTrainDataset(
    MappingTrainDatasetMixin, ListwiseDistillationTrainDataset
):
    pass


