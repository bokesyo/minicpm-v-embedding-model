import math
from typing import List, Optional
import json
import timm
import torch
import torchvision
from PIL import Image
from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
# from torchvision import transforms
from transformers import LlamaTokenizer

from .configuration_minicpm import MiniCPMVConfig
from .modeling_minicpm import MiniCPMForCausalLM, MiniCPMPreTrainedModel
from .resampler import Resampler

from transformers import SiglipVisionModel


class MiniCPMVPreTrainedModel(MiniCPMPreTrainedModel):
    config_class = MiniCPMVConfig


class MiniCPMV(MiniCPMVPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
    
        self.llm = MiniCPMForCausalLM(config)
        self.vision_model = SiglipVisionModel(config.vision_config).vision_model
        # Vision hidden size
        self.vision_dim = self.config.vision_config.hidden_size
        # Language hidden size
        self.embed_dim = self.llm.config.hidden_size
        self.resampler = Resampler(
            grid_size=int(math.sqrt(self.config.query_num)),
            embed_dim=self.embed_dim,
            num_heads=self.embed_dim // 128,
            kv_dim=self.vision_dim,
            adaptive=True
        )
        
        return
    
    def get_vision_embedding(self, pixel_values):
        res = []
        dtype = self.vision_model.embeddings.position_embedding.weight.data.dtype
        for pixel_value in pixel_values:
            H, W = pixel_value.shape[-2:]
            tgt_size = (
                H // self.config.vision_config.patch_size, 
                W // self.config.vision_config.patch_size
            )
            print("tgt_size", tgt_size)
            vision_embedding = self.vision_model(pixel_value.unsqueeze(0).type(dtype))
            
            print(vision_embedding)
            print(vision_embedding.last_hidden_state.shape)
            print(vision_embedding.pooler_output.shape)
            # raise Exception
            res.append(self.resampler(vision_embedding.last_hidden_state, tgt_size))
        
        return torch.vstack(res)

    # @torch.no_grad()
    def get_vllm_embedding(self, data):
        if "vision_hidden_states" not in data:
            pixel_values_list = data["pixel_values"]
            vision_hidden_states = []
            for pixel_values in pixel_values_list:
                if len(pixel_values) > 0:
                    vision_hidden_states.append(self.get_vision_embedding(pixel_values))
                elif self.training:
                    dtype = self.vision_model.embeddings.position_embedding.weight.data.dtype
                    device = self.vision_model.embeddings.position_embedding.weight.data.device
                    dummy_image = torch.zeros(
                        (1, 3, 224, 224), device=device, dtype=dtype
                    )
                    vision_hidden_states.append(self.get_vision_embedding(dummy_image))
                else:
                    vision_hidden_states.append([])

        else:
            vision_hidden_states = data["vision_hidden_states"]

        vllm_embedding = (
            self.llm.model.embed_tokens(data["input_ids"]) * self.llm.config.scale_emb
        )
        vision_hidden_states = [
            i.type(vllm_embedding.dtype) if isinstance(i, torch.Tensor) else i
            for i in vision_hidden_states
        ]

        bs = len(data["input_ids"])
        for i in range(bs):
            cur_vs_hs = vision_hidden_states[i]
            if len(cur_vs_hs) > 0:
                cur_vllm_emb = vllm_embedding[i]
                cur_image_bound = data["image_bound"][i]
                if len(cur_image_bound) > 0:
                    image_indices = torch.stack(
                        [
                            torch.arange(r[0], r[1], dtype=torch.long)
                            for r in cur_image_bound
                        ]
                    ).to(vllm_embedding.device)

                    cur_vllm_emb.scatter_(
                        0,
                        image_indices.view(-1, 1).repeat(1, cur_vllm_emb.shape[-1]),
                        cur_vs_hs.view(-1, cur_vs_hs.shape[-1]),
                    )
                elif self.training:
                    cur_vllm_emb += cur_vs_hs[0].mean() * 0

        return vllm_embedding, vision_hidden_states

    def forward(
        self,
        input_ids,
        pixel_values,
        image_bound,
        **kwargs
        ):

        model_inputs = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "image_bound": image_bound
        }
        
        # with torch.no_grad():
        # merge image embedding and text embedding into unified inputs_embeds
        model_inputs["inputs_embeds"], vision_hidden_states = self.get_vllm_embedding(model_inputs)
        
        # Step5: here it is a pure causal LM problem
        # outputs =  # -> CausalLMOutput
        return self.llm.model.forward( # call MiniCPMBaseModel, we don't need output logits ([B, n_vocab]), we need last_hidden_states ([B, seq_len, d])
            input_ids=None, # because image and text have been merged into model_inputs["inputs_embeds"] here, we don't give input_ids
            position_ids=None,
            inputs_embeds=model_inputs["inputs_embeds"],
            # attention_mask=model_inputs["attention_mask"],
            **kwargs
        )

class LlamaTokenizerWrapper(LlamaTokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.im_start = "<image>"
        self.im_end = "</image>"
        self.ref_start = "<ref>"
        self.ref_end = "</ref>"
        self.box_start = "<box>"
        self.box_end = "</box>"
        self.quad_start = "<quad>"
        self.quad_end = "</quad>"
        self.point_start = "<point>"
        self.point_end = "</point>"
        self.slice_start = "<slice>"
        self.slice_end = "</slice>"

    @property
    def eos_id(self):
        return self.sp_model.eos_id()

    @property
    def bos_id(self):
        return self.sp_model.bos_id()

    @property
    def unk_id(self):
        return self.sp_model.unk_id()

    @property
    def im_start_id(self):
        return self._convert_token_to_id(self.im_start)

    @property
    def im_end_id(self):
        return self._convert_token_to_id(self.im_end)



class MiniCPMVForMMEmbedding(MiniCPMV): # -> MiniCPMV ->  Ultimately a CausalLM
    def forward(
        self,
        input_ids,
        pixel_values,
        image_bound,
        **kwargs): # CausalLM forward, modified from self.generate

        model_inputs = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "image_bound": image_bound
        }
        
        # with torch.no_grad():
        # merge image embedding and text embedding into unified inputs_embeds
        model_inputs["inputs_embeds"], vision_hidden_states = self.get_vllm_embedding(model_inputs)
        
        # Step5: here it is a pure causal LM problem
        # outputs =  # -> CausalLMOutput
        # outputs["attention_mask"] = model_inputs["attention_mask"]
        # print("=============== checkpoint 1 ===============")
        # raise NotImplementedError("stop!")
        return self.llm.model.forward( # call MiniCPMBaseModel, we don't need output logits ([B, n_vocab]), we need last_hidden_states ([B, seq_len, d])
            input_ids=None, # because image and text have been merged into model_inputs["inputs_embeds"] here, we don't give input_ids
            position_ids=None,
            inputs_embeds=model_inputs["inputs_embeds"],
            # attention_mask=model_inputs["attention_mask"],
            **kwargs
        )

