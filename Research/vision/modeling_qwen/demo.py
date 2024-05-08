from transformers import AutoModelForCausalLM
from tokenization_qwen import QWenTokenizer
from modeling_qwen import QWenLMHeadModel
from transformers.generation import GenerationConfig
import torch
torch.manual_seed(1234)

# Note: The default behavior now has injection attack prevention off.
# tokenizer = AutoTokenizer.from_pretrained("/home/jeeves/Qwen-VL-Chat", trust_remote_code=True)
MODEL_PATH = "/home/jeeves/Qwen-VL-Chat"

tokenizer = QWenTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cpu", trust_remote_code=True).eval()
# use cuda device

model = QWenLMHeadModel.from_pretrained(MODEL_PATH, device_map="cuda", trust_remote_code=True).eval()

# Specify hyperparameters for generation (No need to do this if you are using transformers>=4.32.0)
# model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

PROMPT = """
The above image is a page from a book. Your task is to come up with a natural, precise, and ingenious question that can directly "guide" someone to think of this page when she/he sees your question. Do you understand what I mean? The page of book I provide above will answer the question you have come up with. This is actually a reverse process.
Please output your thought process in JSON format:

```json
{{
"discussion": "[Discuss the content in this page, think of meaning, main idea, and thoughts etc. of this page to ensure that you truly understand it.]",
"query": "Think of a 'guiding question' for this page, please make sure it is specific, accurate, and just right. Don't be too short, and also don't be too long."
}}
```

"""


PROMPT_MULTI = """
The above image is a page from a book. Your task is to come up with multiple natural, precise, and ingenious question that can directly "guide" someone to think of this page when she/he sees your question. Do you understand what I mean? The page of book I provide above will answer the question you have come up with. This is actually a reverse process.
Please output your thought process in JSON format:

```json
{{
"discussion": "[Discuss the content in this page, think of meaning, main idea, and thoughts etc. of this page to ensure that you truly understand it.]",
"easy_query": "Think of a 'guiding question' for this page, please make sure it is specific, accurate, and just right. This query usually refers to specific and detailed, the longest.",
"intermediate_query": "Think of another 'guiding question' for this page, but this time, you need to raise the level: it should be a harder question, requiring higher level of reasoning and intuition.",
"hard_query": "Think of another 'guiding question' for this page, but this time, you need to give me an advanced query: you know, this time your query should not focus on lexical similarity, instead you need to consider in semantic space. This query usually should be the shortest among these 3 queries",
}}
```

"""


# IMAGE_PATH = "/home/jeeves/openmatch/Research/Dataset/vision/demo_2.png"
IMAGE_PATH = "/home/jeeves/openmatch/Research/Dataset/vision/demo_2.png"

# 1st dialogue turn
query = tokenizer.from_list_format([
    {'image': IMAGE_PATH},
    {'text': PROMPT},
])

response, history = model.chat(
    tokenizer, 
    query=query, 
    history=None,
    # temperature=1.2,
)
print('='*20)
print(response)
# 图中是一名年轻女子在沙滩上和她的狗玩耍，狗的品种可能是拉布拉多。她们坐在沙滩上，狗的前腿抬起来，似乎在和人类击掌。两人之间充满了信任和爱。

# 2nd dialogue turn
# response, history = model.chat(tokenizer, '输出"击掌"的检测框', history=history)
# print(response)
# # <ref>击掌</ref><box>(517,508),(589,611)</box>
# image = tokenizer.draw_bbox_on_latest_picture(response, history)
# if image:
#   image.save('1.jpg')
# else:
#   print("no box")
