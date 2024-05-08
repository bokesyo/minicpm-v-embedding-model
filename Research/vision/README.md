
# Vision Inverse Query

Model: Qwen-VL

![](./sample_images/page_71.png)



- PROMPT_MULTI
The above image is a page from a book. Your task is to come up with multiple natural, precise, and ingenious question that can directly "guide" someone to think of this page when she/he sees your question. Do you understand what I mean? The page of book I provide above will answer the question you have come up with. This is actually a reverse process.
Please output your thought process in JSON format:

```json
{
"discussion": "[Discuss the content in this page, think of meaning, main idea, and thoughts etc. of this page to ensure that you truly understand it.]",
"easy_query": "Think of a 'guiding question' for this page, please make sure it is specific, accurate, and just right. This query usually refers to specific and detailed, the longest.",
"intermediate_query": "Think of another 'guiding question' for this page, but this time, you need to raise the level: it should be a harder question, requiring higher level of reasoning and intuition.",
"hard_query": "Think of another 'guiding question' for this page, but this time, you need to give me an advanced query: you know, this time your query should not focus on lexical similarity, instead you need to consider in semantic space. This query usually should be the shortest among these 3 queries",
}
```

{ "easy_query": "What is the difference between brightfield and differential interference contrast (DIC) microscopy?", "intermediate_query": "Explain the principle of fluorescent microscopy and give an example.", "hard_query": "Describe the mechanism of Confocal microscopy and explain the advantage of using it over other microscopy techniques." }


- PROMPT_SINGLE
The above image is a page from a book. Your task is to come up with a natural, precise, and ingenious question that can directly "guide" someone to think of this page when she/he sees your question. Do you understand what I mean? The page of book I provide above will answer the question you have come up with. This is actually a reverse process.
Please output your thought process in JSON format:

```json
{
"discussion": "[Discuss the content in this page, think of meaning, main idea, and thoughts etc. of this page to ensure that you truly understand it.]",
"query": "Think of a 'guiding question' for this page, please make sure it is specific, accurate, and just right. Don't be too short, and also don't be too long."
}
```

{ "query": "What are some examples of light microscopy techniques used to visualize cells and tissues?" }



# Run

```bash
sh /home/jeeves/openmatch/scripts/configure_nb_env.sh
```

0. Download dataset

```bash
git clone https://bokesyo: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@huggingface.co/bokesyo/visual_embedding_1

```

1. Produce a jsonl dataset containing data.jsonl

python /home/jeeves/openmatch/Research/Dataset/vision/make_jsonl_ds_direct.py

- merge

python /home/jeeves/openmatch/Research/Dataset/vision/merge_split_jsonl.py

2. Generate reversed queries

python /home/jeeves/openmatch/Research/Dataset/vision/image_inverse_query_driver.py

