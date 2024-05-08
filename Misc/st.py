from sentence_transformers import SentenceTransformer

model = SentenceTransformer("/home/jeeves/cpm_d-2b_with_pad_token", trust_remote_code=True)

texts = [
    "deep learning",
    "artificial intelligence",
    "deep diving",
    "artificial snow",
]

embeddings = model.encode(texts)


model = SentenceTransformer("/home/jeeves/bert-base-uncased-small", trust_remote_code=True)
