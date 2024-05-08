from transformers import AutoTokenizer

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    '/home/jeeves/bert-base-uncased-small', use_fast=False)

# Sample text
text = "Hello, how are you?"

# Encode the text to get token IDs
encoded_text = tokenizer.encode(text, add_special_tokens=True)
print(f"Encoded text to IDs: {encoded_text}")

# Now let's assume you have the token IDs as a list
# and want to use `encode_plus` to replicate the encoding process

# Since `encode_plus` does not directly accept `List[int]` as an argument for text,
# you need to convert the token IDs back to tokens and then to a string if needed.
tokens = tokenizer.convert_ids_to_tokens(encoded_text)
# text_from_tokens = tokenizer.convert_tokens_to_string(tokens)

# Now use `encode_plus` with the reconstructed text
encoded_plus_output = tokenizer.encode_plus(
    tokens, 
    truncation="only_first", 
    max_length=512, 
    padding=False, 
    return_attention_mask=False, 
    return_token_type_ids=False
)

print(f"Encoded plus output: {encoded_plus_output}")
