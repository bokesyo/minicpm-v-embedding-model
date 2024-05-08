import torch

hidden = torch.load("/home/jeeves/example.pt")
attention_mask = torch.load("/home/jeeves/attention_mask.pt")
idx = torch.sum(attention_mask, dim=1)
print(idx.dtype)
print(idx)


seq_lengths = attention_mask.sum(dim=1) - 1
last_token_indices = seq_lengths.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))

# Gather the last valid token representations
last_valid_tokens = hidden_states.gather(1, last_token_indices).squeeze(1)