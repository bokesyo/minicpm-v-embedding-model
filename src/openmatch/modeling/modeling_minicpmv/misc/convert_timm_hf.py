from safetensors import safe_open
from safetensors.torch import save_file


tensors_to_save = {}
source_tensors = safe_open("/home/jeeves/MiniCPM-V-2.0/model-00002-of-00002.safetensors", framework="pt")


for key in source_tensors.keys():
    if "resampler" in key:
        print(key)
        tensors_to_save[key] = source_tensors.get_tensor(key)

# vpm.patch_embed.proj.bias -> vision_model.embeddings.patch_embedding.bias
# [1152] -> [1152]

tensors_to_save["vision_model.embeddings.patch_embedding.bias"] = source_tensors.get_tensor("vpm.patch_embed.proj.bias")

# vpm.patch_embed.proj.weight -> vision_model.embeddings.patch_embedding.weight
# [1152, 3, 14, 14] -> [1152, 3, 14, 14]
tensors_to_save["vision_model.embeddings.patch_embedding.weight"] = source_tensors.get_tensor("vpm.patch_embed.proj.weight")

# vpm.pos_embed -> vision_model.embeddings.position_embedding.weight
# [1, 729, 1152] -> [729, 1152]
tensors_to_save["vision_model.embeddings.position_embedding.weight"] = source_tensors.get_tensor("vpm.pos_embed").squeeze(0)

for layer in range(0, 26):
    print(layer)
    # vpm.blocks.0.attn.qkv.bias [3456] -> {
        # vision_model.encoder.layers.0.self_attn.q_proj.bias [1152]
        # vision_model.encoder.layers.0.self_attn.k_proj.bias [1152]
        # vision_model.encoder.layers.0.self_attn.v_proj.bias [1152]
    # [3456] -> [1152, 1152, 1152]
    dim = source_tensors.get_tensor(f"vpm.blocks.{layer}.attn.qkv.weight").shape[-1]
    print("dim", dim)
    
    print(source_tensors.get_tensor(f"vpm.blocks.{layer}.attn.qkv.bias")[:dim].shape)
    
    tensors_to_save[f"vision_model.encoder.layers.{layer}.self_attn.q_proj.bias"] = source_tensors.get_tensor(f"vpm.blocks.{layer}.attn.qkv.bias")[:dim]
    tensors_to_save[f"vision_model.encoder.layers.{layer}.self_attn.k_proj.bias"] = source_tensors.get_tensor(f"vpm.blocks.{layer}.attn.qkv.bias")[dim:2*dim]
    tensors_to_save[f"vision_model.encoder.layers.{layer}.self_attn.v_proj.bias"] = source_tensors.get_tensor(f"vpm.blocks.{layer}.attn.qkv.bias")[2*dim:]


    # vpm.blocks.0.attn.qkv.weight [3456, 1152] -> {
        # vision_model.encoder.layers.0.self_attn.k_proj.weight [1152, 1152]
        # vision_model.encoder.layers.0.self_attn.k_proj.weight [1152, 1152]
        # vision_model.encoder.layers.0.self_attn.k_proj.weight [1152, 1152]
    # [3456, 1152] -> [1152, 1152] [1152, 1152] [1152, 1152]
    print(source_tensors.get_tensor(f"vpm.blocks.{layer}.attn.qkv.weight")[:dim, :].shape)
    
    tensors_to_save[f"vision_model.encoder.layers.{layer}.self_attn.q_proj.weight"] = source_tensors.get_tensor(f"vpm.blocks.{layer}.attn.qkv.weight")[:dim, :]
    tensors_to_save[f"vision_model.encoder.layers.{layer}.self_attn.k_proj.weight"] = source_tensors.get_tensor(f"vpm.blocks.{layer}.attn.qkv.weight")[dim:2*dim, :]
    tensors_to_save[f"vision_model.encoder.layers.{layer}.self_attn.v_proj.weight"] = source_tensors.get_tensor(f"vpm.blocks.{layer}.attn.qkv.weight")[2*dim:, :]

    tensors_to_save[f"vision_model.encoder.layers.{layer}.self_attn.out_proj.weight"] = source_tensors.get_tensor(f"vpm.blocks.{layer}.attn.proj.weight")
    
    tensors_to_save[f"vision_model.encoder.layers.{layer}.self_attn.out_proj.bias"] = source_tensors.get_tensor(f"vpm.blocks.{layer}.attn.proj.bias")
    
    # vpm.blocks.0.attn.proj.weight [1152, 1152] -> vision_model.encoder.layers.0.self_attn.out_proj.weight [1152, 1152]
    tensors_to_save[f"vision_model.encoder.layers.{layer}.mlp.fc2.bias"] = source_tensors.get_tensor(f"vpm.blocks.{layer}.attn.proj.weight")

    # vpm.blocks.0.mlp.fc1.weight [4304, 1152] -> vision_model.encoder.layers.0.mlp.fc1.weight [4304, 1152]
    tensors_to_save[f"vision_model.encoder.layers.{layer}.mlp.fc1.weight"] = source_tensors.get_tensor(f"vpm.blocks.{layer}.mlp.fc1.weight")

    # vpm.blocks.0.mlp.fc1.bias [4304] -> vision_model.encoder.layers.0.mlp.fc1.bias [4304]
    tensors_to_save[f"vision_model.encoder.layers.{layer}.mlp.fc1.bias"] = source_tensors.get_tensor(f"vpm.blocks.{layer}.mlp.fc1.bias")

    # vpm.blocks.0.mlp.fc2.weight -> vision_model.encoder.layers.0.mlp.fc2.weight
    # [1152, 4304]-> [1152, 4304]
    tensors_to_save[f"vision_model.encoder.layers.{layer}.mlp.fc2.weight"] = source_tensors.get_tensor(f"vpm.blocks.{layer}.mlp.fc2.weight")

    # vpm.blocks.0.mlp.fc2.bias -> vision_model.encoder.layers.10.mlp.fc2.bias
    # [1152] -> [1152]
    tensors_to_save[f"vision_model.encoder.layers.{layer}.mlp.fc2.bias"] = source_tensors.get_tensor(f"vpm.blocks.{layer}.mlp.fc2.bias")

    # vpm.blocks.0.norm1.weight -> vision_model.encoder.layers.0.layer_norm1.weight
    tensors_to_save[f"vision_model.encoder.layers.{layer}.layer_norm1.weight"] = source_tensors.get_tensor(f"vpm.blocks.{layer}.norm1.weight")
    # vpm.blocks.0.norm1.bias -> vision_model.encoder.layers.0.layer_norm1.bias
    tensors_to_save[f"vision_model.encoder.layers.{layer}.layer_norm1.bias"] = source_tensors.get_tensor(f"vpm.blocks.{layer}.norm1.bias")
    # vpm.blocks.0.norm2.weight -> vision_model.encoder.layers.0.layer_norm2.weight
    tensors_to_save[f"vision_model.encoder.layers.{layer}.layer_norm2.weight"] = source_tensors.get_tensor(f"vpm.blocks.{layer}.norm2.weight")
    # vpm.blocks.0.norm2.bias -> vision_model.encoder.layers.0.layer_norm2.bias
    tensors_to_save[f"vision_model.encoder.layers.{layer}.layer_norm2.bias"] = source_tensors.get_tensor(f"vpm.blocks.{layer}.norm2.bias")

    


source_tensors = safe_open("/home/jeeves/MiniCPM-V-2.0/model-00001-of-00002.safetensors", framework="pt")

for key in source_tensors.keys():
    print(key)
    tensors_to_save[key] = source_tensors.get_tensor(key)
    
save_file(tensors_to_save, "/home/jeeves/Siglip-MiniCPM-V-2.0/model.safetensors", metadata={"format": "pt"})