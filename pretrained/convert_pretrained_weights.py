import torch
import re

# Load Hugging Face weights
hf_weights = torch.load('pretrained/mit_b1.bin', map_location='cpu')

# First pass: rename keys from transformers format to intermediate format
intermediate_weights = {}
for key, value in hf_weights.items():
    # Skip classifier head (not needed for encoder-only usage)
    if key.startswith('classifier.'):
        continue
    # Remove 'segformer.encoder.' prefix
    new_key = re.sub(r'^segformer\.encoder\.', '', key)
    # Convert 'block.X.Y' to 'blockX.Y' (block index is 0-based, needs +1)
    new_key = re.sub(r'block\.(\d+)\.', lambda m: f'block{int(m.group(1))+1}.', new_key)
    # Convert 'patch_embeddings.X' to 'patch_embedX'
    new_key = re.sub(r'patch_embeddings\.(\d+)\.', lambda m: f'patch_embed{int(m.group(1))+1}.', new_key)
    # Convert 'layer_norm.X' to 'normX'
    new_key = re.sub(r'^layer_norm\.(\d+)\.', lambda m: f'norm{int(m.group(1))+1}.', new_key)
    # Fix layer norm naming within blocks
    new_key = new_key.replace('.layer_norm_1.', '.norm1.')
    new_key = new_key.replace('.layer_norm_2.', '.norm2.')
    # Fix MLP naming
    new_key = new_key.replace('.mlp.dense1.', '.mlp.fc1.')
    new_key = new_key.replace('.mlp.dense2.', '.mlp.fc2.')
    # Fix projection naming
    new_key = new_key.replace('proj.projection.', 'proj.')
    # Fix attention output projection
    new_key = new_key.replace('.attention.output.dense.', '.attn.proj.')
    # Keep attention.self for now, will handle in second pass
    new_key = new_key.replace('.attention.self.', '.attn.')
    # Fix sr layer_norm to norm
    new_key = new_key.replace('.attn.layer_norm.', '.attn.norm.')
    # Fix patch_embed layer_norm to norm
    new_key = new_key.replace('.layer_norm.', '.norm.')
    intermediate_weights[new_key] = value

# Second pass: combine separate key/value into kv, rename query to q
new_weights = {}
processed_kv = set()

for key, value in intermediate_weights.items():
    # Handle query -> q
    if '.attn.query.' in key:
        new_key = key.replace('.attn.query.', '.attn.q.')
        new_weights[new_key] = value
    # Handle key/value -> kv (concatenate)
    elif '.attn.key.' in key:
        # Find corresponding value tensor
        value_key = key.replace('.attn.key.', '.attn.value.')
        if value_key in intermediate_weights:
            kv_key = key.replace('.attn.key.', '.attn.kv.')
            key_tensor = value
            value_tensor = intermediate_weights[value_key]
            # Concatenate key and value weights along the output dimension
            kv_tensor = torch.cat([key_tensor, value_tensor], dim=0)
            new_weights[kv_key] = kv_tensor
            processed_kv.add(value_key)
    elif '.attn.value.' in key:
        # Skip if already processed as part of kv
        if key in processed_kv:
            continue
        new_weights[key] = value
    else:
        new_weights[key] = value

torch.save(new_weights, 'pretrained/mit_b1.pth')
print(f"Converted {len(new_weights)} keys to pretrained/mit_b1.pth")