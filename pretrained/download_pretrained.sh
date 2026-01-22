!/bin/bash
curl -L -o ./pretrained/mit_b1.bin https://huggingface.co/nvidia/mit-b1/resolve/main/pytorch_model.bin
uv run ./pretrained/convert_pretrained_weights.py