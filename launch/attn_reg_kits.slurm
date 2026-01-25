#!/bin/bash
#SBATCH --job-name=AFA_KITS_224
#SBATCH --partition=pophealth
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:3
#SBATCH --time=20:00:00
#SBATCH --output=logs/AFA_KITS_%A.out
#SBATCH --error=logs/AFA_KITS_%A.err

cd "/group/pgh004/carrow/repo/medical_afa"

source .venv/bin/activate

echo "environment activated"

# Memory optimization: reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=3 --master_port=29522 scripts/dist_train_kits.py --config configs/kits_attn_reg.yaml --pooling gmp --crop_size 224 --work_dir work_dir_kits_224
