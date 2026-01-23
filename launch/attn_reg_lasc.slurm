#!/bin/bash
#SBATCH --job-name=AFA_LASC
#SBATCH --partition=pophealth
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160000
#SBATCH --gres=gpu:a100:1
#SBATCH --time=6:00:00
#SBATCH --output=logs/AFA_LASC_%A.out
#SBATCH --error=logs/AFA_LASC_%A.err

cd "/group/pgh004/carrow/repo/medical_afa"

echo "changed directory to $PWD"

source .venv/bin/activate

echo "environment activated"

# Memory optimization: reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Note: crop_size reduced from 512 to 320 for memory optimization
# With batch_size=16 and scales=[1, 0.5, 1.5], 512 causes OOM on A100
torchrun --nproc_per_node=1 --master_port=29520 scripts/dist_train_lasc.py --config configs/lasc_attn_reg.yaml --pooling gmp --crop_size 320 --work_dir work_dir_lasc
# torchrun --nproc_per_node=1 --master_port=29520 scripts/dist_train_brats.py --config configs/brats_attn_reg.yaml --pooling gmp --crop_size 512 --work_dir work_dir_brats
# torchrun --nproc_per_node=1 --master_port=29520 scripts/dist_train_kits.py --config configs/kits_attn_reg.yaml --pooling gmp --crop_size 512 --work_dir work_dir_kits