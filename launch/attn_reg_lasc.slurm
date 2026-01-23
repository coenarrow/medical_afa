


torchrun --nproc_per_node=1 --master_port=29519 scripts/dist_train_coco.py --config configs/coco_attn_reg.yaml --pooling gmp --crop_size 512 --work_dir work_dir_attn_coco

