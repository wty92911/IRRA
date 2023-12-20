#!/bin/bash
DATASET_NAME="CUHK-PEDES"

CUDA_VISIBLE_DEVICES=1 \
python train.py \
--name iira \
--img_aug \
--batch_size 64 \
--MLM \
--dataset_name $DATASET_NAME \
--root_dir './datasets' \
--loss_names 'sdm+mlm+id' \
--num_epoch 60
