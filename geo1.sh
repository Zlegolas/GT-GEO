#!/bin/bash


 python main.py --dataset "IP_GEO" \
                --gamma 0.6 \
                --n_hop 0 \
                --delay_weight 0.1 \
                --epochs 500 \
                --slope 0.1 \
                --batch_size 16 \
                --eval_batch_size 32 \
                --num_layers 8 \
                --dropout 0.005 \
                --attn_dropout 0.005 \
                --drop_prob 0.005 \
                --lr 0.001 \
                --weight_decay 1e-5 \
                 --run 10 \
                --devices 0 \
                --seed 12344

