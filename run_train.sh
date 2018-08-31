CUDA_VISIBLE_DEVICES=0,1 python main.py \
                         -b 24 \
                         --lr 0.0015 \
                         #2>&1 | tee exps/logs/mula_lip.log \
