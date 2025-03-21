CUDA_VISIBLE_DEVICES=0 python train_nerv_all.py --outf 0320 --data_path ./data/shaman_1 --vid shaman_1_lightfield \
   --conv_type convnext pshuffel --act gelu --norm none  --crop_list 432_864  \
   --resize_list -1 --loss L2  --enc_strds 4 3 3 3 2 --enc_dim 64_16 \
   --dec_strds 4 3 3 3 2 --ks 0_1_5 --reduce 1.2 \
   --modelsize 3  -e 1600 --eval_freq 30  --lower_width 12 -b 2 --lr 0.001 \
   --eval_fps