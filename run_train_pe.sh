CUDA_VISIBLE_DEVICES=1 python train_nerv_all.py --outf 0311 --data_path ./data/shaman_1 --vid shaman_1_lightfield \
   --conv_type convnext pshuffel --act gelu --norm none --crop_list 480_960  \
   --resize_list -1 --loss L2  --fc_hw 2_4 \
   --dec_strds 5 4 3 2 2 --ks 0_1_5 --reduce 1.2 --num_blks 1_1 \
   --modelsize 3  -e 4800 --eval_freq 30 --lower_width 12 -b 2 --lr 0.001 \
   --embed pe_2_32 \
   --eval_fps