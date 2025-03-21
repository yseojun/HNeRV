CUDA_VISIBLE_DEVICES=1 python efficient_nvloader.py --frames 50 \
--ckt output/0303/shaman_1_lightfield/1_1_1__Dim64_16_FC9_16_KS0_1_5_RED1.2_low12_blk1_1_e4800_b2_quant_M8_E6_lr0.001_cosine_0.1_1_0.1_L2_Size3.0_ENC_convnext_5,4,3,2,2_DEC_pshuffel_5,4,3,2,2_gelu1_1/quant_vid.pth \
--decoder output/0303/shaman_1_lightfield/1_1_1__Dim64_16_FC9_16_KS0_1_5_RED1.2_low12_blk1_1_e4800_b2_quant_M8_E6_lr0.001_cosine_0.1_1_0.1_L2_Size3.0_ENC_convnext_5,4,3,2,2_DEC_pshuffel_5,4,3,2,2_gelu1_1/img_decoder.pth \
--dump_dir visualize/shaman_1 \
--eval_interpolation True