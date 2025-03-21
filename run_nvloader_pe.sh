CUDA_VISIBLE_DEVICES=0 python efficient_nvloader.py --frames 50 \
--ckt output/0318/shaman_1_lightfield/1_1_1_pe_2_32_Dim64_16_FC2_4_KS0_1_5_RED1.2_low12_blk1_1_e300_b2_quant_M8_E6_lr0.001_cosine_0.1_1_0.1_L2_Size3.0_ENC_convnext__DEC_pshuffel_4,3,3,3,2_gelu1_1/quant_vid.pth \
--decoder output/0318/shaman_1_lightfield/1_1_1_pe_2_32_Dim64_16_FC2_4_KS0_1_5_RED1.2_low12_blk1_1_e300_b2_quant_M8_E6_lr0.001_cosine_0.1_1_0.1_L2_Size3.0_ENC_convnext__DEC_pshuffel_4,3,3,3,2_gelu1_1/img_decoder.pth \
--dump_dir visualize/shaman_1_pe \
--pe True
