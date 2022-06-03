cd ..

python main.py --exp_name 'VideoBoundaryCoCa_final_large-LoRA_img_ft_label' \
               --savedir ./final_output \
               --do_val \
               --batch_size 16 \
               --num_beams 10 \
               --num_beam_groups 5 \
               --checkpoint_path  /projects/GEBC_CVPR2022/Coca/output/VideoBoundaryCoCa_final/VideoBoundaryCoCa_final_large-LoRA_img_ft_label_step1499.pt \
               --image_modelname vit_large_patch16_224_in21k \
               --unimodal_modelname gpt2 \
               --multimodal_modelname gpt2 \
               --use_saved_frame \
               --num_img_queries 16 \
               --aggregation_frames_method aggregation_frames_method2 \
               --use_frame_position \
               --use_seg_features \
               --use_tsn_features \
               --use_temporal_pairwise_difference \
               --use_replace_01 \
               --use_label \
               --use_img_encoder_lora \
               --num_workers 0 \
               --gpu_ids '0'
