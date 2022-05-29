cd ..

python main.py --exp_name 'VideoBoundaryCoCa_0528_large-LoRA' \
               --do_val \
               --batch_size 64 \
               --num_beams 5 \
               --checkpoint_path ./output/VideoBoundaryCoCa_0528_large-LoRA/VideoBoundaryCoCa_0528_large-LoRA_step6249.pt \
               --unimodal_modelname gpt2-large \
               --multimodal_modelname gpt2-large \
               --use_saved_frame \
               --num_img_queries 16 \
               --aggregation_frames_method aggregation_frames_method2 \
               --use_frame_position \
               --use_seg_features \
               --use_tsn_features \
               --use_temporal_pairwise_difference \
               --use_lora \
               --num_workers 0