cd ..

python main.py --exp_name 'VideoBoundaryCoCa_0529_large-LoRA' --do_train \
                    --image_modelname vit_huge_patch14_224_in21k \
                    --unimodal_modelname gpt2-large \
                    --multimodal_modelname gpt2-large \
                    --batch_size 16 \
                    --lr 8e-4 \
                    --num_training_steps 10000 \
                    --log_interval 5 \
                    --save_interval 250 \
                    --num_warmup_steps 300 \
                    --use_saved_frame \
                    --num_img_queries 16 \
                    --aggregation_frames_method aggregation_frames_method2 \
                    --caption_loss_weight 2 \
                    --use_frame_position \
                    --use_seg_features \
                    --use_tsn_features \
                    --use_temporal_pairwise_difference \
                    --use_text_decoder_lora \
                    --use_replace_01 \
		            --num_workers 0 \
                    --gpu_ids '1 2 3 4 5' \
		            --wandb