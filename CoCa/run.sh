cd ..

python main.py --exp_name 'VideoBoundaryCoCa_final_large-LoRA_img_ft_label' --do_train \
                    --image_modelname vit_large_patch16_224_in21k \
                    --unimodal_modelname gpt2 \
                    --multimodal_modelname gpt2 \
                    --batch_size 8 \
                    --lr 8e-4 \
                    --num_training_steps 10000 \
                    --log_interval 5 \
                    --save_interval 250 \
                    --num_warmup_steps 300 \
                    --use_saved_frame \
                    --num_img_queries 16 \
                    --aggregation_frames_method aggregation_frames_method2 \
                    --contrastive_loss_weight 0.1 \
                    --use_frame_position \
                    --use_seg_features \
                    --use_tsn_features \
                    --use_temporal_pairwise_difference \
                    --use_replace_01 \
                    --use_label \
	                --num_workers 0 \
                    --use_img_encoder_lora \
                    --use_train_val \
                    --gpu_ids '0 1 2 3 4 5' \
                    --wandb

