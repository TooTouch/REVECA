cd ..

python main.py --exp_name 'VideoBoundaryCoCa_0527' --do_train \
                    --image_modelname vit_huge_patch14_224_in21k \
                    --unimodal_modelname gpt2 \
                    --multimodal_modelname gpt2 \
                    --batch_size 16 \
                    --num_training_steps 10000 \
                    --log_interval 5 \
                    --save_interval 1000 \
                    --num_warmup_steps 200 \
                    --use_saved_frame \
                    --num_img_queries 16 \
                    --aggregation_frames_method aggregation_frames_method2 \
                    --use_frame_position \
                    --use_seg_features \
                    --use_tsn_features \
                    --use_temporal_pairwise_difference \
                    --gpu_ids '0 1 2 3' \
                    --wandb
