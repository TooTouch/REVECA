python main.py --exp_name 'VideoBoundaryCoCa' --do_train \
                    --batch_size 16 \
                    --num_training_steps 10000 \
                    --log_interval 5 \
                    --save_interval 1000 \
                    --use_saved_frame \
                    --gpu_ids '0 1 2 3' \
                    --wandb
