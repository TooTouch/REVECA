python run_captioning.py --do_train --evaluate_during_training \
                         --gpu_ids "0 1 2 3" \
                         --per_gpu_train_batch_size 4 \
                         --per_gpu_eval_batch_size 8 \
                         --save_steps 10000 \
                         --wandb
