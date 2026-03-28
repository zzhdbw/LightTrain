export CUDA_VISIBLE_DEVICES=7
python sft_train.py \
    --data_path data/identity.json \
    --model_path /mnt/afs/models/Qwen/Qwen2.5-0.5B-Instruct \
    --gradient_checkpointing \
    --output_dir /mnt/afs/zzh/ckpt/ \
    --num_train_epochs 8 \
    --learning_rate 2e-5 \
    --lr_scheduler_name "cosine" \
    --lr_scheduler_num_warmup_steps 10 \
    --lr_min 5e-6 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --eval_steps 100 \
    --save_steps 200 \
    --save_total_limit 2 \
    --logging_steps 2 \
    --dataloader_num_workers 4 \
    --report_to swanlab \
    --swanlab_project_name "LightLLMTrainer" \
    --swanlab_group_name "dft no_lr_scheduler" \
    --use_dft_loss \
    --dft_alpha 0.8 \
 