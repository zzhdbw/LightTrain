export CUDA_VISIBLE_DEVICES=7
python sft_train.py \
    --data_path data/identity.json \
    --model_path /mnt/afs/models/Qwen/Qwen2.5-0.5B-Instruct \
    --output_dir /mnt/afs/zzh/ckpt/ \
    --num_train_epochs 8 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --eval_steps 100 \
    --save_steps 200 \
    --save_total_limit 2 \
    --logging_steps 2 \
    --lr_scheduler_type cosine \
    --dataloader_num_workers 4 \
    --report_to swanlab \
    --train_on_prompt True \
    --use_dft_loss True \
    --dft_alpha 0.8 \
 