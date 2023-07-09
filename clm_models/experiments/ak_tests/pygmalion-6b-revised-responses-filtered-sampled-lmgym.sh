deepspeed train.py \
  --model_name_or_path PygmalionAI/pygmalion-6b \
  --tokenizer_name AlekseyKorshuk/pygmalion-6b \
  --dataset_name AlekseyKorshuk/revised-responses-filtered-sampled-lmgym \
  --train_to_probs False \
  --do_train \
  --logging_strategy steps \
  --evaluation_strategy no \
  --eval_steps 2100 \
  --save_strategy epoch \
  --save_steps 1 \
  --save_total_limit 10 \
  --logging_steps 10 \
  --logging_first_step \
  --report_to all \
  --output_dir ./checkpoints/pygmalion-6b-revised-responses-filtered-sampled-lmgym-alpaca-2epoch \
  --overwrite_output_dir \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --max_eval_samples 500 \
  --num_train_epochs 2 \
  --eval_first_step False \
  --learning_rate 5e-6 \
  --fp16 \
  --seed 99 \
  --num_eval_prompts 0 \
  --validation_split_percentage 0 \
  --remove_unused_columns False \
  --deepspeed deepspeed_configs/ds_config_soft.json \
  --clean_enabled False \
  --add_reward_scores False \
  --block_size 512 \
  --lr_scheduler_type cosine \
  --gradient_checkpointing False \
  --warmup_ratio 0.03 \
  --weight_decay 0.0 \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --preprocessing_num_workers 32
