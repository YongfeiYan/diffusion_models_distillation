
root_dir=$(pwd)
log_dir=$root_dir/data/log/imagenet/finetune

pretrained_model_name_or_path=$root_dir/data/test-pipeline
script=$root_dir/diffdstl/train/progressdstl.py

mkdir -p $log_dir

if [ -L $log_dir/$resume_from_checkpoint ]; then 
    echo 'remove checkpoint link'
    unlink $log_dir/$resume_from_checkpoint
fi
# ln -s $resume_ckpt_path $log_dir/$resume_from_checkpoint

cd $root_dir
rm -rf $log_dir/replay

# 
  # --unet_config_path $unet_config_path \
  # --unet_ckpt_path "$unet_ckpt_path" \
  # --mixed_precision="fp16"
#   --ldm_config_path $ldm_config_path \
#   --resume_from_checkpoint $resume_from_checkpoint \
#   --start_global_step 350000 \
  # --use_debug_examples 100 \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONPATH=.:third_party/stable-diffusion accelerate launch $script \
  --pretrained_model_name_or_path=$pretrained_model_name_or_path \
  --pipeline_save_name stage_one --distill_mode stage_one --uncond 1000 --origin_loss_weight 0.0 --distill_loss_weight 1.0 --dataset_name imagenet \
  --part_parquets 0 --num-batches-per-replay-part 0 --replay-times 1 --replay-batches 250 \
  --loss_key eval_loss \
  --use_ema --ema_decay 0.995 \
  --validation_prompt_path $root_dir/configs/imagenet/prompts_imagenet.txt \
  --train_batch_size=60 --encoder_batch_size 60 \
  --gradient_accumulation_steps=2 \
  --gradient_checkpointing \
  --max_train_steps=1000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=1000 \
  --snr_gamma 0.0 \
  --dataloader_num_workers 12 \
  --logging_dir $log_dir \
  --output_dir $log_dir  \
  --checkpointing_steps 500 \
  --checkpoints_total_limit 1 \
  --validation_epochs 1 &> $log_dir/run.log &

sleep 0.1 
disown -a
# tail -f $log_dir/run.log 
echo log_file: $log_dir/run.log
