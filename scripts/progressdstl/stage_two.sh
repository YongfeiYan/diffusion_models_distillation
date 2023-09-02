
root_dir=$(pwd)
script=$root_dir/diffdstl/train/progressdstl.py
log_dir=$root_dir/data/log/imagenet/stage_two
pretrained_model_name_or_path=$root_dir/data/log/imagenet/stage_one/pipeline-converted

cd $root_dir
mkdir -p $log_dir
rm -rf $log_dir/replay

mkdir -p $log_dir/step64
if [ -L $log_dir/step64/step64 ]; then 
  echo 'remove previous ckpt: '$log_dir/step64/step64
  unlink $log_dir/step64/step64
fi 
ln -s $pretrained_model_name_or_path $log_dir/step64/step64
ls -al $log_dir/step64

args=(
"32 2000 0.995"
"16 2000 0.995"
"8 5000 0.995"
"4 20000 0.999"
"2 20000 0.999"
"1 20000 0.999"
)
for i in ${!args[@]}; do 
    item=(${args[$i]})
    prev_step=step$((${item[0]} * 2))
    step=${item[0]}
    cur_step=step$step
    max_train_steps=${item[1]}
    ema_decay=${item[2]}
    echo 'prev_step:' $prev_step
    echo $step - $max_train_steps - $ema_decay
    date
    mkdir -p $log_dir/$cur_step
    rm -rf $log_dir/$cur_step/{replay,images,checkpoint,tracker,step,run}*

    # --use_debug_examples 100 \
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONPATH=.:third_party/stable-diffusion accelerate launch $script \
    --pretrained_model_name_or_path=$log_dir/$prev_step/$prev_step \
    --pipeline_save_name $cur_step --run_mode stage_two --uncond 1000 --origin_loss_weight 0.0 --distill_loss_weight 1.0 --dataset_name imagenet --student_steps $step --snr_weight_mode truncated \
    --replay-times 1 --replay-batches 250 \
    --loss_key eval_loss \
    --use_ema --ema_decay $ema_decay \
    --validation_prompt_path $root_dir/configs/imagenet/prompts_imagenet.txt \
    --train_batch_size=60 --encoder_batch_size 60 \
    --gradient_accumulation_steps=2 \
    --gradient_checkpointing \
    --max_train_steps=$max_train_steps \
    --learning_rate=1e-05 \
    --max_grad_norm=1 \
    --lr_scheduler="constant" --lr_warmup_steps=1000 \
    --snr_gamma 0.0 \
    --dataloader_num_workers 12 \
    --logging_dir $log_dir/$cur_step \
    --output_dir $log_dir/$cur_step  \
    --checkpointing_steps 500 \
    --checkpoints_total_limit 1 \
    --validation_epochs 1 &> $log_dir/$cur_step/run.log
    rm -rf $log_dir/$cur_step/replay
done
