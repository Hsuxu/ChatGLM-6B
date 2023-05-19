count=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`
echo $count

PRE_SEQ_LEN=128
LR=2e-2

torchrun --nproc_per_node=$count --rdzv_endpoint 0.0.0.0:47923 ptuning/main.py \
    --do_train \
    --train_file data/LUNG/combine_train.json \
    --validation_file data/LUNG/combine_val.json \
    --prompt_column Input \
    --response_column Output \
    --overwrite_cache \
    --model_name_or_path /mnt/data/xuchi/LLM_CKPT/chatglm-6b \
    --output_dir output/combine_conclusion-chatglm-6b-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 512 \
    --max_target_length 512 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 20000 \
    --logging_steps 10 \
    --save_steps 2000 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 8 \