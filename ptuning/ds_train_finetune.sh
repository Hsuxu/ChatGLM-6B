count=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

LR=2e-2
PRE_SEQ_LEN=128

echo $count
echo $MASTER_PORT
echo $LR
echo $PRE_SEQ_LEN

deepspeed --num_gpus=$count --master_port $MASTER_PORT ptuning/main.py \
    --deepspeed ptuning/deepspeed.json \
    --do_train \
    --train_file data/MRA/train_conclusion.json \
    --validation_file data/LUNG/val_conclusion.json \
    --prompt_column Input \
    --response_column Output \
    --overwrite_cache \
    --model_name_or_path /mnt/users/LLMa/ChatGLM-6B/output/lung_conclusion-chatglm-6b-pt-128-2e-2_0413/checkpoint-15000 \
    --output_dir output/combine_conclusion-chatglm-6b-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 512 \
    --max_target_length 512 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --predict_with_generate \
    --max_steps 20000 \
    --logging_steps 10 \
    --save_steps 2000 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --fp16 \

