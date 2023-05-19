
LR=1e-4

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

CUDA_VISIBLE_DEVICES=0 python3 ptuning/main.py \
    --do_train \
    --train_file data/MRA/train_conclusion.json \
    --validation_file data/MRA/val_conclusion.json \
    --prompt_column Input \
    --response_column Output \
    --overwrite_cache \
    --model_name_or_path /mnt/users/LLMa/ChatGLM-6B/output/lung_conclusion-chatglm-6b-pt1-128-1e-3/checkpoint-10000 \
    --output_dir ./output/combination-chatglm-6b-pt-128-2e-2_0413-ft-$LR \
    --overwrite_output_dir \
    --max_source_length 512 \
    --max_target_length 512 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --predict_with_generate \
    --max_steps 1500 \
    --logging_steps 10 \
    --save_steps 500 \
    --learning_rate $LR \
    --fp16

