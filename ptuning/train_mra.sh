PRE_SEQ_LEN=128
LR=1e-3

CUDA_VISIBLE_DEVICES=0 python3 ptuning/main.py \
    --do_train \
    --train_file data/MRA/train_conclusion.json \
    --validation_file data/MRA/val_conclusion.json \
    --prompt_column Input \
    --response_column Output \
    --overwrite_cache \
    --model_name_or_path /mnt/users/LLMa/ChatGLM-6B/output/lung_conclusion-chatglm-6b-pt-128-2e-2_0413/checkpoint-15000 \
    --output_dir output/mra_conclusion-chatglm-6b-pt-with-lung-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 512 \
    --max_target_length 512 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 1000 \
    --logging_steps 10 \
    --save_steps 500 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 8

