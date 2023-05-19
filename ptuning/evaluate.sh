PRE_SEQ_LEN=128
CHECKPOINT=combine_conclusion-chatglm-6b-pt-128-2e-2
STEP=4000

CUDA_VISIBLE_DEVICES=0 python3 ptuning/main.py \
    --do_predict \
    --validation_file data/MRA/val_conclusion.json \
    --test_file data/MRA/val_conclusion.json \
    --overwrite_cache \
    --prompt_column Input \
    --response_column Output \
    --model_name_or_path ./output/$CHECKPOINT/checkpoint-$STEP  \
    --output_dir ./output/$CHECKPOINT \
    --overwrite_output_dir \
    --max_source_length 4096 \
    --max_target_length 4096 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 8 \