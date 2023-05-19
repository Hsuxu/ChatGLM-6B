import os
import torch
import pandas as pd
from peft import TaskType,LoraConfig,get_peft_model
from transformers import AutoModel, AutoTokenizer
device = torch.device('cuda:3')
model_path = '/mnt/users/LLMa/ChatGLM-6B/output/lung_conclusion-chatglm-6b-pt-128-2e-2_0413/checkpoint-15000'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().to(device)
lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                  r=8,
                                  lora_alpha=32,
                                  lora_dropout=0.1)
for name,para in model.named_parameters():
    print(name,para.data.dtype)
model = get_peft_model(model,lora_cfg)
model.print_trainable_parameters()
model = model.half()
for name,para in model.named_parameters():
    print(name,para.data.dtype)
# print(model)
