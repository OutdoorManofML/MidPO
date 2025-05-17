from __future__ import annotations
import torch
import json
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer,  LlamaForCausalLM,  AutoModelForCausalLM
from safe_rlhf.models import AutoModelForScore
from .input_normalize import PROMPT_INPUT
import os
from safe_rlhf.utils import (
    seed_everything,
)
import argparse
from safe_rlhf.models import load_pretrained_models
from safe_rlhf.utils import to_device
from datasets import load_dataset
from safetensors import safe_open
from datasets import Dataset

# default
seed_everything(42)

reward_device = torch.device("cuda:0")
cost_device = torch.device("cuda:1")

safe_rlhf_data = load_dataset('PKU-Alignment/PKU-SafeRLHF', split='train')
safe_rlhf_data = safe_rlhf_data[:50]

prompt = safe_rlhf_data['prompt']
y_0 = safe_rlhf_data['response_0']
y_1 = safe_rlhf_data['response_1']
better = safe_rlhf_data['better_response_id']
tokenizer = AutoTokenizer.from_pretrained('/root/safe-rlhf/dataroot/models/helpfulness-reward-eval')

cost_model = AutoModelForScore.from_pretrained(
    '/root/safe-rlhf/dataroot/models/safety-reward-eval',
    torch_dtype=torch.bfloat16,
    device_map={"": cost_device}
)

filtered_data = []
for i in range(len(prompt)):
    x = prompt[i]
    indicator = better[i]
    if indicator == 0:
        y_hw = y_0[i]
        y_hl = y_1[i]
    else:
        y_hw = y_1[i]
        y_hl = y_0[i]
    p_hw = PROMPT_INPUT.format(question=x, answer=y_hw)
    p_hl = PROMPT_INPUT.format(question=x, answer=y_hl)

    input_ids_hw = tokenizer(p_hw, return_tensors='pt')
    input_ids_hl = tokenizer(p_hl, return_tensors='pt')

    with torch.no_grad():
        input_ids_gpu_hw = {k: v.to(cost_device) for k, v in input_ids_hw.items()}
        input_ids_gpu_hl = {k: v.to(cost_device) for k, v in input_ids_hl.items()}
        cost_hw = cost_model(
            input_ids=input_ids_gpu_hw['input_ids'],
            attention_mask=input_ids_gpu_hw['attention_mask'],
        ).end_scores.squeeze(dim=-1).item()

        cost_hl = cost_model(
            input_ids=input_ids_gpu_hl['input_ids'],
            attention_mask=input_ids_gpu_hl['attention_mask'],
        ).end_scores.squeeze(dim=-1).item()
    if cost_hl < 0:
        safe_hl = True
    else:
        safe_hl = False
    if cost_hw < 0:
        safe_hw = True
    else:
        safe_hw = False
    if not safe_hw or not safe_hl:
        continue

    if cost_hw - cost_hl < 0:
        yw = y_hw
        yl = y_hl
    else:
        yw = y_hl
        yl = y_hw

    filtered_data.append({
        'prompt': x,
        'response_0': yw,
        'response_1': yl,
        'is_response_0_safe': safe_hw,
        'is_response_1_safe': safe_hl,
        'better_response_id': 0,
        'safer_response_id': 0

    })

filtered_df = pd.DataFrame(filtered_data)
filtered_dataset = Dataset.from_pandas(filtered_df)
filtered_dataset.save_to_disk('./dual_preference_data')



