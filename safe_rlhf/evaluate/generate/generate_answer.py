from __future__ import annotations
from transformers import LlamaForCausalLM
import argparse
import json
import os
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import openai
import torch
from tqdm import tqdm
from safe_rlhf.configs.constants import PROMPT_INPUT
from safe_rlhf.models import load_pretrained_models
from safe_rlhf.utils import to_device
from datasets import load_dataset
from safetensors import safe_open
from peft import PeftModel

HHH_HELP_PATH = os.path.join(os.path.dirname(__file__), 'hhh_helpful.json')
XSTEST_PATH = os.path.join(os.path.dirname(__file__), 'xstest.json')
from safe_rlhf.utils import (
    seed_everything,
)
# default
seed_everything(42)

def parse_arguments() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate models with gpt4',
    )

    # Model
    parser.add_argument(
        '--red_corner_model_name_or_path',
        type=str,
        help='the name or path of the first model (champion) in the arena to load from',
    )
    parser.add_argument(
        '--blue_corner_model_name_or_path',
        type=str,
        help='the name or path of the second model (challenger) in the arena to load from',
    )

    # Logging
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Where to store the eval output.',
    )

    return parser.parse_args()


def load_routers_params(router_path, model):
    saved_state_dict = torch.load(router_path)
    for layer_idx, layer in enumerate(model.model.layers):
        routers = layer.mlp.router_layers
        alpha_state_dict = saved_state_dict[f'layer_{layer_idx}_alpha']
        beta_state_dict = saved_state_dict[f'layer_{layer_idx}_beta']
        routers.alpha_.load_state_dict(alpha_state_dict)
        routers.beta_.load_state_dict(beta_state_dict)


def load_lora_weights(model_path, dtype, device):
    lora_A, lora_B = {}, {}
    tensors = {}
    with safe_open(model_path, framework="pt", device="cuda:2") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    for k, v in tensors.items():
        ks = k.split('.')
        if ks[7] == 'lora_A':
            lora_A[ks[4]] = v.to(device, dtype=dtype)
        if ks[7] == 'lora_B':
            lora_B[ks[4]] = v.to(device, dtype=dtype)
    return lora_A, lora_B


def generate_answer(problems: list[str], model_name_or_path: str) -> list[str]:

    # for the moe model:
    device_moe = torch.device("cuda:2")

    _, tokenizer = load_pretrained_models(
           model_name_or_path,
           auto_device_mapping=True,
           trust_remote_code=True,
       )
    from .llama_with_router import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained('/root/safe-rlhf/dataroot/models/alpaca-7b')
    model = model.to(device_moe)
    lora_0_A, lora_0_B = load_lora_weights(
           '/root/safe-rlhf/output/safety-expert/lora_weight/adapter_model.safetensors',
           dtype=model.dtype, device=device_moe
     )
    lora_1_A, lora_1_B = load_lora_weights(
           '/root/safe-rlhf/output/helpfulness-expert/lora_weight/adapter_model.safetensors',
           dtype=model.dtype, device=device_moe)

    for name, param in model.named_parameters():
        ns = name.split('.')
        if len(ns) >= 5 and ns[3] == 'mlp' and ns[4] == 'lora_0' and ns[5] == 'linear1':
            param.data = lora_0_A[ns[2]].to(param.device, dtype=param.dtype).clone().detach()
        if len(ns) >= 5 and ns[3] == 'mlp' and ns[4] == 'lora_0' and ns[5] == 'linear2':
            param.data = lora_0_B[ns[2]].to(param.device, dtype=param.dtype).clone().detach()
        if len(ns) >= 5 and ns[3] == 'mlp' and ns[4] == 'lora_1' and ns[5] == 'linear1':
            param.data = lora_1_A[ns[2]].to(param.device, dtype=param.dtype).clone().detach()
        if len(ns) >= 5 and ns[3] == 'mlp' and ns[4] == 'lora_1' and ns[5] == 'linear2':
            param.data = lora_1_B[ns[2]].to(param.device, dtype=param.dtype).clone().detach()
    load_routers_params("/root/safe-rlhf/output/router/router.pth", model)

    answers = []
    print(f'Generating answers with {model_name_or_path}')
    for problem in tqdm(problems):
        prompt = PROMPT_INPUT.format(input=problem['prompt'])

        # for other model
        input_ids = to_device(
            tokenizer(prompt, return_tensors='pt'),
            device=(device_moe),
        )

        output_ids = model.generate(
            **input_ids,
            temperature=0.7,
            do_sample=True,
            max_length=2048,
        )
        answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)[len(prompt):]
        answers.append(answer)
    return answers


def read_first_k_entries_json(file_path, k):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    return data[:k]


def read_first_k_entries_hf(k):
    # # SafeRLHF Dataset
    # seen_prompts = set()
    # def is_unique_prompt(sample):
    #     if sample['prompt'] in seen_prompts:
    #         return False
    #     seen_prompts.add(sample['prompt'])
    #     return True
    # saferlhf_dataset = load_dataset('PKU-Alignment/PKU-SafeRLHF', split='test')
    # safety_test_set = saferlhf_dataset.filter(is_unique_prompt)
    # safety_dataset = safety_test_set['prompt']
    # question = safety_dataset
    # category = safety_test_set['response_0_harm_category']

    # # Do-not-answer
    # do_not_answer_dataset = load_dataset('LibrAI/do-not-answer', split='train', cache_dir='/data/ypqi_data')
    # question = do_not_answer_dataset['question']
    # category = do_not_answer_dataset['types_of_harm']


    # WildGuard Mix
    wild_dataset = load_dataset('allenai/wildguardmix', 'wildguardtest', split='test', cache_dir='/data/ypqi_data')
    wild_dataset_test_set = wild_dataset.filter(lambda example: example['prompt_harm_label'] == 'harmful')
    question = wild_dataset_test_set['prompt']
    category = wild_dataset_test_set['subcategory']

    category = category
    safety_dataset = question

    # Used for WildGuard Mix&Do-not-answer
    k_result = [
        {
            'prompt': safety_dataset[i],
            'category': category[i],
        }
        for i in range(min(k, len(safety_dataset)))
    ]
    all_result = [
        {
            'prompt': safety_dataset[i],
            'category': category[i],
        }
        for i in range(len(safety_dataset))
    ]

    # # Used for Safe-RLHF
    # k_result = [
    #     {
    #         'prompt': safety_dataset[i],
    #         'category': ", ".join([key for key, value in category[i].items() if value is True]),
    #     }
    #     for i in range(min(k, len(safety_dataset)))
    # ]
    #
    # all_result = [
    #     {
    #         'prompt': safety_dataset[i],
    #         'category': ", ".join([key for key, value in category[i].items() if value is True]),
    #     }
    #     for i in range(len(safety_dataset))
    #  ]
    #
    result = k_result

    return result

def main() -> None:
    """The main function."""
    args = parse_arguments()
    problems = read_first_k_entries_hf(150)
    # problems = read_first_k_entries_json(XSTEST_PATH, 250)
    red_answer = generate_answer(problems, '/root/safe-rlhf/dataroot/models/alpaca-7b')
    results = []

    for problem, answer1 in tqdm(
        zip(problems, red_answer),
        total=len(problems),
    ):
        # category = 'none'
        category = problem['category']
        results.append(
            {
                'prompt': problem['prompt'],
                'red_corner_model': args.red_corner_model_name_or_path,
                'red_answer': answer1,
                'category': category,
            },
        )
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, 'MidPO_Results.json'), mode='w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
