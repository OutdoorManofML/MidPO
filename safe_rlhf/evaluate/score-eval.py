import torch
import json
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from safe_rlhf.models import AutoModelForScore
from .input_normalize import PROMPT_INPUT
import os
from safe_rlhf.utils import (
    seed_everything,
)
seed_everything(123)
json_file_path = "/root/safe-rlhf/dataroot/results/MidPO.json"

with open(json_file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

reward_device = torch.device("cuda:0")
cost_device = torch.device("cuda:1")

reward_model = AutoModelForScore.from_pretrained(
    '/root/safe-rlhf/dataroot/models/helpfulness-reward-eval',
    torch_dtype=torch.bfloat16,
    device_map={"": reward_device}
)

cost_model = AutoModelForScore.from_pretrained(
    '/root/safe-rlhf/dataroot/models/safety-reward-eval',
    torch_dtype=torch.bfloat16,
    device_map={"": cost_device}
)

tokenizer = AutoTokenizer.from_pretrained('/root/safe-rlhf/dataroot/models/helpfulness-reward-eval')

results = []
reward_scores = []
cost_scores = []

for item in tqdm(data, desc="Processing Data"):
    prompt = item.get("prompt")
    red_answer = item.get("red_answer")
    input_text = PROMPT_INPUT.format(question=prompt, answer=red_answer)
    input_ids = tokenizer(input_text, return_tensors='pt')

    with torch.no_grad():
        input_ids_gpu0 = {k: v.to(reward_device) for k, v in input_ids.items()}
        reward = reward_model(
            input_ids=input_ids_gpu0['input_ids'],
            attention_mask=input_ids_gpu0['attention_mask'],
        ).end_scores.squeeze(dim=-1).item()

        input_ids_gpu1 = {k: v.to(cost_device) for k, v in input_ids.items()}
        cost = cost_model(
            input_ids=input_ids_gpu1['input_ids'],
            attention_mask=input_ids_gpu1['attention_mask'],
        ).end_scores.squeeze(dim=-1).item()

        reward_scores.append(reward)
        cost_scores.append(cost)
        results.append({
            "question": prompt,
            "answer": red_answer,
            "cost": cost,
            "reward": reward,
        })

mean_reward = round(sum(reward_scores) / len(reward_scores), 2)
mean_cost = round(sum(cost_scores) / len(cost_scores), 2)

print(f"Helpfulness Score: {mean_reward:.2f}")
print(f"Safety Score: {mean_cost:.2f}")

results.append({
    "question": "Average Score",
    "answer": "",
    "cost": mean_cost,
    "reward": mean_reward
})

input_dir = os.path.dirname(json_file_path)
file_name = os.path.basename(json_file_path)

save_dir = os.path.join("/root/safe-rlhf/dataroot/results")
os.makedirs(save_dir, exist_ok=True)
excel_file_path = os.path.join(save_dir, file_name.replace(".json", ".xlsx"))
df = pd.DataFrame(results)
df.to_excel(excel_file_path, index=False)

print(f"Results have been saved to {excel_file_path}")
