# MidPO
This repository is the official implementation of MidPO: Dual Preference Optimization for Safety and Helpfulness in Large
Language Models via a Mixture of Experts Framework

## Installation Guide
Run conda-recipe.yaml


Depending on your CUDA version, you may need to modify the `conda-recipe.yaml` file (it is currently configured to work with CUDA version 12.4).

## How to Run
###1. Get the base model
Before running the algorithms, one can down the base model from:
```
https://huggingface.co/PKU-Alignment/alpaca-7b-reproduced
```
###2. Obtain the safety and helpfulness preference models
```
bash scripts/reward-model.sh
bash scripts/cost-model.sh
```

or you can use the open-resource: 
```
https://huggingface.co/PKU-Alignment/beaver-7b-unified-cost
https://huggingface.co/PKU-Alignment/beaver-7b-unified-reward
```
###3.Expert Training of MidPO:
**Safety Expert**
```
bash scripts/mdpo_safety_expert.sh --model_name_or_path "/root/safe-rlhf/dataroot/models/alpaca-7b" --output_dir output/safety-expert --offload optimizer```


**Helpfulness Expert**
```
bash scripts/mdpo_helpfulness_expert.sh --model_name_or_path "/root/safe-rlhf/dataroot/models/alpaca-7b" --output_dir output/helpfulness-expert --offload optimizer
```

###4. Dynamic Routing Mechanism of MidPO
**Reward Ranking → D_{dual} dataset**

```
python -m safe_rlhf.evaluate.reward-ranking
```

**Router Training**
```
bash scripts/mdpo_router.sh --model_name_or_path "/root/safe-rlhf/dataroot/models/alpaca-7b" --output_dir output/router --offload optimizer
```
###5. Evaluation
Generate responses for our MidPO：
```
python3 -m safe_rlhf.evaluate.generate --output_dir dataroot/results
```

Conduct Reward-model-based Evalutaion：
```
python -m safe_rlhf.evaluate.score-eval
```

For a more customized evaluation, you can substitute other models for the reward and cost evaluations.
### Note
All paths should be either local path or Hugging Face model path.