<div align=center>

<h1>MidPO: Dual Preference Optimization for Safety and Helpfulness in Large Language Models via a Mixture of Experts Framework</h1>

<div>
      Yupeng Qi</a><sup>1</sup>,
      Ziyu Lyu</a><sup>1&#8224</sup>,
      Min Yang</a><sup>2</sup>,
      Yanlin Wang</a><sup>1</sup>,
      Lu Bai</a><sup>3</sup>,
      Lixin Cui</a><sup>4</sup>,

<div>
  <sup>1</sup>Sun Yat-sen University, <sup>2</sup>Shenzhen Institute of Advanced Technology, Chinese Academy of Sciences, 
  
  <sup>3</sup>Beijing Normal University, <sup>4</sup>Central University of Finance and Economics
       </div>   
<div>
<sup>+</sup> Corresponding author. 
   </div>

</div>

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

</div>


This repository is the official implementation of EMNLP 2025 Paper "**MidPO: Dual Preference Optimization for Safety and Helpfulness in Large
Language Models via a Mixture of Experts Framework**", the code is based on [Safe RLHF](https://github.com/PKU-Alignment/safe-rlhf), which has been instrumental in our research.

## 👉 Quick Start of MidPO

**Step 0. Environment Setup.** 

Depending on your CUDA version, you may need to modify the `conda-recipe.yaml` file (it is currently configured to work with CUDA version 12.4).
```
Run conda-recipe.yaml 
```

**Step 1. Get the base model, before running MidPO, one can down the initial SFT model from:**
```
https://huggingface.co/PKU-Alignment/alpaca-7b-reproduced
```
**Step 2. Obtain the safety and helpfulness preference reward models**
```
bash scripts/reward-model.sh
bash scripts/cost-model.sh
```

or you can use the open-resource: 
```
https://huggingface.co/PKU-Alignment/beaver-7b-unified-cost
https://huggingface.co/PKU-Alignment/beaver-7b-unified-reward
```
**Step 3. Expert Training of MidPO:**

Safety Expert
```
bash scripts/mdpo_safety_expert.sh --model_name_or_path "/root/safe-rlhf/dataroot/models/alpaca-7b" --output_dir output/safety-expert --offload optimizer
```

Helpfulness Expert
```
bash scripts/mdpo_helpfulness_expert.sh --model_name_or_path "/root/safe-rlhf/dataroot/models/alpaca-7b" --output_dir output/helpfulness-expert --offload optimizer
```

**Step 4. Dynamic Routing Mechanism of MidPO**

Reward Ranking → $D_{dual}$ dataset

```
python -m safe_rlhf.evaluate.reward-ranking
```

Router Training
```
bash scripts/mdpo_router.sh --model_name_or_path "/root/safe-rlhf/dataroot/models/alpaca-7b" --output_dir output/router --offload optimizer
```
**Step 5. Evaluation**

Generate responses for our MidPO：
```
python3 -m safe_rlhf.evaluate.generate --output_dir dataroot/results
```

Conduct Reward-model-based Evalutaion：
```
python -m safe_rlhf.evaluate.score-eval
```

For a more customized evaluation, you can substitute other models for the reward and cost evaluations.
## 🌟 Note
All paths should be either local path or Hugging Face model path.

## ☎️ Contact

Please contact the first author for queries.

- Yupeng Qi, qiyp7@mail2.sysu.edu.cn
