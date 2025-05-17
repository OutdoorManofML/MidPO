from __future__ import annotations
import argparse
from typing import Any
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import deepspeed
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
import torch.nn as nn
from transformers.integrations.deepspeed import HfDeepSpeedConfig
from peft.tuners.lora import Linear
from safe_rlhf.datasets import HelpfulnessExpertPreferenceDataset
from safe_rlhf.models import load_pretrained_models
from safe_rlhf.trainers import SupervisedTrainer
from safe_rlhf.utils import gather_log_probabilities, get_all_reduce_mean
import os
import torch.distributed as dist
from transformers import CONFIG_NAME, WEIGHTS_NAME, PreTrainedModel, PreTrainedTokenizerBase
from safe_rlhf.utils import is_main_process
import abc
import os
import subprocess
import sys
from datetime import datetime
from typing import Any, ClassVar
from safe_rlhf.logger import Logger
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training, PeftModel
from safe_rlhf.models import AutoModelForScore


class HelpfulnessExpertTrainer(SupervisedTrainer):
    TRAINING_TYPE = 'mdpo_helpfulness_expert'

    DATASET_TYPE = HelpfulnessExpertPreferenceDataset

    model: deepspeed.DeepSpeedEngine
    reference_model: deepspeed.DeepSpeedEngine
    ds_train_config: dict[str, Any]
    ds_eval_config: dict[str, Any]

    def __init__(
        self,
        args: argparse.Namespace,
        ds_train_config: dict[str, Any],
        ds_eval_config: dict[str, Any],
    ) -> None:
        """Initialize trainer."""
        self.args = args
        self.ds_train_config = ds_train_config
        self.ds_eval_config = ds_eval_config
        self.scale_coeff = args.scale_coeff
        super().__init__(args, ds_train_config)

    def init_models(self) -> None:
        """Initialize model and tokenizer."""
        if (
            self.ds_train_config is not None
            and self.ds_train_config['zero_optimization']['stage'] == 3
        ):
            self.dstchf_train = HfDeepSpeedConfig(self.ds_train_config)

        if (
            self.ds_eval_config is not None
            and self.ds_eval_config['zero_optimization']['stage'] == 3
        ):
            self.dsechf_eval = HfDeepSpeedConfig(self.ds_eval_config)

        self.model, self.tokenizer = load_pretrained_models(
            self.args.model_name_or_path,
            model_max_length=self.args.max_length,
            padding_side='left',
            auto_model_type=AutoModelForCausalLM,
            trust_remote_code=self.args.trust_remote_code,
        )
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["down_proj"],
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model.enable_input_require_grads()
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        self.model.config.use_cache = False

        self.reference_model, _ = load_pretrained_models(
            self.args.model_name_or_path,
            model_max_length=self.args.max_length,
            padding_side='left',
            auto_model_type=AutoModelForCausalLM,
            trust_remote_code=self.args.trust_remote_code,
        )
        self.reference_model.config.use_cache = False
        for param in self.reference_model.parameters():
            param.requires_grad = False

        self.helpfulness_reward_model = AutoModelForScore.from_pretrained(
            '/root/safe-rlhf/dataroot/models/helpfulness-reward-eval', torch_dtype=torch.bfloat16, device_map='auto')
        for param in self.helpfulness_reward_model.parameters():
            param.requires_grad = False

        rm_tokenizer = AutoTokenizer.from_pretrained('/root/safe-rlhf/dataroot/models/helpfulness-reward-eval')

        rm_vocab = rm_tokenizer.get_vocab()
        vocab = self.tokenizer.get_vocab()

        if vocab == rm_vocab:
            print("Tokenizers have the same vocabulary.")
        else:
            print("Tokenizers have different vocabularies.")

    def init_datasets(self) -> None:
        """Initialize training and evaluation datasets."""
        train_dataset = self.DATASET_TYPE(
            self.args.train_datasets,
            tokenizer=self.tokenizer,
        )

        if self.args.need_eval:
            if self.args.eval_datasets is None and self.args.eval_split_ratio is not None:
                train_dataset, eval_dataset = train_dataset.split_train_test(
                    split_ratio=self.args.eval_split_ratio,
                )
            elif self.args.eval_datasets is not None and self.args.eval_split_ratio is None:
                eval_dataset = self.DATASET_TYPE(
                    self.args.eval_datasets,
                    tokenizer=self.tokenizer,
                )
            else:
                raise ValueError('Either `eval_datasets` or `eval_split_ratio` should be provided.')

            self.eval_dataloader = DataLoader(
                eval_dataset,
                collate_fn=eval_dataset.get_collator(),
                sampler=DistributedSampler(eval_dataset, shuffle=True),
                batch_size=self.args.per_device_eval_batch_size,
            )
        else:
            self.eval_dataloader = None

        self.train_dataloader = DataLoader(
           self.train_dataset,
           collate_fn=train_dataset.get_collator(),
           sampler=DistributedSampler(self.train_dataset, shuffle=True),
           batch_size=self.args.per_device_train_batch_size,
            )

    def init_engines(self) -> None:
        super().init_engines()
        self.reference_model, *_ = deepspeed.initialize(
            model=self.reference_model,
            config=self.ds_eval_config,
        )

    @staticmethod
    def compute_log_probs(
            model: AutoModelForCausalLM,
            input_ids: torch.LongTensor,
            attention_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        logits = model(input_ids, attention_mask=attention_mask).logits
        log_probs = gather_log_probabilities(logits[:, :-1], input_ids[:, 1:])
        return log_probs

    def loss(  # pylint: disable=too-many-locals
        self,
        better_input_ids: torch.LongTensor,  # size = (B, L)
        better_attention_mask: torch.BoolTensor,  # size = (B, L)
        worse_input_ids: torch.LongTensor,  # size = (B, L)
        worse_attention_mask: torch.BoolTensor,  # size = (B, L)
    ) -> dict[str, torch.Tensor]:
        assert better_input_ids.size(0) == worse_input_ids.size(0), 'batch size mismatch!'
        batch_size = better_input_ids.size(0)
        sequence_log_probs = self.compute_log_probs(
            self.model.module.base_model.model,
            input_ids=torch.cat([better_input_ids, worse_input_ids], dim=0),
            attention_mask=torch.cat([better_attention_mask, worse_attention_mask], dim=0),
        )
        (
            better_sequence_log_probs,  # size = (B, L - 1)
            worse_sequence_log_probs,  # size = (B, L - 1)
        ) = sequence_log_probs.chunk(chunks=2, dim=0)

        with torch.no_grad():
            ref_sequence_log_probs = self.compute_log_probs(  # size = (2 * B, L - 1)
                self.reference_model.module,
                input_ids=torch.cat([better_input_ids, worse_input_ids], dim=0),
                attention_mask=torch.cat([better_attention_mask, worse_attention_mask], dim=0),
            )
            (
                ref_better_sequence_log_probs,  # size = (B, L - 1)
                ref_worse_sequence_log_probs,  # size = (B, L - 1)
            ) = ref_sequence_log_probs.chunk(chunks=2, dim=0)
            # New algorithm start
            reward_sequence_loss = self.helpfulness_reward_model(
                input_ids=torch.cat([better_input_ids, worse_input_ids], dim=0),
                attention_mask=torch.cat([better_attention_mask, worse_attention_mask], dim=0)).end_scores
            (
                better_sequence_reward_loss,  # size = (B, L - 1)
                worse_sequence_reward_loss,  # size = (B, L - 1)
            ) = reward_sequence_loss.chunk(chunks=2, dim=0)
            better_sequence_reward_loss = better_sequence_reward_loss.squeeze(dim=-1)
            worse_sequence_reward_loss = worse_sequence_reward_loss.squeeze(dim=-1)
            # New algorithm end
        losses = []
        better_sample_rewards = []
        worse_sample_rewards = []
        for i in range(batch_size):
            assert not torch.all(
                torch.eq(better_input_ids[i], worse_input_ids[i]),
            ).item(), 'The better and worse answers are the same!'
            better_end_index = better_attention_mask[i].nonzero()[-1].squeeze().item()
            worse_end_index = worse_attention_mask[i].nonzero()[-1].squeeze().item()
            diverge_index = (
                (better_input_ids[i] != worse_input_ids[i]).nonzero()[0].squeeze().item()
            )
            assert 0 <= diverge_index <= better_end_index, 'diverge index is out of range!'
            assert 0 <= diverge_index <= worse_end_index, 'diverge index is out of range!'

            better_seq_slice = slice(diverge_index, better_end_index + 1)
            worse_seq_slice = slice(diverge_index, worse_end_index + 1)

            better_log_prob = better_sequence_log_probs[i, better_seq_slice].sum(dim=-1)
            worse_log_prob = worse_sequence_log_probs[i, worse_seq_slice].sum(dim=-1)
            ref_better_log_prob = ref_better_sequence_log_probs[i, better_seq_slice].sum(dim=-1)
            ref_worse_log_prob = ref_worse_sequence_log_probs[i, worse_seq_slice].sum(dim=-1)
            # New algorithm
            better_reward_loss = better_sequence_reward_loss[i]
            worse_reward_loss = worse_sequence_reward_loss[i]
            margin_loss = better_reward_loss - worse_reward_loss
            # filter error examples, ensure helpfulness reward(x,yw) > helpfulness reward(x,yl), see Appendix B.2, Alg 1
            if margin_loss < 0:
                margin_loss = 0
            else:
                margin_loss = margin_loss
            # New algorithm end

            better_log_ratio = better_log_prob - ref_better_log_prob
            worse_log_ratio = worse_log_prob - ref_worse_log_prob
            # Helpful Expert loss
            losses.append(-F.logsigmoid((self.scale_coeff * (better_log_ratio - worse_log_ratio)) - margin_loss))
            better_sample_rewards.append(self.scale_coeff * better_log_ratio.detach())
            worse_sample_rewards.append(self.scale_coeff * worse_log_ratio.detach())

        loss = torch.stack(losses).mean()  # size = ()
        better_sample_reward = torch.stack(better_sample_rewards)  # size = (B,)
        worse_sample_reward = torch.stack(worse_sample_rewards)  # size = (B,)
        reward = better_sample_reward + worse_sample_reward  # size = (B,)
        reward_accuracy = (better_sample_reward > worse_sample_reward).float().mean()  # size = ()
        reward_margin = better_sample_reward - worse_sample_reward  # size = (B,)

        return {
            'loss': loss,
            'reward': reward,
            'better_sample_reward': better_sample_reward,
            'worse_sample_reward': worse_sample_reward,
            'reward_accuracy': reward_accuracy,
            'reward_margin': reward_margin,
        }

    def train_step(
        self,
        better_input_ids: torch.LongTensor,  # size = (B, L)
        better_attention_mask: torch.BoolTensor,  # size = (B, L)
        worse_input_ids: torch.LongTensor,  # size = (B, L)
        worse_attention_mask: torch.BoolTensor,  # size = (B, L)
    ) -> dict[str, Any]:
        """Perform a single training step.

        Args:
            better_input_ids (torch.LongTensor): The input ids of the better answer.
            better_attention_mask (torch.BoolTensor): The attention mask of the better answer.
            worse_input_ids (torch.LongTensor): The input ids of the worse answer.
            worse_attention_mask (torch.BoolTensor): The attention mask of the worse answer.

        Returns:
            dict[str, Any]: training loss, reward, learning rate
        """
        loss_dict = self.loss(
            better_input_ids=better_input_ids,
            better_attention_mask=better_attention_mask,
            worse_input_ids=worse_input_ids,
            worse_attention_mask=worse_attention_mask,
        )
        loss = loss_dict['loss']
        self.model.backward(loss)
        self.model.step()

        with torch.no_grad():
            reward = loss_dict['reward'].mean()
            better_sample_reward = loss_dict['better_sample_reward'].mean()
            worse_sample_reward = loss_dict['worse_sample_reward'].mean()
            reward_accuracy = loss_dict['reward_accuracy']
            reward_margin = loss_dict['reward_margin'].mean()

            loss = get_all_reduce_mean(loss)
            reward = get_all_reduce_mean(reward)
            better_sample_reward = get_all_reduce_mean(better_sample_reward)
            worse_sample_reward = get_all_reduce_mean(worse_sample_reward)
            reward_accuracy = get_all_reduce_mean(reward_accuracy)
            reward_margin = get_all_reduce_mean(reward_margin)

        return {
            'train/loss': loss.item(),
            'train/reward': reward.item(),
            'train/better_sample_reward': better_sample_reward.item(),
            'train/worse_sample_reward': worse_sample_reward.item(),
            'train/reward_accuracy': reward_accuracy.item(),
            'train/reward_margin': reward_margin.item(),
            'train/lr': self.model.optimizer.param_groups[0]['lr'],
        }

    def save(
        self,
        model: deepspeed.DeepSpeedEngine | None = None,
    ) -> None:
        """Save model and tokenizer in Hugging Face format."""
        dist.barrier()

        if model is None:
            model = self.model  # pylint: disable=no-member
        self.logger.print(f'Saving model to "{self.args.output_dir}" ...')

        output_config_file = os.path.join(self.args.output_dir, CONFIG_NAME)
        model_to_save: PreTrainedModel = getattr(model, 'module', model)

        if is_main_process():
            model_to_save.config.to_json_file(output_config_file)
            self.tokenizer.save_pretrained(self.args.output_dir)
            model_to_save.save_pretrained(os.path.join(self.args.output_dir, "lora_weight"))
            print("LoRA Model Saved!")
