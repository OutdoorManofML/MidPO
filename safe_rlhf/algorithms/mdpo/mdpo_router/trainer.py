from __future__ import annotations
import argparse
from typing import Any
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import deepspeed
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers.integrations.deepspeed import HfDeepSpeedConfig
from peft.tuners.lora import Linear
from safe_rlhf.datasets import PreferenceDataset
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
from safetensors import safe_open


class RouterTrainer(SupervisedTrainer):
    TRAINING_TYPE = 'mdpo_router'
    DATASET_TYPE = PreferenceDataset

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

    def load_lora_weights(self, model_path, target_device):
        lora_A, lora_B = {}, {}
        tensors = {}
        with safe_open(model_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        for k, v in tensors.items():
            ks = k.split('.')
            if ks[7] == 'lora_A':
                lora_A[ks[4]] = v.to(target_device)
            if ks[7] == 'lora_B':
                lora_B[ks[4]] = v.to(target_device)
        return lora_A, lora_B

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

        _, self.tokenizer = load_pretrained_models(
            self.args.model_name_or_path,
            model_max_length=self.args.max_length,
            padding_side='left',
            auto_model_type=AutoModelForCausalLM,
            trust_remote_code=self.args.trust_remote_code,
        )

        device_mogu = torch.device("cpu")
        from .llama_with_router import LlamaForCausalLM
        self.model = LlamaForCausalLM.from_pretrained('/root/safe-rlhf/dataroot/models/alpaca-7b',
                                                      trust_remote_code=self.args.trust_remote_code)
        # lora_0 is the safety expert, lora_1 is the helpfulness expert
        lora_0_A, lora_0_B = self.load_lora_weights(
            '/root/safe-rlhf/output/safety-expert/lora_weight/adapter_model.safetensors',
            device_mogu)
        lora_1_A, lora_1_B = self.load_lora_weights(
            '/root/safe-rlhf/output/helpfulness-expert/lora_weight/adapter_model.safetensors',
            device_mogu)

        for name, param in self.model.named_parameters():
            ns = name.split('.')
            if len(ns) >= 5 and ns[3] == 'mlp' and ns[4] == 'lora_0' and ns[5] == 'linear1':
                param.data = lora_0_A[ns[2]].clone().detach()
            if len(ns) >= 5 and ns[3] == 'mlp' and ns[4] == 'lora_0' and ns[5] == 'linear2':
                param.data = lora_0_B[ns[2]].clone().detach()
            if len(ns) >= 5 and ns[3] == 'mlp' and ns[4] == 'lora_1' and ns[5] == 'linear1':
                param.data = lora_1_A[ns[2]].clone().detach()
            if len(ns) >= 5 and ns[3] == 'mlp' and ns[4] == 'lora_1' and ns[5] == 'linear2':
                param.data = lora_1_B[ns[2]].clone().detach()

        self.model.config.use_cache = False
        for param in self.model.model.parameters():
            param.requires_grad = False

        for layer in self.model.model.layers:
            router = layer.mlp.router_layers
            for param in router.alpha_.parameters():
                param.requires_grad = True
            for param in router.beta_.parameters():
                param.requires_grad = True

        self.model.enable_input_require_grads()
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


    def init_engines(self) -> None:
        super().init_engines()
        self.reference_model, *_ = deepspeed.initialize(
            model=self.reference_model,
            config=self.ds_eval_config,
        )

    @staticmethod
    def compute_ref_log_probs(
            model: AutoModelForCausalLM,
            input_ids: torch.LongTensor,
            attention_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """Compute log probabilities of given sequences."""
        logits = model(input_ids, attention_mask=attention_mask).logits
        out_logits = gather_log_probabilities(logits[:, :-1], input_ids[:, 1:])
        return out_logits

    @staticmethod
    def compute_log_probs_and_l1_loss(
            model: AutoModelForCausalLM,
            input_ids: torch.LongTensor,
            attention_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """Compute log probabilities of given sequences."""
        model_out, reg_alpha, reg_beta = model(input_ids, attention_mask=attention_mask)
        logits = model_out.logits
        out_logits = gather_log_probabilities(logits[:, :-1], input_ids[:, 1:])
        l1_loss = reg_alpha + reg_beta
        return out_logits, l1_loss

    def loss(  # pylint: disable=too-many-locals
            self,
            better_input_ids: torch.LongTensor,  # size = (B, L)
            better_attention_mask: torch.BoolTensor,  # size = (B, L)
            worse_input_ids: torch.LongTensor,  # size = (B, L)
            worse_attention_mask: torch.BoolTensor,  # size = (B, L)
    ) -> dict[str, torch.Tensor]:
        """Loss function for the DPO algorithm.

        Args:
            better_input_ids (torch.LongTensor): The input ids of the better answer.
            better_attention_mask (torch.BoolTensor): The attention mask of the better answer.
            worse_input_ids (torch.LongTensor): The input ids of the worse answer.
            worse_attention_mask (torch.BoolTensor): The attention mask of the worse answer.

        Returns:
            dict[str, torch.Tensor]: loss, reward, better sample reward, worse sample reward
        """
        assert better_input_ids.size(0) == worse_input_ids.size(0), 'batch size mismatch!'
        batch_size = better_input_ids.size(0)
        sequence_log_probs, l1_loss = self.compute_log_probs_and_l1_loss(  # size = (2 * B, L - 1)
            self.model.module,
            input_ids=torch.cat([better_input_ids, worse_input_ids], dim=0),
            attention_mask=torch.cat([better_attention_mask, worse_attention_mask], dim=0),
        )
        (
            better_sequence_log_probs,  # size = (B, L - 1)
            worse_sequence_log_probs,  # size = (B, L - 1)
        ) = sequence_log_probs.chunk(chunks=2, dim=0)
        with torch.no_grad():
            ref_sequence_log_probs = self.compute_ref_log_probs(  # size = (2 * B, L - 1)
                self.reference_model.module,
                input_ids=torch.cat([better_input_ids, worse_input_ids], dim=0),
                attention_mask=torch.cat([better_attention_mask, worse_attention_mask], dim=0),
            )
            (
                ref_better_sequence_log_probs,  # size = (B, L - 1)
                ref_worse_sequence_log_probs,  # size = (B, L - 1)
            ) = ref_sequence_log_probs.chunk(chunks=2, dim=0)
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

            # size = ()
            better_log_prob = better_sequence_log_probs[i, better_seq_slice].sum(dim=-1)
            worse_log_prob = worse_sequence_log_probs[i, worse_seq_slice].sum(dim=-1)
            ref_better_log_prob = ref_better_sequence_log_probs[i, better_seq_slice].sum(dim=-1)
            ref_worse_log_prob = ref_worse_sequence_log_probs[i, worse_seq_slice].sum(dim=-1)
            better_log_ratio = better_log_prob - ref_better_log_prob
            worse_log_ratio = worse_log_prob - ref_worse_log_prob

            losses.append(-F.logsigmoid(self.scale_coeff * (better_log_ratio - worse_log_ratio)))
            better_sample_rewards.append(self.scale_coeff * better_log_ratio.detach())
            worse_sample_rewards.append(self.scale_coeff * worse_log_ratio.detach())

        loss = torch.stack(losses).mean()  # size = ()
        loss = l1_loss + loss
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

        model_to_save: PreTrainedModel = getattr(model, 'module', model)

        if is_main_process():
            # Saver router weights
            router_path = os.path.join(self.args.output_dir, 'router.pth')
            routers_state_dict = {}
            for layer_idx, layer in enumerate(model_to_save.model.layers):
                routers = layer.mlp.router_layers
                routers_state_dict[f'layer_{layer_idx}_alpha'] = routers.alpha_.state_dict()
                routers_state_dict[f'layer_{layer_idx}_beta'] = routers.beta_.state_dict()
            torch.save(routers_state_dict, router_path)
            print("Router Saved!")
