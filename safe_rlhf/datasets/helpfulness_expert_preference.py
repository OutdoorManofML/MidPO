from __future__ import annotations

from typing import Callable
from typing_extensions import TypedDict  # Python 3.10+

import torch

from safe_rlhf.datasets.base import CollatorBase, RawSample, TokenizedDataset
from safe_rlhf.datasets.utils import format_prompt, right_padding


__all__ = [
    'HelpfulnessExpertPreferenceDataset',
    'HelpfulnessExpertPreferenceCollator',
    'HelpfulnessExpertPreferenceSample',
    'HelpfulnessExpertPreferenceBatch',
]


class HelpfulnessExpertPreferenceSample(TypedDict, total=True):
    better_input_ids: torch.LongTensor  # size = (L,)
    worse_input_ids: torch.LongTensor  # size = (L,)


class HelpfulnessExpertPreferenceBatch(TypedDict, total=True):
    better_input_ids: torch.LongTensor  # size = (B, L)
    better_attention_mask: torch.BoolTensor  # size = (B, L)

    worse_input_ids: torch.LongTensor  # size = (B, L)
    worse_attention_mask: torch.BoolTensor  # size = (B, L)


class HelpfulnessExpertPreferenceDataset(TokenizedDataset):
    def preprocess(self, raw_sample: RawSample) -> PreferenceSample:
        prompt = format_prompt(input=raw_sample['input'], eos_token=self.tokenizer.eos_token)
        # Preference for training helpfulness expert
        better_answer = raw_sample['answer']
        worse_answer = raw_sample['other_answer']
        better = raw_sample['better']
        if not better:
            better_answer, worse_answer = worse_answer, better_answer

        better_input_ids = self.tokenize(prompt + better_answer + self.tokenizer.eos_token)
        worse_input_ids = self.tokenize(prompt + worse_answer + self.tokenizer.eos_token)
        if (
            better_input_ids.size() == worse_input_ids.size()
            and torch.all(torch.eq(better_input_ids, worse_input_ids)).item()
        ):
            raise ValueError(
                'Two responses get the same `input_ids` after tokenization.\n\n'
                f'Prompt: {prompt}\n\n'
                f'Better answer: {better_answer}\n\n'
                f'Worse answer: {worse_answer}',
            )
        return {
            'better_input_ids': better_input_ids,  # size = (L,)
            'worse_input_ids': worse_input_ids,  # size = (L,)
        }

    def get_collator(self) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
        return HelpfulnessExpertPreferenceCollator(self.tokenizer.pad_token_id)


class HelpfulnessExpertPreferenceCollator(CollatorBase):
    def __call__(self, samples: list[PreferenceSample]) -> PreferenceBatch:
        input_ids = [sample['better_input_ids'] for sample in samples] + [
            sample['worse_input_ids'] for sample in samples
        ]  # size = (2 * B, L)
        attention_mask = [
            input_id.new_ones(input_id.size(), dtype=torch.bool) for input_id in input_ids
        ]  # size = (2 * B, L)

        input_ids = right_padding(input_ids, padding_value=self.pad_token_id)  # size = (2 * B, L)
        attention_mask = right_padding(attention_mask, padding_value=0)  # size = (2 * B, L)

        (
            better_input_ids,  # size = (B, L)
            worse_input_ids,  # size = (B, L)
        ) = input_ids.chunk(chunks=2, dim=0)
        (
            better_attention_mask,  # size = (B, L)
            worse_attention_mask,  # size = (B, L)
        ) = attention_mask.chunk(chunks=2, dim=0)

        return {
            'better_input_ids': better_input_ids,  # size = (B, L)
            'better_attention_mask': better_attention_mask,  # size = (B, L)
            'worse_input_ids': worse_input_ids,  # size = (B, L)
            'worse_attention_mask': worse_attention_mask,  # size = (B, L)
        }
