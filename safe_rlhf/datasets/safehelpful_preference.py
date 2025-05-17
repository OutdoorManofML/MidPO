# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations

from typing import Any, Callable, ClassVar, Collection, Dict, Iterable, Iterator
from typing_extensions import TypedDict  # Python 3.10+
from fractions import Fraction
import transformers
from tqdm import tqdm

import torch

from safe_rlhf.datasets.base import CollatorBase, RawSample, TokenizedDataset, RawDataset
from safe_rlhf.datasets.utils import format_prompt, right_padding
from safe_rlhf.utils import is_main_process
from torch.utils.data import Dataset


__all__ = [
    'SafeHelpfulPreferenceDataset',
    'SafeHelpfulPreferenceCollator',
    'SafeHelpfulPreferenceSample',
    'SafeHelpfulPreferenceBatch',
]


class SafeHelpfulPreferenceSample(TypedDict, total=True):
    better_input_ids: torch.LongTensor  # size = (L,)
    # False (0) for safe / True (+1) for unsafe
    better_is_unsafe: torch.BoolTensor  # size = ()

    worse_input_ids: torch.LongTensor  # size = (L,)
    # False (0) for safe / True (+1) for unsafe
    worse_is_unsafe: torch.BoolTensor  # size = ()


class SafeHelpfulPreferenceBatch(TypedDict, total=True):
    better_input_ids: torch.LongTensor  # size = (B, L)
    better_attention_mask: torch.BoolTensor  # size = (B, L)
    # False (0) for safe / True (+1) for unsafe
    better_is_unsafe: torch.BoolTensor  # size = (B,)

    worse_input_ids: torch.LongTensor  # size = (B, L)
    worse_attention_mask: torch.BoolTensor  # size = (B, L)
    # False (0) for safe / True (+1) for unsafe
    worse_is_unsafe: torch.BoolTensor  # size = (B,)


class SafeHelpfulPreferenceDataset(TokenizedDataset):
    def __init__(  # pylint: disable=too-many-branches
        self,
        dataset_names_and_attributes: (
            dict[str, float | dict[str, Any]] | Iterable[tuple[str, float | dict[str, Any]]]
        ),
        tokenizer: transformers.PreTrainedTokenizerBase,
        lazy_tokenization: bool = True,
        seed: int = 42,
    ) -> None:
        if not isinstance(dataset_names_and_attributes, dict):
            dataset_names_and_attributes = tuple(dataset_names_and_attributes)
            dataset_names = [name for name, _ in dataset_names_and_attributes]
            if len(dataset_names) != len(set(dataset_names)):
                raise ValueError(
                    f'Dataset names should be unique, but got {dataset_names}.',
                )

        Dataset.__init__(self)
        self.dataset_names_and_proportion: dict[str, float | Fraction] = {}
        self.raw_datasets = []
        for name, attributes in dict(dataset_names_and_attributes).items():
            if isinstance(attributes, float):
                kwargs = {'proportion': attributes}
            elif isinstance(attributes, dict):
                kwargs = dict(attributes)  # copy
            else:
                raise TypeError(
                    f'Dataset `{name}` attributes should be a float or a dict, '
                    f'got {type(attributes).__name__}.',
                )
            proportion = kwargs.pop('proportion', 1.0)
            if isinstance(proportion, Fraction):
                if not (proportion < 0 and proportion.denominator == 1):
                    raise ValueError(
                        f'Dataset `{name}` proportion should be a negative integer '
                        f'represents `num_samples / -1`, got {proportion}.',
                    )
            else:
                proportion = float(proportion)
                if proportion < 0.0:
                    raise ValueError(
                        f'Dataset `{name}` proportion should be no less than 0.0, '
                        f'got {proportion}.',
                    )
            if proportion == 0.0:
                continue
            raw_dataset = RawDataset.load(name, **kwargs)
            self.dataset_names_and_proportion[raw_dataset.NAME] = proportion
            self.raw_datasets.append(raw_dataset)

        self.tokenizer = tokenizer
        self.seed = seed

        merged_rawdata = self._merge_raw_datasets(seed=seed)
        rawdata = [merged_rawdata[i] for i in range(len(merged_rawdata))]
        self.rawdata=[]
        for i in range(len(rawdata)):
            if rawdata[i]['is_safe'] + rawdata[i]['is_other_safe'] > 0:
                self.rawdata.append(rawdata[i])
        if lazy_tokenization:
            self.data = [self._SENTINEL for _ in range(len(self.rawdata))]
        else:
            data = list(
                map(
                    self.preprocess,
                    tqdm(
                        self.rawdata,
                        desc='Preprocessing raw dataset...',
                        disable=not is_main_process(),
                    ),
                ),
            )
            self.data = [d for d in data]

    def preprocess(self, raw_sample: RawSample) -> SafeHelpfulPreferenceSample:
        prompt = format_prompt(input=raw_sample['input'], eos_token=self.tokenizer.eos_token)
        better_answer = raw_sample['answer']
        worse_answer = raw_sample['other_answer']
        better_is_unsafe = not raw_sample['is_safe']
        worse_is_unsafe = not raw_sample['is_other_safe']
        better = raw_sample['better']
        if not better:
            better_answer, worse_answer = worse_answer, better_answer
            better_is_unsafe, worse_is_unsafe = worse_is_unsafe, better_is_unsafe
        if (better_is_unsafe and worse_is_unsafe):
            print(raw_sample)
            raise ValueError("SafetyPreferenceSample has error")

        # size = (L,)
        # better_input_ids = self.tokenize(prompt + better_answer + self.tokenizer.eos_token)
        # worse_input_ids = self.tokenize(prompt + worse_answer + self.tokenizer.eos_token)
        better_input_ids = self.tokenize(prompt + better_answer + ' ' + self.tokenizer.eos_token)
        worse_input_ids = self.tokenize(prompt + worse_answer + ' ' + self.tokenizer.eos_token)

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
            'better_is_unsafe': torch.tensor(better_is_unsafe),
            'worse_input_ids': worse_input_ids,  # size = (L,)
            'worse_is_unsafe': torch.tensor(worse_is_unsafe),
        }

    def get_collator(self) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
        return SafeHelpfulPreferenceCollator(self.tokenizer.pad_token_id)


class SafeHelpfulPreferenceCollator(CollatorBase):
    def __call__(self, samples: list[SafeHelpfulPreferenceSample]) -> SafeHelpfulPreferenceBatch:
        input_ids = [sample['better_input_ids'] for sample in samples] + [
            sample['worse_input_ids'] for sample in samples
        ]  # size = (2 * B, L)
        attention_mask = [
            input_id.new_ones(input_id.size(), dtype=torch.bool) for input_id in input_ids
        ]  # size = (2 * B, L)
        is_unsafe = [sample['better_is_unsafe'] for sample in samples] + [
            sample['worse_is_unsafe'] for sample in samples
        ]

        # size = (2 * B, L)
        input_ids = right_padding(input_ids, padding_value=self.pad_token_id)
        attention_mask = right_padding(attention_mask, padding_value=0)
        # size = (2 * B,)
        is_unsafe = torch.tensor(is_unsafe)

        # size = (B, L)
        better_input_ids, worse_input_ids = input_ids.chunk(chunks=2, dim=0)
        better_attention_mask, worse_attention_mask = attention_mask.chunk(chunks=2, dim=0)
        # size = (B,)
        better_is_unsafe, worse_is_unsafe = is_unsafe.chunk(chunks=2, dim=0)
        return {
            'better_input_ids': better_input_ids,  # size = (B, L)
            'better_attention_mask': better_attention_mask,  # size = (B, L)
            'better_is_unsafe': better_is_unsafe,  # size = (B,)
            'worse_input_ids': worse_input_ids,  # size = (B, L)
            'worse_attention_mask': worse_attention_mask,  # size = (B, L)
            'worse_is_unsafe': worse_is_unsafe,  # size = (B,)
        }
