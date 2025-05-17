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
from safe_rlhf.configs import IGNORE_INDEX, PROMPT_ASSISTANT, PROMPT_BEGIN, PROMPT_USER
from safe_rlhf.datasets.base import CollatorBase, RawSample, TokenizedDataset
from safe_rlhf.datasets.utils import format_prompt, right_padding
from safe_rlhf.datasets.supervised import SupervisedCollator, SupervisedSample


__all__ = [
    'SupervisedPairDataset',
]


class SupervisedPairDataset(TokenizedDataset):
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
        self.rawdata = []
        for i in range(len(merged_rawdata)):
            self.rawdata.append({"input": merged_rawdata[i]["input"], "answer": merged_rawdata[i]["answer"]})
            self.rawdata.append({"input": merged_rawdata[i]["input"], "answer": merged_rawdata[i]["other_answer"]})
        self.rawdata = [dict(t) for t in {tuple(d.items()) for d in self.rawdata}]  # remove duplications
        
        if lazy_tokenization:
            self.data = [self._SENTINEL for _ in range(len(self.rawdata))]
        else:
            self.data = list(
                map(
                    self.preprocess,
                    tqdm(
                        self.rawdata,
                        desc='Preprocessing raw dataset...',
                        disable=not is_main_process(),
                    ),
                ),
            )
            
    def preprocess(self, raw_sample: RawSample) -> SupervisedSample:
        if raw_sample.get('input') is None and raw_sample.get('dialogue') is None:
            raise ValueError('Either `input` or `dialogue` must be provided.')
        if raw_sample.get('input') is not None and raw_sample.get('dialogue') is not None:
            raise ValueError('At most one of `input` and `dialogue` can be provided.')

        if raw_sample.get('input') is not None:
            input = raw_sample['input']  # pylint: disable=redefined-builtin
            if not isinstance(input, str):
                raise ValueError(f'Unsupported type of `input`: {type(input)}. Expected: str.')
            prompt = format_prompt(input=input, eos_token=self.tokenizer.eos_token)
            answer = raw_sample['answer']
            text = prompt + answer + ' ' + self.tokenizer.eos_token

            input_ids = self.tokenize(text)
            labels = input_ids.clone()
            # Mask non-assistant input
            labels[: len(self.tokenize(prompt))] = IGNORE_INDEX
            return {'input_ids': input_ids, 'labels': labels}

        dialogue = raw_sample['dialogue']  # is not None
        text = PROMPT_BEGIN
        offsets = [0]
        input_ids = torch.empty(0, dtype=torch.long)
        for i, line in enumerate(dialogue):
            if i % 2 == 0:
                # User input
                text += PROMPT_USER.format(input=line) + PROMPT_ASSISTANT
            else:
                # Assistant input
                text += line + self.tokenizer.eos_token
            input_ids = self.tokenize(text)
            offsets.append(len(input_ids))

        labels = input_ids.clone()
        # Mask non-assistant input
        for begin, end in zip(offsets[::2], offsets[1::2]):
            labels[begin:end] = IGNORE_INDEX

        return {
            'input_ids': input_ids,  # size = (L,)
            'labels': labels,  # size = (L,)
        }

    def get_collator(self) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
        return SupervisedCollator(self.tokenizer.pad_token_id)
