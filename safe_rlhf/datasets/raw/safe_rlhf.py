"""Safe-RLHF preference datasets."""

from __future__ import annotations

from typing import ClassVar
from datasets import load_dataset, concatenate_datasets
from safe_rlhf.datasets.base import RawDataset, RawSample


__all__ = [
    'SafeRLHFDataset',
    'SafeRLHFTrainDataset',
    'SafeRLHFTestDataset',
    'SafeRLHF30KTrainDataset',
    'SafeRLHF30KTestDataset',
    'SafeRLHF10KTrainDataset',
]


class SafeRLHFDataset(RawDataset):
    SPLIT: ClassVar[str]
    PATH: ClassVar[str]

    def __init__(self, path: str | None = None) -> None:
        # Only for training safety expert
        train_dataset = load_dataset(path or self.PATH, split=self.SPLIT)
        safe_data = train_dataset.filter(lambda d: d['is_response_0_safe'] ^ d['is_response_1_safe'])
        self.data = safe_data
        print("Training on the safety preference dataset")

        # # Only for training the helpfulness expert
        # self.data = load_dataset(path or self.PATH, split=self.SPLIT)
        # print("Training on the helpful preference dataset")

        # Only for training the router
        self.data = load_dataset(path or self.PATH, split=self.SPLIT)
        self.data = self.data.filter(
            lambda d: (d['is_response_0_safe'] and d['is_response_1_safe'])
                      and d['safer_response_id'] == d['better_response_id']
        )
        print("Training on the dual preference dataset")

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        return RawSample(
            input=data['prompt'],
            answer=data['response_0'],
            other_answer=data['response_1'],
            better=int(data['better_response_id']) == 0,
            safer=int(data['safer_response_id']) == 0,
            is_safe=bool(data['is_response_0_safe']),
            is_other_safe=bool(data['is_response_1_safe']),
        )

    def __len__(self) -> int:
        return len(self.data)


class SafeRLHFTrainDataset(SafeRLHFDataset):
    NAME: str = 'PKU-SafeRLHF/train'
    ALIASES: tuple[str, ...] = ('PKU-Alignment/PKU-SafeRLHF/train',)
    PATH: str = 'PKU-Alignment/PKU-SafeRLHF'
    SPLIT: str = 'train'


class SafeRLHFTestDataset(SafeRLHFDataset):
    NAME: str = 'PKU-SafeRLHF/test'
    ALIASES: tuple[str, ...] = ('PKU-Alignment/PKU-SafeRLHF/test',)
    PATH: str = 'PKU-Alignment/PKU-SafeRLHF'
    SPLIT: str = 'test'


class SafeRLHF30KTrainDataset(SafeRLHFDataset):
    NAME: str = 'PKU-SafeRLHF-30K/train'
    ALIASES: tuple[str, ...] = ('PKU-Alignment/PKU-SafeRLHF-30K/train',)
    PATH: str = 'PKU-Alignment/PKU-SafeRLHF-30K'
    SPLIT: str = 'train'


class SafeRLHF30KTestDataset(SafeRLHFDataset):
    NAME: str = 'PKU-SafeRLHF-30K/test'
    ALIASES: tuple[str, ...] = ('PKU-Alignment/PKU-SafeRLHF-30K/test',)
    PATH: str = 'PKU-Alignment/PKU-SafeRLHF-30K'
    SPLIT: str = 'test'


class SafeRLHF10KTrainDataset(SafeRLHFDataset):
    NAME: str = 'PKU-SafeRLHF-10K/train'
    ALIASES: tuple[str, ...] = ('PKU-Alignment/PKU-SafeRLHF-10K/train',)
    PATH: str = 'PKU-Alignment/PKU-SafeRLHF-10K'
    SPLIT: str = 'train'
