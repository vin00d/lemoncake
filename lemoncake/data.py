from pathlib import Path

# from collections import Counter

import pandas as pd
import numpy as np
from typing import Optional

import torch.utils.data

DATA_PATH = "../dataset/mimiciv/mit_pretrained/splits"


class MIMICPretrainedDataset(torch.utils.data.Dataset):
    """Custom Dataset for MIMIC IV pretrained features data."""

    def __init__(self, split: str, data_path: str = DATA_PATH):
        self.x = pd.read_csv(Path(data_path) / f"x_{split}.csv")
        self.y = pd.read_csv(Path(data_path) / f"y_{split}.csv")
        assert len(self.x) == len(self.y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return {
            "x": torch.FloatTensor(self.x.loc[index].values),
            "y": torch.FloatTensor(self.y.loc[index].values),
        }


def get_datasets(
    data_path: Optional[str] = None,
    splits: Optional[list] = None,
) -> list:
    """
    Convenience function to read and preprocess the raw Pfam dataset

    Args:
        data_path: Location of downloaded data, defaults to "../dataset/mimiciv/mit_pretrained/splits"
        splits: one or many of "train", "val" and "test" to be read and returned, returns all 3 by default
    Returns:
        datasets: a list of up to 3 "torch.utils.data.Dataset" one each for train, val and test
    """
    datasets = []
    if splits is None:
        splits = ["train", "val", "test"]
    if data_path is None:
        data_path = DATA_PATH

    for split in splits:
        datasets.append(MIMICPretrainedDataset(split=split, data_path=data_path))

    return datasets


def get_dataloaders(
    datasets: dict, batch_size: int = 512, num_workers: Optional[int] = None
) -> dict:
    """
    Returns a dict of train, dev and test dataloaders

    Args:
        datasets: dict with one or many of train, val and test datasets
        batch_size: The value for train dl, doubled for val and test dls
        num_workers: If None, then assigned value of cpu_count()
    """

    if num_workers is None:
        num_workers = torch.multiprocessing.cpu_count()

    dataloaders = {}
    for split in datasets.keys():
        dataloaders[split] = torch.utils.data.DataLoader(
            datasets[split],
            batch_size=batch_size if split == "train" else batch_size * 2,
            shuffle=True if split == "train" else False,
            num_workers=num_workers,
        )

    return dataloaders
