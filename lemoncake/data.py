from pathlib import Path

# from collections import Counter

import pandas as pd
import numpy as np

import torch


class MIMICFeaturesDataset:
    """Class to hold the MIMIC IV pretrained features dataset with required preprocessing methods."""

    def __init__(self, data_path: str = "../dataset/mimiciv/mit_pretrained"):
        """
        Args:
            data_path: Location of data until "cxr_ic_fusion_1103.csv"
        """
 
        self.data_dir = Path(data_path)


    def reader(self, partition: str):
        """Read data and return features and labels"""
        data = []
        location = Path(f"{self.data_dir}/{partition}")
        files = [x for x in location.glob("**/*") if x.is_file()]
        for file_name in files:
            with open(file_name) as file:
                data.append(
                    pd.read_csv(
                        file, index_col=None, usecols=["sequence", "family_accession"]
                    )
                )
        all_data = pd.concat(data)

        return all_data["sequence"], all_data["family_accession"]

    def build_labels(self, targets: pd.DataFrame) -> dict:
        """Build dictionary of unique protein families with indexes"""
        unique_targets = targets.unique()
        fam2label = {target: i for i, target in enumerate(unique_targets, start=1)}
        fam2label["<unk>"] = 0

        return fam2label

    def build_vocab(self, data: pd.DataFrame) -> dict:
        """Build the vocabulary of amino acids"""
        voc = set()
        for sequence in data:
            voc.update(sequence)
        unique_AAs = sorted(voc - self.rare_AAs)

        # Build the mapping
        word2id = {w: i for i, w in enumerate(unique_AAs, start=2)}
        word2id["<pad>"] = 0
        word2id["<unk>"] = 1

        return word2id

    def _get_vocab_classes(self):
        """Returns VOCAB and CLASSES learnt from train data"""
        train_data, train_targets = self.reader("train")
        return self.build_vocab(train_data), self.build_labels(train_targets)


class SequenceDataset(torch.utils.data.Dataset):
    """Custom Dataset for amino acid sequence data."""

    def __init__(self, pfam_dataset: PfamDataset, max_len: int, split: str, lstm: bool):
        self.word2id = pfam_dataset.vocab
        self.fam2label = pfam_dataset.classes
        self.max_len = max_len
        self.lstm = lstm

        self.data, self.label = pfam_dataset.reader(split)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        seq = self.preprocess(self.data.iloc[index])
        label = self.fam2label.get(self.label.iloc[index], self.fam2label["<unk>"])

        return {"sequence": seq, "target": label}

    def preprocess(self, text):
        """Return one-hot encoded matrix for a given sequence if CNN, else just sequence tensor for LSTM"""
        seq = []

        # Encode into IDs
        for word in text[: self.max_len]:
            seq.append(self.word2id.get(word, self.word2id["<unk>"]))

        # Pad to maximal length
        if len(seq) < self.max_len:
            seq += [self.word2id["<pad>"] for _ in range(self.max_len - len(seq))]

        # Convert list into tensor
        seq = torch.from_numpy(np.array(seq))

        if self.lstm:
            return seq
        else:
            # One-hot encode
            one_hot_seq = torch.nn.functional.one_hot(
                seq,
                num_classes=len(self.word2id),
            )
            # Permute channel (one-hot) dim first
            one_hot_seq = one_hot_seq.permute(1, 0)
            return one_hot_seq


def get_datasets(
    seq_max_len=120,
    data_path: str = None,
    lstm=False,
    partitions: list = ["train", "dev", "test"],
) -> tuple[list, int, int, int]:
    """
    Convenience function to read and preprocess the raw Pfam dataset

    Args:
        seq_max_len: max len of the amino acid sequences
        data_path: Location of downloaded data until parent of the "random_split" folder,
                   if nothing is passed in, it will be assumed to be in "/home/<user>/.protfam/dataset"
        lstm: True if using LSTM else False
        partitions: if different from the defaults
    Returns:
        datasets: a list of 3 "torch.utils.data.Dataset" one each for train, dev and test
        num_classes: number of classes for training and prediction
        vocab_len: length of the vocab (unique amino acids in train)
        train_data_len: length of the train dataset (# of samples in train)
    """
    pfam_dataset = PfamDataset(data_path)
    datasets = []
    for partition in partitions:
        datasets.append(
            SequenceDataset(pfam_dataset, seq_max_len, split=partition, lstm=lstm)
        )

    return (
        datasets,
        len(pfam_dataset.classes),
        len(pfam_dataset.vocab),
        len(datasets[0]),
    )


def get_dataloaders(
    datasets: list, batch_size: int = 512, num_workers: int = None
) -> dict:
    """
    Returns a dict of train, dev and test dataloaders

    Args:
        datasets: list with train, dev and test datasets
        batch_size: The value for train dl, doubled for dev and test dls
        num_workers: If None, then assigned value of cpu_count()
    """

    if num_workers is None:
        num_workers = torch.multiprocessing.cpu_count()

    dataloaders = {}
    train_dataset, dev_dataset, test_dataset = datasets

    dataloaders["train"] = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    dataloaders["dev"] = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
    )
    dataloaders["test"] = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
    )

    return dataloaders
