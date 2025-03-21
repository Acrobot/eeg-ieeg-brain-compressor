from pathlib import Path
from typing import Literal

from torch.utils.data import DataLoader, Dataset
from torcheeg import transforms
from torcheeg.datasets import MAHNOBDataset
from torcheeg.model_selection import train_test_split

from datasets import EEGBatch, EEGDataset

_SAMPLING_RATE = 512


class HciEegDataset(EEGDataset):
    def __init__(
        self,
        folder: str | Path,
        batch_size: int = 32,
        segment_size: int = 10000,
        stride: int | None = None,
        num_workers: int | None = 0,
        limit_train_batches: int | float | None = None,
        train_size: float = 0.75,
        validation_size: float = 0.15,
        seed: int = 42,
        io_path: str | None = None,
        predict_dataset: Literal["train", "val", "test"] = "test",
    ) -> None:
        super().__init__()
        self.folder = folder
        self.batch_size = batch_size
        self.segment_size = segment_size
        self.stride = stride if stride is not None else segment_size
        self.num_workers = num_workers
        self.limit_train_batches = limit_train_batches
        self.segment_samples = int(self.segment_size / 1000.0 * _SAMPLING_RATE)
        self.predict_dataset = predict_dataset

        test_size = 1 - train_size - validation_size
        self.dataset = MAHNOBDataset(
            root_path=folder,
            chunk_size=self.segment_samples,
            sampling_rate=_SAMPLING_RATE,
            online_transform=transforms.Compose([transforms.ToTensor()]),
            io_path=io_path,
        )
        self.dataset_train, test = train_test_split(
            dataset=self.dataset, test_size=1 - train_size, random_state=seed
        )
        self.dataset_val, self.dataset_test = train_test_split(
            dataset=test,
            test_size=test_size / (test_size + validation_size),
            random_state=seed,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            MahnobHciWrapper(self.dataset_train),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            MahnobHciWrapper(self.dataset_val),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            MahnobHciWrapper(self.dataset_test),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self) -> DataLoader:
        if self.predict_dataset == "train":
            return self.train_dataloader()
        elif self.predict_dataset == "val":
            return self.val_dataloader()
        elif self.predict_dataset == "test":
            return self.test_dataloader()
        else:
            raise NotImplementedError


class MahnobHciWrapper(Dataset[EEGBatch]):
    def __init__(self, dataset: MAHNOBDataset) -> None:
        self.dataset = dataset

    def __getitem__(self, index: int) -> EEGBatch:
        signal, labels = self.dataset[index]
        sample_id = int(labels["clip_id"].split("_")[-1])
        patient = labels["subject_id"]
        emotion = labels["feltEmo"]
        return EEGBatch(
            data=signal,
            id=index,
            sample_id=sample_id,
            patient=patient,
            dataset=0,
            emotion=emotion,
        )

    def __len__(self) -> int:
        return len(self.dataset)
