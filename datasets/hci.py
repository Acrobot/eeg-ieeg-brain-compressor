from pathlib import Path

from torch.utils.data import DataLoader, Dataset
from torcheeg import transforms
from torcheeg.datasets import MAHNOBDataset
from torcheeg.model_selection import train_test_split

from datasets import EEGBatch, EEGDataset

_SAMPLING_RATE = 256


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
    ) -> None:
        super().__init__()
        self.folder = folder
        self.batch_size = batch_size
        self.segment_size = segment_size
        self.stride = stride if stride is not None else segment_size
        self.num_workers = num_workers
        self.limit_train_batches = limit_train_batches
        self.segment_samples = int(self.segment_size / 1000.0 * _SAMPLING_RATE)

        test_size = 1 - train_size - validation_size
        self.dataset = MAHNOBDataset(
            root_path=folder,
            chunk_size=self.segment_samples,
            sampling_rate=_SAMPLING_RATE,
            online_transform=transforms.Compose([transforms.ToTensor()]),
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
        return self.test_dataloader()


class MahnobHciWrapper(Dataset[EEGBatch]):
    def __init__(self, dataset: MAHNOBDataset) -> None:
        self.dataset = dataset

    def __getitem__(self, index: int) -> EEGBatch:
        signal, labels = self.dataset[index]
        sample_id = int(labels["clip_id"].split("_")[-1])
        patient = labels["subject_id"]
        return EEGBatch(
            data=signal.T, id=index, sample_id=sample_id, patient=patient, dataset=0
        )

    def __len__(self) -> int:
        return len(self.dataset)
