from pathlib import Path

import torch
from pytorch_lightning.callbacks import BasePredictionWriter


class EpochWriter(BasePredictionWriter):
    def __init__(self, output_dir: str) -> None:
        super().__init__("epoch")
        self.output_dir = Path(output_dir)

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        torch.save(predictions, self.output_dir / "predictions.pt")
