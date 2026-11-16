"""CSV-backed images and optional cached modality text features."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class MedImgDataset(Dataset):
    def __init__(
        self,
        csv_path,
        image_root=None,
        text_features_path=None,
        modality_column="modality_id",
        transform=None,
        require_text=True,
    ):
        self.df = pd.read_csv(csv_path)
        if "img_path" not in self.df:
            raise ValueError(f"{csv_path} must contain an img_path column")
        if self.df.empty or self.df["img_path"].isna().any():
            raise ValueError("The pretraining CSV must contain nonempty image paths")
        self.image_root = Path(image_root) if image_root else Path(csv_path).resolve().parent
        self.transform = transform
        if require_text:
            if modality_column not in self.df:
                raise ValueError(f"{csv_path} is missing column {modality_column!r}")
            numeric = pd.to_numeric(self.df[modality_column], errors="raise")
            if numeric.isna().any() or not np.equal(numeric, np.floor(numeric)).all():
                raise ValueError("Modality IDs must be integers")
            self.modality_ids = numeric.to_numpy(dtype=np.int64)
            if not text_features_path:
                raise ValueError("Conditional regularization requires data.text_features_path")
            cache = torch.load(text_features_path, map_location="cpu", weights_only=True)
            if not isinstance(cache, torch.Tensor) or cache.ndim not in (2, 3):
                raise ValueError("Text cache must be a tensor with shape [M, D] or [M, T, D]")
            if not cache.numel() or not torch.isfinite(cache).all():
                raise ValueError("Text cache must be nonempty and finite")
            self.text_features_cache = (
                cache.float().unsqueeze(1) if cache.ndim == 2 else cache.float()
            )
            if self.modality_ids.min() < 0 or self.modality_ids.max() >= len(cache):
                raise ValueError("Modality IDs must index rows of the text feature cache")
        else:
            self.modality_ids = np.zeros(len(self.df), dtype=np.int64)
            self.text_features_cache = torch.zeros(1, 1, 1)
        self.modality_counts = np.bincount(self.modality_ids).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        path = Path(str(self.df.iloc[index]["img_path"]))
        if not path.is_absolute():
            path = self.image_root / path
        with Image.open(path) as image_file:
            image = image_file.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        modality = int(self.modality_ids[index])
        return image, str(path), modality, self.text_features_cache[modality]


def make_dataset(
    transform,
    batch_size,
    csv_path,
    image_root=None,
    text_features_path=None,
    modality_column="modality_id",
    require_text=True,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    drop_last=True,
    persistent_workers=False,
):
    dataset = MedImgDataset(
        csv_path,
        image_root,
        text_features_path,
        modality_column,
        transform,
        require_text,
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collator,
        pin_memory=pin_mem,
        num_workers=num_workers,
        drop_last=drop_last,
        persistent_workers=persistent_workers and num_workers > 0,
    )
    if len(loader) == 0:
        raise ValueError("No pretraining batches: reduce batch_size or add more images")
    return dataset, loader, sampler
