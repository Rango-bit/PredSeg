import os
import re
from logging import getLogger
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from predseg.segmentation.transforms import get_transform


logger = getLogger()


_CXR_DATASET_NAME = "CXR_Masks_and_Labels"


def _select_cxr_bilateral_lung_masks(df):
    """Keep the one bilateral-lung target for each CXR image."""
    expected_mask_names = df["file_name"].map(
        lambda name: f"{Path(str(name)).stem}_lung{Path(str(name)).suffix}"
    )
    return df.loc[df["mask_file"].astype(str) == expected_mask_names].copy()


def dataset_folder_name(dataset_name):
    """Return a filesystem-safe folder name while preserving readability."""
    folder_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(dataset_name))
    folder_name = folder_name.strip().strip(".")
    if not folder_name:
        raise ValueError(f"dataset_name={dataset_name!r} cannot be converted to a folder")
    return folder_name


def discover_dataset_names(
    csv_dir,
    train_csv="train.csv",
    include=None,
    exclude=None,
):
    eligible, _ = discover_dataset_statistics(
        csv_dir=csv_dir,
        train_csv=train_csv,
        include=include,
        exclude=exclude,
        min_train_images=1,
    )
    return list(eligible)


def split_dataset_counts(image_counts, min_train_images):
    if min_train_images < 1:
        raise ValueError("min_train_images must be at least 1")
    eligible = {
        name: int(count)
        for name, count in sorted(image_counts.items())
        if count >= min_train_images
    }
    skipped = {
        name: int(count) for name, count in sorted(image_counts.items()) if count < min_train_images
    }
    return eligible, skipped


def discover_dataset_statistics(
    csv_dir,
    train_csv="train.csv",
    include=None,
    exclude=None,
    min_train_images=100,
):
    """Count unique training images and split eligible/skipped datasets."""
    csv_file = os.path.join(csv_dir, train_csv)
    required_columns = ["dataset_name", "subfolder", "file_name"]
    df = pd.read_csv(csv_file, usecols=required_columns)
    df = df.dropna(subset=["dataset_name", "file_name"]).copy()
    df["dataset_name"] = df["dataset_name"].astype(str).str.strip()
    df["subfolder"] = df["subfolder"].fillna("-").astype(str).str.strip()
    df["file_name"] = df["file_name"].astype(str).str.strip()
    df = df.loc[(df["dataset_name"] != "") & (df["file_name"] != "")]
    if df.empty:
        raise ValueError(f"No dataset images found in {csv_file}")

    image_counts = (
        df.drop_duplicates(subset=["dataset_name", "subfolder", "file_name"])
        .groupby("dataset_name")
        .size()
        .sort_index()
        .to_dict()
    )
    names = sorted(image_counts)
    if not names:
        raise ValueError(f"No dataset names found in {csv_file}")

    if include:
        include_names = {str(name).strip() for name in include}
        missing = sorted(include_names.difference(names))
        if missing:
            raise ValueError(f"Configured datasets are absent from {csv_file}: {missing}")
        names = [name for name in names if name in include_names]

    if exclude:
        exclude_names = {str(name).strip() for name in exclude}
        names = [name for name in names if name not in exclude_names]

    if not names:
        raise ValueError("Dataset selection is empty after include/exclude")

    selected_counts = {name: image_counts[name] for name in names}
    return split_dataset_counts(selected_counts, min_train_images)


class DriveDataset(Dataset):
    """Load one configured dataset from the shared segmentation CSV files."""

    REQUIRED_COLUMNS = {
        "dataset_name",
        "subfolder",
        "file_name",
        "mask_file",
    }

    def __init__(
        self,
        root_dir,
        csv_dir,
        dataset_name,
        transform=None,
        train=False,
        val=False,
        train_csv="train.csv",
        test_csv="test.csv",
    ):
        super().__init__()
        if not dataset_name or not str(dataset_name).strip():
            raise ValueError("dataset_name must be set to one dataset in config")

        self.selected_dataset = str(dataset_name).strip()
        csv_name = train_csv if train else test_csv
        csv_file = os.path.join(csv_dir, csv_name)
        self.img_folder = "train" if train else "test"
        self.mask_folder = "train_mask" if train else "test_mask"

        df = pd.read_csv(csv_file)
        missing_columns = self.REQUIRED_COLUMNS.difference(df.columns)
        if missing_columns:
            raise ValueError(f"{csv_file} is missing columns: {sorted(missing_columns)}")

        normalized_names = df["dataset_name"].astype(str).str.strip()
        available_datasets = sorted(normalized_names.unique().tolist())
        df = df.loc[normalized_names == self.selected_dataset].reset_index(drop=True)
        if self.selected_dataset == _CXR_DATASET_NAME:
            original_count = len(df)
            df = _select_cxr_bilateral_lung_masks(df).reset_index(drop=True)
            logger.info(
                "Filtered CXR masks to bilateral lung targets: %d -> %d samples",
                original_count,
                len(df),
            )
        if df.empty:
            raise ValueError(
                f"No samples for dataset_name={self.selected_dataset!r} in "
                f"{csv_file}. Available datasets: {available_datasets}"
            )

        self.dataset_name = df["dataset_name"].astype(str).str.strip().tolist()
        self.subfolder = df["subfolder"].fillna("-").astype(str).tolist()
        self.img_name = df["file_name"].astype(str).tolist()
        self.mask_name = df["mask_file"].astype(str).tolist()
        self.transform = transform
        self.root_dir = root_dir
        self.train = train
        self.val = val

        logger.info(
            "Loaded %d %s samples for dataset %s from %s",
            len(self.dataset_name),
            "train" if train else "test",
            self.selected_dataset,
            csv_file,
        )

    def __len__(self):
        return len(self.dataset_name)

    def __getitem__(self, idx):
        dataset_name = self.dataset_name[idx]
        subfolder = self.subfolder[idx]
        img_filename = self.img_name[idx]
        mask_filename = self.mask_name[idx]

        if subfolder == "-":
            sub_path = dataset_name
        else:
            sub_path = os.path.join(dataset_name, subfolder)

        img_path = os.path.join(
            self.root_dir,
            sub_path,
            self.img_folder,
            img_filename,
        )
        mask_path = os.path.join(
            self.root_dir,
            sub_path,
            self.mask_folder,
            mask_filename,
        )

        with Image.open(img_path) as img_file:
            image = img_file.convert("RGB")
        with Image.open(mask_path) as mask_file:
            mask = mask_file.convert("L")

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return {
            "pixel_values": image,
            "label": mask,
            "dataset_name": dataset_name,
            "subfolder": subfolder,
            "image_name": img_filename,
            "mask_name": mask_filename,
        }


def make_dataset(
    root_dir,
    csv_dir,
    dataset_name,
    batch_size,
    train=True,
    input_size=224,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    persistent_workers=False,
    prefetch_factor=None,
    drop_last=True,
    train_csv="train.csv",
    test_csv="test.csv",
):
    dataset = DriveDataset(
        root_dir=root_dir,
        csv_dir=csv_dir,
        dataset_name=dataset_name,
        transform=get_transform(train=train, crop_size=input_size),
        train=train,
        train_csv=train_csv,
        test_csv=test_csv,
    )

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=train,
    )
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_mem,
        "drop_last": drop_last,
        "sampler": dist_sampler,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor

    data_loader = torch.utils.data.DataLoader(**loader_kwargs)
    logger.info(
        "Data loader created for %s (samples=%d, workers=%d, prefetch_factor=%s, pin_memory=%s)",
        dataset_name,
        len(dataset),
        num_workers,
        prefetch_factor if num_workers > 0 else None,
        pin_mem,
    )
    return data_loader, dist_sampler
