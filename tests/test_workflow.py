"""Exercise the public CLI from synthetic pretraining through segmentation evaluation."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

from predseg.config import load_config
from predseg.segmentation.checkpoints import validate_segmentation_checkpoint


PROJECT = Path(__file__).resolve().parents[1]


def run_cli(*arguments):
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONIOENCODING="utf-8")
    environment["PYTHONPATH"] = str(PROJECT / "src")
    result = subprocess.run(
        [sys.executable, "-m", "predseg", *map(str, arguments)],
        cwd=PROJECT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_cli_pretrain_segment_evaluate_and_resume(tmp_path):
    data_root = tmp_path / "images"
    dataset_root = data_root / "Synthetic"
    for folder in ["train", "train_mask", "test", "test_mask"]:
        (dataset_root / folder).mkdir(parents=True)
    generator = np.random.default_rng(7)
    pretraining_rows = []
    for split in ["train", "test"]:
        rows = []
        for index in range(2):
            image = generator.integers(0, 256, (56, 56, 3), dtype=np.uint8)
            mask = np.zeros((56, 56), dtype=np.uint8)
            mask[12:40, 16:38] = 255
            name = f"image_{index}.png"
            Image.fromarray(image).save(dataset_root / split / name)
            Image.fromarray(mask).save(dataset_root / f"{split}_mask" / name)
            rows.append(
                {
                    "dataset_name": "Synthetic",
                    "subfolder": "-",
                    "file_name": name,
                    "mask_file": name,
                }
            )
            if split == "train":
                pretraining_rows.append({"img_path": f"Synthetic/train/{name}", "modality_id": 0})
        pd.DataFrame(rows).to_csv(tmp_path / f"{split}.csv", index=False)
    pd.DataFrame(pretraining_rows).to_csv(tmp_path / "pretrain.csv", index=False)
    torch.save(torch.randn(1, 3, 8), tmp_path / "text.pt")

    pretraining = load_config(PROJECT / "configs/pretraining/vit_tiny.yaml")
    pretraining["data"].update(
        csv_path=str(tmp_path / "pretrain.csv"),
        image_root=str(data_root),
        text_features_path=str(tmp_path / "text.pt"),
        batch_size=2,
        crop_size=56,
        num_workers=0,
        pin_mem=False,
    )
    pretraining["mask"].update(min_keep=1, num_pred_masks=1, pred_mask_scale=[0.25, 0.25])
    pretraining["meta"].update(pred_depth=12, pred_emb_dim=48, use_bfloat16=False)
    pretraining["optimization"].update(epochs=1, warmup=0)
    pretraining["train"].update(
        regularization_method="visreg",
        lambda_progressive=False,
        regularization_min_group_size=2,
        visreg_num_projections=4,
    )
    pretraining["logging"].update(folder=str(tmp_path / "pretraining"), checkpoint_freq=0)
    pre_config = tmp_path / "pretrain.yaml"
    pre_config.write_text(yaml.safe_dump(pretraining), encoding="utf-8")
    run_cli("pretrain", "--config", pre_config, "--devices", "cpu")
    weights = tmp_path / "pretraining/jepa-latest.pth.tar"
    pretrained = torch.load(weights, weights_only=True)
    assert pretrained["epoch"] == 1 and pretrained["backbone"]["pred_depth"] == 12
    assert pretrained["loss_context"] > 0
    assert np.isfinite(pretrained["loss_total"])
    assert any(
        not torch.equal(pretrained["encoder"][key], value)
        for key, value in pretrained["target_encoder"].items()
    )
    del pretrained

    run_cli(
        "pretrain",
        "--config",
        pre_config,
        "--devices",
        "cpu",
        "--set",
        "meta.load_checkpoint=true",
        "--set",
        "optimization.epochs=2",
    )
    resumed_pretraining = torch.load(weights, weights_only=True)
    assert resumed_pretraining["epoch"] == 2
    assert np.isfinite(resumed_pretraining["loss_total"])
    del resumed_pretraining

    segmentation = load_config(PROJECT / "configs/segmentation/vit_tiny.yaml")
    segmentation["data"].update(
        root_dir=str(data_root),
        csv_dir=str(tmp_path),
        min_train_images=1,
        input_size=56,
        batch_size=2,
        num_workers=0,
        pin_mem=False,
    )
    segmentation["model"].update(
        pretrained_path=str(weights),
        balanced_decoder_hidden_dim=32,
        balanced_decoder_num_heads=4,
        balanced_decoder_fusion_depth=1,
        balanced_decoder_channels=[32, 24, 16, 8],
        balanced_decoder_dropout=0.0,
    )
    segmentation["meta"].update(
        pred_depth=12, pred_emb_dim=48, use_bfloat16=False, skip_completed=False
    )
    segmentation["optimization"].update(epochs=1, warmup=0)
    segmentation["logging"].update(folder=str(tmp_path / "segmentation"), checkpoint_freq=0)
    segmentation["testing"].update(num_workers=0, batch_size=2)
    seg_config = tmp_path / "segmentation.yaml"
    seg_config.write_text(yaml.safe_dump(segmentation), encoding="utf-8")
    run_cli("segment", "--config", seg_config, "--devices", "cpu")
    decoder = tmp_path / "segmentation/Synthetic/jepa-best.pth.tar"
    checkpoint = torch.load(decoder, weights_only=True)
    assert "target_encoder" not in checkpoint and "decoder" in checkpoint
    segmentation["data"]["dataset_name"] = "Synthetic"
    validate_segmentation_checkpoint(checkpoint, segmentation)
    run_cli(
        "evaluate",
        "--config",
        seg_config,
        "--dataset-name",
        "Synthetic",
        "--devices",
        "cpu",
        "--save-predictions",
    )
    report = tmp_path / "segmentation/Synthetic/test_results"
    summary = json.loads((report / "test_summary.json").read_text(encoding="utf-8"))
    assert summary["num_images"] == 2 and 0 <= summary["mean_dice"] <= 1
    assert len(list((report / "predictions").glob("*.png"))) == 2

    run_cli(
        "segment",
        "--config",
        seg_config,
        "--devices",
        "cpu",
        "--set",
        "meta.load_checkpoint=true",
        "--set",
        "optimization.epochs=2",
    )
    resumed = torch.load(tmp_path / "segmentation/Synthetic/jepa-latest.pth.tar", weights_only=True)
    assert resumed["epoch"] == 2
