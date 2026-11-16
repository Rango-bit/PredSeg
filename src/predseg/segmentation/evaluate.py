import argparse
import json
import math
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from medpy.metric.binary import hd95
from PIL import Image

from predseg.segmentation.data import DriveDataset, dataset_folder_name
from predseg.segmentation.transforms import get_transform
from predseg.segmentation.checkpoints import validate_segmentation_checkpoint
from predseg.segmentation.helper import build_model


RESULT_COLUMNS = [
    "dataset_name",
    "subfolder",
    "image_name",
    "mask_name",
    "dice",
    "iou",
    "hausdorff_95",
    "pixel_accuracy",
    "pred_foreground_pixels",
    "target_foreground_pixels",
    "prediction_path",
    "_correct_pixels",
    "_total_pixels",
    "_sample_index",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate one VisualSeg decoder on its configured dataset"
    )
    parser.add_argument(
        "--fname",
        default="configs/segmentation/vit_base.yaml",
        help="training config used to build the model",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="segmentation checkpoint; defaults to the dataset best checkpoint",
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="dataset decoder to evaluate",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--devices",
        type=str,
        nargs="+",
        default=["cuda:0"],
        help="devices used for parallel inference",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="save binary prediction masks as PNG files",
    )
    return parser.parse_args()


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def load_model_state(model, checkpoint):
    if "decoder" in checkpoint:
        model.decoder.load_state_dict(checkpoint["decoder"], strict=True)
        return
    state_dict = checkpoint.get("model", checkpoint)
    model_has_module = next(iter(model.state_dict())).startswith("module.")
    checkpoint_has_module = next(iter(state_dict)).startswith("module.")
    if checkpoint_has_module and not model_has_module:
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    elif model_has_module and not checkpoint_has_module:
        state_dict = {f"module.{key}": value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)


def binary_metrics(prediction, target):
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(prediction, target).sum(dtype=np.float64)
    pred_size = prediction.sum(dtype=np.float64)
    target_size = target.sum(dtype=np.float64)
    union = np.logical_or(prediction, target).sum(dtype=np.float64)

    denominator = pred_size + target_size
    dice = 1.0 if denominator == 0 else 2.0 * intersection / denominator
    iou = 1.0 if union == 0 else intersection / union
    pixel_accuracy = np.equal(prediction, target).mean(dtype=np.float64)
    if pred_size == 0 and target_size == 0:
        hausdorff_95 = 0.0
    elif pred_size == 0 or target_size == 0:
        hausdorff_95 = math.nan
    else:
        hausdorff_95 = float(hd95(prediction, target))

    return {
        "dice": float(dice),
        "iou": float(iou),
        "hausdorff_95": hausdorff_95,
        "pixel_accuracy": float(pixel_accuracy),
        "pred_foreground_pixels": int(pred_size),
        "target_foreground_pixels": int(target_size),
        "correct_pixels": int(np.equal(prediction, target).sum()),
        "total_pixels": int(target.size),
    }


def get_options(config, cli_args):
    testing = config.get("testing", {})
    dataset_output = os.path.join(
        config["logging"]["folder"],
        dataset_folder_name(config["data"]["dataset_name"]),
    )
    checkpoint_path = (
        cli_args.checkpoint
        or testing.get("checkpoint")
        or os.path.join(
            dataset_output,
            f"{config['logging']['write_tag']}-best.pth.tar",
        )
    )
    output_dir = Path(
        cli_args.output_dir
        or testing.get("output_dir")
        or os.path.join(dataset_output, "test_results")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "checkpoint_path": checkpoint_path,
        "output_dir": output_dir,
        "threshold": (
            cli_args.threshold if cli_args.threshold is not None else testing.get("threshold", 0.5)
        ),
        "batch_size": (
            cli_args.batch_size
            if cli_args.batch_size is not None
            else testing.get("batch_size", 16)
        ),
        "num_workers": (
            cli_args.num_workers
            if cli_args.num_workers is not None
            else testing.get(
                "num_workers",
                config["data"].get("num_workers", 4),
            )
        ),
        "save_predictions": (cli_args.save_predictions or testing.get("save_predictions", False)),
    }


def get_worker_device(device_name):
    if device_name == "cpu":
        return torch.device("cpu")
    if not device_name.startswith("cuda:"):
        raise ValueError(f"Unsupported device {device_name!r}")
    os.environ["CUDA_VISIBLE_DEVICES"] = device_name.split(":", 1)[1]
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable for {device_name}")
    torch.cuda.set_device(0)
    return torch.device("cuda:0")


def save_prediction(prediction, root, subfolder, image_name):
    output_dir = Path(root)
    if subfolder != "-":
        output_dir = output_dir / subfolder
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{Path(image_name).stem}_prediction.png"
    Image.fromarray(prediction.astype(np.uint8) * 255).save(output_path)
    return str(output_path)


def evaluate_worker(rank, world_size, devices, config, cli_args):
    options = get_options(config, cli_args)
    device = get_worker_device(devices[rank])
    dataset_name = config["data"]["dataset_name"]
    test_dataset = DriveDataset(
        root_dir=config["data"]["root_dir"],
        csv_dir=config["data"]["csv_dir"],
        dataset_name=dataset_name,
        transform=get_transform(
            train=False,
            crop_size=config["data"]["input_size"],
        ),
        train=False,
        train_csv=config["data"].get("train_csv", "train.csv"),
        test_csv=config["data"].get("test_csv", "test.csv"),
    )
    sample_indices = list(range(rank, len(test_dataset), world_size))
    worker_dataset = torch.utils.data.Subset(test_dataset, sample_indices)
    test_loader = torch.utils.data.DataLoader(
        worker_dataset,
        batch_size=options["batch_size"],
        shuffle=False,
        num_workers=options["num_workers"],
        pin_memory=config["data"].get("pin_mem", True),
        persistent_workers=(
            options["num_workers"] > 0 and config["data"].get("persistent_workers", False)
        ),
    )

    checkpoint = load_checkpoint(options["checkpoint_path"])
    validate_segmentation_checkpoint(checkpoint, config)
    model = build_model(config, device)
    load_model_state(model, checkpoint)
    model.eval()

    use_autocast = (
        config["meta"].get("use_bfloat16", False)
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )
    prediction_dir = options["output_dir"] / "predictions"
    rows = []
    sample_offset = 0
    with torch.inference_mode():
        for batch in test_loader:
            pixel_values = batch["pixel_values"].to(
                device,
                non_blocking=device.type == "cuda",
            )
            targets = batch["label"].to(
                device,
                non_blocking=device.type == "cuda",
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_autocast,
            ):
                logits = model(pixel_values=pixel_values)["logits"]
                if logits.shape != targets.shape:
                    raise ValueError(
                        f"Logits shape {tuple(logits.shape)} does not match "
                        f"target shape {tuple(targets.shape)}"
                    )
                predictions = torch.sigmoid(logits) >= options["threshold"]

            predictions_np = predictions[:, 0].cpu().numpy()
            targets_np = (targets[:, 0] > 0.5).cpu().numpy()
            for index, prediction in enumerate(predictions_np):
                sample_index = sample_indices[sample_offset + index]
                subfolder = batch["subfolder"][index]
                image_name = batch["image_name"][index]
                metrics = binary_metrics(prediction, targets_np[index])
                prediction_path = ""
                if options["save_predictions"]:
                    prediction_path = save_prediction(
                        prediction,
                        prediction_dir,
                        subfolder,
                        image_name,
                    )
                rows.append(
                    {
                        "dataset_name": batch["dataset_name"][index],
                        "subfolder": subfolder,
                        "image_name": image_name,
                        "mask_name": batch["mask_name"][index],
                        "dice": metrics["dice"],
                        "iou": metrics["iou"],
                        "hausdorff_95": metrics["hausdorff_95"],
                        "pixel_accuracy": metrics["pixel_accuracy"],
                        "pred_foreground_pixels": (metrics["pred_foreground_pixels"]),
                        "target_foreground_pixels": (metrics["target_foreground_pixels"]),
                        "prediction_path": prediction_path,
                        "_correct_pixels": metrics["correct_pixels"],
                        "_total_pixels": metrics["total_pixels"],
                        "_sample_index": sample_index,
                    }
                )
            sample_offset += len(predictions_np)

    parts_dir = options["output_dir"] / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(
        parts_dir / f"rank_{rank:03d}.csv",
        index=False,
    )
    if rank == 0:
        checkpoint_epoch = checkpoint.get("epoch")
        if torch.is_tensor(checkpoint_epoch):
            checkpoint_epoch = checkpoint_epoch.item()
        with open(
            parts_dir / "metadata.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                {
                    "dataset_name": dataset_name,
                    "dataset_size": len(test_dataset),
                    "checkpoint_epoch": checkpoint_epoch,
                },
                file,
                indent=2,
            )


def save_reports(config, cli_args, options, world_size):
    parts_dir = options["output_dir"] / "parts"
    results = pd.concat(
        [pd.read_csv(parts_dir / f"rank_{rank:03d}.csv") for rank in range(world_size)],
        ignore_index=True,
    )
    if results.empty:
        raise RuntimeError("The selected test dataset produced no samples")
    with open(
        parts_dir / "metadata.json",
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    results = results.sort_values("_sample_index").reset_index(drop=True)
    if results["_sample_index"].astype(int).tolist() != list(range(metadata["dataset_size"])):
        raise RuntimeError("Multi-GPU inference produced missing or duplicate samples")

    summary = {
        "dataset_name": config["data"]["dataset_name"],
        "checkpoint": os.path.abspath(options["checkpoint_path"]),
        "checkpoint_epoch": metadata["checkpoint_epoch"],
        "num_images": len(results),
        "devices": cli_args.devices,
        "threshold": options["threshold"],
        "mean_dice": float(results["dice"].mean()),
        "mean_iou": float(results["iou"].mean()),
        "mean_hausdorff_95": (
            float(results["hausdorff_95"].dropna().mean())
            if results["hausdorff_95"].notna().any()
            else None
        ),
        "mean_pixel_accuracy": float(results["pixel_accuracy"].mean()),
        "global_pixel_accuracy": float(
            results["_correct_pixels"].sum() / results["_total_pixels"].sum()
        ),
    }
    visible_columns = [column for column in results.columns if not column.startswith("_")]
    results[visible_columns].to_csv(
        options["output_dir"] / "test_per_image.csv",
        index=False,
    )
    with open(
        options["output_dir"] / "test_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def evaluate(config, cli_args):
    if not cli_args.devices:
        raise ValueError("At least one inference device must be provided")
    options = get_options(config, cli_args)
    if len(cli_args.devices) != len(set(cli_args.devices)):
        raise ValueError("Inference devices must be unique")
    if "cpu" in cli_args.devices and cli_args.devices != ["cpu"]:
        raise ValueError("Use one cpu device or distinct CUDA devices")
    if not 0 <= options["threshold"] <= 1:
        raise ValueError("Evaluation threshold must be in [0, 1]")
    if options["batch_size"] < 1 or options["num_workers"] < 0:
        raise ValueError("Invalid evaluation batch size or worker count")
    if len(cli_args.devices) == 1:
        evaluate_worker(0, 1, cli_args.devices, config, cli_args)
        save_reports(config, cli_args, options, 1)
        return
    context = mp.get_context("spawn")
    processes = []
    for rank in range(len(cli_args.devices)):
        process = context.Process(
            target=evaluate_worker,
            args=(
                rank,
                len(cli_args.devices),
                cli_args.devices,
                config,
                cli_args,
            ),
        )
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
    failed = [process.exitcode for process in processes if process.exitcode]
    if failed:
        raise RuntimeError(f"Inference workers failed with exit codes {failed}")
    save_reports(config, cli_args, options, len(cli_args.devices))


if __name__ == "__main__":
    arguments = parse_args()
    with open(arguments.fname, "r", encoding="utf-8") as config_file:
        test_config = yaml.safe_load(config_file)
    test_config["data"]["dataset_name"] = arguments.dataset_name
    evaluate(test_config, arguments)
