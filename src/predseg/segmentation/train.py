import copy
import csv
import gc
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from monai.losses import DiceCELoss
from torch.nn.parallel import DistributedDataParallel as DDP

from predseg.segmentation.data import (
    dataset_folder_name,
    discover_dataset_statistics,
    make_dataset,
)
from predseg.models.segmentation import normalize_decoder_type
from predseg.segmentation.helper import backbone_configuration, build_model, init_opt
from predseg.utils.distributed import AllReduce, get_device, init_distributed
from predseg.utils.logging import AverageMeter, CSVLogger, gpu_timer, grad_logger


log_timings = True
log_freq = 20

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def append_epoch_metrics(path, row):
    write_header = not os.path.isfile(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


class SmoothedEarlyStopping:
    """Stop after a sustained plateau in EMA-smoothed training loss."""

    def __init__(
        self,
        min_epochs=50,
        patience=20,
        relative_min_delta=0.005,
        smoothing=0.9,
    ):
        if min_epochs < 1:
            raise ValueError("early_stopping.min_epochs must be positive")
        if patience < 1:
            raise ValueError("early_stopping.patience must be positive")
        if not 0.0 <= relative_min_delta < 1.0:
            raise ValueError("early_stopping.relative_min_delta must be in [0, 1)")
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("early_stopping.smoothing must be in [0, 1)")

        self.min_epochs = min_epochs
        self.patience = patience
        self.relative_min_delta = relative_min_delta
        self.smoothing = smoothing
        self.smoothed_loss = None
        self.best_smoothed_loss = None
        self.best_epoch = 0
        self.bad_epochs = 0

    def update(self, epoch, loss):
        if self.smoothed_loss is None:
            self.smoothed_loss = float(loss)
        else:
            self.smoothed_loss = self.smoothing * self.smoothed_loss + (
                1.0 - self.smoothing
            ) * float(loss)

        if self.best_smoothed_loss is None:
            significant_improvement = True
        else:
            required_drop = self.best_smoothed_loss * self.relative_min_delta
            significant_improvement = self.smoothed_loss < self.best_smoothed_loss - required_drop

        if significant_improvement:
            self.best_smoothed_loss = self.smoothed_loss
            self.best_epoch = epoch
            self.bad_epochs = 0
        elif epoch <= self.min_epochs:
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        should_stop = epoch >= self.min_epochs and self.bad_epochs >= self.patience
        return significant_improvement, should_stop

    def state_dict(self):
        return {
            "min_epochs": self.min_epochs,
            "patience": self.patience,
            "relative_min_delta": self.relative_min_delta,
            "smoothing": self.smoothing,
            "smoothed_loss": self.smoothed_loss,
            "best_smoothed_loss": self.best_smoothed_loss,
            "best_epoch": self.best_epoch,
            "bad_epochs": self.bad_epochs,
        }

    def load_state_dict(self, state):
        if not state:
            return
        for key in (
            "smoothed_loss",
            "best_smoothed_loss",
            "best_epoch",
            "bad_epochs",
        ):
            if key in state:
                setattr(self, key, state[key])


def load_checkpoint(path, device):
    return torch.load(path, map_location=device, weights_only=True)


def load_model_state(model, state_dict):
    checkpoint_has_module = next(iter(state_dict)).startswith("module.")
    model_has_module = next(iter(model.state_dict())).startswith("module.")
    if checkpoint_has_module and not model_has_module:
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    elif model_has_module and not checkpoint_has_module:
        state_dict = {f"module.{key}": value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def load_segmentation_state(model, checkpoint):
    if "decoder" in checkpoint:
        unwrap_model(model).decoder.load_state_dict(
            checkpoint["decoder"],
            strict=True,
        )
        return
    load_model_state(model, checkpoint["model"])


def resolve_resume_path(args, dataset_name, latest_path):
    configured_path = args["meta"].get("read_checkpoint")
    if not configured_path:
        return latest_path
    return configured_path.format(
        dataset_name=dataset_name,
        dataset_folder=dataset_folder_name(dataset_name),
    )


def masked_predictive_architecture(model_args):
    if not model_args.get("use_masked_predictive_fusion", False):
        return {}
    return {
        "use_masked_predictive_fusion": True,
        "masked_predictive_method": "single_pass_hierarchical_edge_v1",
        "masked_predictive_top_ratio": model_args.get(
            "masked_predictive_top_ratio",
            0.2,
        ),
    }


def decoder_architecture(model_args):
    decoder_type = normalize_decoder_type(model_args.get("decoder_type", "balanced"))
    if decoder_type == "simple":
        architecture = {
            "name": "simple_visual_prior_decoder_final_only_v2",
            "decoder_type": decoder_type,
            "hidden_dim": model_args.get(
                "simple_decoder_hidden_dim",
                256,
            ),
            "num_heads": model_args.get("simple_decoder_num_heads", 8),
            "dropout": model_args.get("simple_decoder_dropout", 0.0),
        }
        architecture["segmentation_loss"] = {
            "name": "bce_dice_final_only",
            "bce_weight": 1.0,
            "dice_weight": 1.0,
        }
        architecture.update(masked_predictive_architecture(model_args))
        return architecture
    if decoder_type == "balanced":
        architecture = {
            "name": "balanced_visual_decoder_detail_v3",
            "decoder_type": decoder_type,
            "hidden_dim": model_args.get(
                "balanced_decoder_hidden_dim",
                320,
            ),
            "num_heads": model_args.get("balanced_decoder_num_heads", 8),
            "fusion_depth": model_args.get(
                "balanced_decoder_fusion_depth",
                2,
            ),
            "mlp_ratio": model_args.get(
                "balanced_decoder_mlp_ratio",
                3.0,
            ),
            "decoder_channels": model_args.get(
                "balanced_decoder_channels",
                [224, 160, 112, 56],
            ),
            "detail_channels": list(model_args.get("balanced_detail_channels", [48, 64, 96, 128])),
            "dropout": model_args.get(
                "balanced_decoder_dropout",
                0.05,
            ),
            "deep_supervision": model_args.get("balanced_deep_supervision", True),
        }
        auxiliary_weights = model_args.get("balanced_auxiliary_loss_weights", [0.2])
        boundary_weight = model_args.get("balanced_boundary_loss_weight", 0.15)
        aux_head_count = 1
    else:
        architecture = {
            "name": "high_performance_visual_decoder_detail_v4",
            "decoder_type": decoder_type,
            "hidden_dim": model_args.get("decoder_hidden_dim", 384),
            "num_heads": model_args.get("decoder_num_heads", 8),
            "fusion_depth": model_args.get("decoder_fusion_depth", 4),
            "mlp_ratio": model_args.get("decoder_mlp_ratio", 4.0),
            "decoder_channels": model_args.get(
                "decoder_channels",
                [256, 192, 128, 64],
            ),
            "detail_channels": list(model_args.get("detail_channels", [64, 96, 128, 192])),
            "dropout": model_args.get("decoder_dropout", 0.1),
            "deep_supervision": model_args.get("deep_supervision", True),
        }
        auxiliary_weights = model_args.get("auxiliary_loss_weights", [0.1, 0.2, 0.3])
        boundary_weight = model_args.get("boundary_loss_weight", 0.2)
        aux_head_count = 3

    if not architecture["deep_supervision"]:
        auxiliary_weights = []
        aux_head_count = 0
    if not isinstance(auxiliary_weights, (list, tuple)) or len(auxiliary_weights) != aux_head_count:
        raise ValueError(
            f"The {decoder_type} decoder expects {aux_head_count} auxiliary loss weights"
        )
    for weight in (*auxiliary_weights, boundary_weight):
        if not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
            raise ValueError("Auxiliary and boundary loss weights must be finite and nonnegative")
    architecture["segmentation_loss"] = {
        "name": "bce_dice_main_auxiliary_boundary",
        "bce_weight": 1.0,
        "dice_weight": 1.0,
        "auxiliary_weights": list(auxiliary_weights),
        "boundary_weight": boundary_weight,
    }
    architecture.update(masked_predictive_architecture(model_args))
    return architecture


def make_boundary_target(mask):
    """Compute the morphological gradient of a binary mask with a 3x3 window."""
    dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded).clamp(0.0, 1.0)


def segmentation_objective(output, target, criterion, settings):
    logits = output["logits"]
    if logits.shape != target.shape:
        raise ValueError(
            f"Logits shape {tuple(logits.shape)} does not match target shape {tuple(target.shape)}"
        )
    loss = criterion(logits, target)
    auxiliary_logits = output.get("aux_logits", ())
    auxiliary_weights = settings.get("auxiliary_weights", ())
    if len(auxiliary_logits) != len(auxiliary_weights):
        raise ValueError("Auxiliary outputs and configured loss weights have different lengths")
    for weight, auxiliary_logit in zip(auxiliary_weights, auxiliary_logits):
        loss = loss + weight * criterion(auxiliary_logit, target)
    boundary_weight = settings.get("boundary_weight", 0.0)
    if boundary_weight > 0:
        loss = loss + boundary_weight * criterion(
            output["boundary_logits"], make_boundary_target(target)
        )
    return loss


def train_one_dataset(args, dataset_name, world_size, rank, device):
    data_args = args["data"]
    model_args = args["model"]
    meta_args = args["meta"]
    optimization_args = args["optimization"]
    logging_args = args["logging"]
    early_args = args.get("early_stopping", {})

    output_root = logging_args["folder"]
    folder_name = dataset_folder_name(dataset_name)
    folder = os.path.join(output_root, folder_name)
    tag = logging_args["write_tag"]
    architecture = decoder_architecture(model_args)
    architecture["backbone"] = backbone_configuration(args)
    os.makedirs(folder, exist_ok=True)

    latest_path = os.path.join(folder, f"{tag}-latest.pth.tar")
    best_path = os.path.join(folder, f"{tag}-best.pth.tar")
    info_path = os.path.join(folder, "training_info.json")
    epoch_log_path = os.path.join(folder, "epoch_metrics.csv")
    iteration_log_path = os.path.join(folder, f"{tag}_r{rank}.csv")

    if meta_args.get("skip_completed", True) and os.path.isfile(info_path):
        with open(info_path, "r", encoding="utf-8") as file:
            previous_info = json.load(file)
        if (
            previous_info.get("status") in {"completed", "early_stopped"}
            and os.path.isfile(best_path)
            and previous_info.get("decoder_architecture") == architecture
        ):
            logger.info(
                "Skipping completed dataset %s (%s)",
                dataset_name,
                previous_info.get("status"),
            )
            return previous_info

    dataset_config = copy.deepcopy(args)
    dataset_config["data"]["dataset_name"] = dataset_name
    dataset_config["logging"]["folder"] = folder
    if rank == 0:
        with open(
            os.path.join(folder, "params-ijepa.yaml"),
            "w",
            encoding="utf-8",
        ) as file:
            yaml.safe_dump(
                dataset_config,
                file,
                allow_unicode=True,
                sort_keys=False,
            )

    logger.info(
        "Starting dataset %s (%s), decoder=%s, output=%s",
        dataset_name,
        folder_name,
        architecture["decoder_type"],
        folder,
    )
    started_at = utc_now()
    started_time = time.perf_counter()

    train_loader, train_sampler = make_dataset(
        root_dir=data_args["root_dir"],
        csv_dir=data_args["csv_dir"],
        dataset_name=dataset_name,
        batch_size=data_args["batch_size"],
        train=True,
        input_size=data_args["input_size"],
        pin_mem=data_args["pin_mem"],
        num_workers=data_args["num_workers"],
        world_size=world_size,
        rank=rank,
        persistent_workers=data_args["persistent_workers"],
        prefetch_factor=data_args["prefetch_factor"],
        drop_last=False,
        train_csv=data_args.get("train_csv", "train.csv"),
        test_csv=data_args.get("test_csv", "test.csv"),
    )
    iterations_per_epoch = len(train_loader)
    if iterations_per_epoch == 0:
        raise RuntimeError(f"Dataset {dataset_name!r} produced no training batches")

    model = build_model(args, device)
    optimizer, scaler, lr_scheduler = init_opt(
        model=model,
        iterations_per_epoch=iterations_per_epoch,
        lr=optimization_args["lr"],
        start_lr=optimization_args["start_lr"],
        final_lr=optimization_args["final_lr"],
        weight_decay=optimization_args["weight_decay"],
        warmup=optimization_args["warmup"],
        epochs=optimization_args["epochs"],
        use_bfloat16=meta_args["use_bfloat16"],
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        model = DDP(model, static_graph=True)

    early_stopping = SmoothedEarlyStopping(
        min_epochs=early_args.get("min_epochs", 50),
        patience=early_args.get("patience", 20),
        relative_min_delta=early_args.get("relative_min_delta", 0.005),
        smoothing=early_args.get("smoothing", 0.9),
    )
    checkpoint_freq = logging_args.get("checkpoint_freq", 50)
    iteration_logger = CSVLogger(
        iteration_log_path,
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.6f", "loss_pred"),
    )

    start_epoch = 0
    epochs = optimization_args["epochs"]
    best_loss = float("inf")
    best_epoch = 0
    final_loss = None
    resumed_from = None
    if meta_args.get("load_checkpoint", False):
        checkpoint_path = resolve_resume_path(args, dataset_name, latest_path)
        if os.path.isfile(checkpoint_path):
            checkpoint = load_checkpoint(checkpoint_path, device)
            if checkpoint.get("decoder_architecture") != architecture:
                logger.warning(
                    "Ignoring incompatible decoder checkpoint for %s",
                    dataset_name,
                )
                checkpoint = None
            if checkpoint is None:
                logger.info("Starting %s with the new decoder", dataset_name)
            else:
                checkpoint_dataset = checkpoint.get("dataset_name")
                if checkpoint_dataset not in (None, dataset_name):
                    raise ValueError(
                        f"Checkpoint belongs to {checkpoint_dataset!r}, not {dataset_name!r}"
                    )
                load_segmentation_state(model, checkpoint)
                optimizer.load_state_dict(checkpoint["opt"])
                if scaler is not None and checkpoint.get("scaler") is not None:
                    scaler.load_state_dict(checkpoint["scaler"])
                start_epoch = int(checkpoint.get("epoch", 0))
                if start_epoch < 0 or start_epoch > epochs:
                    raise ValueError(f"Checkpoint epoch {start_epoch} is outside [0, {epochs}]")
                best_loss = float(checkpoint.get("best_loss", float("inf")))
                best_epoch = int(checkpoint.get("best_epoch", 0))
                final_loss = float(checkpoint.get("loss_pred", float("nan")))
                early_stopping.load_state_dict(checkpoint.get("early_stopping"))
                if checkpoint.get("scheduler") is not None:
                    lr_scheduler.load_state_dict(checkpoint["scheduler"])
                else:
                    lr_scheduler.step_update(start_epoch * iterations_per_epoch)
                resumed_from = checkpoint_path
                logger.info(
                    "Resumed %s from epoch %d",
                    dataset_name,
                    start_epoch,
                )
        else:
            logger.info(
                "No checkpoint for %s at %s; starting fresh",
                dataset_name,
                checkpoint_path,
            )

    loss_settings = architecture["segmentation_loss"]
    seg_loss = DiceCELoss(
        sigmoid=True,
        lambda_ce=loss_settings["bce_weight"],
        lambda_dice=loss_settings["dice_weight"],
    )
    non_blocking = device.type == "cuda" and data_args["pin_mem"]
    use_autocast = (
        meta_args["use_bfloat16"] and device.type == "cuda" and torch.cuda.is_bf16_supported()
    )
    if meta_args["use_bfloat16"] and not use_autocast:
        logger.warning("bfloat16 unavailable; using float32")

    def checkpoint_state(epoch, epoch_loss):
        return {
            "format_version": 1,
            "backbone": backbone_configuration(args),
            "decoder": unwrap_model(model).decoder.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "loss_pred": epoch_loss,
            "best_loss": best_loss,
            "best_epoch": best_epoch,
            "early_stopping": early_stopping.state_dict(),
            "batch_size": data_args["batch_size"],
            "world_size": world_size,
            "lr": optimizer.param_groups[0]["lr"],
            "dataset_name": dataset_name,
            "backbone_checkpoint": model_args["pretrained_path"],
            "model_name": meta_args["model_name"],
            "decoder_architecture": architecture,
            "decoder_type": architecture["decoder_type"],
        }

    status = "completed"
    stopping_reason = "maximum_epochs_reached"
    completed_epochs = start_epoch
    for epoch in range(start_epoch, epochs):
        train_sampler.set_epoch(epoch)
        loss_sum = torch.zeros((), device=device)
        num_batches = 0
        time_meter = AverageMeter()
        data_time_meter = AverageMeter()
        batch_end = time.perf_counter()

        for itr, sample in enumerate(train_loader):
            data_time_meter.update((time.perf_counter() - batch_end) * 1000.0)
            should_log = itr % log_freq == 0

            def train_step():
                pixel_values = sample["pixel_values"].to(
                    device,
                    non_blocking=non_blocking,
                )
                target = sample["label"].to(
                    device,
                    non_blocking=non_blocking,
                )
                lr_scheduler.step_update(epoch * iterations_per_epoch + itr)
                current_lr = optimizer.param_groups[0]["lr"]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=use_autocast,
                ):
                    output = model(pixel_values=pixel_values)
                    loss = segmentation_objective(output, target, seg_loss, loss_settings)

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                gradients = grad_logger(model.named_parameters()) if should_log else None
                return loss.detach(), current_lr, gradients

            (loss_tensor, current_lr, gradients), elapsed = gpu_timer(
                train_step,
                log_timings=should_log,
            )
            loss_sum.add_(loss_tensor)
            num_batches += 1
            if elapsed >= 0:
                time_meter.update(elapsed)

            if should_log:
                global_loss = float(AllReduce.apply(loss_tensor.clone()))
                iteration_logger.log(epoch + 1, itr, global_loss)
                logger.info(
                    "[%s] epoch %d/%d itr %d loss %.4f lr %.2e gpu %.1fms data %.1fms",
                    dataset_name,
                    epoch + 1,
                    epochs,
                    itr,
                    global_loss,
                    current_lr,
                    time_meter.avg,
                    data_time_meter.avg,
                )
                if gradients is not None:
                    logger.info(
                        "[%s] grad min %.2e max %.2e",
                        dataset_name,
                        gradients.min,
                        gradients.max,
                    )
            batch_end = time.perf_counter()

        final_loss = float(AllReduce.apply(loss_sum) / num_batches)
        significant_improvement, should_stop = early_stopping.update(
            epoch + 1,
            final_loss,
        )
        raw_improvement = final_loss < best_loss
        if raw_improvement:
            best_loss = final_loss
            best_epoch = epoch + 1

        if rank == 0:
            state = checkpoint_state(epoch + 1, final_loss)
            torch.save(state, latest_path)
            if raw_improvement:
                torch.save(state, best_path)
            if checkpoint_freq > 0 and (epoch + 1) % checkpoint_freq == 0:
                periodic_path = os.path.join(
                    folder,
                    f"{tag}-ep{epoch + 1}.pth.tar",
                )
                torch.save(state, periodic_path)

            append_epoch_metrics(
                epoch_log_path,
                {
                    "epoch": epoch + 1,
                    "train_loss": final_loss,
                    "smoothed_loss": early_stopping.smoothed_loss,
                    "best_smoothed_loss": (early_stopping.best_smoothed_loss),
                    "significant_improvement": int(significant_improvement),
                    "bad_epochs": early_stopping.bad_epochs,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "elapsed_seconds": time.perf_counter() - started_time,
                },
            )
            running_info = {
                "dataset_name": dataset_name,
                "dataset_folder": folder_name,
                "status": "running",
                "started_at": started_at,
                "updated_at": utc_now(),
                "configured_epochs": epochs,
                "completed_epochs": epoch + 1,
                "train_samples": len(train_loader.dataset),
                "final_loss": final_loss,
                "best_loss": best_loss,
                "best_epoch": best_epoch,
                "early_stopping": early_stopping.state_dict(),
                "decoder_architecture": architecture,
                "decoder_type": architecture["decoder_type"],
                "resumed_from": resumed_from,
            }
            save_json(running_info, info_path)

        logger.info(
            "[%s] epoch %d loss %.6f ema %.6f bad_epochs %d/%d",
            dataset_name,
            epoch + 1,
            final_loss,
            early_stopping.smoothed_loss,
            early_stopping.bad_epochs,
            early_stopping.patience,
        )

        completed_epochs = epoch + 1
        if should_stop:
            status = "early_stopped"
            stopping_reason = (
                f"EMA loss did not improve by at least "
                f"{early_stopping.relative_min_delta:.2%} for "
                f"{early_stopping.patience} epochs after the minimum "
                f"{early_stopping.min_epochs} epochs"
            )
            logger.info("[%s] early stopping: %s", dataset_name, stopping_reason)
            break
    result = {
        "dataset_name": dataset_name,
        "dataset_folder": folder_name,
        "status": status,
        "stopping_reason": stopping_reason,
        "started_at": started_at,
        "finished_at": utc_now(),
        "duration_seconds": time.perf_counter() - started_time,
        "configured_epochs": epochs,
        "completed_epochs": completed_epochs,
        "train_samples": len(train_loader.dataset),
        "iterations_per_epoch": iterations_per_epoch,
        "final_loss": final_loss,
        "best_loss": best_loss,
        "best_epoch": best_epoch,
        "early_stopping": early_stopping.state_dict(),
        "decoder_architecture": architecture,
        "decoder_type": architecture["decoder_type"],
        "latest_checkpoint": os.path.abspath(latest_path),
        "best_checkpoint": os.path.abspath(best_path),
        "resumed_from": resumed_from,
    }
    if rank == 0:
        save_json(result, info_path)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    del train_loader
    del train_sampler
    model = None
    optimizer = None
    lr_scheduler = None
    seg_loss = None
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def write_global_summary(output_root, results):
    os.makedirs(output_root, exist_ok=True)
    save_json(results, os.path.join(output_root, "all_datasets_summary.json"))
    csv_path = os.path.join(output_root, "all_datasets_summary.csv")
    columns = [
        "dataset_name",
        "dataset_folder",
        "decoder_type",
        "status",
        "completed_epochs",
        "configured_epochs",
        "train_samples",
        "best_loss",
        "best_epoch",
        "final_loss",
        "duration_seconds",
        "stopping_reason",
        "best_checkpoint",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for result in results:
            writer.writerow({column: result.get(column) for column in columns})


def normalize_dataset_selection(value, field_name):
    if value is None:
        return None
    if isinstance(value, str):
        name = value.strip()
        return [name] if name else None
    if isinstance(value, (list, tuple, set)):
        names = []
        for item in value:
            if item is None:
                continue
            name = str(item).strip()
            if name:
                names.append(name)
        return names or None
    raise TypeError(f"data.{field_name} must be empty, a string, or a list of strings")


def train_main(args):
    world_size, rank = init_distributed(port=args["meta"]["port"])
    if rank > 0:
        logger.setLevel(logging.ERROR)
    device = get_device()

    data_args = args["data"]
    minimum_images = data_args.get("min_train_images", 100)
    selected_datasets = normalize_dataset_selection(
        data_args.get("dataset_name"),
        "dataset_name",
    )
    include_datasets = normalize_dataset_selection(
        data_args.get("include_datasets"),
        "include_datasets",
    )
    if selected_datasets is not None and include_datasets is not None:
        logger.warning(
            "Both data.dataset_name and data.include_datasets are set; "
            "using data.dataset_name for dataset selection"
        )
    dataset_counts, skipped_counts = discover_dataset_statistics(
        csv_dir=data_args["csv_dir"],
        train_csv=data_args.get("train_csv", "train.csv"),
        include=selected_datasets or include_datasets,
        exclude=data_args.get("exclude_datasets"),
        min_train_images=minimum_images,
    )
    dataset_names = list(dataset_counts)
    logger.info(
        "Eligible datasets (%d, minimum %d images): %s",
        len(dataset_names),
        minimum_images,
        ", ".join(dataset_names),
    )
    logger.info(
        "Decoder type: %s; skipped datasets: %d",
        decoder_architecture(args["model"])["decoder_type"],
        len(skipped_counts),
    )
    for dataset_name, image_count in skipped_counts.items():
        logger.info(
            "Skipping dataset %s: %d training images < %d",
            dataset_name,
            image_count,
            minimum_images,
        )
    folder_names = [dataset_folder_name(name) for name in dataset_names]
    if len(folder_names) != len(set(folder_names)):
        raise ValueError(f"Multiple dataset names map to the same output folder: {folder_names}")

    results = [
        {
            "dataset_name": dataset_name,
            "dataset_folder": dataset_folder_name(dataset_name),
            "status": "skipped",
            "decoder_type": decoder_architecture(args["model"])["decoder_type"],
            "stopping_reason": (
                f"training image count {image_count} is below "
                f"data.min_train_images={minimum_images}"
            ),
            "configured_epochs": args["optimization"]["epochs"],
            "completed_epochs": 0,
            "train_samples": image_count,
            "best_loss": None,
            "best_epoch": None,
            "final_loss": None,
            "duration_seconds": 0.0,
            "best_checkpoint": None,
            "decoder_architecture": decoder_architecture(args["model"]),
        }
        for dataset_name, image_count in skipped_counts.items()
    ]
    if rank == 0:
        write_global_summary(args["logging"]["folder"], results)

    for index, dataset_name in enumerate(dataset_names, start=1):
        logger.info(
            "Dataset %d/%d: %s",
            index,
            len(dataset_names),
            dataset_name,
        )
        result = train_one_dataset(
            args=args,
            dataset_name=dataset_name,
            world_size=world_size,
            rank=rank,
            device=device,
        )
        results.append(result)
        if rank == 0:
            write_global_summary(args["logging"]["folder"], results)

    logger.info("Finished all %d datasets", len(dataset_names))
    return results
