"""Benchmark the balanced VisualSeg model with and without JEPA predictor use."""

import argparse
import copy
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import yaml

from predseg.segmentation.helper import build_model as build_segmentation_model


NO_PREDICTOR_EXCLUDED_PREFIXES = (
    "predictor.",
    "decoder.masked_prediction_projection.",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure parameters, FLOPs, and inference speed for the balanced "
            "VisualSeg model with and without JEPA predictor execution."
        )
    )
    parser.add_argument("--fname", default="configs/segmentation/vit_base.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional fine-tuned decoder checkpoint. It does not affect complexity.",
    )
    parser.add_argument(
        "--skip-pretrained",
        action="store_true",
        help="Build the configured architecture without loading the I-JEPA checkpoint.",
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument(
        "--precision",
        choices=("auto", "fp32", "bf16"),
        default="auto",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/analysis/model_complexity",
    )
    return parser.parse_args()


def build_model(config, device, skip_pretrained):
    config = copy.deepcopy(config)
    if config["model"].get("decoder_type", "balanced") != "balanced":
        raise ValueError("This benchmark requires decoder_type='balanced'")
    if skip_pretrained:
        config["model"]["pretrained_path"] = None
    return build_segmentation_model(config, device)


def load_decoder_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    if "decoder" in checkpoint:
        model.decoder.load_state_dict(checkpoint["decoder"], strict=True)
        return
    state_dict = checkpoint.get("model", checkpoint)
    cleaned_state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(cleaned_state_dict, strict=True)


class FullModelForward(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values)["logits"]


class NoPredictorForward(nn.Module):
    """Use an encoder-derived prior while bypassing the JEPA predictor entirely."""

    def __init__(self, model):
        super().__init__()
        if not hasattr(model.decoder, "encoder_prior_projection"):
            raise ValueError("The no-predictor route requires use_masked_predictive_fusion=true")
        self.model = model

    def forward(self, pixel_values):
        with torch.no_grad():
            image_levels = self.model.img_encoder(pixel_values)
            image_evidence = torch.cat(image_levels, dim=-1)
        predictor_features = self.model.decoder.encoder_prior_projection(image_evidence)
        return self.model.decoder(
            image_levels=image_levels,
            predictor_features=predictor_features,
            pixel_values=pixel_values,
        )["logits"]


def count_parameters(model, excluded_prefixes=()):
    selected = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith(excluded_prefixes)
    ]
    return {
        "total_params": int(sum(parameter.numel() for parameter in selected)),
        "trainable_params": int(
            sum(parameter.numel() for parameter in selected if parameter.requires_grad)
        ),
    }


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_autocast(device, precision, use_bfloat16):
    enabled = (
        device.type == "cuda"
        and precision != "fp32"
        and (precision == "bf16" or use_bfloat16)
        and torch.cuda.is_bf16_supported()
    )
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled), enabled


def profile_flops(forward_model, inputs, device, autocast_context):
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    synchronize(device)
    with (
        torch.inference_mode(),
        torch.profiler.profile(
            activities=activities,
            with_flops=True,
            record_shapes=False,
            profile_memory=False,
        ) as profiler,
    ):
        with autocast_context:
            forward_model(inputs)
        synchronize(device)
    return int(sum(event.flops for event in profiler.key_averages() if event.flops))


def benchmark_latency(forward_model, inputs, device, autocast_enabled, warmup, iterations):
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be non-negative and iterations must be positive")

    def forward_once():
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ),
        ):
            forward_model(inputs)

    for _ in range(warmup):
        forward_once()
    synchronize(device)

    latency_ms = []
    for _ in range(iterations):
        synchronize(device)
        start = time.perf_counter()
        forward_once()
        synchronize(device)
        latency_ms.append((time.perf_counter() - start) * 1000.0)

    values = np.asarray(latency_ms, dtype=np.float64)
    mean_ms = float(values.mean())
    return {
        "latency_ms_mean": mean_ms,
        "latency_ms_median": float(np.median(values)),
        "latency_ms_p95": float(np.percentile(values, 95)),
        "throughput_images_per_s": float(1000.0 / mean_ms),
    }


def make_row(name, forward_model, parameter_stats, inputs, device, autocast_enabled, args):
    forward_model.eval()
    autocast_context, _ = make_autocast(device, "bf16" if autocast_enabled else "fp32", False)
    flops = profile_flops(forward_model, inputs, device, autocast_context)
    latency = benchmark_latency(
        forward_model,
        inputs,
        device,
        autocast_enabled,
        args.warmup,
        args.iterations,
    )
    return {
        "variant": name,
        "total_params_m": parameter_stats["total_params"] / 1e6,
        "trainable_params_m": parameter_stats["trainable_params"] / 1e6,
        "flops_g": flops / 1e9,
        **latency,
    }


def main():
    args = parse_args()
    with open(args.fname, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable for {args.device}")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    model = build_model(config, device, args.skip_pretrained)
    if args.checkpoint:
        load_decoder_checkpoint(model, args.checkpoint)
    model.eval()

    use_bfloat16 = args.precision == "bf16" or (
        args.precision == "auto" and config["meta"].get("use_bfloat16", False)
    )
    autocast_context, autocast_enabled = make_autocast(device, args.precision, use_bfloat16)
    del autocast_context

    input_size = int(config["data"]["input_size"])
    inputs = torch.randn(1, 3, input_size, input_size, device=device)
    rows = [
        make_row(
            "With JEPA predictor (masked fusion)",
            FullModelForward(model),
            count_parameters(model),
            inputs,
            device,
            autocast_enabled,
            args,
        ),
        make_row(
            "w/o JEPA predictor (encoder-derived prior)",
            NoPredictorForward(model),
            count_parameters(model, NO_PREDICTOR_EXCLUDED_PREFIXES),
            inputs,
            device,
            autocast_enabled,
            args,
        ),
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "model_complexity.csv"
    json_path = output_dir / "model_complexity.json"
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "config": str(Path(args.fname).resolve()),
                "input_shape": [1, 3, input_size, input_size],
                "decoder_type": "balanced",
                "masked_predictive_top_ratio": config["model"].get("masked_predictive_top_ratio"),
                "precision": "bf16" if autocast_enabled else "fp32",
                "warmup_iterations": args.warmup,
                "timed_iterations": args.iterations,
                "no_predictor_parameter_exclusions": list(NO_PREDICTOR_EXCLUDED_PREFIXES),
                "rows": rows,
            },
            file,
            indent=2,
        )

    for row in rows:
        print(
            f"{row['variant']}: {row['total_params_m']:.2f} M params, "
            f"{row['flops_g']:.2f} GFLOPs, "
            f"{row['latency_ms_mean']:.2f} ms/image, "
            f"{row['throughput_images_per_s']:.2f} images/s"
        )
    print(f"Saved {csv_path}")
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()
