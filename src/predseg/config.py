"""Read YAML configurations and validate the two-stage architecture contract."""

from pathlib import Path

import yaml


MODEL_HEADS = {
    "vit_tiny": 3,
    "vit_small": 6,
    "vit_base": 12,
    "vit_large": 16,
    "vit_huge": 16,
    "vit_giant": 16,
}


def load_config(path, overrides=()):
    """Resolve paths against the working directory, as in the original projects."""
    with open(path, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping")
    for override in overrides:
        key, separator, value = override.partition("=")
        if not separator:
            raise ValueError(f"Expected dotted.key=value, got {override!r}")
        target = config
        keys = key.split(".")
        for part in keys[:-1]:
            if not isinstance(target.get(part), dict):
                raise ValueError(f"Unknown config section in {key!r}")
            target = target[part]
        if keys[-1] not in target:
            raise ValueError(f"Unknown config option {key!r}")
        target[keys[-1]] = yaml.safe_load(value)
    return config


def validate_config(config, stage=None, check_files=False):
    stage = stage or ("pretrain" if "mask" in config else "segment")
    required = (
        ("data", "meta", "optimization", "logging", "mask", "train")
        if stage == "pretrain"
        else (
            "data",
            "meta",
            "optimization",
            "logging",
            "model",
        )
    )
    for section in required:
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing config section {section!r} for {stage}")
    meta, data, optim = config["meta"], config["data"], config["optimization"]
    model_name = meta["model_name"]
    if model_name not in MODEL_HEADS:
        raise ValueError(f"Unsupported model_name={model_name!r}")
    if meta["pred_depth"] < 1 or meta["pred_emb_dim"] % MODEL_HEADS[model_name]:
        raise ValueError(
            "Predictor depth must be positive and its width divisible by attention heads"
        )
    image_size = data["crop_size"] if stage == "pretrain" else data["input_size"]
    patch_size = (
        config["mask"]["patch_size"] if stage == "pretrain" else config["model"]["patch_size"]
    )
    if patch_size < 1 or image_size % patch_size or image_size // patch_size < 2:
        raise ValueError(
            "Image size must be divisible by patch size and contain at least 2x2 patches"
        )
    if data["batch_size"] < 1 or data["num_workers"] < 0 or optim["epochs"] < 1:
        raise ValueError("Batch size and epochs must be positive; num_workers must be nonnegative")
    if not 0 <= optim["warmup"] < optim["epochs"]:
        raise ValueError("optimization.warmup must be in [0, epochs)")
    files = []
    if stage == "pretrain":
        mask, train = config["mask"], config["train"]
        if mask["num_enc_masks"] != 1:
            raise ValueError("The distance-weighted context objective requires num_enc_masks=1")
        if mask["num_pred_masks"] < 1:
            raise ValueError("num_pred_masks must be positive")
        if train.get("regularization_method", "sigreg") not in ("sigreg", "visreg"):
            raise ValueError("regularization_method must be sigreg or visreg")
        if train.get("loss_exp", 1) <= 0:
            raise ValueError("loss_exp must be positive")
        files.append(data["csv_path"])
        if train.get("use_distribution_regularization", True) and train.get(
            "use_conditional_regularization", True
        ):
            if not data.get("text_features_path"):
                raise ValueError("Conditional regularization requires data.text_features_path")
            files.append(data["text_features_path"])
        if meta.get("pretrained_checkpoint"):
            files.append(meta["pretrained_checkpoint"])
    else:
        model = config["model"]
        if model.get("decoder_type", "balanced") not in (
            "balanced",
            "balance",
            "simple",
            "complex",
        ):
            raise ValueError("decoder_type must be balanced, complex, or simple")
        if (
            model.get("use_masked_predictive_fusion")
            and not 0 < model["masked_predictive_top_ratio"] < 1
        ):
            raise ValueError("masked_predictive_top_ratio must be in (0, 1)")
        if not model.get("pretrained_path"):
            raise ValueError("Segmentation requires model.pretrained_path")
        files.extend([model["pretrained_path"], Path(data["csv_dir"]) / data["train_csv"]])
    if check_files:
        for path in files:
            if not Path(path).is_file():
                raise FileNotFoundError(f"Required file does not exist: {path}")
    return {
        "stage": stage,
        "model_name": model_name,
        "pred_depth": meta["pred_depth"],
        "pred_emb_dim": meta["pred_emb_dim"],
        "image_size": image_size,
        "patch_size": patch_size,
        "output_dir": config["logging"]["folder"],
    }
