"""Checkpoint I/O shared by pretraining and downstream transfer."""

import torch


def load_checkpoint(path):
    """Load tensor-only checkpoints on CPU without executing pickled code."""
    return torch.load(path, map_location="cpu", weights_only=True)


def unwrap(module):
    return module.module if hasattr(module, "module") else module


def clean_state_dict(state_dict):
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def backbone_metadata(model_name, crop_size, patch_size, pred_depth, pred_emb_dim):
    return {
        "model_name": model_name,
        "crop_size": int(crop_size),
        "patch_size": int(patch_size),
        "pred_depth": int(pred_depth),
        "pred_emb_dim": int(pred_emb_dim),
        "n_levels": 4,
    }


def validate_backbone(checkpoint, expected):
    """Reject incompatible transfer settings, including legacy predictor depths."""
    actual = checkpoint.get("backbone", {})
    mismatches = {
        key: (actual[key], value)
        for key, value in expected.items()
        if key in actual and actual[key] != value
    }
    predictor = clean_state_dict(checkpoint.get("predictor", {}))
    blocks = {int(key.split(".")[1]) for key in predictor if key.startswith("predictor_blocks.")}
    if blocks and len(blocks) != expected["pred_depth"]:
        mismatches["pred_depth"] = (len(blocks), expected["pred_depth"])
    if mismatches:
        raise ValueError(
            "Pretraining/segmentation backbone mismatch "
            f"(checkpoint, configured): {mismatches}. "
            "Use the architecture of the pretrained checkpoint."
        )
