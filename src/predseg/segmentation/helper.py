import logging
import sys

import torch
from timm.scheduler.cosine_lr import CosineLRScheduler

from predseg.models.segmentation import JEPAVisualSeg
from predseg.utils.checkpoint import backbone_metadata


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def backbone_configuration(config):
    return backbone_metadata(
        config["meta"]["model_name"],
        config["data"]["input_size"],
        config["model"]["patch_size"],
        config["meta"].get("pred_depth", 12),
        config["meta"].get("pred_emb_dim", 384),
    )


def build_model(config, device):
    """Use identical construction settings for training, evaluation, and analysis."""
    model_args = dict(config["model"])
    for key in (
        "auxiliary_loss_weights",
        "boundary_loss_weight",
        "balanced_auxiliary_loss_weights",
        "balanced_boundary_loss_weight",
    ):
        model_args.pop(key, None)
    return init_visual_segmodel(
        device=device,
        crop_size=config["data"]["input_size"],
        model_name=config["meta"]["model_name"],
        pred_depth=config["meta"].get("pred_depth", 12),
        pred_emb_dim=config["meta"].get("pred_emb_dim", 384),
        **model_args,
    )


def init_visual_segmodel(
    device,
    patch_size=16,
    crop_size=224,
    model_name="vit_base",
    pred_depth=12,
    pred_emb_dim=384,
    decoder_type="balanced",
    decoder_hidden_dim=384,
    decoder_num_heads=8,
    decoder_fusion_depth=4,
    decoder_mlp_ratio=4.0,
    decoder_channels=(256, 192, 128, 64),
    decoder_dropout=0.1,
    simple_decoder_hidden_dim=256,
    simple_decoder_num_heads=8,
    simple_decoder_dropout=0.0,
    balanced_decoder_hidden_dim=320,
    balanced_decoder_num_heads=8,
    balanced_decoder_fusion_depth=2,
    balanced_decoder_mlp_ratio=3.0,
    balanced_decoder_channels=(224, 160, 112, 56),
    balanced_decoder_dropout=0.05,
    use_masked_predictive_fusion=False,
    masked_predictive_top_ratio=0.2,
    pretrained_path=None,
    *,
    detail_channels=(64, 96, 128, 192),
    deep_supervision=True,
    balanced_detail_channels=(48, 64, 96, 128),
    balanced_deep_supervision=True,
):
    model = JEPAVisualSeg(
        model_name=model_name,
        crop_size=crop_size,
        patch_size=patch_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        decoder_type=decoder_type,
        decoder_hidden_dim=decoder_hidden_dim,
        decoder_num_heads=decoder_num_heads,
        decoder_fusion_depth=decoder_fusion_depth,
        decoder_mlp_ratio=decoder_mlp_ratio,
        decoder_channels=decoder_channels,
        decoder_dropout=decoder_dropout,
        simple_decoder_hidden_dim=simple_decoder_hidden_dim,
        simple_decoder_num_heads=simple_decoder_num_heads,
        simple_decoder_dropout=simple_decoder_dropout,
        balanced_decoder_hidden_dim=balanced_decoder_hidden_dim,
        balanced_decoder_num_heads=balanced_decoder_num_heads,
        balanced_decoder_fusion_depth=balanced_decoder_fusion_depth,
        balanced_decoder_mlp_ratio=balanced_decoder_mlp_ratio,
        balanced_decoder_channels=balanced_decoder_channels,
        balanced_decoder_dropout=balanced_decoder_dropout,
        use_masked_predictive_fusion=use_masked_predictive_fusion,
        masked_predictive_top_ratio=masked_predictive_top_ratio,
        pretrained_path=pretrained_path,
        detail_channels=detail_channels,
        deep_supervision=deep_supervision,
        balanced_detail_channels=balanced_detail_channels,
        balanced_deep_supervision=balanced_deep_supervision,
    )
    return model.to(device)


def init_opt(
    model,
    iterations_per_epoch,
    lr,
    start_lr,
    final_lr,
    weight_decay,
    warmup,
    epochs,
    use_bfloat16=False,
):
    del use_bfloat16
    params_to_optimize = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not params_to_optimize:
        raise ValueError("No trainable parameters found")

    logger.info(
        "Using AdamW (trainable_params=%d, peak_lr=%.2e, warmup_lr=%.2e, final_lr=%.2e)",
        sum(parameter.numel() for parameter in params_to_optimize),
        lr,
        start_lr,
        final_lr,
    )
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=lr,
        weight_decay=weight_decay,
    )
    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=int(epochs * iterations_per_epoch),
        warmup_lr_init=start_lr,
        warmup_t=int(warmup * iterations_per_epoch),
        lr_min=final_lr,
        cycle_limit=1,
        t_in_epochs=False,
    )
    return optimizer, None, lr_scheduler
