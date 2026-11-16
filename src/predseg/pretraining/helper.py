# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import sys

import torch

from predseg.utils.checkpoint import clean_state_dict, load_checkpoint as read_checkpoint, unwrap
import predseg.models.vision_transformer as vit
from predseg.utils.schedulers import WarmupCosineSchedule, CosineWDSchedule

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def load_checkpoint(
    device,
    r_path,
    encoder,
    predictor,
    target_encoder,
    text_projector,
    opt,
    scaler,
):
    del device
    checkpoint = read_checkpoint(r_path)
    epoch = int(checkpoint["epoch"])
    for name, module in [
        ("encoder", encoder),
        ("predictor", predictor),
        ("target_encoder", target_encoder),
        ("text_projector", text_projector),
    ]:
        if module is not None:
            if checkpoint.get(name) is None:
                raise ValueError(f"Resume checkpoint has no {name} state")
            unwrap(module).load_state_dict(clean_state_dict(checkpoint[name]), strict=True)
    opt.load_state_dict(checkpoint["opt"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    logger.info("Resumed pretraining from %s at epoch %d", r_path, epoch)
    return encoder, predictor, target_encoder, text_projector, opt, scaler, epoch


def _strip_prefix_if_present(state_dict, prefix):
    if not any(key.startswith(prefix) for key in state_dict):
        return state_dict
    return {
        key[len(prefix) :] if key.startswith(prefix) else key: value
        for key, value in state_dict.items()
    }


def _clean_state_dict(state_dict):
    state_dict = _strip_prefix_if_present(state_dict, "module.")
    state_dict = _strip_prefix_if_present(state_dict, "encoder.")
    state_dict = _strip_prefix_if_present(state_dict, "target_encoder.")
    state_dict = _strip_prefix_if_present(state_dict, "backbone.")
    return state_dict


def _select_pretrained_state(checkpoint, keys):
    for key in keys:
        state = checkpoint.get(key)
        if isinstance(state, dict):
            return state, key
    return checkpoint, "checkpoint"


def _load_matching_state(module, state_dict, module_name):
    model_state = module.state_dict()
    matched_state = {}
    skipped = []
    for key, value in _clean_state_dict(state_dict).items():
        if key in model_state and model_state[key].shape == value.shape:
            matched_state[key] = value
        else:
            skipped.append(key)

    msg = module.load_state_dict(matched_state, strict=False)
    logger.info(
        "loaded %d/%d tensors into %s from pretrained checkpoint",
        len(matched_state),
        len(model_state),
        module_name,
    )
    if skipped:
        logger.info(
            "skipped %d incompatible or unused tensors for %s",
            len(skipped),
            module_name,
        )
    logger.info("%s load msg: %s", module_name, msg)


def load_pretrained_ijepa(
    r_path,
    encoder,
    predictor=None,
    load_predictor=False,
):
    checkpoint = read_checkpoint(r_path)
    encoder_state, encoder_key = _select_pretrained_state(
        checkpoint,
        ("target_encoder", "encoder", "model", "state_dict"),
    )
    logger.info(
        "loading pretrained encoder weights from %s[%s]",
        r_path,
        encoder_key,
    )
    _load_matching_state(encoder, encoder_state, "encoder")

    if load_predictor and predictor is not None:
        predictor_state = checkpoint.get("predictor")
        if isinstance(predictor_state, dict):
            logger.info("loading pretrained predictor weights from %s[predictor]", r_path)
            _load_matching_state(predictor, predictor_state, "predictor")
        else:
            logger.info(
                "pretrained checkpoint has no predictor state; predictor is randomly initialized"
            )

    del checkpoint
    return encoder, predictor


def init_model(
    device, patch_size=16, model_name="vit_base", crop_size=224, pred_depth=12, pred_emb_dim=384
):
    encoder = vit.__dict__[model_name](img_size=[crop_size], patch_size=patch_size)
    predictor = vit.__dict__["vit_predictor"](
        num_patches=encoder.patch_embed.num_patches,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=pred_emb_dim,
        depth=pred_depth,
        num_heads=encoder.num_heads,
    )

    encoder.to(device)
    predictor.to(device)
    logger.info(encoder)
    return encoder, predictor


def init_opt(
    encoder,
    predictor,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    use_bfloat16=False,
    ipe_scale=1.25,
    text_projector=None,
):
    param_groups = [
        {
            "params": (
                p
                for n, p in encoder.named_parameters()
                if ("bias" not in n) and (len(p.shape) != 1)
            )
        },
        {
            "params": (
                p
                for n, p in predictor.named_parameters()
                if ("bias" not in n) and (len(p.shape) != 1)
            )
        },
        {
            "params": (
                p for n, p in encoder.named_parameters() if ("bias" in n) or (len(p.shape) == 1)
            ),
            "WD_exclude": True,
            "weight_decay": 0,
        },
        {
            "params": (
                p for n, p in predictor.named_parameters() if ("bias" in n) or (len(p.shape) == 1)
            ),
            "WD_exclude": True,
            "weight_decay": 0,
        },
    ]
    if text_projector is not None:
        param_groups.extend(
            [
                {
                    "params": (
                        p
                        for n, p in text_projector.named_parameters()
                        if ("bias" not in n) and (len(p.shape) != 1)
                    )
                },
                {
                    "params": (
                        p
                        for n, p in text_projector.named_parameters()
                        if ("bias" in n) or (len(p.shape) == 1)
                    ),
                    "WD_exclude": True,
                    "weight_decay": 0,
                },
            ]
        )

    logger.info("Using AdamW")
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )
    scaler = torch.amp.GradScaler("cuda") if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler
