# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os


import copy
import logging
import sys
import yaml

import numpy as np

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from predseg.masks.multiblock import MaskCollator as MBMaskCollator
from predseg.masks.utils import apply_masks
from predseg.utils.distributed import (
    init_distributed,
    AllReduce,
    conditional_sigreg_loss,
    conditional_visreg_loss,
    sigreg_loss,
    visreg_loss,
)
from predseg.utils.logging import CSVLogger, gpu_timer, grad_logger, AverageMeter
from predseg.utils.tensors import repeat_interleave_batch
from predseg.pretraining.data import make_dataset

from predseg.pretraining.helper import load_checkpoint, load_pretrained_ijepa, init_model, init_opt
from predseg.pretraining.transforms import make_transforms

from predseg.utils.checkpoint import backbone_metadata, unwrap
from predseg.utils.distributed import get_device
from predseg.utils.masks_dist import compute_mask_distance
from predseg.utils.modules import Lambda_LinearWarmupHold


log_timings = True
log_freq = 50
checkpoint_freq = 50


_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(args, resume_preempt=False):

    #  PASSED IN PARAMS FROM CONFIG FILE

    use_bfloat16 = args["meta"]["use_bfloat16"]
    model_name = args["meta"]["model_name"]
    load_model = args["meta"]["load_checkpoint"] or resume_preempt
    r_file = args["meta"]["read_checkpoint"]
    pred_depth = args["meta"]["pred_depth"]
    pred_emb_dim = args["meta"]["pred_emb_dim"]
    pretrained_checkpoint = args["meta"].get("pretrained_checkpoint")
    load_pretrained_predictor = args["meta"].get("load_pretrained_predictor", False)
    device = get_device()
    use_bfloat16 = bool(use_bfloat16 and device.type == "cuda" and torch.cuda.is_bf16_supported())
    from predseg.models.vision_transformer import VIT_EMBED_DIMS

    embed_dim_encoder = VIT_EMBED_DIMS[model_name]
    cfgs_train = args.get("train", {})
    loss_exp = cfgs_train.get("loss_exp", 1.0)
    predict_all = cfgs_train.get("predict_all", True)
    offset_context_loss = cfgs_train.get("offset_context_loss", False)
    weight_distance_loss = cfgs_train.get("weight_distance_loss", True)
    lambda_progressive = cfgs_train.get("lambda_progressive", True)
    lambda_value = cfgs_train.get("lambda_value", 0.25)
    use_distribution_regularization = cfgs_train.get(
        "use_distribution_regularization",
        cfgs_train.get("use_sigreg_loss", True),
    )
    regularization_method = str(cfgs_train.get("regularization_method", "sigreg")).lower()
    if regularization_method not in ("sigreg", "visreg"):
        raise ValueError(
            "regularization_method must be either 'sigreg' or 'visreg', "
            f"got {regularization_method!r}"
        )
    regularization_weight = cfgs_train.get(
        "regularization_weight",
        cfgs_train.get("sigreg_weight", 0.1),
    )
    regularization_pred_weight = cfgs_train.get(
        "regularization_pred_weight",
        cfgs_train.get("sigreg_pred_weight", 1.0),
    )
    regularization_context_weight = cfgs_train.get(
        "regularization_context_weight",
        cfgs_train.get("sigreg_context_weight", 1.0),
    )
    sigreg_num_projections = cfgs_train.get("sigreg_num_projections", 256)
    sigreg_num_frequencies = cfgs_train.get("sigreg_num_frequencies", 17)
    sigreg_max_frequency = cfgs_train.get("sigreg_max_frequency", 5.0)
    use_conditional_regularization = cfgs_train.get(
        "use_conditional_regularization",
        cfgs_train.get("use_conditional_sigreg", True),
    )
    conditional_regularization_weight = cfgs_train.get(
        "conditional_regularization_weight",
        cfgs_train.get("conditional_sigreg_weight", 1.0),
    )
    regularization_effective_beta = cfgs_train.get(
        "regularization_effective_beta",
        cfgs_train.get("sigreg_effective_beta", 0.9999),
    )
    regularization_min_group_size = cfgs_train.get(
        "regularization_min_group_size",
        cfgs_train.get("sigreg_min_group_size", 4),
    )
    regularization_text_center_weight = cfgs_train.get(
        "regularization_text_center_weight",
        cfgs_train.get("sigreg_text_center_weight", 1.0),
    )
    normalize_text_centers = cfgs_train.get(
        "regularization_normalize_text_centers",
        regularization_method == "sigreg",
    )
    visreg_num_projections = cfgs_train.get("visreg_num_projections", sigreg_num_projections)
    visreg_center_weight = cfgs_train.get("visreg_center_weight", 1.0)
    visreg_scale_weight = cfgs_train.get("visreg_scale_weight", 1.0)
    visreg_shape_weight = cfgs_train.get("visreg_shape_weight", 1.0)
    visreg_eps = cfgs_train.get("visreg_eps", 1e-6)
    if regularization_method == "sigreg":
        regularization_loss_key = "loss_sigreg"
        conditional_loss_key = "loss_conditional"
    else:
        regularization_loss_key = "loss_visreg"
        conditional_loss_key = "loss_conditional_visreg"

    use_gaussian_blur = args["data"]["use_gaussian_blur"]
    use_horizontal_flip = args["data"]["use_horizontal_flip"]
    use_color_distortion = args["data"]["use_color_distortion"]
    color_jitter = args["data"]["color_jitter_strength"]

    batch_size = args["data"]["batch_size"]
    pin_mem = args["data"]["pin_mem"]
    num_workers = args["data"]["num_workers"]
    csv_path = args["data"]["csv_path"]
    crop_size = args["data"]["crop_size"]
    crop_scale = args["data"]["crop_scale"]
    modality_column = args["data"]["modality_column"]

    allow_overlap = args["mask"][
        "allow_overlap"
    ]  # whether to allow overlap b/w context and target blocks
    patch_size = args["mask"]["patch_size"]  # patch-size for model training
    num_enc_masks = args["mask"]["num_enc_masks"]  # number of context blocks
    min_keep = args["mask"]["min_keep"]  # min number of patches in context block
    enc_mask_scale = args["mask"]["enc_mask_scale"]  # scale of context blocks
    num_pred_masks = args["mask"]["num_pred_masks"]  # number of target blocks
    pred_mask_scale = args["mask"]["pred_mask_scale"]  # scale of target blocks
    aspect_ratio = args["mask"]["aspect_ratio"]  # aspect ratio of target blocks

    ema = args["optimization"]["ema"]
    ipe_scale = args["optimization"]["ipe_scale"]  # scheduler scale factor (def: 1.0)
    wd = float(args["optimization"]["weight_decay"])
    final_wd = float(args["optimization"]["final_weight_decay"])
    num_epochs = args["optimization"]["epochs"]
    warmup = args["optimization"]["warmup"]
    start_lr = args["optimization"]["start_lr"]
    lr = args["optimization"]["lr"]
    final_lr = args["optimization"]["final_lr"]

    folder = args["logging"]["folder"]
    tag = args["logging"]["write_tag"]

    os.makedirs(folder, exist_ok=True)
    world_size, rank = init_distributed(port=args["meta"].get("port", 29500))
    if rank == 0:
        dump = os.path.join(folder, "params-ijepa.yaml")
        with open(dump, "w", encoding="utf-8") as file:
            yaml.safe_dump(args, file, sort_keys=False)
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")
    if rank > 0:
        logger.setLevel(logging.ERROR)

    log_file = os.path.join(folder, f"{tag}_r{rank}.csv")
    save_path = os.path.join(folder, f"{tag}" + "-ep{epoch}.pth.tar")
    latest_path = os.path.join(folder, f"{tag}-latest.pth.tar")
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    csv_logger = CSVLogger(
        log_file,
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.5f", "loss_pred"),
        ("%.5f", "loss_context"),
        ("%.5f", regularization_loss_key),
        ("%.5f", conditional_loss_key),
        ("%.5f", "loss_total"),
        ("%.5f", "mask-A"),
        ("%.5f", "mask-B"),
        ("%d", "time (ms)"),
    )

    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name,
    )
    if pretrained_checkpoint is not None and not load_model:
        encoder, predictor = load_pretrained_ijepa(
            r_path=pretrained_checkpoint,
            encoder=encoder,
            predictor=predictor,
            load_predictor=load_pretrained_predictor,
        )
    target_encoder = copy.deepcopy(encoder)

    mask_collator = MBMaskCollator(
        input_size=crop_size,
        patch_size=patch_size,
        pred_mask_scale=pred_mask_scale,
        enc_mask_scale=enc_mask_scale,
        aspect_ratio=aspect_ratio,
        nenc=num_enc_masks,
        npred=num_pred_masks,
        allow_overlap=allow_overlap,
        min_keep=min_keep,
    )

    transform = make_transforms(
        crop_size=crop_size,
        crop_scale=crop_scale,
        gaussian_blur=use_gaussian_blur,
        horizontal_flip=use_horizontal_flip,
        color_distortion=use_color_distortion,
        color_jitter=color_jitter,
    )

    dataset, unsupervised_loader, unsupervised_sampler = make_dataset(
        transform=transform,
        batch_size=batch_size,
        collator=mask_collator,
        pin_mem=pin_mem,
        num_workers=num_workers,
        world_size=world_size,
        rank=rank,
        csv_path=csv_path,
        image_root=args["data"].get("image_root"),
        text_features_path=args["data"].get("text_features_path"),
        require_text=use_distribution_regularization and use_conditional_regularization,
        modality_column=modality_column,
        drop_last=True,
    )
    modality_counts = dataset.modality_counts
    text_projector = None
    if use_distribution_regularization and use_conditional_regularization:
        text_feature_dim = int(dataset.text_features_cache.shape[-1])
        predictor_feature_dim = int(predictor.total_dim)
        text_projector = torch.nn.Sequential(
            torch.nn.LayerNorm(text_feature_dim),
            torch.nn.Linear(text_feature_dim, predictor_feature_dim),
        ).to(device)
        torch.nn.init.trunc_normal_(text_projector[1].weight, std=0.02)
        torch.nn.init.zeros_(text_projector[1].bias)
    if rank == 0:
        logger.info(
            "Distribution regularization method: %s",
            regularization_method,
        )
        logger.info(
            "Distribution regularization losses: %s, %s",
            regularization_loss_key,
            conditional_loss_key,
        )
        logger.info(
            "Modality counts: %s",
            {modality_id: count for modality_id, count in enumerate(modality_counts)},
        )
    ipe = len(unsupervised_loader)

    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16,
        text_projector=text_projector,
    )
    if torch.distributed.is_initialized():
        encoder = DistributedDataParallel(encoder, static_graph=True)
        predictor = DistributedDataParallel(predictor, static_graph=True)
        if text_projector is not None:
            text_projector = DistributedDataParallel(text_projector)
    target_encoder.load_state_dict(unwrap(encoder).state_dict())
    for p in target_encoder.parameters():
        p.requires_grad = False
    target_encoder.eval()

    momentum_scheduler = (
        ema[0] + i * (ema[1] - ema[0]) / (ipe * num_epochs * ipe_scale)
        for i in range(int(ipe * num_epochs * ipe_scale) + 1)
    )
    lambda_sched = Lambda_LinearWarmupHold(lambda_value=lambda_value)

    start_epoch = 0

    if load_model:
        (
            encoder,
            predictor,
            target_encoder,
            text_projector,
            optimizer,
            scaler,
            start_epoch,
        ) = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            text_projector=text_projector,
            opt=optimizer,
            scaler=scaler,
        )
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
            mask_collator.step()

    def save_checkpoint(epoch):
        save_dict = {
            "format_version": 1,
            "backbone": backbone_metadata(
                model_name, crop_size, patch_size, pred_depth, pred_emb_dim
            ),
            "config": args,
            "encoder": unwrap(encoder).state_dict(),
            "predictor": unwrap(predictor).state_dict(),
            "target_encoder": target_encoder.state_dict(),
            "text_projector": (
                None if text_projector is None else unwrap(text_projector).state_dict()
            ),
            "opt": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "epoch": epoch,
            "loss_pred": loss_pred_meter.avg,
            "loss_context": loss_context_meter.avg,
            "regularization_method": regularization_method,
            "loss_regularization_global": loss_regularization_meter.avg,
            "loss_conditional": loss_conditional_meter.avg,
            regularization_loss_key: loss_regularization_meter.avg,
            conditional_loss_key: loss_conditional_meter.avg,
            "loss_total": loss_total_meter.avg,
            "batch_size": batch_size,
            "world_size": world_size,
            "lr": lr,
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (
                args["logging"].get("checkpoint_freq", 50) > 0
                and epoch % args["logging"].get("checkpoint_freq", 50) == 0
            ):
                torch.save(save_dict, save_path.format(epoch=f"{epoch}"))

    for epoch in range(start_epoch, num_epochs):
        logger.info("Epoch %d" % (epoch + 1))

        unsupervised_sampler.set_epoch(epoch)

        loss_pred_meter = AverageMeter()
        loss_context_meter = AverageMeter()
        loss_regularization_meter = AverageMeter()
        loss_conditional_meter = AverageMeter()
        loss_total_meter = AverageMeter()
        maskA_meter = AverageMeter()
        maskB_meter = AverageMeter()
        time_meter = AverageMeter()

        for itr, (udata, masks_enc, masks_pred) in enumerate(unsupervised_loader):

            def load_imgs():

                imgs = udata[0].to(device, non_blocking=True)
                modality_ids = udata[2].to(device, non_blocking=True)
                modality_text_features = udata[3].to(device, non_blocking=True)
                masks_1 = [u.to(device, non_blocking=True) for u in masks_enc]
                masks_2 = [u.to(device, non_blocking=True) for u in masks_pred]
                return (
                    imgs,
                    modality_ids,
                    modality_text_features,
                    masks_1,
                    masks_2,
                )

            (
                imgs,
                modality_ids,
                modality_text_features,
                masks_enc,
                masks_pred,
            ) = load_imgs()
            maskA_meter.update(len(masks_enc[0][0]))
            maskB_meter.update(len(masks_pred[0][0]))

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                def forward_target(embed_dim=embed_dim_encoder):
                    with torch.no_grad():
                        h = target_encoder(imgs)

                        if predict_all:
                            h0 = F.layer_norm(h[:, :, :embed_dim], (embed_dim,))
                            h1 = F.layer_norm(h[:, :, embed_dim : embed_dim * 2], (embed_dim,))
                            h2 = F.layer_norm(h[:, :, embed_dim * 2 : embed_dim * 3], (embed_dim,))
                            h3 = F.layer_norm(h[:, :, embed_dim * 3 : embed_dim * 4], (embed_dim,))
                            h = torch.cat([h0, h1, h2, h3], dim=2)
                        else:
                            h = F.layer_norm(h, (h.size(-1),))
                        return h

                def forward_context():
                    z = encoder(imgs, masks_enc)

                    z_pred, z_context = predictor(z, masks_enc, masks_pred)
                    return z_pred, z_context

                def loss_fn(z, h, masks_to_apply, d_weights):
                    h = apply_masks(h, masks_to_apply)

                    if z.shape[0] != h.shape[0]:
                        h = repeat_interleave_batch(
                            h, B=h.shape[0], repeat=z.shape[0] // h.shape[0]
                        )

                    loss = torch.abs(z - h) ** loss_exp

                    if d_weights is not None:
                        weight = 1.0 / (1.0 + d_weights.unsqueeze(-1))
                        loss = loss * weight

                    loss = loss.mean() / loss_exp
                    loss = AllReduce.apply(loss)

                    return loss

                def pool_predictor_views(z):
                    """Reduce mask views and patch tokens to one vector per image."""
                    batch_size_local = imgs.size(0)
                    if z.size(0) % batch_size_local != 0:
                        raise ValueError(
                            f"Cannot map {z.size(0)} predictor rows to "
                            f"batch size {batch_size_local}"
                        )
                    num_views = z.size(0) // batch_size_local
                    z = z.float().mean(dim=1)
                    return z.reshape(num_views, batch_size_local, z.size(-1)).mean(dim=0)

                def compute_distribution_regularization(z_pred, z_context):
                    pred_global = pool_predictor_views(z_pred)
                    context_global = pool_predictor_views(z_context)

                    if regularization_method == "sigreg":
                        kwargs = dict(
                            num_projections=sigreg_num_projections,
                            num_frequencies=sigreg_num_frequencies,
                            max_frequency=sigreg_max_frequency,
                        )
                        global_loss_fn = sigreg_loss
                        conditional_loss_fn = conditional_sigreg_loss
                    else:
                        kwargs = dict(
                            num_projections=visreg_num_projections,
                            center_weight=visreg_center_weight,
                            scale_weight=visreg_scale_weight,
                            shape_weight=visreg_shape_weight,
                            eps=visreg_eps,
                        )
                        global_loss_fn = visreg_loss
                        conditional_loss_fn = conditional_visreg_loss

                    loss_global = regularization_pred_weight * global_loss_fn(
                        pred_global, **kwargs
                    ) + regularization_context_weight * global_loss_fn(context_global, **kwargs)
                    if not use_conditional_regularization:
                        return loss_global, loss_global.new_zeros(())

                    text_cls = modality_text_features[:, 0, :].float()
                    text_centers = text_projector(text_cls)
                    if normalize_text_centers:
                        text_centers = (
                            F.normalize(text_centers, dim=-1) * text_centers.size(-1) ** 0.5
                        )
                    conditional_kwargs = dict(
                        condition_centers=text_centers,
                        modality_counts=modality_counts,
                        effective_beta=regularization_effective_beta,
                        min_group_size=regularization_min_group_size,
                        text_center_weight=regularization_text_center_weight,
                        **kwargs,
                    )
                    loss_conditional = regularization_pred_weight * conditional_loss_fn(
                        pred_global, modality_ids, **conditional_kwargs
                    ) + regularization_context_weight * conditional_loss_fn(
                        context_global, modality_ids, **conditional_kwargs
                    )
                    return loss_global, loss_conditional

                # Step 1. Forward
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=use_bfloat16
                ):
                    h = forward_target()

                    z_pred, z_context = forward_context()

                    loss_pred = loss_fn(z_pred, h, masks_pred, d_weights=None)

                    loss_context = loss_pred.new_zeros(())
                    lambda_value_step = lambda_value
                    if predict_all:
                        distance_weights = compute_mask_distance(
                            masks_pred,
                            masks_enc,
                            grid_size=crop_size // patch_size,
                            offset_context_loss=offset_context_loss,
                        )

                        if weight_distance_loss:
                            d_weights = distance_weights

                        else:
                            d_weights = None

                        loss_context = loss_fn(z_context, h, masks_enc, d_weights)

                        if lambda_progressive:
                            lambda_value_step = lambda_sched.value(epoch * ipe + itr)

                    loss_regularization_global = loss_pred.new_zeros(())
                    loss_conditional = loss_pred.new_zeros(())
                    if use_distribution_regularization:
                        with torch.autocast(device_type=device.type, enabled=False):
                            (
                                loss_regularization_global,
                                loss_conditional,
                            ) = compute_distribution_regularization(z_pred, z_context)

                    loss_regularization = (
                        loss_regularization_global
                        + conditional_regularization_weight * loss_conditional
                    )
                    loss = (
                        loss_pred
                        + loss_context * lambda_value_step
                        + regularization_weight * loss_regularization
                    )

                #  Step 2. Backward & step
                if use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                grad_stats = grad_logger(encoder.named_parameters())
                optimizer.zero_grad()

                # Step 3. momentum update of target encoder
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)

                return (
                    float(loss_pred),
                    float(loss_context),
                    float(loss_regularization_global),
                    float(loss_conditional),
                    float(loss),
                    _new_lr,
                    _new_wd,
                    grad_stats,
                )

            (
                (
                    loss_pred,
                    loss_context,
                    loss_regularization_global,
                    loss_conditional,
                    loss_total,
                    _new_lr,
                    _new_wd,
                    grad_stats,
                ),
                etime,
            ) = gpu_timer(train_step)
            loss_pred_meter.update(loss_pred)
            loss_context_meter.update(loss_context)
            loss_regularization_meter.update(loss_regularization_global)
            loss_conditional_meter.update(loss_conditional)
            loss_total_meter.update(loss_total)
            time_meter.update(etime)

            def log_stats():
                csv_logger.log(
                    epoch + 1,
                    itr,
                    loss_pred,
                    loss_context,
                    loss_regularization_global,
                    loss_conditional,
                    loss_total,
                    maskA_meter.val,
                    maskB_meter.val,
                    etime,
                )
                if (itr % log_freq == 0) or np.isnan(loss_pred) or np.isinf(loss_pred):
                    logger.info(
                        "[%d, %5d] loss_pred: %.3f "
                        "loss_context: %.3f "
                        "%s: %.3f "
                        "%s: %.3f "
                        "loss_total: %.3f "
                        "[wd: %.2e] [lr: %.2e] "
                        "[mem: %.2e] "
                        "(%.1f ms)"
                        % (
                            epoch + 1,
                            itr,
                            loss_pred_meter.avg,
                            loss_context_meter.avg,
                            regularization_loss_key,
                            loss_regularization_meter.avg,
                            conditional_loss_key,
                            loss_conditional_meter.avg,
                            loss_total_meter.avg,
                            _new_wd,
                            _new_lr,
                            (
                                torch.cuda.max_memory_allocated() / 1024.0**2
                                if device.type == "cuda"
                                else 0.0
                            ),
                            time_meter.avg,
                        )
                    )

                    if grad_stats is not None:
                        logger.info(
                            "[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)"
                            % (
                                epoch + 1,
                                itr,
                                grad_stats.first_layer,
                                grad_stats.last_layer,
                                grad_stats.min,
                                grad_stats.max,
                            )
                        )

            log_stats()

        logger.info("avg. loss %.3f" % loss_total_meter.avg)
        save_checkpoint(epoch + 1)
