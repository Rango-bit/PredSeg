# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
import os

import torch
import torch.distributed as dist

from logging import getLogger

logger = getLogger()


def get_device():
    if not torch.cuda.is_available():
        return torch.device("cpu")
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    return device


def init_distributed(port=29500, rank_and_world_size=(None, None)):
    """Initialize torchrun or local workers; fail visibly on rendezvous errors."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    rank, world_size = rank_and_world_size
    rank = int(os.environ.get("RANK", 0)) if rank is None else rank
    world_size = int(os.environ.get("WORLD_SIZE", 1)) if world_size is None else world_size
    if world_size == 1:
        return 1, 0
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(port))
    device = get_device()
    backend = "nccl" if device.type == "cuda" and dist.is_nccl_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return world_size, rank


class AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            x = x.contiguous()
            outputs = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(outputs, x)
            return torch.cat(outputs, 0)
        return x

    @staticmethod
    def backward(ctx, grads):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            s = (grads.shape[0] // dist.get_world_size()) * dist.get_rank()
            e = (grads.shape[0] // dist.get_world_size()) * (dist.get_rank() + 1)
            grads = grads.contiguous()
            dist.all_reduce(grads)
            return grads[s:e]
        return grads


class AllReduceSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            x = x.contiguous()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads


class AllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            x = x.contiguous() / dist.get_world_size()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads


def _all_gather_no_grad(x):
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        outputs = [torch.empty_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(outputs, x.contiguous())
        return torch.cat(outputs, dim=0)
    return x


def _sample_unit_directions(feature_dim, num_projections, device, dtype):
    directions = torch.empty(feature_dim, num_projections, device=device, dtype=dtype)
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        directions.normal_()
        directions.div_(directions.norm(dim=0, keepdim=True).clamp_min(1e-6))
    if dist.is_available() and dist.is_initialized():
        dist.broadcast(directions, src=0)
    return directions


def _sigreg_from_samples(
    z,
    num_projections=256,
    num_frequencies=17,
    max_frequency=5.0,
):
    directions = _sample_unit_directions(
        feature_dim=z.size(-1),
        num_projections=num_projections,
        device=z.device,
        dtype=z.dtype,
    )
    projected = z @ directions

    frequencies = torch.linspace(
        max_frequency / num_frequencies,
        max_frequency,
        num_frequencies,
        device=z.device,
        dtype=z.dtype,
    )
    angles = projected.unsqueeze(-1) * frequencies
    empirical_real = angles.cos().mean(dim=0)
    empirical_imag = angles.sin().mean(dim=0)
    target_real = torch.exp(-0.5 * frequencies.square()).unsqueeze(0)
    weights = torch.exp(-0.5 * frequencies.square()).unsqueeze(0)

    error = (empirical_real - target_real).square() + empirical_imag.square()
    return (error * weights).sum() / (num_projections * weights.sum().clamp_min(1e-6))


def _standard_normal_quantiles(num_samples, device, dtype):
    quantiles = torch.linspace(
        1,
        num_samples,
        num_samples,
        device=device,
        dtype=torch.float32,
    ) / (num_samples + 1)
    target = torch.erfinv(2 * quantiles - 1).mul_(math.sqrt(2))
    return target.to(dtype=dtype)


def _visreg_from_samples(
    z,
    num_projections=256,
    center_weight=1.0,
    scale_weight=1.0,
    shape_weight=1.0,
    eps=1e-6,
):
    """VISReg-style center/scale/shape regularization for [B, D] features."""
    if z.ndim != 2:
        raise ValueError(f"VISReg expects [B, D], got {tuple(z.shape)}")
    if z.size(0) < 2:
        return z.sum() * 0.0

    z = z.float()
    batch_size = z.size(0)

    mean = z.mean(dim=0, keepdim=True)
    center_loss = mean.square().mean()

    z_centered = z - mean
    # Put epsilon inside sqrt. Using sqrt(var).add(eps) keeps the forward pass
    # finite, but its backward is unstable when a dimension has zero variance.
    var = z_centered.square().mean(dim=0)
    std = (var + eps).sqrt()
    scale_loss = (std - 1.0).square().mean()

    z_normalized = z_centered / std.detach().unsqueeze(0)
    directions = _sample_unit_directions(
        feature_dim=z.size(-1),
        num_projections=num_projections,
        device=z.device,
        dtype=z.dtype,
    )
    projected = z_normalized @ directions
    projected = projected.sort(dim=0).values
    target = _standard_normal_quantiles(
        batch_size,
        device=z.device,
        dtype=z.dtype,
    ).view(batch_size, 1)
    shape_loss = (projected - target).square().mean()

    return center_weight * center_loss + scale_weight * scale_loss + shape_weight * shape_loss


def _effective_number_weight(modality_id, modality_counts, effective_beta, device):
    if modality_counts is None:
        return torch.tensor(1.0, device=device)

    count = max(float(modality_counts[int(modality_id.item())]), 1.0)
    effective_number = 1.0 - math.pow(effective_beta, count)
    return torch.tensor(
        (1.0 - effective_beta) / effective_number,
        device=device,
    )


def sigreg_loss(
    z,
    num_projections=256,
    num_frequencies=17,
    max_frequency=5.0,
):
    """
    Match random one-dimensional slices of z to N(0, 1).

    Inputs must contain one representation per image, shaped [B, D].
    Statistics are computed in FP32 over the global distributed batch.
    """
    if z.ndim != 2:
        raise ValueError(f"SIGReg expects [B, D], got {tuple(z.shape)}")
    z = AllGather.apply(z.float())
    if z.size(0) < 2:
        return z.sum() * 0.0
    return _sigreg_from_samples(
        z,
        num_projections=num_projections,
        num_frequencies=num_frequencies,
        max_frequency=max_frequency,
    )


def visreg_loss(
    z,
    num_projections=256,
    center_weight=1.0,
    scale_weight=1.0,
    shape_weight=1.0,
    eps=1e-6,
):
    """
    Match the global feature distribution with VISReg-style regularization.

    Unlike SIGReg, this decouples center, per-dimension scale, and sliced
    Wasserstein shape matching. Inputs must contain one representation per
    image, shaped [B, D].
    """
    if z.ndim != 2:
        raise ValueError(f"VISReg expects [B, D], got {tuple(z.shape)}")
    z = AllGather.apply(z.float())
    return _visreg_from_samples(
        z,
        num_projections=num_projections,
        center_weight=center_weight,
        scale_weight=scale_weight,
        shape_weight=shape_weight,
        eps=eps,
    )


def conditional_sigreg_loss(
    z,
    modality_ids,
    condition_centers=None,
    modality_counts=None,
    num_projections=256,
    num_frequencies=17,
    max_frequency=5.0,
    effective_beta=0.9999,
    min_group_size=4,
    text_center_weight=1.0,
):
    """
    Match each modality-conditional residual distribution independently.

    Effective-number weights prevent high-resource modalities from dominating
    the regularizer. When condition_centers are provided, each modality is
    centered on its projected text representation, optionally blended with the
    stop-gradient empirical visual mean. Modality IDs must be stable across
    all workers.
    """
    if z.ndim != 2:
        raise ValueError(f"Conditional SIGReg expects [B, D], got {tuple(z.shape)}")
    if modality_ids.ndim != 1 or modality_ids.size(0) != z.size(0):
        raise ValueError("modality_ids must have shape [B]")
    if condition_centers is not None and condition_centers.shape != z.shape:
        raise ValueError(
            "condition_centers must have the same shape as z, "
            f"got {tuple(condition_centers.shape)} and {tuple(z.shape)}"
        )
    if not 0.0 <= text_center_weight <= 1.0:
        raise ValueError("text_center_weight must be in [0, 1]")

    z_global = AllGather.apply(z.float())
    ids_global = _all_gather_no_grad(modality_ids.to(z.device, dtype=torch.long))
    centers_global = (
        AllGather.apply(condition_centers.float()) if condition_centers is not None else None
    )
    losses = []
    weights = []

    for modality_id in ids_global.unique(sorted=True):
        group_mask = ids_global == modality_id
        group = z_global[group_mask]
        if group.size(0) < min_group_size:
            continue

        visual_center = group.mean(dim=0, keepdim=True).detach()
        if centers_global is None:
            conditional_center = visual_center
        else:
            text_center = centers_global[group_mask].mean(dim=0, keepdim=True)
            conditional_center = (
                text_center_weight * text_center + (1.0 - text_center_weight) * visual_center
            )
        group = group - conditional_center
        losses.append(
            _sigreg_from_samples(
                group,
                num_projections=num_projections,
                num_frequencies=num_frequencies,
                max_frequency=max_frequency,
            )
        )

        if modality_counts is None:
            weights.append(group.new_tensor(1.0))
        else:
            weights.append(
                _effective_number_weight(
                    modality_id,
                    modality_counts,
                    effective_beta,
                    device=group.device,
                )
            )

    if not losses:
        loss = z.sum() * 0.0
        if condition_centers is not None:
            loss = loss + condition_centers.sum() * 0.0
        return loss

    weights = torch.stack(weights)
    weights = weights / weights.sum().clamp_min(1e-6)
    return (torch.stack(losses) * weights).sum()


def conditional_visreg_loss(
    z,
    modality_ids,
    condition_centers=None,
    modality_counts=None,
    num_projections=256,
    effective_beta=0.9999,
    min_group_size=4,
    text_center_weight=1.0,
    center_weight=1.0,
    scale_weight=1.0,
    shape_weight=1.0,
    eps=1e-6,
):
    """
    Apply VISReg-style center/scale/shape matching to modality residuals.

    Residuals are built around modality-conditioned centers. With text centers,
    the center is a weighted mixture of projected modality text and a detached
    empirical visual mean; without text centers, it falls back to the detached
    visual mean, matching the old conditional centering behavior.
    """
    if z.ndim != 2:
        raise ValueError(f"Conditional VISReg expects [B, D], got {tuple(z.shape)}")
    if modality_ids.ndim != 1 or modality_ids.size(0) != z.size(0):
        raise ValueError("modality_ids must have shape [B]")
    if condition_centers is not None and condition_centers.shape != z.shape:
        raise ValueError(
            "condition_centers must have the same shape as z, "
            f"got {tuple(condition_centers.shape)} and {tuple(z.shape)}"
        )
    if not 0.0 <= text_center_weight <= 1.0:
        raise ValueError("text_center_weight must be in [0, 1]")

    z_global = AllGather.apply(z.float())
    ids_global = _all_gather_no_grad(modality_ids.to(z.device, dtype=torch.long))
    centers_global = (
        AllGather.apply(condition_centers.float()) if condition_centers is not None else None
    )
    losses = []
    weights = []

    for modality_id in ids_global.unique(sorted=True):
        group_mask = ids_global == modality_id
        group = z_global[group_mask]
        if group.size(0) < min_group_size:
            continue

        visual_center = group.mean(dim=0, keepdim=True).detach()
        if centers_global is None:
            conditional_center = visual_center
        else:
            text_center = centers_global[group_mask].mean(dim=0, keepdim=True)
            conditional_center = (
                text_center_weight * text_center + (1.0 - text_center_weight) * visual_center
            )
        residuals = group - conditional_center
        losses.append(
            _visreg_from_samples(
                residuals,
                num_projections=num_projections,
                center_weight=center_weight,
                scale_weight=scale_weight,
                shape_weight=shape_weight,
                eps=eps,
            )
        )
        weights.append(
            _effective_number_weight(
                modality_id,
                modality_counts,
                effective_beta,
                device=group.device,
            )
        )

    if not losses:
        loss = z.sum() * 0.0
        if condition_centers is not None:
            loss = loss + condition_centers.sum() * 0.0
        return loss

    weights = torch.stack(weights)
    weights = weights / weights.sum().clamp_min(1e-6)
    return (torch.stack(losses) * weights).sum()
