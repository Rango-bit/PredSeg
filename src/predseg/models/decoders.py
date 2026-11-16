import math

import torch
from torch import nn
from torch.nn import functional as F


def group_count(channels, maximum=32):
    groups = min(maximum, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


class FeedForward(nn.Module):
    def __init__(self, hidden_dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        expanded_dim = int(hidden_dim * mlp_ratio)
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, expanded_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expanded_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens):
        return self.layers(tokens)


class CrossAttentionUpdate(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.update_norm = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForward(
            hidden_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

    def forward(self, query, context):
        normalized_context = self.context_norm(context)
        update, _ = self.attention(
            query=self.query_norm(query),
            key=normalized_context,
            value=normalized_context,
            need_weights=False,
        )
        update = query + update
        return update + self.ffn(self.update_norm(update))


class SimpleVisualPriorDecoder(nn.Module):
    """Original lightweight encoder-predictor fusion decoder."""

    def __init__(
        self,
        encoder_dim,
        predictor_dim,
        num_encoder_levels=4,
        hidden_dim=256,
        num_heads=8,
        patch_size=14,
        output_size=224,
        dropout=0.0,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.output_size = (output_size, output_size)
        self.evidence_projection = nn.Linear(
            encoder_dim * num_encoder_levels,
            hidden_dim,
        )
        self.prior_projection = nn.Linear(predictor_dim, hidden_dim)
        self.prior_reader = CrossAttentionUpdate(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.fusion_logit = nn.Parameter(torch.tensor(-2.0))
        self.fused_norm = nn.LayerNorm(hidden_dim)

        upsample_dim = max(hidden_dim // 8, 16)
        self.mask_head = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dim,
                upsample_dim,
                kernel_size=patch_size,
                stride=patch_size,
            ),
            nn.GroupNorm(group_count(upsample_dim, maximum=8), upsample_dim),
            nn.GELU(),
            nn.Conv2d(upsample_dim, 1, kernel_size=3, padding=1),
        )

    def forward(self, image_levels, predictor_features, pixel_values=None):
        del pixel_values
        image_evidence = torch.cat(image_levels, dim=-1)
        evidence = self.evidence_projection(image_evidence)
        prior = self.prior_projection(predictor_features)
        prior_update = self.prior_reader(evidence, prior)
        alpha = torch.sigmoid(self.fusion_logit)
        fused = self.fused_norm((1.0 - alpha) * evidence + alpha * prior_update)

        batch_size, num_tokens, hidden_dim = fused.shape
        grid_size = math.isqrt(num_tokens)
        if grid_size * grid_size != num_tokens:
            raise ValueError(f"Expected a square patch grid, received {num_tokens} tokens")
        feature_map = fused.transpose(1, 2).reshape(
            batch_size,
            hidden_dim,
            grid_size,
            grid_size,
        )
        logits = self.mask_head(feature_map)
        if logits.shape[-2:] != self.output_size:
            logits = F.interpolate(
                logits,
                size=self.output_size,
                mode="bilinear",
                align_corners=False,
            )
        return {
            "logits": logits,
            "fusion_alpha": alpha.detach(),
        }


class BidirectionalFusionBlock(nn.Module):
    """Jointly refine encoder evidence and the JEPA predictor prior."""

    def __init__(
        self,
        hidden_dim,
        num_heads,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()
        self.evidence_norm = nn.LayerNorm(hidden_dim)
        self.evidence_self_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.prior_query_norm = nn.LayerNorm(hidden_dim)
        self.evidence_context_norm = nn.LayerNorm(hidden_dim)
        self.prior_reads_evidence = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.evidence_query_norm = nn.LayerNorm(hidden_dim)
        self.prior_context_norm = nn.LayerNorm(hidden_dim)
        self.evidence_reads_prior = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.adaptive_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.adaptive_gate[-2].weight)
        nn.init.constant_(self.adaptive_gate[-2].bias, -2.0)
        self.evidence_ffn_norm = nn.LayerNorm(hidden_dim)
        self.prior_ffn_norm = nn.LayerNorm(hidden_dim)
        self.evidence_ffn = FeedForward(
            hidden_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.prior_ffn = FeedForward(
            hidden_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, evidence, prior):
        normalized_evidence = self.evidence_norm(evidence)
        self_update, _ = self.evidence_self_attention(
            normalized_evidence,
            normalized_evidence,
            normalized_evidence,
            need_weights=False,
        )
        evidence = evidence + self.dropout(self_update)

        prior_update, _ = self.prior_reads_evidence(
            query=self.prior_query_norm(prior),
            key=self.evidence_context_norm(evidence),
            value=self.evidence_context_norm(evidence),
            need_weights=False,
        )
        prior = prior + self.dropout(prior_update)
        prior = prior + self.prior_ffn(self.prior_ffn_norm(prior))

        evidence_update, _ = self.evidence_reads_prior(
            query=self.evidence_query_norm(evidence),
            key=self.prior_context_norm(prior),
            value=self.prior_context_norm(prior),
            need_weights=False,
        )
        gate = self.adaptive_gate(torch.cat([evidence, evidence_update], dim=-1))
        evidence = evidence + self.dropout(gate * evidence_update)
        evidence = evidence + self.evidence_ffn(self.evidence_ffn_norm(evidence))
        return evidence, prior


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
    ):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(group_count(out_channels), out_channels),
            nn.GELU(),
        )


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.projection = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.block = nn.Sequential(
            ConvNormAct(in_channels, out_channels),
            nn.Dropout2d(dropout),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(group_count(out_channels), out_channels),
        )
        self.activation = nn.GELU()

    def forward(self, feature_map):
        return self.activation(self.projection(feature_map) + self.block(feature_map))


class DetailPyramid(nn.Module):
    """Trainable high-resolution branch for boundaries and small objects."""

    def __init__(self, channels=(64, 96, 128, 192), dropout=0.0):
        super().__init__()
        full, half, quarter, eighth = channels
        self.full_stage = nn.Sequential(
            ConvNormAct(3, full),
            ResidualConvBlock(full, full, dropout=dropout),
        )
        self.half_stage = nn.Sequential(
            ConvNormAct(full, half, stride=2),
            ResidualConvBlock(half, half, dropout=dropout),
        )
        self.quarter_stage = nn.Sequential(
            ConvNormAct(half, quarter, stride=2),
            ResidualConvBlock(quarter, quarter, dropout=dropout),
        )
        self.eighth_stage = nn.Sequential(
            ConvNormAct(quarter, eighth, stride=2),
            ResidualConvBlock(eighth, eighth, dropout=dropout),
        )

    def forward(self, image):
        full = self.full_stage(image)
        half = self.half_stage(full)
        quarter = self.quarter_stage(half)
        eighth = self.eighth_stage(quarter)
        return {
            "full": full,
            "half": half,
            "quarter": quarter,
            "eighth": eighth,
        }


class UpFusionStage(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        num_blocks=2,
        dropout=0.0,
    ):
        super().__init__()
        self.input_projection = ConvNormAct(
            in_channels,
            out_channels,
            kernel_size=1,
        )
        self.skip_projection = ConvNormAct(
            skip_channels,
            out_channels,
            kernel_size=1,
        )
        blocks = [
            ResidualConvBlock(
                out_channels * 2,
                out_channels,
                dropout=dropout,
            )
        ]
        blocks.extend(
            ResidualConvBlock(
                out_channels,
                out_channels,
                dropout=dropout,
            )
            for _ in range(num_blocks - 1)
        )
        self.refinement = nn.Sequential(*blocks)

    def forward(self, feature_map, skip):
        feature_map = F.interpolate(
            feature_map,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        feature_map = self.input_projection(feature_map)
        skip = self.skip_projection(skip)
        return self.refinement(torch.cat([feature_map, skip], dim=1))


class BalancedVisualDecoder(nn.Module):
    """Decode cross-attended encoder features with a four-scale detail pyramid."""

    def __init__(
        self,
        encoder_dim,
        predictor_dim,
        num_encoder_levels=4,
        hidden_dim=320,
        num_heads=8,
        fusion_depth=2,
        mlp_ratio=3.0,
        decoder_channels=(224, 160, 112, 56),
        detail_channels=(48, 64, 96, 128),
        output_size=224,
        dropout=0.05,
        deep_supervision=True,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        if len(decoder_channels) != 4 or len(detail_channels) != 4:
            raise ValueError("decoder_channels and detail_channels must each have 4 values")

        self.output_size = (output_size, output_size)
        self.num_encoder_levels = num_encoder_levels
        self.deep_supervision = deep_supervision
        self.level_adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(encoder_dim),
                    nn.Linear(encoder_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for _ in range(num_encoder_levels)
            ]
        )
        self.level_score = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.level_score[-1].weight)
        nn.init.zeros_(self.level_score[-1].bias)
        self.level_fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * num_encoder_levels),
            nn.Linear(hidden_dim * num_encoder_levels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.evidence_norm = nn.LayerNorm(hidden_dim)
        self.prior_projection = nn.Sequential(
            nn.LayerNorm(predictor_dim),
            nn.Linear(predictor_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion_blocks = nn.ModuleList(
            [
                CrossAttentionUpdate(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(fusion_depth)
            ]
        )
        self.fused_norm = nn.LayerNorm(hidden_dim)

        self.detail_pyramid = DetailPyramid(
            channels=detail_channels,
            dropout=dropout,
        )
        stage_28, stage_56, stage_112, stage_224 = decoder_channels
        detail_224, detail_112, detail_56, detail_28 = detail_channels
        self.stage_28 = UpFusionStage(
            hidden_dim,
            detail_28,
            stage_28,
            num_blocks=1,
            dropout=dropout,
        )
        self.stage_56 = UpFusionStage(
            stage_28,
            detail_56,
            stage_56,
            num_blocks=1,
            dropout=dropout,
        )
        self.stage_112 = UpFusionStage(
            stage_56,
            detail_112,
            stage_112,
            num_blocks=1,
            dropout=dropout,
        )
        self.stage_224 = UpFusionStage(
            stage_112,
            detail_224,
            stage_224,
            num_blocks=1,
            dropout=dropout,
        )
        self.boundary_head = nn.Sequential(
            nn.Conv2d(stage_224, stage_224 // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(stage_224 // 2, 1, kernel_size=1),
        )
        self.mask_head = nn.Sequential(
            ResidualConvBlock(
                stage_224,
                stage_224,
                dropout=dropout,
            ),
            nn.Conv2d(stage_224, 1, kernel_size=1),
        )
        self.auxiliary_head = nn.Conv2d(stage_112, 1, kernel_size=1) if deep_supervision else None

    def fuse_encoder_levels(self, image_levels):
        if len(image_levels) != self.num_encoder_levels:
            raise ValueError(
                f"Expected {self.num_encoder_levels} encoder levels, received {len(image_levels)}"
            )
        projected_levels = [
            adapter(level) for adapter, level in zip(self.level_adapters, image_levels)
        ]
        stacked_levels = torch.stack(projected_levels, dim=2)
        level_weights = torch.softmax(
            self.level_score(stacked_levels).squeeze(-1),
            dim=2,
        )
        weighted_levels = (stacked_levels * level_weights.unsqueeze(-1)).sum(dim=2)
        concatenated_levels = self.level_fusion(torch.cat(projected_levels, dim=-1))
        return (
            self.evidence_norm(weighted_levels + concatenated_levels),
            level_weights,
        )

    def forward(self, image_levels, predictor_features, pixel_values):
        evidence, level_weights = self.fuse_encoder_levels(image_levels)
        prior = self.prior_projection(predictor_features)
        if evidence.shape[:2] != prior.shape[:2]:
            raise ValueError("Encoder evidence and predictor prior need the same patch grid")
        for block in self.fusion_blocks:
            evidence = block(evidence, prior)
        evidence = self.fused_norm(evidence)

        batch_size, num_tokens, hidden_dim = evidence.shape
        grid_size = math.isqrt(num_tokens)
        if grid_size * grid_size != num_tokens:
            raise ValueError(f"Expected a square patch grid, received {num_tokens} tokens")
        feature_map = evidence.transpose(1, 2).reshape(
            batch_size,
            hidden_dim,
            grid_size,
            grid_size,
        )

        feature_112, feature_224 = self.upsample_features(feature_map, pixel_values)
        boundary_logits = self.boundary_head(feature_224)
        logits = self.mask_head(feature_224 * (1.0 + torch.sigmoid(boundary_logits)))
        if logits.shape[-2:] != self.output_size:
            logits = F.interpolate(
                logits,
                size=self.output_size,
                mode="bilinear",
                align_corners=False,
            )
            boundary_logits = F.interpolate(
                boundary_logits,
                size=self.output_size,
                mode="bilinear",
                align_corners=False,
            )

        output = {
            "logits": logits,
            "boundary_logits": boundary_logits,
            "level_weights": level_weights.mean(dim=(0, 1)).detach(),
        }
        if self.deep_supervision and self.training:
            output["aux_logits"] = [
                F.interpolate(
                    self.auxiliary_head(feature_112),
                    size=self.output_size,
                    mode="bilinear",
                    align_corners=False,
                )
            ]
        return output

    def upsample_features(self, feature_map, pixel_values):
        details = self.detail_pyramid(pixel_values)
        feature_28 = self.stage_28(feature_map, details["eighth"])
        feature_56 = self.stage_56(feature_28, details["quarter"])
        feature_112 = self.stage_112(feature_56, details["half"])
        feature_224 = self.stage_224(feature_112, details["full"])
        return feature_112, feature_224


class HighPerformanceVisualDecoder(nn.Module):
    """Performance-oriented decoder using semantic and spatial detail paths."""

    def __init__(
        self,
        encoder_dim,
        predictor_dim,
        num_encoder_levels=4,
        hidden_dim=384,
        num_heads=8,
        fusion_depth=4,
        mlp_ratio=4.0,
        decoder_channels=(256, 192, 128, 64),
        detail_channels=(64, 96, 128, 192),
        output_size=224,
        dropout=0.1,
        deep_supervision=True,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        if len(decoder_channels) != 4 or len(detail_channels) != 4:
            raise ValueError("decoder_channels and detail_channels must each have 4 values")

        self.output_size = (output_size, output_size)
        self.num_encoder_levels = num_encoder_levels
        self.deep_supervision = deep_supervision
        self.level_adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(encoder_dim),
                    nn.Linear(encoder_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for _ in range(num_encoder_levels)
            ]
        )
        self.level_score = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.zeros_(self.level_score[-1].weight)
        nn.init.zeros_(self.level_score[-1].bias)
        self.concatenated_level_fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * num_encoder_levels),
            nn.Linear(hidden_dim * num_encoder_levels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.evidence_norm = nn.LayerNorm(hidden_dim)
        self.prior_projection = nn.Sequential(
            nn.LayerNorm(predictor_dim),
            nn.Linear(predictor_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion_blocks = nn.ModuleList(
            [
                BidirectionalFusionBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(fusion_depth)
            ]
        )
        self.final_token_fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.detail_pyramid = DetailPyramid(
            channels=detail_channels,
            dropout=dropout,
        )
        stage_28, stage_56, stage_112, stage_224 = decoder_channels
        detail_224, detail_112, detail_56, detail_28 = detail_channels
        self.stage_28 = UpFusionStage(
            hidden_dim,
            detail_28,
            stage_28,
            dropout=dropout,
        )
        self.stage_56 = UpFusionStage(
            stage_28,
            detail_56,
            stage_56,
            dropout=dropout,
        )
        self.stage_112 = UpFusionStage(
            stage_56,
            detail_112,
            stage_112,
            dropout=dropout,
        )
        self.stage_224 = UpFusionStage(
            stage_112,
            detail_224,
            stage_224,
            dropout=dropout,
        )
        self.boundary_head = nn.Sequential(
            nn.Conv2d(stage_224, stage_224 // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(stage_224 // 2, 1, kernel_size=1),
        )
        self.mask_head = nn.Sequential(
            ResidualConvBlock(
                stage_224,
                stage_224,
                dropout=dropout,
            ),
            nn.Conv2d(stage_224, stage_224 // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(stage_224 // 2, 1, kernel_size=1),
        )

        if deep_supervision:
            self.auxiliary_heads = nn.ModuleList(
                [
                    nn.Conv2d(stage_28, 1, kernel_size=1),
                    nn.Conv2d(stage_56, 1, kernel_size=1),
                    nn.Conv2d(stage_112, 1, kernel_size=1),
                ]
            )
        else:
            self.auxiliary_heads = nn.ModuleList()

    def fuse_encoder_levels(self, image_levels):
        if len(image_levels) != self.num_encoder_levels:
            raise ValueError(
                f"Expected {self.num_encoder_levels} encoder levels, received {len(image_levels)}"
            )
        projected_levels = [
            adapter(level) for adapter, level in zip(self.level_adapters, image_levels)
        ]
        stacked_levels = torch.stack(projected_levels, dim=2)
        level_weights = torch.softmax(
            self.level_score(stacked_levels).squeeze(-1),
            dim=2,
        )
        weighted_levels = (stacked_levels * level_weights.unsqueeze(-1)).sum(dim=2)
        concatenated_levels = self.concatenated_level_fusion(torch.cat(projected_levels, dim=-1))
        evidence = self.evidence_norm(weighted_levels + concatenated_levels)
        return evidence, level_weights

    def forward(self, image_levels, predictor_features, pixel_values):
        if not isinstance(image_levels, (list, tuple)):
            raise ValueError("image_levels must be a list or tuple")
        evidence, level_weights = self.fuse_encoder_levels(image_levels)
        prior = self.prior_projection(predictor_features)
        if evidence.shape[:2] != prior.shape[:2]:
            raise ValueError("Encoder evidence and predictor prior need the same patch grid")

        for block in self.fusion_blocks:
            evidence, prior = block(evidence, prior)

        fused_tokens = self.final_token_fusion(
            torch.cat(
                [
                    evidence,
                    prior,
                    evidence * prior,
                    evidence - prior,
                ],
                dim=-1,
            )
        )
        batch_size, num_tokens, hidden_dim = fused_tokens.shape
        grid_size = math.isqrt(num_tokens)
        if grid_size * grid_size != num_tokens:
            raise ValueError(f"Expected a square patch grid, received {num_tokens} tokens")
        feature_map = fused_tokens.transpose(1, 2).reshape(
            batch_size,
            hidden_dim,
            grid_size,
            grid_size,
        )

        details = self.detail_pyramid(pixel_values)
        feature_28 = self.stage_28(feature_map, details["eighth"])
        feature_56 = self.stage_56(feature_28, details["quarter"])
        feature_112 = self.stage_112(feature_56, details["half"])
        feature_224 = self.stage_224(feature_112, details["full"])
        boundary_logits = self.boundary_head(feature_224)
        boundary_attention = torch.sigmoid(boundary_logits)
        logits = self.mask_head(feature_224 * (1.0 + boundary_attention))
        if logits.shape[-2:] != self.output_size:
            logits = F.interpolate(
                logits,
                size=self.output_size,
                mode="bilinear",
                align_corners=False,
            )
            boundary_logits = F.interpolate(
                boundary_logits,
                size=self.output_size,
                mode="bilinear",
                align_corners=False,
            )

        output = {
            "logits": logits,
            "boundary_logits": boundary_logits,
            "level_weights": level_weights.mean(dim=(0, 1)).detach(),
        }
        if self.deep_supervision and self.training:
            auxiliary_features = [
                feature_28,
                feature_56,
                feature_112,
            ]
            output["aux_logits"] = [
                F.interpolate(
                    head(feature),
                    size=self.output_size,
                    mode="bilinear",
                    align_corners=False,
                )
                for head, feature in zip(
                    self.auxiliary_heads,
                    auxiliary_features,
                )
            ]
        return output
