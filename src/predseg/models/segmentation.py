import math

import torch
from torch import nn
from torch.nn import functional as F

import predseg.models.vision_transformer as vit
from predseg.utils.checkpoint import backbone_metadata, load_checkpoint, validate_backbone
from predseg.models.decoders import (
    BalancedVisualDecoder,
    HighPerformanceVisualDecoder,
    SimpleVisualPriorDecoder,
)


DECODER_TYPES = {"balanced", "complex", "simple"}
DECODER_TYPE_ALIASES = {"balance": "balanced"}


def normalize_decoder_type(decoder_type):
    decoder_type = str(decoder_type).strip().lower()
    decoder_type = DECODER_TYPE_ALIASES.get(decoder_type, decoder_type)
    if decoder_type not in DECODER_TYPES:
        raise ValueError(
            f"Unsupported decoder_type={decoder_type!r}; expected one of {sorted(DECODER_TYPES)}"
        )
    return decoder_type


def _clean_state_dict(state_dict):
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


class JEPAVisualSeg(nn.Module):
    """Frozen I-JEPA backbone with one trainable decoder per dataset."""

    def __init__(
        self,
        model_name,
        crop_size,
        patch_size,
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
        super().__init__()
        if crop_size % patch_size != 0:
            raise ValueError(f"crop_size={crop_size} must be divisible by patch_size={patch_size}")
        encoder_dim = vit.VIT_EMBED_DIMS[model_name]
        self.img_encoder = vit.__dict__[model_name](
            img_size=[crop_size],
            patch_size=patch_size,
            return_levels=True,
        )
        if pred_emb_dim % self.img_encoder.num_heads != 0:
            raise ValueError(
                f"pred_emb_dim={pred_emb_dim} must be divisible by the "
                f"{self.img_encoder.num_heads} predictor attention heads"
            )
        self.predictor = vit.__dict__["vit_predictor"](
            num_patches=self.img_encoder.patch_embed.num_patches,
            embed_dim=self.img_encoder.embed_dim,
            predictor_embed_dim=pred_emb_dim,
            depth=pred_depth,
            num_heads=self.img_encoder.num_heads,
        )

        if pretrained_path is not None:
            checkpoint = load_checkpoint(pretrained_path)
            validate_backbone(
                checkpoint,
                backbone_metadata(
                    model_name,
                    crop_size,
                    patch_size,
                    pred_depth,
                    pred_emb_dim,
                ),
            )
            for key in ("target_encoder", "predictor"):
                if key not in checkpoint:
                    raise KeyError(f"Checkpoint {pretrained_path} has no {key!r} state")
            self.img_encoder.load_state_dict(
                _clean_state_dict(checkpoint["target_encoder"]),
                strict=True,
            )
            self.predictor.load_state_dict(
                _clean_state_dict(checkpoint["predictor"]),
                strict=True,
            )

        for module in (self.img_encoder, self.predictor):
            for parameter in module.parameters():
                parameter.requires_grad = False

        self.use_masked_predictive_fusion = use_masked_predictive_fusion
        self.masked_predictive_top_ratio = float(masked_predictive_top_ratio)
        if self.use_masked_predictive_fusion and not 0.0 < self.masked_predictive_top_ratio < 1.0:
            raise ValueError(
                "masked_predictive_top_ratio must be in (0, 1) when "
                "use_masked_predictive_fusion is enabled"
            )

        self.decoder_type = normalize_decoder_type(decoder_type)
        if self.decoder_type == "complex":
            self.decoder = HighPerformanceVisualDecoder(
                encoder_dim=encoder_dim,
                predictor_dim=pred_emb_dim,
                num_encoder_levels=4,
                hidden_dim=decoder_hidden_dim,
                num_heads=decoder_num_heads,
                fusion_depth=decoder_fusion_depth,
                mlp_ratio=decoder_mlp_ratio,
                decoder_channels=tuple(decoder_channels),
                detail_channels=tuple(detail_channels),
                output_size=crop_size,
                dropout=decoder_dropout,
                deep_supervision=deep_supervision,
            )
        elif self.decoder_type == "balanced":
            self.decoder = BalancedVisualDecoder(
                encoder_dim=encoder_dim,
                predictor_dim=pred_emb_dim,
                num_encoder_levels=4,
                hidden_dim=balanced_decoder_hidden_dim,
                num_heads=balanced_decoder_num_heads,
                fusion_depth=balanced_decoder_fusion_depth,
                mlp_ratio=balanced_decoder_mlp_ratio,
                decoder_channels=tuple(balanced_decoder_channels),
                detail_channels=tuple(balanced_detail_channels),
                output_size=crop_size,
                dropout=balanced_decoder_dropout,
                deep_supervision=balanced_deep_supervision,
            )
        else:
            self.decoder = SimpleVisualPriorDecoder(
                encoder_dim=encoder_dim,
                predictor_dim=pred_emb_dim,
                num_encoder_levels=4,
                hidden_dim=simple_decoder_hidden_dim,
                num_heads=simple_decoder_num_heads,
                patch_size=patch_size,
                output_size=crop_size,
                dropout=simple_decoder_dropout,
            )
        if self.use_masked_predictive_fusion:
            self.decoder.encoder_prior_projection = nn.Sequential(
                nn.LayerNorm(encoder_dim * 4),
                nn.Linear(encoder_dim * 4, pred_emb_dim),
                nn.GELU(),
            )
            self.decoder.masked_prediction_projection = nn.Sequential(
                nn.LayerNorm(encoder_dim * 4),
                nn.Linear(encoder_dim * 4, pred_emb_dim),
                nn.GELU(),
            )
        self.train()

    def train(self, mode=True):
        super().train(mode)
        self.img_encoder.eval()
        self.predictor.eval()
        return self

    @staticmethod
    def _gather_tokens(tokens, indices):
        gather_indices = indices.unsqueeze(-1).expand(
            -1,
            -1,
            tokens.size(-1),
        )
        return torch.gather(tokens, dim=1, index=gather_indices)

    @staticmethod
    def _scatter_tokens(tokens, indices, updates):
        if updates.dtype != tokens.dtype:
            updates = updates.to(tokens.dtype)
        scatter_indices = indices.unsqueeze(-1).expand_as(updates)
        return tokens.scatter(dim=1, index=scatter_indices, src=updates)

    @staticmethod
    def _normalize_scores(scores):
        score_min = scores.min(dim=1, keepdim=True).values
        score_max = scores.max(dim=1, keepdim=True).values
        return (scores - score_min) / (score_max - score_min).clamp_min(1e-6)

    @staticmethod
    def _feature_edge_score(tokens):
        batch_size, num_tokens, channels = tokens.shape
        grid_size = math.isqrt(num_tokens)
        if grid_size * grid_size != num_tokens:
            raise ValueError(f"Expected a square patch grid, received {num_tokens} tokens")

        tokens = F.normalize(tokens, dim=-1).reshape(
            batch_size,
            grid_size,
            grid_size,
            channels,
        )
        edge_score = tokens.new_zeros(batch_size, grid_size, grid_size)
        edge_count = tokens.new_zeros(batch_size, grid_size, grid_size)
        if grid_size > 1:
            horizontal = 1.0 - (tokens[:, :, 1:] * tokens[:, :, :-1]).sum(dim=-1)
            edge_score[:, :, 1:] += horizontal
            edge_score[:, :, :-1] += horizontal
            edge_count[:, :, 1:] += 1.0
            edge_count[:, :, :-1] += 1.0

            vertical = 1.0 - (tokens[:, 1:, :] * tokens[:, :-1, :]).sum(dim=-1)
            edge_score[:, 1:, :] += vertical
            edge_score[:, :-1, :] += vertical
            edge_count[:, 1:, :] += 1.0
            edge_count[:, :-1, :] += 1.0

        return (edge_score / edge_count.clamp_min(1.0)).flatten(1)

    @staticmethod
    def _hierarchical_disagreement_score(image_levels):
        if len(image_levels) < 2:
            return image_levels[0].new_zeros(image_levels[0].shape[:2])

        normalized_levels = torch.stack(
            [F.normalize(level, dim=-1) for level in image_levels],
            dim=2,
        )
        level_center = F.normalize(normalized_levels.mean(dim=2), dim=-1)
        disagreement = 1.0 - (normalized_levels * level_center.unsqueeze(2)).sum(dim=-1)
        return disagreement.mean(dim=2)

    def _select_mask_indices(self, image_levels, image_evidence):
        disagreement_score = self._hierarchical_disagreement_score(
            image_levels,
        )
        feature_edge_score = self._feature_edge_score(image_evidence)
        disagreement_score = self._normalize_scores(disagreement_score)
        feature_edge_score = self._normalize_scores(feature_edge_score)
        token_score = disagreement_score * (1.0 + feature_edge_score)

        batch_size, num_tokens = token_score.shape
        num_targets = max(
            1,
            min(
                num_tokens - 1,
                math.ceil(num_tokens * self.masked_predictive_top_ratio),
            ),
        )
        target_indices = torch.topk(
            token_score,
            k=num_targets,
            dim=1,
        ).indices
        target_indices = target_indices.sort(dim=1).values

        target_mask = torch.zeros(
            batch_size,
            num_tokens,
            dtype=torch.bool,
            device=token_score.device,
        )
        target_mask.scatter_(dim=1, index=target_indices, value=True)
        all_indices = torch.arange(
            num_tokens,
            device=token_score.device,
        ).expand(batch_size, -1)
        context_indices = all_indices[~target_mask].view(
            batch_size,
            num_tokens - num_targets,
        )
        return context_indices, target_indices

    def _masked_predictor_features(self, image_levels, image_evidence):
        with torch.no_grad():
            context_indices, target_indices = self._select_mask_indices(
                image_levels,
                image_evidence,
            )
            context_evidence = self._gather_tokens(
                image_evidence,
                context_indices,
            )
            predicted_target = self.predictor(
                context_evidence,
                masks_x=context_indices,
                masks=target_indices,
            )
            if isinstance(predicted_target, tuple):
                predicted_target = predicted_target[0]

        encoder_prior = self.decoder.encoder_prior_projection(image_evidence)
        projected_target = self.decoder.masked_prediction_projection(
            predicted_target,
        )
        baseline_target = self._gather_tokens(
            encoder_prior,
            target_indices,
        )
        observed_target = self._gather_tokens(
            image_evidence,
            target_indices,
        )
        credibility = F.cosine_similarity(
            observed_target,
            predicted_target,
            dim=-1,
        )
        credibility = ((credibility + 1.0) * 0.5).clamp(0.0, 1.0)
        credibility = credibility.unsqueeze(-1)
        fused_target = (1.0 - credibility) * baseline_target + credibility * projected_target
        return self._scatter_tokens(
            encoder_prior,
            target_indices,
            fused_target,
        )

    def forward(self, pixel_values):
        with torch.no_grad():
            image_levels = self.img_encoder(pixel_values)
            if not isinstance(image_levels, (list, tuple)):
                raise ValueError("The image encoder must return hierarchical features")
            image_evidence = torch.cat(image_levels, dim=-1)
        if self.use_masked_predictive_fusion:
            predictor_features = self._masked_predictor_features(
                image_levels,
                image_evidence,
            )
        else:
            with torch.no_grad():
                predictor_features = self.predictor.forward_full(
                    image_evidence,
                )

        return self.decoder(
            image_levels=image_levels,
            predictor_features=predictor_features,
            pixel_values=pixel_values,
        )
