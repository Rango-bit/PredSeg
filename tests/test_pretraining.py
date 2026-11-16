"""Regression checks for transfer, regularization, and portable input data."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from predseg.config import load_config, validate_config
from predseg.models.segmentation import JEPAVisualSeg
from predseg.pretraining.data import MedImgDataset
from predseg.pretraining.helper import init_model, load_checkpoint
from predseg.utils.checkpoint import backbone_metadata, validate_backbone
from predseg.utils.distributed import (
    conditional_sigreg_loss,
    conditional_visreg_loss,
    sigreg_loss,
    visreg_loss,
)
from predseg.utils.masks_dist import compute_mask_distance


PROJECT = Path(__file__).resolve().parents[1]


def test_all_recipes_validate_and_share_pretraining_architecture():
    for path in sorted((PROJECT / "configs/pretraining").glob("*.yaml")):
        pretraining = load_config(path)
        validate_config(pretraining)
        assert pretraining["meta"]["pred_depth"] == 12
        segmentation_path = PROJECT / "configs/segmentation" / path.name
        if segmentation_path.exists():
            segmentation = load_config(segmentation_path)
            validate_config(segmentation)
            for name in ("model_name", "pred_depth", "pred_emb_dim"):
                assert segmentation["meta"][name] == pretraining["meta"][name]
            assert segmentation["model"]["patch_size"] == pretraining["mask"]["patch_size"]
            assert segmentation["data"]["input_size"] == pretraining["data"]["crop_size"]


@pytest.mark.parametrize("ddp_prefix", [False, True])
def test_pretraining_weights_transfer_strictly_to_frozen_segmentation(tmp_path, ddp_prefix):
    encoder, predictor = init_model(
        torch.device("cpu"),
        model_name="vit_tiny",
        crop_size=56,
        patch_size=14,
        pred_depth=12,
        pred_emb_dim=48,
    )
    state = {}
    for key, module in [("target_encoder", encoder), ("predictor", predictor)]:
        state[key] = {
            ("module." + name if ddp_prefix else name): tensor
            for name, tensor in module.state_dict().items()
        }
    if not ddp_prefix:
        state["backbone"] = backbone_metadata("vit_tiny", 56, 14, 12, 48)
    path = tmp_path / "pretrained.pth.tar"
    torch.save(state, path)
    model = JEPAVisualSeg(
        model_name="vit_tiny",
        crop_size=56,
        patch_size=14,
        pred_depth=12,
        pred_emb_dim=48,
        pretrained_path=path,
        balanced_decoder_hidden_dim=32,
        balanced_decoder_num_heads=4,
        balanced_decoder_fusion_depth=1,
        balanced_decoder_channels=(32, 24, 16, 8),
        use_masked_predictive_fusion=True,
    ).train()
    assert not model.img_encoder.training and not model.predictor.training
    assert all(
        torch.equal(value, model.img_encoder.state_dict()[key])
        for key, value in encoder.state_dict().items()
    )
    assert all(
        torch.equal(value, model.predictor.state_dict()[key])
        for key, value in predictor.state_dict().items()
    )
    image = torch.randn(1, 3, 56, 56)
    with torch.no_grad():
        assert torch.equal(encoder(image), torch.cat(model.img_encoder(image), dim=-1))
        assert model(image)["logits"].shape == (1, 1, 56, 56)
    with pytest.raises(ValueError, match="pred_depth"):
        validate_backbone(state, backbone_metadata("vit_tiny", 56, 14, 8, 48))


def test_failed_pretraining_resume_is_not_silently_ignored(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint("cpu", tmp_path / "missing.pth.tar", None, None, None, None, None, None)


@pytest.mark.parametrize("method", ["sigreg", "visreg"])
@pytest.mark.parametrize("constant", [False, True])
def test_distribution_regularization_has_finite_visual_and_text_gradients(method, constant):
    feature = (torch.ones(8, 24) if constant else torch.randn(8, 24)).requires_grad_()
    centers = torch.randn(8, 24, requires_grad=True)
    ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    global_loss = sigreg_loss if method == "sigreg" else visreg_loss
    conditional_loss = conditional_sigreg_loss if method == "sigreg" else conditional_visreg_loss
    loss = global_loss(feature, num_projections=4) + conditional_loss(
        feature,
        ids,
        condition_centers=centers,
        modality_counts=[4, 4],
        num_projections=4,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert feature.grad is not None and torch.isfinite(feature.grad).all()
    assert centers.grad is not None and torch.isfinite(centers.grad).all()


def test_portable_csv_paths_text_cache_and_image_only_mode(tmp_path):
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(tmp_path / "image.png")
    csv = tmp_path / "images.csv"
    pd.DataFrame({"img_path": ["image.png"], "modality_id": [1]}).to_csv(csv, index=False)
    cache = tmp_path / "text.pt"
    torch.save(torch.randn(2, 3, 8), cache)
    dataset = MedImgDataset(csv, text_features_path=cache)
    image, path, modality, text = dataset[0]
    assert image.size == (16, 16) and Path(path).is_file()
    assert modality == 1 and text.shape == (3, 8)
    torch.save(torch.randn(1, 8), cache)
    with pytest.raises(ValueError, match="index rows"):
        MedImgDataset(csv, text_features_path=cache)
    pd.DataFrame({"img_path": ["image.png"]}).to_csv(csv, index=False)
    assert MedImgDataset(csv, require_text=False)[0][0].size == (16, 16)


def test_context_distance_uses_actual_patch_grid():
    # Indices 15 and 16 are on different rows on a 16-wide grid.
    distance = compute_mask_distance([torch.tensor([[16]])], [torch.tensor([[15]])], grid_size=16)
    assert torch.allclose(distance, torch.tensor([[226.0**0.5]]))


def test_config_overrides_and_malformed_architecture():
    config = load_config(
        PROJECT / "configs/pretraining/vit_small.yaml", ["meta.pred_depth=12", "data.batch_size=2"]
    )
    assert config["data"]["batch_size"] == 2
    config["meta"]["pred_emb_dim"] = 385
    with pytest.raises(ValueError, match="attention heads"):
        validate_config(config)
    with pytest.raises(ValueError, match="Unknown config"):
        load_config(PROJECT / "configs/pretraining/vit_small.yaml", ["data.typo=2"])
