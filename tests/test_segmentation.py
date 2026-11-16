"""Synthetic regression checks for segmentation transfer and supervision."""

import copy
import inspect
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml
from monai.losses import DiceCELoss, DiceLoss
from torch.utils.data import DataLoader, DistributedSampler

PROJECT = Path(__file__).resolve().parents[1]

from predseg.models.segmentation import JEPAVisualSeg  # noqa: E402
from predseg.analysis.model_complexity_benchmark import NoPredictorForward  # noqa: E402
import predseg.segmentation.evaluate as evaluation  # noqa: E402
import predseg.segmentation.train as train  # noqa: E402


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(7)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous_threads)


def small_model(decoder_type="balanced", masked=True):
    return JEPAVisualSeg(
        model_name="vit_tiny",
        crop_size=56,
        patch_size=14,
        pred_depth=2,
        pred_emb_dim=48,
        decoder_type=decoder_type,
        decoder_hidden_dim=32,
        decoder_num_heads=4,
        decoder_fusion_depth=1,
        decoder_channels=(32, 24, 16, 8),
        detail_channels=(8, 12, 16, 24),
        decoder_dropout=0.0,
        balanced_decoder_hidden_dim=32,
        balanced_decoder_num_heads=4,
        balanced_decoder_fusion_depth=1,
        balanced_decoder_channels=(32, 24, 16, 8),
        balanced_detail_channels=(8, 12, 16, 24),
        balanced_decoder_dropout=0.0,
        simple_decoder_hidden_dim=32,
        simple_decoder_num_heads=4,
        use_masked_predictive_fusion=masked,
    )


@pytest.mark.parametrize("decoder_type", ["balanced", "complex", "simple"])
@pytest.mark.parametrize("masked", [True, False])
def test_forward_backward_preserves_frozen_backbone_and_prior_gradients(decoder_type, masked):
    model = small_model(decoder_type, masked).train()
    assert not model.img_encoder.training
    assert not model.predictor.training
    has_details = decoder_type != "simple"
    assert hasattr(model.decoder, "detail_pyramid") == has_details
    assert hasattr(model.decoder, "boundary_head") == has_details

    calls = []
    if masked:
        handle = model.predictor.register_forward_hook(lambda *_: calls.append(True))
    image = torch.randn(2, 3, 56, 56, requires_grad=True)
    output = model(image)
    if masked:
        handle.remove()
        assert len(calls) == 1
    assert output["logits"].shape == (2, 1, 56, 56)
    target = torch.zeros_like(output["logits"])
    target[0, :, 12:30, 12:30] = 1  # Include both a foreground and an empty mask.
    criterion = DiceCELoss(sigmoid=True, lambda_ce=1.0, lambda_dice=1.0)
    settings = train.decoder_architecture({"decoder_type": decoder_type})["segmentation_loss"]
    loss = train.segmentation_objective(output, target, criterion, settings)
    expected = criterion(output["logits"], target)
    if has_details:
        aux_weights = (0.2,) if decoder_type == "balanced" else (0.1, 0.2, 0.3)
        assert len(output["aux_logits"]) == len(aux_weights)
        for weight, aux in zip(aux_weights, output["aux_logits"]):
            assert aux.shape == target.shape
            expected = expected + weight * criterion(aux, target)
        boundary_target = torch.zeros_like(target)
        boundary_target[0, :, 11:31, 11:31] = 1
        boundary_target[0, :, 13:29, 13:29] = 0
        assert torch.equal(train.make_boundary_target(target), boundary_target)
        assert output["boundary_logits"].shape == target.shape
        boundary_weight = 0.15 if decoder_type == "balanced" else 0.2
        expected = expected + boundary_weight * criterion(
            output["boundary_logits"], boundary_target
        )
    else:
        assert "boundary_logits" not in output and "aux_logits" not in output
    torch.testing.assert_close(loss, expected)
    assert torch.isfinite(loss)
    loss.backward()
    if has_details:
        assert image.grad is not None and image.grad.abs().sum() > 0
    else:
        assert image.grad is None
    assert all(not p.requires_grad and p.grad is None for p in model.img_encoder.parameters())
    assert all(not p.requires_grad and p.grad is None for p in model.predictor.parameters())
    missing = [
        name for name, p in model.decoder.named_parameters() if p.requires_grad and p.grad is None
    ]
    assert not missing, f"Unused trainable decoder parameters: {missing}"
    assert all(
        torch.isfinite(p.grad).all() for p in model.decoder.parameters() if p.grad is not None
    )
    if masked:
        for projection in (
            model.decoder.encoder_prior_projection,
            model.decoder.masked_prediction_projection,
        ):
            assert any(
                p.grad is not None and p.grad.abs().sum() > 0 for p in projection.parameters()
            )
    if has_details:
        assert torch.allclose(output["level_weights"].sum(), torch.tensor(1.0))
        with torch.no_grad():
            levels = model.img_encoder(image)
            evidence = torch.cat(levels, dim=-1)
            context, selected = model._select_mask_indices(levels, evidence)
            assert selected.shape[1] == math.ceil(0.2 * 16)
            assert torch.equal(
                torch.sort(torch.cat([context, selected], dim=1)).values,
                torch.arange(16).expand(2, -1),
            )


@pytest.mark.parametrize("target_kind", ["empty", "full", "mixed"])
def test_equal_weight_binary_loss(target_kind):
    logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    target = torch.zeros_like(logits)
    if target_kind == "full":
        target.fill_(1)
    elif target_kind == "mixed":
        target[:, :, 2:8, 3:9] = 1
    loss = DiceCELoss(sigmoid=True, lambda_ce=1.0, lambda_dice=1.0)(logits, target)
    expected = F.binary_cross_entropy_with_logits(logits, target) + DiceLoss(sigmoid=True)(
        logits, target
    )
    assert torch.allclose(loss, expected)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_default_vits_224_resolution_and_no_predictor_benchmark():
    config = yaml.safe_load(
        (PROJECT / "configs/segmentation/vit_small.yaml").read_text(encoding="utf-8")
    )
    config["model"]["pretrained_path"] = None
    model = evaluation.build_model(config, torch.device("cpu")).eval()
    captured = {}

    def capture_encoder(_module, _args, output):
        captured["levels"] = output

    def capture_predictor(_module, _args, kwargs):
        captured["targets"] = kwargs["masks"].shape[1]

    handles = [
        model.img_encoder.register_forward_hook(capture_encoder),
        model.predictor.register_forward_pre_hook(capture_predictor, with_kwargs=True),
    ]
    sizes = []
    decoder = model.decoder
    handles.extend(
        stage.register_forward_hook(lambda _m, _a, out: sizes.append(out.shape[-2:]))
        for stage in (decoder.stage_28, decoder.stage_56, decoder.stage_112, decoder.stage_224)
    )
    image = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        output = model(image)
    for handle in handles:
        handle.remove()
    assert output["logits"].shape == (1, 1, 224, 224)
    assert output["boundary_logits"].shape == output["logits"].shape
    assert "aux_logits" not in output
    assert len(captured["levels"]) == 4
    assert all(level.shape == (1, 256, 384) for level in captured["levels"])
    assert captured["targets"] == 52
    assert sizes == [(28, 28), (56, 56), (112, 112), (224, 224)]

    def unexpected_predictor(*_args, **_kwargs):
        pytest.fail("NoPredictorForward must bypass the frozen predictor")

    handle = model.predictor.register_forward_pre_hook(unexpected_predictor)
    with torch.inference_mode():
        no_predictor = NoPredictorForward(model)(image)
    handle.remove()
    assert no_predictor.shape == output["logits"].shape
    assert torch.isfinite(no_predictor).all()


def test_all_supplied_configs_and_builders(monkeypatch):
    from predseg.segmentation.helper import init_visual_segmodel
    import predseg.segmentation.helper as helper

    monkeypatch.setattr(helper, "init_visual_segmodel", lambda **kwargs: kwargs)
    for path in (PROJECT / "configs/segmentation").glob("*.yaml"):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        kwargs = evaluation.build_model(config, torch.device("cpu"))
        inspect.signature(init_visual_segmodel).bind(**kwargs)
        assert "segmentation" in config["logging"]["folder"]
        architecture = train.decoder_architecture(config["model"])
        assert architecture["decoder_type"] == "balanced"
        assert architecture["detail_channels"] == [48, 64, 96, 128]
        assert architecture["deep_supervision"] is True
        assert architecture["segmentation_loss"] == {
            "name": "bce_dice_main_auxiliary_boundary",
            "bce_weight": 1.0,
            "dice_weight": 1.0,
            "auxiliary_weights": [0.2],
            "boundary_weight": 0.15,
        }


def test_real_training_loop_checkpoint_reload_and_resume(tmp_path, monkeypatch):
    config = yaml.safe_load(
        (PROJECT / "configs/segmentation/vit_small.yaml").read_text(encoding="utf-8")
    )
    config["data"].update(input_size=56, batch_size=2, pin_mem=False)
    config["model"].update(
        pretrained_path=None,
        balanced_decoder_hidden_dim=32,
        balanced_decoder_num_heads=4,
        balanced_decoder_fusion_depth=1,
        balanced_decoder_channels=[32, 24, 16, 8],
        balanced_detail_channels=[8, 12, 16, 24],
        balanced_decoder_dropout=0.0,
    )
    config["meta"].update(
        model_name="vit_tiny",
        pred_depth=2,
        pred_emb_dim=48,
        use_bfloat16=False,
        skip_completed=False,
        load_checkpoint=False,
    )
    config["optimization"].update(epochs=1, warmup=0)
    config["logging"].update(folder=str(tmp_path), checkpoint_freq=0)
    config["early_stopping"].update(min_epochs=1, patience=20)
    samples = [
        {"pixel_values": torch.randn(3, 56, 56), "label": (torch.rand(1, 56, 56) > 0.8).float()}
        for _ in range(2)
    ]
    sampler = DistributedSampler(samples, num_replicas=1, rank=0, seed=7)
    loader = DataLoader(samples, batch_size=2, sampler=sampler)
    monkeypatch.setattr(train, "make_dataset", lambda **_kwargs: (loader, sampler))
    monkeypatch.setattr(train, "log_timings", False)
    result = train.train_one_dataset(config, "synthetic", 1, 0, torch.device("cpu"))
    assert result["completed_epochs"] == 1
    assert math.isfinite(result["final_loss"])
    checkpoint = torch.load(result["best_checkpoint"], weights_only=True)
    model = evaluation.build_model(config, torch.device("cpu")).eval()
    evaluation.load_model_state(model, checkpoint)
    with torch.inference_mode():
        assert model(torch.randn(1, 3, 56, 56))["logits"].shape == (1, 1, 56, 56)
    # Strict loading rejects a state missing restored prediction heads.
    old_state = copy.copy(checkpoint["decoder"])
    del old_state["boundary_head.0.weight"]
    with pytest.raises(RuntimeError, match="Missing key"):
        evaluation.load_model_state(model, {"decoder": old_state})

    config["meta"]["load_checkpoint"] = True
    config["optimization"]["epochs"] = 2
    resumed = train.train_one_dataset(config, "synthetic", 1, 0, torch.device("cpu"))
    assert resumed["completed_epochs"] == 2
    assert resumed["resumed_from"] == result["latest_checkpoint"]
