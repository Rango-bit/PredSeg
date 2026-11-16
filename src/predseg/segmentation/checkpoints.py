"""Validate a dataset decoder against its architecture and frozen backbone."""

from predseg.segmentation.helper import backbone_configuration


def validate_segmentation_checkpoint(checkpoint, config):
    from predseg.segmentation.train import decoder_architecture

    expected = decoder_architecture(config["model"])
    expected["backbone"] = backbone_configuration(config)
    if checkpoint.get("decoder_architecture") != expected:
        raise ValueError(
            "Segmentation checkpoint architecture does not match the configuration. "
            "Use the original backbone, decoder, and fusion settings. Legacy decoder "
            "checkpoints without backbone metadata require retraining or explicit migration."
        )
    dataset = config["data"]["dataset_name"]
    if checkpoint.get("dataset_name") != dataset:
        raise ValueError(
            f"Checkpoint belongs to {checkpoint.get('dataset_name')!r}, not {dataset!r}"
        )
