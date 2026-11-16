"""Command-line entry points for the full PredSeg workflow."""

import argparse
import json
import logging

from predseg.config import load_config, validate_config


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="PredSeg: pretraining and medical image segmentation"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, description in [
        ("pretrain", "Pretrain the encoder and predictor"),
        ("segment", "Train one frozen-backbone decoder per dataset"),
        ("evaluate", "Evaluate one dataset-specific decoder"),
        ("validate", "Check a YAML configuration without training"),
    ]:
        command = commands.add_parser(name, help=description)
        command.add_argument("--config", required=True, help="Path to a YAML configuration")
        command.add_argument(
            "--set",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="Override a dotted YAML option; repeat as needed",
        )
        if name in ("pretrain", "segment", "evaluate"):
            command.add_argument(
                "--devices", nargs="+", default=None, help="cpu or cuda:0 cuda:1 (default: cuda:0)"
            )
        if name == "evaluate":
            command.add_argument("--dataset-name", required=True)
            command.add_argument("--checkpoint", default=None)
            command.add_argument("--output-dir", default=None)
            command.add_argument("--batch-size", type=int, default=None)
            command.add_argument("--num-workers", type=int, default=None)
            command.add_argument("--threshold", type=float, default=None)
            command.add_argument("--save-predictions", action="store_true")
        if name == "validate":
            command.add_argument("--check-files", action="store_true")
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(arguments.config, arguments.set)
    stage = (
        "pretrain"
        if arguments.command == "pretrain"
        else (None if arguments.command == "validate" else "segment")
    )
    summary = validate_config(config, stage, check_files=False)
    if arguments.command == "validate":
        validate_config(config, check_files=arguments.check_files)
        print(json.dumps(summary, indent=2))
    elif arguments.command == "evaluate":
        from predseg.segmentation.evaluate import evaluate

        arguments.devices = arguments.devices or ["cuda:0"]
        config["data"]["dataset_name"] = arguments.dataset_name
        evaluate(config, arguments)
    else:
        from predseg.launcher import launch

        launch(config, arguments.command, arguments.devices)
