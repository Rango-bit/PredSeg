"""Generate modality token features from local text-model checkpoints."""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch


def modality_texts(path):
    frame = pd.read_csv(path)
    if not {"modality_id", "modality_text"}.issubset(frame):
        raise ValueError("Modality CSV requires modality_id and modality_text")
    frame = frame.sort_values("modality_id")
    if frame["modality_id"].tolist() != list(range(len(frame))):
        raise ValueError("Modality IDs must be unique contiguous integers starting at zero")
    if (
        frame.empty
        or frame["modality_text"].isna().any()
        or frame["modality_text"].str.strip().eq("").any()
    ):
        raise ValueError("Modality descriptions must be nonempty strings")
    return frame["modality_text"].tolist()


def encode_texts(texts, encoder, model_path, tokenizer_path=None, device="cpu", max_length=77):
    """Expose the same last-hidden-state token features used in pretraining."""
    from transformers import AutoModel, AutoTokenizer, CLIPTextModel

    tokenizer_path = tokenizer_path or model_path
    if encoder == "biomedclip":
        from open_clip import create_model
        from open_clip.factory import _MODEL_CONFIGS

        with open(Path(model_path) / "open_clip_config.json", encoding="utf-8") as file:
            model_config = json.load(file)["model_cfg"]
        model_config["text_cfg"]["hf_model_name"] = str(tokenizer_path)
        model_config["text_cfg"]["hf_tokenizer_name"] = str(tokenizer_path)
        # Preserve the local BiomedCLIP loading route used by the research implementation.
        _MODEL_CONFIGS["predseg_biomedclip_local"] = model_config
        model = (
            create_model(
                "predseg_biomedclip_local",
                pretrained=str(Path(model_path) / "open_clip_pytorch_model.bin"),
            )
            .to(device)
            .eval()
        )
        text_model = model.text.transformer
    else:
        model_class = CLIPTextModel if encoder == "clip" else AutoModel
        model = (
            model_class.from_pretrained(str(model_path), local_files_only=True).to(device).eval()
        )
        text_model = model
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    features = []
    with torch.inference_mode():
        for text in texts:
            tokens = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            # CLIP text models do not consume BERT-style token_type_ids.
            if encoder == "clip":
                tokens.pop("token_type_ids", None)
            output = text_model(**tokens, return_dict=True)
            features.append(output.last_hidden_state[0].float().cpu())
    return torch.stack(features)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modalities", required=True)
    parser.add_argument("--encoder", choices=["biomedclip", "bert", "clip"], required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-length", type=int, default=77)
    args = parser.parse_args()
    if args.max_length < 1:
        parser.error("--max-length must be positive")
    features = encode_texts(
        modality_texts(args.modalities),
        args.encoder,
        args.model_path,
        args.tokenizer_path,
        args.device,
        args.max_length,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features, output)
    print(f"Saved {tuple(features.shape)} modality token features to {output}")


if __name__ == "__main__":
    main()
