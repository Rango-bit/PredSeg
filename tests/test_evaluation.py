"""Metric and data-selection conventions used by public evaluation."""

import math

import numpy as np
import pytest

from predseg.segmentation.evaluate import binary_metrics
from predseg.prepare_text import modality_texts


def test_binary_metric_empty_and_perfect_predictions():
    empty = np.zeros((8, 8), dtype=bool)
    foreground = empty.copy()
    foreground[2:5, 2:5] = True
    both_empty = binary_metrics(empty, empty)
    assert both_empty["dice"] == 1 and both_empty["hausdorff_95"] == 0
    one_empty = binary_metrics(empty, foreground)
    assert one_empty["dice"] == 0 and math.isnan(one_empty["hausdorff_95"])
    perfect = binary_metrics(foreground, foreground)
    assert perfect["dice"] == perfect["iou"] == perfect["pixel_accuracy"] == 1
    assert perfect["hausdorff_95"] == 0


def test_modality_schema_rejects_ambiguous_cache_indices(tmp_path):
    path = tmp_path / "modalities.csv"
    path.write_text(
        "modality_id,modality_text\n0,A radiograph.\n1,A skin image.\n", encoding="utf-8"
    )
    assert modality_texts(path) == ["A radiograph.", "A skin image."]
    path.write_text("modality_id,modality_text\n1,A radiograph.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="contiguous"):
        modality_texts(path)
