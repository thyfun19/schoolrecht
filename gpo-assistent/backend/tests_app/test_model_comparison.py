"""Synthetic numerical checks; no model download, document DB or expected-case tuning."""

import json
from pathlib import Path

import numpy as np
import pytest

from gpo_app.model_assets import acquire
from gpo_app.model_comparison import case_ranking, frozen, metrics, pool


def test_masked_pool_excludes_padding_before_normalization():
    hidden = np.asarray([[[1, 0], [0, 1], [999, 999]]], dtype=np.float32)
    mask = np.asarray([[1, 1, 0]], dtype=np.int64)
    actual = pool(hidden, mask, "masked_mean")
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, [[2**-0.5, 2**-0.5]], atol=1e-7)
    np.testing.assert_array_equal(pool(hidden, mask, "cls"), [[1, 0]])
    with pytest.raises(ValueError):
        pool(hidden, np.zeros_like(mask), "masked_mean")
    with pytest.raises(ValueError):
        pool(hidden, mask, "unknown")


def test_case_ranking_uses_maximum_and_stable_ties():
    chunks = [{"case": c} for c in ["b", "a", "b", "c"]]
    assert case_ranking(chunks, [0.5, 0.5, 0.1, 0]) == ["a", "b", "c"]
    assert case_ranking(chunks, [0.5, 0.5, 0.1, 0], True) == ["a", "b"]


def test_complete_context_does_not_inject_missing_case_or_cross_page_anchor():
    contexts = {
        c: {
            "parts": {
                "commissieoverwegingen": [{"page_number": 1, "text": "anker"}],
                "conclusies": [],
            }
        }
        for c in ["a", "b"]
    }
    gold = [{"case": "b", "page": 1, "anchor": "anker"}]
    result = metrics(["a"], None, contexts, gold)
    assert result["full_anchor_recall"] == 0 and result["missing_cases"] == ["b"]
    assert metrics(["b"], None, contexts, gold)["full_anchor_recall"] == 1
    assert (
        metrics(["b"], None, contexts, [{"case": "b", "page": 2, "anchor": "anker"}])[
            "full_anchor_recall"
        ]
        == 0
    )
    assert metrics(["b"], None, contexts, []) is None


def test_changed_frozen_question_is_rejected_before_model_execution(tmp_path):
    (tmp_path / "questions.json").write_text("changed")
    (tmp_path / "frozen.json").write_text(
        json.dumps({"hashes": {"questions.json": "0" * 64}})
    )
    with pytest.raises(ValueError, match="Bevroren"):
        frozen(tmp_path)


def test_bad_existing_asset_is_not_overwritten_and_windows_lock_closes(tmp_path):
    path = tmp_path / "e5"
    path.mkdir()
    asset = path / "model.onnx"
    asset.write_bytes(b"bad")
    with pytest.raises(ValueError, match="niet overschrijven"):
        acquire("e5", tmp_path)
    assert asset.read_bytes() == b"bad"
    assert not (path / "download.lock").exists()
