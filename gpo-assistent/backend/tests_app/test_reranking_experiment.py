from types import SimpleNamespace

import numpy as np
import pytest
from tokenizers import Tokenizer

from gpo_app.reranking_experiment import LocalReranker, choose_pairs, rank_cases
from conftest import BASE


def test_two_representations_share_candidates_without_injecting_other_cases():
    primary = [
        dict(
            id="p1",
            case="a",
            text="overweging",
            page=1,
            role="commissie",
            section="OVERWEGINGEN",
        ),
        dict(
            id="p2",
            case="b",
            text="ander",
            page=2,
            role="commissie",
            section="OVERWEGINGEN",
        ),
    ]
    aux = dict(
        id="aux_1",
        case="a",
        text="ouderstandpunt",
        page=3,
        role="ouders",
        section="STANDPUNTEN",
    )
    pairs, reps = choose_pairs(
        "vraag",
        ["a"],
        primary,
        primary + [aux],
        [0.5, 0.8],
        [0.1, 0.8, 0.9],
        [0.6, 0.9],
    )
    assert {p["case"] for p in pairs} == {"a"}
    assert set(reps["rerank_primary"]) == set(reps["rerank_with_auxiliary"]) == {"a"}
    assert (
        len(reps["rerank_primary"]["a"]) == 1
        and len(reps["rerank_with_auxiliary"]["a"]) == 2
    )
    assert next(p for p in pairs if p["is_auxiliary"])["role"] == "ouders"
    with pytest.raises(ValueError):
        choose_pairs("vraag", ["a"] * 2, primary, primary, [1, 2], [1, 2], [1, 2])
    with pytest.raises(ValueError):
        choose_pairs("vraag", list("abcdef"), primary, primary, [1, 2], [1, 2], [1, 2])


def test_reranking_accounts_for_every_candidate_and_never_adds_one():
    reps = {"a": ["p1", "p2"], "b": ["p3"]}
    scores = {"p1": -1, "p2": 4, "p3": 3, "outside": 99}
    ranking, maxima = rank_cases(reps, scores, ["a", "b"])
    assert ranking == ["a", "b"] and maxima == {"a": 4, "b": 3}
    assert rank_cases({"a": ["p1"], "b": ["p1"]}, scores, ["b", "a"])[0] == ["a", "b"]
    with pytest.raises(ValueError):
        rank_cases(reps, scores, ["a", "b", "missing"])
    with pytest.raises(ValueError):
        rank_cases({"a": []}, scores, ["a"])
    with pytest.raises(KeyError):
        rank_cases({"a": ["unknown"]}, scores, ["a"])


class FakeSession:
    called = False

    def get_inputs(self):
        return [SimpleNamespace(name=n) for n in ["input_ids", "attention_mask"]]

    def run(self, outputs, inputs):
        self.called = True
        assert inputs["input_ids"].dtype == inputs["attention_mask"].dtype == np.int64
        return [inputs["attention_mask"].sum(axis=1).astype(np.float32)[:, None]]


def fake_reranker():
    reranker = object.__new__(LocalReranker)
    reranker.tokenizer = Tokenizer.from_file(str(BASE / "data/e5/model/tokenizer.json"))
    reranker.session = FakeSession()
    reranker.output = "logits"
    return reranker


def test_sequence_pair_padding_keeps_each_pair_intact():
    model = fake_reranker()
    pairs = [
        dict(question="Vraag?", passage="Kort."),
        dict(
            question="Vraag?",
            passage="Een langere passage over ondersteuning en voorwaarden.",
        ),
    ]
    actual = model.score(pairs)
    expected = np.asarray([model.score([p])[0] for p in pairs])
    np.testing.assert_array_equal(actual, expected)
    assert len(actual) == 2


def test_overlong_sequence_pair_is_rejected_before_inference():
    model = fake_reranker()
    with pytest.raises(ValueError, match="geen truncatie"):
        model.score([dict(question="vraag", passage="ondersteuning " * 1000)])
    assert not model.session.called
