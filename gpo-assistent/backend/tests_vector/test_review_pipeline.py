import copy
import json
from pathlib import Path

import pytest

from gpo_local.common import read
from gpo_local.retrieval import Retrieval
from gpo_local.source_registry import SourceRegistry
from gpo_local.question_grounding import prepare_question_assessment
from gpo_local.review_pipeline import replay_offline
from gpo_local.check_question_focus import scenario

BASE = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not (BASE / "data/e5/bundle.json").exists(), reason="Lokale broncache ontbreekt."
)


@pytest.fixture(scope="module")
def packets():
    engine = Retrieval(BASE / "data/e5", BASE / "data/gpo.sqlite3")
    registry = SourceRegistry(engine.documents, engine.sections)
    saved = read(BASE / "local-results/grounding-v2-trial/packets.json")
    output = {}
    for item in saved:
        old = item["prepared"]
        cases = list(dict.fromkeys(s["case_number"] for s in old["sources"]))
        # Same full frozen source selection; no expected cases inserted into retrieval.
        question = (
            item["question"]
            if "question" in item
            else json.loads(old["request"]["input"])["question"]
        )
        prepared = prepare_question_assessment(
            question,
            dict(method="vector", results=[dict(case_number=c) for c in cases]),
            registry,
        )
        assert prepared["sources"] == old["sources"]
        output[item["id"]] = (registry, prepared, scenario(item["id"], prepared))
    return output


class ReplayClient:
    def __init__(self, first, second=None, error=None):
        self.first = first
        self.second = second
        self.error = error
        self.calls = []

    def replay(self, stage, context):
        self.calls.append((stage, copy.deepcopy(context)))
        if self.error == stage:
            raise RuntimeError("Synthetic stage failure")
        return copy.deepcopy(self.first if stage == "assessment" else self.second)


def review(assessments):
    return {
        r["assessment_id"]: dict(
            verdict="supported",
            reviewer="Offline reference author, not independent",
            note="Synthetische integratietest; bronrelevantie en beperkingen gecontroleerd voor dit scenario.",
        )
        for r in assessments["case_assessments"]
    }


def comparison(payload):
    contributions = [
        c for c in payload["cases"] if c["status"] == "relevant_contribution"
    ]
    return dict(
        similarities=[],
        differences=[
            dict(
                text="In deze historische zaken verschilden de beoordelingen van het onderzoek.",
                source_ids=[c["finding"]["source_ids"][0] for c in contributions],
            )
        ],
        missing_information=[
            "Deze vergelijking beoordeelt geen nieuwe persoonlijke situatie."
        ],
    )


def test_valid_chain_passes_only_reviewed_contributions_and_conditions(packets):
    registry, prepared, payload = packets["V1"]
    client = ReplayClient(payload, comparison(payload))
    result = replay_offline(prepared, registry, client, review)
    assert [s for s, _ in client.calls] == ["assessment", "comparison"]
    context = client.calls[1][1]
    assert len(context["context"]) == 2
    assert all(c["limitations"] for c in context["context"])
    rejected = [c for c in payload["cases"] if c["status"] == "insufficient_relevance"][
        0
    ]
    assert rejected["case_source_id"] not in str(context)
    expected_ids = {
        sid
        for c in payload["cases"]
        if c["status"] == "relevant_contribution"
        for statement in [c["finding"]] + c["limitations"]
        for sid in statement["source_ids"]
    }
    assert {s["source_id"] for s in context["sources"]} == expected_ids
    assert (
        result["comparison"]["answer_quality"] == "not_validated"
        and result["paid_calls"] == 0
    )


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "duplicate",
        "unexpected_case",
        "swapped_marker",
        "unknown_source",
        "outside_packet",
        "other_case_source",
        "review_rejected",
        "review_missing",
    ],
)
def test_bad_first_stage_makes_zero_comparison_calls(packets, fault):
    registry, prepared, original = packets["V1"]
    payload = copy.deepcopy(original)
    first = payload["cases"][0]
    other = next(c for c in payload["cases"] if c is not first)
    if fault == "missing":
        payload["cases"].pop()
    if fault == "duplicate":
        payload["cases"][-1] = copy.deepcopy(first)
    if fault == "unexpected_case":
        first["case_source_id"] = registry.select(["109206"])[0]["source_id"]
    if fault == "swapped_marker":
        first["case_source_id"], other["case_source_id"] = (
            other["case_source_id"],
            first["case_source_id"],
        )
    relevant = next(
        c for c in payload["cases"] if c["status"] == "relevant_contribution"
    )
    if fault == "unknown_source":
        relevant["finding"]["source_ids"] = ["src_unknown"]
    if fault == "outside_packet":
        relevant["finding"]["source_ids"] = [
            registry.select(["109206"])[0]["source_id"]
        ]
    if fault == "other_case_source":
        relevant["finding"]["source_ids"] = [
            next(c["case_source_id"] for c in payload["cases"] if c is not relevant)
        ]

    def decision(a):
        reviews = review(a)
        if fault == "review_rejected":
            next(iter(reviews.values()))["verdict"] = "unsupported"
        if fault == "review_missing":
            reviews.pop(next(iter(reviews)))
        return reviews

    client = ReplayClient(payload, comparison(original))
    with pytest.raises(ValueError):
        replay_offline(prepared, registry, client, decision)
    assert [s for s, _ in client.calls] == ["assessment"]


def test_comparison_cannot_reintroduce_rejected_case(packets):
    registry, prepared, payload = packets["V1"]
    second = comparison(payload)
    rejected = next(
        c for c in payload["cases"] if c["status"] == "insufficient_relevance"
    )
    second["differences"][0]["source_ids"].append(rejected["basis_source_ids"][0])
    client = ReplayClient(payload, second)
    with pytest.raises(ValueError):
        replay_offline(prepared, registry, client, review)
    assert [s for s, _ in client.calls] == ["assessment", "comparison"]


@pytest.mark.parametrize("stage", ["assessment", "comparison"])
def test_failure_has_no_retry(packets, stage):
    registry, prepared, payload = packets["V1"]
    client = ReplayClient(payload, comparison(payload), error=stage)
    with pytest.raises(RuntimeError):
        replay_offline(prepared, registry, client, review)
    assert [s for s, _ in client.calls].count(stage) == 1


def test_n3_frozen_vector_packet_has_no_relevant_comparison(packets):
    registry, prepared, payload = packets["N3"]
    client = ReplayClient(payload)
    result = replay_offline(prepared, registry, client, review)
    assert result["comparison"] is None
    assert [s for s, _ in client.calls] == ["assessment"]
    assert result["assessments"]["evidence_status"] == "insufficient_evidence"
