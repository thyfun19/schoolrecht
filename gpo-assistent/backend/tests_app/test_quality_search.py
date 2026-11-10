from pathlib import Path
import builtins

import pytest
from fastapi.testclient import TestClient

from gpo_app.api import create_app
from gpo_app.schemas import CandidateSearchRequest
from gpo_app.quality_search import CandidateSearch


def test_intake_precedes_loading_search_models(monkeypatch):
    def forbidden(*args):
        raise AssertionError("Search must not run before clarification.")

    monkeypatch.setattr("gpo_app.quality_search.candidate_search", forbidden)
    with TestClient(create_app(object())) as client:
        result = client.post(
            "/api/v1/evidence-plans/search/prepare",
            json=dict(question="De school wil mijn kind niet meer."),
        )
        assert (
            result.status_code == 200
            and result.json()["status"] == "needs_clarification"
        )
        assert (
            client.post(
                "/api/v1/evidence-plans/search/prepare",
                json=dict(question="Welke ondersteuning?", expected_case="112994"),
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/v1/evidence-plans/search/prepare", json=dict(question="   ")
            ).status_code
            == 422
        )


def test_candidate_limit_and_method_are_validated():
    with pytest.raises(ValueError):
        CandidateSearchRequest(question="Welke ondersteuning?", limit=6)
    with pytest.raises(ValueError):
        CandidateSearchRequest(question="Welke ondersteuning?", method="favorite_cases")


def test_missing_optional_search_dependencies_return_clear_status(monkeypatch):
    original = builtins.__import__

    def without_search(name, *args, **kwargs):
        if name == "quality_search":
            raise ImportError("Synthetic missing optional dependency.")
        return original(name, *args, **kwargs)

    with TestClient(create_app(object())) as client:
        monkeypatch.setattr(builtins, "__import__", without_search)
        assert (
            client.post(
                "/api/v1/evidence-plans/search/prepare",
                json=dict(question="De school wil mijn kind niet meer."),
            ).status_code
            == 200
        )
        response = client.post(
            "/api/v1/evidence-plans/search/prepare",
            json=dict(question="Welke ondersteuning onderzocht de school?"),
        )
        assert (
            response.status_code == 503
            and "DependencyPath" in response.json()["detail"]
        )


@pytest.mark.skipif(
    not (
        Path(__file__).resolve().parents[1]
        / "data/model-comparison/models/bge/model.onnx"
    ).exists(),
    reason="Pinned local model cache missing; no download in tests.",
)
def test_actual_hybrid_candidate_search_reproducible_and_bound_to_processing_ids():
    search = CandidateSearch("hybrid_bge")
    question = "De school heeft alleen gezegd dat de situatie onveilig is. Hoe beoordeelde de commissie eerder een verwijderingsbrief zonder concrete onderbouwing?"
    first = search.search(question, 3)
    assert first == search.search(question, 3)
    assert (
        len(first["cases"])
        == len(first["processing_ids"])
        == len(set(first["cases"]))
        == 3
    )
    assert first["processing_ids"] == [
        search.contexts[c]["processing_id"] for c in first["cases"]
    ]
    assert (
        first["answer_generated"] is False and first["score_is_answerability"] is False
    )
