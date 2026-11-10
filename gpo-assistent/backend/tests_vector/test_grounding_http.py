from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gpo_local.api import create_app

BASE = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not (BASE / "data/e5/bundle.json").exists(), reason="Lokale broncache ontbreekt."
)


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as c:
        yield c


def example(client):
    request = dict(
        question="Welke onderzoeken deed een school naar de ondersteuning?",
        method="text",
        limit=3,
    )
    prepared = client.post("/explanation/prepare", json=request).json()
    grouped = {}
    for source in prepared["sources"]:
        grouped.setdefault(source["case_number"], source)
    payload = dict(
        cases=[
            dict(
                status="insufficient_relevance",
                case_source_id=s["source_id"],
                reason="Synthetische afbakeningstest; geen werkelijke relevantiebeoordeling.",
                question_part_not_answered="Welke onderzoeken de concrete ondersteuning onderbouwen.",
                basis_source_ids=[s["source_id"]],
            )
            for s in grouped.values()
        ],
        unanswered_question_parts=[
            "Deze synthetische test bepaalt geen werkelijk antwoord op de vraag."
        ],
    )
    return request, prepared, payload


def test_http_inserts_canonical_citations_and_keeps_claim_unreviewed(client):
    request, prepared, payload = example(client)
    response = client.post(
        "/explanation/validate", json=dict(**request, payload=payload)
    )
    assert response.status_code == 200
    data = response.json()
    assert data["answer_quality"] == "not_validated"
    lookup = {s["source_id"]: s for s in prepared["sources"]}
    for case in data["case_assessments"]:
        assert case["semantic_support"] == "requires_review"
        for citation in case["citations"]:
            assert citation == lookup[citation["source_id"]]
            assert citation["case_number"] == case["case_number"]


def test_http_rejects_unknown_reference_and_model_supplied_quote(client):
    request, _, payload = example(client)
    payload["cases"][0]["basis_source_ids"] = ["unknown"]
    assert (
        client.post(
            "/explanation/validate", json=dict(**request, payload=payload)
        ).status_code
        == 422
    )
    request, _, payload = example(client)
    payload["cases"][0]["quote"] = "Verzonnen citaat"
    assert (
        client.post(
            "/explanation/validate", json=dict(**request, payload=payload)
        ).status_code
        == 422
    )
