from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from gpo_local.api import create_app

BASE = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not (BASE / "data/e5/bundle.json").exists(),
    reason="Importeer eerst de lokale model-/broncache; geen download in tests.",
)


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as c:
        yield c


def test_actual_index_health(client):
    assert client.get("/health").json() == dict(
        status="ok",
        documents=10,
        fragments=166,
        answer_generation=False,
        paid_api_enabled=False,
    )


@pytest.mark.parametrize("method", ["text", "vector"])
def test_search_preserves_source_roles_and_full_reasoning(client, method):
    r = client.post(
        "/search",
        json=dict(
            question="Moest de school een OPP met ouders evalueren?", method=method
        ),
    )
    assert r.status_code == 200
    data = r.json()
    assert data["answerability"] == "not_assessed"
    assert data["score_is_answerability"] is False
    for case in data["results"]:
        assert case["case_number"] and case["advice_date"]
        assert case["parts"]["commissieoverwegingen"]
        for s in case["parts"]["commissieoverwegingen"]:
            assert s["role"] == "commissie" and s["page"] >= 1
            assert s["url"] == case["source_url"] + f"#page={s['page']}"
        assert all(s["role"] != "commissie" for s in case["parts"]["partijstandpunten"])


@pytest.mark.parametrize(
    "q",
    [
        "De school heeft mijn kind geweigerd. Tot welke datum kan ik nu bezwaar maken en waar moet ik het indienen?",
        "Kan ik met deze adviezen schadevergoeding krijgen voor de maanden dat mijn kind geen les kreeg?",
    ],
)
def test_actual_boundary_preparation(client, q):
    r = client.post("/explanation/prepare", json=dict(question=q, method="vector"))
    assert r.status_code == 200
    assert r.json()["status"] == "needs_additional_sources"
    assert r.json()["paid_calls"] == 0 and r.json()["request"] is None


def test_request_validation_and_no_generation_route(client):
    assert client.post("/search", json={"question": "   "}).status_code == 422
    assert (
        client.post(
            "/search", json={"question": "Ondersteuning", "method": "hybrid"}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/search", json={"question": "Ondersteuning", "expected_case": "46731"}
        ).status_code
        == 422
    )
    assert (
        client.post("/explanation", json={"question": "Ondersteuning"}).status_code
        == 404
    )


def test_overlong_token_input_is_rejected_without_truncation(client):
    assert (
        client.post(
            "/search", json={"question": "x " * 900, "method": "vector"}
        ).status_code
        == 422
    )


def test_query_is_independent_of_batch_companions_order_and_padding(client):
    engine = client.app.state.engine
    query = "We hebben vaak gesprekken met school, maar ik krijg het OPP pas laat en er is geen duidelijke evaluatie met mij. Zijn er vergelijkbare zaken?"
    shorter = "Wie maakte het OPP?"
    longer = "Een leerling krijgt extra ondersteuning. " * 30
    alone = engine.embed_query(query)
    for group in [
        [query, query],
        [query, shorter],
        [query, longer],
        [longer, query],
        [shorter, longer, query],
    ]:
        vectors = engine.embed_queries(group)
        for i, q in enumerate(group):
            if q == query:
                assert np.array_equal(vectors[i], alone)
    original = client.post("/search", json=dict(question=query, method="vector")).json()
    for group in [[query, shorter], [longer, query], [query, query]]:
        batch = client.post(
            "/search/batch",
            json={"questions": [dict(question=q, method="vector") for q in group]},
        ).json()
        assert all(batch[i] == original for i, q in enumerate(group) if q == query)
