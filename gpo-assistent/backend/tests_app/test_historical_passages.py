from fastapi.testclient import TestClient
from gpo_app.api import create_app
from gpo_app.historical_passages import evidence_windows, rank_evidence

DOC = dict(
    advice_date=None,
    sha256="0" * 64,
    source_url="https://onderwijsgeschillen.nl/test.pdf",
)


def test_unknown_headings_never_turn_party_statements_into_evidence():
    original = [
        (
            1,
            "Standpunt verweerder\nDe school vindt dat deze leerling geen ondersteuning nodig heeft.",
        )
    ]
    assert evidence_windows("100", DOC, original) == []


def test_quote_requires_registered_id_and_explicit_committee_heading():
    original = [
        (
            2,
            "4. OVERWEGINGEN VAN DE COMMISSIE\nDe commissie heeft het verslag van het overleg in deze zaak meegewogen.",
        )
    ]
    raw = evidence_windows("100", DOC, original)
    citation = raw[0]["citations"][0]
    assert citation["source_url"].endswith("#page=2")
    assert evidence_windows("100", DOC, original, {citation["source_id"]}) == raw
    assert evidence_windows("100", DOC, original, {"unknown"}) == []


def test_no_supported_case_returns_auditable_abstention():
    result = rank_evidence(
        "Hoe woog de commissie het overleg?", ["100"], lambda case: [], 3
    )
    assert result["status"] == "insufficient_verified_evidence"
    assert result["candidates"] == [{"case_number": "100", "verified_windows": 0}]


def test_intake_works_before_comparison_models_or_registry():
    with TestClient(create_app(object())) as client:
        path = "/api/v1/vergelijk"
        for question, status in [
            ("Tot wanneer kan ik bezwaar maken?", "needs_additional_sources"),
            ("De school heeft iets besloten. Wat nu?", "needs_clarification"),
        ]:
            response = client.post(path, json={"question": question})
            assert response.status_code == 200
            assert response.json()["status"] == status
