import copy
import json

import pytest
from fastapi.testclient import TestClient

from gpo_app.api import create_app
from gpo_app.ingestion import import_prepared
from gpo_app.isolated_assessment import prepare, validate


def test_full_registered_context_and_new_http_entry_points(engine, prepared):
    docs = [d for d in prepared if d.document["case_number"] in ["110222", "126525"]]
    import_prepared(engine, docs)
    ids = [d.processing["processing_id"] for d in docs]
    with TestClient(create_app(engine)) as client:
        response = client.post(
            "/api/v1/assessments/prepare",
            json=dict(
                question="Welke ondersteuning onderzocht de school vóór de weigering?",
                processing_ids=ids,
            ),
        )
        assert response.status_code == 200
        packet = response.json()
        assert len(packet["case_packets"]) == 2 and packet["request"] is None
        outputs = []
        for item in packet["case_packets"]:
            context = json.loads(item["request"]["input"])["context"]
            assert len(context["sections"]) >= 2
            assert {s["category"] for s in context["sections"]} == {
                "commissieoverwegingen",
                "conclusies",
            }
            if item["case_number"] == "110222":
                assert "onderwijsopbouw en zorgafbouw" in item["request"]["input"]
                assert "gedegen onderzoek" in item["request"]["input"]
            outputs.append(
                dict(
                    status="insufficient_relevance",
                    case_source_id=item["case_markers"][0],
                    reason="Synthetische afwijzing, geen echt relevantieoordeel.",
                    question_part_not_answered="De concrete onderzoekshandelingen.",
                    basis_source_ids=item["case_markers"],
                )
            )
        payload = dict(
            cases=outputs,
            unanswered_question_parts=[
                "Dit synthetische voorbeeld beantwoordt de vraag niet."
            ],
        )
        result = client.post(
            "/api/v1/assessments/validate", json=dict(prepared=packet, payload=payload)
        )
        assert result.status_code == 200
        assert len(result.json()["case_assessments"]) == 2
        assert result.json()["evidence_status"] == "insufficient_evidence"
        assert not result.json()["comparison_allowed"]
        outputs[0]["basis_source_ids"] = packet["case_packets"][1]["case_markers"]
        assert (
            client.post(
                "/api/v1/assessments/validate",
                json=dict(prepared=packet, payload=payload),
            ).status_code
            == 422
        )
        altered = copy.deepcopy(packet)
        altered["case_packets"][0]["request"]["input"] = "Gewijzigd."
        with pytest.raises(ValueError):
            validate(engine, payload, altered)


def test_intake_precedes_database_access():
    result = prepare(None, "De school wil mijn kind niet meer.", [])
    assert result["status"] == "needs_clarification"
    result = prepare(None, "Hoeveel dagen heb ik nog om ertegen in te gaan?", [])
    assert result["status"] == "needs_additional_sources"
