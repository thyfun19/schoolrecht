from fastapi.testclient import TestClient

from gpo_app.api import create_app
from gpo_app.ingestion import import_prepared


def test_new_workflow_routes_require_review_before_writing(engine, prepared):
    doc = next(d for d in prepared if d.document["case_number"] == "110222")
    import_prepared(engine, [doc])
    with TestClient(create_app(engine)) as client:
        r = client.post(
            "/api/v1/evidence-plans/prepare",
            json=dict(
                question="Welke ondersteuning onderzocht de school vóór weigering?",
                processing_ids=[doc.processing["processing_id"]],
            ),
        )
        assert r.status_code == 200
        workflow = r.json()
        case = workflow["baseline"]["case_packets"][0]
        rule = next(
            r
            for r in workflow["selection_requests"][0]["selection_policy"]
            if r["source_id"] == "src_16378b3d0cb0d7bf96c2a163ed01bb58"
        )
        assert rule["function"] == "reference_candidate"
        payload = dict(
            case_source_id=case["case_markers"][0],
            relevance="insufficient",
            issue_match="related_issue",
            answered_question_part="De concrete onderzoekshandelingen.",
            relevance_reason="Synthetische fixture, geen inhoudelijk oordeel.",
            issue_source_ids=case["case_markers"],
            rule_source_ids=[],
            application_source_ids=[],
            condition_source_ids=[],
            difference_source_ids=[],
            missing_question_parts=["Deze fixture beantwoordt de vraag niet."],
        )
        r = client.post(
            "/api/v1/evidence-plans/validate",
            json=dict(workflow=workflow, payloads=[payload]),
        )
        assert r.status_code == 200
        validated = r.json()
        r = client.post(
            "/api/v1/evidence-plans/writing/prepare",
            json=dict(workflow=workflow, validated=validated, reviews={}),
        )
        assert r.status_code == 422
        record = validated["plans"][0]
        review = {
            record["plan_id"]: dict(
                verdict="supported",
                reviewer="Synthetic integration test",
                note="Not content validation.",
                checks=dict(
                    question_match=True,
                    source_support=True,
                    concrete_application=True,
                    conditions_and_differences=True,
                ),
            )
        }
        r = client.post(
            "/api/v1/evidence-plans/writing/prepare",
            json=dict(workflow=workflow, validated=validated, reviews=review),
        )
        assert r.status_code == 200 and r.json()["status"] == "insufficient_evidence"
        validated["plans"][0]["evidence"]["issue_source_ids"][0]["quote"] = "Verzonnen"
        assert (
            client.post(
                "/api/v1/evidence-plans/writing/prepare",
                json=dict(workflow=workflow, validated=validated, reviews=review),
            ).status_code
            == 422
        )


def test_reviewed_selection_to_writer_and_deterministic_output(engine, prepared):
    doc = next(d for d in prepared if d.document["case_number"] == "110222")
    import_prepared(engine, [doc])
    with TestClient(create_app(engine)) as client:
        workflow = client.post(
            "/api/v1/evidence-plans/prepare",
            json=dict(
                question="Welk onderzoek droeg de weigering wegens ondersteuning?",
                processing_ids=[doc.processing["processing_id"]],
            ),
        ).json()
        case = workflow["baseline"]["case_packets"][0]
        application = next(
            s["source_id"]
            for s in case["sources"]
            if "aangetoond gedegen onderzoek" in s["quote"]
        )
        condition = next(
            s["source_id"]
            for s in case["sources"]
            if "onderwijsopbouw en zorgafbouw" in s["quote"]
        )
        payload = dict(
            case_source_id=case["case_markers"][0],
            relevance="direct",
            issue_match="same_issue",
            answered_question_part="Het concrete onderzoek.",
            relevance_reason="Synthetic integration fixture.",
            issue_source_ids=case["case_markers"],
            rule_source_ids=[],
            application_source_ids=[application],
            condition_source_ids=[condition],
            difference_source_ids=[],
            missing_question_parts=["De persoonlijke situatie is niet beoordeeld."],
        )
        response = client.post(
            "/api/v1/evidence-plans/validate",
            json=dict(workflow=workflow, payloads=[payload]),
        )
        assert response.status_code == 200
        validated = response.json()
        review = {
            validated["plans"][0]["plan_id"]: dict(
                verdict="supported",
                reviewer="Synthetic integration fixture",
                note="Mechanism test, no independent content validation.",
                checks=dict(
                    question_match=True,
                    source_support=True,
                    concrete_application=True,
                    conditions_and_differences=True,
                ),
            )
        }
        response = client.post(
            "/api/v1/evidence-plans/writing/prepare",
            json=dict(workflow=workflow, validated=validated, reviews=review),
        )
        assert response.status_code == 200
        writer = response.json()
        body = dict(
            cases=[
                dict(
                    status="relevant_contribution",
                    case_source_id=writer["case_markers"][0],
                    answers_question_part="De beoordeling van het concrete onderzoek.",
                    relevance_explanation="Synthetic fixture.",
                    finding=dict(
                        text="In deze zaak vond de commissie dat gedegen onderzoek was gedaan.",
                        source_ids=[application],
                    ),
                    supporting_arguments=[],
                    limitations=[
                        dict(
                            text="Onderwijsopbouw en zorgafbouw speelden een rol.",
                            source_ids=[condition],
                        )
                    ],
                )
            ],
            unanswered_question_parts=["De persoonlijke situatie is niet beoordeeld."],
        )
        response = client.post(
            "/api/v1/evidence-plans/writing/validate",
            json=dict(
                workflow=workflow, validated=validated, reviews=review, payload=body
            ),
        )
        assert response.status_code == 200
        result = response.json()
        assert result["case_assessments"][0]["case_outcome"]["origin"] == "backend"
        assert result["case_assessments"][0]["finding"]["citations"][0]["page"] == 6
        assert result["answer_quality"] == "not_validated"
