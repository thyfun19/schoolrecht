import copy
import pytest

from gpo_app.case_assessment import prepare, validate
from gpo_app.ingestion import import_prepared


def rejected(packet):
    return dict(
        cases=[
            dict(
                status="insufficient_relevance",
                case_source_id=c["case_source_id"],
                reason="Synthetische validatorfixture; geen inhoudelijk relevantieoordeel.",
                question_part_not_answered="De concrete verantwoordelijkheid is niet beoordeeld.",
                basis_source_ids=[c["case_source_id"]],
            )
            for c in packet["case_contexts"]
        ],
        unanswered_question_parts=[
            "De concrete vraag is met deze fixture niet beantwoord."
        ],
    )


def test_full_context_and_complete_case_accounting(engine, prepared):
    docs = [d for d in prepared if d.document["case_number"] in ["110222", "109206"]]
    import_prepared(engine, docs)
    packet = prepare(
        engine,
        "Wie bleef verantwoordelijk voor het OPP?",
        [d.processing["processing_id"] for d in docs],
    )
    assert packet["request"] is None and len(packet["case_contexts"]) == 2
    text = " ".join(
        s["text"]
        for c in packet["case_contexts"]
        if c["case_number"] == "110222"
        for s in c["parts"]["commissieoverwegingen"]
    )
    assert "gedegen onderzoek" in text
    assert all(s["role"] == "commissie" for s in packet["sources"])
    result = validate(engine, rejected(packet), packet)
    assert len(result["case_assessments"]) == 2 and result["semantic_review_required"]
    assert not result["comparison_allowed"] and not result["answer_generated"]
    contribution = rejected(packet)
    marker = packet["case_markers"][0]
    contribution["cases"][0] = dict(
        status="relevant_contribution",
        case_source_id=marker,
        answers_question_part="Synthetische bijdrage voor de validator.",
        finding=dict(
            text="In deze zaak vond de commissie een uitsluitend synthetische testbewering.",
            source_ids=[marker],
        ),
        limitations=[
            dict(text="Geen inhoudelijke juistheid aangetoond.", source_ids=[marker])
        ],
    )
    checked = validate(engine, contribution, packet)
    claim = checked["case_assessments"][0]["finding"]
    assert claim["citations"][0]["quote"] == next(
        s["quote"] for s in packet["sources"] if s["source_id"] == marker
    )
    assert checked["semantic_review_required"] and not checked["comparison_allowed"]
    for mutation in ["missing", "unknown", "cross_case", "quote"]:
        payload = rejected(packet)
        if mutation == "missing":
            payload["cases"].pop()
        elif mutation == "unknown":
            payload["cases"][0]["basis_source_ids"] = ["src_unknown"]
        elif mutation == "cross_case":
            payload["cases"][0]["basis_source_ids"] = [packet["case_markers"][1]]
        else:
            payload["cases"][0]["quote"] = "Model mag geen citaatvelden toevoegen."
        with pytest.raises(ValueError):
            validate(engine, payload, packet)
    altered = copy.deepcopy(packet)
    altered["question"] = "Een andere vraag"
    with pytest.raises(ValueError, match="gewijzigd"):
        validate(engine, rejected(packet), altered)


def test_boundaries_and_no_duplicate_case_versions(engine, prepared):
    for question in [
        "Tot welke datum kan ik nu bezwaar maken?",
        "Kan ik schadevergoeding krijgen?",
    ]:
        result = prepare(engine, question, [])
        assert result["status"] == "needs_additional_sources" and not result["sources"]
        with pytest.raises(ValueError):
            validate(engine, {}, result)
    with pytest.raises(ValueError):
        prepare(engine, "Een historische vraag?", [])
    with pytest.raises(ValueError):
        prepare(engine, "Een historische vraag?", ["missing", "missing"])
