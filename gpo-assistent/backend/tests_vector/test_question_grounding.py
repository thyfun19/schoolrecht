import copy

import pytest

from gpo_local.question_grounding import (
    prepare_question_assessment,
    hydrate_question_assessment,
    assemble_reviewed_explanation,
)
from gpo_local.source_registry import SourceRegistry


def fixture():
    docs = {
        case: dict(
            sha256=case * 21 + "a",
            advice_date="2020-01-01",
            source_url=f"https://example.org/{case}.pdf",
        )
        for case in ["111", "222"]
    }
    sections = [
        dict(
            case=case,
            page=3,
            section="OVERWEGINGEN",
            role="commissie",
            section_id=case + "-p3",
            text=text,
        )
        for case, text in [
            ("111", "Bij plaatsing elders bleef het schoolbestuur verantwoordelijk."),
            ("222", "Dit geschil ging over een andere maatregel."),
        ]
    ]
    registry = SourceRegistry(docs, sections)
    prepared = prepare_question_assessment(
        "Wie bleef verantwoordelijk bij plaatsing elders?",
        dict(method="text", results=[dict(case_number=case) for case in docs]),
        registry,
    )
    a, b = prepared["case_markers"]
    payload = dict(
        cases=[
            dict(
                status="relevant_contribution",
                case_source_id=a,
                answers_question_part="Wie verantwoordelijk bleef bij plaatsing elders.",
                finding=dict(
                    text="In deze zaak vond de commissie dat het schoolbestuur verantwoordelijk bleef.",
                    source_ids=[a],
                ),
                limitations=[
                    dict(text="Het ging om plaatsing elders.", source_ids=[a])
                ],
            ),
            dict(
                status="insufficient_relevance",
                case_source_id=b,
                reason="De aangeboden overweging betreft een andere maatregel en onderbouwt de verantwoordelijkheidsvraag niet.",
                question_part_not_answered="Wie verantwoordelijk bleef bij plaatsing elders.",
                basis_source_ids=[b],
            ),
        ],
        unanswered_question_parts=[],
    )
    return registry, prepared, payload


def reviews(hydrated):
    return {
        r["assessment_id"]: dict(
            verdict="supported",
            reviewer="Synthetic offline test",
            note="Test van controlemechanisme, geen echte juridische beoordeling.",
        )
        for r in hydrated["case_assessments"]
    }


def test_every_found_case_has_contribution_or_explained_rejection():
    registry, prepared, payload = fixture()
    schema = prepared["request"]["text"]["format"]["schema"]
    assert (
        schema["properties"]["cases"]["minItems"]
        == schema["properties"]["cases"]["maxItems"]
        == 2
    )
    payload["cases"] = payload["cases"][:1]
    with pytest.raises(ValueError, match="Iedere opgehaalde"):
        hydrate_question_assessment(payload, prepared, registry)


def test_irrelevant_case_is_audited_but_not_used_in_explanation():
    registry, prepared, payload = fixture()
    hydrated = hydrate_question_assessment(payload, prepared, registry)
    result = assemble_reviewed_explanation(hydrated, reviews(hydrated))
    assert [c["case_number"] for c in result["explanation"]["contributions"]] == ["111"]
    assert result["audit"]["assessed_cases"] == 2
    assert result["audit"]["excluded_cases"][0]["case_number"] == "222"
    assert "andere maatregel" not in str(result["explanation"])


def test_no_relevant_case_requires_explicit_unanswered_question_part():
    registry, prepared, payload = fixture()
    first = payload["cases"][0]
    payload["cases"][0] = dict(
        status="insufficient_relevance",
        case_source_id=first["case_source_id"],
        reason="Deze bron onderbouwt het gevraagde verband niet.",
        question_part_not_answered="Wie verantwoordelijk bleef.",
        basis_source_ids=[first["case_source_id"]],
    )
    with pytest.raises(ValueError, match="expliciet"):
        hydrate_question_assessment(payload, prepared, registry)
    payload["unanswered_question_parts"] = [
        "Met deze opgehaalde bronnen kan niet worden vastgesteld wie bij plaatsing elders verantwoordelijk bleef."
    ]
    hydrated = hydrate_question_assessment(payload, prepared, registry)
    result = assemble_reviewed_explanation(hydrated, reviews(hydrated))
    assert result["evidence_status"] == "insufficient_evidence"
    assert result["explanation"]["contributions"] == []


@pytest.mark.parametrize(
    "bad",
    [
        "unknown",
        "other_case",
        "no_condition",
        "general_rule",
        "ask_parent",
        "ask_case",
        "question_not_boundary",
        "quote",
    ],
)
def test_reject_bad_question_focused_output(bad):
    registry, prepared, payload = fixture()
    contribution = payload["cases"][0]
    if bad == "unknown":
        contribution["finding"]["source_ids"] = ["unknown"]
    if bad == "other_case":
        contribution["finding"]["source_ids"] = [payload["cases"][1]["case_source_id"]]
    if bad == "no_condition":
        contribution["limitations"] = []
    if bad == "general_rule":
        contribution["finding"]["text"] = "Het schoolbestuur moet dit altijd doen."
    if bad == "ask_parent":
        payload["unanswered_question_parts"] = [
            "Kunt u zaaknummers of bronpassages aanleveren?"
        ]
    if bad == "ask_case":
        payload["unanswered_question_parts"] = [
            "Over welke eerdere zaak wilt u een vergelijking?"
        ]
    if bad == "question_not_boundary":
        payload["unanswered_question_parts"] = ["Welke partij bleef verantwoordelijk?"]
    if bad == "quote":
        contribution["finding"]["quote"] = "Modelcitaat"
    with pytest.raises(ValueError):
        hydrate_question_assessment(payload, prepared, registry)


def test_existing_source_does_not_prove_relevance_or_content():
    registry, prepared, payload = fixture()
    payload["cases"][0]["finding"]["text"] = (
        "In deze zaak vond de commissie dat de school altijd gelijk heeft."
    )
    hydrated = hydrate_question_assessment(payload, prepared, registry)
    assert hydrated["answer_quality"] == "not_validated"
    with pytest.raises(ValueError):
        assemble_reviewed_explanation(hydrated, {})
    changed = copy.deepcopy(hydrated)
    changed["case_assessments"][0]["limitations"][0]["text"] = (
        "Gewijzigd na beoordeling"
    )
    with pytest.raises(ValueError):
        assemble_reviewed_explanation(changed, reviews(hydrated))
