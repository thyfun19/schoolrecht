import copy
import json

import pytest

from gpo_local.isolated_assessment import prepare_isolated, hydrate_isolated
from gpo_local.question_grounding import assemble_reviewed_explanation
from gpo_local.question_intake import question_intake
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
    sections = []
    for case in docs:
        sections.extend(
            [
                dict(
                    case=case,
                    page=3,
                    section="OVERWEGINGEN",
                    role="commissie",
                    section_id=case + "-p3",
                    text="De school onderzocht de ondersteuningsbehoefte. De concrete omstandigheden waren belangrijk.",
                ),
                dict(
                    case=case,
                    page=4,
                    section="OORDEEL",
                    role="commissie",
                    section_id=case + "-p4",
                    text="De commissie verklaart dit verzoek ongegrond.",
                ),
            ]
        )
    registry = SourceRegistry(docs, sections)
    packet = prepare_isolated(
        "Welke ondersteuning onderzocht de school?",
        list(docs),
        registry.select(list(docs)),
    )
    first, second = packet["case_packets"]
    marker = first["case_markers"][0]
    conclusion = next(
        s["source_id"] for s in first["sources"] if s["category"] == "conclusies"
    )
    payload = dict(
        cases=[
            dict(
                status="relevant_contribution",
                case_source_id=marker,
                answers_question_part="De onderzochte ondersteuningsbehoefte.",
                relevance_explanation="Synthetische test van de vraaggerichte bijdrage.",
                finding=dict(
                    text="In deze zaak vond de commissie dat de school de behoefte onderzocht.",
                    source_ids=[marker],
                ),
                limitations=[
                    dict(
                        text="De concrete omstandigheden waren belangrijk.",
                        source_ids=[marker],
                    )
                ],
                case_outcome=dict(
                    text="In deze zaak vond de commissie het verzoek ongegrond.",
                    source_ids=[conclusion],
                ),
            ),
            dict(
                status="insufficient_relevance",
                case_source_id=second["case_markers"][0],
                reason="Synthetische afwijzing om uitsluiting te testen.",
                question_part_not_answered="De concrete onderzoekshandelingen.",
                basis_source_ids=second["case_markers"],
            ),
        ],
        unanswered_question_parts=[
            "De concrete persoonlijke situatie is niet beoordeeld."
        ],
    )
    return registry, packet, payload


def test_each_request_contains_only_one_case_and_its_source_choices():
    _, packet, _ = fixture()
    assert packet["request"] is None and packet["max_calls"] == 2
    assert packet["estimated_max_cost_usd"] == sum(
        p["estimated_max_cost_usd"] for p in packet["case_packets"]
    )
    for own in packet["case_packets"]:
        request = own["request"]
        assert request["store"] is False
        assert (
            request["text"]["format"]["schema"]["properties"]["cases"]["maxItems"] == 1
        )
        allowed = {s["source_id"] for s in own["sources"]}
        foreign = {s["source_id"] for s in packet["sources"]} - allowed
        serialized = json.dumps(request)
        assert all(sid not in serialized for sid in foreign)
        assert all(sid in serialized for sid in allowed)


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown",
        "other_case",
        "wrong_outcome",
        "model_quote",
        "missing_result",
        "duplicate_result",
        "missing_relevance",
        "missing_outcome",
        "parent_request",
        "nonhistorical_outcome",
    ],
)
def test_reject_regressions_before_review(mutation):
    registry, packet, payload = fixture()
    item = payload["cases"][0]
    if mutation == "unknown":
        item["finding"]["source_ids"] = ["unknown"]
    elif mutation == "other_case":
        item["finding"]["source_ids"] = packet["case_packets"][1]["case_markers"]
    elif mutation == "wrong_outcome":
        item["case_outcome"]["source_ids"] = packet["case_packets"][0]["case_markers"]
    elif mutation == "model_quote":
        item["finding"]["quote"] = "Modelcitaat"
    elif mutation == "missing_result":
        payload["cases"].pop()
    elif mutation == "duplicate_result":
        payload["cases"][1] = copy.deepcopy(item)
    elif mutation == "missing_relevance":
        item.pop("relevance_explanation")
    elif mutation == "missing_outcome":
        item.pop("case_outcome")
    elif mutation == "parent_request":
        item["relevance_explanation"] = "Kunt u bronpassages aanleveren?"
    else:
        item["case_outcome"]["text"] = "Uw kind mag dus altijd terugkeren."
    with pytest.raises(ValueError):
        hydrate_isolated(payload, packet, registry)


def test_review_required_per_dimension_and_irrelevant_case_excluded():
    registry, packet, payload = fixture()
    result = hydrate_isolated(payload, packet, registry)
    assert (
        result["answer_quality"] == "not_validated" and not result["comparison_allowed"]
    )
    assert result["case_assessments"][0]["case_outcome"]["citations"][0]["page"] == 4
    reviews = {
        r["assessment_id"]: dict(
            verdict="supported",
            reviewer="Synthetic test",
            note="Controle van de poort, geen inhoudelijke validatie.",
        )
        for r in result["case_assessments"]
    }
    with pytest.raises(ValueError, match="afzonderlijk"):
        assemble_reviewed_explanation(result, reviews)
    for review in reviews.values():
        review["checks"] = dict(
            question_relevance=True,
            source_support=True,
            conditions_and_exceptions=True,
            case_outcome=True,
        )
    explanation = assemble_reviewed_explanation(result, reviews)
    assert [r["case_number"] for r in explanation["explanation"]["contributions"]] == [
        "111"
    ]
    assert explanation["audit"]["excluded_cases"][0]["case_number"] == "222"
    changed = copy.deepcopy(result)
    changed["case_assessments"][0]["case_outcome"]["text"] = "Na review gewijzigd."
    with pytest.raises(ValueError, match="gewijzigd"):
        assemble_reviewed_explanation(changed, reviews)


def test_packet_tampering_and_missing_context_cannot_silently_drop_case():
    registry, packet, payload = fixture()
    altered = copy.deepcopy(packet)
    altered["question"] = "Ander onderwerp"
    with pytest.raises(ValueError, match="gewijzigd"):
        hydrate_isolated(payload, altered, registry)
    truncated = [
        s
        for s in packet["sources"]
        if not (s["case_number"] == "222" and s["category"] == "conclusies")
    ]
    missing = prepare_isolated(packet["question"], ["111", "222"], truncated)
    assert missing["status"] == "insufficient_sources"
    assert len(missing["case_packets"]) == 2
    assert missing["case_packets"][1]["request"] is None
    with pytest.raises(ValueError):
        hydrate_isolated(payload, missing, registry)


def test_valid_citation_does_not_establish_semantic_support():
    registry, packet, payload = fixture()
    payload["cases"][0]["finding"]["text"] = (
        "In deze zaak vond de commissie dat de school altijd gelijk heeft."
    )
    result = hydrate_isolated(payload, packet, registry)
    assert result["case_assessments"][0]["semantic_support"] == "requires_review"
    with pytest.raises(ValueError):
        assemble_reviewed_explanation(result, {})


@pytest.mark.parametrize("field", ["question", "unanswered_question_parts"])
def test_review_cannot_be_reused_after_question_or_limitations_change(field):
    registry, packet, payload = fixture()
    result = hydrate_isolated(payload, packet, registry)
    reviews = {
        r["assessment_id"]: dict(
            verdict="supported",
            reviewer="Synthetic test",
            note="Fixture.",
            checks=dict(
                question_relevance=True,
                source_support=True,
                conditions_and_exceptions=True,
                case_outcome=True,
            ),
        )
        for r in result["case_assessments"]
    }
    result[field] = "Een andere vraag" if field == "question" else []
    with pytest.raises(ValueError, match="gewijzigd"):
        assemble_reviewed_explanation(result, reviews)


@pytest.mark.parametrize(
    "question",
    [
        "De afwijzing kwam gisteren binnen. Hoeveel dagen heb ik nog om ertegen in te gaan en naar welk loket moet ik?",
        "Als mijn kind naar een school verder weg moet, betaalt de gemeente dan onze taxiritten en hoeveel krijgen wij terug?",
        "Hoeveel schadevergoeding krijg ik omdat mijn kind een jaar heeft gemist?",
        "Hoe lang heb ik nog tijd om iets tegen deze beslissing te doen?",
        "Worden onze vervoerskosten door de gemeente vergoed?",
    ],
)
def test_insufficient_scope_blocks_request_before_source_access(question):
    result = prepare_isolated(question, [], [])
    assert result["status"] == "needs_additional_sources"
    assert result["request"] is None and not result["case_packets"]


@pytest.mark.parametrize(
    "question",
    [
        "De school wil mijn kind niet meer. Wat hebben andere ouders hieraan gehad?",
        "Ik ben het niet eens met het plan van school. Kun je mij helpen?",
    ],
)
def test_ambiguous_question_gets_one_clarification(question):
    result = question_intake(question)
    assert result["status"] == "needs_clarification"
    assert result["clarification_question"].count("?") == 1
    assert "zaaknummer" not in result["clarification_question"]


@pytest.mark.parametrize(
    "question",
    [
        "Welke ondersteuning onderzocht de school?",
        "Hoe woog de commissie eerdere incidenten bij een verwijdering?",
        "Heeft een school het OPP met ouders geëvalueerd?",
    ],
)
def test_clear_historical_questions_are_not_blocked(question):
    assert question_intake(question) is None
