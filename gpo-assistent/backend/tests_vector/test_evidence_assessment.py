import copy
import json

import pytest

from gpo_local.evidence_assessment import prepare_evidence, hydrate_evidence
from gpo_local.evidence_policy import classify_sources
from gpo_local.question_grounding import assemble_reviewed_explanation
from gpo_local.source_registry import SourceRegistry


def fixture():
    docs = {
        "111": dict(
            sha256="a" * 64,
            advice_date="2020-01-01",
            source_url="https://example.org/111.pdf",
        )
    }
    sections = [
        dict(
            case="111",
            page=3,
            section="OVERWEGINGEN",
            role="commissie",
            section_id="reasoning",
            text="De school onderzocht de ondersteuningsbehoefte. De klasgrootte was een relevante voorwaarde.",
        ),
        dict(
            case="111",
            page=4,
            section="OORDEEL",
            role="commissie",
            section_id="outcome",
            text="De Commissie verklaart het verzoek ongegrond. Aldus gedaan te Utrecht door mr. J. Voorzitter.",
        ),
        dict(
            case="111",
            page=4,
            section="AANBEVELINGEN",
            role="commissie",
            section_id="recommendation",
            text="De Commissie beveelt aan om de begeleiding te evalueren.",
        ),
    ]
    registry = SourceRegistry(docs, sections)
    sources = registry.select(["111"])
    packet = prepare_evidence(
        "Welke ondersteuning onderzocht de school?", ["111"], sources
    )
    case = packet["case_packets"][0]
    marker = case["case_markers"][0]
    recommendation = next(
        s["source_id"] for s in sources if s["section_id"] == "recommendation"
    )
    payload = dict(
        cases=[
            dict(
                status="relevant_contribution",
                case_source_id=marker,
                answers_question_part="Het onderzoek naar de behoefte.",
                relevance_explanation="Synthetische inhoudelijke fixture.",
                finding=dict(
                    text="In deze zaak vond de commissie dat de school de behoefte onderzocht.",
                    source_ids=[marker],
                ),
                supporting_arguments=[
                    dict(
                        text="De begeleiding moest worden geëvalueerd.",
                        source_ids=[recommendation],
                    )
                ],
                limitations=[
                    dict(
                        text="De klasgrootte was een relevante voorwaarde.",
                        source_ids=[marker],
                    )
                ],
            )
        ],
        unanswered_question_parts=["De persoonlijke situatie is niet beoordeeld."],
    )
    return registry, packet, payload


def test_signature_units_preserved_in_registry_but_absent_from_request():
    registry, packet, _ = fixture()
    case = packet["case_packets"][0]
    policy = {r["source_id"]: r for r in case["source_policy"]}
    excluded = [
        s for s in case["sources"] if not policy[s["source_id"]]["basis_eligible"]
    ]
    assert excluded
    request = json.dumps(case["request"])
    for source in excluded:
        assert registry.get(source["source_id"], {source["source_id"]}) == source
        assert source["source_id"] not in request
        assert (
            source["quote"]
            not in json.loads(case["request"]["input"])["context"].__str__()
        )


def test_formal_outcome_is_server_owned_and_argument_has_separate_citation():
    registry, packet, payload = fixture()
    result = hydrate_evidence(payload, packet, registry)
    record = result["case_assessments"][0]
    assert record["case_outcome"]["origin"] == "backend"
    assert (
        record["case_outcome"]["citations"][0]["quote"]
        == "De Commissie verklaart het verzoek ongegrond."
    )
    assert (
        record["supporting_arguments"][0]["citations"][0]["section"] == "AANBEVELINGEN"
    )
    assert result["answer_quality"] == "not_validated"
    schema = packet["case_packets"][0]["request"]["text"]["format"]["schema"]
    assert "case_outcome" not in schema["$defs"]["RelevantEvidence"]["properties"]


@pytest.mark.parametrize(
    "mutation",
    ["signature", "formal_as_argument", "invented_outcome", "unknown", "wrong_role"],
)
def test_reject_ineligible_evidence_and_free_outcome_text(mutation):
    registry, packet, payload = fixture()
    case = packet["case_packets"][0]
    if mutation == "invented_outcome":
        payload["cases"][0]["case_outcome"] = dict(
            text="Dus de school kan alle extra hulp bieden.",
            source_ids=case["formal_decision_ids"],
        )
    elif mutation == "signature":
        sid = next(
            r["source_id"]
            for r in case["source_policy"]
            if r["function"] == "administrative_or_mixed"
        )
        payload["cases"][0]["finding"]["source_ids"] = [sid]
    elif mutation == "unknown":
        payload["cases"][0]["supporting_arguments"][0]["source_ids"] = ["unknown"]
    elif mutation == "wrong_role":
        payload["cases"][0]["limitations"][0]["source_ids"] = case[
            "formal_decision_ids"
        ]
    else:
        payload["cases"][0]["supporting_arguments"][0]["source_ids"] = case[
            "formal_decision_ids"
        ]
    with pytest.raises(ValueError):
        hydrate_evidence(payload, packet, registry)


def test_unsupported_inference_still_requires_semantic_review():
    registry, packet, payload = fixture()
    payload["cases"][0]["finding"]["text"] = (
        "In deze zaak vond de commissie dat de school alle extra ondersteuning kon bieden."
    )
    result = hydrate_evidence(payload, packet, registry)
    assert result["case_assessments"][0]["semantic_support"] == "requires_review"
    with pytest.raises(ValueError):
        assemble_reviewed_explanation(result, {})


def test_changed_supporting_argument_invalidates_review():
    registry, packet, payload = fixture()
    result = hydrate_evidence(payload, packet, registry)
    record = result["case_assessments"][0]
    review = dict(
        verdict="supported",
        reviewer="Synthetic test",
        note="Test of review binding.",
        checks=dict(
            question_relevance=True,
            source_support=True,
            conditions_and_exceptions=True,
            case_outcome=True,
        ),
    )
    reviews = {record["assessment_id"]: review}
    assert assemble_reviewed_explanation(result, reviews)["explanation"][
        "contributions"
    ]
    changed = copy.deepcopy(result)
    changed["case_assessments"][0]["supporting_arguments"][0]["text"] = (
        "Gewijzigde motivering."
    )
    with pytest.raises(ValueError, match="gewijzigd"):
        assemble_reviewed_explanation(changed, reviews)


def test_insufficient_relevance_stays_accounted_for_with_canonical_outcome():
    registry, packet, _ = fixture()
    case = packet["case_packets"][0]
    payload = dict(
        cases=[
            dict(
                status="insufficient_relevance",
                case_source_id=case["case_markers"][0],
                reason="De concrete vraag wordt niet onderbouwd.",
                question_part_not_answered="De gevraagde onderzoekshandelingen.",
                basis_source_ids=case["formal_decision_ids"],
            )
        ],
        unanswered_question_parts=[
            "De gevraagde onderzoekshandelingen blijven onbeantwoord."
        ],
    )
    result = hydrate_evidence(payload, packet, registry)
    assert result["evidence_status"] == "insufficient_evidence"
    assert result["case_assessments"][0]["case_outcome"]["origin"] == "backend"


def test_unknown_decision_requires_review_instead_of_guessing():
    registry, _, _ = fixture()
    sources = [s for s in registry.select(["111"]) if s["section_id"] != "outcome"]
    result = prepare_evidence(
        "Welke ondersteuning onderzocht de school?", ["111"], sources
    )
    assert result["status"] == "insufficient_sources"
    assert (
        len(result["case_packets"]) == 1
        and result["case_packets"][0]["request"] is None
    )


def test_signature_crossing_unit_cannot_be_silently_shortened():
    registry, _, _ = fixture()
    sources = registry.select(["111"])
    outcome = next(s for s in sources if s["section_id"] == "outcome")
    outcome["quote"] += " Aldus gedaan te Utrecht."
    outcome["end_offset"] = outcome["start_offset"] + len(outcome["quote"])
    policy = classify_sources(sources)
    assert (
        next(r for r in policy if r["source_id"] == outcome["source_id"])["function"]
        == "administrative_or_mixed"
    )
    assert (
        prepare_evidence("Welke ondersteuning onderzocht de school?", ["111"], sources)[
            "status"
        ]
        == "insufficient_sources"
    )


def test_mixed_recommendation_also_blocks_generation_when_formal_decision_exists():
    registry = SourceRegistry(
        {
            "111": dict(
                sha256="a" * 64,
                advice_date="2020-01-01",
                source_url="https://example.org/111.pdf",
            )
        },
        [
            dict(
                case="111",
                page=3,
                section="OVERWEGINGEN",
                role="commissie",
                section_id="reasoning",
                text="De school onderzocht de behoefte.",
            ),
            dict(
                case="111",
                page=4,
                section="OORDEEL",
                role="commissie",
                section_id="outcome",
                text="De Commissie verklaart het verzoek gegrond.",
            ),
            dict(
                case="111",
                page=4,
                section="AANBEVELINGEN",
                role="commissie",
                section_id="recommendation",
                text="De Commissie beveelt begeleiding aan Aldus gedaan te Utrecht.",
            ),
        ],
    )
    result = prepare_evidence(
        "Welke ondersteuning onderzocht de school?", ["111"], registry.select(["111"])
    )
    assert result["status"] == "insufficient_sources"
    assert result["case_packets"][0]["request"] is None


def test_partial_decision_is_kept_verbatim_without_generalizing():
    registry = SourceRegistry(
        {
            "111": dict(
                sha256="a" * 64,
                advice_date="2020-01-01",
                source_url="https://example.org/111.pdf",
            )
        },
        [
            dict(
                case="111",
                page=4,
                section="OORDEEL",
                role="commissie",
                section_id="outcome",
                text="De Commissie verklaart het verzoek gedeeltelijk gegrond. Aldus gedaan te Utrecht.",
            )
        ],
    )
    sources = registry.select(["111"])
    outcome = next(s for s in sources if s["section_id"] == "outcome")
    policy = classify_sources(sources)
    assert (
        next(r for r in policy if r["source_id"] == outcome["source_id"])["function"]
        == "formal_decision"
    )
    assert outcome["quote"].endswith("gedeeltelijk gegrond.")
