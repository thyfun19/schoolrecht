import pytest

from gpo_local.api import SearchRequest
from gpo_local.explanation import (
    prepare_explanation,
    validate_explanation,
    scope_boundary,
)


def response():
    source = dict(
        case="123",
        page=4,
        section="OVERWEGINGEN VAN DE COMMISSIE",
        role="commissie",
        section_id="123-p4-0",
        text="De school heeft onvoldoende onderzoek gedaan.",
        url="https://example.org/advies.pdf#page=4",
    )
    party = {
        **source,
        "role": "school",
        "section_id": "partij",
        "text": "School beweert iets anders.",
    }
    return {
        "results": [
            dict(
                case_number="123",
                advice_date="2020-01-01",
                parts=dict(
                    commissieoverwegingen=[source],
                    conclusies=[],
                    partijstandpunten=[party],
                    feiten=[party],
                ),
            )
        ]
    }


@pytest.mark.parametrize(
    "q",
    [
        "De school heeft mijn kind geweigerd. Tot welke datum kan ik nu bezwaar maken en waar moet ik het indienen?",
        "Kan ik met deze adviezen schadevergoeding krijgen voor de maanden dat mijn kind geen les kreeg?",
        "Welke termijn geldt voor mijn beroep?",
    ],
)
def test_boundaries_need_sources_even_with_search_results(q):
    prepared = prepare_explanation(q, response())
    assert prepared["status"] == "needs_additional_sources"
    assert prepared["request"] is None and prepared["paid_calls"] == 0


def test_context_excludes_party_positions_and_does_not_assume_answerability():
    packet = prepare_explanation("Zijn er vergelijkbare zaken?", response())
    assert packet["status"] == "awaiting_cost_approval"
    assert packet["answerability"] == "not_assessed"
    assert len(packet["sources"]) == 1
    assert "School beweert" not in packet["request"]["input"]
    assert packet["request"]["store"] is False
    assert packet["request"]["max_output_tokens"] == 2000


def payload():
    return dict(
        similarities=[
            dict(
                text="In deze historische zaak ontbrak voldoende onderzoek.",
                citations=[
                    dict(source_id="123-p4-0", quote="onvoldoende onderzoek gedaan")
                ],
            )
        ],
        differences=[],
        missing_information=["Welke ondersteuning is nodig?"],
        historical_notice="Dit betreft historische zaken.",
    )


@pytest.mark.parametrize(
    "bad", ["unknown_id", "invented_quote", "no_citation", "party_citation"]
)
def test_reject_untraceable_claims(bad):
    data = payload()
    citation = data["similarities"][0]["citations"][0]
    if bad == "unknown_id":
        citation["source_id"] = "niet-bestaand"
    if bad == "invented_quote":
        citation["quote"] = "De ouder wint altijd."
    if bad == "no_citation":
        data["similarities"][0]["citations"] = []
    if bad == "party_citation":
        citation["source_id"] = "partij"
    with pytest.raises(ValueError):
        validate_explanation(
            data, prepare_explanation("Zijn er vergelijkbare zaken?", response())
        )


def test_traceability_is_not_a_semantic_quality_pass():
    result = validate_explanation(
        payload(), prepare_explanation("Zijn er vergelijkbare zaken?", response())
    )
    assert result["citation_integrity"] == "passed"
    assert result["answer_quality"] == "not_validated"
    assert result["citations"][0]["page"] == 4


def test_oversized_context_is_not_silently_truncated():
    huge = response()
    huge["results"][0]["parts"]["commissieoverwegingen"][0]["text"] = "x" * 70000
    assert (
        prepare_explanation("Vergelijk deze zaak", huge)["status"]
        == "context_too_large"
    )


def test_blank_and_unknown_request_fields_rejected():
    with pytest.raises(ValueError):
        SearchRequest(question="   ")
    with pytest.raises(ValueError):
        SearchRequest(question="Wat gebeurde er?", expected_case="123")
