import copy

import pytest

from gpo_local.grounding import (
    prepare_case_findings,
    hydrate_case_findings,
    prepare_comparison,
    hydrate_comparison,
)
from gpo_local.source_registry import SourceRegistry


def context():
    documents = {}
    sections = []
    for case, quote in [
        ("111", "De school heeft onvoldoende onderzoek gedaan."),
        ("222", "De school heeft voldoende onderzoek gedaan."),
    ]:
        documents[case] = dict(
            advice_date="2020-01-01",
            sha256=case * 21 + "x",
            source_url=f"https://example.org/{case}.pdf",
        )
        sections.append(
            dict(
                case=case,
                page=4,
                section="OVERWEGINGEN VAN DE COMMISSIE",
                role="commissie",
                section_id=case + "-p4-0",
                text=quote,
            )
        )
        sections.append(
            dict(
                case=case,
                page=3,
                section="PARTIJEN",
                role="school",
                section_id=case + "-p3-0",
                text="School stelt iets anders.",
            )
        )
    registry = SourceRegistry(documents, sections)
    response = {"results": [{"case_number": case} for case in documents]}
    prepared = prepare_case_findings("Welke verschillen zijn er?", response, registry)
    ids = [s["source_id"] for s in registry.select(["111", "222"])]
    payload = {
        "cases": [
            dict(
                case_source_id=sid,
                conclusions=[dict(text="Testclaim " + str(i), source_ids=[sid])],
            )
            for i, sid in enumerate(ids)
        ],
        "missing_information": ["Welke ondersteuning is nodig?"],
    }
    return registry, prepared, payload, ids, documents, sections


def test_fixed_ids_survive_order_and_change_with_source_version():
    registry, _, _, ids, documents, sections = context()
    other = SourceRegistry(documents, list(reversed(sections)))
    assert set(ids) == {s["source_id"] for s in other.select(documents)}
    changed = copy.deepcopy(documents)
    changed["111"]["sha256"] = "new-version"
    new = SourceRegistry(changed, sections)
    with pytest.raises(ValueError):
        new.get(ids[0], ids)
    assert len(registry.select(documents)) == 2  # No party positions.


def test_backend_owns_quotes_pages_cases_and_links_even_if_packet_metadata_changes():
    registry, prepared, payload, ids, _, _ = context()
    prepared["sources"][0].update(
        quote="VERZONNEN",
        page=99,
        case_number="other",
        source_url="https://wrong.invalid",
    )
    result = hydrate_case_findings(payload, prepared, registry)
    citation = result["cases"][0]["conclusions"][0]["citations"][0]
    assert citation["quote"] == "De school heeft onvoldoende onderzoek gedaan."
    assert citation["page"] == 4 and citation["case_number"] == "111"
    assert citation["source_url"] == "https://example.org/111.pdf#page=4"


@pytest.mark.parametrize(
    "bad", ["unknown", "cross_case", "model_quote", "model_page", "missing_case"]
)
def test_reject_bad_model_references(bad):
    registry, prepared, payload, ids, _, _ = context()
    claim = payload["cases"][0]["conclusions"][0]
    if bad == "unknown":
        claim["source_ids"] = ["src_unknown"]
    if bad == "cross_case":
        claim["source_ids"] = [ids[1]]
    if bad == "model_quote":
        claim["quote"] = "Zelfgeschreven citaat"
    if bad == "model_page":
        claim["page"] = 4
    if bad == "missing_case":
        payload["cases"] = payload["cases"][:1]
    with pytest.raises(ValueError):
        hydrate_case_findings(payload, prepared, registry)


def test_valid_source_does_not_approve_false_claim_or_allow_comparison():
    registry, prepared, payload, ids, _, _ = context()
    payload["cases"][0]["conclusions"][0]["text"] = "De school heeft altijd gelijk."
    findings = hydrate_case_findings(payload, prepared, registry)
    assert findings["answer_quality"] == "not_validated"
    assert (
        findings["cases"][0]["conclusions"][0]["semantic_support"] == "requires_review"
    )
    with pytest.raises(ValueError, match="inhoudelijk"):
        prepare_comparison("Vergelijk", findings, registry, {})
    rejected = {
        c["claim_id"]: dict(
            verdict="unsupported", reviewer="Offline test", note="Bron zegt dit niet."
        )
        for case in findings["cases"]
        for c in case["conclusions"]
    }
    with pytest.raises(ValueError):
        prepare_comparison("Vergelijk", findings, registry, rejected)


def test_comparison_follows_reviewed_separate_cases_and_rehydrates_sources():
    registry, prepared, payload, ids, _, _ = context()
    findings = hydrate_case_findings(payload, prepared, registry)
    reviews = {
        c["claim_id"]: dict(
            verdict="supported",
            reviewer="Synthetic test reviewer",
            note="Only a test of gating.",
        )
        for case in findings["cases"]
        for c in case["conclusions"]
    }
    comparison = prepare_comparison("Vergelijk", findings, registry, reviews)
    result = hydrate_comparison(
        dict(
            similarities=[],
            differences=[
                dict(text="De historische oordelen verschillen.", source_ids=ids)
            ],
            missing_information=["Wat heeft de school onderzocht?"],
        ),
        comparison,
        registry,
    )
    assert len(result["differences"][0]["citations"]) == 2
    assert result["answer_quality"] == "not_validated"
    with pytest.raises(ValueError):
        hydrate_comparison(
            dict(
                similarities=[],
                differences=[dict(text="Een zaak", source_ids=ids[:1])],
                missing_information=["Wat ontbreekt?"],
            ),
            comparison,
            registry,
        )
    findings["cases"][0]["conclusions"][0]["text"] = "Gewijzigd na beoordeling"
    with pytest.raises(ValueError):
        prepare_comparison("Vergelijk", findings, registry, reviews)


def test_model_schema_permits_only_known_source_ids_and_no_quote_fields():
    _, prepared, _, ids, _, _ = context()
    schema = prepared["request"]["text"]["format"]["schema"]
    assert set(
        schema["$defs"]["SourceClaim"]["properties"]["source_ids"]["items"]["enum"]
    ) == set(ids)
    assert set(schema["$defs"]["SourceClaim"]["properties"]) == {"text", "source_ids"}
