import copy
from collections import defaultdict

import pytest

from gpo_app.chunking import LocalTokenCounter
from gpo_app.discovery_context import (
    auxiliary_fragments,
    evidence_packets,
    rrf,
    validate_auxiliary,
)
from conftest import BASE


@pytest.fixture(scope="module")
def counter():
    return LocalTokenCounter(BASE / "data/e5/model/tokenizer.json")


def context(doc):
    return dict(
        processing_id=doc.processing["processing_id"],
        version_id=doc.version["version_id"],
        source_url=doc.version["source_url"],
        advice_date=str(doc.version["advice_date"]),
    )


def fragments(doc, counter):
    return auxiliary_fragments(
        doc.document["case_number"],
        doc.processing["processing_id"],
        doc.version["version_id"],
        doc.version["source_url"],
        doc.sections,
        doc.pages,
        counter,
    )


def test_all_auxiliary_text_has_exact_coverage_roles_and_no_explanation_citations(
    prepared, counter
):
    for doc in prepared:
        aux = fragments(doc, counter)
        assert aux == fragments(doc, counter)
        case = doc.document["case_number"]
        ctx = {case: context(doc)}
        validate_auxiliary(aux, [dict(case=case, sections=doc.sections)], counter, ctx)
        texts = defaultdict(list)
        for c in aux:
            assert (
                c["role"] != "commissie"
                and c["evidence_eligible"] is False
                and not c["citation_source_ids"]
            )
            assert c["token_count"] <= 384 and c["id"].startswith("aux_")
            texts[c["section_key"]].append(c)
        for s in doc.sections:
            if s["category"] in ("feiten", "partijstandpunten"):
                assert "".join(c["text"] for c in texts[s["section_key"]]) == s["text"]
            else:
                assert s["section_key"] not in texts


@pytest.mark.parametrize(
    "field,value",
    [
        ("role", "commissie"),
        ("page", 999),
        ("version_id", "wrong"),
        ("source_url", "https://example.invalid"),
        ("citation_source_ids", ["src_unknown"]),
        ("evidence_eligible", True),
    ],
)
def test_changed_provenance_is_rejected(prepared, counter, field, value):
    doc = prepared[0]
    aux = fragments(doc, counter)
    aux[0][field] = value
    case = doc.document["case_number"]
    with pytest.raises(ValueError):
        validate_auxiliary(
            aux, [dict(case=case, sections=doc.sections)], counter, {case: context(doc)}
        )


def test_missing_auxiliary_fragment_is_rejected(prepared, counter):
    doc = prepared[0]
    aux = fragments(doc, counter)
    case = doc.document["case_number"]
    with pytest.raises(ValueError):
        validate_auxiliary(
            aux[1:],
            [dict(case=case, sections=doc.sections)],
            counter,
            {case: context(doc)},
        )


def test_rrf_ties_and_case_uniqueness_are_explicit():
    assert rrf(["a", "b"], ["b", "a"]) == ["a", "b"]
    assert rrf(["a", "c"], ["b", "c"])[0] == "c"
    with pytest.raises(ValueError):
        rrf(["a", "a"])


def test_evidence_includes_only_selected_canonical_committee_parts():
    span = dict(
        role="commissie", source_ids=["src_existing"], text="Canonieke overweging."
    )
    primary = dict(
        processing_id="p",
        version_id="v",
        source_url="https://example.invalid",
        advice_date="2026-01-01",
        parts={
            "commissieoverwegingen": [span],
            "conclusies": [],
            "partijstandpunten": [dict(role="ouders", text="Onbewezen stelling")],
        },
    )
    contexts = {"a": primary, "b": copy.deepcopy(primary)}
    result = evidence_packets(["a"], contexts)
    assert len(result) == 1 and result[0]["case_number"] == "a"
    assert set(result[0]["parts"]) == {"commissieoverwegingen", "conclusies"}
    assert result[0]["semantic_support"] == "not_assessed"
    with pytest.raises(KeyError):
        evidence_packets(["missing"], contexts)
    with pytest.raises(ValueError):
        evidence_packets(["a", "a"], contexts)
    contexts["a"]["parts"]["commissieoverwegingen"][0]["source_ids"] = [
        "aux_search_only"
    ]
    with pytest.raises(ValueError):
        evidence_packets(["a"], contexts)


def test_party_statement_cannot_enter_explanation_as_committee_evidence():
    ctx = dict(
        processing_id="p",
        version_id="v",
        source_url="url",
        advice_date=None,
        parts={
            "commissieoverwegingen": [dict(role="ouders", source_ids=[])],
            "conclusies": [],
        },
    )
    with pytest.raises(ValueError):
        evidence_packets(["a"], {"a": ctx})
