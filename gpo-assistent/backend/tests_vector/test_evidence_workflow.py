import copy

import pytest

from gpo_local.evidence_assessment import prepare_evidence, hydrate_evidence
from gpo_local.evidence_workflow import (
    prepare_plans,
    validate_plans,
    prepare_writing,
    selection_policy,
)
from gpo_local.source_registry import SourceRegistry
from gpo_local.question_intake import question_intake


def fixture():
    docs = {
        c: dict(
            sha256=c * 21 + "a",
            advice_date="2020-01-01",
            source_url=f"https://example.org/{c}.pdf",
        )
        for c in ["111", "222"]
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
                    section_id=case + "-reasoning",
                    text="Het bestuur moet de ondersteuning onderzoeken. De school heeft observaties besproken. "
                    "De kleine groep was bepalend. Het geschil betreft toelating. "
                    "45 lid 2 Wet op de expertisecentra 4 Kamerstukken II 2011/12.",
                ),
                dict(
                    case=case,
                    page=4,
                    section="OORDEEL",
                    role="commissie",
                    section_id=case + "-outcome",
                    text="De Commissie verklaart het verzoek ongegrond.",
                ),
            ]
        )
    registry = SourceRegistry(docs, sections)
    baseline = prepare_evidence(
        "Welke ondersteuning onderzocht de school vóór toelating?",
        list(docs),
        registry.select(list(docs)),
    )
    workflow = prepare_plans(baseline)
    payloads = []
    for case in baseline["case_packets"]:
        ids = {s["quote"].split(".")[0]: s["source_id"] for s in case["sources"]}
        payloads.append(
            dict(
                case_source_id=case["case_markers"][0],
                relevance="direct",
                issue_match="same_issue",
                answered_question_part="De onderzochte ondersteuning.",
                relevance_reason="Synthetic test.",
                issue_source_ids=[ids["Het geschil betreft toelating"]],
                rule_source_ids=[ids["Het bestuur moet de ondersteuning onderzoeken"]],
                application_source_ids=[ids["De school heeft observaties besproken"]],
                condition_source_ids=[ids["De kleine groep was bepalend"]],
                difference_source_ids=[],
                missing_question_parts=[],
            )
        )
    payloads[1].update(
        relevance="background",
        issue_match="related_issue",
        missing_question_parts=["De concrete toelatingsvraag blijft onbeantwoord."],
    )
    return registry, baseline, workflow, payloads


def reviews(validated):
    return {
        r["plan_id"]: dict(
            verdict="supported",
            reviewer="Synthetic test",
            note="Mechanism test only.",
            checks=dict(
                question_match=True,
                source_support=True,
                concrete_application=True,
                conditions_and_differences=True,
            ),
        )
        for r in validated["plans"]
    }


def test_reference_footnote_cannot_be_selected_as_argument():
    registry, baseline, workflow, payloads = fixture()
    case = baseline["case_packets"][0]
    reference = next(s for s in case["sources"] if s["quote"].startswith("45 lid"))
    assert (
        selection_policy(case)[reference["source_id"]]["function"]
        == "reference_candidate"
    )
    schema = workflow["selection_requests"][0]["request"]["text"]["format"]["schema"]
    assert (
        reference["source_id"]
        not in schema["properties"]["rule_source_ids"]["items"]["enum"]
    )
    payloads[0]["rule_source_ids"] = [reference["source_id"]]
    with pytest.raises(ValueError):
        validate_plans(payloads, workflow, registry)


@pytest.mark.parametrize(
    "mutation",
    [
        "no_application",
        "related_as_direct",
        "missing_case",
        "duplicate_case",
        "no_conditions_or_limitation",
        "cross_case",
        "parent_sources",
    ],
)
def test_incomplete_or_wrong_evidence_plan_is_blocked(mutation):
    registry, _, workflow, payloads = fixture()
    if mutation == "no_application":
        payloads[0]["application_source_ids"] = []
    elif mutation == "related_as_direct":
        payloads[0]["issue_match"] = "related_issue"
    elif mutation == "missing_case":
        payloads.pop()
    elif mutation == "duplicate_case":
        payloads[1] = copy.deepcopy(payloads[0])
    elif mutation == "no_conditions_or_limitation":
        payloads[0]["condition_source_ids"] = []
    elif mutation == "cross_case":
        payloads[0]["application_source_ids"] = payloads[1]["application_source_ids"]
    else:
        payloads[0]["missing_question_parts"] = ["Kunt u bronpassages aanleveren?"]
    with pytest.raises(ValueError):
        validate_plans(payloads, workflow, registry)


def test_reviewer_gate_and_background_exclusion_before_writing():
    registry, baseline, workflow, payloads = fixture()
    validated = validate_plans(payloads, workflow, registry)
    with pytest.raises(ValueError):
        prepare_writing(validated, workflow, {})
    writer = prepare_writing(validated, workflow, reviews(validated))
    assert [p["case_number"] for p in writer["case_packets"]] == ["111"]
    assert writer["selection_audit"]["excluded_cases"][0]["case_number"] == "222"
    selected = {
        sid
        for name, ids in payloads[0].items()
        if name.endswith("_source_ids")
        for sid in ids
    }
    selected |= set(baseline["case_packets"][0]["formal_decision_ids"])
    assert {s["source_id"] for s in writer["sources"]} == selected
    assert writer["answer_generated"] is False


def test_changed_reviewed_plan_or_question_cannot_be_reused():
    registry, _, workflow, payloads = fixture()
    validated = validate_plans(payloads, workflow, registry)
    review = reviews(validated)
    changed = copy.deepcopy(validated)
    changed["plans"][0]["plan"]["relevance"] = "insufficient"
    with pytest.raises(ValueError):
        prepare_writing(changed, workflow, review)
    changed = copy.deepcopy(validated)
    changed["question"] = "Another question"
    with pytest.raises(ValueError):
        prepare_writing(changed, workflow, review)


def test_writer_cannot_cite_unselected_reference_even_when_it_exists():
    registry, baseline, workflow, payloads = fixture()
    validated = validate_plans(payloads, workflow, registry)
    writer = prepare_writing(validated, workflow, reviews(validated))
    case = writer["case_packets"][0]
    reference = next(
        s["source_id"]
        for s in baseline["sources"]
        if s["case_number"] == "111" and s["quote"].startswith("45 lid")
    )
    body = dict(
        cases=[
            dict(
                status="relevant_contribution",
                case_source_id=case["case_markers"][0],
                answers_question_part="Het onderzoek.",
                relevance_explanation="Synthetic test.",
                finding=dict(
                    text="In deze zaak vond de commissie iets over ondersteuning.",
                    source_ids=[reference],
                ),
                supporting_arguments=[],
                limitations=[
                    dict(text="Een voorwaarde.", source_ids=case["case_markers"])
                ],
            )
        ],
        unanswered_question_parts=[],
    )
    with pytest.raises(ValueError):
        hydrate_evidence(body, writer, registry)


def test_only_background_evidence_does_not_prepare_a_writer():
    registry, _, workflow, payloads = fixture()
    payloads[0].update(
        relevance="insufficient",
        issue_match="different_issue",
        missing_question_parts=["Geen direct bewijs beschikbaar."],
    )
    validated = validate_plans(payloads, workflow, registry)
    writer = prepare_writing(validated, workflow, reviews(validated))
    assert writer["status"] == "insufficient_evidence" and writer["case_packets"] == []
    assert len(writer["excluded_cases"]) == 2


def test_reviewer_can_exclude_a_misclassified_plan_without_rewriting_it():
    registry, _, workflow, payloads = fixture()
    payloads[1].update(relevance="direct", issue_match="same_issue")
    validated = validate_plans(payloads, workflow, registry)
    review = reviews(validated)
    wrong = validated["plans"][1]
    review[wrong["plan_id"]] = dict(
        verdict="exclude",
        reviewer="Synthetic reviewer",
        note="The selected issue does not answer the concrete question.",
    )
    writer = prepare_writing(validated, workflow, review)
    assert [p["case_number"] for p in writer["case_packets"]] == ["111"]
    assert writer["selection_audit"]["excluded_cases"][0]["excluded_by_review"]
    assert validated["plans"][1] == wrong and wrong["plan"]["relevance"] == "direct"


def test_substantive_legal_statement_is_not_a_reference_only():
    _, baseline, _, _ = fixture()
    case = copy.deepcopy(baseline["case_packets"][0])
    source = next(
        s for s in case["sources"] if s["category"] == "commissieoverwegingen"
    )
    source["quote"] = "Artikel 40 bepaalt dat het bestuur onderzoek moet doen."
    assert selection_policy(case)[source["source_id"]]["argument_eligible"]


def test_unsupported_source_role_assignment_stays_pending_semantic_review():
    registry, _, workflow, payloads = fixture()
    payloads[0]["application_source_ids"] = payloads[0]["rule_source_ids"]
    payloads[0]["difference_source_ids"] = list(payloads[0]["application_source_ids"])
    validated = validate_plans(payloads, workflow, registry)
    assert (
        validated["semantic_review_required"]
        and validated["answer_quality"] == "not_validated"
    )
    with pytest.raises(ValueError):
        prepare_writing(validated, workflow, {})


@pytest.mark.parametrize(
    "fragment", ["3, p.", "17.", "292", "19).", "2015, 95, p.", "22-23)."]
)
def test_trailing_footnote_page_fragments_are_not_argument_eligible(fragment):
    _, baseline, _, _ = fixture()
    case = copy.deepcopy(baseline["case_packets"][0])
    source = next(
        s for s in case["sources"] if s["category"] == "commissieoverwegingen"
    )
    source["quote"] = fragment
    rule = selection_policy(case)[source["source_id"]]
    assert rule["function"] == "reference_candidate" and not rule["argument_eligible"]


@pytest.mark.parametrize("mutation", ["too_many_issue_ids", "duplicate_in_list"])
def test_unbounded_or_duplicate_evidence_plan_lists_are_blocked(mutation):
    registry, _, workflow, payloads = fixture()
    if mutation == "too_many_issue_ids":
        payloads[0]["issue_source_ids"] = payloads[0]["issue_source_ids"] * 4
    else:
        payloads[0]["application_source_ids"] = (
            payloads[0]["application_source_ids"] * 2
        )
    with pytest.raises(ValueError):
        validate_plans(payloads, workflow, registry)


@pytest.mark.parametrize(
    "question",
    [
        "De school zegt dat hulp uit het samenwerkingsverband niet werkt. Zijn er eerdere zaken?",
        "Mijn behandelplek stopt en de school vindt de stap naar de gewone klas te groot. Wat vond de commissie?",
        "Hoe keek de commissie naar de afbouw van zorg en opbouw van onderwijs op school?",
    ],
)
def test_school_support_context_without_explicit_support_word(question):
    assert question_intake(question) is None


def test_reviewed_missing_part_reaches_writer_and_cannot_disappear():
    registry, _, workflow, plans = fixture()
    limit = "Of de genoemde begeleiding ook beschikbaar was, blijft onbekend."
    plans[0]["missing_question_parts"] = [limit]
    checked = validate_plans(plans, workflow, registry)
    writer = prepare_writing(checked, workflow, reviews(checked))
    (packet,) = writer["case_packets"]
    assert packet["reviewed_scope"]["missing_question_parts"] == [limit]
    assert limit in str(packet["request"]["input"])
    assert (
        packet["request"]["text"]["format"]["schema"]["properties"][
            "unanswered_question_parts"
        ]["minItems"]
        == 1
    )
    statement = dict(
        text="In deze zaak vond de commissie de begeleiding relevant.",
        source_ids=packet["case_markers"],
    )
    body = dict(
        unanswered_question_parts=[],
        cases=[
            dict(
                status="relevant_contribution",
                case_source_id=packet["case_markers"][0],
                answers_question_part="De rol van begeleiding.",
                relevance_explanation="Test.",
                finding=statement,
                supporting_arguments=[],
                limitations=[statement],
            )
        ],
    )
    with pytest.raises(ValueError, match="onbeantwoord vraagdeel"):
        hydrate_evidence(body, writer, registry)
