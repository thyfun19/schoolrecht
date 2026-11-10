"""Offline-only replay of review gates. No network client or paid execution route."""

from .grounding import Comparison, NOTICE
from .question_grounding import (
    PROTOCOL,
    assemble_reviewed_explanation,
    hydrate_question_assessment,
    no_parent_source_requests,
)


def prepare_reviewed_comparison(assessments, reviews, registry):
    reviewed = assemble_reviewed_explanation(assessments, reviews)
    contributions = reviewed["explanation"]["contributions"]
    if len(contributions) < 2:
        return dict(
            protocol=PROTOCOL,
            status="not_enough_relevant_cases",
            context=[],
            sources=[],
            paid_calls=0,
        )
    context = []
    sources = {}
    for case in contributions:
        statements = [case["finding"]] + case["limitations"]
        for statement in statements:
            for citation in statement["citations"]:
                source = registry.get(citation["source_id"], [citation["source_id"]])
                if citation != source or source["case_number"] != case["case_number"]:
                    raise ValueError("Niet-canonieke bron in gecontroleerde bijdrage.")
                sources[source["source_id"]] = source
        context.append(
            dict(
                case_source_id=case["finding"]["citations"][0]["source_id"],
                answers_question_part=case["answers_question_part"],
                finding=dict(
                    text=case["finding"]["text"],
                    source_ids=[c["source_id"] for c in case["finding"]["citations"]],
                ),
                limitations=[
                    dict(
                        text=s["text"],
                        source_ids=[c["source_id"] for c in s["citations"]],
                    )
                    for s in case["limitations"]
                ],
            )
        )
    return dict(
        protocol=PROTOCOL,
        status="offline_context_only",
        question=assessments["question"],
        context=context,
        sources=list(sources.values()),
        unanswered_question_parts=reviewed["explanation"]["unanswered_question_parts"],
        paid_calls=0,
    )


def validate_reviewed_comparison(payload, prepared, registry):
    if prepared.get("status") != "offline_context_only":
        raise ValueError("Geen gecontroleerde vergelijkingscontext.")
    data = Comparison.model_validate(payload)
    if not data.similarities and not data.differences:
        raise ValueError("Geen vergelijking.")
    allowed = {s["source_id"] for s in prepared["sources"]}
    output = {}
    for kind in ["similarities", "differences"]:
        output[kind] = []
        for claim in getattr(data, kind):
            refs = [registry.get(sid, allowed) for sid in claim.source_ids]
            if len({s["case_number"] for s in refs}) < 2:
                raise ValueError("Vergelijking moet op twee zaken steunen.")
            no_parent_source_requests(claim.text)
            output[kind].append(
                dict(
                    text=claim.text, citations=refs, semantic_support="requires_review"
                )
            )
    for part in data.missing_information:
        no_parent_source_requests(part)
    return dict(
        **output,
        missing_information=data.missing_information,
        historical_notice=NOTICE,
        answer_quality="not_validated",
        citation_integrity="deterministic",
    )


def replay_offline(prepared, registry, client, reviewer):
    """client supplies prerecorded fixtures only; this module never constructs an API request.

    Reviewer is trusted local code. Do not expose reviewer verdicts as model-controlled fields.
    Each stage is invoked once; failures propagate immediately, with no retries.
    """
    first = client.replay(
        "assessment",
        dict(
            question=prepared["question"],
            sources=prepared["sources"],
            case_markers=prepared["case_markers"],
        ),
    )
    assessments = hydrate_question_assessment(first, prepared, registry)
    reviews = reviewer(assessments)
    context = prepare_reviewed_comparison(assessments, reviews, registry)
    if context["status"] != "offline_context_only":
        return dict(
            assessments=assessments,
            comparison=None,
            comparison_status=context["status"],
            paid_calls=0,
        )
    second = client.replay("comparison", context)
    return dict(
        assessments=assessments,
        comparison=validate_reviewed_comparison(second, context, registry),
        comparison_status="references_passed_content_review_pending",
        paid_calls=0,
    )
