"""Registered full-context adapter for isolated case assessments."""

from gpo_local.evidence_assessment import prepare_evidence, hydrate_evidence
from gpo_local.evidence_policy import classify_sources
from gpo_local.question_intake import question_intake

from .assessment_trial_plan import marked_section
from .case_assessment import prepare as prepare_context, RegisteredSources
from .ingestion import json_hash


def prepare(engine, question, processing_ids):
    decision = question_intake(question)
    if decision:
        return decision
    registered = prepare_context(engine, question, processing_ids)
    sources = {s["source_id"]: s for s in registered["sources"]}
    policy = classify_sources(registered["sources"])
    eligible = {r["source_id"] for r in policy if r["basis_eligible"]}
    cutoffs = {
        (r["case_number"], r["section_id"]): r["administrative_start"] for r in policy
    }
    contexts = {}
    for case in registered["case_contexts"]:
        contexts[case["case_number"]] = []
        for category, sections in case["parts"].items():
            for section in sections:
                cutoff = cutoffs.get((case["case_number"], section["section_key"]))
                text = section["text"] if cutoff is None else section["text"][:cutoff]
                if not text.strip():
                    continue
                content = dict(
                    section,
                    text=text,
                    source_ids=[
                        sid for sid in section["source_ids"] if sid in eligible
                    ],
                )
                contexts[case["case_number"]].append(
                    dict(category=category, text=marked_section(content, sources))
                )
    result = prepare_evidence(
        question,
        [c["case_number"] for c in registered["case_contexts"]],
        registered["sources"],
        full_contexts=contexts,
    )
    result.pop("packet_sha256")
    result["processing_ids"] = processing_ids
    result["registry_context_sha256"] = registered["context_sha256"]
    # Same deterministic serialization as the local packet builder, with registry binding included.
    result["packet_sha256"] = json_hash(result)
    return result


def validate(engine, payload, prepared):
    fresh = prepare(engine, prepared["question"], prepared["processing_ids"])
    if fresh != prepared:
        raise ValueError("Bronpakket of registercontext gewijzigd na voorbereiding.")
    return hydrate_evidence(payload, fresh, RegisteredSources(fresh["sources"]))
