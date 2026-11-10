"""Registered adapter for the experimental evidence-first workflow."""

from gpo_local.evidence_workflow import prepare_plans, validate_plans, prepare_writing
from gpo_local.evidence_assessment import hydrate_evidence

from .case_assessment import RegisteredSources
from .isolated_assessment import prepare as prepare_baseline


def prepare(engine, question, processing_ids):
    return prepare_plans(prepare_baseline(engine, question, processing_ids))


def fresh_workflow(engine, workflow):
    baseline = workflow["baseline"]
    fresh = prepare(engine, workflow["question"], baseline["processing_ids"])
    if fresh != workflow:
        raise ValueError("Workflow of bronregister gewijzigd.")
    return fresh


def validate(engine, payloads, workflow):
    fresh = fresh_workflow(engine, workflow)
    return validate_plans(
        payloads, fresh, RegisteredSources(fresh["baseline"]["sources"])
    )


def writing(engine, validated, workflow, reviews):
    fresh = fresh_workflow(engine, workflow)
    rechecked = validate(engine, [r["plan"] for r in validated["plans"]], fresh)
    if rechecked != validated:
        raise ValueError("Beoordeeld bewijsplan gewijzigd.")
    return prepare_writing(rechecked, fresh, reviews)


def validate_output(engine, payload, validated, workflow, reviews):
    prepared = writing(engine, validated, workflow, reviews)
    return hydrate_evidence(
        payload, prepared, RegisteredSources(workflow["baseline"]["sources"])
    )
