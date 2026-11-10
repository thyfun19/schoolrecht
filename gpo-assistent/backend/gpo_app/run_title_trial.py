"""Evaluation runner for title-based single-case baseline trials."""

import sys
from pathlib import Path

import httpx

from gpo_local.common import digest
from gpo_local.evidence_assessment import hydrate_evidence
from gpo_local.evidence_workflow import prepare_writing, validate_plans
from gpo_local.isolated_assessment import fingerprint
from gpo_local.run_experiment import api_key
from .case_assessment import RegisteredSources
from .run_quality_trial import MODEL, call_one, read, request_hash, save

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "local-results/title-human-review/generation-proposal"
RESULT = SOURCE.parent / "approved-trial"
PROPOSAL = ROOT / "experiments/title-evidence-proposal.json"
APPROVED_HASH = "aa603afbc2d28d99a3c7fd19e781f554ea5187ca381419dc2077080428c4b6ea"


def packets():
    if digest(PROPOSAL) != APPROVED_HASH:
        raise ValueError("Voorstel gewijzigd")
    result = []
    for item in read(PROPOSAL)["packets"]:
        workflow = read(SOURCE / (item["id"] + ".json"))
        if workflow["packet_sha256"] != item["packet_hash"] or (
            item["selection_limit"]
            and fingerprint({k: v for k, v in workflow.items() if k != "packet_sha256"})
            != item["packet_hash"]
        ):
            raise ValueError("Bronpakket gewijzigd")
        if item["selection_limit"]:
            (packet,) = workflow["selection_requests"]
            if (
                packet["case_number"] != item["case"]
                or packet["estimated_max_cost_usd"] != item["selection_limit"]
                or packet["request"]["model"] != MODEL
                or request_hash(packet["request"]) != packet["request_sha256"]
            ):
                raise ValueError("Aanvraag gewijzigd")
            result.append((item, workflow, packet))
    return result


def validate(_, draft, payload):
    if draft["stage"] == "selection":
        w = draft["workflow"]
        return validate_plans([payload], w, RegisteredSources(w["baseline"]["sources"]))
    w = draft["writer"]
    return hydrate_evidence(payload, w, RegisteredSources(w["sources"]))


def run(phase):
    fixed = packets()
    if phase == "check":
        print(len(fixed), "verified packets")
        return
    if phase not in ("selection", "writing"):
        raise ValueError("Onbekende fase")
    key = api_key(ROOT / ".env")
    drafts = []
    if phase == "selection":
        RESULT.mkdir(exist_ok=False)
        ledger = dict(
            status="running_selection",
            approval_sha256=APPROVED_HASH,
            budget_usd=0.42,
            reserved_usd=0,
            actual_cost_usd=0,
            attempts=[],
            skipped=[],
        )
        for item, workflow, packet in fixed:
            drafts.append(dict(packet, stage=phase, qid=item["id"], workflow=workflow))
    else:
        ledger = read(RESULT / "ledger.json")
        if (
            ledger["status"] != "awaiting_content_review"
            or ledger["approval_sha256"] != APPROVED_HASH
        ):
            raise ValueError("Proef gesloten of nog niet beoordeeld")
        reviews = read(RESULT / "reviews.json")
        for item, workflow, packet in fixed:
            qid = item["id"]
            prefix = "selection-" + qid + "-" + item["case"]
            validated = validate(
                None,
                dict(stage="selection", workflow=workflow),
                read(RESULT / (prefix + "-parsed.json")),
            )
            writer = prepare_writing(validated, workflow, reviews[qid])
            for case in writer["case_packets"]:
                if not case.get("request") or case["estimated_max_cost_usd"] > 0.03:
                    ledger["skipped"].append(qid)
                    continue
                if (
                    request_hash(case["request"]) != case["request_sha256"]
                    or case["request"]["model"] != MODEL
                ):
                    raise ValueError("Schrijfaanvraag gewijzigd")
                drafts.append(dict(case, stage=phase, qid=qid, writer=writer))
        ledger["status"] = "running_writing"
    (RESULT / (phase + ".lock")).touch(exist_ok=False)
    save(RESULT / "ledger.json", ledger)
    limits = dict(
        inputs=[
            dict(
                id=i["id"], selection_max_usd=i["selection_limit"], writing_max_usd=0.03
            )
            for i, _, _ in fixed
        ]
    )
    with httpx.Client(
        timeout=90, follow_redirects=False, transport=httpx.HTTPTransport(retries=0)
    ) as client:
        for draft in drafts:
            if not call_one(
                None,
                RESULT,
                ledger,
                limits,
                draft,
                client,
                key,
                budget=0.42,
                max_calls=16,
                validator=validate,
            ):
                print(ledger["status"], draft["qid"], flush=True)
                return
            print(phase, draft["qid"], "validated", flush=True)
    ledger["status"] = (
        "awaiting_content_review" if phase == "selection" else "closed_after_trial"
    )
    save(RESULT / "ledger.json", ledger)


if __name__ == "__main__":
    run(sys.argv[1])
