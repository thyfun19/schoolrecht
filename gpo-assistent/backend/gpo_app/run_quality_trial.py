"""Staged retrieval and comparison evaluation across registered GPO decisions."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import httpx
from sqlalchemy.exc import SQLAlchemyError

from gpo_local.common import digest
from gpo_local.run_experiment import api_key
from .database import create_engine
from .ingestion import json_hash
from .isolated_assessment import (
    prepare as baseline_prepare,
    validate as baseline_validate,
)
from .evidence_workflow import (
    prepare as workflow_prepare,
    validate as plan_validate,
    writing,
)
from .case_assessment import RegisteredSources
from gpo_local.evidence_assessment import hydrate_evidence
from .run_assessment_trial import read, save, cost

MODEL = "gpt-5.4-mini-2026-03-17"
BUDGET = 0.50


def request_hash(request):
    return hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()


def preflight(engine, source):
    proposal = read(source / "proposal.json")
    if proposal["proposal_sha256"] != json_hash(
        {k: v for k, v in proposal.items() if k != "proposal_sha256"}
    ):
        raise ValueError("Voorstel gewijzigd.")
    if (
        proposal["model"] != MODEL
        or proposal["max_calls"] != 18
        or proposal["proposed_budget_usd"] != BUDGET
        or proposal["retries"]
        or proposal["comparison_calls"]
        or proposal["parent_final_answer"]
        or [q["id"] for q in proposal["inputs"]] != ["D62", "D65"]
    ):
        raise ValueError("Voorstel buiten akkoord.")
    for name, expected in proposal["code_hashes"].items():
        if digest(Path(name)) != expected:
            raise ValueError("Beoordelingscode gewijzigd na akkoord.")
    original_contract = Path("experiments/quality-target-v6.json")
    if digest(original_contract) != proposal["protocol_sha256"] or read(
        source / "contract.json"
    ) != read(original_contract):
        raise ValueError("Vaste vragen gewijzigd.")
    drafts, workflows = [], {}
    for question in proposal["inputs"]:
        qid = question["id"]
        baseline = read(source / (qid + "-baseline.json"))
        workflow = read(source / (qid + "-selection.json"))
        fresh = workflow_prepare(
            engine, question["question"], baseline["processing_ids"]
        )
        if (
            fresh != workflow
            or fresh["baseline"] != baseline
            or workflow["packet_sha256"] != question["selection_sha256"]
        ):
            raise ValueError("Broncontext of selectieaanvraag gewijzigd.")
        if (
            baseline["packet_sha256"] != question["baseline_sha256"]
            or [c["case_number"] for c in baseline["case_packets"]] != question["cases"]
        ):
            raise ValueError("Nulmeting of kandidaten gewijzigd.")
        workflows[qid] = workflow
        for stage, packets, expected_hashes in [
            ("baseline", baseline["case_packets"], question["baseline_request_hashes"]),
            (
                "selection",
                workflow["selection_requests"],
                question["selection_request_hashes"],
            ),
        ]:
            if (
                len(packets) != 3
                or [p["request_sha256"] for p in packets] != expected_hashes
            ):
                raise ValueError("Vaste aanvragen gewijzigd.")
            for pid, packet in zip(baseline["processing_ids"], packets):
                if request_hash(packet["request"]) != packet["request_sha256"]:
                    raise ValueError("Aanvraaghash ongeldig.")
                single = workflow_prepare(engine, question["question"], [pid])
                target = (
                    single["baseline"]["case_packets"][0]
                    if stage == "baseline"
                    else single["selection_requests"][0]
                )
                if target["request"] != packet["request"]:
                    raise ValueError("Afzonderlijke aanvraag wijkt af.")
                drafts.append(
                    dict(
                        stage=stage,
                        qid=qid,
                        case_number=packet["case_number"],
                        request=packet["request"],
                        request_sha256=packet["request_sha256"],
                        estimated_max_cost_usd=packet["estimated_max_cost_usd"],
                        single=single,
                    )
                )
    case_order = {
        (q["id"], case): i
        for q in proposal["inputs"]
        for i, case in enumerate(q["cases"])
    }
    drafts.sort(
        key=lambda d: (
            d["stage"] != "baseline",
            d["qid"],
            case_order[(d["qid"], d["case_number"])],
        )
    )
    return proposal, drafts, workflows


def initialize(engine, source, base):
    proposal, _, _ = preflight(engine, source)
    base.mkdir()
    save(
        base / "approval.json",
        dict(
            status="approved_by_user",
            date="2026-11-05",
            basis="Explicit user reply: Akkoord, maximaal $0,50",
            budget_usd=BUDGET,
            max_calls=18,
            model=MODEL,
            retries=0,
            comparison_calls=0,
            proposal_sha256=proposal["proposal_sha256"],
        ),
    )
    save(
        base / "ledger.json",
        dict(
            status="not_started",
            attempts=[],
            reserved_usd=0,
            actual_cost_usd=0,
            comparison_calls=0,
        ),
    )


def check_approval(base, proposal):
    approval = read(base / "approval.json")
    if (
        approval["status"] != "approved_by_user"
        or approval["proposal_sha256"] != proposal["proposal_sha256"]
        or approval["budget_usd"] != BUDGET
        or approval["max_calls"] != 18
        or approval["model"] != MODEL
        or approval["retries"]
        or approval["comparison_calls"]
    ):
        raise ValueError("Geen passend akkoord.")


def validate_draft(engine, draft, payload):
    if draft["stage"] == "baseline":
        return baseline_validate(engine, payload, draft["single"]["baseline"])
    if draft["stage"] == "selection":
        return plan_validate(engine, [payload], draft["single"])
    return hydrate_evidence(
        payload, draft["writer"], RegisteredSources(draft["writer"]["sources"])
    )


def call_one(
    engine,
    base,
    ledger,
    proposal,
    draft,
    client,
    key,
    *,
    budget=BUDGET,
    max_calls=18,
    validator=None,
):
    entry = dict(
        stage=draft["stage"],
        qid=draft["qid"],
        case_number=draft["case_number"],
        request_sha256=draft["request_sha256"],
        reserved_usd=draft["estimated_max_cost_usd"],
        status="reserved_before_call",
    )
    allocated = next(q for q in proposal["inputs"] if q["id"] == draft["qid"])[
        draft["stage"] + "_max_usd"
    ]
    prior = sum(
        a["reserved_usd"]
        for a in ledger["attempts"]
        if a["stage"] == draft["stage"] and a["qid"] == draft["qid"]
    )
    if (
        len(ledger["attempts"]) >= max_calls
        or ledger["reserved_usd"] + entry["reserved_usd"] > budget
        or prior + entry["reserved_usd"] > allocated + 1e-9
    ):
        raise ValueError("Proef- of deelbudget overschreden.")
    ledger["attempts"].append(entry)
    ledger["reserved_usd"] += entry["reserved_usd"]
    save(base / "ledger.json", ledger)
    prefix = "-".join([entry["stage"], entry["qid"], entry["case_number"]])
    try:
        response = client.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": "Bearer " + key},
            json=draft["request"],
        )
        if response.status_code != 200:
            entry.update(
                status="api_error_no_retry",
                http_status=response.status_code,
                actual_cost_unknown=True,
            )
            ledger["status"] = "closed_after_failure"
            return False
        raw = response.json()
        save(base / (prefix + "-raw.json"), raw)
        entry["usage"] = raw["usage"]
        entry["actual_cost_usd"] = cost(raw["usage"])
        ledger["actual_cost_usd"] += entry["actual_cost_usd"]
        if (
            entry["actual_cost_usd"] > entry["reserved_usd"]
            or ledger["actual_cost_usd"] > budget
        ):
            raise ValueError("Kostengrens overschreden.")
        if raw.get("status") != "completed" or raw.get("model") != MODEL:
            raise ValueError("Antwoord onvolledig of verkeerd model.")
        text = "".join(
            c["text"]
            for item in raw.get("output", [])
            if item.get("type") == "message"
            for c in item.get("content", [])
            if c.get("type") == "output_text"
        )
        payload = json.loads(text)
        save(base / (prefix + "-parsed.json"), payload)
        checked = (validator or validate_draft)(engine, draft, payload)
        save(base / (prefix + "-validated.json"), checked)
        entry["status"] = "technical_checks_passed_content_review_pending"
        return True
    except (ValueError, KeyError, httpx.HTTPError, SQLAlchemyError) as error:
        entry.update(status="failed_no_retry", error_type=type(error).__name__)
        if "actual_cost_usd" not in entry:
            entry["actual_cost_unknown"] = True
        ledger["status"] = "closed_after_failure"
        return False
    finally:
        save(base / "ledger.json", ledger)


def execute(engine, source, base, phase, *, client, key):
    if phase not in ["initial", "writing"]:
        raise ValueError("Onbekende proeffase.")
    proposal, drafts, workflows = preflight(engine, source)
    check_approval(base, proposal)
    handle = os.open(base / "trial.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        ledger = read(base / "ledger.json")
        required = "not_started" if phase == "initial" else "awaiting_content_review"
        if ledger["status"] != required:
            raise ValueError("Verkeerde fase; geen herhaling of retry.")
        if phase == "initial" and ledger["attempts"]:
            raise ValueError("Proef al gestart.")
        if phase == "writing":
            reviews = read(base / "reviews.json")
            drafts = []
            for question in proposal["inputs"]:
                qid = question["id"]
                validated = read(base / (qid + "-plans.json"))
                writer = writing(engine, validated, workflows[qid], reviews[qid])
                save(base / (qid + "-writing-prepared.json"), writer)
                for packet in writer["case_packets"]:
                    # Validate this response against exactly its one-case source subset.
                    single = dict(
                        writer,
                        case_packets=[packet],
                        case_markers=packet["case_markers"],
                        sources=packet["sources"],
                    )
                    single.pop("packet_sha256")
                    single["packet_sha256"] = json_hash(single)
                    drafts.append(
                        dict(
                            stage="writing",
                            qid=qid,
                            case_number=packet["case_number"],
                            request=packet["request"],
                            request_sha256=packet["request_sha256"],
                            estimated_max_cost_usd=packet["estimated_max_cost_usd"],
                            writer=single,
                        )
                    )
            save(base / "writing-drafts.json", drafts)
            ledger["reviews_sha256"] = digest(base / "reviews.json")
        ledger["status"] = "running_" + phase
        save(base / "ledger.json", ledger)
        for draft in drafts:
            if not call_one(engine, base, ledger, proposal, draft, client, key):
                return ledger
        if phase == "initial":
            for question in proposal["inputs"]:
                qid = question["id"]
                payloads = [
                    read(base / ("selection-" + qid + "-" + case + "-parsed.json"))
                    for case in question["cases"]
                ]
                save(
                    base / (qid + "-plans.json"),
                    plan_validate(engine, payloads, workflows[qid]),
                )
            ledger["status"] = "awaiting_content_review"
        else:
            ledger["status"] = "closed_after_trial"
        save(base / "ledger.json", ledger)
        return ledger
    except Exception:
        if "ledger" in locals() and ledger["status"].startswith("running_"):
            ledger["status"] = "closed_after_failure"
            save(base / "ledger.json", ledger)
        raise
    finally:
        os.close(handle)
        (base / "trial.lock").unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["init", "preflight", "initial", "writing"])
    parser.add_argument(
        "--source", type=Path, default=Path("local-results/quality-v6-final")
    )
    parser.add_argument(
        "--base", type=Path, default=Path("local-results/quality-comparison-v6")
    )
    args = parser.parse_args()
    engine = create_engine()
    try:
        if args.action == "init":
            initialize(engine, args.source, args.base)
            print("Approval and new staged ledger recorded; zero API calls.")
        elif args.action == "preflight":
            preflight(engine, args.source)
            print(
                "Fixed questions, code hashes, source packets and requests verified; zero API calls."
            )
        else:
            proposal, _, _ = preflight(engine, args.source)
            check_approval(args.base, proposal)
            key = api_key(Path(".env"))
            with httpx.Client(
                timeout=90,
                follow_redirects=False,
                transport=httpx.HTTPTransport(retries=0),
            ) as client:
                print(
                    json.dumps(
                        execute(
                            engine,
                            args.source,
                            args.base,
                            args.action,
                            client=client,
                            key=key,
                        ),
                        indent=2,
                    )
                )
    finally:
        engine.dispose()
