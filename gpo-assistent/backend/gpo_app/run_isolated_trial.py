"""Evaluation runner for five-case isolated retrieval trials."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import httpx

from gpo_local.run_experiment import api_key
from .database import create_engine
from .ingestion import json_hash
from .isolated_assessment import prepare, validate
from .run_assessment_trial import cost, read, save

MODEL = "gpt-5.4-mini-2026-03-17"
BUDGET = 0.20


def preflight(engine, source):
    proposal = read(source / "proposal.json")
    if proposal["proposal_sha256"] != json_hash(
        {k: v for k, v in proposal.items() if k != "proposal_sha256"}
    ):
        raise ValueError("Proefvoorstel gewijzigd.")
    if (
        proposal["model"] != MODEL
        or proposal["question_id"] != "V1"
        or proposal["max_calls"] != 5
        or proposal["retries"] != 0
        or proposal["comparison_calls"] != 0
        or proposal["proposed_budget_usd"] != BUDGET
    ):
        raise ValueError("Proefvoorstel valt buiten akkoord.")
    original = read(source / "V1.json")
    fresh = prepare(engine, original["question"], original["processing_ids"])
    if fresh != original or fresh["packet_sha256"] != proposal["source_packet_sha256"]:
        raise ValueError("Bronpakketten gewijzigd.")
    drafts = fresh["case_packets"]
    if (
        [p["case_number"] for p in drafts] != proposal["candidates"]
        or [p["request_sha256"] for p in drafts] != proposal["request_hashes"]
        or sum(p["estimated_max_cost_usd"] for p in drafts) > BUDGET
    ):
        raise ValueError("Aanvragen of kosten gewijzigd.")
    singles = []
    for pid, draft in zip(fresh["processing_ids"], drafts):
        expected = hashlib.sha256(
            json.dumps(draft["request"], sort_keys=True).encode()
        ).hexdigest()
        if expected != draft["request_sha256"]:
            raise ValueError("Aanvraaghash klopt niet.")
        single = prepare(engine, fresh["question"], [pid])
        if single["case_packets"] != [draft]:
            raise ValueError(
                "Afzonderlijke aanvraag wijkt af van goedgekeurde aanvraag."
            )
        singles.append(single)
    return proposal, drafts, singles


def initialize(engine, source, base):
    proposal, _, _ = preflight(engine, source)
    base.mkdir()  # Exclusive new run; never reset an existing ledger.
    save(
        base / "approval.json",
        dict(
            status="approved_by_user",
            date="2026-11-05",
            basis="Explicit user reply: Akkoord, maximaal $0,20",
            authorized_ceiling_usd=BUDGET,
            operational_budget_usd=BUDGET,
            model=MODEL,
            max_calls=5,
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


def execute(engine, source, base, *, client, key):
    proposal, drafts, singles = preflight(engine, source)
    approval = read(base / "approval.json")
    if (
        approval["status"] != "approved_by_user"
        or approval["operational_budget_usd"] != BUDGET
        or approval["authorized_ceiling_usd"] < BUDGET
        or approval["max_calls"] != 5
        or approval["retries"] != 0
        or approval["comparison_calls"] != 0
        or approval["model"] != MODEL
        or approval["proposal_sha256"] != proposal["proposal_sha256"]
    ):
        raise ValueError("Geen passend akkoord.")
    handle = os.open(base / "trial.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        ledger = read(base / "ledger.json")
        if ledger["status"] != "not_started" or ledger["attempts"]:
            raise ValueError("Proef al gestart; geen herhaling.")
        ledger["status"] = "running"
        for draft, single in zip(drafts, singles):
            if (
                len(ledger["attempts"]) >= 5
                or ledger["reserved_usd"] + draft["estimated_max_cost_usd"] > BUDGET
            ):
                raise ValueError("Plafond bereikt.")
            case = draft["case_number"]
            entry = dict(
                case_number=case,
                status="reserved_before_call",
                request_sha256=draft["request_sha256"],
                reserved_usd=draft["estimated_max_cost_usd"],
            )
            ledger["attempts"].append(entry)
            ledger["reserved_usd"] += entry["reserved_usd"]
            save(base / "ledger.json", ledger)
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
                    break
                raw = response.json()
                save(base / (case + "-raw.json"), raw)
                entry["usage"] = raw["usage"]
                entry["actual_cost_usd"] = cost(raw["usage"])
                ledger["actual_cost_usd"] += entry["actual_cost_usd"]
                if (
                    entry["actual_cost_usd"] > entry["reserved_usd"]
                    or ledger["actual_cost_usd"] > BUDGET
                ):
                    entry["status"] = "cost_deviation_stop"
                    break
                if raw.get("status") != "completed" or raw.get("model") != MODEL:
                    entry["status"] = "incomplete_or_wrong_model_stop"
                    break
                text = "".join(
                    c["text"]
                    for item in raw.get("output", [])
                    if item.get("type") == "message"
                    for c in item.get("content", [])
                    if c.get("type") == "output_text"
                )
                payload = json.loads(text)
                save(base / (case + "-parsed.json"), payload)
                hydrated = validate(engine, payload, single)
                save(base / (case + "-validated.json"), hydrated)
                entry["status"] = "citation_checks_passed_content_review_pending"
            except (ValueError, KeyError, httpx.HTTPError) as error:
                entry.update(status="failed_no_retry", error_type=type(error).__name__)
                if "actual_cost_usd" not in entry:
                    entry["actual_cost_unknown"] = True
                break
            finally:
                save(base / "ledger.json", ledger)
        ledger["status"] = "closed_after_trial"
        save(base / "ledger.json", ledger)
        return ledger
    finally:
        os.close(handle)
        (base / "trial.lock").unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["init", "preflight", "run"])
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("local-results/isolated-assessment-v4-final"),
    )
    parser.add_argument(
        "--base", type=Path, default=Path("local-results/isolated-assessment-trial-v4")
    )
    args = parser.parse_args()
    engine = create_engine()
    try:
        if args.action == "init":
            initialize(engine, args.source, args.base)
            print("New approval and ledger recorded; zero calls.")
        elif args.action == "preflight":
            preflight(engine, args.source)
            print(
                "All five approved requests and source contexts verified; zero calls."
            )
        else:
            preflight(engine, args.source)
            key = api_key(Path(".env"))
            with httpx.Client(
                timeout=90,
                follow_redirects=False,
                transport=httpx.HTTPTransport(retries=0),
            ) as client:
                print(
                    json.dumps(
                        execute(engine, args.source, args.base, client=client, key=key),
                        indent=2,
                    )
                )
    finally:
        engine.dispose()
