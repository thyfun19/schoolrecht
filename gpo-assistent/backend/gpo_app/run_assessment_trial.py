"""Execute case-assessment evaluation drafts against the registered GPO corpus."""

import argparse
import json
import os
from pathlib import Path

import httpx

from gpo_local.run_experiment import api_key
from .assessment_trial_plan import MODEL, prepare_request
from .case_assessment import prepare, validate
from .database import create_engine
from .ingestion import json_hash


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def preflight(engine, base, source):
    proposal = read(base / "proposal.json")
    drafts = read(base / "drafts.json")
    if (
        proposal["model"] != MODEL
        or proposal["max_calls"] != 2
        or proposal["budget_usd"] != 0.20
        or proposal["retries"] != 0
        or proposal["comparison_calls"] != 0
        or proposal["drafts_sha256"] != json_hash(drafts)
        or [d["id"] for d in drafts] != ["V1", "V4"]
    ):
        raise ValueError("Voorstel wijkt af van akkoord.")
    if sum(d["estimated_max_usd"] for d in drafts) > 0.20:
        raise ValueError("Raming buiten akkoord.")
    for draft in drafts:
        packet = read(source / (draft["id"] + ".json"))
        fresh = prepare(
            engine,
            packet["question"],
            [c["processing_id"] for c in packet["case_contexts"]],
        )
        if fresh["context_sha256"] != packet["context_sha256"]:
            raise ValueError("Broncontext gewijzigd.")
        rebuilt = dict(id=draft["id"], **prepare_request(fresh))
        if rebuilt != draft:
            raise ValueError("Conceptaanvraag gewijzigd.")
    return proposal, drafts


def record_approval(base):
    ledger = read(base / "ledger.json")
    if ledger["status"] != "not_started" or ledger["attempts"]:
        raise ValueError("Geen budgetreset of herstart toegestaan.")
    value = dict(
        status="approved_by_user",
        basis="Explicit chat reply: Akkoord: maximaal $0,20",
        date="2026-11-01",
        model=MODEL,
        max_calls=2,
        retries=0,
        budget_usd=0.20,
        drafts_sha256=read(base / "proposal.json")["drafts_sha256"],
        comparison_calls=0,
    )
    with (base / "approval.json").open("x", encoding="utf-8") as out:
        json.dump(value, out, indent=2)


def cost(usage):
    input_tokens = usage["input_tokens"]
    output_tokens = usage["output_tokens"]
    cached = usage.get("input_tokens_details", {}).get("cached_tokens", 0)
    if not 0 <= cached <= input_tokens or min(input_tokens, output_tokens) < 0:
        raise ValueError("Ongeldige usage.")
    return round(
        ((input_tokens - cached) * 0.75 + cached * 0.075 + output_tokens * 4.5) / 1e6,
        10,
    )


def execute(engine, base, source, *, client, key):
    proposal, drafts = preflight(engine, base, source)
    approval = read(base / "approval.json")
    if (
        approval["status"] != "approved_by_user"
        or approval["drafts_sha256"] != proposal["drafts_sha256"]
        or approval["max_calls"] != 2
        or approval["budget_usd"] != 0.20
        or approval["retries"] != 0
        or approval["model"] != MODEL
        or approval["comparison_calls"] != 0
    ):
        raise ValueError("Geen passend akkoord.")
    handle = os.open(base / "trial.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        ledger = read(base / "ledger.json")
        if ledger["status"] != "not_started" or ledger["attempts"]:
            raise ValueError("Proef al gestart; geen retry of herhaling.")
        ledger.update(status="running", reserved_usd=0, actual_cost_usd=0)
        for draft in drafts:
            if (
                len(ledger["attempts"]) >= 2
                or ledger["reserved_usd"] + draft["estimated_max_usd"] > 0.20
            ):
                raise ValueError("Plafond bereikt.")
            entry = dict(
                id=draft["id"],
                status="reserved_before_call",
                request_sha256=draft["request_sha256"],
                reserved_usd=draft["estimated_max_usd"],
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
                    save(base / (draft["id"] + "-error.json"), response.json())
                    break
                raw = response.json()
                save(base / (draft["id"] + "-raw.json"), raw)
                entry["usage"] = raw["usage"]
                entry["actual_cost_usd"] = cost(raw["usage"])
                ledger["actual_cost_usd"] += entry["actual_cost_usd"]
                if (
                    entry["actual_cost_usd"] > entry["reserved_usd"]
                    or ledger["actual_cost_usd"] > 0.20
                ):
                    entry["status"] = "cost_deviation_stop"
                    break
                if raw.get("status") != "completed":
                    entry["status"] = "incomplete_no_retry"
                    break
                text = "".join(
                    c["text"]
                    for item in raw.get("output", [])
                    if item.get("type") == "message"
                    for c in item.get("content", [])
                    if c.get("type") == "output_text"
                )
                payload = json.loads(text)
                save(base / (draft["id"] + "-parsed.json"), payload)
                checked = validate(
                    engine, payload, read(source / (draft["id"] + ".json"))
                )
                save(base / (draft["id"] + "-validated.json"), checked)
                entry["status"] = "citation_checks_passed_content_review_pending"
            except (ValueError, KeyError, httpx.HTTPError) as error:
                entry.update(status="failed_no_retry", error_type=type(error).__name__)
                if "actual_cost_usd" not in entry:
                    entry["actual_cost_unknown"] = True
                break
            finally:
                save(base / "ledger.json", ledger)
        ledger["status"] = "closed_after_trial"
        ledger["comparison_calls"] = 0
        save(base / "ledger.json", ledger)
        return ledger
    finally:
        os.close(handle)
        (base / "trial.lock").unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["approve", "run", "preflight"])
    parser.add_argument(
        "--base", type=Path, default=Path("local-results/case-assessment-trial-v1")
    )
    parser.add_argument(
        "--source", type=Path, default=Path("local-results/case-assessment-v1")
    )
    args = parser.parse_args()
    if args.action == "approve":
        record_approval(args.base)
        print("Explicit approval recorded; zero calls.")
    else:
        engine = create_engine()
        try:
            preflight(engine, args.base, args.source)
            if args.action == "preflight":
                print(
                    "Both frozen requests and current source contexts verified; zero calls."
                )
            else:
                key = api_key(Path(".env"))
                with httpx.Client(
                    timeout=90,
                    follow_redirects=False,
                    transport=httpx.HTTPTransport(retries=0),
                ) as client:
                    print(
                        json.dumps(
                            execute(
                                engine, args.base, args.source, client=client, key=key
                            ),
                            indent=2,
                        )
                    )
        finally:
            engine.dispose()
