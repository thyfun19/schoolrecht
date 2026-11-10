"""Approved V1/N3 trial: four unique stage calls, $0.20, no retries."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import httpx

from .common import read, digest
from .grounding import (
    MODEL,
    hydrate_case_findings,
    prepare_comparison,
    hydrate_comparison,
)
from .retrieval import Retrieval
from .source_registry import SourceRegistry
from .run_experiment import api_key

BASE = Path("local-results/grounding-v2-trial")


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def request_hash(request):
    return hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()


def initialize(base=BASE):
    if base.exists():
        raise ValueError("Nieuwe proefmap bestaat al; geen budgetreset toegestaan.")
    packets = [
        p
        for p in read(Path("local-results/offline-v2/source-id-packets.json"))
        if p["id"] in ["V1", "N3"]
    ]
    if [p["id"] for p in packets] != ["V1", "N3"]:
        raise ValueError("Niet de vastgelegde twee vragen.")
    for p in packets:
        assert request_hash(p["prepared"]["request"]) == p["prepared"]["packet_sha256"]
    base.mkdir(parents=True)
    save(base / "packets.json", packets)
    save(
        base / "approval.json",
        dict(
            status="approved_by_user",
            date="2026-10-30",
            basis="Gebruiker geeft expliciet akkoord voor V1/N3, maximaal vier aanroepen zonder retries en USD 0.20 exclusief belasting.",
            model=MODEL,
            max_calls=4,
            budget_usd=0.20,
            packets_sha256=digest(base / "packets.json"),
            code_hashes={
                name: digest(Path(__file__).parent / name)
                for name in ["grounding.py", "source_registry.py"]
            },
            criteria=[
                "Iedere zaak apart; iedere claim steun uit eigen aangehaalde fragmenten.",
                "Geen nieuwe feiten, te ruime externe-verantwoordelijkheidsconclusie of persoonlijke juridische beoordeling.",
                "Alleen een vergelijkingsaanroep als alle afzonderlijke claims inhoudelijk worden gesteund.",
                "Vaste bronnen niet wijzigen of ontbrekende zaken toevoegen; N3-brongrens expliciet beoordelen.",
            ],
        ),
    )
    save(
        base / "ledger.json", dict(attempts=[], reserved_usd=0, known_actual_cost_usd=0)
    )


def call_once(base, question_id, stage, registry, *, key, client):
    approval = read(base / "approval.json")
    if (
        approval["status"] != "approved_by_user"
        or approval["budget_usd"] != 0.20
        or approval["max_calls"] != 4
        or approval["model"] != MODEL
        or digest(base / "packets.json") != approval["packets_sha256"]
    ):
        raise ValueError("Proef wijkt af van het akkoord.")
    for name, expected in approval["code_hashes"].items():
        if digest(Path(__file__).parent / name) != expected:
            raise ValueError("Voorbereidingscode gewijzigd na vastlegging.")
    packet = next(
        (p for p in read(base / "packets.json") if p["id"] == question_id), None
    )
    if packet is None or stage not in ["per_case", "comparison"]:
        raise ValueError("Vraag of stap buiten akkoord.")
    stage_name = f"{question_id}-{stage}"
    if stage == "per_case":
        prepared = packet["prepared"]
    else:
        review = read(base / f"{question_id}-review.json")
        if review["overall_verdict"] != "pass":
            raise ValueError(
                "Inhoudelijke review is niet geslaagd; geen vergelijkingsaanroep."
            )
        findings = read(base / f"{question_id}-per_case-hydrated.json")
        if review["findings_sha256"] != digest(
            base / f"{question_id}-per_case-hydrated.json"
        ):
            raise ValueError("Inhoudelijke review is niet geslaagd of niet actueel.")
        question = json.loads(packet["prepared"]["request"]["input"])["question"]
        prepared = prepare_comparison(question, findings, registry, review["claims"])
    if prepared.get("request") is None or prepared["request"]["model"] != MODEL:
        raise ValueError("Geen toegestane request.")
    allowed = {s["source_id"] for s in packet["prepared"]["sources"]}
    if any(
        s["source_id"] not in allowed or s != registry.get(s["source_id"], allowed)
        for s in prepared["sources"]
    ):
        raise ValueError("Broncontext wijkt af van de vastgelegde registratie.")
    lock = base / "trial.lock"
    handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        ledger = read(base / "ledger.json")
        if ledger.get("status") == "closed_after_report":
            raise ValueError(
                "Deze beperkte proef is afgesloten; geen verdere aanroepen."
            )
        if any(
            a["status"] == "cost_estimate_exceeded_stop" for a in ledger["attempts"]
        ):
            raise ValueError("Kostenafwijking: hele proef blijft gestopt.")
        if any(a["stage"] == stage_name for a in ledger["attempts"]):
            raise ValueError("Deze vraag/stap is al geprobeerd; geen retry.")
        maximum = prepared["estimated_max_cost_usd"]
        if len(ledger["attempts"]) >= 4 or ledger["reserved_usd"] + maximum > 0.20:
            raise ValueError("Goedgekeurd aanroep-/kostenplafond bereikt.")
        entry = dict(
            stage=stage_name,
            status="reserved_before_call",
            reserved_usd=maximum,
            request_sha256=request_hash(prepared["request"]),
        )
        ledger["attempts"].append(entry)
        ledger["reserved_usd"] += maximum
        save(base / "ledger.json", ledger)
        save(base / f"{stage_name}-prepared.json", prepared)
        try:
            response = client.post(
                "https://api.openai.com/v1/responses",
                headers={"Authorization": "Bearer " + key},
                json=prepared["request"],
            )
            if response.status_code != 200:
                entry.update(
                    status="api_error_no_retry",
                    http_status=response.status_code,
                    actual_cost_unknown=True,
                )
                return entry
            raw = response.json()
            save(base / f"{stage_name}-raw.json", raw)
            usage = raw.get("usage")
            if not usage or "input_tokens" not in usage or "output_tokens" not in usage:
                entry.update(status="usage_unknown_stop", actual_cost_unknown=True)
                return entry
            actual = (usage["input_tokens"] * 0.75 + usage["output_tokens"] * 4.5) / 1e6
            entry.update(usage=usage, actual_cost_usd=actual)
            ledger["known_actual_cost_usd"] += actual
            if actual > maximum:
                entry["status"] = "cost_estimate_exceeded_stop"
                return entry
            if raw.get("status") != "completed":
                entry["status"] = "incomplete_no_retry"
                return entry
            text = "".join(
                c["text"]
                for item in raw.get("output", [])
                if item.get("type") == "message"
                for c in item.get("content", [])
                if c.get("type") == "output_text"
            )
            parsed = json.loads(text)
            save(base / f"{stage_name}-parsed.json", parsed)
            hydrated = (
                hydrate_case_findings(parsed, prepared, registry)
                if stage == "per_case"
                else hydrate_comparison(parsed, prepared, registry)
            )
            save(base / f"{stage_name}-hydrated.json", hydrated)
            entry["status"] = "references_passed_content_review_pending"
        except (ValueError, KeyError, httpx.HTTPError) as error:
            entry["status"] = "validation_or_transport_failure_no_retry"
            entry["error_type"] = type(error).__name__
            if "actual_cost_usd" not in entry:
                entry["actual_cost_unknown"] = True
        finally:
            save(base / "ledger.json", ledger)
        return entry
    finally:
        os.close(handle)
        lock.unlink()


def close_trial(base=BASE):
    lock = base / "trial.lock"
    handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        ledger = read(base / "ledger.json")
        ledger["status"] = "closed_after_report"
        ledger["closure_reason"] = (
            "Beide eerste stappen afgekeurd; geen vergelijkingsaanroepen of retries. Gebruiker vroeg te stoppen na proefrapport."
        )
        save(base / "ledger.json", ledger)
    finally:
        os.close(handle)
        lock.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["init", "per_case", "comparison"])
    parser.add_argument("--question", choices=["V1", "N3"])
    args = parser.parse_args()
    if args.action == "init":
        initialize()
        print("Nieuw akkoord en ledger vastgelegd; geen aanroep gedaan.")
    else:
        engine = Retrieval(Path("data/e5"), Path("data/gpo.sqlite3"))
        registry = SourceRegistry(engine.documents, engine.sections)
        with httpx.Client(
            timeout=60, follow_redirects=False, transport=httpx.HTTPTransport(retries=0)
        ) as client:
            print(
                json.dumps(
                    call_once(
                        BASE,
                        args.question,
                        args.action,
                        registry,
                        key=api_key(Path(".env")),
                        client=client,
                    ),
                    indent=2,
                )
            )
