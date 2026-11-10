"""One approved fixed batch, max eight calls/$0.25; no retries or public generation route."""

import argparse
import json
import os
from pathlib import Path

import httpx

from .common import digest, read
from .explanation import MODEL, validate_explanation


def api_key(env_file):
    value = os.getenv("OPENAI_API_KEY")
    if not value and env_file.exists():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            if line.strip().startswith("OPENAI_API_KEY="):
                value = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
    if not value:
        raise ValueError(
            "OPENAI_API_KEY ontbreekt; geen API-aanroep gedaan. Stel de sleutel lokaal in, niet in de chat."
        )
    return value


def run(
    packets_file,
    approval_file,
    output,
    *,
    key,
    client,
    continue_unattempted=False,
    review_note="",
):
    approval = read(approval_file)
    if (
        approval.get("status") != "approved_by_user"
        or approval["model"] != MODEL
        or approval["max_calls"] != 8
        or approval["budget_usd"] != 0.25
        or approval["packets_sha256"] != digest(packets_file)
    ):
        raise ValueError("Akkoord past niet bij deze vaste proef.")
    packets = [p for p in read(packets_file) if p["prepared"]["request"] is not None]
    if (
        len(packets) != 8
        or sum(p["prepared"]["estimated_max_cost_usd"] for p in packets) > 0.25
    ):
        raise ValueError("Proef valt buiten akkoord.")
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "experiment.lock"
    # Exclusive lock prevents parallel CLI runs from spending twice.
    handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    ledger_file = output / "ledger.json"

    def persist(ledger):
        temp = output / "ledger.tmp"
        temp.write_text(
            json.dumps(ledger, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(temp, ledger_file)

    try:
        ledger = (
            read(ledger_file)
            if ledger_file.exists()
            else {"attempts": [], "reserved_usd": 0, "actual_cost_usd": 0}
        )
        if ledger["attempts"] and not continue_unattempted:
            raise ValueError(
                "Proef is al gestart. Geen automatische herhaling; beoordeel eerst het ledger."
            )
        attempted = {a["id"] for a in ledger["attempts"]}
        if len(attempted) != len(ledger["attempts"]) or not attempted.issubset(
            {p["id"] for p in packets}
        ):
            raise ValueError("Ledger bevat dubbele of onbekende pogingen.")
        if continue_unattempted:
            if len(review_note.strip()) < 10:
                raise ValueError(
                    "Leg eerst de inhoudelijke controle van de eerdere fout vast."
                )
            ledger.setdefault("manual_resumptions", []).append(
                dict(attempted_before_resume=sorted(attempted), review_note=review_note)
            )
            persist(ledger)
        for packet in packets:
            if packet["id"] in attempted:
                continue
            prepared = packet["prepared"]
            maximum = prepared["estimated_max_cost_usd"]
            if len(ledger["attempts"]) >= 8 or ledger["reserved_usd"] + maximum > 0.25:
                raise ValueError("Goedgekeurd budget bereikt.")
            attempt = dict(id=packet["id"], status="reserved", reserved_usd=maximum)
            ledger["attempts"].append(attempt)
            ledger["reserved_usd"] += maximum
            persist(ledger)  # Even a timeout may be billable. Never retry silently.
            try:
                response = client.post(
                    "https://api.openai.com/v1/responses",
                    headers={"Authorization": "Bearer " + key},
                    json=prepared["request"],
                )
                if response.status_code != 200:
                    attempt.update(status="api_error", http_status=response.status_code)
                    persist(ledger)
                    break
                raw = response.json()
                (output / f"{packet['id']}-raw.json").write_text(
                    json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                usage = raw.get("usage", {})
                actual = (
                    usage.get("input_tokens", 0) * 0.75
                    + usage.get("output_tokens", 0) * 4.5
                ) / 1e6
                ledger["actual_cost_usd"] += actual
                attempt.update(usage=usage, actual_cost_usd=actual)
                if raw.get("status") != "completed":
                    raise ValueError("Antwoord onvolledig of geweigerd.")
                text = "".join(
                    c["text"]
                    for item in raw.get("output", [])
                    if item.get("type") == "message"
                    for c in item.get("content", [])
                    if c.get("type") == "output_text"
                )
                validated = validate_explanation(json.loads(text), prepared)
                (output / f"{packet['id']}-checked.json").write_text(
                    json.dumps(validated, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                attempt["status"] = "citation_checks_passed_content_review_pending"
            except (ValueError, KeyError, httpx.HTTPError):
                attempt["status"] = "failed_or_unknown_review_required"
                persist(ledger)
                break
            persist(ledger)
            if attempt["actual_cost_usd"] > maximum:
                attempt["status"] = "cost_estimate_exceeded_stop"
                persist(ledger)
                break
        return ledger
    finally:
        os.close(handle)
        lock.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--packets",
        type=Path,
        default=Path("local-results/e5/explanation-packets.json"),
    )
    parser.add_argument(
        "--approval", type=Path, default=Path("local-results/e5/approval.json")
    )
    parser.add_argument("--output", type=Path, default=Path("local-results/e5/answers"))
    parser.add_argument("--continue-unattempted", action="store_true")
    parser.add_argument("--review-note", default="")
    args = parser.parse_args()
    try:
        key = api_key(Path(".env"))
        with httpx.Client(
            timeout=90, follow_redirects=False, transport=httpx.HTTPTransport(retries=0)
        ) as client:
            result = run(
                args.packets,
                args.approval,
                args.output,
                key=key,
                client=client,
                continue_unattempted=args.continue_unattempted,
                review_note=args.review_note,
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except ValueError as error:
        raise SystemExit(str(error))
