"""Offline report of actual v2 responses and separately recorded semantic reviews."""

import json
from pathlib import Path

from .common import read, digest
from .run_grounding_trial import BASE


def report(base=BASE):
    packets = {p["id"]: p["prepared"] for p in read(base / "packets.json")}
    ledger = read(base / "ledger.json")
    rows = []
    lines = [
        "# Werkelijke uitvoer beperkte v2-proef",
        "",
        "Afgekeurde modeluitvoer, geen gevalideerd advies. Citaten hieronder zijn door de backend uit de vaste registratie ingevoegd.",
        "",
    ]
    for qid in ["V1", "N3"]:
        packet = packets[qid]
        parsed = read(base / f"{qid}-per_case-parsed.json")
        sources = {s["source_id"]: s for s in packet["sources"]}
        review = read(base / f"{qid}-review.json")
        returned = []
        refs = 0
        valid = 0
        lines += [
            f"## {qid}",
            "",
            f"Beoordeling: {review['overall_verdict']}. {review['summary']}",
            "",
        ]
        for case in parsed["cases"]:
            marker = sources.get(case["case_source_id"])
            case_number = marker["case_number"] if marker else "onbekend"
            returned.append(case_number)
            lines += [f"### Zaak {case_number}", ""]
            for i, claim in enumerate(case["conclusions"], 1):
                lines += [f"**Claim {i}:** {claim['text']}", ""]
                for sid in claim["source_ids"]:
                    refs += 1
                    source = sources.get(sid)
                    if source is not None and source["case_number"] == case_number:
                        valid += 1
                        lines.append(
                            f"- [{source['case_number']}, p.{source['page']}]({source['source_url']}) — {source['quote']}"
                        )
                    else:
                        lines.append(
                            "- Ongeldige bronverwijzing; geen citaat ingevoegd."
                        )
                lines.append("")
        lines += ["**Ontbrekende informatie volgens het model:**", ""]
        lines += ["- " + s for s in parsed["missing_information"]]
        lines += [
            "",
            "**Beoordeling door dezelfde AI, geen onafhankelijke review:**",
            "",
            review["summary"],
            "",
        ]
        rows.append(
            dict(
                id=qid,
                retrieved_cases=sorted({s["case_number"] for s in sources.values()}),
                returned_cases=returned,
                reference_count=refs,
                known_ids_owned_by_correct_case=valid,
                review=review["overall_verdict"],
                comparison_called=any(
                    a["stage"] == qid + "-comparison" for a in ledger["attempts"]
                ),
            )
        )
    assert len(ledger["attempts"]) <= 4 and ledger["reserved_usd"] <= 0.20
    result = dict(
        rows=rows,
        calls=len(ledger["attempts"]),
        cost_usd=ledger["known_actual_cost_usd"],
        input_tokens=sum(
            a.get("usage", {}).get("input_tokens", 0) for a in ledger["attempts"]
        ),
        output_tokens=sum(
            a.get("usage", {}).get("output_tokens", 0) for a in ledger["attempts"]
        ),
        fixed_packets_unchanged=digest(base / "packets.json")
        == read(base / "approval.json")["packets_sha256"],
    )
    (base / "modeluitvoer-en-review.md").write_text("\n".join(lines), encoding="utf-8")
    (base / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    report()
