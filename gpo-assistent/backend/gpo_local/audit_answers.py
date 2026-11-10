"""Offline audit of real model output, including rejected answers. No API calls."""

import argparse
import json
from pathlib import Path

from .common import read


def audit(base):
    results = []
    for packet in read(base / "explanation-packets.json"):
        path = base / "answers" / f"{packet['id']}-raw.json"
        if not path.exists():
            continue
        raw = read(path)
        text = "".join(
            c["text"]
            for item in raw.get("output", [])
            if item.get("type") == "message"
            for c in item.get("content", [])
            if c.get("type") == "output_text"
        )
        entry = dict(
            id=packet["id"],
            status=raw.get("status"),
            question=packet["question"],
            claims=[],
        )
        try:
            answer = json.loads(text)
        except ValueError:
            entry["parse_error"] = True
            results.append(entry)
            continue
        sources = {s["source_id"]: s for s in packet["prepared"]["sources"]}
        for category in ["similarities", "differences"]:
            for index, claim in enumerate(answer.get(category, []), 1):
                citations = []
                for ref in claim["citations"]:
                    source = sources.get(ref["source_id"])
                    literal = source is not None and ref["quote"] in source["text"]
                    elsewhere = [
                        s["source_id"]
                        for s in sources.values()
                        if ref["quote"] in s["text"]
                    ]
                    citations.append(
                        dict(
                            **ref,
                            literal=literal,
                            other_exact_locations=elsewhere,
                            page=source["page"] if source else None,
                            source_url=source["source_url"] if source else None,
                        )
                    )
                entry["claims"].append(
                    dict(
                        category=category,
                        index=index,
                        text=claim["text"],
                        citations=citations,
                    )
                )
        refs = [ref for c in entry["claims"] for ref in c["citations"]]
        entry.update(
            citations=len(refs),
            literal_citations=sum(c["literal"] for c in refs),
            missing_information=answer.get("missing_information"),
            historical_notice=answer.get("historical_notice"),
            content_quality="requires_manual_review",
        )
        results.append(entry)
    (base / "answer-audit.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Werkelijke modelantwoorden — afgekeurde antwoorden inbegrepen",
        "",
        "Onderzoeksuitvoer, geen gevalideerde uitleg. Een bronlink of letterlijk citaat bewijst geen inhoudelijke juistheid.",
        "",
    ]
    for row in results:
        lines += [
            f"## {row['id']} — {row['question']}",
            "",
            f"Letterlijke citaten in opgegeven bronsectie: {row.get('literal_citations', 0)}/{row.get('citations', 0)}.",
            "",
        ]
        for claim in row["claims"]:
            lines += [f"**{claim['category']} {claim['index']}** — {claim['text']}", ""]
            for ref in claim["citations"]:
                status = (
                    "letterlijk"
                    if ref["literal"]
                    else "AFGEKEURD: niet letterlijk op opgegeven locatie"
                )
                lines.append(
                    f"- [{ref['source_id']}]({ref['source_url']}) ({status}): {ref['quote']}"
                )
            lines.append("")
        lines += ["**Ontbrekende informatie volgens het model:**", ""]
        lines += ["- " + value for value in row.get("missing_information", [])]
        lines += ["", row.get("historical_notice", ""), ""]
    (base / "modelantwoorden.md").write_text("\n".join(lines), encoding="utf-8")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("local-results/e5"))
    parser.add_argument("--only")
    args = parser.parse_args()
    rows = audit(args.base)
    for row in rows:
        if args.only is None or row["id"] == args.only:
            print(json.dumps(row, ensure_ascii=False, indent=2))
