"""Round-trip the real corpus with synthetic ID selections; never call a model."""

import argparse
import json
from pathlib import Path

from .common import read, digest
from .grounding import prepare_case_findings, hydrate_case_findings, prepare_comparison
from .retrieval import Retrieval
from .source_registry import SourceRegistry


def check(study, output):
    engine = Retrieval(Path("data/e5"), Path("data/gpo.sqlite3"))
    registry = SourceRegistry(engine.documents, engine.sections)
    sections = {s["section_id"]: s for s in engine.sections}
    all_sources = registry.select(engine.documents)
    assert len({s["source_id"] for s in all_sources}) == len(all_sources)
    for s in all_sources:
        section = sections[s["section_id"]]
        assert s["quote"] == section["text"][s["start_offset"] : s["end_offset"]]
        assert s["role"] == "commissie" and s["page"] == section["page"]
        assert (
            s["source_url"]
            == engine.documents[s["case_number"]]["source_url"] + f"#page={s['page']}"
        )
    frozen = read(study / "bevroren.json")["hashes"]
    assert digest(study / "nieuwe_vragen.json") == frozen["nieuwe_vragen.json"]
    assert (
        digest(study.parent / "besliscontrole/referentie.json")
        == frozen["oude_referentie"]
    )
    questions = (
        read(study.parent / "besliscontrole/referentie.json")["questions"]
        + read(study / "nieuwe_vragen.json")["questions"]
    )
    report = []
    packets = []
    for q in questions:
        response = engine.search(q["question"], "vector")
        packet = prepare_case_findings(q["question"], response, registry)
        packets.append(dict(id=q["id"], prepared=packet))
        row = dict(id=q["id"], status=packet["status"], sources=len(packet["sources"]))
        if not q["gold"]:
            assert (
                packet["status"] == "needs_additional_sources"
                and packet["request"] is None
            )
        else:
            assert packet["status"] == "awaiting_new_cost_approval"
            payload = dict(
                cases=[],
                missing_information=[
                    "Synthetische offline controle, geen modelantwoord."
                ],
            )
            for c in response["results"]:
                sid = next(
                    s["source_id"]
                    for s in packet["sources"]
                    if s["case_number"] == c["case_number"]
                )
                payload["cases"].append(
                    dict(
                        case_source_id=sid,
                        conclusions=[
                            dict(
                                text="Dit is een onbeoordeelde technische testclaim.",
                                source_ids=[sid],
                            )
                        ],
                    )
                )
            hydrated = hydrate_case_findings(payload, packet, registry)
            assert all(
                c["citations"][0]["quote"]
                == registry.get(c["source_ids"][0], c["source_ids"])["quote"]
                for case in hydrated["cases"]
                for c in case["conclusions"]
            )
            try:
                prepare_comparison(q["question"], hydrated, registry, {})
            except ValueError:
                row["comparison_blocked_until_content_review"] = True
            else:
                raise AssertionError("Een onbeoordeelde claim mag niet door.")
            row["synthetic_round_trip_passed"] = True
            row["estimated_first_stage_max_usd"] = packet["estimated_max_cost_usd"]
        report.append(row)
    output.mkdir(parents=True, exist_ok=True)
    result = dict(
        registry_units=len(all_sources),
        all_quotes_and_provenance_verified=True,
        frozen_questions_unchanged=True,
        synthetic_checks_not_model_quality=True,
        paid_calls=0,
        results=report,
    )
    (output / "grounding-check.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (output / "source-id-packets.json").write_text(
        json.dumps(packets, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("local-results/offline-v2"))
    args = parser.parse_args()
    check(args.study, args.output)
