"""Offline candidate coverage experiment. No production ranking or generation change."""

import hashlib
import json
from pathlib import Path

from .common import digest, norm, read
from .retrieval import Retrieval
from .source_registry import SourceRegistry
from .explanation import scope_boundary
from .check_question_focus import scenario

EXTRA_HASH = "b9ed2e01cbc2f2e55fda27e1eb80ead172193b2133e8fa0cda0014a905c355a8"


def stable_union(first, second):
    return list(dict.fromkeys(first + second))


def measure(question, candidates, registry):
    expected = {g["case"] for g in question["gold"]}
    sections = [s for c in candidates for s in c["parts"]["commissieoverwegingen"]]
    sources = registry.select([c["case_number"] for c in candidates])
    # Resolve every source through the canonical registry; no model-provided metadata.
    assert all(
        registry.get(s["source_id"], set(x["source_id"] for x in sources)) == s
        for s in sources
    )
    for source in sources:
        assert source["source_id"].startswith("src_") and source["page"] > 0
        assert (
            source["source_url"].endswith(f"#page={source['page']}")
            and source["role"] == "commissie"
        )
    anchors = [
        any(
            s["case"] == g["case"]
            and s["page"] == g["page"]
            and g["anchor"].lower() in norm(s["text"]).lower()
            for s in sections
        )
        for g in question["gold"]
    ]
    cases = [c["case_number"] for c in candidates]
    return dict(
        cases=cases,
        case_count=len(cases),
        reference_case_coverage=len(expected & set(cases)) / len(expected)
        if expected
        else None,
        anchor_coverage=sum(anchors) / len(anchors) if anchors else None,
        anchors_found=anchors,
        non_reference_cases=[c for c in cases if c not in expected],
        non_reference_is_not_automatically_irrelevant=True,
        committee_chars=sum(len(s["text"]) for s in sections),
        reasoning_and_conclusion_chars=sum(len(s["quote"]) for s in sources),
        source_count=len(sources),
        source_ids_sha256=hashlib.sha256(
            json.dumps([s["source_id"] for s in sources]).encode()
        ).hexdigest(),
    )


def compare(output=Path("local-results/candidate-union")):
    study = Path("../../onderzoek_gpo/vectoronderzoek")
    frozen = read(study / "bevroren.json")["hashes"]
    assert digest(study / "nieuwe_vragen.json") == frozen["nieuwe_vragen.json"]
    assert (
        digest(study.parent / "besliscontrole/referentie.json")
        == frozen["oude_referentie"]
    )
    extra = Path("experiments/candidate-union-questions.json")
    assert digest(extra) == EXTRA_HASH, "Extra vragen gewijzigd na vastlegging."
    original = (
        read(study.parent / "besliscontrole/referentie.json")["questions"]
        + read(study / "nieuwe_vragen.json")["questions"]
    )
    engine = Retrieval(Path("data/e5"), Path("data/gpo.sqlite3"))
    registry = SourceRegistry(engine.documents, engine.sections)
    rows = []
    for q in original + read(extra)["questions"]:
        if scope_boundary(q["question"]):
            rows.append(
                dict(
                    id=q["id"],
                    question=q["question"],
                    blocked=True,
                    answerability="needs_additional_sources",
                )
            )
            continue
        retrieved = {
            m: engine.search(q["question"], m, 6)["results"] for m in ["text", "vector"]
        }
        ranks = {m: [c["case_number"] for c in cases] for m, cases in retrieved.items()}
        union = stable_union(ranks["text"][:3], ranks["vector"][:3])
        lookup = {c["case_number"]: c for cases in retrieved.values() for c in cases}
        candidates = dict(
            text3=retrieved["text"][:3],
            vector3=retrieved["vector"][:3],
            union=[lookup[c] for c in union],
            text6=retrieved["text"],
            vector6=retrieved["vector"],
            text_equal_count=retrieved["text"][: len(union)],
            vector_equal_count=retrieved["vector"][: len(union)],
        )
        assert len(union) <= 6
        # Repeat both actual queries, not only the union helper, to check runtime determinism.
        repeated = {
            m: [c["case_number"] for c in engine.search(q["question"], m, 6)["results"]]
            for m in ranks
        }
        assert repeated == ranks
        results = {m: measure(q, items, registry) for m, items in candidates.items()}
        if q["id"] in ["V1", "N3"]:
            sources = registry.select(union)
            markers = [
                next(s["source_id"] for s in sources if s["case_number"] == c)
                for c in union
            ]
            reference = scenario(
                q["id"],
                dict(sources=sources, case_markers=markers, question=q["question"]),
            )
            status = {
                registry.get(c["case_source_id"], set(markers))["case_number"]: c[
                    "status"
                ]
                for c in reference["cases"]
            }
            results["union"]["manual_reference_relevance"] = status
            results["union"]["relevance_label"] = (
                "Same AI manual reference scenario; not independently validated or model output"
            )
        rows.append(
            dict(
                id=q["id"],
                question=q["question"],
                blocked=False,
                answerability="not_assessed",
                ranking=ranks,
                deterministic=True,
                methods=results,
            )
        )
    groups = {}
    for label, selected in [
        (
            "existing",
            [r for r in rows if not r["blocked"] and not r["id"].startswith("H")],
        ),
        ("extra", [r for r in rows if r["id"].startswith("H")]),
    ]:
        groups[label] = {
            m: {
                key: sum(r["methods"][m][key] for r in selected) / len(selected)
                for key in [
                    "reference_case_coverage",
                    "anchor_coverage",
                    "case_count",
                    "committee_chars",
                    "reasoning_and_conclusion_chars",
                ]
            }
            for m in [
                "text3",
                "vector3",
                "union",
                "text6",
                "vector6",
                "text_equal_count",
                "vector_equal_count",
            ]
        }
    output.mkdir(parents=True, exist_ok=True)
    result = dict(
        paid_calls=0,
        production_changed=False,
        extra_questions_sha256=EXTRA_HASH,
        caveat="Reference is incomplete and made by the same AI. More candidates do not prove better ranking.",
        summary=groups,
        results=rows,
    )
    (output / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            dict(summary=groups, details=[r for r in rows if r["id"] in ["V1", "N3"]]),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    compare()
