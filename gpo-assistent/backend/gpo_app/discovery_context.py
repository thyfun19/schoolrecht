"""Offline lexical discovery scope experiment; explanation evidence stays canonical."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import sqlalchemy as sa

from gpo_local.common import digest, read
from gpo_local.retrieval import BM25
from . import models as m
from .chunking import LocalTokenCounter, paragraph_cuts, split_span
from .database import create_engine
from .ingestion import identity, json_hash
from .model_comparison import frozen as original_inputs, case_ranking, metrics, save

DISCOVERY_CATEGORIES = {"feiten", "partijstandpunten"}


def auxiliary_fragments(case, processing_id, version_id, url, sections, pages, counter):
    page_map = {p["page_number"]: p for p in pages}
    output = []
    for section in sorted(sections, key=lambda s: s["ordinal"]):
        if section["category"] not in DISCOVERY_CATEGORIES:
            continue
        if section["role"] == "commissie":
            raise ValueError(
                "Aanvullende zoekcontext heeft een tegenstrijdige bronrol."
            )
        cuts, status = paragraph_cuts(section, page_map[section["page_number"]])
        for a, b in zip(cuts, cuts[1:]):
            for start, end, reason in split_span(section["text"], a, b, counter, 384):
                text = section["text"][start:end]
                output.append(
                    dict(
                        id=identity(
                            "aux",
                            processing_id,
                            section["section_key"],
                            start,
                            end,
                            json_hash(text),
                        ),
                        case=case,
                        processing_id=processing_id,
                        version_id=version_id,
                        page=section["page_number"],
                        section_key=section["section_key"],
                        section=section["heading"],
                        role=section["role"],
                        category=section["category"],
                        start_offset=start,
                        end_offset=end,
                        text=text,
                        token_count=counter(text),
                        source_url=url + f"#page={section['page_number']}",
                        evidence_eligible=False,
                        citation_source_ids=[],
                        split_reason=status + ":" + reason,
                    )
                )
    return output


def freeze(root, previous, protocol_path, tokenizer_path):
    if (root / "frozen.json").exists():
        raise ValueError("Deze proef is al bevroren; niet overschrijven.")
    questions, primary, contexts = original_inputs(previous)
    counter = LocalTokenCounter(tokenizer_path)
    if not (previous / "results.json").exists():
        raise ValueError("Vorige vergelijking ontbreekt.")
    auxiliary = []
    snapshots = []
    engine = create_engine()
    try:
        with engine.connect() as con:
            for case, context in sorted(contexts.items()):
                pid = context["processing_id"]
                sections = [
                    dict(r)
                    for r in con.execute(
                        sa.select(m.sections)
                        .where(m.sections.c.processing_id == pid)
                        .order_by(m.sections.c.ordinal)
                    ).mappings()
                ]
                pages = [
                    dict(r)
                    for r in con.execute(
                        sa.select(m.pages)
                        .where(m.pages.c.processing_id == pid)
                        .order_by(m.pages.c.page_number)
                    ).mappings()
                ]
                lookup = {s["section_key"]: s for s in sections}
                for parent in context["parts"].values():
                    for span in parent:
                        section = lookup[span["section_key"]]
                        if (
                            section["text"][span["start_offset"] : span["end_offset"]]
                            != span["text"]
                            or section["role"] != span["role"]
                            or section["page_number"] != span["page_number"]
                        ):
                            raise ValueError(
                                "Bronregister wijkt af van de oorspronkelijke context."
                            )
                snapshots.append(dict(case=case, sections=sections, pages=pages))
                auxiliary.extend(
                    auxiliary_fragments(
                        case,
                        pid,
                        context["version_id"],
                        context["source_url"],
                        sections,
                        pages,
                        counter,
                    )
                )
    finally:
        engine.dispose()
    auxiliary.sort(key=lambda c: c["id"])
    root.mkdir(parents=True, exist_ok=True)
    save(root / "protocol.json", read(protocol_path))
    save(root / "auxiliary.json", auxiliary)
    save(root / "source-snapshot.json", snapshots)
    save(
        root / "frozen.json",
        dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            previous_root=str(previous),
            previous_hashes={
                name: digest(previous / name)
                for name in [
                    "frozen.json",
                    "questions.json",
                    "chunks.json",
                    "contexts.json",
                    "results.json",
                    "bge-run.json",
                    "bge-vectors.npy",
                    "bge-queries.npy",
                ]
            },
            hashes={
                name: digest(root / name)
                for name in ["protocol.json", "auxiliary.json", "source-snapshot.json"]
            },
            original_protocol_path=str(protocol_path),
            original_protocol_sha256=digest(protocol_path),
            tokenizer_sha256=counter.sha256,
            previous_manifest_sha256=digest(previous / "frozen.json"),
            paid_calls=0,
        ),
    )
    print(
        json.dumps(
            dict(
                frozen=True,
                questions=len(questions),
                primary=len(primary),
                auxiliary=len(auxiliary),
                roles=dict(Counter(c["role"] for c in auxiliary)),
                paid_calls=0,
            )
        ),
        flush=True,
    )


def load(root):
    manifest = read(root / "frozen.json")
    previous = Path(manifest["previous_root"])
    for name, value in manifest["hashes"].items():
        if digest(root / name) != value:
            raise ValueError("Bevroren zoekcontext gewijzigd.")
    for name, value in manifest["previous_hashes"].items():
        if digest(previous / name) != value:
            raise ValueError("Eerdere proef gewijzigd.")
    if (
        digest(Path(manifest["original_protocol_path"]))
        != manifest["original_protocol_sha256"]
    ):
        raise ValueError("Protocol gewijzigd.")
    questions, primary, contexts = original_inputs(previous)
    report = read(previous / "bge-run.json")
    if (
        digest(previous / "bge-vectors.npy") != report["vectors_sha256"]
        or digest(previous / "bge-queries.npy") != report["queries_sha256"]
    ):
        raise ValueError("BGE-vectoren gewijzigd.")
    vectors = np.load(previous / "bge-vectors.npy", allow_pickle=False)
    queries = np.load(previous / "bge-queries.npy", allow_pickle=False)
    return (
        manifest,
        previous,
        questions,
        primary,
        contexts,
        read(root / "auxiliary.json"),
        queries @ vectors.T,
    )


def rrf(*rankings):
    scores = defaultdict(float)
    for ranking in rankings:
        if len(ranking) != len(set(ranking)):
            raise ValueError("Dubbele zaak in ranglijst.")
        for position, case in enumerate(ranking, 1):
            scores[case] += 1 / (60 + position)
    return sorted(scores, key=lambda case: (-scores[case], case))


def evidence_packets(selected, contexts):
    if len(selected) != len(set(selected)) or len(selected) > 3:
        raise ValueError("Ongeldige zaakselectie.")
    packets = []
    for case in selected:
        context = contexts[case]
        parts = {
            key: context["parts"][key]
            for key in ["commissieoverwegingen", "conclusies"]
        }
        if any(
            s["role"] != "commissie"
            or any(ref.startswith("aux_") for ref in s["source_ids"])
            for spans in parts.values()
            for s in spans
        ):
            raise ValueError("Zoekcontext mag niet als commissie-evidence dienen.")
        packets.append(
            dict(
                case_number=case,
                processing_id=context["processing_id"],
                version_id=context["version_id"],
                source_url=context["source_url"],
                advice_date=context["advice_date"],
                parts=parts,
                semantic_support="not_assessed",
            )
        )
    return packets


def validate_auxiliary(auxiliary, snapshots, counter, contexts):
    lookup = {
        (doc["case"], s["section_key"]): s for doc in snapshots for s in doc["sections"]
    }
    by_section = defaultdict(list)
    for c in auxiliary:
        section = lookup[(c["case"], c["section_key"])]
        a, b = c["start_offset"], c["end_offset"]
        context = contexts[c["case"]]
        if (
            section["category"] not in DISCOVERY_CATEGORIES
            or c["category"] != section["category"]
            or c["role"] != section["role"]
            or c["page"] != section["page_number"]
            or c["processing_id"] != section["processing_id"]
            or c["processing_id"] != context["processing_id"]
            or c["version_id"] != context["version_id"]
            or c["source_url"] != context["source_url"] + f"#page={c['page']}"
            or c["text"] != section["text"][a:b]
            or not 0 <= a < b <= len(section["text"])
            or c["evidence_eligible"] is not False
            or c["citation_source_ids"] != []
            or c["id"]
            != identity(
                "aux", c["processing_id"], c["section_key"], a, b, json_hash(c["text"])
            )
            or c["token_count"] != counter(c["text"])
            or c["token_count"] > 384
        ):
            raise ValueError("Aanvullend fragment heeft ongeldige herkomst.")
        by_section[(c["case"], c["section_key"])].append(c)
    expected = {
        key for key, s in lookup.items() if s["category"] in DISCOVERY_CATEGORIES
    }
    if set(by_section) != expected:
        raise ValueError("Aanvullende sectie ontbreekt.")
    for key, parts in by_section.items():
        parts.sort(key=lambda c: c["start_offset"])
        cursor = 0
        for part in parts:
            if part["start_offset"] != cursor:
                raise ValueError("Zoekcontext mist of verdubbelt tekst.")
            cursor = part["end_offset"]
        if cursor != len(lookup[key]["text"]):
            raise ValueError("Zoekcontext is onvolledig.")


def score(root, tokenizer_path):
    if (root / "results.json").exists():
        raise ValueError("Deze proef is al uitgevoerd; niet overschrijven.")
    manifest, previous, questions, primary, contexts, auxiliary, vector_scores = load(
        root
    )
    counter = LocalTokenCounter(tokenizer_path)
    if counter.sha256 != manifest["tokenizer_sha256"]:
        raise ValueError("Tokenizer gewijzigd.")
    validate_auxiliary(
        auxiliary, read(root / "source-snapshot.json"), counter, contexts
    )
    index_primary = BM25(primary)
    expanded = sorted(primary + auxiliary, key=lambda c: c["id"])
    index_expanded = BM25(expanded)
    baseline = read(previous / "results.json")
    baseline_map = {q["id"]: q for q in baseline["results"]}
    results = []
    packets = []
    for i, q in enumerate(questions):
        start = time.perf_counter()
        p_scores = index_primary.scores(q["question"])
        primary_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        a_scores = index_expanded.scores(q["question"])
        expanded_ms = (time.perf_counter() - start) * 1000
        p_rank = case_ranking(primary, p_scores, True)
        a_rank = case_ranking(expanded, a_scores, True)
        bge = case_ranking(primary, vector_scores[i])
        start = time.perf_counter()
        expanded_hybrid = rrf(a_rank, bge)
        fusion_ms = (time.perf_counter() - start) * 1000
        original = baseline_map[q["id"]]["methods"]
        if (
            rrf(p_rank, bge) != original["hybrid_bge"]["all_cases"]
            or p_rank != original["text_shared"]["all_cases"]
        ):
            raise ValueError(
                "Oorspronkelijke zoekrangschikking niet exact gereproduceerd."
            )
        rankings = {
            "text_primary": p_rank,
            "hybrid_primary": rrf(p_rank, bge),
            "text_expanded": a_rank,
            "hybrid_expanded": expanded_hybrid,
            "text_existing": original["text_existing"]["all_cases"],
        }
        methods = {}
        for name, ranking in rankings.items():
            selected = ranking[:3]
            result = metrics(selected, None, contexts, q["gold"])
            evidence = evidence_packets(selected, contexts)
            methods[name] = dict(
                cases=selected,
                all_cases=ranking,
                metrics=result,
                evidence_source_ids=sorted(
                    {
                        ref
                        for p in evidence
                        for spans in p["parts"].values()
                        for s in spans
                        for ref in s["source_ids"]
                    }
                ),
            )
            packets.append(dict(question_id=q["id"], method=name, evidence=evidence))
        # Explain actual discovery provenance, not a favorable expected-case feature.
        discoveries = []
        for case in expanded_hybrid[:3]:
            choices = [j for j, c in enumerate(expanded) if c["case"] == case]
            best = min(choices, key=lambda j: (-float(a_scores[j]), expanded[j]["id"]))
            c = expanded[best]
            discoveries.append(
                dict(
                    case=case,
                    fragment_id=c["id"],
                    role=c["role"],
                    page=c["page"],
                    section=c["section"],
                    is_auxiliary=c["id"].startswith("aux_"),
                    score=float(a_scores[best]),
                )
            )
        results.append(
            dict(
                id=q["id"],
                question=q["question"],
                boundary=not bool(q["gold"]),
                methods=methods,
                expanded_hybrid_lexical_matches=discoveries,
                primary_bm25_ms=primary_ms,
                expanded_bm25_ms=expanded_ms,
                fusion_ms=fusion_ms,
            )
        )
    summary = {}
    for name in results[0]["methods"]:
        values = [q["methods"][name]["metrics"] for q in results if not q["boundary"]]
        summary[name] = {
            key: sum(v[key] for v in values) / len(values)
            for key in [
                "case_recall",
                "mrr",
                "full_anchor_recall",
                "reasoning_characters",
                "conclusion_characters",
            ]
        }
        summary[name] |= dict(
            questions=len(values),
            anchor_hits=sum(sum(v["full_anchor_hits"]) for v in values),
            anchor_total=sum(len(v["full_anchor_hits"]) for v in values),
            complete=sum(v["complete"] for v in values),
        )
    regressions = []
    improvements = []
    for q in results:
        if q["boundary"]:
            continue
        before = q["methods"]["hybrid_primary"]["metrics"]["full_anchor_recall"]
        after = q["methods"]["hybrid_expanded"]["metrics"]["full_anchor_recall"]
        if after < before:
            regressions.append(q["id"])
        if after > before:
            improvements.append(q["id"])
    report = dict(
        protocol_sha256=digest(root / "protocol.json"),
        frozen_sha256=digest(root / "frozen.json"),
        previous_frozen_sha256=manifest["previous_manifest_sha256"],
        summary=summary,
        results=results,
        regressions_vs_primary_hybrid=regressions,
        improvements_vs_primary_hybrid=improvements,
        meets_prespecified_development_gate="V4" in improvements and not regressions,
        primary_fragments=len(primary),
        auxiliary_fragments=len(auxiliary),
        auxiliary_roles=dict(Counter(c["role"] for c in auxiliary)),
        source_scope="facts_and_party_positions_for_discovery_only",
        answer_quality="not_measured",
        paid_calls=0,
        score_is_answerability=False,
        original_rankings_reproduced=True,
        active_api_changed=False,
    )
    save(root / "evidence-packets.json", packets)
    save(root / "results.json", report)
    print(
        json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["freeze", "score"])
    parser.add_argument(
        "--root", type=Path, default=Path("local-results/discovery-context-v1")
    )
    parser.add_argument(
        "--previous", type=Path, default=Path("local-results/model-comparison-v1")
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("experiments/discovery-context-protocol.json"),
    )
    parser.add_argument(
        "--tokenizer", type=Path, default=Path("data/e5/model/tokenizer.json")
    )
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze(args.root, args.previous, args.protocol, args.tokenizer)
    else:
        score(args.root, args.tokenizer)
