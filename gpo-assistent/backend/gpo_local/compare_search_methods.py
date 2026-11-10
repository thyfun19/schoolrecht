"""Offline retrieval comparison, before changing answer logic. No query rewriting."""

import collections
import json
import math
from pathlib import Path

from .common import read, norm, digest
from .retrieval import Retrieval, words, ordered


def compare(output=Path("local-results/question-focus")):
    engine = Retrieval(Path("data/e5"), Path("data/gpo.sqlite3"))
    study = Path("../../onderzoek_gpo/vectoronderzoek")
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
    rows = []
    for q in questions:
        methods = {}
        for method in ["text", "text_e5_chunks", "vector"]:
            chunks = engine.legacy_chunks if method == "text" else engine.chunks
            index = engine.legacy_index if method == "text" else engine.text_index
            scores = (
                engine.vectors @ engine.embed_query(q["question"])
                if method == "vector"
                else index.scores(q["question"])
            )
            ranking = []
            best = {}
            for i in ordered(scores):
                if method != "vector" and scores[i] <= 0:
                    continue
                case = chunks[i]["case"]
                if case not in best:
                    ranking.append(case)
                    best[case] = i
            response = engine.search(q["question"], method)
            full = [
                s
                for c in response["results"]
                for s in c["parts"]["commissieoverwegingen"]
            ]
            hits = [
                any(
                    s["case"] == g["case"]
                    and s["page"] == g["page"]
                    and g["anchor"].lower() in norm(s["text"]).lower()
                    for s in full
                )
                for g in q["gold"]
            ]
            item = dict(
                top3=ranking[:3],
                all_cases=ranking,
                anchor_recall=sum(hits) / len(hits) if hits else None,
            )
            if q["id"] == "N3":
                target = "109206"  # Evaluation/inspection only, never a ranking input.
                i = best[target]
                chunk = chunks[i]
                item.update(
                    reference_case_rank=ranking.index(target) + 1,
                    best_reference_chunk=dict(
                        page=chunk["page"],
                        role=chunk["role"],
                        section=chunk["section"],
                        text=chunk["text"],
                        score=float(scores[i]),
                    ),
                    leading_matches=[
                        dict(
                            case=case,
                            page=chunks[best[case]]["page"],
                            role=chunks[best[case]]["role"],
                            score=float(scores[best[case]]),
                        )
                        for case in ranking[:3]
                    ],
                )
                if method != "vector":
                    counts = index.docs[i]
                    length = index.lengths[i]
                    n = len(chunks)
                    item["reference_chunk_term_contributions"] = {
                        w: math.log(1 + (n - index.df[w] + 0.5) / (index.df[w] + 0.5))
                        * counts[w]
                        * 2.5
                        / (counts[w] + 1.5 * (0.25 + 0.75 * length / index.average))
                        for w in words(q["question"])
                        if counts[w]
                    }
                    item["query_terms"] = words(q["question"])
                    for match in item["leading_matches"]:
                        position = best[match["case"]]
                        counts = index.docs[position]
                        length = index.lengths[position]
                        match["term_contributions"] = {
                            w: math.log(
                                1 + (n - index.df[w] + 0.5) / (index.df[w] + 0.5)
                            )
                            * counts[w]
                            * 2.5
                            / (counts[w] + 1.5 * (0.25 + 0.75 * length / index.average))
                            for w in words(q["question"])
                            if counts[w]
                        }
            methods[method] = item
        rows.append(dict(id=q["id"], question=q["question"], methods=methods))
    summary = {
        m: sum(
            r["methods"][m]["anchor_recall"]
            for r in rows
            if r["methods"][m]["anchor_recall"] is not None
        )
        / 8
        for m in ["text", "text_e5_chunks", "vector"]
    }
    result = dict(
        default_method="text",
        queries_unchanged=True,
        ranking_unchanged=True,
        paid_calls=0,
        summary=summary,
        results=rows,
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "search-comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            dict(summary=summary, N3=next(r for r in rows if r["id"] == "N3")),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    compare()
