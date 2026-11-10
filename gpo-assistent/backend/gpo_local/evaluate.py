"""Compare actual HTTP responses to frozen study artifacts; no expected-case boosts."""

import argparse
import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from .api import create_app
from .common import read, norm, digest


def evaluate(study, output):
    frozen = read(study / "bevroren.json")["hashes"]
    assert digest(study / "nieuwe_vragen.json") == frozen["nieuwe_vragen.json"]
    assert (
        digest(study.parent / "besliscontrole/referentie.json")
        == frozen["oude_referentie"]
    )
    reference = read(study / "resultaten.json")
    baseline = {
        q["id"]: q for q in read(study / "bestaande_textcontrole.json")["results"]
    }
    vectors = np.load(study / "query_vectors.npy")
    questions = (
        read(study.parent / "besliscontrole/referentie.json")["questions"]
        + read(study / "nieuwe_vragen.json")["questions"]
    )
    app = create_app()
    results, packets = [], []
    with TestClient(app) as client:
        batch_vectors = app.state.engine.embed_queries(
            [q["question"] for q in questions]
        )
        original_batch_vectors_match = np.allclose(batch_vectors, vectors, atol=1e-6)
        batch = client.post(
            "/search/batch",
            json={
                "questions": [
                    dict(question=q["question"], method="vector", limit=3)
                    for q in questions
                ]
            },
        )
        assert batch.status_code == 200, batch.text
        batch = batch.json()
        for index, (q, expected) in enumerate(zip(questions, reference["results"])):
            assert q["question"] == expected["question"]
            actual_vector = app.state.engine.embed_query(q["question"])
            entry = dict(
                id=q["id"],
                methods={},
                single_vs_batch_query_cosine=float(actual_vector @ vectors[index]),
            )
            assert np.array_equal(actual_vector, batch_vectors[index])
            for method, study_method in [
                ("vector", "vector"),
                ("text", "legacy_sqlite"),
                ("text_e5_chunks", "text"),
            ]:
                response = client.post(
                    "/search", json=dict(question=q["question"], method=method, limit=3)
                )
                assert response.status_code == 200, response.text
                response = response.json()
                if method == "vector":
                    assert response == batch[index], (
                        "Batch en losse vraag moeten exact gelijk zijn."
                    )
                cases = [c["case_number"] for c in response["results"]]
                wanted = (
                    baseline[q["id"]]["cases"]
                    if method == "text"
                    else expected["methods"][study_method]["cases"]
                )
                if method != "vector":
                    assert cases == wanted, (q["id"], method, cases, wanted)
                reasoning = [
                    s
                    for c in response["results"]
                    for s in c["parts"]["commissieoverwegingen"]
                ]
                if method == "text_e5_chunks":
                    wanted_sections = expected["methods"][study_method][
                        "full_reasoning"
                    ]
                    assert {s["section_id"]: s["text"] for s in reasoning} == {
                        s["section_id"]: s["text"] for s in wanted_sections
                    }
                hits = [
                    any(
                        s["case"] == g["case"]
                        and s["page"] == g["page"]
                        and g["anchor"].lower() in norm(s["text"]).lower()
                        for s in reasoning
                    )
                    for g in q["gold"]
                ]
                entry["methods"][method] = dict(
                    cases=cases,
                    matches_study=cases == wanted,
                    anchor_recall=sum(hits) / len(hits) if hits else None,
                    complete=all(hits) if hits else None,
                )
            prepared = client.post(
                "/explanation/prepare",
                json={"question": q["question"], "method": "vector"},
            ).json()
            assert prepared["answer_generated"] is False and prepared["paid_calls"] == 0
            if not q["gold"]:
                assert (
                    prepared["status"] == "needs_additional_sources"
                    and prepared["request"] is None
                )
            else:
                assert prepared["status"] == "awaiting_new_cost_approval"
                assert all(s["role"] == "commissie" for s in prepared["sources"])
            packets.append(dict(id=q["id"], question=q["question"], prepared=prepared))
            entry["explanation"] = dict(
                status=prepared["status"],
                answer_quality="not_measured",
                max_cost_usd=prepared["estimated_max_cost_usd"],
            )
            results.append(entry)
    summary = {}
    for method in ["text", "text_e5_chunks", "vector"]:
        positive = [
            r["methods"][method]
            for r in results
            if r["methods"][method]["anchor_recall"] is not None
        ]
        summary[method] = dict(
            anchor_recall=sum(v["anchor_recall"] for v in positive) / len(positive),
            complete=sum(v["complete"] for v in positive),
            n=len(positive),
        )
    output.mkdir(parents=True, exist_ok=True)
    report = dict(
        search_quality=summary,
        single_and_batch_responses_identical=True,
        original_batch_embeddings_match=bool(original_batch_vectors_match),
        single_query_ranking_matches=sum(
            r["methods"]["vector"]["matches_study"] for r in results
        ),
        model_batch_sensitive_but_backend_isolates_queries=True,
        explanation_quality="not_measured_no_generated_answers",
        preparation_boundary_checks="2/2",
        paid_calls=0,
        estimated_batch_max_usd=sum(
            p["prepared"]["estimated_max_cost_usd"] for p in packets
        ),
        results=results,
    )
    (output / "backend-evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "explanation-packets.json").write_text(
        json.dumps(packets, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("local-results/offline-v2"))
    args = parser.parse_args()
    evaluate(args.study.resolve(), args.output.resolve())
