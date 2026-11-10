"""Offline embedding model comparison utilities across GPO decision corpora."""

import argparse
from collections import defaultdict
import ctypes
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np

from gpo_local.common import digest, norm, read
from gpo_local.retrieval import BM25, Retrieval, ordered
from .chunking import read_preparation
from .database import create_engine
from .ingestion import json_hash
from .model_assets import SPECS, sha256
from . import models as m
import sqlalchemy as sa


def save(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def freeze(root, protocol_path, study):
    root.mkdir(parents=True, exist_ok=True)
    if (root / "frozen.json").exists():
        raise ValueError("Proef bestaat al; niet overschrijven.")
    protocol = read(protocol_path)
    old = study.parent / "besliscontrole/referentie.json"
    new = study / "nieuwe_vragen.json"
    prior = read(study / "bevroren.json")["hashes"]
    if (
        digest(old) != prior["oude_referentie"]
        or digest(new) != prior["nieuwe_vragen.json"]
    ):
        raise ValueError("Vragen/bewijsankers gewijzigd.")
    questions = read(old)["questions"] + read(new)["questions"]
    engine = create_engine()
    contexts = {}
    chunks = []
    try:
        with engine.connect() as con:
            ids = list(con.scalars(sa.select(m.chunk_runs.c.chunk_run_id)))
        for run_id in sorted(ids):
            run = read_preparation(engine, run_id)
            case = run["case_number"]
            if case in contexts:
                raise ValueError(
                    "Meerdere fragmentversies per zaak: kies expliciet vóór de proef."
                )
            parts = {p["category"]: p["spans"] for p in run["contexts"]}
            contexts[case] = dict(
                case_number=case,
                processing_id=run["processing_id"],
                version_id=run["version_id"],
                advice_date=str(run["advice_date"]),
                source_url=run["source_url"],
                parts=parts,
            )
            for parent in run["contexts"]:
                for c in parent["chunks"]:
                    chunks.append(
                        dict(
                            id=c["chunk_id"],
                            context_id=c["context_id"],
                            case=case,
                            page=c["span"]["page_number"],
                            section=c["span"]["heading"],
                            role=c["span"]["role"],
                            text=c["text"],
                            span=c["span"],
                            processing_id=run["processing_id"],
                            version_id=run["version_id"],
                        )
                    )
    finally:
        engine.dispose()
    chunks.sort(key=lambda c: c["id"])
    if len(contexts) != 10 or len(chunks) != 111:
        raise ValueError("Deze proef vereist de vastgelegde tien zaken/111 fragmenten.")
    for q in questions:
        for g in q["gold"]:
            if not any(
                s["page_number"] == g["page"]
                and g["anchor"].lower() in norm(s["text"]).lower()
                for s in contexts[g["case"]]["parts"]["commissieoverwegingen"]
            ):
                raise ValueError("Bewijsanker ontbreekt in broncontext.")
    for name, value in [
        ("questions.json", questions),
        ("chunks.json", chunks),
        ("contexts.json", contexts),
        ("protocol.json", protocol),
    ]:
        save(root / name, value)
    save(
        root / "frozen.json",
        dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            hashes={
                name: digest(root / name)
                for name in [
                    "questions.json",
                    "chunks.json",
                    "contexts.json",
                    "protocol.json",
                ]
            },
            original_hashes={
                str(old): digest(old),
                str(new): digest(new),
                str(protocol_path): digest(protocol_path),
            },
            model_specs=SPECS,
            paid_calls=0,
        ),
    )
    print(
        json.dumps(
            dict(
                questions=len(questions),
                chunks=len(chunks),
                cases=len(contexts),
                frozen=True,
            )
        ),
        flush=True,
    )


def frozen(root):
    manifest = read(root / "frozen.json")
    for name, expected in manifest["hashes"].items():
        if digest(root / name) != expected:
            raise ValueError("Bevroren proefbestand gewijzigd: " + name)
    for name, expected in manifest["original_hashes"].items():
        if digest(Path(name)) != expected:
            raise ValueError("Originele referentie gewijzigd.")
    if json_hash(manifest["model_specs"]) != json_hash(SPECS):
        raise ValueError("Modelspecificatie gewijzigd na bevriezen.")
    return (
        read(root / "questions.json"),
        read(root / "chunks.json"),
        read(root / "contexts.json"),
    )


class Encoder:
    def __init__(self, name, path):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.name = name
        self.spec = SPECS[name]
        self.path = path
        for remote, (size, expected) in self.spec["files"].items():
            p = path / Path(remote).name
            if p.stat().st_size != size or sha256(p) != expected:
                raise ValueError("Modelbestand wijkt af.")
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        opts.enable_cpu_mem_arena = False
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        opts.add_session_config_entry("session.disable_prepacking", "1")
        start = time.perf_counter()
        self.session = ort.InferenceSession(
            str(path / "model.onnx"),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.load_seconds = time.perf_counter() - start
        outputs = self.session.get_outputs()
        self.output = next(
            (o.name for o in outputs if o.name == "last_hidden_state"), outputs[0].name
        )
        self.description = dict(
            inputs=[
                dict(name=i.name, type=i.type, shape=i.shape)
                for i in self.session.get_inputs()
            ],
            outputs=[dict(name=i.name, type=i.type, shape=i.shape) for i in outputs],
        )

    def encode(self, texts, kind, batch_size=1):
        prefix = self.spec["query_prefix" if kind == "query" else "passage_prefix"]
        tokenizer = self.tokenizer
        tokenizer.no_padding()
        tokenizer.no_truncation()
        lengths = [len(tokenizer.encode(prefix + t).ids) for t in texts]
        if max(lengths, default=0) > 512:
            raise ValueError("Modelinvoer langer dan 512 tokens; stop zonder afkappen.")
        pad = tokenizer.token_to_id("<pad>")
        if pad is None:
            raise ValueError("Geen expliciete paddingtoken gevonden.")
        tokenizer.enable_padding(direction="right", pad_id=pad, pad_token="<pad>")
        vectors = []
        for start in range(0, len(texts), batch_size):
            if kind == "passage" and len(texts) > 20 and start % 20 == 0:
                print(
                    self.name + ": passages " + str(start) + "/" + str(len(texts)),
                    flush=True,
                )
            batch = tokenizer.encode_batch(
                [prefix + t for t in texts[start : start + batch_size]]
            )
            available = {
                key: np.asarray(value, dtype=np.int64)
                for key, value in [
                    ("input_ids", [e.ids for e in batch]),
                    ("attention_mask", [e.attention_mask for e in batch]),
                    ("token_type_ids", [e.type_ids for e in batch]),
                ]
            }
            hidden = self.session.run(
                [self.output],
                {i.name: available[i.name] for i in self.session.get_inputs()},
            )[0]
            if hidden.ndim != 3 or hidden.dtype != np.float32:
                raise ValueError("Geen float32 tokenrepresentaties ontvangen.")
            vectors.extend(
                pool(hidden, available["attention_mask"], self.spec["pooling"])
            )
        result = np.asarray(vectors, dtype=np.float32)
        if (
            result.shape != (len(texts), self.spec["dimensions"])
            or not np.isfinite(result).all()
        ):
            raise ValueError("Ongeldige embeddings.")
        return result, lengths


def pool(hidden, mask, mode):
    if mode not in ("cls", "masked_mean"):
        raise ValueError("Onbekende pooling.")
    weights = mask.astype(np.float32)
    if (weights.sum(axis=1) <= 0).any():
        raise ValueError("Leeg attention mask.")
    value = (
        hidden[:, 0, :]
        if mode == "cls"
        else (hidden * weights[:, :, None]).sum(axis=1) / weights.sum(axis=1)[:, None]
    )
    length = np.linalg.norm(value, axis=1, keepdims=True)
    if (length <= 0).any():
        raise ValueError("Nulvector.")
    return (value / length).astype(np.float32)


def case_ranking(chunks, scores, positive=False):
    maxima = {}
    for c, score in zip(chunks, scores):
        if positive and score <= 0:
            continue
        maxima[c["case"]] = max(maxima.get(c["case"], float("-inf")), float(score))
    return sorted(maxima, key=lambda case: (-maxima[case], case))


def peak_memory():
    if os.name != "nt":
        return None

    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
            (n, ctypes.c_size_t)
            for n in [
                "PeakWorkingSetSize",
                "WorkingSetSize",
                "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage",
                "PagefileUsage",
                "PeakPagefileUsage",
            ]
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    # Real process pseudo-handle fits pointer width on Windows x64.
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(
        ctypes.c_void_p(-1), ctypes.byref(counters), counters.cb
    )
    return counters.PeakWorkingSetSize if ok else None


def embed(root, name, model_root):
    questions, chunks, _ = frozen(root)
    if (root / (name + "-run.json")).exists():
        raise ValueError("Modelproef al afgerond; niet overschrijven.")
    preflight(
        root, model_root
    )  # Both models must accept the shared texts before any score.
    encoder = Encoder(name, model_root / name)
    start = time.perf_counter()
    vectors, lengths = encoder.encode([c["text"] for c in chunks], "passage")
    corpus_seconds = time.perf_counter() - start
    query_vectors = []
    query_seconds = []
    for q in questions:
        start = time.perf_counter()
        v, _ = encoder.encode([q["question"]], "query")
        query_seconds.append(time.perf_counter() - start)
        query_vectors.append(v[0])
    query_vectors = np.asarray(query_vectors)
    stability = []
    for size in [2, 3, len(questions)]:
        order = list(reversed(range(len(questions))))
        batched, _ = encoder.encode(
            [questions[i]["question"] for i in order], "query", size
        )
        actual = np.empty_like(query_vectors)
        for j, index in enumerate(order):
            actual[index] = batched[j]
        changed = []
        for i, q in enumerate(questions):
            if case_ranking(chunks, vectors @ actual[i]) != case_ranking(
                chunks, vectors @ query_vectors[i]
            ):
                changed.append(q["id"])
        stability.append(
            dict(
                batch_size=size,
                max_abs_difference=float(np.max(np.abs(actual - query_vectors))),
                max_score_difference=float(
                    np.max(np.abs(actual @ vectors.T - query_vectors @ vectors.T))
                ),
                min_cosine=float(np.min((actual * query_vectors).sum(axis=1))),
                ranking_changed_questions=changed,
            )
        )
    chosen = [0, len(chunks) // 2, len(chunks) - 1]
    document_batch, _ = encoder.encode(
        [chunks[i]["text"] for i in chosen], "passage", 3
    )
    corpus_batch_difference = float(np.max(np.abs(document_batch - vectors[chosen])))
    np.save(root / (name + "-vectors.npy"), vectors, allow_pickle=False)
    np.save(root / (name + "-queries.npy"), query_vectors, allow_pickle=False)
    report = dict(
        model=name,
        manifest=read(model_root / name / "manifest.json"),
        session=encoder.description,
        runtime={
            p: importlib.metadata.version(p)
            for p in ["numpy", "onnxruntime", "tokenizers"]
        },
        corpus_seconds=corpus_seconds,
        model_load_seconds=encoder.load_seconds,
        query_seconds=query_seconds,
        max_passage_tokens=max(lengths),
        peak_working_set_bytes=peak_memory(),
        stability=stability,
        corpus_batch_check=dict(
            fragment_indices=chosen, max_abs_difference=corpus_batch_difference
        ),
        vectors_sha256=digest(root / (name + "-vectors.npy")),
        queries_sha256=digest(root / (name + "-queries.npy")),
        frozen_sha256=digest(root / "frozen.json"),
        paid_calls=0,
    )
    save(root / (name + "-run.json"), report)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("session", "manifest")}
        ),
        flush=True,
    )


def preflight(root, model_root):
    from tokenizers import Tokenizer

    questions, chunks, _ = frozen(root)
    result = {}
    for name, spec in SPECS.items():
        tokenizer = Tokenizer.from_file(str(model_root / name / "tokenizer.json"))
        tokenizer.no_truncation()
        tokenizer.no_padding()
        lengths = [
            len(tokenizer.encode(spec["passage_prefix"] + c["text"]).ids)
            for c in chunks
        ]
        qlengths = [
            len(tokenizer.encode(spec["query_prefix"] + q["question"]).ids)
            for q in questions
        ]
        result[name] = dict(
            max_passage_tokens=max(lengths),
            max_query_tokens=max(qlengths),
            rejected_chunks=[chunks[i]["id"] for i, n in enumerate(lengths) if n > 512],
        )
    save(root / "token-preflight.json", result)
    if any(
        v["rejected_chunks"] or v["max_query_tokens"] > 512 for v in result.values()
    ):
        raise ValueError(
            "Gezamenlijke fragmentversie past niet; stop vóór embedding/ranking."
        )
    print(json.dumps(result), flush=True)


def metrics(cases, fragments, contexts, gold):
    if not gold:
        return None
    evidence = [
        dict(case=case, page=s["page_number"], text=s["text"])
        for case in cases
        for s in contexts[case]["parts"]["commissieoverwegingen"]
    ]

    def hits(source):
        return [
            any(
                s["case"] == g["case"]
                and s["page"] == g["page"]
                and g["anchor"].lower() in norm(s["text"]).lower()
                for s in source
            )
            for g in gold
        ]

    full = hits(evidence)
    expected = {g["case"] for g in gold}
    return dict(
        case_recall=len(set(cases) & expected) / len(expected),
        mrr=next((1 / (i + 1) for i, c in enumerate(cases) if c in expected), 0),
        full_anchor_recall=sum(full) / len(full),
        complete=all(full),
        full_anchor_hits=full,
        passage_anchor_recall=sum(hits(fragments)) / len(gold)
        if fragments is not None
        else None,
        missing_cases=sorted(expected - set(cases)),
        reasoning_characters=sum(len(s["text"]) for s in evidence),
        conclusion_characters=sum(
            len(s["text"]) for c in cases for s in contexts[c]["parts"]["conclusies"]
        ),
    )


def score(root):
    questions, chunks, contexts = frozen(root)
    index = BM25(chunks)
    text = [index.scores(q["question"]) for q in questions]
    vectors = {}
    reports = {}
    for name in SPECS:
        report = read(root / (name + "-run.json"))
        if (
            report["frozen_sha256"] != digest(root / "frozen.json")
            or report["vectors_sha256"] != digest(root / (name + "-vectors.npy"))
            or report["queries_sha256"] != digest(root / (name + "-queries.npy"))
        ):
            raise ValueError("Modeluitvoer gewijzigd.")
        reports[name] = report
        vectors[name] = (
            np.load(root / (name + "-queries.npy"), allow_pickle=False)
            @ np.load(root / (name + "-vectors.npy"), allow_pickle=False).T
        )
    # Rerun actual earlier backend, preserving its single-query dynamic-quantization workaround.
    legacy = Retrieval(Path("data/e5"), Path("data/gpo.sqlite3"))
    results = []
    for i, q in enumerate(questions):
        text_order = case_ranking(chunks, text[i], True)
        methods = {}
        scores = {"text_shared": text[i]} | {name: vectors[name][i] for name in SPECS}
        rankings = {
            name: case_ranking(chunks, value, name == "text_shared")
            for name, value in scores.items()
        }
        for name, ranking in rankings.items():
            selected = ranking[:3]
            fragments = [
                chunks[j]
                for j in ordered(scores[name])[:10]
                if name != "text_shared" or scores[name][j] > 0
            ]
            methods[name] = dict(
                cases=selected,
                all_cases=ranking,
                metrics=metrics(selected, fragments, contexts, q["gold"]),
            )
        for name in SPECS:
            rrf = defaultdict(float)
            for ranking in [text_order, rankings[name]]:
                for rank, case in enumerate(ranking, 1):
                    rrf[case] += 1 / (60 + rank)
            ranking = sorted(rrf, key=lambda c: (-rrf[c], c))
            selected = ranking[:3]
            methods["hybrid_" + name] = dict(
                cases=selected,
                all_cases=ranking,
                metrics=metrics(selected, None, contexts, q["gold"]),
            )
        for name, method in [
            ("text_existing", "text"),
            ("e5_quantized_existing", "vector"),
        ]:
            response = legacy.search(q["question"], method, 10)
            ranking = [c["case_number"] for c in response["results"]]
            selected = ranking[:3]
            methods[name] = dict(
                cases=selected,
                all_cases=ranking,
                metrics=metrics(selected, None, contexts, q["gold"]),
            )
        results.append(
            dict(
                id=q["id"],
                question=q["question"],
                boundary=not bool(q["gold"]),
                methods=methods,
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
            n=len(values), complete=sum(v["complete"] for v in values)
        )
    report = dict(
        protocol_sha256=digest(root / "protocol.json"),
        frozen_sha256=digest(root / "frozen.json"),
        summary=summary,
        results=results,
        answer_quality="not_measured",
        paid_calls=0,
        score_is_answerability=False,
    )
    save(root / "results.json", report)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["freeze", "preflight", "embed", "score"])
    parser.add_argument(
        "--root", type=Path, default=Path("local-results/model-comparison-v1")
    )
    parser.add_argument(
        "--model-root", type=Path, default=Path("data/model-comparison/models")
    )
    parser.add_argument("--model", choices=list(SPECS))
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("experiments/model-comparison-protocol.json"),
    )
    parser.add_argument(
        "--study", type=Path, default=Path("../../onderzoek_gpo/vectoronderzoek")
    )
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze(args.root, args.protocol, args.study)
    elif args.phase == "preflight":
        preflight(args.root, args.model_root)
    elif args.phase == "embed":
        if args.model is None:
            parser.error("--model is vereist")
        embed(args.root, args.model, args.model_root)
    else:
        score(args.root)
