"""Local cross-encoder reranking utilities for hybrid passage retrieval."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np

from gpo_local.common import read, digest
from gpo_local.retrieval import BM25
from .discovery_context import load as load_discovery, evidence_packets
from .ingestion import identity, json_hash
from .model_assets import acquire, sha256
from .model_comparison import save, peak_memory, metrics

RERANKER = dict(
    repository="onnx-community/bge-reranker-v2-m3-ONNX",
    revision="6f5ff65298512715a1e669753bc754d2bc8f367b",
    base_model="BAAI/bge-reranker-v2-m3",
    output="classification_logits",
    files={
        "onnx/model.onnx": (
            656891,
            "faae32b124a9d54afb7e89b5e9896e03c18a9552d56d1d6b273a709a83012486",
        ),
        "onnx/model.onnx_data": (
            2271088656,
            "f009aa6c6cf21986fd7e0021fa66b20ccce27abc6900a57c7109c8496811bcbe",
        ),
        "tokenizer.json": (
            17082900,
            "8bf8afbfd11306bd872018c53bfdf2e160a56f8edbcf49933324404791c148d3",
        ),
    },
)


def choose_pairs(
    question,
    candidates,
    primary,
    expanded,
    primary_lexical,
    expanded_lexical,
    vector_scores,
):
    if len(candidates) != len(set(candidates)) or len(candidates) > 5:
        raise ValueError("Ongeldig kandidaatbudget.")
    output = []
    representations = {key: {} for key in ["rerank_primary", "rerank_with_auxiliary"]}

    def best(chunks, scores, case):
        choices = [i for i, c in enumerate(chunks) if c["case"] == case]
        if not choices:
            raise ValueError("Kandidaat heeft geen zoekfragment.")
        return chunks[min(choices, key=lambda i: (-float(scores[i]), chunks[i]["id"]))]

    for case in candidates:
        dense = best(primary, vector_scores, case)
        variants = {
            "rerank_primary": [dense, best(primary, primary_lexical, case)],
            "rerank_with_auxiliary": [dense, best(expanded, expanded_lexical, case)],
        }
        for name, chunks in variants.items():
            refs = []
            for c in chunks:
                pair_id = identity("pair", question, c["id"])
                refs.append(pair_id)
                output.append(
                    dict(
                        pair_id=pair_id,
                        question=question,
                        passage=c["text"],
                        case=case,
                        fragment_id=c["id"],
                        page=c["page"],
                        role=c["role"],
                        section=c["section"],
                        is_auxiliary=c["id"].startswith("aux_"),
                    )
                )
            representations[name][case] = sorted(set(refs))
    pairs = {p["pair_id"]: p for p in output}
    return list(pairs.values()), representations


def freeze(root, discovery, protocol, model_root):
    from tokenizers import Tokenizer

    if (root / "frozen.json").exists():
        raise ValueError("Rerankerproef bestaat al; niet overschrijven.")
    _, previous, questions, primary, contexts, auxiliary, vectors = load_discovery(
        discovery
    )
    reference = read(discovery / "results.json")
    refmap = {q["id"]: q for q in reference["results"]}
    tokenizer = Tokenizer.from_file(str(model_root / "reranker/tokenizer.json"))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    if (
        sha256(model_root / "reranker/tokenizer.json")
        != RERANKER["files"]["tokenizer.json"][1]
    ):
        raise ValueError("Rerankertokenizer gewijzigd.")
    expanded = sorted(primary + auxiliary, key=lambda c: c["id"])
    p_index = BM25(primary)
    a_index = BM25(expanded)
    packets = []
    all_pairs = {}
    for i, q in enumerate(questions):
        candidates = refmap[q["id"]]["methods"]["hybrid_primary"]["all_cases"][:5]
        pairs, reps = choose_pairs(
            q["question"],
            candidates,
            primary,
            expanded,
            p_index.scores(q["question"]),
            a_index.scores(q["question"]),
            vectors[i],
        )
        for p in pairs:
            p["token_count"] = len(tokenizer.encode(p["question"], p["passage"]).ids)
            if p["token_count"] > 512:
                raise ValueError(
                    "Rerankerpaar langer dan 512 tokens; stop zonder afkappen."
                )
            all_pairs[p["pair_id"]] = p
        packets.append(
            dict(question_id=q["id"], candidates=candidates, representations=reps)
        )
    root.mkdir(parents=True, exist_ok=True)
    save(root / "protocol.json", read(protocol))
    save(root / "packets.json", packets)
    save(root / "pairs.json", [all_pairs[k] for k in sorted(all_pairs)])
    save(
        root / "frozen.json",
        dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            discovery_root=str(discovery),
            discovery_hashes={
                n: digest(discovery / n) for n in ["frozen.json", "results.json"]
            },
            hashes={
                n: digest(root / n)
                for n in ["protocol.json", "packets.json", "pairs.json"]
            },
            model_spec=RERANKER,
            original_protocol=str(protocol),
            original_protocol_sha256=digest(protocol),
            paid_calls=0,
        ),
    )
    print(
        json.dumps(
            dict(
                frozen=True,
                queries=len(packets),
                unique_pairs=len(all_pairs),
                max_pair_tokens=max(p["token_count"] for p in all_pairs.values()),
                paid_calls=0,
            )
        ),
        flush=True,
    )


def frozen(root):
    manifest = read(root / "frozen.json")
    discovery = Path(manifest["discovery_root"])
    for name, value in manifest["hashes"].items():
        if digest(root / name) != value:
            raise ValueError("Bevroren rerankerinvoer gewijzigd.")
    for name, value in manifest["discovery_hashes"].items():
        if digest(discovery / name) != value:
            raise ValueError("Eerdere selectieproef gewijzigd.")
    if digest(Path(manifest["original_protocol"])) != manifest[
        "original_protocol_sha256"
    ] or json_hash(manifest["model_spec"]) != json_hash(RERANKER):
        raise ValueError("Rerankerprotocol/modelspecificatie gewijzigd.")
    return discovery, read(root / "pairs.json"), read(root / "packets.json")


class LocalReranker:
    def __init__(self, path):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        for remote, (size, expected) in RERANKER["files"].items():
            p = path / Path(remote).name
            if p.stat().st_size != size or sha256(p) != expected:
                raise ValueError("Rerankermodel gewijzigd.")
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = False
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        options.add_session_config_entry("session.disable_prepacking", "1")
        start = time.perf_counter()
        self.session = ort.InferenceSession(
            str(path / "model.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.load_seconds = time.perf_counter() - start
        outputs = self.session.get_outputs()
        self.output = next(
            (o.name for o in outputs if o.name == "logits"), outputs[0].name
        )
        self.description = dict(
            inputs=[
                dict(name=i.name, type=i.type, shape=i.shape)
                for i in self.session.get_inputs()
            ],
            outputs=[dict(name=i.name, type=i.type, shape=i.shape) for i in outputs],
        )

    def score(self, pairs):
        tokenizer = self.tokenizer
        tokenizer.no_truncation()
        tokenizer.no_padding()
        encoded = tokenizer.encode_batch([(p["question"], p["passage"]) for p in pairs])
        if max((len(e.ids) for e in encoded), default=0) > 512:
            raise ValueError("Rerankerpaar te lang; geen truncatie.")
        tokenizer.enable_padding(
            direction="right", pad_id=tokenizer.token_to_id("<pad>"), pad_token="<pad>"
        )
        encoded = tokenizer.encode_batch([(p["question"], p["passage"]) for p in pairs])
        inputs = {
            name: np.asarray(value, dtype=np.int64)
            for name, value in [
                ("input_ids", [e.ids for e in encoded]),
                ("attention_mask", [e.attention_mask for e in encoded]),
                ("token_type_ids", [e.type_ids for e in encoded]),
            ]
        }
        raw = self.session.run(
            [self.output], {i.name: inputs[i.name] for i in self.session.get_inputs()}
        )[0]
        if (
            raw.dtype != np.float32
            or raw.shape != (len(pairs), 1)
            or not np.isfinite(raw).all()
        ):
            raise ValueError("Ongeldige rerankerscores.")
        return raw[:, 0]


def rank_cases(representations, scores, candidates):
    if set(representations) != set(candidates):
        raise ValueError("Reranker moet iedere kandidaat verantwoorden.")
    maxima = {}
    for case, refs in representations.items():
        if not refs:
            raise ValueError("Kandidaat zonder scores.")
        maxima[case] = max(float(scores[ref]) for ref in refs)
    return sorted(maxima, key=lambda c: (-maxima[c], c)), maxima


def run(root, model_root):
    if (root / "results.json").exists() or (root / "pair-scores.json").exists():
        raise ValueError(
            "Rerankerproef al gestart/afgerond; niet automatisch herhalen."
        )
    discovery, pairs, packets = frozen(root)
    _, _, questions, _, contexts, _, _ = load_discovery(discovery)
    reference = {q["id"]: q for q in read(discovery / "results.json")["results"]}
    qmap = {q["id"]: q for q in questions}
    reranker = LocalReranker(model_root / "reranker")
    scores = {}
    seconds = {}
    for i, pair in enumerate(pairs):
        if i % 20 == 0:
            print("reranker: pairs " + str(i) + "/" + str(len(pairs)), flush=True)
        start = time.perf_counter()
        scores[pair["pair_id"]] = float(reranker.score([pair])[0])
        seconds[pair["pair_id"]] = time.perf_counter() - start
    save(
        root / "pair-scores.json",
        dict(
            scores=scores, seconds=seconds, frozen_sha256=digest(root / "frozen.json")
        ),
    )
    probes = [
        pairs[i] for i in np.linspace(0, len(pairs) - 1, min(10, len(pairs)), dtype=int)
    ]
    actual = reranker.score(probes)
    expected = np.asarray([scores[p["pair_id"]] for p in probes], dtype=np.float32)
    results = []
    evidence = []
    for packet in packets:
        q = qmap[packet["question_id"]]
        methods = {}
        for name, reps in packet["representations"].items():
            ranking, maxima = rank_cases(reps, scores, packet["candidates"])
            selected = ranking[:3]
            methods[name] = dict(
                cases=selected,
                candidate_ranking=ranking,
                case_logits=maxima,
                metrics=metrics(selected, None, contexts, q["gold"]),
            )
            evidence.append(
                dict(
                    question_id=q["id"],
                    method=name,
                    evidence=evidence_packets(selected, contexts),
                )
            )
        refs = {
            ref
            for reps in packet["representations"].values()
            for group in reps.values()
            for ref in group
        }
        results.append(
            dict(
                id=q["id"],
                question=q["question"],
                boundary=not bool(q["gold"]),
                candidates=packet["candidates"],
                candidate_ceiling=metrics(
                    packet["candidates"], None, contexts, q["gold"]
                ),
                methods=methods,
                unique_pairs=len(refs),
                pair_seconds=sum(seconds[ref] for ref in refs),
            )
        )
    summary = {}
    gates = {}
    for name in ["rerank_primary", "rerank_with_auxiliary"]:
        vals = [q["methods"][name]["metrics"] for q in results if not q["boundary"]]
        summary[name] = {
            key: sum(v[key] for v in vals) / len(vals)
            for key in [
                "case_recall",
                "mrr",
                "full_anchor_recall",
                "reasoning_characters",
                "conclusion_characters",
            ]
        }
        summary[name] |= dict(
            anchor_hits=sum(sum(v["full_anchor_hits"]) for v in vals),
            anchor_total=sum(len(v["full_anchor_hits"]) for v in vals),
            complete=sum(v["complete"] for v in vals),
            questions=len(vals),
        )
        gains = []
        losses = []
        for q in results:
            if q["boundary"]:
                continue
            before = reference[q["id"]]["methods"]["hybrid_primary"]["metrics"][
                "full_anchor_recall"
            ]
            after = q["methods"][name]["metrics"]["full_anchor_recall"]
            if after > before:
                gains.append(q["id"])
            if after < before:
                losses.append(q["id"])
        gates[name] = dict(
            improvements=gains, regressions=losses, passes="V4" in gains and not losses
        )
    report = dict(
        summary=summary,
        gates=gates,
        results=results,
        model_spec=RERANKER,
        session=reranker.description,
        runtime={
            p: importlib.metadata.version(p)
            for p in ["numpy", "onnxruntime", "tokenizers"]
        },
        model_load_seconds=reranker.load_seconds,
        total_pair_seconds=sum(seconds.values()),
        unique_pairs=len(pairs),
        peak_working_set_bytes=peak_memory(),
        probe_batch_max_abs_difference=float(np.max(np.abs(actual - expected))),
        probe_pair_ids=[p["pair_id"] for p in probes],
        frozen_sha256=digest(root / "frozen.json"),
        score_is_answerability=False,
        answer_quality="not_measured",
        paid_calls=0,
        active_api_changed=False,
    )
    save(root / "evidence-packets.json", evidence)
    save(root / "results.json", report)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("results", "model_spec", "session")
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["download", "freeze", "run"])
    parser.add_argument("--root", type=Path, default=Path("local-results/reranking-v1"))
    parser.add_argument(
        "--discovery", type=Path, default=Path("local-results/discovery-context-v1")
    )
    parser.add_argument(
        "--protocol", type=Path, default=Path("experiments/reranking-protocol.json")
    )
    parser.add_argument(
        "--model-root", type=Path, default=Path("data/model-comparison/models")
    )
    args = parser.parse_args()
    if args.phase == "download":
        acquire("reranker", args.model_root, spec=RERANKER)
    elif args.phase == "freeze":
        freeze(args.root, args.discovery, args.protocol, args.model_root)
    else:
        run(args.root, args.model_root)
