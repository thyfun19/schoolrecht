"""Offline controlled batch/padding/quantization ablations. No re-embedding corpus."""

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper
from tokenizers import Tokenizer

from .common import REVISION, digest, read
from .retrieval import Retrieval, ordered


def inputs(tokenizer, texts, fixed_length=None):
    tokenizer.no_truncation()
    tokenizer.no_padding()
    tokenizer.enable_padding(pad_id=1, pad_token="<pad>", length=fixed_length)
    enc = tokenizer.encode_batch(["query: " + q for q in texts])
    return {
        k: np.asarray(values, dtype=np.int64)
        for k, values in [
            ("input_ids", [x.ids for x in enc]),
            ("attention_mask", [x.attention_mask for x in enc]),
            ("token_type_ids", [x.type_ids for x in enc]),
        ]
    }


def pool(hidden, mask):
    value = (hidden * mask[:, :, None]).sum(axis=1) / mask.sum(axis=1)[:, None]
    value /= np.linalg.norm(value, axis=1, keepdims=True)
    return value.astype(np.float32)


def investigate(study, bundle, database, output):
    output.mkdir(parents=True, exist_ok=True)
    engine = Retrieval(bundle, database)
    tokenizer = Tokenizer.from_file(str(bundle / "model/tokenizer.json"))
    questions = (
        read(study.parent / "besliscontrole/referentie.json")["questions"]
        + read(study / "nieuwe_vragen.json")["questions"]
    )
    q = questions[2]["question"]  # V3, fixed ahead of ablation, never altered.
    short = questions[8]["question"]
    long = questions[5]["question"]
    variants = {
        "alone": [q],
        "repeat": [q, q],
        "with_short": [q, short],
        "with_long": [q, long],
        "reversed": [long, q],
        "original_first8": [x["question"] for x in questions[:8]],
    }
    anchor = inputs(tokenizer, [q])
    runs = []
    for padding in [None, 128]:
        for name, texts in variants.items():
            target = texts.index(q)
            feed = inputs(tokenizer, texts, padding)
            mask = feed["attention_mask"][target]
            n = int(mask.sum())
            for key in ["input_ids", "attention_mask", "token_type_ids"]:
                assert np.array_equal(feed[key][target][:n], anchor[key][0][:n])
            assert np.all(mask[:n] == 1) and np.all(mask[n:] == 0)
            assert np.all(feed["input_ids"][target, n:] == 1)
            hidden = engine.session.run(None, feed)[0]
            v = pool(hidden, feed["attention_mask"])[target]
            manual = hidden[target, :n].mean(axis=0)
            manual /= np.linalg.norm(manual)
            assert np.allclose(v, manual, atol=1e-6)
            scores = engine.vectors @ v
            cases = []
            for i in ordered(scores):
                if engine.chunks[i]["case"] not in cases:
                    cases.append(engine.chunks[i]["case"])
            runs.append(
                dict(
                    name=name,
                    fixed_padding=padding,
                    target_index=target,
                    sequence_length=feed["input_ids"].shape[1],
                    token_count=n,
                    embedding=v.tolist(),
                    cases=cases[:3],
                    pool_matches_unpadded_mean=True,
                    unit_norm=float(np.linalg.norm(v)),
                )
            )
    baseline = np.asarray(runs[0]["embedding"])
    for run in runs:
        run["cosine_to_alone"] = float(baseline @ np.asarray(run["embedding"]))
    model = onnx.load(bundle / "model/model_quantized.onnx")
    counts = collections.Counter(n.op_type for n in model.graph.node)
    quant = next(n for n in model.graph.node if n.op_type == "DynamicQuantizeLinear")
    # Expose the activation before quantization and its tensor-wide parameters.
    model.graph.output.extend(
        [
            helper.make_tensor_value_info(quant.input[0], TensorProto.FLOAT, None),
            helper.make_tensor_value_info(quant.output[0], TensorProto.UINT8, None),
            helper.make_tensor_value_info(quant.output[1], TensorProto.FLOAT, None),
            helper.make_tensor_value_info(quant.output[2], TensorProto.UINT8, None),
        ]
    )
    debug_path = output / "instrumented.onnx"
    onnx.save(model, debug_path)
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    opts.intra_op_num_threads = 2
    debug = ort.InferenceSession(
        str(debug_path), sess_options=opts, providers=["CPUExecutionProvider"]
    )
    evidence = []
    floating = []
    for texts in [[q], [q, long]]:
        feed = inputs(tokenizer, texts, 128)
        hidden, activation, quantized, scale, zero = debug.run(None, feed)
        floating.append(activation[0].copy())
        evidence.append(
            dict(
                batch_size=len(texts),
                first_scale=float(scale),
                first_zero_point=int(zero),
                activation_min=float(activation.min()),
                activation_max=float(activation.max()),
                target_quantized_sha256=__import__("hashlib")
                .sha256(quantized[0].tobytes())
                .hexdigest(),
                embedding=pool(hidden, feed["attention_mask"])[0].tolist(),
            )
        )
    difference = float(np.max(np.abs(floating[0] - floating[1])))
    result = dict(
        revision=REVISION,
        model_sha256=digest(bundle / "model/model_quantized.onnx"),
        tokenizer_sha256=digest(bundle / "model/tokenizer.json"),
        query_prefix="query: ",
        corpus_prefix="passage: ",
        max_tokens=512,
        dynamic_quantize_nodes=counts["DynamicQuantizeLinear"],
        integer_matmul_nodes=counts["MatMulInteger"],
        first_quantizer_input=quant.input[0],
        fixed_padding_instrumentation=evidence,
        prequant_target_max_abs_difference=difference,
        target_tokens_identical=True,
        correct_attention_mask=True,
        correct_pooling=True,
        normalized=True,
        rank_rule="maximum fragment cosine per case; deterministic fragment index ties",
        runs=runs,
    )
    (output / "batch-cause.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ["runs", "fixed_padding_instrumentation"]
            },
            indent=2,
        )
    )
    print(
        "Instrumentation:",
        [{k: v for k, v in x.items() if k != "embedding"} for x in evidence],
    )
    print(
        "Ranks:",
        [
            (x["name"], x["fixed_padding"], round(x["cosine_to_alone"], 6), x["cases"])
            for x in runs
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=Path("data/e5"))
    parser.add_argument("--database", type=Path, default=Path("data/gpo.sqlite3"))
    parser.add_argument(
        "--output", type=Path, default=Path("local-results/e5-stability")
    )
    args = parser.parse_args()
    investigate(args.study, args.bundle, args.database, args.output)
