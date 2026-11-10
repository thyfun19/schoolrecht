import collections
import math
import re
import sqlite3
import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from .common import REVISION, digest, norm, read

STOP = set(
    "de het een en van voor op in is zijn wordt worden dat die dit met om te tot aan als bij door naar over of er ook niet wel kan moet heeft hebben was welke wat waarom hoe ik mijn deze dan".split()
)


def words(s):
    return [w for w in re.findall(r"\w+", s.lower()) if w not in STOP and len(w) > 1]


class BM25:
    def __init__(self, chunks):
        self.docs = [
            collections.Counter(words(c["case"] + " " + c["text"])) for c in chunks
        ]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.average = sum(self.lengths) / len(self.docs)
        self.df = collections.Counter(w for d in self.docs for w in d)

    def scores(self, question):
        n = len(self.docs)
        return np.asarray(
            [
                sum(
                    math.log(1 + (n - self.df[w] + 0.5) / (self.df[w] + 0.5))
                    * d[w]
                    * 2.5
                    / (d[w] + 1.5 * (0.25 + 0.75 * length / self.average))
                    for w in words(question)
                    if d[w]
                )
                for d, length in zip(self.docs, self.lengths)
            ]
        )


def ordered(scores):
    return sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))


class Retrieval:
    def __init__(self, bundle: Path, database: Path):
        manifest = read(bundle / "bundle.json")
        if manifest["revision"] != REVISION:
            raise ValueError("Modelrevision niet ondersteund.")
        for name, expected in manifest["hashes"].items():
            if digest(bundle / name) != expected:
                raise ValueError(f"Indexbestand gewijzigd: {name}")
        for f in manifest["model"]["files"]:
            if digest(bundle / "model" / f["file"]) != f["sha256"]:
                raise ValueError("Modelhash wijkt af.")
        self.chunks = read(bundle / "chunks.json")
        self.vectors = np.load(bundle / "vectors.npy", allow_pickle=False)
        if (
            self.vectors.shape != (len(self.chunks), 384)
            or not np.isfinite(self.vectors).all()
        ):
            raise ValueError("Ongeldige vectorindex.")
        corpus = read(bundle / "corpus.json")
        self.documents = {d["case_number"]: d for d in corpus["documents"]}
        self.sections = corpus["sections"]
        self.text_index = BM25(self.chunks)
        self.tokenizer = Tokenizer.from_file(str(bundle / "model/tokenizer.json"))
        self.lock = threading.Lock()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads, opts.inter_op_num_threads = 2, 1
        self.session = ort.InferenceSession(
            str(bundle / "model/model_quantized.onnx"),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as con:
            con.row_factory = sqlite3.Row
            registered = con.execute(
                "SELECT case_number,sha256 FROM documents"
            ).fetchall()
            if {d["case_number"]: d["sha256"] for d in registered} != {
                k: v["sha256"] for k, v in self.documents.items()
            }:
                raise ValueError("SQLite-registratie past niet bij vectorcorpus.")
            self.legacy_chunks = [
                dict(row)
                for row in con.execute("""SELECT p.text, p.id,
                d.case_number AS "case", s.page_number AS page,s.role,s.heading AS section,
                d.source_url AS url FROM search_passages p JOIN sections s ON p.section_id=s.id
                JOIN documents d ON s.document_id=d.id ORDER BY d.case_number,p.id""")
            ]
        self.legacy_index = BM25(self.legacy_chunks)

    def embed_queries(self, questions):
        # Tokenizer padding configuration is mutable; serialize query access.
        with self.lock:
            self.tokenizer.no_truncation()
            self.tokenizer.no_padding()
            if any(
                len(self.tokenizer.encode("query: " + q).ids) > 512 for q in questions
            ):
                raise ValueError(
                    "Vraag is langer dan 512 modeltokens; verkort de vraag."
                )
            self.tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
            vectors = []
            # DynamicQuantizeLinear uses tensor-wide activation ranges. Other queries
            # or padding in a shared forward pass change this query's embedding.
            # One forward per query makes single and batch endpoints equivalent.
            for question in questions:
                encoded = self.tokenizer.encode_batch(["query: " + question])
                available = {
                    k: np.asarray(value, dtype=np.int64)
                    for k, value in [
                        ("input_ids", [x.ids for x in encoded]),
                        ("attention_mask", [x.attention_mask for x in encoded]),
                        ("token_type_ids", [x.type_ids for x in encoded]),
                    ]
                }
                hidden = self.session.run(
                    None, {x.name: available[x.name] for x in self.session.get_inputs()}
                )[0]
                mask = available["attention_mask"]
                pooled = (hidden * mask[:, :, None]).sum(axis=1) / mask.sum(axis=1)[
                    :, None
                ]
                pooled /= np.maximum(
                    np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9
                )
                vectors.extend(pooled)
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, question):
        return self.embed_queries([question])[0]

    def search_many(self, questions, method="vector", limit=3):
        vectors = (
            self.embed_queries(questions)
            if method == "vector"
            else [None] * len(questions)
        )
        return [
            self.search(q, method, limit, embedding=v)
            for q, v in zip(questions, vectors)
        ]

    def search(self, question, method="vector", limit=3, *, embedding=None):
        chunks = self.legacy_chunks if method == "text" else self.chunks
        if method == "vector":
            scores = self.vectors @ (
                self.embed_query(question) if embedding is None else embedding
            )
        else:
            scores = (
                self.legacy_index if method == "text" else self.text_index
            ).scores(question)
        case_order, matches = [], {}
        for i in ordered(scores):
            if method != "vector" and scores[i] <= 0:
                continue
            c = chunks[i]
            if c["case"] not in matches:
                case_order.append(c["case"])
                matches[c["case"]] = []
            if len(matches[c["case"]]) < 3:
                matches[c["case"]].append(
                    {
                        **c,
                        "url": self.documents[c["case"]]["source_url"]
                        + f"#page={c['page']}",
                        "score": float(scores[i]),
                    }
                )
        results = []
        for case in case_order[:limit]:
            parts = dict(
                commissieoverwegingen=[],
                conclusies=[],
                feiten=[],
                partijstandpunten=[],
                overige=[],
            )
            for s in self.sections:
                if s["case"] != case:
                    continue
                category = (
                    "commissieoverwegingen"
                    if s["role"] == "commissie" and "OVERWEGINGEN" in s["section"]
                    else "conclusies"
                    if s["role"] == "commissie"
                    and s["section"] in ["ADVIES", "OORDEEL", "AANBEVELINGEN"]
                    else "feiten"
                    if s["role"] == "feiten"
                    else "partijstandpunten"
                    if s["role"]
                    in ["school", "ouders", "leerling", "samenwerkingsverband"]
                    else "overige"
                )
                parts[category].append(s)
            results.append(
                {**self.documents[case], "matches": matches[case], "parts": parts}
            )
        return dict(
            method=method,
            model_revision=REVISION if method == "vector" else None,
            answerability="not_assessed",
            score_is_answerability=False,
            answer_generated=False,
            query_batch_sensitive=False,
            embedding_execution="one_query_per_forward_v1"
            if method == "vector"
            else None,
            vector_experimental=method == "vector",
            results=results,
        )
