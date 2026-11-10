"""Optional local hybrid candidate search using the existing pinned research assets."""

from functools import lru_cache
from pathlib import Path
import threading

import numpy as np

from gpo_local.common import digest, read
from gpo_local.retrieval import BM25, Retrieval
from .discovery_context import rrf
from .model_comparison import Encoder, case_ranking
from .model_assets import SPECS
from .ingestion import json_hash

BASE = Path(__file__).resolve().parents[1]


class CandidateSearch:
    def __init__(self, method):
        self.method = method
        root = BASE / "local-results/model-comparison-v1"
        manifest = read(root / "frozen.json")
        for name, expected in manifest["hashes"].items():
            if digest(root / name) != expected:
                raise ValueError("Bevroren zoekcorpus gewijzigd.")
        if json_hash(manifest["model_specs"]) != json_hash(SPECS):
            raise ValueError("Modelspecificatie wijkt af van de corpusvectoren.")
        self.contexts = read(root / "contexts.json")
        self.lock = threading.Lock()
        if method == "text":
            self.legacy = Retrieval(BASE / "data/e5", BASE / "data/gpo.sqlite3")
        elif method == "hybrid_bge":
            self.chunks = read(root / "chunks.json")
            self.text = BM25(self.chunks)
            run = read(root / "bge-run.json")
            if (
                digest(root / "bge-vectors.npy") != run["vectors_sha256"]
                or digest(root / "frozen.json") != run["frozen_sha256"]
            ):
                raise ValueError("Vectorbestand gewijzigd.")
            self.vectors = np.load(root / "bge-vectors.npy", allow_pickle=False)
            if (
                self.vectors.shape != (len(self.chunks), 1024)
                or not np.isfinite(self.vectors).all()
            ):
                raise ValueError("Ongeldige vectorindex.")
            self.encoder = Encoder("bge", BASE / "data/model-comparison/models/bge")
        else:
            raise ValueError("Onbekende kandidaatzoekmethode.")

    def search(self, question, limit=3):
        question = question.strip()
        if not 3 <= len(question.strip()) <= 2000 or not 1 <= limit <= 5:
            raise ValueError("Ongeldige vraag of kandidaatlimiet.")
        if self.method == "text":
            response = self.legacy.search(question, "text", limit)
            ranking = [c["case_number"] for c in response["results"]]
        else:
            with self.lock:
                query, _ = self.encoder.encode([question], "query", batch_size=1)
            text = case_ranking(self.chunks, self.text.scores(question), True)
            vector = case_ranking(self.chunks, self.vectors @ query[0])
            ranking = rrf(text, vector)[:limit]
        return dict(
            method=self.method,
            cases=ranking,
            processing_ids=[self.contexts[c]["processing_id"] for c in ranking],
            score_is_answerability=False,
            answer_generated=False,
            paid_calls=0,
            scope="Frozen ten-document research corpus; no automatic collection expansion",
        )


@lru_cache(maxsize=2)
def candidate_search(method):
    return CandidateSearch(method)
