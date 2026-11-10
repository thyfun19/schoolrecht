"""Case discovery from official titles; evidence always comes from the registered PDF."""

import hashlib
import html
import json
import re
from functools import lru_cache
from pathlib import Path
import threading

import numpy as np

from .model_comparison import Encoder
from .model_assets import SPECS
from gpo_local.common import digest
from gpo_local.retrieval import BM25

BASE = Path(__file__).resolve().parents[1]
EXPORT = BASE / "data/full_gpo/verdicts_raw.json"
INDEX = BASE / "data/full_gpo/title-bge.npz"


def available_pdfs():
    paths = {}
    for directory in (
        BASE / "data/full_gpo/pdfs",
        BASE.parents[1] / "onderzoek_gpo/data/pdf",
    ):
        for path in sorted(directory.glob("*.pdf")):
            if path.stem in paths and digest(path) != digest(paths[path.stem]):
                raise ValueError("Meerdere PDF-versies vereisen expliciete selectie.")
            paths[path.stem] = path
    return paths


def plain(value):
    return " ".join(re.sub(r"<[^>]+>", " ", html.unescape(value)).split())


def records(export):
    items = json.loads(export.read_bytes())
    result = {}
    for item in items:
        overview = item.get("overview") or {}
        case = str(overview.get("publicationNumber") or "").strip()
        title = plain(item["title"]["rendered"])
        if not case or not title or not (item.get("file") or {}).get("url"):
            continue
        if case in result:
            raise ValueError("Dubbele zaak in catalogus.")
        result[case] = (title, plain(item["content"]["rendered"]), item["file"]["url"])
    return result


def fingerprint(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def build():
    rows = records(EXPORT)
    encoder = Encoder("bge", BASE / "data/model-comparison/models/bge")
    vectors, _ = encoder.encode(
        [rows[c][0] for c in sorted(rows)], "passage", batch_size=4
    )
    temp = INDEX.with_suffix(".tmp.npz")
    np.savez_compressed(
        temp,
        vectors=vectors,
        fingerprint=fingerprint(rows),
        revision=SPECS["bge"]["revision"],
    )
    temp.replace(INDEX)


class TitleIndex:
    def __init__(self, encoder):
        self.rows = records(EXPORT)
        self.cases = sorted(self.rows)
        with np.load(INDEX, allow_pickle=False) as archive:
            self.vectors = archive["vectors"]
            if (
                archive["fingerprint"].item() != fingerprint(self.rows)
                or archive["revision"].item() != SPECS["bge"]["revision"]
                or self.vectors.shape != (len(self.cases), SPECS["bge"]["dimensions"])
                or not np.isfinite(self.vectors).all()
            ):
                raise ValueError("Catalogusindex komt niet overeen met de bronversie.")
        self.lexical = BM25(
            [{"case": c, "text": " ".join(self.rows[c][:2])} for c in self.cases]
        )
        self.encoder = encoder
        self.lock = threading.Lock()

    def search(self, question, limit, eligible=None):
        with self.lock:
            query, _ = self.encoder.encode([question], "query")
        text_scores = self.lexical.scores(question)
        ranks = np.zeros(len(self.cases))
        for scores in (self.vectors @ query[0], text_scores):
            order = [
                i
                for i in np.argsort(-scores, kind="stable")
                if eligible is None or self.cases[i] in eligible
            ]
            for rank, i in enumerate(order[:50], 1):
                if scores[i] > 0:
                    ranks[i] += 1 / (60 + rank)
        return [
            self.cases[i]
            for i in np.argsort(-ranks, kind="stable")[:limit]
            if ranks[i] > 0
        ]


@lru_cache(maxsize=1)
def title_index():
    return TitleIndex(Encoder("bge", BASE / "data/model-comparison/models/bge"))


if __name__ == "__main__":
    build()
