"""Import cached study artifacts, never download or re-embed. Run explicitly."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from .common import REVISION, digest, norm, read, sections


def prepare(study: Path, output: Path, catalog_path: Path):
    if output.exists():
        raise ValueError(
            "Doel bestaat al; kies een nieuwe map. Bestaande gegevens blijven behouden."
        )
    catalog = {d["case_number"]: d for d in read(catalog_path)["documents"]}
    manifest = read(study / "model/manifest.json")
    if manifest["revision"] != REVISION:
        raise ValueError("Onverwachte modelrevision.")
    for f in manifest["files"]:
        if digest(study / "model" / f["file"]) != f["sha256"]:
            raise ValueError("Modelbestand wijkt af van gepinde hash.")
    pages = read(study.parent / "data/pages.json")
    chunks = read(study / "fragmenten.json")
    vectors = np.load(study / "vectors.npy", allow_pickle=False)
    if vectors.shape != (len(chunks), 384) or not np.isfinite(vectors).all():
        raise ValueError("Vectorbestand ongeldig.")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
        raise ValueError("Vectoren zijn niet genormaliseerd.")
    full = sections(pages)
    lookup = {s["section_id"]: s for s in full}
    if len({c["id"] for c in chunks}) != len(chunks):
        raise ValueError("Dubbele fragment-ID.")
    for case, d in catalog.items():
        if digest(study.parent / "data/pdf" / f"{case}.pdf") != d["sha256"]:
            raise ValueError("PDF verschilt van catalogus.")
    for c in chunks:
        d = catalog[c["case"]]
        s = lookup[c["section_id"]]
        if (
            c["pdf_sha256"] != d["sha256"]
            or c["page"] != s["page"]
            or c["role"] != s["role"]
            or c["section"] != s["section"]
            or c["url"] != d["source_url"] + f"#page={c['page']}"
            or c["text"] != s["text"][c["start_offset"] : c["end_offset"]]
        ):
            raise ValueError("Fragment of herkomst wijkt af.")
    for s in full:
        s["url"] = catalog[s["case"]]["source_url"] + f"#page={s['page']}"
    # Validation precedes every write. Output is a new ignored data directory.
    output.mkdir(parents=True)
    shutil.copytree(study / "model", output / "model")
    shutil.copy2(study / "vectors.npy", output / "vectors.npy")
    shutil.copy2(study / "fragmenten.json", output / "chunks.json")
    (output / "corpus.json").write_text(
        json.dumps(
            dict(documents=list(catalog.values()), sections=full), ensure_ascii=False
        ),
        encoding="utf-8",
    )
    hashes = {
        name: digest(output / name)
        for name in ["vectors.npy", "chunks.json", "corpus.json"]
    }
    (output / "bundle.json").write_text(
        json.dumps(
            dict(
                revision=REVISION,
                hashes=hashes,
                model=manifest,
                passage_prefix="passage: ",
                query_prefix="query: ",
                dimensions=384,
                chunk_tokens=320,
                overlap_tokens=48,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    return dict(
        documents=len(catalog),
        pages=len(pages),
        fragments=len(chunks),
        reembedded=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/e5"))
    parser.add_argument(
        "--catalog", type=Path, default=Path("../sources/catalogus.json")
    )
    args = parser.parse_args()
    print(prepare(args.study.resolve(), args.output.resolve(), args.catalog.resolve()))
