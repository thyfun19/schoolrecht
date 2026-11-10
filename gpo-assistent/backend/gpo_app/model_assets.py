"""Public model asset management for local ONNX cross-encoders and tokenizers."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

SPECS = {
    "e5": dict(
        repository="Xenova/multilingual-e5-small",
        revision="761b726dd34fb83930e26aab4e9ac3899aa1fa78",
        base_model="intfloat/multilingual-e5-small",
        dimensions=384,
        pooling="masked_mean",
        query_prefix="query: ",
        passage_prefix="passage: ",
        files={
            "onnx/model.onnx": (
                470268533,
                "4aa845c27760e06e9a686b9d8b5d440eae4b6612cd09e5b522b716d3941f77ff",
            ),
            "tokenizer.json": (
                17082730,
                "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39",
            ),
        },
    ),
    "bge": dict(
        repository="Xenova/bge-m3",
        revision="4de13258303883538bd53b696b452bf8099f0858",
        base_model="BAAI/bge-m3",
        dimensions=1024,
        pooling="cls",
        query_prefix="",
        passage_prefix="",
        files={
            "onnx/model.onnx": (
                607298,
                "5d89a0010dd39aa2cfa8b22bb49f06904c5bbf5877135f877da419480f40cde3",
            ),
            "onnx/model.onnx_data": (
                2266820608,
                "1eebfb28493f67bba03ce0ef64bfdc7fc5a3bd9d7493f818bb1d78cd798416b4",
            ),
            "tokenizer.json": (
                17082821,
                "6710678b12670bc442b99edc952c4d996ae309a7020c1fa0096dd245c2faf790",
            ),
        },
    ),
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def acquire(name, root, existing_e5=Path("data/e5/model"), *, spec=None):
    spec = SPECS[name] if spec is None else spec
    target = Path(root) / name
    target.mkdir(parents=True, exist_ok=True)
    lock = target / "download.lock"
    with lock.open("x") as handle:
        try:
            records = []
            for remote, (size, expected) in spec["files"].items():
                path = target / Path(remote).name
                if path.exists():
                    if path.stat().st_size != size or sha256(path) != expected:
                        raise ValueError(
                            "Bestaand modelbestand wijkt af; niet overschrijven."
                        )
                    print(name + ": verified " + path.name, flush=True)
                else:
                    reuse = existing_e5 / path.name
                    if (
                        name == "e5"
                        and reuse.is_file()
                        and reuse.stat().st_size == size
                        and sha256(reuse) == expected
                    ):
                        shutil.copyfile(reuse, path)
                        print(name + ": reused " + path.name, flush=True)
                    else:
                        url = f"https://huggingface.co/{spec['repository']}/resolve/{spec['revision']}/{remote}"
                        request = urllib.request.Request(
                            url,
                            headers={"User-Agent": "GPO-local-model-comparison/1.0"},
                        )
                        partial = path.with_suffix(path.suffix + ".part")
                        print(
                            name
                            + ": downloading "
                            + path.name
                            + " ("
                            + str(size)
                            + " bytes)",
                            flush=True,
                        )
                        with (
                            urllib.request.urlopen(request, timeout=60) as response,
                            partial.open("wb") as out,
                        ):
                            shutil.copyfileobj(response, out, length=1024 * 1024)
                        if (
                            partial.stat().st_size != size
                            or sha256(partial) != expected
                        ):
                            raise ValueError(
                                "Modeldownload heeft verkeerde grootte/hash; niet gebruiken."
                            )
                        partial.replace(path)
                records.append(dict(file=path.name, bytes=size, sha256=expected))
            manifest = {k: v for k, v in spec.items() if k != "files"} | dict(
                files=records, precision="float32", storage="local_only"
            )
            (target / "manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
        finally:
            handle.close()  # Windows cannot unlink an open file.
            lock.unlink()
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(SPECS) + ["all"], default="all")
    parser.add_argument(
        "--root", type=Path, default=Path("data/model-comparison/models")
    )
    args = parser.parse_args()
    for name in SPECS if args.model == "all" else [args.model]:
        acquire(name, args.root)
