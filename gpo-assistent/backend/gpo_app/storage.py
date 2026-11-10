"""Immutable local PDF snapshots addressed by SHA-256; originals are never modified."""

import hashlib
import os
from pathlib import Path
import tempfile

DEFAULT_STORAGE_ROOT = Path(__file__).resolve().parents[1] / "data/registry/pdfs"


def snapshot_pdf(pdf):
    try:
        content = Path(pdf["local_path"]).read_bytes()
    except OSError:
        raise ValueError(
            "Oorspronkelijke PDF niet beschikbaar tijdens import."
        ) from None
    if hashlib.sha256(content).hexdigest() != pdf["sha256"]:
        raise ValueError(
            "PDF gewijzigd tijdens import; niets als gecontroleerde versie registreren."
        )
    directory = DEFAULT_STORAGE_ROOT.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (pdf["sha256"] + ".pdf")
    if not target.exists():
        with tempfile.NamedTemporaryFile(
            dir=directory, prefix="snapshot-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        try:
            # Publish without overwriting another writer's completed snapshot.
            try:
                os.link(temporary, target)
            except FileExistsError:
                pass
        finally:
            temporary.unlink(missing_ok=True)
    if hashlib.sha256(target.read_bytes()).hexdigest() != pdf["sha256"]:
        raise ValueError("Bestaande PDF-snapshot is gewijzigd; niet overschrijven.")
    return {**pdf, "local_path": str(target)}
