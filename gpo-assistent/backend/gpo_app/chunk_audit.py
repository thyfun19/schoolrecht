"""Read-only technical coverage audit; does not assess retrieval or legal meaning."""

from collections import Counter
import sqlalchemy as sa
from . import models as m
from .chunking import read_preparation


def audit(engine):
    with engine.connect() as con:
        ids = list(
            con.scalars(
                sa.select(m.chunk_runs.c.chunk_run_id).order_by(
                    m.chunk_runs.c.chunk_run_id
                )
            )
        )
    rows = []
    for run_id in ids:
        run = read_preparation(engine, run_id)
        chunks = [c for p in run["contexts"] for c in p["chunks"]]
        spans = [s for p in run["contexts"] for s in p["spans"]]
        rows.append(
            dict(
                case_number=run["case_number"],
                chunk_run_id=run_id,
                processing_id=run["processing_id"],
                contexts=len(run["contexts"]),
                chunks=len(chunks),
                characters=sum(len(s["text"]) for s in spans),
                max_tokens=max((c["token_count"] for c in chunks), default=0),
                token_limit=run["config"]["token_limit"],
                cross_page_contexts=sum(
                    len({s["page_number"] for s in p["spans"]}) > 1
                    for p in run["contexts"]
                ),
                split_reasons=dict(Counter(c["split_reason"] for c in chunks)),
                source_ids=len(
                    {ref for c in chunks for ref in c["span"]["source_ids"]}
                ),
            )
        )
    return dict(
        runs=len(rows),
        contexts=sum(r["contexts"] for r in rows),
        chunks=sum(r["chunks"] for r in rows),
        source_character_coverage="exact_nonoverlapping",
        semantic_argument_boundaries="not_validated",
        retrieval_quality="not_evaluated",
        documents=sorted(rows, key=lambda r: r["case_number"]),
    )
