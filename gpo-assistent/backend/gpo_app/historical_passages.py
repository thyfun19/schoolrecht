"""Local, extractive comparison. Case discovery and admissible evidence are separate."""

from collections import defaultdict
from functools import lru_cache
import re
import threading

import sqlalchemy as sa

from gpo_local.common import digest, sections
from gpo_local.source_registry import SourceRegistry
from gpo_local.retrieval import BM25
from . import models as m
from . import repository
from .ingestion import clean_legacy, identity
from .reranking_experiment import LocalReranker
from .title_search import BASE, available_pdfs, title_index


def evidence_windows(case, doc, pages, registered=None):
    # Reparse raw pages with explicit headings. Positional fallback roles from
    # a bulk import are insufficient for attributing a passage to the committee.
    parts = sections(
        [dict(case=case, page=p, text=clean_legacy(raw)) for p, raw in pages]
    )
    entries = SourceRegistry({case: doc}, parts).select([case])
    by_section = defaultdict(list)
    for e in entries:
        if e["category"] == "commissieoverwegingen":
            by_section[e["section_id"]].append(e)
    windows = []
    for group in by_section.values():
        for start in range(0, len(group), 2):
            block = group[start : start + 3]
            if registered is not None and any(
                e["source_id"] not in registered for e in block
            ):
                continue
            text = " ".join(e["quote"] for e in block)
            if len(text.split()) < 8:
                continue
            windows.append(
                dict(case=case, text=text, page=block[0]["page"], citations=block)
            )
    return windows


def registered_cases(engine, topic=None, sector=None):
    stmt = (
        sa.select(
            m.documents.c.case_number,
            m.versions.c.pdf_sha256,
            m.versions.c.source_url,
            m.versions.c.advice_date,
            m.versions.c.version_id,
            m.documents.c.topic,
            m.documents.c.sector,
            m.documents.c.publication_date,
            m.runs.c.processing_id,
        )
        .select_from(m.runs.join(m.versions).join(m.documents))
        .where(m.runs.c.state == "validated_not_indexed")
    )
    if topic:
        stmt = stmt.where(m.documents.c.topic == topic)
    if sector:
        stmt = stmt.where(m.documents.c.sector == sector)
    with engine.connect() as con:
        return [dict(row) for row in con.execute(stmt).mappings()]


def original_pages(engine, pid):
    with engine.connect() as con:
        return con.execute(
            sa.select(m.pages.c.page_number, m.pages.c.raw_text)
            .where(m.pages.c.processing_id == pid)
            .order_by(m.pages.c.page_number)
        ).all()


def registered_windows(engine, case, row):
    pid = row["processing_id"]
    pages = original_pages(engine, pid)
    with engine.connect() as con:
        ids = set(
            con.scalars(
                sa.select(m.citations.c.source_id).where(
                    m.citations.c.processing_id == pid
                )
            )
        )
    doc = dict(
        advice_date=row["advice_date"],
        source_url=row["source_url"],
        sha256=row["pdf_sha256"],
    )
    return evidence_windows(case, doc, pages, ids)


@lru_cache(maxsize=1)
def reranker():
    return LocalReranker(BASE / "data/model-comparison/models/reranker")


model_lock = threading.Lock()


def education_levels(text):
    return set(
        re.findall(r"\b(?:vmbo|mavo|havo|vwo|gymnasium|atheneum)\b", text.casefold())
    )


def rank_evidence(question, cases, load_windows, limit):
    levels = education_levels(question)
    candidates = []
    audit = []
    for case in cases:
        windows = load_windows(case)
        audit.append(dict(case_number=case, verified_windows=len(windows)))
        if not windows:
            continue
        scores = BM25(windows).scores(question)
        order = sorted(
            range(len(scores)),
            key=lambda i: (
                len(levels - education_levels(windows[i]["text"])),
                -scores[i],
            ),
        )
        candidates.extend(windows[i] for i in order[:2])
    ranked = {}
    if candidates:
        model = reranker()
        with model_lock:
            model.tokenizer.no_padding()
            for w in candidates:
                pair = dict(question=question, passage=w["text"])
                if len(model.tokenizer.encode(question, w["text"]).ids) > 512:
                    continue
                score = float(model.score([pair])[0])
                missing = sorted(levels - education_levels(w["text"]))
                priority = (-len(missing), score)
                if w["case"] not in ranked or priority > ranked[w["case"]][0]:
                    ranked[w["case"]] = (priority, w, missing)
    selected = sorted(ranked.values(), key=lambda item: item[0], reverse=True)[:limit]
    return dict(
        status="historical_passages" if selected else "insufficient_verified_evidence",
        results=[
            dict(
                case_number=w["case"],
                page=w["page"],
                citations=w["citations"],
                missing_education_levels=missing,
            )
            for _, w, missing in selected
        ],
        candidates=audit,
        answer_generated=False,
        semantic_support="requires_review",
        paid_calls=0,
    )


@lru_cache(maxsize=1024)
def pdf_hash(path, size, modified_ns):
    return digest(path)


def same_catalogue_document(index, case, row, pdfs):
    if case not in index.rows or not case.isdecimal():
        return False
    path = pdfs.get(case)
    if path is not None:
        stat = path.stat()
        return row["pdf_sha256"] == pdf_hash(path, stat.st_size, stat.st_mtime_ns)
    return index.rows[case][2].split("#")[0] == row["source_url"]


def compare(engine, question, limit=3, topic=None, sector=None):
    index = title_index()
    rows = registered_cases(engine, topic, sector)
    by_case = defaultdict(list)
    pdfs = available_pdfs()
    for row in rows:
        case = row["case_number"]
        if same_catalogue_document(index, case, row, pdfs):
            by_case[case].append(row)
    # Ambiguous document versions require resolution before showing citations.
    unique = {case: items[0] for case, items in by_case.items() if len(items) == 1}
    cases = index.search(question, 12, eligible=unique)
    result = rank_evidence(
        question,
        cases,
        lambda case: registered_windows(engine, case, unique[case]),
        limit,
    )
    for entry in result["results"]:
        row = unique[entry["case_number"]]
        entry.update(
            {
                k: row[k]
                for k in (
                    "processing_id",
                    "version_id",
                    "topic",
                    "sector",
                    "publication_date",
                    "advice_date",
                    "source_url",
                )
            }
        )
        entry["title"] = index.rows[entry["case_number"]][0]
        entry["context_url"] = f"/api/v1/extractions/{row['processing_id']}/context"
    return result


def search_response(engine, body):
    """Expose the candidate to the existing search client on opt-in deployments."""
    result = compare(
        engine, body.question, min(body.limit, 12), body.topic, body.sector
    )
    for entry in result["results"]:
        refs = entry["citations"]
        ids = [ref["source_id"] for ref in refs]
        entry["matched_chunks"] = [
            dict(
                chunk_id=identity("excerpt", *ids),
                source_ids=ids,
                text=" ".join(ref["quote"] for ref in refs),
                role="commissie",
                section=refs[0]["section"],
                page=entry["page"],
            )
        ]
        pid = entry["processing_id"]
        # Only display sections whose text, speaker and page also follow from
        # explicit headings in the raw registered pages.
        parsed = sections(
            [
                dict(case=entry["case_number"], page=n, text=clean_legacy(raw))
                for n, raw in original_pages(engine, pid)
            ]
        )
        allowed = {
            (p["section_id"], p["page"], p["section"], p["text"])
            for p in parsed
            if p["role"] == "commissie"
        }
        parts = repository.get_context(engine, pid)["parts"]
        for category in ("commissieoverwegingen", "conclusies"):
            entry[category] = [
                s
                for s in parts[category]
                if (s["section_key"], s["page_number"], s["heading"], s["text"])
                in allowed
            ]
    result.update(
        method="catalogue_bge_rerank",
        score_is_answerability=False,
        cases=[r["case_number"] for r in result["results"]],
        processing_ids=[r["processing_id"] for r in result["results"]],
    )
    return result
