"""Offline search children with exact source spans and conservative full-role parents.

Paragraph layout is a structural hint, never evidence of a complete legal argument.
No model inference, network, source rewriting or expected-case features.
"""

import hashlib
import re
from pathlib import Path

import sqlalchemy as sa

from gpo_local.common import norm
from . import models as m
from .ingestion import identity, json_hash

POLICY = "paragraph-children-full-role-fallback-v1"
CONFIG_FIELDS = (
    "policy",
    "token_limit",
    "tokenizer_sha256",
    "prefix",
    "special_tokens",
    "overlap",
)


def chunk_run_identity(processing_id, config, input_sha256):
    # JSONB does not retain object key order. Freeze the v1 serialization order
    # explicitly, including after a database round trip; retain existing v1 IDs.
    if set(config) != set(CONFIG_FIELDS):
        raise ValueError("Verwerkingsconfiguratie gewijzigd.")
    ordered = {key: config[key] for key in CONFIG_FIELDS}
    return identity("chk", processing_id, ordered, input_sha256)


class LocalTokenCounter:
    def __init__(self, path):
        # Optional offline tooling, not imported by API startup.
        from tokenizers import Tokenizer

        path = Path(path)
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.tokenizer = Tokenizer.from_file(str(path))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()

    def __call__(self, text):
        return len(
            self.tokenizer.encode("passage: " + text, add_special_tokens=True).ids
        )


def paragraph_cuts(section, page):
    """Recover only exactly alignable blank-line boundaries; otherwise use section."""
    text = section["text"]
    clean = page["cleaned_text"]
    flat = norm(clean)
    start = flat.find(text)
    if start < 0 or flat.find(text, start + 1) >= 0:
        return [0, len(text)], "section_alignment_uncertain"
    cuts = [0]
    for match in re.finditer(r"\n[ \t\r]*\n", clean):
        prefix = norm(clean[: match.start()])
        # Normalization around a boundary must not invent or remove source characters.
        if not flat.startswith(prefix):
            continue
        offset = len(prefix) - start
        if 0 < offset < len(text) and text[offset : offset + 1] == " ":
            cuts.append(offset)
    return sorted(set(cuts + [len(text)])), "layout_only"


def split_span(text, start, end, count, limit):
    """Exact, non-overlapping coverage; prefer sentence/word ends, no truncation."""
    while start < end:
        if count(text[start:end]) <= limit:
            yield start, end, "paragraph_or_section"
            return
        # Token counts need not be monotonic at every character. Binary search finds
        # a candidate only; verify the emitted slice explicitly below.
        lo, hi = start + 1, end
        if count(text[start:lo]) > limit:
            raise ValueError("Tokenbudget te klein voor één teken.")
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if count(text[start:mid]) <= limit:
                lo = mid
            else:
                hi = mid
        candidate = lo
        preferred = [
            m.end() for m in re.finditer(r"[.!?](?:\s+|$)|\s+", text[start:candidate])
        ]
        if preferred and preferred[-1] > 0:
            candidate = start + preferred[-1]
        while candidate > start and count(text[start:candidate]) > limit:
            candidate -= 1
        if candidate <= start:
            raise ValueError("Geen veilig zoekfragment binnen tokenbudget.")
        yield start, candidate, "token_budget_continuation"
        start = candidate


def build(
    processing_id, sections, pages, citations, count, tokenizer_sha256, limit=384
):
    if limit < 8:
        raise ValueError("Tokenbudget moet ten minste acht zijn.")
    config = dict(
        policy=POLICY,
        token_limit=limit,
        tokenizer_sha256=tokenizer_sha256,
        prefix="passage: ",
        special_tokens=True,
        overlap=0,
    )
    fingerprint = json_hash(dict(sections=sections, pages=pages, citations=citations))
    run = dict(
        chunk_run_id=chunk_run_identity(processing_id, config, fingerprint),
        processing_id=processing_id,
        config=config,
        input_sha256=fingerprint,
    )
    selected = [
        s for s in sections if s["category"] in ("commissieoverwegingen", "conclusies")
    ]
    page_map = {p["page_number"]: p for p in pages}
    parents = []
    children = []
    # All reasoning of the same source role stays accessible, even across pages.
    for category, role in sorted({(s["category"], s["role"]) for s in selected}):
        group = sorted(
            [s for s in selected if (s["category"], s["role"]) == (category, role)],
            key=lambda s: s["ordinal"],
        )
        cid = identity("ctx", run["chunk_run_id"], category, role)
        spans = []
        ordinal = 0
        for section in group:

            def span(a, b):
                refs = [
                    c["source_id"]
                    for c in citations
                    if c["section_key"] == section["section_key"]
                    and c["start_offset"] < b
                    and c["end_offset"] > a
                ]
                if not refs:
                    raise ValueError(
                        "Zoekfragment mist geregistreerde bronverwijzingen."
                    )
                return dict(
                    section_key=section["section_key"],
                    page_number=section["page_number"],
                    start_offset=a,
                    end_offset=b,
                    source_ids=refs,
                )

            spans.append(span(0, len(section["text"])))
            cuts, status = paragraph_cuts(section, page_map[section["page_number"]])
            for a, b in zip(cuts, cuts[1:]):
                for start, end, reason in split_span(
                    section["text"], a, b, count, limit
                ):
                    piece = section["text"][start:end]
                    children.append(
                        dict(
                            chunk_id=identity(
                                "frag", cid, section["section_key"], start, end
                            ),
                            context_id=cid,
                            ordinal=ordinal,
                            text=piece,
                            token_count=count(piece),
                            span=span(start, end),
                            split_reason=status + ":" + reason,
                        )
                    )
                    ordinal += 1
        parents.append(
            dict(
                context_id=cid,
                chunk_run_id=run["chunk_run_id"],
                category=category,
                role=role,
                boundary_status="argument_boundaries_uncertain_full_role_context",
                spans=spans,
            )
        )
    return run, parents, children


def prepare_chunks(engine, counter, limit=384):
    totals = dict(runs_added=0, contexts_added=0, chunks_added=0, already_registered=0)
    with engine.begin() as con:
        con.execute(
            sa.text(
                "SELECT pg_advisory_xact_lock(hashtext('gpo-chunk-preparation-v1'))"
            )
        )
        runs = (
            con.execute(
                sa.select(m.runs)
                .where(m.runs.c.state == "validated_not_indexed")
                .order_by(m.runs.c.processing_id)
            )
            .mappings()
            .all()
        )
        for source in runs:
            pid = source["processing_id"]

            def rows(table, order):
                return [
                    dict(r)
                    for r in con.execute(
                        sa.select(table)
                        .where(table.c.processing_id == pid)
                        .order_by(order)
                    ).mappings()
                ]

            result = build(
                pid,
                rows(m.sections, m.sections.c.ordinal),
                rows(m.pages, m.pages.c.page_number),
                rows(m.citations, m.citations.c.source_id),
                counter,
                counter.sha256,
                limit,
            )
            run, parents, children = result
            if con.scalar(
                sa.select(m.chunk_runs.c.chunk_run_id).where(
                    m.chunk_runs.c.chunk_run_id == run["chunk_run_id"]
                )
            ):
                stored_parents = [
                    dict(r)
                    for r in con.execute(
                        sa.select(m.contexts).where(
                            m.contexts.c.chunk_run_id == run["chunk_run_id"]
                        )
                    ).mappings()
                ]
                stored_children = [
                    dict(r)
                    for r in con.execute(
                        sa.select(m.chunks)
                        .join(m.contexts)
                        .where(m.contexts.c.chunk_run_id == run["chunk_run_id"])
                    ).mappings()
                ]
                if sorted(stored_parents, key=lambda r: r["context_id"]) != sorted(
                    parents, key=lambda r: r["context_id"]
                ) or sorted(stored_children, key=lambda r: r["chunk_id"]) != sorted(
                    children, key=lambda r: r["chunk_id"]
                ):
                    raise ValueError(
                        "Bestaande fragmentverwerking gewijzigd; niet overschrijven."
                    )
                totals["already_registered"] += 1
                continue
            con.execute(m.chunk_runs.insert(), run)
            if parents:
                con.execute(m.contexts.insert(), parents)
            if children:
                con.execute(m.chunks.insert(), children)
            totals["runs_added"] += 1
            totals["contexts_added"] += len(parents)
            totals["chunks_added"] += len(children)
    return totals


def read_preparation(engine, chunk_run_id):
    """Resolve source text server-side; fail closed if stored derived spans changed."""
    with engine.connect() as con:
        run = (
            con.execute(
                sa.select(m.chunk_runs).where(
                    m.chunk_runs.c.chunk_run_id == chunk_run_id
                )
            )
            .mappings()
            .first()
        )
        if not run:
            return None
        if chunk_run_id != chunk_run_identity(
            run["processing_id"], run["config"], run["input_sha256"]
        ):
            raise ValueError("Verwerkingsconfiguratie gewijzigd.")
        info = (
            con.execute(
                sa.select(
                    m.documents.c.case_number,
                    m.versions.c.source_url,
                    m.versions.c.version_id,
                    m.versions.c.advice_date,
                )
                .select_from(m.runs.join(m.versions).join(m.documents))
                .where(m.runs.c.processing_id == run["processing_id"])
            )
            .mappings()
            .one()
        )
        sections = {
            r["section_key"]: r
            for r in con.execute(
                sa.select(m.sections).where(
                    m.sections.c.processing_id == run["processing_id"]
                )
            ).mappings()
        }
        refs = {
            r["source_id"]: r
            for r in con.execute(
                sa.select(m.citations).where(
                    m.citations.c.processing_id == run["processing_id"]
                )
            ).mappings()
        }

        def resolve(span):
            s = sections[span["section_key"]]
            a, b = span["start_offset"], span["end_offset"]
            if (
                not (0 <= a < b <= len(s["text"]))
                or span["page_number"] != s["page_number"]
            ):
                raise ValueError("Ongeldig bronspan.")
            expected = [
                key
                for key, c in refs.items()
                if c["section_key"] == s["section_key"]
                and c["start_offset"] < b
                and c["end_offset"] > a
            ]
            if not expected or set(expected) != set(span["source_ids"]):
                raise ValueError("Bronverwijzingen niet volledig.")
            for ref in span["source_ids"]:
                c = refs.get(ref)
                if (
                    not c
                    or c["section_key"] != s["section_key"]
                    or c["quote"] != s["text"][c["start_offset"] : c["end_offset"]]
                ):
                    raise ValueError("Ongeldige bronverwijzing.")
            return dict(
                **span,
                text=s["text"][a:b],
                heading=s["heading"],
                role=s["role"],
                source_url=info["source_url"] + f"#page={s['page_number']}",
            )

        contexts = []
        for parent in con.execute(
            sa.select(m.contexts)
            .where(m.contexts.c.chunk_run_id == chunk_run_id)
            .order_by(m.contexts.c.category)
        ).mappings():
            group = sorted(
                [
                    s
                    for s in sections.values()
                    if (s["category"], s["role"])
                    == (parent["category"], parent["role"])
                ],
                key=lambda s: s["ordinal"],
            )
            if (
                parent["category"] not in ("commissieoverwegingen", "conclusies")
                or parent["context_id"]
                != identity("ctx", chunk_run_id, parent["category"], parent["role"])
                or [s["section_key"] for s in parent["spans"]]
                != [s["section_key"] for s in group]
                or any(
                    s["start_offset"] != 0
                    or s["end_offset"] != len(sections[s["section_key"]]["text"])
                    for s in parent["spans"]
                )
            ):
                raise ValueError("Context wijkt af van volledige bronrol.")
            children = []
            for row in con.execute(
                sa.select(m.chunks)
                .where(m.chunks.c.context_id == parent["context_id"])
                .order_by(m.chunks.c.ordinal)
            ).mappings():
                span = resolve(row["span"])
                if (
                    span["text"] != row["text"]
                    or span["role"] != parent["role"]
                    or sections[span["section_key"]]["category"] != parent["category"]
                    or row["chunk_id"]
                    != identity(
                        "frag",
                        parent["context_id"],
                        span["section_key"],
                        span["start_offset"],
                        span["end_offset"],
                    )
                ):
                    raise ValueError("Zoektekst wijkt af van bronspan.")
                children.append(dict(row) | {"span": span})
            cursor = 0
            for section in group:
                offset = 0
                while (
                    cursor < len(children)
                    and children[cursor]["span"]["section_key"]
                    == section["section_key"]
                ):
                    span = children[cursor]["span"]
                    if span["start_offset"] != offset:
                        raise ValueError(
                            "Zoekfragmenten missen of verdubbelen brontekst."
                        )
                    offset = span["end_offset"]
                    cursor += 1
                if offset != len(section["text"]):
                    raise ValueError("Zoekfragmenten missen of verdubbelen brontekst.")
            if cursor != len(children):
                raise ValueError("Zoekfragmenten hebben een onjuiste volgorde.")
            contexts.append(
                dict(parent)
                | {"spans": [resolve(s) for s in parent["spans"]], "chunks": children}
            )
        expected_groups = {
            (s["category"], s["role"])
            for s in sections.values()
            if s["category"] in ("commissieoverwegingen", "conclusies")
        }
        if {(p["category"], p["role"]) for p in contexts} != expected_groups:
            raise ValueError("Volledige broncontext ontbreekt.")
        return (
            dict(run)
            | dict(info)
            | dict(contexts=contexts, semantic_support="not_assessed")
        )


def list_preparations(engine, processing_id):
    with engine.connect() as con:
        return [
            dict(r)
            for r in con.execute(
                sa.select(m.chunk_runs)
                .where(m.chunk_runs.c.processing_id == processing_id)
                .order_by(m.chunk_runs.c.chunk_run_id)
            ).mappings()
        ]
