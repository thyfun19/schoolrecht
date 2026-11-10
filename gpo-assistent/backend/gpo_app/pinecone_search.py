"""Live Pinecone + Supabase PostgreSQL semantic search using local BGE-M3 embeddings."""

from functools import lru_cache
import os
from pathlib import Path
import threading

import httpx
import sqlalchemy as sa

from . import models as m
from .citation_search import RegisteredCitationSearch

BASE = Path(__file__).resolve().parents[1]


def _read_dotenv() -> dict[str, str]:
    env_path = BASE / ".env"
    if not env_path.exists():
        return {}
    result = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        result[k.strip()] = v.strip()
    return result


def pinecone_settings() -> tuple[str | None, str]:
    env = _read_dotenv()
    api_key = os.environ.get("PINECONE_API_KEY") or env.get("PINECONE_API_KEY")
    index_name = (
        os.environ.get("PINECONE_INDEX_NAME")
        or env.get("PINECONE_INDEX_NAME")
        or "gpo-uitspraken"
    )
    return (api_key.strip() if api_key else None), index_name.strip()


def pinecone_configured() -> bool:
    key, _ = pinecone_settings()
    return bool(key)


_RAW_VERDICTS_MAP = None


def get_raw_verdicts_map() -> dict[str, dict]:
    global _RAW_VERDICTS_MAP
    if _RAW_VERDICTS_MAP is None:
        _RAW_VERDICTS_MAP = {}
        path = BASE / "data/full_gpo/verdicts_raw.json"
        if path.exists():
            import json, re, html

            try:
                items = json.load(open(path, encoding="utf-8"))
                for it in items:
                    pn = it.get("overview", {}).get("publicationNumber")
                    if pn:
                        t = html.unescape(
                            it.get("title", {}).get("rendered", "")
                        ).strip()
                        c = it.get("content", {}).get("rendered", "")
                        m_sit = re.search(
                            r"Situatie.*?<\/p>\s*<p>(.*?)<\/p>",
                            c,
                            re.DOTALL | re.IGNORECASE,
                        )
                        sit = (
                            html.unescape(
                                re.sub(r"<[^>]+>", "", m_sit.group(1))
                            ).strip()
                            if m_sit
                            else ""
                        )
                        _RAW_VERDICTS_MAP[str(pn)] = {"title": t, "summary": sit}
            except Exception:
                pass
    return _RAW_VERDICTS_MAP


def fuse_case_matches(text_matches, vector_matches):
    """Fuse case ranks, so long opinions get only one vote per search method."""
    scores = {}
    passages = {}
    for matches in (text_matches, vector_matches):
        seen = set()
        for match in matches:
            meta = match.get("metadata") or {}
            case = meta.get("case") or meta.get("case_number")
            if not case or not meta.get("processing_id"):
                continue
            if case not in seen:
                seen.add(case)
                scores[case] = scores.get(case, 0) + 1 / (60 + len(seen))
            passages.setdefault(case, {}).setdefault(match["id"], match)
    result = []
    for case in sorted(scores, key=lambda c: -scores[c]):
        for match in passages[case].values():
            result.append(dict(match, case_score=scores[case]))
    return result


def registry_text_matches(engine, question, top_k, topic, sector):
    """Rank registered passages using all Dutch lexemes from the question."""
    statement = sa.text("""
        WITH terms AS (
            SELECT unnest(tsvector_to_array(to_tsvector('dutch', :question))) AS term
        ), candidates AS MATERIALIZED (
            SELECT c.chunk_id, c.span, d.case_number, v.version_id, r.processing_id,
                   to_tsvector('dutch', c.text) AS tokens
            FROM gpo_search_chunks c
            JOIN gpo_search_contexts ctx USING (context_id)
            JOIN gpo_chunk_runs cr USING (chunk_run_id)
            JOIN gpo_extractions r USING (processing_id)
            JOIN gpo_document_versions v USING (version_id)
            JOIN gpo_documents d USING (document_id)
            WHERE ctx.category IN ('commissieoverwegingen','conclusies')
              AND r.state = 'validated_not_indexed'
              AND (CAST(:topic AS text) IS NULL OR strpos(lower(d.topic),lower(:topic)) > 0)
              AND (CAST(:sector AS text) IS NULL OR strpos(lower(d.sector),lower(:sector)) > 0)
              AND to_tsvector('dutch', c.text) @@
                  (SELECT to_tsquery('simple', string_agg(quote_literal(term), ' | ')) FROM terms)
        ), hits AS (
            SELECT c.chunk_id, t.term,
                   ts_rank(c.tokens, to_tsquery('simple', quote_literal(t.term)), 1) AS weight,
                   count(*) OVER (PARTITION BY t.term) AS frequency
            FROM candidates c JOIN terms t ON c.tokens @@ to_tsquery('simple', quote_literal(t.term))
        ), scores AS (
            SELECT chunk_id, sum(weight * ln(1 + (SELECT count(*) FROM candidates)::float / frequency)) AS score
            FROM hits GROUP BY chunk_id
        )
        SELECT c.chunk_id, c.span, c.case_number, c.version_id, c.processing_id, s.score
        FROM candidates c JOIN scores s USING (chunk_id)
        ORDER BY s.score DESC, c.chunk_id
        LIMIT :top_k
    """)
    with engine.connect() as con:
        rows = con.execute(
            statement, dict(question=question, top_k=top_k, topic=topic, sector=sector)
        )
        return [
            dict(
                id=row.chunk_id,
                score=row.score,
                metadata=dict(
                    case=row.case_number,
                    processing_id=row.processing_id,
                    version_id=row.version_id,
                    page=row.span["page_number"],
                    section=row.span.get("heading"),
                    role=row.span.get("role"),
                ),
            )
            for row in rows
        ]


registered_citation_search = RegisteredCitationSearch()


class PineconeSearch:
    def __init__(self):
        self.index_name = os.environ.get("PINECONE_INDEX_NAME", "gpo-uitspraken")
        self.lock = threading.Lock()
        self.encoder = None
        model_path = BASE / "data/model-comparison/models/bge"
        if (model_path / "model.onnx").exists():
            try:
                from .model_comparison import Encoder

                self.encoder = Encoder("bge", model_path)
            except Exception:
                self.encoder = None

    @lru_cache(maxsize=1)
    def connection(self):
        key, self.index_name = pinecone_settings()
        if not key:
            raise ValueError("Pinecone is niet geconfigureerd.")
        headers = {"Api-Key": key, "X-Pinecone-API-Version": "2024-07"}
        info = httpx.get(
            f"https://api.pinecone.io/indexes/{self.index_name}",
            headers=headers,
            timeout=15,
        )
        info.raise_for_status()
        return info.json()["host"], headers

    def query_vectors(self, question, top_k):
        if self.encoder is None:
            from .pinned_vectors import get_query_vector

            vector = get_query_vector(question)
        else:
            with self.lock:
                vectors, _ = self.encoder.encode([question], "query", batch_size=1)
            vector = vectors[0].tolist()
        host, headers = self.connection()
        resp = httpx.post(
            f"https://{host}/query",
            headers=headers,
            json={
                "vector": vector,
                "topK": top_k,
                "includeMetadata": True,
            },
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json().get("matches", [])

    def search(
        self,
        engine,
        question: str,
        limit: int = 5,
        topic: str | None = None,
        sector: str | None = None,
    ) -> dict:
        if not 1 <= limit <= 50:
            raise ValueError("Ongeldige kandidaatlimiet.")
        question = question.strip()
        if not 3 <= len(question) <= 2000:
            raise ValueError("Ongeldige vraaglengte.")
        candidate_limit = max(20, limit)
        top_k = candidate_limit * 10
        # Keep the measured baseline on existing deployments until the hybrid
        # candidate also passes a full-corpus comparison.
        hybrid_enabled = os.environ.get("GPO_HYBRID_SEARCH") == "1"
        if self.encoder is None and (hybrid_enabled or not pinecone_configured()):
            method = "postgres_text"
            matches = registry_text_matches(engine, question, top_k, topic, sector)
            if pinecone_configured() and matches:
                try:
                    related = self.query_vectors(question, top_k)
                except httpx.HTTPError:
                    related = []
                if related:
                    matches = fuse_case_matches(matches, related)
                    method = "hybrid_text_topic"
        else:
            method = "pinecone_bge" if self.encoder is not None else "pinecone_topic"
            matches = self.query_vectors(question, top_k)
        scope = (
            "GPO-bronregister"
            if method == "postgres_text"
            else "Pinecone BGE-M3 + GPO-bronregister"
        )
        by_case: dict[str, dict] = {}
        chunk_ids: list[str] = []
        for match in matches:
            meta = match.get("metadata") or {}
            case = str(meta.get("case") or meta.get("case_number") or "").strip()
            pid = meta.get("processing_id")
            if not case or not pid:
                continue
            chunk_ids.append(match["id"])
            if case not in by_case:
                by_case[case] = {
                    "case_number": case,
                    "processing_id": pid,
                    "version_id": meta.get("version_id"),
                    "score": round(
                        float(match.get("case_score", match.get("score", 0.0))), 6
                    ),
                    "matched_chunks": [],
                }
            if pid != by_case[case]["processing_id"]:
                continue
            if len(by_case[case]["matched_chunks"]) < 3:
                by_case[case]["matched_chunks"].append(
                    {
                        "chunk_id": match["id"],
                        "score": round(float(match.get("score", 0.0)), 4),
                        "page": meta.get("page"),
                        "section": meta.get("section"),
                        "role": meta.get("role"),
                    }
                )

        # Keep independent places for additional cases before selecting the cards.
        # An existing case keeps its original extraction and vector matches.
        base_cases = (
            dict(list(by_case.items())[:candidate_limit])
            if not (topic or sector)
            else by_case
        )
        extras = registered_citation_search.candidates(
            engine, question, candidate_limit, topic, sector
        )
        extra_cases = {r["case_number"] for r in extras}
        for row in extras:
            case = row["case_number"]
            base_cases.setdefault(
                case,
                by_case.get(case)
                or dict(
                    case_number=case,
                    processing_id=row["processing_id"],
                    version_id=row["version_id"],
                    score=row["lexical_score"],
                    retrieval_source="registered_citation",
                    matched_chunks=[],
                ),
            )
        by_case = base_cases

        if not by_case:
            return {
                "method": method,
                "index_name": self.index_name,
                "cases": [],
                "processing_ids": [],
                "results": [],
                "score_is_answerability": False,
                "answer_generated": False,
                "paid_calls": 0,
                "scope": scope,
            }

        with engine.connect() as con:
            doc_rows = (
                con.execute(
                    sa.select(
                        m.documents.c.case_number,
                        m.documents.c.topic,
                        m.documents.c.sector,
                        m.documents.c.publication_date,
                        m.documents.c.detail_url,
                        m.versions.c.version_id,
                        m.versions.c.source_url,
                        m.versions.c.advice_date,
                        m.runs.c.processing_id,
                    )
                    .select_from(m.runs.join(m.versions).join(m.documents))
                    .where(
                        m.documents.c.case_number.in_(list(by_case.keys())),
                        m.runs.c.state == "validated_not_indexed",
                    )
                )
                .mappings()
                .all()
            )
            doc_map = {
                (r["case_number"], r["processing_id"]): dict(r) for r in doc_rows
            }

            chunk_text_map = {}
            if chunk_ids:
                c_rows = (
                    con.execute(
                        sa.select(
                            m.chunks.c.chunk_id, m.chunks.c.text, m.chunks.c.span
                        ).where(m.chunks.c.chunk_id.in_(chunk_ids))
                    )
                    .mappings()
                    .all()
                )
                chunk_text_map = {r["chunk_id"]: dict(r) for r in c_rows}

            filtered_cases = []
            for case, entry in by_case.items():
                d = doc_map.get((case, entry["processing_id"]))
                if not d or (
                    entry["version_id"] and entry["version_id"] != d["version_id"]
                ):
                    continue
                if topic and topic.lower() not in (d.get("topic") or "").lower():
                    continue
                if sector and sector.lower() not in (d.get("sector") or "").lower():
                    continue
                entry["processing_id"] = d["processing_id"]
                entry["version_id"] = d["version_id"]
                entry["topic"] = d.get("topic")
                entry["sector"] = d.get("sector")
                entry["publication_date"] = (
                    str(d["publication_date"]) if d.get("publication_date") else None
                )
                entry["advice_date"] = (
                    str(d["advice_date"]) if d.get("advice_date") else None
                )
                entry["detail_url"] = d.get("detail_url")
                entry["source_url"] = d.get("source_url")
                raw_info = get_raw_verdicts_map().get(case) or {}
                entry["title"] = (
                    raw_info.get("title")
                    or f"Zaak {case}: {d.get('topic') or 'Geschil passend onderwijs'}"
                )
                entry["summary"] = raw_info.get("summary")
                for mc in entry["matched_chunks"]:
                    c_info = chunk_text_map.get(mc["chunk_id"])
                    if c_info:
                        mc["text"] = c_info["text"]
                        mc["source_ids"] = (c_info.get("span") or {}).get(
                            "source_ids", []
                        )
                if len(filtered_cases) < candidate_limit or case in extra_cases:
                    filtered_cases.append(entry)

            from .card_search import select_cards

            filtered_cases = select_cards(con, question, filtered_cases, limit)
            selected_pids = [c["processing_id"] for c in filtered_cases]
            if selected_pids:
                sec_rows = (
                    con.execute(
                        sa.select(
                            m.sections.c.processing_id,
                            m.sections.c.section_key,
                            m.sections.c.page_number,
                            m.sections.c.heading,
                            m.sections.c.role,
                            m.sections.c.category,
                            m.sections.c.text,
                        )
                        .where(
                            m.sections.c.processing_id.in_(selected_pids),
                            m.sections.c.category.in_(
                                ["commissieoverwegingen", "conclusies"]
                            ),
                        )
                        .order_by(m.sections.c.ordinal)
                    )
                    .mappings()
                    .all()
                )
                secs_by_pid: dict[str, list[dict]] = {}
                for s in sec_rows:
                    secs_by_pid.setdefault(s["processing_id"], []).append(dict(s))
                for entry in filtered_cases:
                    secs = secs_by_pid.get(entry["processing_id"], [])
                    entry["conclusies"] = [
                        s for s in secs if s["category"] == "conclusies"
                    ]
                    entry["commissieoverwegingen"] = [
                        s for s in secs if s["category"] == "commissieoverwegingen"
                    ]

        return {
            "method": method,
            "index_name": self.index_name,
            "cases": [c["case_number"] for c in filtered_cases],
            "processing_ids": [c["processing_id"] for c in filtered_cases],
            "results": filtered_cases,
            "score_is_answerability": False,
            "answer_generated": False,
            "paid_calls": 0,
            "scope": scope,
        }


@lru_cache(maxsize=1)
def pinecone_search() -> PineconeSearch:
    return PineconeSearch()
