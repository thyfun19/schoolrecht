"""Find additional cases in the complete registered committee reasoning corpus."""

from array import array
from collections import Counter, defaultdict
from math import log
from threading import Lock

import sqlalchemy as sa

from . import models as m
from .card_search import query_terms, terms


class CitationIndex:
    def __init__(self, rows):
        self.postings = defaultdict(lambda: array("I"))
        self.owners = []
        case_terms = defaultdict(set)
        for pid, quote in rows:
            words = terms(quote)
            if not words or len(quote.split()) < 6:
                continue
            ordinal = len(self.owners)
            self.owners.append(pid)
            case_terms[pid].update(words)
            for word in words:
                self.postings[word].append(ordinal)
        frequencies = Counter(word for words in case_terms.values() for word in words)
        self.weights = {
            word: log(1 + (len(case_terms) + 0.5) / (frequency + 0.5))
            for word, frequency in frequencies.items()
        }

    def ranking(self, question, eligible, limit):
        scores = defaultdict(float)
        for term in sorted(query_terms(question)):
            if term not in self.weights:
                continue
            weight = self.weights[term] * (4 if term.startswith("topic:") else 1)
            for ordinal in self.postings[term]:
                scores[ordinal] += weight
        best = {}
        for ordinal, score in scores.items():
            pid = self.owners[ordinal]
            if pid in eligible:
                best[pid] = max(best.get(pid, 0), score)
        return sorted(best.items(), key=lambda item: (-item[1], item[0]))[:limit]


class RegisteredCitationSearch:
    def __init__(self):
        self.lock = Lock()
        self.key = self.index = None

    def candidates(self, engine, question, limit, topic, sector):
        with engine.connect() as con:
            docs = (
                con.execute(
                    sa.select(
                        m.runs.c.processing_id,
                        m.versions.c.version_id,
                        m.documents.c.case_number,
                        m.documents.c.topic,
                        m.documents.c.sector,
                    )
                    .select_from(m.runs.join(m.versions).join(m.documents))
                    .where(m.runs.c.state == "validated_not_indexed")
                )
                .mappings()
                .all()
            )
            # Registered extraction IDs identify immutable text, sections and citations.
            key = (engine, frozenset(row["processing_id"] for row in docs))
            with self.lock:
                if self.key != key:
                    rows = con.execute(
                        sa.select(m.citations.c.processing_id, m.citations.c.quote)
                        .select_from(m.citations.join(m.sections))
                        .where(
                            m.sections.c.role == "commissie",
                            m.sections.c.category == "commissieoverwegingen",
                            m.citations.c.processing_id.in_(key[1]),
                        )
                    )
                    self.index = CitationIndex(rows)
                    self.key = key
                index = self.index
        eligible = {
            r["processing_id"]: r
            for r in docs
            if (not topic or topic.casefold() in (r["topic"] or "").casefold())
            and (not sector or sector.casefold() in (r["sector"] or "").casefold())
        }
        return [
            dict(eligible[pid], lexical_score=score)
            for pid, score in index.ranking(question, eligible, limit)
        ]
