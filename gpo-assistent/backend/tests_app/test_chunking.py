from collections import defaultdict

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from gpo_app import models as m
from gpo_app.api import create_app
from gpo_app.chunk_audit import audit
from gpo_app.chunking import (
    LocalTokenCounter,
    build,
    chunk_run_identity,
    paragraph_cuts,
    prepare_chunks,
    read_preparation,
    split_span,
)
from gpo_app.ingestion import import_prepared
from conftest import BASE


@pytest.fixture(scope="module")
def counter():
    return LocalTokenCounter(BASE / "data/e5/model/tokenizer.json")


def test_json_object_key_order_does_not_change_processing_identity(prepared, counter):
    doc = prepared[0]
    run, _, _ = build(
        doc.processing["processing_id"],
        doc.sections,
        doc.pages,
        doc.citations,
        counter,
        counter.sha256,
    )
    reversed_config = dict(reversed(list(run["config"].items())))
    assert (
        chunk_run_identity(run["processing_id"], reversed_config, run["input_sha256"])
        == run["chunk_run_id"]
    )


def test_real_sources_have_exact_coverage_token_limits_and_stable_ids(
    prepared, counter
):
    for doc in prepared:
        args = (
            doc.processing["processing_id"],
            doc.sections,
            doc.pages,
            doc.citations,
            counter,
            counter.sha256,
        )
        first = build(*args)
        assert first == build(*args)
        run, parents, children = first
        assert run["config"]["token_limit"] == 384
        assert all(
            p["role"] == "commissie" and "uncertain" in p["boundary_status"]
            for p in parents
        )
        by_section = defaultdict(list)
        for c in children:
            s = c["span"]
            by_section[s["section_key"]].append(c)
            assert c["token_count"] == counter(c["text"]) <= 384
            assert c["span"]["source_ids"]
        for section in doc.sections:
            if section["category"] not in ("commissieoverwegingen", "conclusies"):
                assert section["section_key"] not in by_section
                continue
            parts = by_section[section["section_key"]]
            assert "".join(p["text"] for p in parts) == section["text"]
            assert parts[0]["span"]["start_offset"] == 0
            assert parts[-1]["span"]["end_offset"] == len(section["text"])
            assert all(
                a["span"]["end_offset"] == b["span"]["start_offset"]
                for a, b in zip(parts, parts[1:])
            )


def test_page_boundary_and_conclusions_remain_separate(prepared, counter):
    doc = next(d for d in prepared if d.document["case_number"] == "109206")
    _, parents, children = build(
        doc.processing["processing_id"],
        doc.sections,
        doc.pages,
        doc.citations,
        counter,
        counter.sha256,
    )
    parent = next(p for p in parents if p["category"] == "commissieoverwegingen")
    assert {s["page_number"] for s in parent["spans"]} == {4, 5}
    section_map = {s["section_key"]: s for s in doc.sections}
    text = " ".join(section_map[s["section_key"]]["text"] for s in parent["spans"])
    assert "Het is de Commissie gebleken" in text
    assert any(
        p["category"] == "conclusies" and p["context_id"] != parent["context_id"]
        for p in parents
    )
    assert all(
        section_map[s["section_key"]]["category"] == "commissieoverwegingen"
        for s in parent["spans"]
    )


def test_blank_lines_are_hints_and_uncertain_alignment_falls_back():
    section = {"text": "Eerste argument. Tweede argument."}
    cuts, status = paragraph_cuts(
        section, {"cleaned_text": "Kop\nEerste argument.\n\nTweede argument."}
    )
    assert cuts == [0, 16, len(section["text"])] and status == "layout_only"
    assert paragraph_cuts(section, {"cleaned_text": "andere tekst"}) == (
        [0, len(section["text"])],
        "section_alignment_uncertain",
    )


@pytest.mark.parametrize(
    "text",
    [
        "zonder spaties" * 100,
        "Niet toegestaan, tenzij aan alle voorwaarden is voldaan. " * 80,
    ],
)
def test_oversized_passage_never_truncates(text, counter):
    parts = list(split_span(text, 0, len(text), counter, 32))
    assert "".join(text[a:b] for a, b, _ in parts) == text
    assert all(counter(text[a:b]) <= 32 for a, b, _ in parts)


def test_registry_persists_idempotently_versions_config_and_serves_sources(
    engine, prepared, counter
):
    import_prepared(engine, prepared)
    before = None
    with engine.connect() as con:
        before = list(
            con.execute(sa.select(m.citations).order_by(m.citations.c.source_id))
        )
    result = prepare_chunks(engine, counter)
    assert (
        result["runs_added"] == 10
        and result["contexts_added"] == 20
        and result["chunks_added"] > 20
    )
    assert prepare_chunks(engine, counter)["already_registered"] == 10
    report = audit(engine)
    assert report["runs"] == 10 and report["chunks"] == result["chunks_added"]
    assert report["semantic_argument_boundaries"] == "not_validated"
    with engine.connect() as con:
        assert (
            list(con.execute(sa.select(m.citations).order_by(m.citations.c.source_id)))
            == before
        )
        ids = list(con.scalars(sa.select(m.chunk_runs.c.chunk_run_id)))
    with TestClient(create_app(engine)) as client:
        assert client.get("/api/v1/chunk-preparations/missing").status_code == 404
        for run_id in ids:
            response = client.get("/api/v1/chunk-preparations/" + run_id)
            assert response.status_code == 200, response.text
            result = response.json()
            assert result["semantic_support"] == "not_assessed"
            listing = client.get(
                "/api/v1/extractions/" + result["processing_id"] + "/chunk-preparations"
            )
            assert (
                listing.status_code == 200
                and listing.json()[0]["chunk_run_id"] == run_id
            )
            for parent in result["contexts"]:
                for chunk in parent["chunks"]:
                    span = chunk["span"]
                    assert span["source_url"].endswith(
                        "#page=" + str(span["page_number"])
                    )
                    for source_id in span["source_ids"]:
                        ref = client.get(
                            "/api/v1/sources/" + source_id,
                            params={"processing_id": result["processing_id"]},
                        )
                        assert (
                            ref.status_code == 200
                            and ref.json()["case_number"] == result["case_number"]
                        )
    assert prepare_chunks(engine, counter, 320)["runs_added"] == 10
    with engine.connect() as con:
        assert con.scalar(sa.select(sa.func.count()).select_from(m.chunk_runs)) == 20


def test_changed_derived_text_is_rejected(engine, prepared, counter):
    import_prepared(engine, [prepared[0]])
    prepare_chunks(engine, counter)
    with engine.begin() as con:
        run_id = con.scalar(sa.select(m.chunk_runs.c.chunk_run_id))
        chunk_id = con.scalar(sa.select(m.chunks.c.chunk_id))
        con.execute(
            m.chunks.update()
            .where(m.chunks.c.chunk_id == chunk_id)
            .values(text="ander citaat")
        )
    with pytest.raises(ValueError, match="Zoektekst"):
        read_preparation(engine, run_id)
    with TestClient(create_app(engine)) as client:
        assert client.get("/api/v1/chunk-preparations/" + run_id).status_code == 409
    with pytest.raises(ValueError, match="niet overschrijven"):
        prepare_chunks(engine, counter)


@pytest.mark.parametrize(
    "change",
    [
        "unknown_source",
        "wrong_page",
        "missing_chunk",
        "wrong_role",
        "missing_context",
        "changed_config",
    ],
)
def test_invalid_provenance_or_incomplete_coverage_fails_closed(
    engine, prepared, counter, change
):
    import_prepared(engine, [prepared[0]])
    prepare_chunks(engine, counter)
    with engine.begin() as con:
        run_id = con.scalar(sa.select(m.chunk_runs.c.chunk_run_id))
        chunk = dict(con.execute(sa.select(m.chunks)).mappings().first())
        if change == "changed_config":
            con.execute(
                m.chunk_runs.update()
                .where(m.chunk_runs.c.chunk_run_id == run_id)
                .values(config={"token_limit": 999})
            )
        elif change == "missing_context":
            con.execute(
                m.chunks.delete().where(m.chunks.c.context_id == chunk["context_id"])
            )
            con.execute(
                m.contexts.delete().where(
                    m.contexts.c.context_id == chunk["context_id"]
                )
            )
        elif change == "wrong_role":
            con.execute(
                m.contexts.update()
                .where(m.contexts.c.context_id == chunk["context_id"])
                .values(role="ouders")
            )
        elif change == "missing_chunk":
            con.execute(
                m.chunks.delete().where(m.chunks.c.chunk_id == chunk["chunk_id"])
            )
        else:
            span = dict(chunk["span"])
            if change == "unknown_source":
                span["source_ids"] = ["src_bestaat_niet"]
            else:
                span["page_number"] = 999
            con.execute(
                m.chunks.update()
                .where(m.chunks.c.chunk_id == chunk["chunk_id"])
                .values(span=span)
            )
    with pytest.raises(ValueError):
        read_preparation(engine, run_id)
