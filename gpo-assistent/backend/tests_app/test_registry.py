import copy
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient
from pypdf import PdfReader, PdfWriter
from sqlalchemy.exc import IntegrityError

from gpo_app import models as m
from gpo_app.api import create_app
from gpo_app.ingestion import (
    import_prepared,
    prepare_existing,
    register_records,
    official_url,
)
from gpo_app.repository import (
    counts,
    get_citation,
    get_context,
    list_documents,
    verified_pdf,
)
from conftest import BASE, ROOT


def test_real_corpus_is_idempotent_and_all_citations_resolve(engine, prepared):
    first = import_prepared(engine, prepared)
    assert first["documents_added"] == 10 and first["versions_added"] == 10
    assert import_prepared(engine, prepared)["already_registered"] == 10
    total = counts(engine)
    assert (
        total["documents"],
        total["pdf_files"],
        total["pages"],
        total["sections"],
        total["citations"],
    ) == (10, 10, 57, 119, 501)
    for doc in prepared:
        pid = doc.processing["processing_id"]
        context = get_context(engine, pid)
        got = [s for part in context["parts"].values() for s in part]
        assert {s["section_key"]: s["text"] for s in got} == {
            s["section_key"]: s["text"] for s in doc.sections
        }
        assert all(
            s["role"] == "commissie" for s in context["parts"]["commissieoverwegingen"]
        )
        assert all(
            s["role"] != "commissie" for s in context["parts"]["partijstandpunten"]
        )
        for original in doc.citations:
            source = get_citation(engine, pid, original["source_id"])
            section = next(
                s for s in doc.sections if s["section_key"] == original["section_key"]
            )
            assert source["quote"] == original["quote"]
            assert source["page_number"] == section["page_number"]
            assert source["case_number"] == doc.document["case_number"]
            assert source["pdf_sha256"] == doc.pdf["sha256"]
            assert (
                source["source_url"]
                == doc.version["source_url"] + f"#page={section['page_number']}"
            )
            assert source["semantic_support"] == "not_assessed"


def test_date_discrepancy_is_preserved_and_publication_is_not_invented(
    engine, prepared
):
    doc = next(p for p in prepared if p.document["case_number"] == "123719")
    import_prepared(engine, [doc])
    record = list_documents(engine)[0]
    assert record["publication_date"] is None
    assert record["versions"][0]["advice_date"] == date(2025, 10, 21)
    assert record["versions"][0]["alternative_advice_date"] == date(2025, 10, 22)
    assert record["versions"][0]["date_note"]


def test_parallel_import_does_not_duplicate(engine, prepared):
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: import_prepared(engine, prepared), range(2)))
    assert sum(r["documents_added"] for r in results) == 10
    assert sum(r["already_registered"] for r in results) == 10
    assert counts(engine)["citations"] == 501


def test_changed_cache_rolls_back_whole_batch(engine, prepared):
    import_prepared(engine, [prepared[0]])
    changed = copy.deepcopy(prepared[0])
    changed.processing["input_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Cache gewijzigd"):
        import_prepared(engine, [prepared[1], changed])
    assert counts(engine)["documents"] == 1
    assert counts(engine)["extractions"] == 1


def test_same_pdf_advice_metadata_is_not_silently_overwritten(engine, prepared):
    import_prepared(engine, [prepared[0]])
    changed = copy.deepcopy(prepared[0])
    changed.version["advice_date"] = date(2000, 1, 1)
    with pytest.raises(ValueError, match="Adviesmetadata gewijzigd"):
        import_prepared(engine, [changed])
    assert (
        list_documents(engine)[0]["versions"][0]["advice_date"]
        == prepared[0].version["advice_date"]
    )


@pytest.mark.parametrize(
    "fault",
    [
        "hash",
        "missing",
        "path_escape",
        "raw_cache",
        "clean_cache",
        "role",
        "page_number",
    ],
)
def test_bad_existing_input_is_rejected_before_registration(tmp_path, prepared, fault):
    catalog = json.loads(
        (BASE.parent / "sources/catalogus.json").read_text(encoding="utf-8")
    )
    catalog["documents"] = catalog["documents"][:1]
    pages = json.loads(
        (ROOT / "onderzoek_gpo/data/pages.json").read_text(encoding="utf-8")
    )
    corpus = json.loads((BASE / "data/e5/corpus.json").read_text(encoding="utf-8"))
    case = catalog["documents"][0]["case_number"]
    page = next(p for p in pages if p["case"] == case)
    section = next(s for s in corpus["sections"] if s["case"] == case)
    if fault == "hash":
        catalog["documents"][0]["sha256"] = "0" * 64
    if fault == "missing":
        catalog["documents"][0]["file_name"] = "not-present.pdf"
    if fault == "path_escape":
        catalog["documents"][0]["file_name"] = "../pages.json"
    if fault == "raw_cache":
        page["raw"] = "Gewijzigde broncache"
    if fault == "clean_cache":
        page["text"] = "Niet zomaar de negatie verwijderen."
    if fault == "role":
        section["role"] = "school" if section["role"] != "school" else "commissie"
    if fault == "page_number":
        page["page"] = 99
    paths = []
    for name, value in [("catalog", catalog), ("pages", pages), ("corpus", corpus)]:
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(value), encoding="utf-8")
        paths.append(path)
    with pytest.raises(ValueError):
        prepare_existing(paths[0], ROOT / "onderzoek_gpo/data/pdf", paths[1], paths[2])


def test_changed_pdf_is_new_version_and_old_version_remains(engine, prepared, tmp_path):
    first = prepared[0]
    import_prepared(engine, [first])
    writer = PdfWriter()
    writer.clone_document_from_reader(PdfReader(first.pdf["local_path"]))
    writer.add_metadata(
        {"/GPOTest": "Synthetic metadata-only version; not a new official advice"}
    )
    path = tmp_path / "version.pdf"
    with path.open("wb") as stream:
        writer.write(stream)
    from gpo_local.common import digest

    # Build a real new PDF version through the same validation entry point.
    catalog = json.loads(
        (BASE.parent / "sources/catalogus.json").read_text(encoding="utf-8")
    )
    record = next(
        d
        for d in catalog["documents"]
        if d["case_number"] == first.document["case_number"]
    )
    record.update(sha256=digest(path), file_name="version.pdf")
    catalog["documents"] = [record]
    corpus = json.loads((BASE / "data/e5/corpus.json").read_text(encoding="utf-8"))
    next(d for d in corpus["documents"] if d["case_number"] == record["case_number"])[
        "sha256"
    ] = digest(path)
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps(corpus), encoding="utf-8")
    second = prepare_existing(
        catalog_path, tmp_path, ROOT / "onderzoek_gpo/data/pages.json", corpus_path
    )
    result = import_prepared(engine, second)
    assert result["documents_added"] == 0 and result["versions_added"] == 1
    assert counts(engine)["document_versions"] == 2
    assert (
        digest(verified_pdf(engine, first.version["version_id"])) == first.pdf["sha256"]
    )
    assert (
        digest(verified_pdf(engine, second[0].version["version_id"]))
        == second[0].pdf["sha256"]
    )
    assert first.citations[0]["source_id"] != second[0].citations[0]["source_id"]
    # A later changed input file cannot replace the registered snapshot.
    path.write_bytes(b"changed input file")
    assert (
        digest(verified_pdf(engine, second[0].version["version_id"]))
        == second[0].pdf["sha256"]
    )


def observed(doc, **extra):
    return dict(
        commission="GPO",
        case_number=doc.document["case_number"],
        detail_url="https://www.onderwijsgeschillen.nl/uitspraken/registry-test-"
        + doc.document["case_number"]
        + "/",
        pdf_url=doc.version["source_url"],
        **extra,
    )


@pytest.mark.parametrize("metadata_first", [True, False])
def test_detail_record_and_exact_pdf_url_share_identity(
    engine, prepared, metadata_first
):
    doc = prepared[0]
    record = observed(doc, publication_date="2026-11-01")
    if metadata_first:
        register_records(engine, [record])
    import_prepared(engine, [doc])
    register_records(engine, [record])
    assert counts(engine)["documents"] == 1
    assert counts(engine)["availability"] == {"pdf_verified": 1}
    listed = list_documents(engine)[0]
    assert listed["publication_date"] == date(2026, 11, 1)
    assert listed["versions"][0]["advice_date"] == doc.version["advice_date"]
    assert listed["versions"][0]["extractions"][0]["state"] == "validated_not_indexed"
    assert import_prepared(engine, [doc])["already_registered"] == 1


def test_link_only_and_summary_are_not_complete_pdfs(engine):
    register_records(
        engine,
        [
            dict(
                commission="GPO",
                case_number="9991",
                detail_url="https://www.onderwijsgeschillen.nl/uitspraken/test-1/",
                pdf_url="https://wordpress.onderwijsgeschillen.nl/app/uploads/test.pdf",
            ),
            dict(
                commission="GPO",
                case_number="9992",
                detail_url="https://www.onderwijsgeschillen.nl/uitspraken/test-2/",
                availability="summary_only",
            ),
        ],
    )
    total = counts(engine)
    assert total["documents"] == 2 and total["pdf_files"] == total["pages"] == 0
    assert total["availability"] == {"pdf_link_found": 1, "summary_only": 1}


def test_metadata_cannot_claim_pdf_verification_or_other_commission(engine):
    for record in [
        dict(commission="LKC"),
        dict(commission="GPO", availability="pdf_verified"),
    ]:
        with pytest.raises(ValueError):
            register_records(
                engine,
                [
                    dict(
                        case_number="999",
                        detail_url="https://www.onderwijsgeschillen.nl/uitspraken/test/",
                        **record,
                    )
                ],
            )
    assert counts(engine)["documents"] == 0


def test_duplicate_links_are_idempotent_but_conflicting_case_rolls_back(
    engine, prepared
):
    record = observed(prepared[0])
    register_records(engine, [record, record])
    bad = {**record, "case_number": "99999"}
    with pytest.raises(ValueError):
        register_records(engine, [observed(prepared[1]), bad])
    assert counts(engine)["documents"] == 1


def test_postgres_rejects_orphan_section(engine):
    with pytest.raises(IntegrityError):
        with engine.begin() as con:
            con.execute(
                sa.insert(m.sections).values(
                    processing_id="absent",
                    section_key="s",
                    page_number=1,
                    ordinal=1,
                    heading="OVERWEGINGEN",
                    role="commissie",
                    category="commissieoverwegingen",
                    text="Orphan",
                )
            )


def test_http_canonical_citation_context_and_exact_pdf(engine, prepared):
    doc = prepared[0]
    import_prepared(engine, [doc])
    with TestClient(create_app(engine)) as client:
        assert client.get("/api/v1/health").json()["counts"]["documents"] == 1
        assert client.get("/api/v1/documents?limit=0").status_code == 422
        assert client.get("/api/v1/documents?case_number=not-a-case").status_code == 422
        assert (
            client.get("/api/v1/documents?case_number=999999").json()["documents"] == []
        )
        assert (
            len(
                client.get(
                    "/api/v1/documents",
                    params={"case_number": doc.document["case_number"]},
                ).json()["documents"]
            )
            == 1
        )
        record = client.get("/api/v1/documents").json()["documents"][0]
        assert "local_path" not in json.dumps(record)
        pid = record["versions"][0]["extractions"][0]["processing_id"]
        sid = doc.citations[0]["source_id"]
        response = client.get("/api/v1/sources/" + sid, params={"processing_id": pid})
        assert (
            response.status_code == 200
            and response.json()["quote"] == doc.citations[0]["quote"]
        )
        assert (
            client.get(
                "/api/v1/sources/" + sid, params={"processing_id": "wrong"}
            ).status_code
            == 404
        )
        assert (
            client.get(
                "/api/v1/sources/unknown", params={"processing_id": pid}
            ).status_code
            == 404
        )
        context = client.get("/api/v1/extractions/" + pid + "/context").json()
        assert context["parts"]["commissieoverwegingen"]
        assert any(
            sid in section["source_ids"]
            for section in context["parts"]["commissieoverwegingen"]
            + context["parts"]["conclusies"]
        )
        pdf = client.get("/api/v1/documents/" + doc.version["version_id"] + "/pdf")
        assert pdf.status_code == 200 and pdf.content[:5] == b"%PDF-"
        assert (
            client.post("/api/v1/explanations", json={"question": "test"}).status_code
            == 404
        )


def test_modified_registered_quote_is_blocked(engine, prepared):
    doc = prepared[0]
    import_prepared(engine, [doc])
    original = doc.citations[0]
    with engine.begin() as con:
        con.execute(
            sa.update(m.citations)
            .where(
                m.citations.c.processing_id == doc.processing["processing_id"],
                m.citations.c.source_id == original["source_id"],
            )
            .values(quote="X" * len(original["quote"]))
        )
    with TestClient(create_app(engine)) as client:
        assert (
            client.get(
                "/api/v1/sources/" + original["source_id"],
                params={"processing_id": doc.processing["processing_id"]},
            ).status_code
            == 409
        )


def test_missing_original_pdf_is_not_replaced_by_other_version(
    engine, prepared, tmp_path
):
    doc = prepared[0]
    import_prepared(engine, [doc])
    with engine.begin() as con:
        con.execute(
            sa.update(m.files)
            .where(m.files.c.sha256 == doc.pdf["sha256"])
            .values(local_path=str(tmp_path / "missing.pdf"))
        )
    with TestClient(create_app(engine)) as client:
        assert (
            client.get(
                "/api/v1/documents/" + doc.version["version_id"] + "/pdf"
            ).status_code
            == 409
        )


def test_corrupt_managed_snapshot_is_not_silently_overwritten(engine, prepared):
    doc = prepared[0]
    import_prepared(engine, [doc])
    path = verified_pdf(engine, doc.version["version_id"])
    path.write_bytes(b"corrupt snapshot")
    with pytest.raises(ValueError, match="snapshot is gewijzigd"):
        import_prepared(engine, [doc])
    assert path.read_bytes() == b"corrupt snapshot"


@pytest.mark.parametrize(
    "url",
    [
        "http://www.onderwijsgeschillen.nl/x",
        "https://example.org/advice.pdf",
        "https://user:secret@www.onderwijsgeschillen.nl/x",
    ],
)
def test_registry_rejects_nonofficial_or_credential_urls(url):
    with pytest.raises(ValueError, match="officiële HTTPS"):
        official_url(url)
