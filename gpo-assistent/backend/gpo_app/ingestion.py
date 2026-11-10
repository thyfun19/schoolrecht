"""Connect existing PDFs and audited caches atomically. No downloading or re-embedding."""

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import pypdf
import sqlalchemy as sa
from pypdf import PdfReader
from sqlalchemy.dialects.postgresql import insert

from gpo_local.common import digest, norm, read, sections as legacy_sections
from gpo_local.source_registry import SourceRegistry
from . import models as m
from .storage import snapshot_pdf

PIPELINE = "audited-legacy-sections-v1"


def identity(prefix, *parts):
    value = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode()
    return prefix + "_" + hashlib.sha256(value).hexdigest()[:32]


def json_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def official_url(value):
    parsed = urlparse(value)
    if (
        parsed.scheme != "https"
        or parsed.username
        or parsed.password
        or parsed.hostname
        not in {
            "www.onderwijsgeschillen.nl",
            "onderwijsgeschillen.nl",
            "wordpress.onderwijsgeschillen.nl",
        }
    ):
        raise ValueError("Bronlink moet naar een officiële HTTPS-bron verwijzen.")
    return parsed._replace(fragment="").geturl()


def date_value(value):
    return date.fromisoformat(value) if value else None


def clean_legacy(raw):
    # Same versioned policy as the existing audited extraction, not new chunking.
    value = re.sub(
        r"(?im)^.*(?:pagina \d+ van \d+|advies d\.d\.|zaak \d+ \| d\.d\.).*$", "", raw
    )
    value = re.sub(
        r"(?im)^\s*(?:Geschillencommissie(?: passend)?|passend onderwijs|onderwijs|Stichting Onderwijsgeschillen)\s*$",
        "",
        value,
    )
    return value.strip()


def category(section):
    if section["role"] == "commissie":
        if "OVERWEGINGEN" in section["section"]:
            return "commissieoverwegingen"
        if section["section"] in ["ADVIES", "OORDEEL", "AANBEVELINGEN"]:
            return "conclusies"
    if section["role"] == "feiten":
        return "feiten"
    if section["role"] in ["school", "ouders", "leerling", "samenwerkingsverband"]:
        return "partijstandpunten"
    return "overige"


@dataclass
class PreparedDocument:
    document: dict
    pdf: dict
    version: dict
    processing: dict
    pages: list
    sections: list
    citations: list


def prepare_existing(catalog_path, pdf_root, pages_path, corpus_path):
    catalog = read(catalog_path)["documents"]
    cached_pages = read(pages_path)
    corpus = read(corpus_path)
    corpus_docs = {d["case_number"]: d for d in corpus["documents"]}
    root = Path(pdf_root).resolve()
    prepared = []
    seen = set()
    for doc in catalog:
        case = doc["case_number"]
        if not re.fullmatch(r"\d+", case):
            raise ValueError("Ongeldig zaaknummer in broncatalogus.")
        if doc.get("commission", "GPO") != "GPO":
            raise ValueError("Geen GPO-document.")
        source = official_url(doc["source_url"])
        key = doc.get("document_key") or doc.get("detail_url") or source
        if key in seen:
            raise ValueError("Dubbele documentidentiteit in dezelfde import.")
        seen.add(key)
        path = (root / doc.get("file_name", case + ".pdf")).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"PDF ontbreekt of ligt buiten bronmap: {case}.")
        pdf_hash = digest(path)
        if pdf_hash != doc["sha256"]:
            raise ValueError(f"PDF-hash verschilt van catalogus: {case}.")
        if corpus_docs.get(case, {}).get("sha256") != pdf_hash:
            raise ValueError(f"Corpus hoort niet bij deze PDF-versie: {case}.")
        if path.read_bytes()[:5] != b"%PDF-":
            raise ValueError(f"Bestand is geen PDF: {case}.")
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise ValueError(f"Versleutelde PDF vereist controle: {case}.")
        page_data = [p for p in cached_pages if p["case"] == case]
        if len(reader.pages) != doc["pages"] or [p["page"] for p in page_data] != list(
            range(1, doc["pages"] + 1)
        ):
            raise ValueError(
                f"Paginaregistratie niet volledig of verkeerd geordend: {case}."
            )
        for physical, stored in zip(reader.pages, page_data):
            raw = physical.extract_text() or ""
            if not raw.strip():
                raise ValueError(
                    f"Lege tekstlaag vereist controle: {case}, p.{stored['page']}."
                )
            # One original includes a blank closing page with only a footer; the raw layer still exists.
            if raw != stored["raw"] or clean_legacy(raw) != stored["text"]:
                raise ValueError(
                    f"Opgeslagen tekst hoort niet bij PDF/extractiebeleid: {case}, p.{stored['page']}."
                )
        expected = legacy_sections(page_data)
        registered = [s for s in corpus["sections"] if s["case"] == case]
        fields = ["case", "page", "section", "role", "section_id", "text"]
        if [{k: s[k] for k in fields} for s in expected] != [
            {k: s[k] for k in fields} for s in registered
        ]:
            raise ValueError(
                f"Secties of bronrollen wijken af van gecontroleerde cache: {case}."
            )
        if not any(category(s) == "commissieoverwegingen" for s in registered):
            raise ValueError(f"Geen volledige commissieoverwegingen: {case}.")
        document_id = identity("doc", key)
        version_id = identity("ver", document_id, pdf_hash)
        processing_id = identity("run", version_id, PIPELINE)
        source_registry = SourceRegistry({case: doc}, registered)
        cite_data = source_registry.select([case])
        lookup = {s["section_id"]: s for s in registered}
        for citation in cite_data:
            s = lookup[citation["section_id"]]
            if (
                s["text"][citation["start_offset"] : citation["end_offset"]]
                != citation["quote"]
            ):
                raise ValueError("Citaat komt niet overeen met geregistreerde bron.")
        prepared.append(
            PreparedDocument(
                document=dict(
                    document_id=document_id,
                    source_key=key,
                    commission="GPO",
                    case_number=case,
                    topic=doc.get("topic"),
                    sector=doc.get("sector"),
                    detail_url=doc.get("detail_url"),
                    publication_date=date_value(doc.get("publication_date")),
                    availability="pdf_verified",
                ),
                pdf=dict(
                    sha256=pdf_hash,
                    local_path=str(path),
                    byte_size=path.stat().st_size,
                    page_count=len(reader.pages),
                ),
                version=dict(
                    version_id=version_id,
                    document_id=document_id,
                    pdf_sha256=pdf_hash,
                    advice_date=date_value(doc.get("advice_date")),
                    alternative_advice_date=date_value(
                        doc.get("alternative_advice_date")
                    ),
                    date_note=doc.get("date_note"),
                    source_url=source,
                    retrieved_on=date_value(doc.get("retrieved")),
                ),
                processing=dict(
                    processing_id=processing_id,
                    version_id=version_id,
                    pipeline_version=PIPELINE,
                    input_sha256=json_hash(
                        dict(
                            pages=page_data,
                            sections=[{k: s[k] for k in fields} for s in registered],
                        )
                    ),
                    state="validated_not_indexed",
                    provenance=dict(
                        pypdf_version=pypdf.__version__,
                        cache_verified_against_pdf=True,
                        new_argument_segmentation=False,
                        catalog_sha256=digest(catalog_path),
                        pages_file_sha256=digest(pages_path),
                        corpus_sha256=digest(corpus_path),
                    ),
                ),
                pages=[
                    dict(
                        processing_id=processing_id,
                        page_number=p["page"],
                        raw_text=p["raw"],
                        cleaned_text=p["text"],
                        raw_sha256=hashlib.sha256(p["raw"].encode()).hexdigest(),
                    )
                    for p in page_data
                ],
                sections=[
                    dict(
                        processing_id=processing_id,
                        section_key=s["section_id"],
                        page_number=s["page"],
                        ordinal=i,
                        heading=s["section"],
                        role=s["role"],
                        category=category(s),
                        text=s["text"],
                    )
                    for i, s in enumerate(registered)
                ],
                citations=[
                    dict(
                        processing_id=processing_id,
                        source_id=c["source_id"],
                        section_key=c["section_id"],
                        start_offset=c["start_offset"],
                        end_offset=c["end_offset"],
                        quote=c["quote"],
                    )
                    for c in cite_data
                ],
            )
        )
    if not prepared:
        raise ValueError("Broncatalogus bevat geen documenten.")
    return prepared


def lock_sources(con, keys):
    for key in sorted(set(keys)):
        lock = int(hashlib.sha256(key.encode()).hexdigest()[:15], 16)
        con.execute(sa.text("SELECT pg_advisory_xact_lock(:key)"), {"key": lock})


def resolve_source(con, key, urls):
    linked = sa.select(m.links.c.document_id).where(
        m.links.c.url.in_([u for u in urls if u])
    )
    matches = (
        con.execute(
            sa.select(m.documents).where(
                sa.or_(
                    m.documents.c.source_key == key,
                    m.documents.c.document_id.in_(linked),
                )
            )
        )
        .mappings()
        .all()
    )
    if len(matches) > 1:
        raise ValueError(
            "Bronlinks verwijzen naar meerdere documenten; handmatige controle nodig."
        )
    return matches[0] if matches else None


def rebind(item, document):
    """A detail-page record and an existing PDF URL share one stable registry identity."""
    value = deepcopy(item)
    value.document.update(
        document_id=document["document_id"], source_key=document["source_key"]
    )
    version_id = identity("ver", document["document_id"], value.pdf["sha256"])
    processing_id = identity("run", version_id, PIPELINE)
    value.version.update(document_id=document["document_id"], version_id=version_id)
    value.processing.update(version_id=version_id, processing_id=processing_id)
    for rows in [value.pages, value.sections, value.citations]:
        for row in rows:
            row["processing_id"] = processing_id
    return value


def import_prepared(engine, prepared):
    added_docs = added_versions = added_runs = skipped = 0
    # Sorted per-document transaction locks serialize concurrent retries without duplicate rows.
    with engine.begin() as con:
        lock_sources(
            con,
            [
                key
                for p in prepared
                for key in [
                    p.document["source_key"],
                    p.document["detail_url"],
                    p.version["source_url"],
                ]
                if key
            ],
        )
        for item in prepared:
            prior = resolve_source(
                con,
                item.document["source_key"],
                [item.document["detail_url"], item.version["source_url"]],
            )
            if prior and (
                prior["case_number"] != item.document["case_number"]
                or prior["commission"] != "GPO"
            ):
                raise ValueError(
                    "Bestaande documentidentiteit hoort bij een andere zaak; niet overschreven."
                )
            if prior:
                item = rebind(item, prior)
            old_version = (
                con.execute(
                    sa.select(m.versions).where(
                        m.versions.c.version_id == item.version["version_id"]
                    )
                )
                .mappings()
                .first()
            )
            if old_version and any(
                item.version[k] is not None and old_version[k] != item.version[k]
                for k in ["advice_date", "alternative_advice_date", "date_note"]
            ):
                raise ValueError(
                    "Adviesmetadata gewijzigd bij dezelfde PDF; afzonderlijke correctiecontrole vereist."
                )
            existing = (
                con.execute(
                    sa.select(m.runs).where(
                        m.runs.c.processing_id == item.processing["processing_id"]
                    )
                )
                .mappings()
                .first()
            )
            if existing and existing["input_sha256"] != item.processing["input_sha256"]:
                raise ValueError(
                    "Cache gewijzigd binnen dezelfde extractieversie; bestaande registratie niet overschreven."
                )
            con.execute(
                insert(m.documents).values(**item.document).on_conflict_do_nothing()
            )
            con.execute(
                sa.update(m.documents)
                .where(m.documents.c.document_id == item.document["document_id"])
                .values(availability="pdf_verified")
            )
            for kind, url in [
                ("pdf", item.version["source_url"]),
                ("detail", item.document["detail_url"]),
            ]:
                if url:
                    con.execute(
                        insert(m.links)
                        .values(
                            document_id=item.document["document_id"],
                            url=official_url(url),
                            kind=kind,
                        )
                        .on_conflict_do_nothing()
                    )
            snapshot = snapshot_pdf(item.pdf)
            con.execute(
                insert(m.files)
                .values(**snapshot)
                .on_conflict_do_update(
                    index_elements=[m.files.c.sha256],
                    set_={"local_path": snapshot["local_path"]},
                )
            )
            if existing:
                skipped += 1
                continue
            if not prior:
                added_docs += 1
            if old_version is None:
                added_versions += 1
            con.execute(
                insert(m.versions).values(**item.version).on_conflict_do_nothing()
            )
            con.execute(sa.insert(m.runs).values(**item.processing))
            con.execute(sa.insert(m.pages), item.pages)
            con.execute(sa.insert(m.sections), item.sections)
            if item.citations:
                con.execute(sa.insert(m.citations), item.citations)
            added_runs += 1
    return dict(
        documents_added=added_docs,
        versions_added=added_versions,
        extractions_added=added_runs,
        already_registered=skipped,
        downloads=0,
        reembedded=False,
        state="validated_not_indexed",
    )


def import_existing(engine, catalog_path, pdf_root, pages_path, corpus_path):
    return import_prepared(
        engine, prepare_existing(catalog_path, pdf_root, pages_path, corpus_path)
    )


def register_records(engine, records):
    """Register observed metadata only; a PDF link never proves a complete PDF."""
    values = []
    for record in records:
        if record.get("commission") != "GPO":
            raise ValueError("Metadatarecord moet expliciet GPO betreffen.")
        case = record["case_number"]
        if not re.fullmatch(r"\d+", case):
            raise ValueError("Ongeldig zaaknummer.")
        detail = (
            official_url(record["detail_url"]) if record.get("detail_url") else None
        )
        pdf = official_url(record["pdf_url"]) if record.get("pdf_url") else None
        key = record.get("document_key") or detail or pdf
        if not key:
            raise ValueError("Record heeft geen bronidentiteit.")
        state = record.get("availability") or ("pdf_link_found" if pdf else "unchecked")
        if state not in {
            "unchecked",
            "pdf_link_found",
            "no_pdf_link",
            "summary_only",
            "download_failed",
        }:
            raise ValueError("Metadatarecord kan geen geverifieerde PDF verklaren.")
        if state == "pdf_link_found" and not pdf:
            raise ValueError("PDF-linkstatus zonder PDF-link.")
        if state in {"no_pdf_link", "summary_only"} and pdf:
            raise ValueError("Tegenstrijdige linkstatus; controle nodig.")
        values.append(
            (
                dict(
                    document_id=identity("doc", key),
                    source_key=key,
                    commission="GPO",
                    case_number=case,
                    topic=record.get("topic"),
                    sector=record.get("sector"),
                    detail_url=detail,
                    publication_date=date_value(record.get("publication_date")),
                    availability=state,
                ),
                pdf,
            )
        )
    with engine.begin() as con:
        lock_sources(
            con,
            [
                key
                for value, pdf in values
                for key in [value["source_key"], value["detail_url"], pdf]
                if key
            ],
        )
        for value, pdf in values:
            old = resolve_source(con, value["source_key"], [value["detail_url"], pdf])
            if old and old["case_number"] != value["case_number"]:
                raise ValueError("Bronidentiteit heeft een ander zaaknummer.")
            if old:
                value = {
                    **value,
                    "document_id": old["document_id"],
                    "source_key": old["source_key"],
                }
            con.execute(insert(m.documents).values(**value).on_conflict_do_nothing())
            if old:
                update = {
                    k: v
                    for k, v in value.items()
                    if v is not None
                    and k in ["topic", "sector", "detail_url", "publication_date"]
                }
                if old["availability"] != "pdf_verified":
                    update["availability"] = value["availability"]
                update["checked_at"] = sa.func.now()
                con.execute(
                    sa.update(m.documents)
                    .where(m.documents.c.document_id == value["document_id"])
                    .values(**update)
                )
            for kind, url in [("pdf", pdf), ("detail", value["detail_url"])]:
                if url:
                    con.execute(
                        insert(m.links)
                        .values(document_id=value["document_id"], url=url, kind=kind)
                        .on_conflict_do_nothing()
                    )
    return dict(records_observed=len(values), pdfs_downloaded=0)
