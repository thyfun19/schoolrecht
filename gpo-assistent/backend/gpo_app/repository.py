"""Read the canonical registry; external URLs and local file paths are server-owned."""

from pathlib import Path
import sqlalchemy as sa
from . import models as m
from gpo_local.common import digest
from gpo_local.source_registry import source_id as canonical_source_id


def counts(engine):
    with engine.connect() as con:
        values = {
            table.name.removeprefix("gpo_"): con.scalar(
                sa.select(sa.func.count()).select_from(table)
            )
            for table in [
                m.documents,
                m.files,
                m.versions,
                m.runs,
                m.pages,
                m.sections,
                m.citations,
            ]
        }
        values["availability"] = {
            r[0]: r[1]
            for r in con.execute(
                sa.select(m.documents.c.availability, sa.func.count()).group_by(
                    m.documents.c.availability
                )
            )
        }
        return values


def list_documents(engine, offset=0, limit=50, case_number=None):
    with engine.connect() as con:
        query = sa.select(m.documents)
        if case_number is not None:
            query = query.where(m.documents.c.case_number == case_number)
        docs = [
            dict(r)
            for r in con.execute(
                query.order_by(m.documents.c.case_number, m.documents.c.document_id)
                .offset(offset)
                .limit(limit)
            ).mappings()
        ]
        for doc in docs:
            doc["versions"] = [
                dict(r)
                for r in con.execute(
                    sa.select(m.versions)
                    .where(m.versions.c.document_id == doc["document_id"])
                    .order_by(m.versions.c.version_id)
                ).mappings()
            ]
            for version in doc["versions"]:
                version["extractions"] = [
                    dict(r)
                    for r in con.execute(
                        sa.select(
                            m.runs.c.processing_id,
                            m.runs.c.pipeline_version,
                            m.runs.c.state,
                        ).where(m.runs.c.version_id == version["version_id"])
                    ).mappings()
                ]
        return docs


def get_context(engine, processing_id):
    with engine.connect() as con:
        info = (
            con.execute(
                sa.select(
                    m.runs.c.processing_id,
                    m.runs.c.state,
                    m.versions,
                    m.documents.c.case_number,
                )
                .join(m.versions, m.versions.c.version_id == m.runs.c.version_id)
                .join(
                    m.documents, m.documents.c.document_id == m.versions.c.document_id
                )
                .where(m.runs.c.processing_id == processing_id)
            )
            .mappings()
            .first()
        )
        if not info:
            return None
        refs = {}
        for source in con.execute(
            sa.select(m.citations.c.section_key, m.citations.c.source_id)
            .where(m.citations.c.processing_id == processing_id)
            .order_by(m.citations.c.start_offset)
        ).mappings():
            refs.setdefault(source["section_key"], []).append(source["source_id"])
        parts = {
            k: []
            for k in [
                "commissieoverwegingen",
                "conclusies",
                "feiten",
                "partijstandpunten",
                "overige",
            ]
        }
        for row in con.execute(
            sa.select(m.sections)
            .where(m.sections.c.processing_id == processing_id)
            .order_by(m.sections.c.ordinal)
        ).mappings():
            section = dict(row)
            section["source_url"] = info["source_url"] + f"#page={row['page_number']}"
            section["source_ids"] = refs.get(row["section_key"], [])
            parts[row["category"]].append(section)
        return dict(**info, parts=parts, source_roles_are_not_semantic_endorsement=True)


def get_citation(engine, processing_id, source_id):
    with engine.connect() as con:
        row = (
            con.execute(
                sa.select(
                    m.citations,
                    m.sections.c.page_number,
                    m.sections.c.heading,
                    m.sections.c.role,
                    m.sections.c.text,
                    m.versions.c.version_id,
                    m.versions.c.advice_date,
                    m.versions.c.pdf_sha256,
                    m.versions.c.source_url,
                    m.documents.c.case_number,
                )
                .select_from(
                    m.citations.join(
                        m.sections,
                        sa.and_(
                            m.citations.c.processing_id == m.sections.c.processing_id,
                            m.citations.c.section_key == m.sections.c.section_key,
                        ),
                    )
                    .join(m.runs, m.runs.c.processing_id == m.citations.c.processing_id)
                    .join(m.versions, m.versions.c.version_id == m.runs.c.version_id)
                    .join(
                        m.documents,
                        m.documents.c.document_id == m.versions.c.document_id,
                    )
                )
                .where(
                    m.citations.c.processing_id == processing_id,
                    m.citations.c.source_id == source_id,
                )
            )
            .mappings()
            .first()
        )
        if not row:
            return None
        data = dict(row)
        canonical = dict(
            case_number=data["case_number"],
            pdf_sha256=data["pdf_sha256"],
            page=data["page_number"],
            section_id=data["section_key"],
            section=data["heading"],
            role=data["role"],
            start_offset=data["start_offset"],
            end_offset=data["end_offset"],
            quote=data["quote"],
        )
        if (
            data["text"][data["start_offset"] : data["end_offset"]] != data["quote"]
            or canonical_source_id(canonical) != data["source_id"]
        ):
            raise ValueError("Citaatregistratie wijkt af; niet tonen.")
        data["source_url"] = data["source_url"] + f"#page={data['page_number']}"
        data["pdf_url"] = (
            f"/api/v1/documents/{data['version_id']}/pdf#page={data['page_number']}"
        )
        data["semantic_support"] = "not_assessed"
        return data


def verified_pdf(engine, version_id):
    with engine.connect() as con:
        row = (
            con.execute(
                sa.select(
                    m.files.c.local_path, m.files.c.sha256, m.versions.c.version_id
                )
                .select_from(
                    m.versions.join(
                        m.files, m.files.c.sha256 == m.versions.c.pdf_sha256
                    )
                )
                .where(m.versions.c.version_id == version_id)
            )
            .mappings()
            .first()
        )
    if not row:
        return None
    path = Path(row["local_path"])
    if not path.is_file() or digest(path) != row["sha256"]:
        raise ValueError("Gebruikte PDF-versie ontbreekt of is gewijzigd.")
    return path
