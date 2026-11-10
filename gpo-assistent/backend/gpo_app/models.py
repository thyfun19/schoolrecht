"""PostgreSQL source identity and immutable extraction provenance; no vector storage."""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

metadata = sa.MetaData(
    naming_convention={
        "pk": "pk_%(table_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ix": "ix_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
    }
)

documents = sa.Table(
    "gpo_documents",
    metadata,
    sa.Column("document_id", sa.String(40), primary_key=True),
    sa.Column("source_key", sa.Text, nullable=False, unique=True),
    sa.Column("commission", sa.String(10), nullable=False),
    sa.Column("case_number", sa.String(50), nullable=False, index=True),
    sa.Column("topic", sa.Text),
    sa.Column("sector", sa.Text),
    sa.Column("detail_url", sa.Text),
    sa.Column("publication_date", sa.Date),
    sa.Column("availability", sa.String(30), nullable=False),
    sa.Column(
        "checked_at",
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.func.now(),
    ),
    sa.CheckConstraint("commission = 'GPO'", name="gpo_only"),
    sa.CheckConstraint(
        "availability IN ('unchecked','pdf_link_found','no_pdf_link','summary_only','download_failed','pdf_verified')",
        name="availability",
    ),
)

links = sa.Table(
    "gpo_source_links",
    metadata,
    sa.Column(
        "document_id",
        sa.String(40),
        sa.ForeignKey(documents.c.document_id),
        primary_key=True,
    ),
    sa.Column("url", sa.Text, primary_key=True),
    sa.Column("kind", sa.String(10), nullable=False),
    sa.CheckConstraint("kind IN ('pdf','detail')", name="kind"),
)

files = sa.Table(
    "gpo_pdf_files",
    metadata,
    sa.Column("sha256", sa.String(64), primary_key=True),
    sa.Column("local_path", sa.Text, nullable=False),
    sa.Column("byte_size", sa.BigInteger, nullable=False),
    sa.Column("page_count", sa.Integer, nullable=False),
    sa.CheckConstraint("length(sha256) = 64", name="hash_length"),
    sa.CheckConstraint("byte_size > 0 AND page_count > 0", name="positive_size"),
)

versions = sa.Table(
    "gpo_document_versions",
    metadata,
    sa.Column("version_id", sa.String(40), primary_key=True),
    sa.Column(
        "document_id",
        sa.String(40),
        sa.ForeignKey(documents.c.document_id),
        nullable=False,
    ),
    sa.Column(
        "pdf_sha256", sa.String(64), sa.ForeignKey(files.c.sha256), nullable=False
    ),
    sa.Column("advice_date", sa.Date),
    sa.Column("alternative_advice_date", sa.Date),
    sa.Column("date_note", sa.Text),
    sa.Column("source_url", sa.Text, nullable=False),
    sa.Column("retrieved_on", sa.Date),
    sa.UniqueConstraint("document_id", "pdf_sha256"),
)

runs = sa.Table(
    "gpo_extractions",
    metadata,
    sa.Column("processing_id", sa.String(40), primary_key=True),
    sa.Column(
        "version_id",
        sa.String(40),
        sa.ForeignKey(versions.c.version_id),
        nullable=False,
    ),
    sa.Column("pipeline_version", sa.Text, nullable=False),
    sa.Column("input_sha256", sa.String(64), nullable=False),
    sa.Column("state", sa.String(30), nullable=False),
    sa.Column("provenance", JSONB, nullable=False),
    sa.Column(
        "created_at",
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.func.now(),
    ),
    sa.UniqueConstraint("version_id", "pipeline_version"),
    sa.CheckConstraint(
        "state IN ('validated_not_indexed','needs_review','failed')", name="state"
    ),
)

pages = sa.Table(
    "gpo_pages",
    metadata,
    sa.Column(
        "processing_id",
        sa.String(40),
        sa.ForeignKey(runs.c.processing_id),
        primary_key=True,
    ),
    sa.Column("page_number", sa.Integer, primary_key=True),
    sa.Column("raw_text", sa.Text, nullable=False),
    sa.Column("cleaned_text", sa.Text, nullable=False),
    sa.Column("raw_sha256", sa.String(64), nullable=False),
    sa.CheckConstraint("page_number > 0", name="physical_page"),
)

sections = sa.Table(
    "gpo_sections",
    metadata,
    sa.Column("processing_id", sa.String(40), primary_key=True),
    sa.Column("section_key", sa.Text, primary_key=True),
    sa.Column("page_number", sa.Integer, nullable=False),
    sa.Column("ordinal", sa.Integer, nullable=False),
    sa.Column("heading", sa.Text, nullable=False),
    sa.Column("role", sa.Text, nullable=False),
    sa.Column("category", sa.Text, nullable=False),
    sa.Column("text", sa.Text, nullable=False),
    sa.ForeignKeyConstraint(
        ["processing_id", "page_number"], [pages.c.processing_id, pages.c.page_number]
    ),
    sa.UniqueConstraint("processing_id", "ordinal"),
)

citations = sa.Table(
    "gpo_citations",
    metadata,
    sa.Column("processing_id", sa.String(40), primary_key=True),
    sa.Column("source_id", sa.String(40), primary_key=True),
    sa.Column("section_key", sa.Text, nullable=False),
    sa.Column("start_offset", sa.Integer, nullable=False),
    sa.Column("end_offset", sa.Integer, nullable=False),
    sa.Column("quote", sa.Text, nullable=False),
    sa.ForeignKeyConstraint(
        ["processing_id", "section_key"],
        [sections.c.processing_id, sections.c.section_key],
    ),
    sa.CheckConstraint(
        "start_offset >= 0 AND end_offset > start_offset", name="offsets"
    ),
    sa.CheckConstraint(
        "end_offset - start_offset = length(quote)", name="quote_length"
    ),
)

# Derived, versioned search preparation. Canonical pages/sections/citations stay unchanged.
chunk_runs = sa.Table(
    "gpo_chunk_runs",
    metadata,
    sa.Column("chunk_run_id", sa.String(40), primary_key=True),
    sa.Column(
        "processing_id",
        sa.String(40),
        sa.ForeignKey(runs.c.processing_id),
        nullable=False,
    ),
    sa.Column("config", JSONB, nullable=False),
    sa.Column("input_sha256", sa.String(64), nullable=False),
    sa.Column(
        "created_at",
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.func.now(),
    ),
)
contexts = sa.Table(
    "gpo_search_contexts",
    metadata,
    sa.Column("context_id", sa.String(40), primary_key=True),
    sa.Column(
        "chunk_run_id",
        sa.String(40),
        sa.ForeignKey(chunk_runs.c.chunk_run_id),
        nullable=False,
        index=True,
    ),
    sa.Column("category", sa.Text, nullable=False),
    sa.Column("role", sa.Text, nullable=False),
    sa.Column("boundary_status", sa.Text, nullable=False),
    sa.Column("spans", JSONB, nullable=False),
)
chunks = sa.Table(
    "gpo_search_chunks",
    metadata,
    sa.Column("chunk_id", sa.String(40), primary_key=True),
    sa.Column(
        "context_id",
        sa.String(40),
        sa.ForeignKey(contexts.c.context_id),
        nullable=False,
        index=True,
    ),
    sa.Column("ordinal", sa.Integer, nullable=False),
    sa.Column("text", sa.Text, nullable=False),
    sa.Column("token_count", sa.Integer, nullable=False),
    sa.Column("span", JSONB, nullable=False),
    sa.Column("split_reason", sa.Text, nullable=False),
    sa.UniqueConstraint("context_id", "ordinal"),
    sa.CheckConstraint("token_count > 0 AND length(text) > 0", name="nonempty"),
)
