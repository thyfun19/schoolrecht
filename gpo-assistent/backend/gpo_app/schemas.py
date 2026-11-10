"""Public read-only API contracts. Local file paths and credentials have no fields."""

from datetime import date, datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


class PublicResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AssessmentRequest(PublicResponse):
    question: str = Field(min_length=3, max_length=2000)
    processing_ids: list[str] = Field(min_length=1, max_length=5)


class AssessmentValidation(PublicResponse):
    prepared: dict
    payload: dict


class PlanValidation(PublicResponse):
    workflow: dict
    payloads: list[dict] = Field(min_length=1, max_length=5)


class PlanWriting(PublicResponse):
    workflow: dict
    validated: dict
    reviews: dict


class PlanOutputValidation(PlanWriting):
    payload: dict


class CandidateSearchRequest(PublicResponse):
    question: str = Field(min_length=3, max_length=2000)
    method: Literal["text", "hybrid_bge", "pinecone_bge"] = "text"
    limit: int = Field(default=3, ge=1, le=5)

    @field_validator("question")
    @classmethod
    def clean_question(cls, value):
        value = value.strip()
        if len(value) < 3:
            raise ValueError("Vul minstens drie tekens in.")
        return value


class PineconeSearchRequest(PublicResponse):
    question: str = Field(min_length=3, max_length=2000)
    limit: int = Field(default=5, ge=1, le=50)
    topic: str | None = Field(default=None, max_length=100)
    sector: str | None = Field(default=None, max_length=100)

    @field_validator("question")
    @classmethod
    def clean_question(cls, value):
        value = value.strip()
        if len(value) < 3:
            raise ValueError("Vul minstens drie tekens in.")
        return value


class RegistryCounts(PublicResponse):
    documents: int
    pdf_files: int
    document_versions: int
    extractions: int
    pages: int
    sections: int
    citations: int
    availability: dict[str, int]


class Health(PublicResponse):
    status: Literal["ok"]
    schema_revision: str
    counts: RegistryCounts
    pinecone_enabled: bool
    answer_generation: bool
    paid_api_enabled: bool


class Extraction(PublicResponse):
    processing_id: str
    pipeline_version: str
    state: str


class DocumentVersion(PublicResponse):
    version_id: str
    document_id: str
    pdf_sha256: str
    advice_date: date | None
    alternative_advice_date: date | None
    date_note: str | None
    source_url: str
    retrieved_on: date | None


class ListedVersion(DocumentVersion):
    extractions: list[Extraction]


class Document(PublicResponse):
    document_id: str
    source_key: str
    commission: Literal["GPO"]
    case_number: str
    topic: str | None
    sector: str | None
    detail_url: str | None
    publication_date: date | None
    availability: str
    checked_at: datetime
    versions: list[ListedVersion]


class Documents(PublicResponse):
    documents: list[Document]
    offset: int
    limit: int


class Section(PublicResponse):
    processing_id: str
    section_key: str
    page_number: int
    ordinal: int
    heading: str
    role: str
    category: str
    text: str
    source_url: str
    source_ids: list[str]


class Context(DocumentVersion):
    processing_id: str
    state: str
    case_number: str
    parts: dict[str, list[Section]]
    source_roles_are_not_semantic_endorsement: bool


class Citation(PublicResponse):
    processing_id: str
    source_id: str
    section_key: str
    start_offset: int
    end_offset: int
    quote: str
    page_number: int
    heading: str
    role: str
    text: str
    version_id: str
    advice_date: date | None
    pdf_sha256: str
    source_url: str
    case_number: str
    pdf_url: str
    semantic_support: Literal["not_assessed"]


class SourceSpan(PublicResponse):
    section_key: str
    page_number: int
    start_offset: int
    end_offset: int
    source_ids: list[str]
    text: str
    heading: str
    role: str
    source_url: str


class SearchChunk(PublicResponse):
    chunk_id: str
    context_id: str
    ordinal: int
    text: str
    token_count: int
    span: SourceSpan
    split_reason: str


class SearchContext(PublicResponse):
    context_id: str
    chunk_run_id: str
    category: str
    role: str
    boundary_status: str
    spans: list[SourceSpan]
    chunks: list[SearchChunk]


class ChunkRun(PublicResponse):
    chunk_run_id: str
    processing_id: str
    config: dict
    input_sha256: str
    created_at: datetime


class ChunkPreparation(ChunkRun):
    case_number: str
    source_url: str
    version_id: str
    advice_date: date | None
    contexts: list[SearchContext]
    semantic_support: Literal["not_assessed"]
