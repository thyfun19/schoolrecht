import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .evidence_assessment import prepare_evidence, hydrate_evidence
from .question_intake import question_intake
from .retrieval import Retrieval
from .source_registry import SourceRegistry

BASE = Path(__file__).resolve().parents[1]


class SearchRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "question": "We hebben vaak gesprekken met school, maar ik krijg het OPP pas laat en er is geen duidelijke evaluatie met mij. Zijn er vergelijkbare zaken?",
                "method": "vector",
                "limit": 3,
            }
        },
    )
    question: str = Field(min_length=3, max_length=2000)
    method: Literal["vector", "text", "text_e5_chunks"] = "text"
    limit: int = Field(default=3, ge=1, le=10)

    @field_validator("question")
    @classmethod
    def clean(cls, value):
        value = value.strip()
        if len(value) < 3:
            raise ValueError("Vul minstens drie tekens in.")
        return value


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    questions: list[SearchRequest] = Field(min_length=1, max_length=10)


class ValidateRequest(SearchRequest):
    payload: dict


def create_app(bundle=None, database=None, engine=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.engine = engine or Retrieval(
            Path(bundle or os.getenv("GPO_VECTOR_BUNDLE", BASE / "data/e5")),
            Path(database or os.getenv("GPO_DATABASE", BASE / "data/gpo.sqlite3")),
        )
        app.state.source_registry = SourceRegistry(
            app.state.engine.documents, app.state.engine.sections
        )
        yield

    app = FastAPI(title="GPO lokaal — E5 en tekstzoeken", lifespan=lifespan)

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/docs")

    @app.get("/health")
    def health():
        return dict(
            status="ok",
            documents=len(app.state.engine.documents),
            fragments=len(app.state.engine.chunks),
            answer_generation=False,
            paid_api_enabled=False,
        )

    @app.get("/documents")
    def documents():
        return dict(documents=list(app.state.engine.documents.values()))

    def run(body):
        try:
            return app.state.engine.search(body.question, body.method, body.limit)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/search")
    def search(body: SearchRequest):
        return run(body)

    @app.post(
        "/search/batch",
        summary="Zoek meerdere vragen met dezelfde verwerking als losse vragen",
    )
    def search_batch(body: BatchRequest):
        methods = {q.method for q in body.questions}
        limits = {q.limit for q in body.questions}
        if len(methods) != 1 or len(limits) != 1:
            raise HTTPException(
                422, detail="Gebruik dezelfde methode en limiet voor de hele batch."
            )
        try:
            return app.state.engine.search_many(
                [q.question for q in body.questions],
                body.questions[0].method,
                body.questions[0].limit,
            )
        except ValueError as error:
            raise HTTPException(422, detail=str(error)) from error

    @app.post(
        "/explanation/prepare",
        summary="Bereid brongebonden uitleg voor; doet geen API-aanroep",
    )
    def explanation(body: SearchRequest):
        decision = question_intake(body.question)
        if decision:
            return decision
        response = run(body)
        cases = [c["case_number"] for c in response["results"]]
        return prepare_evidence(
            body.question, cases, app.state.source_registry.select(cases)
        )

    @app.post(
        "/explanation/validate",
        summary="Voeg geregistreerde citaten toe; geen inhoudelijke goedkeuring of API-aanroep",
    )
    def validate_explanation(body: ValidateRequest):
        registry = app.state.source_registry
        prepared = explanation(body)
        try:
            return hydrate_evidence(body.payload, prepared, registry)
        except ValueError as error:
            raise HTTPException(422, detail=str(error)) from error

    return app


app = create_app()
