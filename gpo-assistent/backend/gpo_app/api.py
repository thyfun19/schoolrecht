from contextlib import asynccontextmanager
import os

import httpx

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from .database import create_engine
from .pinecone_search import pinecone_configured
from . import repository
from . import schemas


def create_app(engine=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.engine = engine if engine is not None else create_engine()
        yield
        if engine is None:
            app.state.engine.dispose()

    app = FastAPI(title="GPO-bronregister", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(SQLAlchemyError)
    async def database_error(request, error):
        return JSONResponse(
            status_code=503,
            content={"detail": "Bronregister tijdelijk niet beschikbaar."},
        )

    @app.get("/api/v1/health", response_model=schemas.Health)
    def health():
        try:
            with app.state.engine.connect() as con:
                revision = con.scalar(
                    text("SELECT version_num FROM gpo_alembic_version")
                )
            return dict(
                status="ok",
                schema_revision=revision,
                counts=repository.counts(app.state.engine),
                pinecone_enabled=pinecone_configured(),
                answer_generation=False,
                paid_api_enabled=False,
            )
        except SQLAlchemyError:
            raise HTTPException(
                503, "Bronregister niet beschikbaar of niet gemigreerd."
            ) from None

    @app.get("/api/v1/documents", response_model=schemas.Documents)
    def documents(
        offset: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=100),
        case_number: str | None = Query(None, pattern=r"^\d+$", max_length=50),
    ):
        return dict(
            documents=repository.list_documents(
                app.state.engine, offset, limit, case_number
            ),
            offset=offset,
            limit=limit,
        )

    @app.get(
        "/api/v1/extractions/{processing_id}/context", response_model=schemas.Context
    )
    def context(processing_id: str):
        value = repository.get_context(app.state.engine, processing_id)
        if value is None:
            raise HTTPException(404, "Extractie niet gevonden.")
        return value

    @app.get("/api/v1/sources/{source_id}", response_model=schemas.Citation)
    def source(source_id: str, processing_id: str = Query(...)):
        try:
            value = repository.get_citation(app.state.engine, processing_id, source_id)
        except ValueError:
            raise HTTPException(409, "Bronintegriteit niet in orde.") from None
        if value is None:
            raise HTTPException(404, "Bron niet gevonden binnen deze extractie.")
        return value

    @app.get(
        "/api/v1/chunk-preparations/{chunk_run_id}",
        response_model=schemas.ChunkPreparation,
    )
    def chunk_preparation(chunk_run_id: str):
        from .chunking import read_preparation

        try:
            value = read_preparation(app.state.engine, chunk_run_id)
        except (ValueError, KeyError):
            raise HTTPException(
                409, "Zoekfragmenten wijken af van de bronregistratie."
            ) from None
        if value is None:
            raise HTTPException(404, "Fragmentverwerking niet gevonden.")
        return value

    @app.get(
        "/api/v1/extractions/{processing_id}/chunk-preparations",
        response_model=list[schemas.ChunkRun],
    )
    def chunk_runs(processing_id: str):
        from .chunking import list_preparations

        return list_preparations(app.state.engine, processing_id)

    @app.get("/api/v1/documents/{version_id}/pdf")
    def pdf(version_id: str):
        try:
            path = repository.verified_pdf(app.state.engine, version_id)
        except ValueError:
            raise HTTPException(
                409, "Deze oorspronkelijke PDF-versie is niet beschikbaar of gewijzigd."
            ) from None
        if path is None:
            raise HTTPException(404, "Documentversie niet gevonden.")
        return FileResponse(
            path, media_type="application/pdf", headers={"Cache-Control": "no-store"}
        )

    @app.post(
        "/api/v1/assessments/prepare",
        summary="Eén beoordelingsaanvraag per gevonden zaak; geen modelaanroep",
    )
    def assessment_prepare(body: schemas.AssessmentRequest):
        from .isolated_assessment import prepare

        try:
            return prepare(app.state.engine, body.question, body.processing_ids)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.post(
        "/api/v1/assessments/validate",
        summary="Controleer afzonderlijke zaakresultaten en voeg backendcitaten toe",
    )
    def assessment_validate(body: schemas.AssessmentValidation):
        from .isolated_assessment import validate

        try:
            return validate(app.state.engine, body.payload, body.prepared)
        except (ValueError, KeyError, TypeError) as error:
            raise HTTPException(
                422, "Zaakresultaat of bronpakket ongeldig of gewijzigd."
            ) from error

    @app.post(
        "/api/v1/evidence-plans/prepare",
        summary="Experimenteel: bereid bewijsselectie voor, geen modelaanroep",
    )
    def plan_prepare(body: schemas.AssessmentRequest):
        from .evidence_workflow import prepare

        try:
            return prepare(app.state.engine, body.question, body.processing_ids)
        except (ValueError, KeyError, TypeError) as error:
            raise HTTPException(422, "Bewijsworkflow niet beschikbaar.") from error

    @app.post(
        "/api/v1/vergelijk",
        summary="Lokale vergelijking met letterlijke commissiepassages",
    )
    def compare_cases(body: schemas.PineconeSearchRequest):
        from gpo_local.question_intake import question_intake

        decision = question_intake(body.question)
        if decision:
            return decision
        try:
            from .historical_passages import compare

            return compare(
                app.state.engine,
                body.question,
                min(body.limit, 5),
                body.topic,
                body.sector,
            )
        except (ImportError, OSError, ValueError) as error:
            raise HTTPException(
                503, "De lokale vergelijkfunctie of haar bronindex is niet beschikbaar."
            ) from error

    @app.post(
        "/api/v1/search/pinecone",
        summary="Live semantische vectorzoekopdracht via Pinecone (BGE-M3 1024d) en Supabase",
    )
    def search_pinecone(body: schemas.PineconeSearchRequest):
        q = body.question.strip()
        from gpo_local.explanation import scope_boundary

        boundary = scope_boundary(q)
        if boundary:
            raise HTTPException(422, boundary)
        try:
            if os.getenv("GPO_SEARCH_MODE") == "catalogue_bge_rerank":
                from gpo_local.question_intake import question_intake

                decision = question_intake(q)
                if decision:
                    return decision
                from .historical_passages import search_response

                return search_response(app.state.engine, body)
            from .pinecone_search import pinecone_search

            return pinecone_search().search(
                app.state.engine, q, body.limit, body.topic, body.sector
            )
        except ImportError as error:
            raise HTTPException(
                503,
                "Lokale zoekdependencies ontbreken. Start met de bestaande DependencyPath.",
            ) from error
        except (ValueError, OSError, RuntimeError, httpx.HTTPError) as error:
            raise HTTPException(
                503, "Zoeken in het bronregister is tijdelijk niet beschikbaar."
            ) from error

    @app.post(
        "/api/v1/evidence-plans/search/prepare",
        summary="Experimentele lokale tekst/hybride of Pinecone zoekroute naar bewijsselectie",
    )
    def plan_search(body: schemas.CandidateSearchRequest):
        from .evidence_workflow import prepare
        from gpo_local.question_intake import question_intake

        decision = question_intake(body.question)
        if decision:
            return decision
        try:
            if body.method == "pinecone_bge":
                from .pinecone_search import pinecone_search

                found = pinecone_search().search(
                    app.state.engine, body.question, body.limit
                )
            else:
                from .quality_search import candidate_search

                found = candidate_search(body.method).search(body.question, body.limit)
            if not found["processing_ids"]:
                return dict(
                    status="insufficient_sources",
                    retrieval=found,
                    workflow=None,
                    paid_calls=0,
                )
            return dict(
                retrieval=found,
                workflow=prepare(
                    app.state.engine, body.question, found["processing_ids"]
                ),
            )
        except ImportError as error:
            raise HTTPException(
                503,
                "Lokale zoekdependencies ontbreken. Start met de bestaande DependencyPath.",
            ) from error
        except (ValueError, KeyError, OSError) as error:
            raise HTTPException(
                422, "Lokale zoekcache of broncontext niet beschikbaar."
            ) from error

    @app.post(
        "/api/v1/evidence-plans/validate",
        summary="Controleer bewijsselectie; inhoudelijke review blijft nodig",
    )
    def plan_validate(body: schemas.PlanValidation):
        from .evidence_workflow import validate

        try:
            return validate(app.state.engine, body.payloads, body.workflow)
        except (ValueError, KeyError, TypeError) as error:
            raise HTTPException(422, "Bewijsselectie ongeldig of gewijzigd.") from error

    @app.post(
        "/api/v1/evidence-plans/writing/prepare",
        summary="Bereid schrijven voor na expliciete bewijsreview",
    )
    def plan_writing(body: schemas.PlanWriting):
        from .evidence_workflow import writing

        try:
            return writing(
                app.state.engine, body.validated, body.workflow, body.reviews
            )
        except (ValueError, KeyError, TypeError) as error:
            raise HTTPException(422, "Beoordeelde bewijsselectie vereist.") from error

    @app.post(
        "/api/v1/evidence-plans/writing/validate",
        summary="Valideer uitleg uitsluitend tegen beoordeelde bewijsselectie",
    )
    def plan_output(body: schemas.PlanOutputValidation):
        from .evidence_workflow import validate_output

        try:
            return validate_output(
                app.state.engine,
                body.payload,
                body.validated,
                body.workflow,
                body.reviews,
            )
        except (ValueError, KeyError, TypeError) as error:
            raise HTTPException(
                422, "Uitleg buiten beoordeelde bewijsselectie of ongeldig."
            ) from error

    return app


app = create_app()
