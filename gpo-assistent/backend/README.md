# SchoolRecht Backend API

De backend van **SchoolRecht** is een performante RESTful API gebouwd met **Python 3.11+**, **FastAPI**, **SQLAlchemy 2.0** en **Pydantic v2**.
De service verzorgt de semantische vectorretrieval via Pinecone (`BAAI/bge-m3`), lexicaal zoeken via PostgreSQL/BM25 en cryptografische bronverificatie op originele GPO-adviezen.

---

## Architectuur & Verantwoordelijkheden

- **`gpo_app/`**: Kern FastAPI applicatie:
  - `api.py`: REST routes (`/api/v1/health`, `/api/v1/documents`, `/api/v1/search/pinecone`, `/api/v1/sources/{id}`).
  - `pinecone_search.py`: Hybride retrieval die BGE-M3 dense embeddings combineert met BM25 en Cross-Encoder reranking.
  - `historical_passages.py`: Extractie van commissieoverwegingen gekoppeld aan paginanummers en bronposities.
  - `schemas.py`: Strikte Pydantic validatiemodellen voor publieke requests en responses.
  - `database.py` & `models.py`: PostgreSQL tabellen voor documenten, versies, pagina's en citaten.
- **`gpo_local/`**: Lokale ingestie- en parsingtools:
  - `source_registry.py` & `grounding.py`: PDF-ingestie, cryptografische SHA-256 verificatie en sectieclassificatie.
  - `question_intake.py` & `explanation.py`: Procedurele scope-bewaking voor juridische contrastvragen.
- **`tests_app/`**: Integratietests voor het bronregister, de REST API-routes en documentverificatie.
- **`tests_vector/`**: Unittests voor vectorchunking, retrieval-evaluatie en grenswaarden.

---

## Lokaal Ontwikkelen

### 1. Virtuele Omgeving & Dependencies

```bash
python -m venv .venv

# Windows:
.\.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Lokale PostgreSQL Database & Migraties

Voor lokale ontwikkeling en integratietests kan de PostgreSQL container worden gestart:

```bash
docker compose -p gpo-source-registry -f compose-registry.yaml up -d --wait
alembic -c alembic-registry.ini upgrade head
```

### 3. API Server Starten

```bash
uvicorn gpo_app.api:app --host 127.0.0.1 --port 8002 --reload
```

Open de Swagger documentatie op `http://127.0.0.1:8002/docs`.

---

## Geautomatiseerde Tests

De testsuite omvat 241 geautomatiseerde tests:

```bash
# Draai alle tests
pytest -q tests_app/ tests_vector/
```
