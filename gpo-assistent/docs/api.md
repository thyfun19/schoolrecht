# API Documentatie — SchoolRecht REST API

De SchoolRecht backend stelt een RESTful API beschikbaar op basis van FastAPI.
Interactieve OpenAPI / Swagger documentatie is live beschikbaar via `/docs`.

- **Live URL:** `https://schoolrecht-api.onrender.com/docs`
- **Lokale URL:** `http://127.0.0.1:8002/docs`

---

## Overzicht van Endpoints

| Methode | Pad | Beschrijving |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Status van de service, database en Pinecone vectorindex |
| `GET` | `/api/v1/documents` | Paginatie van geregistreerde GPO-uitspraken |
| `GET` | `/api/v1/documents/{version_id}/pdf` | Download de cryptografisch geverifieerde PDF |
| `GET` | `/api/v1/extractions/{processing_id}/context` | Volledige zaakcontext en ingedeelde secties |
| `GET` | `/api/v1/sources/{source_id}` | Exact broncitaat met paginanummer en karakter-offsets |
| `POST` | `/api/v1/search/pinecone` | Hybride vector- en tekstzoekopdracht voor oudervragen |
| `POST` | `/api/v1/vergelijk` | Extractieve vergelijking van commissieoverwegingen |

---

## Endpoint Details

### 1. Systeemstatus: `GET /api/v1/health`

Controleert de status van de backend, de huidige databaseschema-versie en de verbinding met Pinecone.

**Voorbeeld Response (`200 OK`):**
```json
{
  "status": "ok",
  "schema_revision": "0002_search_children",
  "counts": {
    "documents": 10,
    "versions": 10,
    "pages": 57,
    "citations": 1420
  },
  "pinecone_enabled": true,
  "answer_generation": false,
  "paid_api_enabled": false
}
```

---

### 2. Documentenoverzicht: `GET /api/v1/documents`

Geeft een gepagineerde lijst van geregistreerde uitspraken terug.

**Query Parameters:**
- `offset` *(integer, default: 0)*: Startindex.
- `limit` *(integer, default: 50, max: 100)*: Aantal documenten.
- `case_number` *(string, optioneel)*: Filter op zaaknummer (bijv. `109206`).

**Voorbeeld Response (`200 OK`):**
```json
{
  "documents": [
    {
      "case_number": "109206",
      "advice_date": "2020-04-14",
      "topic": "Toelating en plaatsing",
      "sector": "Primair onderwijs",
      "page_count": 6,
      "pdf_sha256": "3a4f8c...",
      "source_url": "https://onderwijsgeschillen.nl/uitspraak/109206"
    }
  ],
  "offset": 0,
  "limit": 50
}
```

---

### 3. Hybride Zoeken: `POST /api/v1/search/pinecone`

Zoekt relevante GPO-overwegingen op basis van een vraag in natuurlijke taal.

**Request Body:**
```json
{
  "question": "Mag de school weigeren omdat de klas vol zit?",
  "limit": 5,
  "topic": "Toelating en plaatsing",
  "sector": "Primair onderwijs"
}
```

**Voorbeeld Response (`200 OK`):**
```json
{
  "status": "success",
  "question": "Mag de school weigeren omdat de klas vol zit?",
  "results": [
    {
      "case_number": "109206",
      "score": 0.842,
      "topic": "Toelating en plaatsing",
      "sector": "Primair onderwijs",
      "advice_date": "2020-04-14",
      "summary_quote": "Het enkele argument dat groepen vol zijn, ontslaat het bevoegd gezag niet van zijn wettelijke onderzoeksplicht.",
      "page": 4,
      "source_id": "src_109206_p4_s3_c1",
      "char_start": 340,
      "char_end": 458
    }
  ]
}
```

---

### 4. Broncitaat Ophalen: `GET /api/v1/sources/{source_id}`

Haalt het exacte citaat op inclusief bronverificatie.

**Query Parameters:**
- `processing_id` *(string, verplicht)*: ID van de specifieke verwerkingsrun.

**Voorbeeld Response (`200 OK`):**
```json
{
  "source_id": "src_109206_p4_s3_c1",
  "case_number": "109206",
  "page_number": 4,
  "category": "commissieoverwegingen",
  "quote": "Het enkele argument dat groepen vol zijn, ontslaat het bevoegd gezag niet van zijn wettelijke onderzoeksplicht.",
  "char_start": 340,
  "char_end": 458,
  "pdf_sha256": "3a4f8c1b..."
}
```
