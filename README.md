<p align="center">
  <img src="gpo-assistent/frontend/public/schoolrecht-logo.png" alt="SchoolRecht Logo" width="320" />
</p>

# SchoolRecht — Zaakvergelijker Passend Onderwijs

<p align="center">
  <strong>🌐 Live Applicatie:</strong> <a href="https://public-lab-timi-fung-kraak-de-code-c3174c.dev.hihva.nl">public-lab-timi-fung-kraak-de-code-c3174c.dev.hihva.nl</a>
  &nbsp;|&nbsp;
  <strong>⚡ Live API (Swagger):</strong> <a href="https://schoolrecht-api.onrender.com/docs">schoolrecht-api.onrender.com/docs</a>
</p>

**SchoolRecht** is een full-stack LegalTech-webapplicatie en bronregister waarmee ouders, leerlingen en onderwijsprofessionals openbare uitspraken van de **Geschillencommissie Passend Onderwijs (GPO)** kunnen doorzoeken en vergelijken op basis van alledaagse hulpvragen.

In tegenstelling tot generieke chatbots die juridische adviezen vrij samenvatten (met risico op hallucinaties), werkt SchoolRecht volgens het principe **Data bij de Bron**: elke getoonde overweging is gekoppeld aan een cryptografisch geverifieerd broncitaat (`SHA-256` PDF-hash, karakter-offsets en fysiek paginanummer) uit de originele uitspraak.

---

## Hoe ik op het idee kwam

Tijdens een project aan de HvA werkten we in teamverband voor een opdrachtgever van de Huurcommissie. Onze Product Owner noemde destijds een interessant voorbeeld: de zaakvergelijker van de Rechtspraak.

Wat me daarin direct opviel:
- Het was geen chatbot die zelf teksten samenvat of dingen kan verzinnen, maar een zoeksysteem dat puur de relevante passages en eerdere uitspraken naast elkaar zet. In het juridische domein is betrouwbaarheid immers essentieel.

Dat bracht me op het idee om dit toe te passen op een ander domein: schoolgeschillen in het passend onderwijs (adviezen van de Geschillencommissie Passend Onderwijs, GPO). Ouders van kinderen met een extra ondersteuningsbehoefte kennen de formele juridische termen vaak niet en weten niet waar ze moeten zoeken als een school bijvoorbeeld zegt dat een klas vol zit.

Daarnaast was het een praktische keuze: een enorm corpus aan uitspraken lokaal verwerken en indexeren vraagt veel van een eigen laptop. De collectie van de GPO vormde een behapbare dataset. Hierdoor kon ik het hele proces — van PDF-extractie en SHA-256 verificatie tot vectorretrieval en een NL Design System frontend — lokaal opzetten en van begin tot eind zelf bouwen.

---

## Kernfunctionaliteiten

- **Hybride Zoekarchitectuur (Oudertaal → Juridische Begrippen)**
  Combineert semantische vector-retrieval (`BGE-M3` / `multilingual-e5` via Pinecone), PostgreSQL/BM25 lexicale ranking met Reciprocal Rank Fusion (RRF) en een lokale ONNX Cross-Encoder reranker om vragen in oudertaal (bijv. *"klas zit vol"*, *"terugkomen met een aangepast rooster"*, *"verbeterpunten op het OPP"*) direct te koppelen aan de relevante rechtsoverwegingen van de Commissie.
- **Cryptografische Bronverificatie (Zero-Hallucination Citaatregister)**
  Elke uitspraak is opgedeeld in gestructureerde secties (`feiten`, `partijstandpunten`, `commissieoverwegingen`, `conclusies`) en geregistreerde zinsfragmenten met exacte karakter-offsets en paginanummers, gevalideerd tegen de `SHA-256` checksum van de originele PDF.
- **Procedurele Scope-Bewaking**
  Vragen over persoonlijke bezwaartermijnen of het bevoegde loket worden vóór de zoekstap onderschept met een gerichte verwijzing naar actuele procedurebronnen, zodat historische uitspraken nooit ten onrechte als persoonlijk procesadvies worden gepresenteerd.
- **Toegankelijke NL Design System Frontend**
  Gebouwd met React 18, TypeScript en **NL Design System / Utrecht Design Tokens** voor een toegankelijke, overheidsconforme gebruikerservaring inclusief zijpaneel (`CitationDrawer`) voor directe controle van het broncitaat en de PDF-pagina.

---

## Architectuur & Tech Stack

| Laag | Technologieën |
| :--- | :--- |
| **Frontend** | React 18, TypeScript, Vite, NL Design System (`@nl-design-system-candidate/*`), Utrecht Component Library & Design Tokens |
| **Backend API** | Python 3.11+, FastAPI, Pydantic v2, SQLAlchemy 2.0, Alembic migraties |
| **Database & Opslag** | PostgreSQL (productie) / SQLite (test- & ontwikkelmodus), SHA-256 geverifieerde PDF-opslag |
| **Search & NLP** | Pinecone Vector Index, `BAAI/bge-m3` embeddings, lokale ONNX Cross-Encoder reranking, Reciprocal Rank Fusion (RRF), domeinspecifieke intent-mapping |
| **Kwaliteit & DevOps** | Pytest (241 geautomatiseerde unit- en integratietests), Docker & Docker Compose, GitLab CI/CD & Render Blueprint |

```text
├── gpo-assistent/
│   ├── backend/
│   │   ├── gpo_app/              # FastAPI applicatie, hybride zoekroute, citaatselectie & database-modellen
│   │   ├── gpo_local/            # PDF-ingestie, sectie-parser, bronregistratie & scope-validatie
│   │   ├── alembic/              # Database-migraties voor het GPO-bronregister
│   │   ├── tests_app/            # Integratietests voor het bronregister, API-routes en zoekselectie
│   │   └── tests_vector/         # Tests voor chunking, vector-indexering en evaluatiegates
│   ├── frontend/
│   │   ├── src/components/       # CaseCard, CitationDrawer en UI-componenten (NL Design System)
│   │   ├── src/pages/            # ZaakvergelijkerPage (zoeken, vergelijken en bronverificatie)
│   │   └── src/styles/           # SchoolRecht thema en NLDS styling
│   ├── docs/                     # Technische ontwerpdocumentatie en zoekevaluaties
│   └── sources/                  # Publieke broncatalogus met metadata en SHA-256 bestandshashes
├── docker-compose.yml            # Reproduceerbare containeropzet voor frontend + backend
└── render.yaml                   # Cloud deployment configuratie
```

---

## Lokaal Opstarten

### 1. Backend (FastAPI)

```bash
cd gpo-assistent/backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
uvicorn gpo_app.api:app --host 127.0.0.1 --port 8002 --reload
```

De OpenAPI-documentatie is lokaal beschikbaar op `http://127.0.0.1:8002/docs`.

### 2. Frontend (React + Vite)

```bash
cd gpo-assistent/frontend
npm install
npm run dev
```

Open `http://localhost:5173` in de browser. In ontwikkelmodus stuurt Vite alle `/api`-verzoeken automatisch door naar `http://127.0.0.1:8002`.

### 3. Testsuite Uitvoeren

```bash
cd gpo-assistent/backend
pytest -q tests_app/ tests_vector/
```

---

## Auteur & Verantwoording

Ontwikkeld door **Timi Fung** binnen **Studio Public Lab** aan de **Hogeschool van Amsterdam (HvA — HBO-ICT)**.

*Disclaimer: SchoolRecht is een informatief onderzoeksprototype op basis van openbare uitspraken van de Geschillencommissie Passend Onderwijs en vervangt geen individueel juridisch advies.*
