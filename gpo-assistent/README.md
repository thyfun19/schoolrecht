# SchoolRecht — Applicatiemap (`gpo-assistent`)

Deze map bevat de volledige broncode van **SchoolRecht** (backend, frontend, broncatalogus en ontwerpdocumentatie).

## Onderdelen

- **[`backend/`](backend/)**: FastAPI-service (`gpo_app`) en bronverwerkingspijplijn (`gpo_local`). Verzorgt PDF-extractie, sectieclassificatie (`commissieoverwegingen`, `conclusies`, `feiten`, `partijstandpunten`), hybride vector- en tekst-retrieval, en kaartselectie met paginaverificatie.
- **[`frontend/`](frontend/)**: React 18 + TypeScript single-page applicatie gebouwd met het **NL Design System** en Utrecht Design Tokens.
- **[`sources/catalogus.json`](sources/catalogus.json)**: Publieke broncatalogus met zaaknummers, adviesdatums, onderwerpen, officiële bronlinks en SHA-256 bestandshashes.
- **[`docs/`](docs/)**: Verantwoording van het technisch ontwerp, het PostgreSQL-bronregister en de evaluaties van de zoekkwaliteit op oudervragen.

Zie de hoofd-[`README.md`](../README.md) in de root van de repository voor installatie- en startinstructies.
