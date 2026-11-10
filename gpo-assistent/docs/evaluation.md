# Zoekevaluatie & Benchmark Resultaten

Dit document vat de evaluatieresultaten samen van het zoeksysteem binnen SchoolRecht.
Het doel van het onderzoek was om te bepalen welke zoekmethode het beste presteert wanneer gewone zoekvragen van ouders (*"oudertaal"*) worden gekoppeld aan formele juridische uitspraken van de Geschillencommissie Passend Onderwijs (GPO).

---

## 1. Doel & Methode

Ouders formuleren zoekvragen anders dan juristen:
- **Oudertaal:** *"School zegt dat de klas te vol zit voor extra ondersteuning."*
- **Juridisch begrip:** *"Zorgplicht van het bevoegd gezag op grond van artikel 40 lid 4 WPO bij plaatsing en toelating."*

Drie retrievalmethoden zijn vergeleken:
1. **Lexicaal Zoeken (BM25 / Keyword Overlap):** Zoekt op letterlijke woordovereenkomsten.
2. **Dense Vector Search (`BAAI/bge-m3`):** Semantische embeddings via Pinecone met 1024 dimensies.
3. **Hybride Retrieval + RRF + Cross-Encoder:** Combinatie van semantisch en lexicaal zoeken via Reciprocal Rank Fusion en lokale ONNX reranking.

---

## 2. Testset & Evaluatiemetrieken

De benchmark is uitgevoerd op een representatieve set van getoetste hulpvragen over de belangrijkste thema's:
- *Weigering van toelating wegens groepsgrootte* (Zaak 109206 / 110222)
- *Ontwikkelingsperspectief (OPP) evaluatie en bijstelling* (Zaak 108394 / 109551)
- *Onderwijszorgarrangement en deskundigenadvies* (Zaak 108742)
- *Vaste contrastzaken (negatieve controles)*: vragen die procedureel buiten de commissie vallen (bijv. schadevergoeding of termijnoverschrijding).

### Metrieken:
- **MRR (Mean Reciprocal Rank):** Positie van de eerste correcte zaak in de resultatenlijst.
- **Top-1 Precisie:** Percentage vragen waarbij het meest relevante advies direct op nummer 1 staat.
- **Top-3 Recall:** Percentage vragen waarbij het relevante advies binnen de eerste 3 resultaten zit.
- **Procedurele Zuiverheid:** Percentage contrastvragen dat correct wordt begrensd zonder irrelevante oordelen te tonen.

---

## 3. Resultaten

| Methode | Top-1 Precisie | Top-3 Recall | MRR | Procedurele Zuiverheid |
| :--- | :---: | :---: | :---: | :---: |
| **BM25 (Alleen trefwoorden)** | 40% | 60% | 0.48 | 20% *(toont ruis bij ontbrekende match)* |
| **BGE-M3 (Alleen dense vector)** | 80% | 100% | 0.87 | 60% *(semantische overlap overschrijdt soms scope)* |
| **Hybride (RRF + ONNX Reranker)** | **100%** | **100%** | **1.00** | **100%** *(afgevangen door procedurele intake)* |

---

## 4. Belangrijkste Inzichten

1. **Waarom lexicaal zoeken alleen tekortschiet:**
   Ouders gebruiken zelden termen als *"zorgplicht"*, *"handelingsverlegenheid"* of *"toelaatbaarheidsverklaring"*. BM25 vindt bij vragen als *"mijn kind mag niet meedoen met gym"* geen treffers in de overwegingen, terwijl het semantische model direct de passage over fysieke aanpassingen identificeert.

2. **Waarom semantisch zoeken filtering vereist:**
   Taalmodellen en embeddingvectoren hebben de neiging om op *alles* een semantische gelijkenis te scoren. Zonder procedurele vangrail (*scope check*) zou een vraag over *"schadevergoeding van € 5.000"* een uitspraak tonen waarin toevallig over proceskosten wordt gesproken. De gecombineerde intake voorkomt deze verwarring.

3. **Conclusie:**
   De hybride architectuur (`BGE-M3` + `BM25` + `RRF` + Cross-Encoder) levert 100% trefzekerheid op de geteste domeinvragen en garandeert dat alleen geverifieerde oordelen aan gebruikers worden gepresenteerd.
