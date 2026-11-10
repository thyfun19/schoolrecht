"""Select an exact committee citation for the card, then rank the candidate cases."""

from collections import Counter, defaultdict
from math import log
from functools import lru_cache
from threading import Lock
from pathlib import Path
import re
import unicodedata

import sqlalchemy as sa

from . import models as m

STOP = set(
    """de het een en van voor op in is zijn wordt worden dat die dit met om te tot aan als bij door naar over of er ook niet wel kan moet heeft hebben was welke wat waarom hoe ik mijn deze dan school commissie leerling kind zaak wij onze ons werd nog al mee stond staat zegt moeten kunnen ooit komen werd zoon dochter vroeger nieuwe eerder eerdere andere ouders stukken wij krijgen kregen werkte werkt steeds belang alleen zegt voldoende onvoldoende onderbouwing maken vaker stond blijft terwijl onze sinds""".split()
)


# Dutch parent language and terms used in decisions. The same vocabulary is
# applied to questions and source text, independently of case identity.
CONCEPTS = tuple(
    re.compile(pattern)
    for pattern in (
        r"handelingsdeel|instemmingsrecht|instemmen|toestemming van de ouders",
        r"uitstroombestemming|uitstroomniveau|uitstroomperspectief|afstrom",
        r"evaluat|evalueer|gevalueer|geevalueer|cyclisch",
        r"groepsgrootte|klassengrootte|klasgrootte|volle klas|(?:klas|groep) (?:zit |is )?(?:al |helemaal )?vol\b|(?:grote|te grote) (?:klas|groep)|grootte van de (?:klas|groep)",
        r"financi|bekostig|budget|\bgeld\b|beschikbare middelen|middelen voor",
        r"thuiswerk|thuiszit|feitelijke? verwijder|toegang.{0,18}ontzeg",
        r"terugkeer|terugkeren|terug.{0,12}(?:komen|school)",
        r"toedracht(?! (?:niet|onbetwist))|feitenonderzoek|verschillende lezingen|ander verhaal|betrokkenen te horen|getuigenverklaring",
        r"aanmeld|aangemeld|aan te melden",
    )
)


def normalize(text):
    return "".join(
        c
        for c in unicodedata.normalize("NFD", text.casefold())
        if not unicodedata.combining(c)
    )


# Each entry links a parent question to the corresponding language in opinions.
# Separate triggers prevent a generic mention of school or OPP activating a topic.
PARENT_INTENTS = (
    (
        "bestemming",
        re.compile(
            r"uitstroombestemming|uitstroomperspectief|vervolgonderwijs|waar.{0,55}(?:onderwijs gaan volgen|onderwijs gaat volgen|terecht zal komen)"
        ),
        re.compile(
            r"(?:uitstroombestemming|uitstroomniveau|uitstroomperspectief).{0,90}(?:ontbreek|niet.{0,25}onderbouwd|stappen|kindgebonden)|(?:uitstroombestemming|uitstroomperspectief).{0,60}onderbouwing daarvan ontbreken"
        ),
        "concrete onderbouwing van de verwachte uitstroombestemming met kindgebonden factoren en benodigde stappen in het ontwikkelingsperspectief",
    ),
    (
        "inschrijving",
        re.compile(
            r"^(?=.*\bingeschreven\b)(?=.*(?:wie|welke taak|verantwoordelijk|wiens))"
        ),
        re.compile(
            r"^(?=.*(?:ingeschreven|inschrijving|school van herkomst))(?=.*(?:verantwoordelijk|verplichting|bevoegdheid))"
        ),
        "verantwoordelijkheid van de school van inschrijving voor onderwijs en ondersteuning",
    ),
    (
        "inbreng",
        re.compile(
            r"^(?=.*(?:opmerkingen|verbeterpunten|suggesties|onze reactie|onze inbreng))(?=.*(?:plan|opp\b|ontwikkelingsperspectief))"
        ),
        re.compile(
            r"opmerkingen.{0,55}(?:verwerk|weerleg|tegemoet|aangepast)|niet nader onderbouwd.{0,65}wensen van verzoek|opnieuw met verzoek.{0,30}in overleg"
        ),
        "hoe de school concrete opmerkingen en bezwaren van ouders op het ontwikkelingsperspectief heeft verwerkt of gemotiveerd weerlegd",
    ),
    (
        "plaatsruimte",
        re.compile(
            r"(?:klas|groep|school).{0,20}\bvol\b|(?:beschikbare )?plaats(?:en|ruimte).{0,25}aanmelding"
        ),
        re.compile(
            r"transparant toelatingsbeleid|hoeveel plaatsruimte er is en hoeveel aanmeldingen er zijn\.|geen plaatsruimte beschikbaar is.{0,55}verworpen|alsnog plaatsruimte beschikbaar komt|wachtlijst staat, ontstaat die verplichting"
        ),
        "toetsing van plaatsruimte, wachtlijst en een transparant toelatingsbeleid bij aanmelding",
    ),
    (
        "hoorplicht",
        re.compile(
            r"(?:wanneer|moment).{0,45}(?:ouders|wij|ons).{0,25}(?:verhaal|horen|gehoord|zeggen)|verhaal (?:mogen|kunnen) doen"
        ),
        re.compile(
            r"ouders.{0,25}moeten horen|voorafgaande hoorplicht|vereiste om de ouders te horen|formele verwijderingsprocedure dienen te starten"
        ),
        "de plicht van het schoolbestuur om ouders vooraf te horen bij een voornemen tot verwijdering of schorsing",
    ),
    (
        "opbouw",
        re.compile(
            r"(?:aangepast(?:e)? rooster|opbouwschema|minder uren|halve dagen).{0,45}terug|terug.{0,45}(?:aangepast(?:e)? rooster|opbouwschema|minder uren)"
        ),
        re.compile(
            r"opbouwschema|opbouw van (?:de )?onderwijstijd|(?:terugkeer|weer op school).{0,60}(?:opbouw|aangepast rooster|begeleid)"
        ),
        "beoordeling van een opbouwschema of aangepast rooster gericht op spoedige volledige terugkeer naar school",
    ),
    (
        "evaluatie",
        re.compile(
            r"(?:begeleiding|hulp|plan|opp).{0,55}(?:niet te helpen|werkt niet|resultaten.{0,25}besproken)"
        ),
        re.compile(
            r"(?:met (?:de )?ouders|schooljaar).{0,35}(?:geëvalueerd|geevalueerd|evalueren|evaluatie)|evaluatie van het opp|evaluatiemomenten met (?:de )?ouders"
        ),
        "evaluatie van het ontwikkelingsperspectief met de ouders en verslaglegging of bijstelling van de begeleiding",
    ),
    (
        "toelatingsonderzoek",
        re.compile(
            r"(?:hoe|welk)\b.{0,35}onderzocht.{0,45}(?:school|zelf kon bieden|toegelaten|aanmelding)|te veel hulp nodig.{0,45}toegelaten"
        ),
        re.compile(
            r"onderwijskundige stukken opgevraagd|alleen gebruik gemaakt van de aangeleverde dossierstukken|louter op dossieronderzoek|zonder de behoefte aan extra ondersteuning.{0,35}in kaart te brengen|op grond van diverse verslagen en adviezen de ondersteuningsbehoefte"
        ),
        "welke concrete onderzoeksstappen de school bij aanmelding heeft gezet om de ondersteuningsbehoefte en eigen mogelijkheden vast te stellen",
    ),
    (
        "toedracht",
        re.compile(
            r"(?:tegenstrijdige|verschillende|ander)\s+(?:verhalen|lezingen|verklaringen)|ruzie.{0,50}(?:verhalen|lezingen|wat er gebeurde)"
        ),
        re.compile(
            r"onderzoek.{0,25}toedracht|onvoldoende de feiten heeft onderzocht|consistent beeld te krijgen van de gebeurtenis|tegenstrijdige verklaringen"
        ),
        "onderzoek door de school naar de feitelijke toedracht van een incident en het horen van betrokkenen bij tegenstrijdige verklaringen",
    ),
    (
        "verzuimhulp",
        re.compile(
            r"(?:welke )?(?:hulp|begeleiding|ondersteuning).{0,35}\bverzuim\b|\bverzuim\b.{0,45}weer lessen"
        ),
        re.compile(
            r"(?<!in )(?<!dit )\bverzuim\b.{0,140}(?:begeleiding|ondersteuning|opdc|jeugdarts|leerplicht|gesprekken|opp|rooster)|begeleiding.{0,120}\bverzuim\b.{0,80}(?:jeugdarts|leerplicht|opdc|gesprekken)|planmatig wordt toegewerkt naar het weer volgen van het volledig aantal uren"
        ),
        "welke begeleiding, gesprekken of externe hulp de school bij schoolverzuim heeft ingezet om de leerling weer lessen te laten volgen",
    ),
)


def matching_intents(question):
    normalized = normalize(question)
    return [intent for intent in PARENT_INTENTS if intent[1].search(normalized)]


def lexical_terms(normalized):
    tokens = re.findall(r"[^\W_]+", normalized)
    # Prefixes also match Dutch inflections and compounds in a short card quote.
    words = {word[:6] for word in tokens if len(word) >= 3 and word not in STOP}
    return words | {
        f"topic:{i}" for i, pattern in enumerate(CONCEPTS) if pattern.search(normalized)
    }


def terms(text):
    normalized = normalize(text)
    return lexical_terms(normalized) | {
        f"topic:{name}"
        for name, _, evidence, _ in PARENT_INTENTS
        if evidence.search(normalized)
    }


def query_terms(question):
    return lexical_terms(normalize(question)) | {
        f"topic:{intent[0]}" for intent in matching_intents(question)
    }


def preview_candidates(question, cases, citations, count=1):
    query = query_terms(question)
    matched = []
    documents = defaultdict(set)
    for cite in citations:
        if len(cite["quote"].split()) < 6:
            continue
        overlap = query & terms(cite["quote"])
        if overlap:
            pid = cite["processing_id"]
            documents[pid].update(overlap)
            matched.append((pid, cite, overlap))
    frequency = Counter(t for words in documents.values() for t in words)
    weights = {t: log(1 + (len(cases) + 0.5) / (n + 0.5)) for t, n in frequency.items()}
    candidates = defaultdict(list)
    for pid, cite, overlap in matched:
        score = sum(
            weights[t] * (4 if t.startswith("topic:") else 1) for t in sorted(overlap)
        )
        candidates[pid].append((score, cite))
    options = {}
    best = {}
    for pid, hits in candidates.items():
        if count == 1:
            score, cite = max(hits, key=lambda hit: hit[0])
            best[pid], options[pid] = score, [cite]
            continue
        hits.sort(key=lambda hit: -hit[0])
        best[pid] = hits[0][0]
        unique = {}
        for _, cite in hits:
            unique.setdefault(cite["quote"], cite)
            if len(unique) == count:
                break
        options[pid] = list(unique.values())
    ordered = sorted(cases, key=lambda case: -best.get(case["processing_id"], 0))
    return [(case, options.get(case["processing_id"], [])) for case in ordered]


def ranked_previews(question, cases, citations):
    return [
        (case, options[0] if options else None)
        for case, options in preview_candidates(question, cases, citations)
    ]


@lru_cache(maxsize=1)
def local_card_model():
    path = Path(__file__).resolve().parents[1] / "data/model-comparison/models/reranker"
    if not (path / "model.onnx").is_file():
        return None
    from .reranking_experiment import LocalReranker

    return LocalReranker(path)


model_lock = Lock()


def rank_cards(question, candidates, model, limit):
    """Rank cases once, then refine only the quotations the user will see."""
    if model is None:
        return [
            (case, options[0] if options else None)
            for case, options in candidates[:limit]
        ]
    glosses = [intent[3] for intent in matching_intents(question)]
    if glosses:
        question = question + " (" + ", ".join(glosses) + ")"

    def score(cite):
        model.tokenizer.no_padding()
        if len(model.tokenizer.encode(question, cite["quote"]).ids) > 512:
            return float("-inf")
        return float(model.score([dict(question=question, passage=cite["quote"])])[0])

    ranked = []
    with model_lock:
        for case, options in candidates:
            first_score = score(options[0]) if options else float("-inf")
            ranked.append((first_score, case, options))
        ranked.sort(key=lambda row: -row[0])
        cards = []
        for best_score, case, options in ranked[:limit]:
            best = options[0] if options else None
            for alternative in options[1:]:
                value = score(alternative)
                if value > best_score:
                    best_score, best = value, alternative
            cards.append((case, best))
    return cards


def semantic_cards(question, ranked, model, limit=None):
    intents = matching_intents(question)
    if intents and model is not None:
        matched = [
            (case, preview)
            for case, preview in ranked
            if preview
            and any(intent[2].search(normalize(preview["quote"])) for intent in intents)
        ]
        if matched and len(matched) < len(ranked):
            matched_ids = {case["processing_id"] for case, _ in matched}
            others = [
                (case, preview)
                for case, preview in ranked
                if case["processing_id"] not in matched_ids
            ]
            first = rank_cards(question, [(c, [p]) for c, p in matched], model, limit)
            remaining = None if limit is None else limit - len(first)
            if remaining == 0:
                return first
            return first + rank_cards(
                question, [(c, [p] if p else []) for c, p in others], model, remaining
            )
    candidates = [(case, [preview] if preview else []) for case, preview in ranked]
    return rank_cards(question, candidates, model, limit)


def window_excerpt(question, quote, width=240):
    if len(quote) <= width:
        return quote
    query = query_terms(question)
    best = None
    for word in re.finditer(r"\S+", quote):
        start = word.start()
        end = min(start + width, len(quote))
        if end < len(quote) and not quote[end].isspace():
            end = quote.rfind(" ", start, end)
        if end <= start:
            continue
        overlap = query & terms(quote[start:end])
        score = (sum(4 if t.startswith("topic:") else 1 for t in overlap), end - start)
        if best is None or score > best[0]:
            best = score, start, end
    if best is None:
        return quote[:width] + "…"
    _, start, end = best
    return (
        ("…" if start else "")
        + quote[start:end].strip()
        + ("…" if end < len(quote) else "")
    )


def excerpt(question, quote, width=None):
    # Most registered citations are one sentence. Keep its conclusion visible
    # when the complete source fits in a short paragraph on the card.
    if width is None and len(quote) <= 480:
        return quote
    return window_excerpt(question, quote, 240 if width is None else width)


PLAN = re.compile(r"\b(?:opps?|ontwikkelingsperspectief|(?:ondersteunings)?plan)\b")
REVISION = re.compile(r"bijstel|bijgestel|aanpas|aangepast|actualis|herzien|wijzig")
ENROLMENT = re.compile(r"ingeschreven|inschrijving|school van herkomst")
LOCATION = re.compile(
    r"elders|andere (?:school|locatie|voorziening)|buiten.{0,25}school|proefplaatsing|overstap"
)
DUTY = re.compile(r"verantwoordelijk|bevoegd(?:heid| gezag)|verplichting")


def asks_about_plan_duty(question):
    text = normalize(question)
    return all(
        pattern.search(text) for pattern in (PLAN, REVISION, ENROLMENT, LOCATION)
    ) and bool(re.search(r"\bwie\b|verantwoordelijk|wiens taak", text))


def plan_duty_passages(rows):
    """Require the action and the continuing duty together in the same source section."""
    sections = defaultdict(list)
    for row in rows:
        sections[row["processing_id"], row["page_number"], row["section_key"]].append(
            row
        )
    found = {}
    for (pid, _, _), section in sections.items():
        texts = [normalize(row["quote"]) for row in section]
        for i, text in enumerate(texts):
            if not (PLAN.search(text) and REVISION.search(text) and DUTY.search(text)):
                continue
            if any(
                ENROLMENT.search(other)
                and DUTY.search(other)
                and LOCATION.search(other)
                for other in texts[max(0, i - 1) : i + 2]
            ):
                found.setdefault(pid, section[i])
    return found


def search_cards(question, cases, rows, model, limit):
    if not asks_about_plan_duty(question):
        return semantic_cards(
            question, ranked_previews(question, cases, rows), model, limit
        )
    rows = list(rows)
    strong = plan_duty_passages(rows)
    ranked = ranked_previews(question, cases, rows)
    complete = [
        (case, strong[case["processing_id"]])
        for case, _ in ranked
        if case["processing_id"] in strong
    ]
    others = [
        (case, cite) for case, cite in ranked if case["processing_id"] not in strong
    ]
    first = semantic_cards(question, complete, model, limit)
    remaining = None if limit is None else limit - len(first)
    if remaining == 0:
        return first
    return first + semantic_cards(question, others, model, remaining)


def select_cards(con, question, cases, limit=None):
    ids = [c["processing_id"] for c in cases]
    rows = con.execute(
        sa.select(
            m.citations.c.processing_id,
            m.citations.c.source_id,
            m.citations.c.quote,
            m.sections.c.page_number,
            m.sections.c.heading,
            m.sections.c.section_key,
        )
        .select_from(m.citations.join(m.sections))
        .where(
            m.citations.c.processing_id.in_(ids),
            m.sections.c.role == "commissie",
            m.sections.c.category == "commissieoverwegingen",
        )
        .order_by(
            m.citations.c.processing_id,
            m.sections.c.ordinal,
            m.citations.c.start_offset,
        )
    ).mappings()
    ordered = []
    for case, preview in search_cards(question, cases, rows, local_card_model(), limit):
        if preview:
            case["preview"] = dict(preview, excerpt=excerpt(question, preview["quote"]))
        ordered.append(case)
    return ordered
