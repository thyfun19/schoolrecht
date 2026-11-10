"""Build a reserved manual review set from existing paid output. Never scores meaning."""

import json
import re
from pathlib import Path

from .common import digest, norm, read
from .retrieval import Retrieval
from .source_registry import SourceRegistry

BASE = Path("local-results")


def raw_text(path):
    raw = read(path)
    return json.loads(
        "".join(
            c["text"]
            for item in raw["output"]
            if item.get("type") == "message"
            for c in item["content"]
            if c.get("type") == "output_text"
        )
    )


def build(output=Path("experiments/reserved-review")):
    engine = Retrieval(Path("data/e5"), Path("data/gpo.sqlite3"))
    registry = SourceRegistry(engine.documents, engine.sections)
    all_sources = registry.select(list(engine.documents))
    lookup = {s["source_id"]: s for s in all_sources}
    questions = read(BASE / "question-focus/search-comparison.json")["results"]
    question_by_id = {q["id"]: q["question"] for q in questions}
    pairs = []
    input_hashes = {}

    def v2(qid, case, index):
        path = BASE / "grounding-v2-trial" / f"{qid}-per_case-parsed.json"
        data = read(path)
        claim = next(
            c
            for c in data["cases"]
            if lookup[c["case_source_id"]]["case_number"] == case
        )["conclusions"][index]
        input_hashes[str(path)] = digest(path)
        return (
            claim["text"],
            claim["source_ids"],
            dict(version="v2", path=str(path), case=case, claim_index=index),
        )

    def v1(qid, category, index):
        path = BASE / "e5/answers" / f"{qid}-raw.json"
        claim = raw_text(path)[category][index]
        input_hashes[str(path)] = digest(path)
        ids = []
        mapping_notes = []
        for citation_index, citation in enumerate(claim["citations"]):
            case, page = re.match(r"(.+)-p(\d+)-", citation["source_id"]).groups()
            matches = [
                s
                for s in all_sources
                if s["case_number"] == case
                and s["page"] == int(page)
                and norm(citation["quote"]).lower() in norm(s["quote"]).lower()
            ]
            if not matches and (qid, category, index, citation_index) == (
                "N3",
                "similarities",
                1,
                0,
            ):
                # Explicit manual alignment, never approximate automatic quote repair.
                # Original says "de school"; source says "een school". Preserve original.
                matches = [
                    s
                    for s in all_sources
                    if s["case_number"] == case
                    and s["page"] == int(page)
                    and "een school vóór het vaststellen van een opp" in s["quote"]
                ]
                mapping_notes.append(
                    "Original v1 quote is nonliteral (de school / een school). Common contrast evidence manually aligned to the canonical source; original retained unchanged."
                )
            if len(matches) != 1:
                raise ValueError("V1-citaat niet eenduidig canoniek te koppelen.")
            ids.append(matches[0]["source_id"])
        return (
            claim["text"],
            list(dict.fromkeys(ids)),
            dict(
                version="v1",
                path=str(path),
                category=category,
                claim_index=index,
                original_citations=claim["citations"],
                mapping="Same-case same-page canonical units; explicit manual exception recorded",
                mapping_notes=mapping_notes,
            ),
        )

    def contextual(case, needle):
        return next(
            s["source_id"]
            for s in all_sources
            if s["case_number"] == case and needle in s["quote"].lower()
        )

    def add(
        pid, qid, original, kind, why, condition, good, decision="beperken", extra=None
    ):
        text, original_ids, origin = original
        extra = [sid for sid in (extra or []) if sid not in original_ids]
        ids = list(dict.fromkeys(original_ids + extra))
        # Both variants get exactly the same real IDs: structural checks cannot judge meaning.
        bad = dict(text=text, source_ids=ids)
        improved = dict(text=good, source_ids=ids)
        assert bad["source_ids"] == improved["source_ids"]
        sources = [registry.get(sid, set(lookup)) for sid in ids]
        assert all(s["role"] == "commissie" for s in sources)
        pairs.append(
            dict(
                id=pid,
                question_id=qid,
                question=question_by_id[qid],
                origin=origin,
                original_source_ids=original_ids,
                added_context_ids=extra or [],
                original_claim=bad,
                manual_reference_variant=improved,
                canonical_evidence=sources,
                issue_type=kind,
                why_insufficient=why,
                reviewer_must_notice=condition,
                expected_original_decision=decision,
                expected_reference_decision="accepteren_als_begrensde_historische_bijdrage_of_bewijsgrens",
                structural_check=dict(
                    both_have_known_ids=True,
                    same_ids=True,
                    semantic_verdict_not_computed=True,
                ),
            )
        )

    add(
        "P1",
        "V1",
        v2("V1", "46731", 0),
        "ontbrekende uitzondering / historische reikwijdte",
        "De onderzoeksplicht staat in het historische advies, maar de losse formulering bewaart de uitzondering op aanpassingen niet en is geen actuele persoonlijke toets.",
        "Doeltreffende aanpassingen zijn niet onbegrensd: onevenredige belasting is uitdrukkelijk genoemd.",
        "In deze zaak vond de commissie dat de school moest onderzoeken of zij extra ondersteuning kon bieden, al dan niet met middelen van het samenwerkingsverband; het aangehaalde kader zonderde aanpassingen met een onevenredige belasting uit.",
        extra=[contextual("46731", "onevenredige belasting")],
    )
    add(
        "P2",
        "V1",
        v2("V1", "46731", 1),
        "historische bevinding als algemene verplichting",
        "De claim is niet als rechtsregel onwaar verklaard, maar de huidige bronnen bewijzen geen algemene actuele onderzoekslijst voor de ouder.",
        "Het oordeel betreft ontoereikend onderzoek in dit dossier; geen zelfstandig oordeel over een nieuwe weigering.",
        "In deze zaak vond de commissie dat de school de aard van de ondersteuningsbehoefte onvoldoende gedegen had onderzocht.",
    )
    add(
        "P3",
        "V1",
        v2("V1", "46731", 3),
        "casusvoorwaarde onvoldoende afgebakend",
        "De leerling in deze zaak zat langdurig thuis; het onderzoekspunt mag niet als stap in elke mogelijke aanmelding worden gepresenteerd.",
        "Behoud het verband met de lange periode van thuiszitten en de concrete bevinding dat dit niet onderzocht was.",
        "In deze zaak vond de commissie dat niet was gebleken dat de school de invloed van de lange periode van thuiszitten op het functioneren in een schoolse setting had onderzocht.",
    )
    add(
        "P4",
        "N3",
        v2("N3", "110651", 0),
        "bevoegdheid verward met rechtvaardiging / irrelevant vraagdeel",
        "Een wettelijke bevoegdheid is geen goedkeuring van deze verwijdering; de latere noodzakelijkheidstoets was juist ongunstig voor de school. Dit draagt bovendien niet bij aan externe OPP-verantwoordelijkheid.",
        "Onderscheid bevoegdheid, noodzaak en de concrete OPP-vraag.",
        "In deze zaak vond de commissie dat verweerder naar de strikte letter van de wet kon beslissen tot verwijdering, maar niet voldoende aannemelijk had gemaakt dat verwijdering noodzakelijk was. Deze verwijderingsafweging beantwoordt de vraag naar een externe OPP-opsteller niet.",
        decision="afwijzen_als_antwoord_op_N3",
        extra=[
            contextual("110651", "niet voldoende aannemelijk geworden dat verwijdering")
        ],
    )
    add(
        "P5",
        "N3",
        v2("N3", "123719", 0),
        "juiste algemene passage beantwoordt specifieke vraag niet",
        "De aangehaalde vaststellingsregel is op zichzelf weergegeven, maar behandelt geen verantwoordelijkheidsverdeling wanneer een externe voorziening de opsteller is.",
        "Markeer onvoldoende relevantie voor het gevraagde verband; verzin geen externe opsteller of overdracht.",
        "In deze zaak vond de commissie het aangehaalde kader voor OPP-vaststelling van toepassing. Dit citaat onderbouwt niet wie verantwoordelijk bleef wanneer een externe onderwijsvoorziening het OPP maakte.",
        decision="afwijzen_als_antwoord_op_N3",
    )
    add(
        "P6",
        "N3",
        v1("N3", "similarities", 1),
        "ongefundeerde gevolgtrekking uit ouderoverleg",
        "Overleg- en evaluatievereisten bewijzen niet hoe verantwoordelijkheid bij een externe opsteller verdeeld was.",
        "Ouderbetrokkenheid en externe opstellersverantwoordelijkheid zijn verschillende relaties.",
        "In deze historische zaken werd overleg met ouders bij OPP-vaststelling of bijstelling genoemd. Daaruit kan de verantwoordelijkheid bij een externe OPP-opsteller niet worden afgeleid.",
        decision="afwijzen",
    )
    add(
        "P7",
        "N3",
        v1("N3", "differences", 1),
        "ontbrekend bewijs ingevuld met verantwoordelijkheidsoordeel",
        "Gebreken bij vaststelling en evaluatie dragen niet zonder meer de gevolgtrekking over overdracht of behoud van verantwoordelijkheid bij een externe voorziening.",
        "Ontbrekend bewijs voor overdracht is geen bewijs voor het gevraagde verantwoordelijkheidsverband.",
        "De aangehaalde overwegingen betreffen gebreken bij OPP-vaststelling en evaluatie. Met deze citaten kan niet worden vastgesteld wie verantwoordelijk bleef bij een externe OPP-opsteller.",
        decision="afwijzen",
    )
    add(
        "P8",
        "V4",
        v1("V4", "differences", 2),
        "bron uit derde zaak bewijst geen feit in eerste zaak",
        "De claim verbindt opstelling en evaluatie in één van twee zaken, terwijl het citaat over ouderbetrokkenheid uit de derde zaak 109493 komt. Dat bewijst geen evaluatie in 109206.",
        "Zet ieder feit bij zijn eigen zaak; noem de derde zaak expliciet of trek de evaluatieclaim in.",
        "In 109206 was een OPP opgesteld door de bovenschoolse voorziening. Het citaat over ouderbetrokkenheid komt uit de andere zaak 109493 en bewijst geen evaluatie in 109206. In 126525 was een extra ondersteuningsbehoefte tot begin schooljaar 2024-2025 niet gebleken.",
        decision="beperken",
    )
    assert len(pairs) == 8
    result = dict(
        label="Voorlopige handmatige contrastparen van dezelfde AI; geen nieuwe modeluitvoer of onafhankelijke kwaliteitsmeting",
        intended_use="Reserved for future review evaluation; do not include these examples in review-prompt development",
        paid_calls=0,
        semantic_scoring_implemented=False,
        source_mapping_not_a_repair_of_original_output=True,
        extra_context_policy="Both contrast variants share canonical original IDs plus explicitly listed context IDs. Original paid outputs and selected IDs remain unchanged.",
        input_sha256=input_hashes,
        pairs=pairs,
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "semantic-pairs.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Acht voorlopige inhoudelijke contrastparen",
        "",
        result["label"],
        "",
        "Geldige IDs bewijzen geen bronsteun. De referentievariant is handmatig samengesteld, geen gerepareerde modeluitvoer.",
        "",
    ]
    for p in pairs:
        lines += [
            f"## {p['id']} — {p['issue_type']}",
            "",
            f"Vraag: {p['question']}",
            "",
            f"Oorspronkelijk: {p['original_claim']['text']}",
            "",
            f"Handmatige referentie: {p['manual_reference_variant']['text']}",
            "",
            f"Waarom: {p['why_insufficient']}",
            "",
            f"Te herkennen: {p['reviewer_must_notice']}",
            "",
            f"Beslissing origineel: {p['expected_original_decision']}; referentie: {p['expected_reference_decision']}.",
            "",
            "Beide varianten hebben dezelfde canonieke bron-IDs. Eventuele extra context staat afzonderlijk in de JSON; deze was niet achteraf aan de modeluitvoer toegevoegd.",
            "",
        ]
        lines += [f"Herkomst: {p['origin']}", ""]
        if p["added_context_ids"]:
            lines += [
                f"Aanvullende context voor beide contrastvarianten, niet voor het oorspronkelijke modelantwoord: {p['added_context_ids']}",
                "",
            ]
        for s in p["canonical_evidence"]:
            lines += [
                f"- [{s['case_number']}, fysieke p.{s['page']}]({s['source_url']}) `{s['source_id']}` — {s['quote']}"
            ]
        lines += [""]
    (output / "semantic-pairs.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            dict(
                pair_count=len(pairs),
                paid_calls=0,
                both_variants_known_same_ids=True,
                input_sha256=input_hashes,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    build()
