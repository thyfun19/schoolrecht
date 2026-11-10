"""Offline reference scenarios, explicitly not newly generated model answers."""

import json
from pathlib import Path

from .common import read, digest
from .retrieval import Retrieval
from .source_registry import SourceRegistry
from .question_grounding import (
    prepare_question_assessment,
    hydrate_question_assessment,
    assemble_reviewed_explanation,
)


def scenario(qid, prepared):
    sources = prepared["sources"]

    def sid(case, needle):
        return next(
            s["source_id"]
            for s in sources
            if s["case_number"] == case and needle in s["quote"].lower()
        )

    cases = []
    for marker in prepared["case_markers"]:
        case = next(s["case_number"] for s in sources if s["source_id"] == marker)
        if qid == "N3" and case == "109206":
            finding = "In deze zaak vond de commissie dat het bevoegd gezag van de school van inschrijving verantwoordelijk bleef voor het OPP."
            refs = [sid(case, "verantwoordelijk voor het opp")]
            limitation = "Dit gold bij plaatsing op een bovenschoolse voorziening, die het OPP had opgesteld."
            limitrefs = [
                sid(case, "opgesteld door de bovenschoolse voorziening"),
                sid(case, "verantwoordelijk voor het opp"),
            ]
            part = "Wie verantwoordelijk bleef voor het OPP bij een bovenschoolse opsteller."
        elif qid == "V1" and case == "46731":
            finding = "In deze zaak vond de commissie dat de school de ondersteuningsbehoefte onvoldoende onderzocht en de beschikbare ondersteuningsmiddelen onvoldoende meegewogen had."
            refs = [
                sid(case, "onvoldoende gedegen onderzoek"),
                sid(case, "aanwending van deze beschikbare middelen"),
            ]
            limitation = "Het betrof deze toelatingsweigering na langdurig thuiszitten; de school baseerde zich mede op verouderde informatie."
            limitrefs = [
                sid(case, "verouderde informatie"),
                sid(case, "lange periode van thuiszitten"),
            ]
            part = "Welk onderzoek en welke afweging bij de weigering van extra ondersteuning ontbraken."
        elif qid == "V1" and case == "110222":
            finding = "In deze zaak vond de commissie dat de school gedegen onderzoek naar de ondersteuningsbehoefte had aangetoond."
            refs = [sid(case, "gedegen onderzoek")]
            limitation = "Het oordeel betrof een leerling voor wie de onderwijssetting nog een te grote stap was bij de opbouw van schoolgang."
            limitrefs = [sid(case, "een te grote stap")]
            part = "Een contrasterende zaak waarin de commissie het onderzoek voldoende vond."
        else:
            needles = {
                "109493": "niet periodiek",
                "122465": "niet gelijk te stellen",
                "110651": "verwijdering noodzakelijk",
                "123719": "procedurele vereisten",
                "126525": "basisondersteuning overstijgende",
            }
            reason = (
                "De aangeleverde overwegingen behandelen OPP-evaluatie of een verwijderings-/vaststellingsgeschil en onderbouwen de specifieke verantwoordelijkheid bij een externe OPP-opsteller niet."
                if qid == "N3"
                else "De aangeleverde overwegingen betreffen een ander geschil en geven geen directe bijdrage over het onderzoek bij deze vraag naar toelatingsweigering."
            )
            cases.append(
                dict(
                    status="insufficient_relevance",
                    case_source_id=marker,
                    reason=reason,
                    question_part_not_answered=prepared["question"],
                    basis_source_ids=[sid(case, needles[case])],
                )
            )
            continue
        cases.append(
            dict(
                status="relevant_contribution",
                case_source_id=marker,
                answers_question_part=part,
                finding=dict(text=finding, source_ids=refs),
                limitations=[dict(text=limitation, source_ids=limitrefs)],
            )
        )
    relevant = any(c["status"] == "relevant_contribution" for c in cases)
    unanswered = []
    if not relevant:
        unanswered = [
            "Met de opgehaalde commissieoverwegingen kan niet worden vastgesteld wie verantwoordelijk bleef wanneer een externe onderwijsvoorziening het OPP maakte."
        ]
    elif qid == "V1":
        unanswered = [
            "Welke onderzoekshandelingen in een nieuwe persoonlijke situatie nodig zijn, kan met deze historische selectie niet worden vastgesteld."
        ]
    return dict(cases=cases, unanswered_question_parts=unanswered)


def check(output=Path("local-results/question-focus")):
    engine = Retrieval(Path("data/e5"), Path("data/gpo.sqlite3"))
    registry = SourceRegistry(engine.documents, engine.sections)
    comparison = read(output / "search-comparison.json")
    rows = []
    packets = []
    examples = []
    old = Path("local-results/grounding-v2-trial")
    for q in [q for q in comparison["results"] if q["id"] in ["V1", "N3"]]:
        for method in ["text", "vector"]:
            response = engine.search(q["question"], method)
            prepared = prepare_question_assessment(q["question"], response, registry)
            payload = scenario(q["id"], prepared)
            hydrated = hydrate_question_assessment(payload, prepared, registry)
            reviews = {
                a["assessment_id"]: dict(
                    verdict="supported",
                    reviewer="AI reference-scenario author; not independent",
                    note="Handmatige bronselectie voor een offline referentiescenario, geen modeluitvoer of modelkwaliteitsmeting.",
                )
                for a in hydrated["case_assessments"]
            }
            result = assemble_reviewed_explanation(hydrated, reviews)
            assert result["audit"]["assessed_cases"] == len(response["results"])
            rows.append(
                dict(
                    id=q["id"],
                    method=method,
                    assessed=len(payload["cases"]),
                    included=[
                        v["case_number"] for v in result["explanation"]["contributions"]
                    ],
                    excluded=[
                        v["case_number"] for v in result["audit"]["excluded_cases"]
                    ],
                    evidence_status=result["evidence_status"],
                )
            )
            examples.append(
                dict(
                    id=q["id"],
                    method=method,
                    synthetic_reference_not_model_output=True,
                    payload=payload,
                    result=result,
                )
            )
            if method == "text":
                packets.append(
                    dict(id=q["id"], question=q["question"], prepared=prepared)
                )
            if method == "vector":
                assert (
                    prepared["sources"]
                    == read(old / f"{q['id']}-per_case-prepared.json")["sources"]
                )
                rows[-1]["fixed_old_source_context_identical"] = True
                try:
                    hydrate_question_assessment(
                        read(old / f"{q['id']}-per_case-parsed.json"),
                        prepared,
                        registry,
                    )
                except ValueError:
                    rows[-1]["old_v2_output_rejected_as_not_question_focused"] = True
                else:
                    raise AssertionError("Oude uitvoer is geen geldig v3-resultaat.")
    output.mkdir(parents=True, exist_ok=True)
    result = dict(
        paid_calls=0,
        model_quality_not_measured=True,
        default_search_unchanged="text",
        results=rows,
        retest_first_stage_max_usd=sum(
            p["prepared"]["estimated_max_cost_usd"] for p in packets
        ),
    )
    (output / "offline-check.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (output / "reference-scenarios.json").write_text(
        json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "proposed-retest-packets.json").write_text(
        json.dumps(packets, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "proposed-retest-provenance.json").write_text(
        json.dumps(
            dict(
                status="not_approved_not_run",
                packets_sha256=digest(output / "proposed-retest-packets.json"),
                model=packets[0]["prepared"]["request"]["model"],
                paid_calls=0,
                limit_candidate_usd=0.12,
                max_calls_candidate=2,
                no_retries=True,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    check()
