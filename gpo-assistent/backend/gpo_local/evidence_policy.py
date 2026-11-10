"""Classify existing citation units without rewriting text or changing source identity."""

from collections import defaultdict
import re

SIGNATURE_START = re.compile(
    r"\b(?:aldus\s+(?:gedaan|vastgesteld)|vastgesteld\s+te|ondertekend\s+te)\b", re.I
)
FORMAL_DECISION = re.compile(
    r"\bcommissie\b.*\bverzoek(?:er|ers|ster)?\b.*\b(?:ongegrond|gegrond|niet[\s-]ontvankelijk)\b",
    re.I | re.S,
)


def classify_sources(sources):
    groups = defaultdict(list)
    for source in sources:
        groups[(source["case_number"], source["section_id"])].append(source)
    classified = []
    for (case, section_id), units in groups.items():
        boundaries = [
            s["start_offset"] + match.start()
            for s in units
            if s["category"] == "conclusies"
            for match in SIGNATURE_START.finditer(s["quote"])
        ]
        cutoff = min(boundaries) if boundaries else None
        for source in units:
            if source["role"] != "commissie":
                function = "non_committee"
            elif source["category"] == "commissieoverwegingen":
                function = "reasoning"
            elif cutoff is not None and source["end_offset"] > cutoff:
                function = "administrative_or_mixed"
            elif source["section"].upper() in [
                "OORDEEL",
                "ADVIES",
            ] and FORMAL_DECISION.search(source["quote"]):
                function = "formal_decision"
            elif source["section"].upper() == "AANBEVELINGEN":
                function = "recommendation"
            else:
                function = "unclassified_conclusion"
            argument = function in ["reasoning", "recommendation"]
            classified.append(
                dict(
                    source_id=source["source_id"],
                    case_number=case,
                    section_id=section_id,
                    function=function,
                    argument_eligible=argument,
                    basis_eligible=argument or function == "formal_decision",
                    boundary_straddled=cutoff is not None
                    and source["start_offset"] < cutoff < source["end_offset"],
                    administrative_start=cutoff,
                )
            )
    return classified
