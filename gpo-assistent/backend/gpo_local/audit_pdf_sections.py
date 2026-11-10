"""Read-only PDF-to-registry completeness audit. No corpus/index rewriting."""

import collections
import difflib
import json
import re
from pathlib import Path

from pypdf import PdfReader

from .common import digest, norm, read


def audit(output=Path("local-results/pdf-audit")):
    corpus = read(Path("data/e5/corpus.json"))
    prior = {
        (p["case"], p["page"]): p
        for p in read(Path("../../onderzoek_gpo/data/pages.json"))
    }
    rows = []
    output.mkdir(parents=True, exist_ok=True)
    for doc in corpus["documents"]:
        case = doc["case_number"]
        path = Path("../../onderzoek_gpo/data/pdf") / (case + ".pdf")
        assert digest(path) == doc["sha256"]
        reader = PdfReader(path)
        assert len(reader.pages) == doc["pages"]
        pages = []
        lines = []
        for number, page in enumerate(reader.pages, 1):
            raw = page.extract_text() or ""
            stored = prior[(case, number)]
            sections = [
                s
                for s in corpus["sections"]
                if s["case"] == case and s["page"] == number
            ]
            # Audit the existing cleaning policy; preserve every substantive difference for review.
            clean = []
            removed = []
            headings = []
            for line in raw.splitlines():
                if re.search(
                    r"pagina \d+ van \d+|advies d\.d\.|zaak \d+ \| d\.d\.", line, re.I
                ) or re.fullmatch(
                    r"\s*(?:Geschillencommissie(?: passend)?|passend onderwijs|onderwijs|Stichting Onderwijsgeschillen)\s*",
                    line,
                    re.I,
                ):
                    removed.append(line)
                else:
                    clean.append(line)
            clean = "\n".join(clean).strip()
            content = []
            for line in clean.splitlines():
                is_heading = re.fullmatch(r"\s*\d+\.\s*([A-ZÁÉ][A-ZÁÉ -]{4,})\s*", line)
                is_actor = re.fullmatch(
                    r"\s*(?:(?:Standpunt\s+)?(?:Verzoekers?|Verzoekster|Verweerder|verzoekers?|verzoekster|verweerder)|Mening van de leerling)\s*",
                    line,
                )
                if is_heading:
                    headings.append(is_heading[1])
                elif not is_actor:
                    content.append(line)
            actual = norm(" ".join(s["text"] for s in sections))
            expected = norm("\n".join(content))
            difference = []
            if actual != expected:
                difference = list(
                    difflib.unified_diff(
                        expected.split(),
                        actual.split(),
                        fromfile="PDF content",
                        tofile="registered sections",
                        n=5,
                    )
                )
            normalized = norm(clean)
            spans = []
            cursor = 0
            for s in sections:
                position = normalized.find(s["text"], cursor)
                spans.append(
                    dict(
                        section_id=s["section_id"],
                        role=s["role"],
                        section=s["section"],
                        found_once_in_order=position >= cursor,
                        start=position,
                        end=position + len(s["text"]),
                    )
                )
                if position >= cursor:
                    cursor = position + len(s["text"])
            pages.append(
                dict(
                    page=number,
                    raw_identical_to_prior=raw == stored["raw"],
                    clean_identical_to_prior=clean == stored["text"],
                    all_content_equal=actual == expected,
                    removed_header_footer_lines=removed,
                    headings=headings,
                    section_spans=spans,
                    differences=difference,
                )
            )
            lines += [f"\n=== PHYSICAL PAGE {number} ===\n", raw]
        commission = [
            s
            for s in corpus["sections"]
            if s["case"] == case and s["role"] == "commissie"
        ]
        reasoning = [s for s in commission if "OVERWEGINGEN" in s["section"]]
        transitions = [
            dict(
                from_page=a["page"],
                to_page=b["page"],
                tail=a["text"][-180:],
                head=b["text"][:180],
            )
            for a, b in zip(reasoning, reasoning[1:])
            if a["page"] != b["page"]
        ]
        rows.append(
            dict(
                case=case,
                pdf_sha256=doc["sha256"],
                physical_pages=len(reader.pages),
                pages=pages,
                committee_pages=[s["page"] for s in reasoning],
                cross_page_transitions=transitions,
                conclusions=[
                    dict(page=s["page"], section=s["section"], text=s["text"])
                    for s in commission
                    if "OVERWEGINGEN" not in s["section"]
                ],
            )
        )
        (output / (case + "-fresh.txt")).write_text("\n".join(lines), encoding="utf-8")
    result = dict(
        paid_calls=0,
        pdfs_modified=False,
        corpus_modified=False,
        caveat="Fresh extraction uses the same pypdf library; visual inspection is required for section/role fidelity.",
        summary=dict(
            documents=len(rows),
            pages=sum(r["physical_pages"] for r in rows),
            all_raw_identical=all(
                p["raw_identical_to_prior"] for r in rows for p in r["pages"]
            ),
            all_clean_identical=all(
                p["clean_identical_to_prior"] for r in rows for p in r["pages"]
            ),
            all_registered_content_equal=all(
                p["all_content_equal"] for r in rows for p in r["pages"]
            ),
            all_sections_in_page_order=all(
                s["found_once_in_order"]
                for r in rows
                for p in r["pages"]
                for s in p["section_spans"]
            ),
        ),
        results=rows,
    )
    (output / "audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result["summary"], indent=2))
    for row in rows:
        print(
            row["case"],
            [
                (p["page"], p["headings"], p["differences"][:20])
                for p in row["pages"]
                if p["headings"] or p["differences"]
            ],
        )


if __name__ == "__main__":
    audit()
