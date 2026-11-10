import hashlib
import json
import re
from pathlib import Path

REVISION = "761b726dd34fb83930e26aab4e9ac3899aa1fa78"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def norm(text):
    text = re.sub(r"-\s*\n\s*", "", text).replace("\u00ad", "").replace("\x02", "")
    return re.sub(r"\s+", " ", text).strip()


def sections(pages):
    """Match the frozen study's page, heading and speaker boundaries."""
    output = []
    case, heading, role = None, "VOORBLAD", "document"
    for page in pages:
        if case != page["case"]:
            case, heading, role = page["case"], "VOORBLAD", "document"
        lines, start, offset = [], 0, 0

        def flush():
            text = norm("\n".join(lines))
            if text:
                output.append(
                    dict(
                        case=case,
                        page=page["page"],
                        section=heading,
                        role=role,
                        section_id=f"{case}-p{page['page']}-{start}",
                        text=text,
                    )
                )
            lines.clear()

        for line in page["text"].splitlines(keepends=True):
            title = re.match(r"^\d+\.\s*([A-ZÁÉ][A-ZÁÉ -]{4,})$", line.strip())
            actor = re.fullmatch(
                r"(?:Standpunt\s+)?(Verzoekers?|Verzoekster|Verweerder|verzoekers?|verzoekster|verweerder)|Mening van de leerling",
                line.strip(),
            )
            if title:
                flush()
                heading = title[1].strip()
                role = (
                    "commissie"
                    if "OVERWEGINGEN" in heading
                    or heading in ["ADVIES", "OORDEEL", "AANBEVELINGEN"]
                    else "feiten"
                    if "FEITEN" in heading
                    else "samenwerkingsverband"
                    if "SAMENWERKINGSVERBAND" in heading
                    else "leerling"
                    if "MENING" in heading
                    else "document"
                )
                start = offset + len(line)
            elif actor:
                flush()
                role = (
                    "school"
                    if "verweerder" in line.lower()
                    else "leerling"
                    if "leerling" in line.lower()
                    else "ouders"
                )
                start = offset + len(line)
            else:
                lines.append(line.strip())
            offset += len(line)
        flush()
    return output
