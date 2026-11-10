"""Canonical citeable units: stable versioned IDs and server-owned provenance."""

import hashlib
import json
import re
from types import MappingProxyType

VERSION = "gpo-citation-v2"


def source_id(source):
    identity = {
        k: source[k]
        for k in [
            "case_number",
            "pdf_sha256",
            "page",
            "section_id",
            "section",
            "role",
            "start_offset",
            "end_offset",
            "quote",
        ]
    }
    identity["version"] = VERSION
    return (
        "src_"
        + hashlib.sha256(
            json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()[:32]
    )


def spans(text, maximum=800):
    """No page/role crossing, no model-created ellipses or restored punctuation."""
    start = 0
    for end in [m.end() for m in re.finditer(r"[.!?](?:\s+|$)", text)] + [len(text)]:
        if end <= start:
            continue
        while end - start > maximum:
            stop = text.rfind(" ", start, start + maximum)
            if stop <= start:
                stop = start + maximum
            yield start, stop
            start = stop
            while start < end and text[start].isspace():
                start += 1
        stop = end
        while stop > start and text[stop - 1].isspace():
            stop -= 1
        if stop > start:
            yield start, stop
        start = end


class SourceRegistry:
    def __init__(self, documents, sections):
        entries = {}
        for section in sections:
            category = (
                "commissieoverwegingen"
                if "OVERWEGINGEN" in section["section"]
                else "conclusies"
                if section["section"] in ["ADVIES", "OORDEEL", "AANBEVELINGEN"]
                else None
            )
            if section["role"] != "commissie" or category is None:
                continue
            doc = documents[section["case"]]
            for start, end in spans(section["text"]):
                entry = dict(
                    case_number=section["case"],
                    advice_date=doc["advice_date"],
                    pdf_sha256=doc["sha256"],
                    page=section["page"],
                    section=section["section"],
                    section_id=section["section_id"],
                    role=section["role"],
                    category=category,
                    source_url=doc["source_url"].split("#")[0]
                    + f"#page={section['page']}",
                    start_offset=start,
                    end_offset=end,
                    quote=section["text"][start:end],
                )
                sid = source_id(entry)
                if sid in entries and dict(entries[sid]) != dict(entry, source_id=sid):
                    raise ValueError("Conflicterend bron-ID.")
                entries[sid] = MappingProxyType(dict(entry, source_id=sid))
        self._entries = MappingProxyType(entries)

    def select(self, cases):
        return [dict(s) for s in self._entries.values() if s["case_number"] in cases]

    def get(self, sid, allowed_ids):
        if sid not in allowed_ids or sid not in self._entries:
            raise ValueError("Onbekend bron-ID of bron buiten de opgehaalde context.")
        return dict(self._entries[sid])
