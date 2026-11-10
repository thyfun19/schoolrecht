import copy
import pytest

from gpo_app.assessment_trial_plan import marked_section, prepare_request
from gpo_app.ingestion import json_hash


def packet():
    text = "Voorwaarde. Uitzondering."
    sources = [
        dict(
            source_id="src_a",
            section_key="s",
            page_number=2,
            role="commissie",
            start_offset=0,
            end_offset=11,
            quote="Voorwaarde.",
        ),
        dict(
            source_id="src_b",
            section_key="s",
            page_number=2,
            role="commissie",
            start_offset=12,
            end_offset=len(text),
            quote="Uitzondering.",
        ),
    ]
    section = dict(
        text=text, section_key="s", page_number=2, source_ids=["src_a", "src_b"]
    )
    result = dict(
        status="offline_assessment_context_ready",
        sources=sources,
        case_contexts=[
            dict(
                case_source_id="src_a",
                parts={"commissieoverwegingen": [section], "conclusies": []},
            )
        ],
        instructions="Historische beoordeling.",
        question="Welke voorwaarden?",
        output_schema={"type": "object"},
    )
    result["context_sha256"] = json_hash(result)
    return result


def test_all_source_characters_and_labels_survive_without_metadata_duplication():
    p = packet()
    request = prepare_request(p)["request"]
    assert request["store"] is False and request["max_output_tokens"] == 5000
    assert "Voorwaarde.[/source] [src_b]Uitzondering." in request["input"]
    assert request["input"].count("Uitzondering.") == 1
    assert "page_number" not in request["input"]
    assert "api_key" not in str(request).lower()


def test_mutated_context_and_fake_quote_are_rejected():
    p = packet()
    p["question"] = "gewijzigd"
    with pytest.raises(ValueError):
        prepare_request(p)
    p = packet()
    p["sources"][0]["quote"] = "Verzonnen."
    with pytest.raises(ValueError):
        marked_section(
            p["case_contexts"][0]["parts"]["commissieoverwegingen"][0],
            {s["source_id"]: s for s in p["sources"]},
        )
