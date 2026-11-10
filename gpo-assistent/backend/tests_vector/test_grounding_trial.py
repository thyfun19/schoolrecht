from pathlib import Path

import httpx
import pytest

from gpo_local.common import digest, read
from gpo_local.grounding import prepare_case_findings, MODEL
from gpo_local.run_grounding_trial import call_once, save
from gpo_local.source_registry import SourceRegistry


def fixture(tmp_path):
    registry = SourceRegistry(
        {
            "111": dict(
                sha256="a" * 64,
                advice_date="2020-01-01",
                source_url="https://example.org/a.pdf",
            )
        },
        [
            dict(
                case="111",
                page=1,
                section="OVERWEGINGEN",
                role="commissie",
                section_id="a",
                text="De school deed onderzoek.",
            )
        ],
    )
    packet = prepare_case_findings(
        "Welke onderzoeken zijn gedaan?",
        {"results": [{"case_number": "111"}]},
        registry,
    )
    save(tmp_path / "packets.json", [dict(id="V1", prepared=packet)])
    module = Path(__file__).resolve().parents[1] / "gpo_local"
    save(
        tmp_path / "approval.json",
        dict(
            status="approved_by_user",
            budget_usd=0.20,
            max_calls=4,
            model=MODEL,
            packets_sha256=digest(tmp_path / "packets.json"),
            code_hashes={
                n: digest(module / n) for n in ["grounding.py", "source_registry.py"]
            },
        ),
    )
    save(
        tmp_path / "ledger.json",
        dict(attempts=[], reserved_usd=0, known_actual_cost_usd=0),
    )
    return registry


def test_no_retry_and_reservation_precedes_api_error(tmp_path):
    registry = fixture(tmp_path)
    calls = []

    def fake(request):
        calls.append(request)
        assert len(read(tmp_path / "ledger.json")["attempts"]) == 1
        return httpx.Response(429)

    with httpx.Client(transport=httpx.MockTransport(fake)) as client:
        call_once(tmp_path, "V1", "per_case", registry, key="synthetic", client=client)
        with pytest.raises(ValueError):
            call_once(
                tmp_path, "V1", "per_case", registry, key="synthetic", client=client
            )
    assert len(calls) == 1
    assert read(tmp_path / "ledger.json")["reserved_usd"] > 0


@pytest.mark.parametrize(
    "constraint", ["budget", "four_calls", "failed_review", "closed"]
)
def test_limits_and_review_gate_never_call_api(tmp_path, constraint):
    registry = fixture(tmp_path)
    stage = "per_case"
    ledger = read(tmp_path / "ledger.json")
    if constraint == "budget":
        ledger["reserved_usd"] = 0.20
    if constraint == "four_calls":
        ledger["attempts"] = [dict(stage=str(i), status="done") for i in range(4)]
    if constraint == "closed":
        ledger["status"] = "closed_after_report"
    if constraint == "failed_review":
        stage = "comparison"
        save(tmp_path / "V1-per_case-hydrated.json", {})
        save(
            tmp_path / "V1-review.json",
            dict(overall_verdict="fail", findings_sha256="not-relevant"),
        )
    save(tmp_path / "ledger.json", ledger)
    with httpx.Client(
        transport=httpx.MockTransport(lambda req: pytest.fail("No network allowed"))
    ) as client:
        with pytest.raises(ValueError):
            call_once(tmp_path, "V1", stage, registry, key="synthetic", client=client)
