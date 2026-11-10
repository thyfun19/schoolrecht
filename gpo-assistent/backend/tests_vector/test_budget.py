import json

import httpx
import pytest

from gpo_local.common import digest
from gpo_local.explanation import MODEL
from gpo_local.run_experiment import run, api_key


def fixture(tmp_path):
    packets = tmp_path / "packets.json"
    packets.write_text(
        json.dumps(
            [
                dict(
                    id=str(i),
                    prepared=dict(
                        request={"model": MODEL}, estimated_max_cost_usd=0.02
                    ),
                )
                for i in range(8)
            ]
        )
    )
    approval = tmp_path / "approval.json"
    approval.write_text(
        json.dumps(
            dict(
                status="approved_by_user",
                model=MODEL,
                max_calls=8,
                budget_usd=0.25,
                packets_sha256=digest(packets),
            )
        )
    )
    return packets, approval, tmp_path / "answers"


def test_unapproved_packet_never_calls_network(tmp_path):
    p, a, out = fixture(tmp_path)
    p.write_text("[]")
    with httpx.Client(
        transport=httpx.MockTransport(lambda req: pytest.fail("Network not allowed"))
    ) as client:
        with pytest.raises(ValueError):
            run(p, a, out, key="synthetic-test", client=client)


def test_error_spends_one_reserved_attempt_and_cannot_repeat(tmp_path):
    p, a, out = fixture(tmp_path)
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(429, json={"error": "test"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        ledger = run(p, a, out, key="synthetic-test", client=client)
        assert len(calls) == 1 and ledger["reserved_usd"] == 0.02
        with pytest.raises(ValueError):
            run(p, a, out, key="synthetic-test", client=client)
        assert len(calls) == 1


def test_key_file_missing_does_not_request_secret_in_chat(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ontbreekt"):
        api_key(tmp_path / "missing")


def test_manual_review_can_only_start_unattempted_questions(tmp_path):
    p, a, out = fixture(tmp_path)
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(429, json={"error": "test"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        run(p, a, out, key="synthetic-test", client=client)
        with pytest.raises(ValueError):
            run(
                p,
                a,
                out,
                key="synthetic-test",
                client=client,
                continue_unattempted=True,
            )
        assert len(calls) == 1
        ledger = run(
            p,
            a,
            out,
            key="synthetic-test",
            client=client,
            continue_unattempted=True,
            review_note="Eerste fout gecontroleerd; niet opnieuw uitvoeren.",
        )
        assert [v["id"] for v in ledger["attempts"]] == ["0", "1"]
        assert len(calls) == 2 and ledger["reserved_usd"] == 0.04


def test_eight_call_ceiling_survives_manual_continuations(tmp_path):
    p, a, out = fixture(tmp_path)
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(429, json={"error": "test"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        run(p, a, out, key="synthetic-test", client=client)
        for _ in range(8):
            ledger = run(
                p,
                a,
                out,
                key="synthetic-test",
                client=client,
                continue_unattempted=True,
                review_note="Fout beoordeeld; alleen nog ongestarte vraag vervolgen.",
            )
        assert len(calls) == 8 and len(ledger["attempts"]) == 8
        assert len({v["id"] for v in ledger["attempts"]}) == 8
        assert ledger["reserved_usd"] < 0.25
