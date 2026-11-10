import json
import pytest
from gpo_app import run_assessment_trial as trial


def setup(tmp_path, monkeypatch, status=200):
    proposal = dict(
        model=trial.MODEL,
        max_calls=2,
        budget_usd=0.20,
        retries=0,
        comparison_calls=0,
        drafts_sha256="frozen",
    )
    drafts = [
        dict(id=q, request={}, request_sha256=q, estimated_max_usd=0.075)
        for q in ["V1", "V4"]
    ]
    trial.save(tmp_path / "proposal.json", proposal)
    trial.save(
        tmp_path / "ledger.json",
        dict(status="not_started", attempts=[], actual_cost_usd=0),
    )
    trial.record_approval(tmp_path)
    monkeypatch.setattr(trial, "preflight", lambda *args: (proposal, drafts))
    trial.save(tmp_path / "V1.json", {})
    trial.save(tmp_path / "V4.json", {})
    monkeypatch.setattr(
        trial,
        "validate",
        lambda *args: dict(semantic_review_required=True, comparison_allowed=False),
    )

    class Client:
        calls = 0

        def post(self, *args, **kwargs):
            self.calls += 1

            class Response:
                status_code = status

                def json(self):
                    return dict(
                        status="completed",
                        usage=dict(input_tokens=1000, output_tokens=100),
                        output=[
                            dict(
                                type="message",
                                content=[dict(type="output_text", text="{}")],
                            )
                        ],
                    )

            return Response()

    return Client()


def test_two_calls_closed_and_no_replay(tmp_path, monkeypatch):
    client = setup(tmp_path, monkeypatch)
    result = trial.execute(
        None, tmp_path, tmp_path, client=client, key="synthetic-not-a-key"
    )
    assert client.calls == 2 and result["comparison_calls"] == 0
    assert result["actual_cost_usd"] == pytest.approx(0.0024)
    with pytest.raises(ValueError):
        trial.execute(None, tmp_path, tmp_path, client=client, key="synthetic")
    assert client.calls == 2


def test_http_failure_stops_without_second_call(tmp_path, monkeypatch):
    client = setup(tmp_path, monkeypatch, 400)
    result = trial.execute(None, tmp_path, tmp_path, client=client, key="synthetic")
    assert client.calls == 1 and result["attempts"][0]["status"] == "api_error_no_retry"


def test_completed_paid_response_with_invalid_sources_stops_and_keeps_usage(
    tmp_path, monkeypatch
):
    client = setup(tmp_path, monkeypatch)

    def reject(*args):
        raise ValueError("Synthetic cross-case citation failure.")

    monkeypatch.setattr(trial, "validate", reject)
    result = trial.execute(None, tmp_path, tmp_path, client=client, key="synthetic")
    assert client.calls == 1 and result["status"] == "closed_after_trial"
    assert result["actual_cost_usd"] == pytest.approx(0.0012)
    assert result["attempts"][0]["status"] == "failed_no_retry"
    assert (tmp_path / "V1-raw.json").exists() and not (
        tmp_path / "V4-raw.json"
    ).exists()


def test_cached_cost_and_invalid_usage():
    assert trial.cost(
        dict(
            input_tokens=1000,
            output_tokens=100,
            input_tokens_details={"cached_tokens": 1000},
        )
    ) == pytest.approx(0.000525)
    with pytest.raises(ValueError):
        trial.cost(
            dict(
                input_tokens=1,
                output_tokens=1,
                input_tokens_details={"cached_tokens": 2},
            )
        )
