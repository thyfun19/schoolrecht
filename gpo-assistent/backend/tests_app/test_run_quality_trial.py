import pytest

from gpo_app import run_quality_trial as trial


def setup(tmp_path, monkeypatch, failure=None):
    inputs = [
        dict(
            id=q,
            cases=["1", "2", "3"],
            baseline_max_usd=0.06,
            selection_max_usd=0.06,
            writing_max_usd=0.06,
        )
        for q in ["D62", "D65"]
    ]
    proposal = dict(proposal_sha256="frozen", inputs=inputs)
    drafts = [
        dict(
            stage=stage,
            qid=q["id"],
            case_number=case,
            request={},
            request_sha256="fixed",
            estimated_max_cost_usd=0.01,
        )
        for stage in ["baseline", "selection"]
        for q in inputs
        for case in q["cases"]
    ]
    workflows = {q["id"]: dict(qid=q["id"]) for q in inputs}
    monkeypatch.setattr(trial, "preflight", lambda *args: (proposal, drafts, workflows))
    monkeypatch.setattr(trial, "plan_validate", lambda *args: dict(plans=[]))

    def validate(engine, draft, payload):
        if failure == draft["stage"]:
            raise ValueError("Synthetic validation failure.")
        return dict(semantic_review_required=True)

    monkeypatch.setattr(trial, "validate_draft", validate)

    def writer(engine, validated, workflow, reviews):
        return dict(
            packet_sha256="synthetic",
            case_packets=[
                dict(
                    case_number=c,
                    request={},
                    request_sha256="writer",
                    estimated_max_cost_usd=0.01,
                    case_markers=[c],
                    sources=[],
                )
                for c in ["1", "2", "3"]
            ],
        )

    monkeypatch.setattr(trial, "writing", writer)
    trial.initialize(None, tmp_path, tmp_path / "run")

    class Client:
        calls = 0

        def post(self, *args, **kwargs):
            self.calls += 1

            class Response:
                status_code = 400 if failure == "http" else 200

                def json(self):
                    return dict(
                        status="completed",
                        model=trial.MODEL,
                        usage=dict(input_tokens=1000, output_tokens=100),
                        output=[
                            dict(
                                type="message",
                                content=[dict(type="output_text", text="{}")],
                            )
                        ],
                    )

            return Response()

    return Client(), proposal, drafts


def test_pause_for_review_then_at_most_eighteen_calls_and_no_replay(
    tmp_path, monkeypatch
):
    client, _, _ = setup(tmp_path, monkeypatch)
    base = tmp_path / "run"
    result = trial.execute(
        None, tmp_path, base, "initial", client=client, key="synthetic"
    )
    assert result["status"] == "awaiting_content_review" and client.calls == 12
    with pytest.raises(ValueError):
        trial.execute(None, tmp_path, base, "initial", client=client, key="synthetic")
    with pytest.raises(FileNotFoundError):
        trial.execute(None, tmp_path, base, "writing", client=client, key="synthetic")
    assert client.calls == 12
    trial.save(base / "reviews.json", dict(D62={}, D65={}))
    result = trial.execute(
        None, tmp_path, base, "writing", client=client, key="synthetic"
    )
    assert client.calls == 18 and result["status"] == "closed_after_trial"
    assert result["actual_cost_usd"] == pytest.approx(0.0216)
    with pytest.raises(ValueError):
        trial.execute(None, tmp_path, base, "writing", client=client, key="synthetic")
    assert client.calls == 18


@pytest.mark.parametrize(
    "failure,expected_calls", [("baseline", 1), ("selection", 7), ("http", 1)]
)
def test_any_api_or_validation_failure_stops_without_retry(
    tmp_path, monkeypatch, failure, expected_calls
):
    client, _, _ = setup(tmp_path, monkeypatch, failure)
    result = trial.execute(
        None, tmp_path, tmp_path / "run", "initial", client=client, key="synthetic"
    )
    assert result["status"] == "closed_after_failure" and client.calls == expected_calls
    if failure != "http":
        assert result["actual_cost_usd"] == pytest.approx(expected_calls * 0.0012)
        assert "actual_cost_unknown" not in result["attempts"][-1]


def test_call_and_stage_budget_checked_before_request(tmp_path, monkeypatch):
    client, proposal, drafts = setup(tmp_path, monkeypatch)
    ledger = dict(
        attempts=[dict(stage="baseline", qid="D62", reserved_usd=0.001)] * 18,
        reserved_usd=0,
        actual_cost_usd=0,
    )
    with pytest.raises(ValueError):
        trial.call_one(
            None, tmp_path / "run", ledger, proposal, drafts[0], client, "synthetic"
        )
    ledger = dict(
        attempts=[dict(stage="baseline", qid="D62", reserved_usd=0.01)] * 6,
        reserved_usd=0.06,
        actual_cost_usd=0,
    )
    with pytest.raises(ValueError):
        trial.call_one(
            None, tmp_path / "run", ledger, proposal, drafts[0], client, "synthetic"
        )
    ledger = dict(attempts=[], reserved_usd=0.499, actual_cost_usd=0)
    with pytest.raises(ValueError):
        trial.call_one(
            None, tmp_path / "run", ledger, proposal, drafts[0], client, "synthetic"
        )
    assert client.calls == 0


def test_approval_cannot_be_changed(tmp_path, monkeypatch):
    client, _, _ = setup(tmp_path, monkeypatch)
    base = tmp_path / "run"
    approval = trial.read(base / "approval.json")
    approval["budget_usd"] = 1
    trial.save(base / "approval.json", approval)
    with pytest.raises(ValueError):
        trial.execute(None, tmp_path, base, "initial", client=client, key="synthetic")
    assert client.calls == 0
