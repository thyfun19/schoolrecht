import pytest

from gpo_app import run_isolated_trial as trial


def setup(tmp_path, monkeypatch, *, reject=False, http_status=200):
    proposal = dict(proposal_sha256="frozen")
    drafts = [
        dict(
            case_number=str(i),
            request_sha256=str(i),
            request={},
            estimated_max_cost_usd=0.025,
        )
        for i in range(5)
    ]
    monkeypatch.setattr(trial, "preflight", lambda *args: (proposal, drafts, [{}] * 5))
    trial.initialize(None, tmp_path, tmp_path / "run")

    def validate(*args):
        if reject:
            raise ValueError("Synthetic invalid citation.")
        return dict(semantic_review_required=True)

    monkeypatch.setattr(trial, "validate", validate)

    class Client:
        calls = 0

        def post(self, *args, **kwargs):
            self.calls += 1

            class Response:
                status_code = http_status

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

    return Client()


def test_exactly_five_calls_and_no_replay(tmp_path, monkeypatch):
    client = setup(tmp_path, monkeypatch)
    approval = trial.read(tmp_path / "run/approval.json")
    assert approval["authorized_ceiling_usd"] == 0.20
    result = trial.execute(
        None, tmp_path, tmp_path / "run", client=client, key="synthetic"
    )
    assert client.calls == 5 and result["actual_cost_usd"] == pytest.approx(0.006)
    assert result["status"] == "closed_after_trial" and result["comparison_calls"] == 0
    with pytest.raises(ValueError):
        trial.execute(None, tmp_path, tmp_path / "run", client=client, key="synthetic")
    assert client.calls == 5


def test_validation_error_stops_and_keeps_paid_usage(tmp_path, monkeypatch):
    client = setup(tmp_path, monkeypatch, reject=True)
    result = trial.execute(
        None, tmp_path, tmp_path / "run", client=client, key="synthetic"
    )
    assert client.calls == 1 and result["actual_cost_usd"] == pytest.approx(0.0012)
    assert result["attempts"][0]["status"] == "failed_no_retry"
    assert not (tmp_path / "run/1-raw.json").exists()


def test_http_error_is_not_retried(tmp_path, monkeypatch):
    client = setup(tmp_path, monkeypatch, http_status=400)
    result = trial.execute(
        None, tmp_path, tmp_path / "run", client=client, key="synthetic"
    )
    assert client.calls == 1 and result["attempts"][0]["actual_cost_unknown"]


@pytest.mark.parametrize(
    "field,value", [("operational_budget_usd", 0.50), ("authorized_ceiling_usd", 0.10)]
)
def test_approval_budget_cannot_be_exceeded(tmp_path, monkeypatch, field, value):
    client = setup(tmp_path, monkeypatch)
    approval = trial.read(tmp_path / "run/approval.json")
    approval[field] = value
    trial.save(tmp_path / "run/approval.json", approval)
    with pytest.raises(ValueError):
        trial.execute(None, tmp_path, tmp_path / "run", client=client, key="synthetic")
    assert client.calls == 0
