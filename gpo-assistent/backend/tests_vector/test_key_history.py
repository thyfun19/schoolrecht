import subprocess

from gpo_local.audit_key_history import audit


def repo(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    subprocess.run(
        ["git", "init", "-q", str(tmp_path)], check=True, capture_output=True
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Offline test"], check=True
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "offline@example.invalid"],
        check=True,
    )
    key = "sk-" + ("a" * 60)  # Deliberately synthetic; no real credential.
    path = tmp_path / "gpo-assistent/backend"
    path.mkdir(parents=True)
    (path / ".env").write_text("OPENAI_API_KEY=" + key)
    return key


def test_detects_unreachable_key_without_claiming_commit_or_push(tmp_path, monkeypatch):
    key = repo(tmp_path, monkeypatch)
    subprocess.run(
        ["git", "-C", str(tmp_path), "hash-object", "-w", "--stdin"],
        input=key.encode(),
        capture_output=True,
        check=True,
    )
    result = audit(tmp_path)
    assert len(result["hits"]) == 1 and result["hits"][0]["exact_current_key"]
    assert result["commits_containing_key"] == []
    assert result["staged_secret_paths"] == []
    assert key not in str(result)


def test_detects_key_in_index_and_committed_history(tmp_path, monkeypatch):
    key = repo(tmp_path, monkeypatch)
    (tmp_path / "example.txt").write_text("OPENAI_API_KEY=" + key)
    subprocess.run(["git", "-C", str(tmp_path), "add", "example.txt"], check=True)
    result = audit(tmp_path)
    assert result["staged_secret_paths"] == ["example.txt"]
    assert result["commits_containing_key"] == []
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-qm", "synthetic fixture"], check=True
    )
    result = audit(tmp_path)
    assert result["commits_containing_key"][0]["path"] == "example.txt"
    assert any(result["ref_reachability"].values())
    assert key not in str(result)
