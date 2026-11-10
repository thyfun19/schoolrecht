"""Read-only Git object scan. Never print, serialize or hash-display secret values."""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


def git(*args, root=None):
    return subprocess.check_output(
        ["git", *args], cwd=root, stderr=subprocess.PIPE, timeout=30
    )


def audit(root, remote=None):
    run = lambda *args: git(*args, root=root)
    candidates = set()
    for path in [
        root / "gpo-assistent/backend/.env",
        root / "gpo-assistent/backend/.env.example",
    ]:
        if path.exists():
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                if line.strip().startswith("OPENAI_API_KEY="):
                    value = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
                    if len(value) > 20:
                        candidates.add(value.encode())
    if os.getenv("OPENAI_API_KEY"):
        candidates.add(os.environ["OPENAI_API_KEY"].encode())
    if not candidates:
        raise ValueError(
            "Geen lokale sleutel beschikbaar voor exacte vergelijking; niets getoond."
        )
    objects = (
        run(
            "cat-file",
            "--batch-all-objects",
            "--batch-check=%(objectname) %(objecttype)",
        )
        .decode()
        .splitlines()
    )
    hits = []
    # Include unreachable objects and tool checkpoint refs, not only main branch history.
    for row in objects:
        oid, kind = row.split()
        if kind not in ["blob", "commit", "tag"]:
            continue
        value = run("cat-file", kind, oid)
        exact = any(key in value for key in candidates)
        pattern = bool(re.search(rb"sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}", value))
        if exact or pattern:
            hits.append(
                dict(
                    object_id=oid,
                    object_type=kind,
                    exact_current_key=exact,
                    key_like_pattern=pattern,
                )
            )
    index_hits = []
    for entry in run("ls-files", "-s", "-z").split(b"\x00"):
        if not entry:
            continue
        meta, path = entry.split(b"\t", 1)
        oid = meta.split()[1].decode()
        if any(h["object_id"] == oid for h in hits):
            index_hits.append(path.decode(errors="replace"))
    ids = {h["object_id"] for h in hits}
    refs = run("for-each-ref", "--format=%(refname)").decode().splitlines()
    reachable = lambda ref: (
        ids
        & {
            line.split()[0]
            for line in run("rev-list", "--objects", ref).decode().splitlines()
        }
    )
    linked = []
    for row in objects:
        oid, kind = row.split()
        if kind != "commit":
            continue
        if oid in ids:
            linked.append(dict(commit=oid, location="commit_metadata"))
        for item in run("ls-tree", "-r", "-z", oid).split(b"\x00"):
            if item and item.split(b"\t", 1)[0].split()[2].decode() in ids:
                linked.append(dict(commit=oid, path=item.split(b"\t", 1)[1].decode()))
    result = dict(
        local_key_candidates=len(candidates),
        object_count=len(objects),
        all_local_objects_scanned=True,
        includes_unreachable_and_checkpoint_objects=True,
        hits=hits,
        staged_secret_paths=index_hits,
        commits_containing_key=linked,
        ref_reachability={ref: sorted(reachable(ref)) for ref in refs},
        local_remote_tracking_refs=run(
            "for-each-ref", "--format=%(refname) %(objectname)", "refs/remotes/"
        )
        .decode()
        .splitlines(),
        limitations="Remote live refs, deleted server refs, forks, server backups and provider access logs need separate verification. No history rewritten.",
    )
    result["ref_types"] = {
        ref: run("cat-file", "-t", ref).decode().strip() for ref in refs
    }
    result["ref_secret_paths"] = {}
    for ref in refs:
        if result["ref_types"][ref] not in ["tree", "commit", "tag"]:
            continue
        result["ref_secret_paths"][ref] = [
            item.split(b"\t", 1)[1].decode()
            for item in run("ls-tree", "-r", "-z", ref).split(b"\x00")
            if item and item.split(b"\t", 1)[0].split()[2].decode() in ids
        ]
    if remote:
        lines = (
            run(
                "-c",
                "core.sshCommand=ssh -o BatchMode=yes -o ConnectTimeout=10",
                "ls-remote",
                "--heads",
                "--tags",
                remote,
            )
            .decode()
            .splitlines()
        )
        result["live_remote_refs"] = lines
        result["remote_reachability"] = {}
        for line in lines:
            oid, ref = line.split()
            try:
                result["remote_reachability"][ref] = sorted(reachable(oid))
            except subprocess.CalledProcessError:
                result["remote_reachability"][ref] = (
                    "not_available_locally_not_verified"
                )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("local-results/key-history.json")
    )
    parser.add_argument(
        "--remote", help="Optional read-only live branch/tag comparison"
    )
    args = parser.parse_args()
    root = Path(git("rev-parse", "--show-toplevel").decode().strip())
    result = audit(root, args.remote)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
