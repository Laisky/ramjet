"""Behavior contracts for the real deploy shell using task-local Docker doubles."""

import json
import os
from pathlib import Path
import subprocess
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
FAKE = r"""#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
state = Path(os.environ["TEST_STATE"])
calls = Path(os.environ["TEST_CALLS"])
with calls.open("a") as f:
    f.write(json.dumps([Path(sys.argv[0]).name, *args]) + "\n")
mode = os.environ["TEST_MODE"]
if Path(sys.argv[0]).name == "docker-compose":
    if "ps" in args:
        print("ramjet-container")
    elif "up" in args:
        override = args[args.index("-f", args.index("-f") + 1) + 1] if args.count("-f") > 1 else ""
        wanted = "new-image-id" if not override or "ppcelery/ramjet:accepted" in Path(override).read_text() else "previous-image-id"
        if mode == "compose-fail" and wanted == "new-image-id":
            raise SystemExit(19)
        state.write_text(wanted)
elif args and args[0] == "pull":
    if mode == "pull-fail":
        raise SystemExit(17)
elif args[:2] == ["image", "inspect"]:
    print("new-image-id")
elif args and args[0] == "inspect":
    print("wrong-image-id" if mode == "wrong-image" and state.read_text() == "new-image-id" else state.read_text())
elif args and args[0] == "run":
    print("raise SystemExit(0)")
elif args and args[0] == "exec":
    if mode == "health-fail" and state.read_text() == "new-image-id":
        raise SystemExit(23)
"""


def deployment_script():
    """Load the actual workflow's shell, including its failure and rollback paths."""
    data = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    return data["jobs"]["deploy"]["steps"][0]["with"]["script"]


def run_deployment(tmp_path, mode):
    """Execute the production shell with synthetic image state and command outcomes."""
    commands = tmp_path / "bin"
    commands.mkdir()
    for name in ("docker", "docker-compose"):
        file = commands / name
        file.write_text(FAKE)
        file.chmod(0o755)
    pause = commands / "sleep"
    pause.write_text("#!/bin/sh\nexit 0\n")
    pause.chmod(0o755)
    state = tmp_path / "state"
    state.write_text("previous-image-id")
    calls = tmp_path / "calls"
    script = deployment_script().replace("cd /home/ubuntu/repo/VPS", f'cd "{tmp_path}"')
    script = script.replace(
        "${{ needs.build_hash.outputs.image }}", "ppcelery/ramjet:accepted"
    )
    script = script.replace(
        "${{ needs.build_hash.outputs.digest }}", "sha256:accepted-digest"
    )
    env = dict(
        os.environ,
        PATH=str(commands) + ":" + os.environ["PATH"],
        TEST_MODE=mode,
        TEST_STATE=str(state),
        TEST_CALLS=str(calls),
    )
    result = subprocess.run(
        ["sh", "-c", script], env=env, text=True, capture_output=True, timeout=15
    )
    assert calls.exists(), result.stderr
    return (
        result,
        state.read_text(),
        [json.loads(line) for line in calls.read_text().splitlines()],
    )


def test_deploy_failed_pull_stops_before_recreation(tmp_path):
    result, state, calls = run_deployment(tmp_path, "pull-fail")
    assert result.returncode != 0
    assert state == "previous-image-id"
    assert not any(call[0] == "docker-compose" and "up" in call for call in calls)


@pytest.mark.parametrize("mode", ["compose-fail", "health-fail", "wrong-image"])
def test_deploy_unaccepted_candidate_restores_previous_image(tmp_path, mode):
    result, state, calls = run_deployment(tmp_path, mode)
    assert result.returncode != 0
    assert state == "previous-image-id"
    assert "Previous Ramjet image and health restored." in result.stdout


def test_deploy_accepts_immutable_healthy_image(tmp_path):
    result, state, calls = run_deployment(tmp_path, "healthy")
    assert result.returncode == 0, result.stderr
    assert state == "new-image-id"
    assert any(call[0] == "docker" and call[1] == "exec" for call in calls)
    assert not any("--remove-orphans" in call for call in calls)
