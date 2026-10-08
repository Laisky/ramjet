"""Exercise the actual deploy shell against the verified Home layout and CLI."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
HOME_REPO = "/home/laisky/repo/laisky/VPS"
FAKE = r"""#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with Path(os.environ["TEST_CALLS"]).open("a") as f:
    f.write(json.dumps([Path(sys.argv[0]).name, *args]) + "\n")
if Path(sys.argv[0]).name == "docker-compose":
    print("Home has Compose v2; legacy docker-compose is unavailable", file=sys.stderr)
    raise SystemExit(127)
if args[:3] != ["compose", "-f", "home-docker-compose.yml"]:
    print("Unexpected deployment command or compose file", file=sys.stderr)
    raise SystemExit(64)
if not Path(args[2]).is_file():
    raise SystemExit(66)
mode = os.environ["TEST_MODE"]
if args[3:] == ["pull", "ramjet"]:
    raise SystemExit(17 if mode == "pull-fail" else 0)
if args[3:] == ["up", "-d", "--no-deps", "--force-recreate", "ramjet"]:
    if mode == "compose-fail":
        raise SystemExit(19)
    Path(os.environ["TEST_STATE"]).write_text("latest-image-id")
elif args[3:] == ["exec", "-T", "ramjet", "python", "/app/scripts/check_container_health.py"]:
    if mode == "health-fail":
        raise SystemExit(23)
    if mode == "health-retry":
        attempts = Path(os.environ["TEST_STATE"] + "-health")
        count = int(attempts.read_text()) if attempts.exists() else 0
        attempts.write_text(str(count + 1))
        if count == 0:
            raise SystemExit(23)
else:
    raise SystemExit(64)
"""


def deployment_script():
    """deployment_script loads the shell that the release workflow executes."""
    data = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    return data["jobs"]["deploy"]["steps"][0]["with"]["script"]


def run_deployment(tmp_path, mode, emulate_home=True):
    """run_deployment emulates only Home's existing path and Compose v2 commands."""
    commands = tmp_path / "bin"
    commands.mkdir()
    for name in ("docker", "docker-compose"):
        file = commands / name
        file.write_text(FAKE)
        file.chmod(0o755)
    pause = commands / "sleep"
    pause.write_text("#!/bin/sh\nexit 0\n")
    pause.chmod(0o755)
    home_repo = tmp_path / HOME_REPO.lstrip("/")
    home_repo.mkdir(parents=True)
    (home_repo / "home-docker-compose.yml").write_text(
        "services: {ramjet: {image: ppcelery/ramjet:latest}}\n"
    )
    state = tmp_path / "state"
    state.write_text("previous-image-id")
    calls = tmp_path / "calls"
    script = deployment_script()
    if emulate_home:
        script = script.replace("/home/", str(tmp_path / "home") + "/")
    else:
        # Bypass only cd so a stale legacy CLI can be reproduced independently.
        script = re.sub(r"(?m)^cd /home/[^\n]+$", f'cd "{home_repo}"', script)
    script = re.sub(r"\$\{\{[^}]+\}\}", "unused-build-output", script)
    env = dict(
        os.environ,
        PATH=str(commands) + ":" + str(Path(sys.executable).parent) + ":/usr/bin:/bin",
        TEST_MODE=mode,
        TEST_STATE=str(state),
        TEST_CALLS=str(calls),
    )
    result = subprocess.run(
        ["sh", "-c", script], env=env, text=True, capture_output=True, timeout=5
    )
    recorded = (
        [json.loads(line) for line in calls.read_text().splitlines()]
        if calls.exists()
        else []
    )
    return result, state.read_text(), recorded


def test_deploy_uses_existing_home_layout(tmp_path):
    """test_deploy_uses_existing_home_layout rejects the removed Ubuntu/BJ path."""
    result, state, calls = run_deployment(tmp_path, "healthy")
    assert result.returncode == 0, result.stderr
    assert state == "latest-image-id"
    assert len(calls) == 3


def test_deploy_uses_available_compose_v2(tmp_path):
    """test_deploy_uses_available_compose_v2 rejects the unavailable legacy CLI."""
    result, state, calls = run_deployment(tmp_path, "healthy", emulate_home=False)
    assert result.returncode == 0, result.stderr
    assert state == "latest-image-id"
    assert all(call[:2] == ["docker", "compose"] for call in calls)


def test_deploy_failed_pull_stops_before_recreation(tmp_path):
    """test_deploy_failed_pull_stops_before_recreation preserves the old service."""
    result, state, calls = run_deployment(tmp_path, "pull-fail")
    assert result.returncode == 17
    assert state == "previous-image-id"
    assert calls == [
        ["docker", "compose", "-f", "home-docker-compose.yml", "pull", "ramjet"]
    ]


def test_deploy_recreate_failure_is_reported(tmp_path):
    """test_deploy_recreate_failure_is_reported keeps a failed recreation visible."""
    result, state, calls = run_deployment(tmp_path, "compose-fail")
    assert result.returncode == 19
    assert state == "previous-image-id"
    assert len(calls) == 2


def test_deploy_pulls_latest_and_recreates_only_ramjet(tmp_path):
    """test_deploy_pulls_latest_and_recreates_only_ramjet protects sibling services."""
    result, state, calls = run_deployment(tmp_path, "healthy")
    assert result.returncode == 0, result.stderr
    assert state == "latest-image-id"
    assert calls == [
        ["docker", "compose", "-f", "home-docker-compose.yml", "pull", "ramjet"],
        [
            "docker",
            "compose",
            "-f",
            "home-docker-compose.yml",
            "up",
            "-d",
            "--no-deps",
            "--force-recreate",
            "ramjet",
        ],
        [
            "docker",
            "compose",
            "-f",
            "home-docker-compose.yml",
            "exec",
            "-T",
            "ramjet",
            "python",
            "/app/scripts/check_container_health.py",
        ],
    ]


def test_deploy_retries_health_until_accepted(tmp_path):
    """test_deploy_retries_health_until_accepted allows a brief startup delay."""
    result, state, calls = run_deployment(tmp_path, "health-retry")
    assert result.returncode == 0, result.stderr
    assert state == "latest-image-id"
    assert len([call for call in calls if "exec" in call]) == 2


def test_deploy_exhausted_health_is_reported(tmp_path):
    """test_deploy_exhausted_health_is_reported bounds retries and fails the job."""
    result, state, calls = run_deployment(tmp_path, "health-fail")
    assert result.returncode != 0
    assert state == "latest-image-id"
    assert len([call for call in calls if "exec" in call]) == 30
    assert len([call for call in calls if "up" in call]) == 1
