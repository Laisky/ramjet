"""Run changed-file formatting and three independent packaging contracts."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

TEST_FILE = "tests/test_packaging_metadata.py"
TESTS = (
    "test_pyproject_has_no_license_classifier_conflict",
    "test_production_settings_are_excluded_from_build",
    "test_dockerfile_copies_license_before_pdm_install",
)


def validate_xml(output, expected=TESTS):
    """validate_xml rejects incomplete, skipped, duplicated or failing receipts."""
    root = ET.fromstring(output)
    cases = list(root.iter("testcase"))
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("nonempty unique expected tests required")
    if sorted(case.get("name") for case in cases) != sorted(expected):
        raise ValueError("test completion differs from explicit allowlist")
    for suite in root.iter("testsuite"):
        for key in ("failures", "errors", "skipped"):
            if int(suite.get(key, "0")):
                raise ValueError(f"unsuccessful suite: {key}")
    if any(list(case) for case in cases):
        raise ValueError("test skip/error/failure or unexpected report content")


def run(command, evidence, label, records, timeout=120):
    """run retains native output, exits and timing, including failures/timeouts."""
    record = {"command": command, "exit_code": None}
    records.append(record)
    started = time.perf_counter()
    try:
        result = subprocess.run(command, capture_output=True, timeout=timeout)
        record["exit_code"] = result.returncode
        (evidence / f"{label}.stdout").write_bytes(result.stdout)
        (evidence / f"{label}.stderr").write_bytes(result.stderr)
        if result.returncode:
            sys.stderr.write(result.stdout.decode(errors="replace"))
            sys.stderr.write(result.stderr.decode(errors="replace"))
            raise RuntimeError(f"{label} exited {result.returncode}")
        return result.stdout.decode("utf-8")
    except subprocess.TimeoutExpired as exc:
        record["error"] = "timeout; no successful exit"
        (evidence / f"{label}.stdout").write_bytes(exc.stdout or b"")
        (evidence / f"{label}.stderr").write_bytes(exc.stderr or b"")
        raise RuntimeError(f"{label} timed out") from exc
    finally:
        record["seconds"] = round(time.perf_counter() - started, 3)


def main():
    """main discovers exact tests and writes an honest receipt on every outcome."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--evidence", required=True, type=Path)
    args = parser.parse_args()
    args.evidence.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    receipt = {"passed": False, "commands": []}
    records = receipt["commands"]
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    try:
        receipt["source"] = run(
            ["git", "rev-parse", "HEAD"], args.evidence, "source", records
        ).strip()
        run(
            ["git", "rev-parse", "--verify", args.base + "^{commit}"],
            args.evidence,
            "base",
            records,
        )
        changed = run(
            [
                "git",
                "diff",
                "--name-only",
                "-z",
                "--diff-filter=ACMR",
                args.base,
                "HEAD",
                "--",
                "*.py",
            ],
            args.evidence,
            "changed-python",
            records,
        )
        paths = sorted(
            set(filter(None, changed.split("\0")))
            | {TEST_FILE, ".scripts/fast_ci.py", ".scripts/test_fast_ci.py"}
        )
        receipt["format_paths"] = paths
        run(
            [sys.executable, "-m", "black", "--check", *paths],
            args.evidence,
            "format",
            records,
        )
        nodes = [f"{TEST_FILE}::{name}" for name in TESTS]
        common = [
            sys.executable,
            "-m",
            "pytest",
            "--noconftest",
            "-p",
            "no:cacheprovider",
            "--color=no",
            "-q",
        ]
        output = run(
            [*common, "--collect-only", *nodes], args.evidence, "discovery", records
        )
        found = [line for line in output.splitlines() if "::" in line]
        if sorted(found) != sorted(nodes):
            raise ValueError("test discovery differs from explicit allowlist")
        xml = args.evidence / "tests.xml"
        run([*common, "--junitxml", str(xml), *nodes], args.evidence, "tests", records)
        validate_xml(xml.read_bytes())
        receipt["passed"] = True
    except Exception as exc:
        receipt["error"] = str(exc)
        print(str(exc), file=sys.stderr)
    finally:
        receipt["seconds"] = round(time.perf_counter() - started, 3)
        (args.evidence / "receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {k: receipt[k] for k in ("passed", "seconds", "error") if k in receipt}
            )
        )
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
