"""Resolve reviewed stable pins through PDM and regenerate the production export."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from packaging.requirements import Requirement

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

PDM_VERSION = "2.26.2"
FILES = ("pyproject.toml", "pdm.lock", "requirements.txt")
PIN = re.compile(r"([A-Za-z0-9][A-Za-z0-9_.-]*)==([0-9]+(?:\.[0-9]+)+)$")


def normalize(name):
    """normalize returns the PEP 503 spelling of a dependency name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def requested_pins(values):
    """requested_pins returns unique stable exact pins or rejects ambiguous input."""
    pins = {}
    for value in values:
        match = PIN.fullmatch(value)
        if not match:
            raise ValueError(f"stable name==version required: {value}")
        name, version = match.groups()
        name = normalize(name)
        if name in pins:
            raise ValueError(f"duplicate dependency: {name}")
        pins[name] = version
    if not pins:
        raise ValueError("at least one reviewed pin required")
    return pins


def production_pins(root):
    """production_pins returns package versions used by the frozen image."""
    lock = tomllib.loads((root / "pdm.lock").read_text())
    return {
        normalize(package["name"]): package["version"]
        for package in lock["package"]
        if "default" in package.get("groups", [])
    }


def validate_locked_constraints(root):
    """validate_locked_constraints rejects override conflicts on supported Python versions."""
    lock = tomllib.loads((root / "pdm.lock").read_text())
    versions = production_pins(root)
    for package in lock["package"]:
        if "default" not in package.get("groups", []):
            continue
        for line in package.get("dependencies", []):
            dependency = Requirement(line)
            name = normalize(dependency.name)
            for python in ("3.10", "3.11", "3.12", "3.13", "3.14"):
                environment = {
                    "extra": "",
                    "python_version": python,
                    "python_full_version": python + ".0",
                }
                if dependency.marker and not dependency.marker.evaluate(environment):
                    continue
                if name not in versions or versions[name] not in dependency.specifier:
                    raise ValueError(
                        f"{package['name']}: {line} conflicts with {name}=={versions.get(name)} on Python {python}"
                    )


def update(root, pins, apply=False, run=subprocess.run):
    """update prints a plan or applies pins, rolling back failed metadata changes."""
    current = production_pins(root)
    unknown = pins.keys() - current.keys()
    if unknown:
        raise ValueError(f"dependencies absent from production lock: {sorted(unknown)}")
    plan = {
        name: {"from": current[name], "to": version}
        for name, version in sorted(pins.items())
    }
    print(json.dumps({"reviewed_updates": plan, "apply": apply}, indent=2))
    if not apply:
        return plan
    dirty = run(
        ["git", "status", "--porcelain", "--", *FILES],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    if dirty.stdout.strip():
        raise ValueError("dependency metadata must be clean; use an isolated worktree")
    pdm = [sys.executable, "-m", "pdm"]
    version = run(
        [*pdm, "--version"], cwd=root, check=True, capture_output=True, text=True
    )
    if version.stdout.strip() != f"PDM, version {PDM_VERSION}":
        raise ValueError(f"run with the pinned PDM {PDM_VERSION}")
    saved = {name: (root / name).read_bytes() for name in FILES}
    interpreter = root / ".pdm-python"
    saved_interpreter = interpreter.read_bytes() if interpreter.exists() else None
    env = dict(os.environ, PDM_IGNORE_SAVED_PYTHON="1", PDM_CHECK_UPDATE="false")
    try:
        with tempfile.TemporaryDirectory(prefix="ramjet-pdm-update-") as temporary:
            constraints = Path(temporary) / "constraints.txt"
            constraints.write_text(
                "".join(f"{name}=={value}\n" for name, value in sorted(pins.items()))
            )
            run(
                [
                    *pdm,
                    "update",
                    "--no-sync",
                    "--update-reuse",
                    "--override",
                    str(constraints),
                    *sorted(pins),
                ],
                cwd=root,
                env=env,
                check=True,
                timeout=900,
            )
            validate_locked_constraints(root)
            run(
                [
                    *pdm,
                    "export",
                    "--prod",
                    "--without-hashes",
                    "-o",
                    "requirements.txt",
                ],
                cwd=root,
                env=env,
                check=True,
                timeout=120,
            )
        resolved = production_pins(root)
        for name, value in pins.items():
            if resolved.get(name) != value:
                raise ValueError(f"requested pin not resolved: {name}=={value}")
        if (root / "pyproject.toml").read_bytes() != saved["pyproject.toml"]:
            raise ValueError("resolver unexpectedly changed project dependency policy")
        run([*pdm, "lock", "--check"], cwd=root, env=env, check=True, timeout=120)
    except BaseException:
        for name, content in saved.items():
            (root / name).write_bytes(content)
        raise
    finally:
        if saved_interpreter is None:
            interpreter.unlink(missing_ok=True)
        else:
            interpreter.write_bytes(saved_interpreter)
    print("Review the full lock/export diff and qualify the frozen image before merge.")
    return plan


def main():
    """main parses reviewed pins and applies them only when explicitly requested."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pins", nargs="+")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    update(root, requested_pins(args.pins), apply=args.apply)


if __name__ == "__main__":
    main()
