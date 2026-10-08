from __future__ import annotations

"""Tests that guard against packaging regressions."""

from pathlib import Path

try:  # pragma: no cover - fallback only used on Python 3.10.
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


def _load_pyproject() -> dict:
    """Load the parsed pyproject.toml into a dictionary."""
    project_root = Path(__file__).resolve().parents[1]
    with (project_root / "pyproject.toml").open("rb") as fp:
        return tomllib.load(fp)


def test_pyproject_has_no_license_classifier_conflict() -> None:
    """Ensure no trove license classifiers sneak back in alongside SPDX metadata."""
    data = _load_pyproject()
    classifiers = data["project"].get("classifiers", [])
    assert all(
        not classifier.startswith("License ::") for classifier in classifiers
    ), "SPDX licenses cannot be combined with trove license classifiers under pdm-backend"


def test_production_settings_are_excluded_from_build() -> None:
    """Verify the production settings module stays out of built distributions."""
    data = _load_pyproject()
    build_cfg = data.get("tool", {}).get("pdm", {}).get("build", {})
    excludes = build_cfg.get("excludes", [])
    assert (
        "ramjet/settings/prd.py" in excludes
    ), "Absolute prd settings symlink must stay excluded to keep builds reproducible"


def test_dockerfile_copies_license_before_pdm_install() -> None:
    """Ensure Docker build stage provides LICENSE before running pdm install."""
    project_root = Path(__file__).resolve().parents[1]
    dockerfile = (project_root / "Dockerfile").read_text().splitlines()
    license_copy_line = None
    pdm_install_line = None
    for idx, raw_line in enumerate(dockerfile):
        line = raw_line.strip().lower()
        if license_copy_line is None and line.startswith("copy") and "license" in line:
            license_copy_line = idx
        if line.startswith("run") and "pdm install" in line:
            pdm_install_line = idx
            # Ensure the first pdm install uses --no-self to avoid needing README/source
            assert (
                "--no-self" in line
            ), "First pdm install must use --no-self to skip project build"
            break

    assert (
        license_copy_line is not None
    ), "Dockerfile must copy LICENSE before installing dependencies"
    assert pdm_install_line is not None, "Dockerfile must invoke pdm install"
    assert (
        license_copy_line < pdm_install_line
    ), "COPY LICENSE must come before the pdm install RUN layer"


def test_requirements_match_frozen_lock() -> None:
    """test_requirements_match_frozen_lock rejects exports that bypass production pins."""
    project_root = Path(__file__).resolve().parents[1]
    with (project_root / "pdm.lock").open("rb") as fp:
        lock = tomllib.load(fp)
    locked = {
        package["name"].lower().replace("_", "-").replace(".", "-"): package["version"]
        for package in lock["package"]
        if "default" in package.get("groups", [])
    }
    exported = {}
    for raw_line in (project_root / "requirements.txt").read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        requirement = line.split(";", 1)[0].strip()
        name, version = requirement.split("==", 1)
        exported[name.lower().replace("_", "-").replace(".", "-")] = version
    assert (
        exported == locked
    ), "requirements.txt must be regenerated from production pdm.lock"
