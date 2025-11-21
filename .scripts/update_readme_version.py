#!/usr/bin/env python3
"""Update the README version badge to match the package version."""

from __future__ import annotations

import re
from pathlib import Path

from ramjet import __version__


def update_readme_version(version: str) -> None:
    """Replace the README shield badge with the supplied version string."""
    ver_reg = re.compile(
        r"(https://img\.shields\.io/badge/version-v"
        r"[0-9]+\.[0-9]+(\.[0-9]+)?((dev|rc)[0-9]+)?"
        r"-blue\.svg)"
    )
    badge_url = f"https://img.shields.io/badge/version-v{version}-blue.svg"
    readme_path = Path("README.md")
    readme_path.write_text(
        ver_reg.sub(badge_url, readme_path.read_text(encoding="utf-8")),
        encoding="utf-8",
    )


if __name__ == "__main__":
    update_readme_version(__version__)
