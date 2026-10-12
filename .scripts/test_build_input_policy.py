"""Fast offline contracts for immutable privileged build inputs."""

from pathlib import Path
import re
import hashlib
import sys
import zipfile
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def mutable_actions(source):
    """mutable_actions returns nonlocal workflow refs lacking a full commit identity."""
    refs = re.findall(r"^\s*(?:-\s*)?uses:\s*([^\s#]+)", source, re.M)
    return [
        ref
        for ref in refs
        if not ref.startswith("./") and not re.fullmatch(r"[^@]+@[0-9a-f]{40}", ref)
    ]


def immutable_base(source):
    """immutable_base checks each external Docker base has its explicit manifest digest."""
    bases = re.findall(r"^FROM\s+(?:--platform=\S+\s+)?(\S+)", source, re.M)
    return bool(bases) and all(
        re.fullmatch(r"python:3\.12\.15-bookworm@sha256:[0-9a-f]{64}", base)
        for base in bases
    )


def hashed_bootstrap(source):
    """hashed_bootstrap checks every tool requirement has an exact version and SHA256."""
    lines = source.replace("\\\n", " ").splitlines()
    requirements = [
        line.strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return (
        bool(requirements)
        and all(
            re.fullmatch(
                r"[A-Za-z0-9_.-]+==[0-9][A-Za-z0-9.!+-]*(?:\s+--hash=sha256:[0-9a-f]{64})+",
                line,
            )
            for line in requirements
        )
        and any(line.startswith("pdm==2.26.2 ") for line in requirements)
    )


class BuildInputPolicyTests(unittest.TestCase):
    """BuildInputPolicyTests rejects mutable repository inputs and negative fixtures."""

    def test_all_external_action_refs_are_full_commits(self):
        """test_all_external_action_refs checks actual workflows without executing actions."""
        for path in sorted((ROOT / ".github/workflows").glob("*.yml")):
            with self.subTest(workflow=path.name):
                self.assertEqual(mutable_actions(path.read_text()), [])

    def test_python_base_manifest_is_immutable(self):
        """test_python_base_manifest checks the unchanged Python version has a digest."""
        self.assertTrue(immutable_base((ROOT / "Dockerfile").read_text()))

    def test_pdm_bootstrap_is_fully_hashed(self):
        """test_pdm_bootstrap checks the existing PDM version and every bootstrap artifact."""
        path = ROOT / ".scripts/pdm-bootstrap.txt"
        self.assertTrue(
            path.exists(), "frozen hashed bootstrap requirements are required"
        )
        self.assertTrue(hashed_bootstrap(path.read_text()))
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn("--require-hashes", dockerfile)
        self.assertIn("--only-binary=:all:", dockerfile)
        self.assertIn(".scripts/pdm-bootstrap.txt", dockerfile)

    def test_movable_or_truncated_action_refs_fail(self):
        """test_movable_or_truncated_action_refs rejects tag and abbreviated identities."""
        for reference in [
            "actions/checkout@v4",
            "actions/checkout@main",
            "actions/checkout@" + "a" * 39,
        ]:
            self.assertEqual(mutable_actions("  uses: " + reference), [reference])
            self.assertEqual(mutable_actions("  - uses: " + reference), [reference])
        self.assertEqual(mutable_actions("  uses: actions/checkout@" + "a" * 40), [])

    def test_digestless_or_malformed_base_fails(self):
        """test_digestless_or_malformed_base rejects tag-only or truncated digest bases."""
        for source in [
            "FROM python:3.12.15-bookworm",
            "FROM python:3.12.15-bookworm@sha256:" + "a" * 63,
        ]:
            self.assertFalse(immutable_base(source))
        self.assertTrue(
            immutable_base("FROM python:3.12.15-bookworm@sha256:" + "a" * 64)
        )

    def test_unhashed_or_floating_tool_requirement_fails(self):
        """test_unhashed_or_floating_tool_requirement rejects incomplete bootstrap locks."""
        pin = "pdm==2.26.2 --hash=sha256:" + "a" * 64
        self.assertTrue(hashed_bootstrap(pin))
        for source in [
            pin.replace("==2.26.2", ""),
            pin + "\nhttpx==0.28.1",
            pin.replace("a" * 64, "a" * 63),
            pin.replace("==2.26.2", ">=2.26.2"),
            pin.replace("==2.26.2", "==2.26.*"),
        ]:
            self.assertFalse(hashed_bootstrap(source))

    def test_pip_rejects_changed_wheel_with_reviewed_hash(self):
        """test_pip_rejects_changed_wheel proves hash enforcement on inert local wheels."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = root / "fixture_bootstrap-1.0-py3-none-any.whl"

            def write_wheel(content):
                """write_wheel creates a local inert wheel with no executable installer."""
                with zipfile.ZipFile(wheel, "w") as archive:
                    archive.writestr("fixture_bootstrap/data.txt", content)
                    archive.writestr(
                        "fixture_bootstrap-1.0.dist-info/METADATA",
                        "Metadata-Version: 2.1\nName: fixture-bootstrap\nVersion: 1.0\n",
                    )
                    archive.writestr(
                        "fixture_bootstrap-1.0.dist-info/WHEEL",
                        "Wheel-Version: 1.0\nGenerator: local-fixture\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
                    )
                    archive.writestr("fixture_bootstrap-1.0.dist-info/RECORD", "")

            write_wheel("reviewed benign content")
            reviewed = hashlib.sha256(wheel.read_bytes()).hexdigest()
            requirement = root / "requirements.txt"
            requirement.write_text(
                "fixture-bootstrap==1.0 --hash=sha256:" + reviewed + "\n"
            )
            command = [
                sys.executable,
                "-m",
                "pip",
                "--isolated",
                "download",
                "--no-index",
                "--no-deps",
                "--no-cache-dir",
                "--find-links",
                str(root),
                "-r",
                str(requirement),
            ]
            good = subprocess.run(
                command + ["--require-hashes", "-d", str(root / "good")],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(good.returncode, 0, good.stdout + good.stderr)
            write_wheel("changed benign content")
            bad = subprocess.run(
                command + ["--require-hashes", "-d", str(root / "bad")],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("DO NOT MATCH THE HASHES", bad.stdout + bad.stderr)
            self.assertFalse((root / "bad" / wheel.name).exists())

    def test_same_tag_can_select_different_content_while_commit_stays_stable(self):
        """test_same_tag_can_select_different_content reproduces retargeting locally."""
        with tempfile.TemporaryDirectory() as directory:

            def git(*args):
                """git returns bounded local fixture output without remote or credentials."""
                return subprocess.check_output(
                    ["git", "-C", directory, *args],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()

            git("init", "-q")
            git("config", "user.name", "Local fixture")
            git("config", "user.email", "fixture@example.invalid")
            path = Path(directory) / "input.txt"
            path.write_text("reviewed benign input")
            git("add", "input.txt")
            git("commit", "-qm", "reviewed")
            pinned = git("rev-parse", "HEAD")
            git("tag", "v1")
            before = git("show", "v1:input.txt")
            path.write_text("changed benign input")
            git("add", "input.txt")
            git("commit", "-qm", "changed")
            git("tag", "-f", "v1")
            self.assertNotEqual(git("show", "v1:input.txt"), before)
            self.assertEqual(git("show", pinned + ":input.txt"), before)


if __name__ == "__main__":
    unittest.main()
