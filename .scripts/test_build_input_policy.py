"""Fast offline contracts for immutable privileged build inputs."""

from pathlib import Path
import hashlib
import sys
import zipfile
import subprocess
import tempfile
import textwrap
import unittest
from unittest.mock import patch

from build_input_policy import (
    bootstrap_errors,
    hashed_bootstrap,
    immutable_base,
    mutable_actions,
    workflow_files,
)

ROOT = Path(__file__).resolve().parents[1]


def workflow_source(steps):
    """workflow_source embeds YAML step text in a complete workflow and returns it."""
    return (
        "name: fixture\non: push\njobs:\n  build:\n"
        "    runs-on: ubuntu-latest\n    steps:\n"
        + textwrap.indent(steps, "      ")
        + "\n"
    )


def repository_policy_result(test_name, files):
    """repository_policy_result runs one real policy against isolated fixture files."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for name, content in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        with patch(__name__ + ".ROOT", root):
            result = unittest.TestResult()
            BuildInputPolicyTests(test_name).run(result)
        return result


class BuildInputPolicyTests(unittest.TestCase):
    """BuildInputPolicyTests rejects mutable repository inputs and negative fixtures."""

    def test_all_external_action_refs_are_full_commits(self):
        """test_all_external_action_refs checks actual workflows without executing actions."""
        for path in workflow_files(ROOT):
            with self.subTest(workflow=path.name):
                self.assertEqual(mutable_actions(path.read_text()), [])

    def test_python_base_manifest_is_immutable(self):
        """test_python_base_manifest checks every external Docker base has a digest."""
        self.assertTrue(immutable_base((ROOT / "Dockerfile").read_text()))

    def test_pdm_bootstrap_is_fully_hashed(self):
        """test_pdm_bootstrap checks hashed requirements and their executable install."""
        path = ROOT / ".scripts/pdm-bootstrap.txt"
        self.assertTrue(
            path.exists(), "frozen hashed bootstrap requirements are required"
        )
        self.assertTrue(hashed_bootstrap(path.read_text()))
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertEqual(bootstrap_errors(dockerfile), [])

    def test_movable_or_truncated_action_refs_fail(self):
        """test_movable_or_truncated_action_refs rejects tag and abbreviated identities."""
        for reference in [
            "actions/checkout@v4",
            "actions/checkout@main",
            "actions/checkout@" + "a" * 39,
        ]:
            source = workflow_source("- uses: " + reference)
            self.assertEqual(mutable_actions(source), [reference])
        self.assertEqual(
            mutable_actions(workflow_source("- uses: actions/checkout@" + "a" * 40)),
            [],
        )

    def test_workflow_formats_and_extensions_reject_movable_refs(self):
        """test_workflow_formats_and_extensions rejects real mutable workflow fixtures."""
        for suffix in ("yml", "yaml"):
            for step in (
                "- uses: actions/checkout@v4",
                "- {uses: actions/checkout@v4}",
                '- "uses": actions/checkout@v4',
            ):
                with self.subTest(suffix=suffix, step=step):
                    result = repository_policy_result(
                        "test_all_external_action_refs_are_full_commits",
                        {f".github/workflows/release.{suffix}": workflow_source(step)},
                    )
                    self.assertEqual(result.errors, [])
                    self.assertTrue(result.failures, "mutable workflow was accepted")

    def test_workflow_formatting_preserves_valid_pins(self):
        """test_workflow_formatting_preserves_valid_pins accepts equivalent YAML values."""
        reference = "actions/checkout@" + "a" * 40
        for step in (
            f'- uses: "{reference}"',
            f"- {{uses: '{reference}'}}",
            f'- "uses": {reference}',
            f"- uses: >-\n    {reference}",
            "- run: |\n    echo 'uses: actions/checkout@v4'",
        ):
            with self.subTest(step=step):
                self.assertEqual(mutable_actions(workflow_source(step)), [])

    def test_action_policy_accepts_local_and_digest_container_actions(self):
        """test_action_policy_accepts_local_and_digest_container_actions preserves types."""
        for reference in (
            "./.github/actions/local",
            "docker://alpine@sha256:" + "a" * 64,
        ):
            with self.subTest(reference=reference):
                self.assertEqual(
                    mutable_actions(workflow_source("- uses: " + reference)), []
                )
        reference = "docker://alpine:latest"
        self.assertEqual(
            mutable_actions(workflow_source("- uses: " + reference)), [reference]
        )

    def test_action_policy_checks_reusable_workflows(self):
        """test_action_policy_checks_reusable_workflows inspects job-level references."""
        reference = "example/reusable/.github/workflows/build.yml@main"
        source = f"jobs: {{build: {{uses: {reference}}}}}"
        self.assertEqual(mutable_actions(source), [reference])
        self.assertEqual(mutable_actions(source.replace("@main", "@" + "a" * 40)), [])

    def test_malformed_workflow_structures_are_reported(self):
        """test_malformed_workflow_structures_are_reported avoids silently skipped jobs."""
        for source in (
            "jobs: [",
            "jobs: []",
            "jobs: {build: []}",
            "jobs: {build: {steps: invalid}}",
            "jobs: {build: {steps: [{uses: null}]}}",
        ):
            with self.subTest(source=source):
                with self.assertRaises(ValueError):
                    mutable_actions(source)

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

    def test_every_external_build_stage_requires_a_digest(self):
        """test_every_external_build_stage requires pins despite keyword formatting."""
        first = "FROM python:3.12.15-bookworm@sha256:" + "a" * 64 + " AS build\n"
        for stage in (
            "from alpine:latest",
            "  FROM alpine:latest",
            "FrOm alpine:latest",
        ):
            with self.subTest(stage=stage):
                self.assertFalse(immutable_base(first + stage))

    def test_digest_policy_allows_version_updates_and_stage_reuse(self):
        """test_digest_policy_allows_version_updates_and_stage_reuse checks identities."""
        pin = "python:3.12.16-bookworm@sha256:" + "a" * 64
        for source in (
            "from " + pin,
            "  FROM --platform=linux/amd64 " + pin,
            "FROM \\\n    " + pin,
            "FROM " + pin + " AS build\nFROM build AS runtime",
            "FROM " + pin + " AS build\nFROM scratch",
        ):
            with self.subTest(source=source):
                self.assertTrue(immutable_base(source))

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

    def test_hashed_bootstrap_allows_reviewed_tool_updates(self):
        """test_hashed_bootstrap_allows_reviewed_tool_updates avoids version coupling."""
        pin = "pdm==2.27.0 --hash=sha256:" + "a" * 64
        self.assertTrue(hashed_bootstrap(pin))
        self.assertTrue(hashed_bootstrap(pin + "  # via reviewed local update\n"))

    def test_bootstrap_install_requires_executable_hash_options_and_lock(self):
        """test_bootstrap_install_requires_executable_hash_options_and_lock rejects bypasses."""
        original = (ROOT / "Dockerfile").read_text()
        secure = next(
            line for line in original.splitlines() if line.startswith("RUN pip")
        )
        mutations = (
            "# --only-binary=:all: --require-hashes\nRUN pip install pdm==2.26.2",
            "RUN echo --only-binary=:all: --require-hashes && pip install pdm==2.26.2",
            "RUN pip install pdm==2.26.2; echo --only-binary=:all: --require-hashes",
            secure.replace("RUN pip", "RUN echo '&&' pip"),
            secure.replace("RUN pip", "RUN printf '%s\\n' '&&' pip"),
            secure + " --only-binar=:none:",
            secure + " --no-binar=pdm",
            secure.replace("/tmp/pdm-bootstrap.txt", "/tmp/other-requirements.txt"),
            "# " + secure,
        )
        for replacement in mutations:
            with self.subTest(replacement=replacement):
                result = repository_policy_result(
                    "test_pdm_bootstrap_is_fully_hashed",
                    {
                        "Dockerfile": original.replace(secure, replacement),
                        ".scripts/pdm-bootstrap.txt": (
                            ROOT / ".scripts/pdm-bootstrap.txt"
                        ).read_text(),
                    },
                )
                self.assertEqual(result.errors, [])
                self.assertTrue(
                    result.failures, "unsafe bootstrap command was accepted"
                )

    def test_bootstrap_lock_must_be_copied_before_install(self):
        """test_bootstrap_lock_must_be_copied_before_install checks actual file provenance."""
        original = (ROOT / "Dockerfile").read_text()
        copy = "COPY .scripts/pdm-bootstrap.txt /tmp/pdm-bootstrap.txt"
        install = next(
            line for line in original.splitlines() if line.startswith("RUN pip")
        )
        for changed in (
            original.replace(copy, "# " + copy),
            original.replace(copy + "\n" + install, install + "\n" + copy),
            original.replace(copy, "COPY another.txt /tmp/pdm-bootstrap.txt"),
        ):
            with self.subTest(dockerfile=changed):
                result = repository_policy_result(
                    "test_pdm_bootstrap_is_fully_hashed",
                    {
                        "Dockerfile": changed,
                        ".scripts/pdm-bootstrap.txt": (
                            ROOT / ".scripts/pdm-bootstrap.txt"
                        ).read_text(),
                    },
                )
                self.assertEqual(result.errors, [])
                self.assertTrue(
                    result.failures, "unavailable reviewed lock was accepted"
                )

    def test_safe_bootstrap_cannot_mask_a_later_unhashed_install(self):
        """test_safe_bootstrap_cannot_mask_a_later_unhashed_install rejects reinstalls."""
        original = (ROOT / "Dockerfile").read_text()
        install = next(
            line for line in original.splitlines() if line.startswith("RUN pip")
        )
        changed = original.replace(install, install + "\nRUN pip install --upgrade pdm")
        result = repository_policy_result(
            "test_pdm_bootstrap_is_fully_hashed",
            {
                "Dockerfile": changed,
                ".scripts/pdm-bootstrap.txt": (
                    ROOT / ".scripts/pdm-bootstrap.txt"
                ).read_text(),
            },
        )
        self.assertEqual(result.errors, [])
        self.assertTrue(result.failures, "unhashed PDM reinstall was accepted")

    def test_equivalent_bootstrap_instructions_remain_supported(self):
        """test_equivalent_bootstrap_instructions_remain_supported preserves valid syntax."""
        original = (ROOT / "Dockerfile").read_text()
        install = next(
            line for line in original.splitlines() if line.startswith("RUN pip")
        )
        copy = "COPY .scripts/pdm-bootstrap.txt /tmp/pdm-bootstrap.txt"
        variants = (
            original.replace(
                install,
                "run python -m pip install --no-cache-dir --only-binary :all: "
                "--require-hashes --requirement /tmp/pdm-bootstrap.txt",
            ),
            original.replace(
                install,
                'RUN ["pip", "install", "--no-cache-dir", "--only-binary=:all:", '
                '"--require-hashes", "-r", "/tmp/pdm-bootstrap.txt"]',
            ).replace(
                copy,
                'COPY [".scripts/pdm-bootstrap.txt", "/tmp/pdm-bootstrap.txt"]',
            ),
            original.replace(
                install,
                "RUN pip install "
                + "\\\n    "
                + "--no-cache-dir --only-binary=:all: "
                + "\\\n    "
                + "--require-hashes -r /tmp/pdm-bootstrap.txt",
            ),
            original.replace(copy, copy + "\nCOPY unrelated.txt /tmp/"),
            original.replace(copy, copy.replace("COPY ", "COPY --link ")),
            original.replace(
                install, install.replace("pip install", "pip --isolated install")
            ),
            original.replace(install, 'RUN echo "a << b"\n' + install),
        )
        for changed in variants:
            with self.subTest(dockerfile=changed):
                result = repository_policy_result(
                    "test_pdm_bootstrap_is_fully_hashed",
                    {
                        "Dockerfile": changed,
                        ".scripts/pdm-bootstrap.txt": (
                            ROOT / ".scripts/pdm-bootstrap.txt"
                        ).read_text(),
                    },
                )
                self.assertEqual(result.errors, [])
                self.assertEqual(result.failures, [])

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
                    [
                        "git",
                        "-c",
                        "commit.gpgsign=false",
                        "-c",
                        "tag.gpgsign=false",
                        "-C",
                        directory,
                        *args,
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
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
