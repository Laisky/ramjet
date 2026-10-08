"""Offline safety and failure contracts for targeted PDM maintenance."""

import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

SOURCE = Path(__file__).with_name("update_dependencies.py")
SPEC = importlib.util.spec_from_file_location("dependency_update", SOURCE)
UPDATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(UPDATE)


class DependencyUpdateTests(unittest.TestCase):
    def setUp(self):
        """setUp creates disposable metadata for isolated maintenance tests."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "pyproject.toml").write_text("[project]\nname = 'fixture'\n")
        (self.root / "requirements.txt").write_text("example==1.0.0\n")
        self.lock("1.0.0")
        (self.root / ".pdm-python").write_text("/preserved/interpreter")
        self.commands = []

    def lock(self, version):
        """lock writes a minimal production fixture for the requested version."""
        (self.root / "pdm.lock").write_text(
            "[[package]]\nname = 'example'\n"
            f"version = '{version}'\ngroups = ['default']\n"
        )

    def runner(self, command, **kwargs):
        """runner simulates PDM commands without network access or installation."""
        self.commands.append(command)
        if command[0] == "git":
            output = ""
        elif command[-1] == "--version":
            output = f"PDM, version {UPDATE.PDM_VERSION}\n"
        else:
            output = ""
            if "update" in command:
                self.assertIn("--no-sync", command)
                self.assertIn("--update-reuse", command)
                constraint = Path(command[command.index("--override") + 1])
                self.assertEqual(constraint.read_text(), "example==2.0.0\n")
                self.lock("2.0.0")
                (self.root / ".pdm-python").write_text("/temporary/interpreter")
            if "export" in command:
                (self.root / "requirements.txt").write_text("example==2.0.0\n")
        return subprocess.CompletedProcess(command, 0, stdout=output)

    def test_plan_has_no_side_effects(self):
        """test_plan_has_no_side_effects requires review before command execution."""
        UPDATE.update(self.root, {"example": "2.0.0"}, run=self.runner)
        self.assertEqual(self.commands, [])
        self.assertEqual(UPDATE.production_pins(self.root), {"example": "1.0.0"})

    def test_apply_updates_lock_and_export_preserving_interpreter(self):
        """test_apply_updates_lock_and_export_preserving_interpreter checks success."""
        UPDATE.update(self.root, {"example": "2.0.0"}, apply=True, run=self.runner)
        self.assertEqual(UPDATE.production_pins(self.root), {"example": "2.0.0"})
        self.assertEqual(
            (self.root / "requirements.txt").read_text(), "example==2.0.0\n"
        )
        self.assertEqual(
            (self.root / ".pdm-python").read_text(), "/preserved/interpreter"
        )

    def test_resolver_failure_rolls_back_without_export(self):
        """test_resolver_failure_rolls_back_without_export retains original bytes."""
        before = {name: (self.root / name).read_bytes() for name in UPDATE.FILES}

        def fail(command, **kwargs):
            """fail simulates a resolver that modifies its lock before failing."""
            result = self.runner(command, **kwargs)
            if "update" in command:
                raise subprocess.CalledProcessError(1, command)
            return result

        with self.assertRaises(subprocess.CalledProcessError):
            UPDATE.update(self.root, {"example": "2.0.0"}, apply=True, run=fail)
        self.assertEqual(
            before, {name: (self.root / name).read_bytes() for name in before}
        )
        self.assertFalse(any("export" in command for command in self.commands))
        self.assertEqual(
            (self.root / ".pdm-python").read_text(), "/preserved/interpreter"
        )

    def test_dirty_metadata_is_never_overwritten(self):
        """test_dirty_metadata_is_never_overwritten protects existing local edits."""

        def dirty(command, **kwargs):
            """dirty reports a local lock edit belonging to another owner."""
            self.commands.append(command)
            return subprocess.CompletedProcess(command, 0, stdout=" M pdm.lock\n")

        with self.assertRaisesRegex(ValueError, "metadata must be clean"):
            UPDATE.update(self.root, {"example": "2.0.0"}, apply=True, run=dirty)
        self.assertEqual(len(self.commands), 1)

    def test_unknown_dependency_is_rejected_before_commands(self):
        """test_unknown_dependency_is_rejected_before_commands limits update scope."""
        with self.assertRaisesRegex(ValueError, "absent from production lock"):
            UPDATE.update(self.root, {"unknown": "2.0.0"}, apply=True, run=self.runner)
        self.assertEqual(self.commands, [])

    def test_only_unique_stable_pins_are_accepted(self):
        """test_only_unique_stable_pins_are_accepted rejects URLs and prereleases."""
        for pins in (
            ["example>=2"],
            ["example==2.0.0rc1"],
            ["--config==2.0.0"],
            ["example @ https://example.invalid"],
            ["Example==2.0.0", "example==2.0.0"],
        ):
            with self.subTest(pins=pins), self.assertRaises(ValueError):
                UPDATE.requested_pins(pins)
        self.assertEqual(
            UPDATE.requested_pins(["Some_Package==2.0.0"]), {"some-package": "2.0.0"}
        )


if __name__ == "__main__":
    unittest.main()
