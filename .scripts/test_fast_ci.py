"""Negative controls for exact packaging-test completion and process exits."""

from pathlib import Path
import sys
import tempfile
import unittest
from fast_ci import run, validate_xml


class GateTests(unittest.TestCase):
    """GateTests ensures missing, skipped and failed execution cannot pass."""

    def test_success(self):
        """test_success accepts one complete explicitly named test."""
        validate_xml(
            '<testsuite><testcase name="required"/></testsuite>', ("required",)
        )

    def test_empty_missing_duplicate_and_unexpected(self):
        """test_empty_missing_duplicate_and_unexpected rejects false discovery."""
        for cases in (
            "",
            '<testcase name="other"/>',
            '<testcase name="required"/>' * 2,
        ):
            with self.assertRaises(ValueError):
                validate_xml(f"<testsuite>{cases}</testsuite>", ("required",))
        with self.assertRaises(ValueError):
            validate_xml("<testsuite/>", ())

    def test_skips_failures_errors(self):
        """test_skips_failures_errors rejects every unsuccessful test outcome."""
        for tag in ("skipped", "failure", "error"):
            with self.assertRaises(ValueError):
                validate_xml(
                    f'<testsuite><testcase name="required"><{tag}/></testcase></testsuite>',
                    ("required",),
                )
        for key in ("skipped", "failures", "errors"):
            with self.assertRaises(ValueError):
                validate_xml(
                    f'<testsuite {key}="1"><testcase name="required"/></testsuite>',
                    ("required",),
                )

    def test_native_failure_and_timeout(self):
        """test_native_failure_and_timeout retains real exits without success."""
        with tempfile.TemporaryDirectory() as directory:
            records = []
            with self.assertRaises(RuntimeError):
                run(
                    [sys.executable, "-c", "raise SystemExit(7)"],
                    Path(directory),
                    "failure",
                    records,
                )
            self.assertEqual(records[-1]["exit_code"], 7)
            with self.assertRaises(RuntimeError):
                run(
                    [sys.executable, "-c", "import time;time.sleep(10)"],
                    Path(directory),
                    "timeout",
                    records,
                    timeout=0.1,
                )
            self.assertIsNone(records[-1]["exit_code"])


if __name__ == "__main__":
    unittest.main()
