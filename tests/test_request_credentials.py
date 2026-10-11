"""Standard-library contracts for the independent request credential API."""

import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from ramjet.credentials import (
    DEFAULT_API_BASE,
    require_api_key,
    resolve_model_credentials,
    resolve_request_credentials,
)

KEY = "synthetic-caller-key"
BASE = "http://127.0.0.1:17980/selected/v1"


class RequestCredentialContracts(unittest.TestCase):
    """RequestCredentialContracts checks validation without task or SDK imports."""

    def test_root_import_does_not_load_settings_tasks_or_change_environment(self):
        """test_root_import_does_not_load_settings_tasks_or_change_environment checks an isolated interpreter."""
        code = """
import importlib.abc, os, sys
class RejectRuntimeImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("ramjet.tasks", "ramjet.settings")):
            raise AssertionError("runtime import forbidden")
sys.meta_path.insert(0, RejectRuntimeImports())
before = dict(os.environ)
from ramjet.credentials import resolve_request_credentials
assert dict(os.environ) == before
assert not any(name.startswith(("ramjet.tasks", "ramjet.settings")) for name in sys.modules)
assert resolve_request_credentials("synthetic-caller-key", "http://127.0.0.1:9/selected")["base_url"] == "http://127.0.0.1:9/selected/v1"
"""
        environment = {
            "PATH": os.defpath,
            "OPENAI_API_KEY": KEY,
            "OPENAI_API_BASE": BASE,
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        result = subprocess.run(
            [sys.executable, "-S", "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_key_validation_never_uses_environment_fallback(self):
        """test_key_validation_never_uses_environment_fallback rejects malformed caller keys."""
        invalid = (
            None,
            "",
            "  ",
            "bearer",
            "null",
            "undefined",
            "default_proxy_token",
            "FREETIER",
            "synthetic key",
            "synthetic\nkey",
            "synthetic\x7fkey",
        )
        with patch.dict(os.environ, {"OPENAI_API_KEY": "synthetic-server-key"}):
            for key in invalid:
                with self.subTest(kind=type(key).__name__):
                    with self.assertRaises(ValueError) as caught:
                        require_api_key(key)
                    self.assertNotIn("synthetic-server-key", str(caught.exception))
        self.assertEqual(require_api_key(KEY), KEY)

    def test_request_headers_and_backend_bytes_remain_compatible(self):
        """test_request_headers_and_backend_bytes_remain_compatible preserves configured destinations."""
        providers = (
            ("http://internal.fixture:1234", "http://internal.fixture:1234/v1"),
            (BASE, BASE),
            (BASE + "/", BASE),
            (
                "https://fixture/gateway?tenant=one&tenant=&tenant=two#local",
                "https://fixture/gateway/v1?tenant=one&tenant=&tenant=two#local",
            ),
        )
        for authorization in (KEY, "Bearer " + KEY, "bearer " + KEY):
            for base, expected in providers:
                with self.subTest(header_kind=authorization.split()[0], provider=base):
                    options = resolve_request_credentials(authorization, base)
                    self.assertEqual(options, {"api_key": KEY, "base_url": expected})
                    self.assertNotIn(KEY, options["base_url"])
        self.assertEqual(resolve_request_credentials(KEY)["base_url"], DEFAULT_API_BASE)

    def test_malformed_headers_and_urls_fail_without_echoing_values(self):
        """test_malformed_headers_and_urls_fail_without_echoing_values excludes sensitive error details."""
        for header in (
            "Bearer\t" + KEY,
            "Bearer " + KEY + "\n",
            "Basic " + KEY,
            "Bearer \x85" + KEY,
        ):
            with self.assertRaises(ValueError) as caught:
                resolve_request_credentials(header, BASE)
            self.assertNotIn(KEY, str(caught.exception))
        for base in (
            "not-a-url",
            "ftp://internal.fixture",
            "http://",
            "http://fixture:invalid",
            "https://fixture/\nprivate",
        ):
            with self.assertRaises(ValueError) as caught:
                resolve_model_credentials(KEY, base)
            self.assertNotIn(base, str(caught.exception))
            self.assertNotIn(KEY, str(caught.exception))
        self.assertEqual(
            resolve_model_credentials(KEY, None)["base_url"], DEFAULT_API_BASE
        )


if __name__ == "__main__":
    unittest.main()
