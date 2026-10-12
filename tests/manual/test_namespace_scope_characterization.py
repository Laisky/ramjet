"""Opt-in request-scope characterization, not ownership repair or acceptance.

Only public tracked functions run. Permission construction is mocked to capture
selected arguments; no private configuration, provider or storage is loaded.
"""
import ast
import hashlib
import importlib.util
import logging
import os
from pathlib import Path
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[2]


class SyntheticHTTPError(Exception):
    """SyntheticHTTPError substitutes only the handler's error carrier."""

    def __init__(self, *, reason):
        """__init__ retains the selected reason without HTTP or network work."""
        super().__init__(reason)


def tracked_function(relative, name, environment):
    """tracked_function compiles one unchanged public definition without imports."""
    tree = ast.parse((ROOT / relative).read_text())
    selected = [
        n for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    ]
    assert len(selected) == 1
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        *selected,
    ], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), relative, "exec"), environment)
    return environment[name]


@unittest.skipUnless(
    os.environ.get("RAMJET_NAMESPACE_CHARACTERIZATION") == "1",
    "manual characterization; does not assert secure ownership",
)
class NamespaceScopeCharacterization(unittest.TestCase):
    """NamespaceScopeCharacterization records current selectors and isolation controls."""

    def setUp(self):
        """setUp supplies strict synthetic permission and semaphore dependencies."""
        specification = importlib.util.spec_from_file_location(
            "namespace_public_credentials",
            ROOT / "ramjet/tasks/gptchat/credentials.py",
        )
        credentials = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(credentials)
        self.permission = Mock(side_effect=lambda **fields: NS(**fields))
        self.buckets = {}
        environment = {
            "hashlib": hashlib,
            "resolve_request_credentials": credentials.resolve_request_credentials,
            "prd": NS(UserPermission=self.permission),
            "aiohttp": NS(web=NS(
                HTTPUnauthorized=SyntheticHTTPError,
                HTTPTooManyRequests=SyntheticHTTPError,
            )),
            "logger": logging.getLogger("namespace-characterization"),
            "threading": threading,
            "user_prcess_file_sema": self.buckets,
            "user_prcess_file_sema_lock": threading.RLock(),
        }
        self.select = tracked_function(
            "ramjet/tasks/gptchat/utils.py", "get_user_by_appkey", environment,
        )
        self.acquire = tracked_function(
            "ramjet/tasks/gptchat/router.py", "uid_ratelimiter", environment,
        )

    def user(self, uid, free=None, key="sk-synthetic-owner-A"):
        """user invokes the actual selector with syntactic, unverified keys."""
        headers = {"Authorization": "Bearer " + key}
        if uid is not None:
            headers["X-Laisky-User-Id"] = uid
        if free is not None:
            headers["X-Laisky-User-Is-Free"] = free
        return self.select(NS(headers=headers, query={}))

    def test_free_header_changes_selected_paid_argument(self):
        """test_free_header records argument selection, not constructor authority."""
        for value, paid in [(None, True), ("false", True), ("true", False),
                            ("TrUe", False), ("unrecognized", True)]:
            with self.subTest(value=value):
                user = self.user("synthetic-requested-scope", value)
                self.assertEqual(user.uid, "synthetic-requested-scope")
                self.assertEqual(user.is_paid, paid)
                self.assertEqual(user.n_concurrent, 100)
                self.assertEqual(self.permission.call_args.kwargs["is_paid"], paid)

    def test_supplied_uid_overrides_key_but_distinct_uid_stays_distinct(self):
        """test_supplied_uid distinguishes scope selection from key validation."""
        first = self.user("synthetic-scope-A", key="sk-synthetic-owner-A")
        second = self.user("synthetic-scope-A", key="sk-synthetic-owner-B")
        other = self.user("synthetic-scope-B", key="sk-synthetic-owner-B")
        self.assertNotEqual(first.apikey, second.apikey)
        self.assertEqual(first.uid, second.uid)
        self.assertNotEqual(first.uid, other.uid)
        fallback = self.user(None)
        self.assertEqual(fallback.uid, hashlib.sha1(first.apikey.encode()).hexdigest())

    def test_empty_key_rejects_before_permission_construction(self):
        """test_empty_key excludes an indiscriminate accepting fixture."""
        with self.assertRaises(SyntheticHTTPError):
            self.select(NS(headers={}, query={}))
        self.permission.assert_not_called()

    def test_absent_public_bucket_free_users_fall_back_to_uid(self):
        """test_absent_public_bucket disproves blanket shared-free-bucket claims."""
        first = NS(uid="synthetic-A", is_paid=False, n_concurrent=1)
        same = NS(uid="synthetic-A", is_paid=False, n_concurrent=1)
        other = NS(uid="synthetic-B", is_paid=False, n_concurrent=1)
        bucket = self.acquire(first)
        with self.assertRaises(SyntheticHTTPError):
            self.acquire(same)
        separate = self.acquire(other)
        self.assertIsNot(bucket, separate)
        self.assertEqual(set(self.buckets), {"synthetic-A", "synthetic-B"})
        bucket.release()
        separate.release()

    def test_existing_public_bucket_free_users_share_it(self):
        """test_existing_public_bucket proves the preseeded branch separately."""
        public = threading.Semaphore(1)
        self.buckets["public"] = public
        first = NS(uid="synthetic-A", is_paid=False, n_concurrent=1)
        other = NS(uid="synthetic-B", is_paid=False, n_concurrent=1)
        bucket = self.acquire(first)
        self.assertIs(bucket, public)
        with self.assertRaises(SyntheticHTTPError):
            self.acquire(other)
        bucket.release()
        self.assertIs(self.acquire(other), public)
        public.release()
        self.assertEqual(set(self.buckets), {"public"})

    def test_paid_users_ignore_preseeded_public_bucket(self):
        """test_paid_users excludes overclaiming public-bucket scope behavior."""
        public = threading.Semaphore(1)
        self.buckets["public"] = public
        private = self.acquire(NS(uid="synthetic-A", is_paid=True, n_concurrent=1))
        self.assertIsNot(private, public)
        self.assertTrue(public.acquire(blocking=False))
        public.release()
        private.release()


if __name__ == "__main__":
    unittest.main()
