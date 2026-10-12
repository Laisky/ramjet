"""Synthetic ownership regressions using actual decorators and storage handlers."""

import ast
import asyncio
import functools
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import re
import threading
from types import SimpleNamespace as NS
import unittest

from aiohttp import web
from multidict import CIMultiDict

ROOT = Path(__file__).resolve().parents[1]
KEY_A = "sk-SYNTHETIC-01-A"
KEY_B = "sk-SYNTHETIC-01-B"
PREFIX = "synthetic-embeddings"


def public_module(relative):
    """public_module loads a pure tracked module without settings or SDK imports."""
    module_name = "ownership_" + Path(relative).stem
    if Path(relative).stem == "ownership":
        import sys
        import types

        package = types.ModuleType("ownership_fixture")
        package.__path__ = []
        sys.modules["ownership_fixture"] = package
        sys.modules["ownership_fixture.namespace"] = public_module(
            "ramjet/tasks/gptchat/namespace.py"
        )
        module_name = "ownership_fixture.ownership"
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def definition(relative, name, env, owner=None):
    """definition executes the actual function including its authentication decorators."""
    tree = ast.parse((ROOT / relative).read_text())
    nodes = tree.body
    if owner:
        nodes = next(
            n for n in nodes if isinstance(n, ast.ClassDef) and n.name == owner
        ).body
    node = next(n for n in nodes if getattr(n, "name", None) == name)
    exec(
        compile(
            "from __future__ import annotations\n" + ast.unparse(node), relative, "exec"
        ),
        env,
    )
    return env[name]


class MemoryObjects:
    """MemoryObjects tracks operations on synthetic private and legacy names."""

    def __init__(self):
        """__init__ creates only synthetic objects and operation receipts."""
        self.objects = set()
        self.markers = {}
        self.operations = []

    def list_objects(self, *, bucket_name, prefix, recursive):
        """list_objects records the selected scope and returns matching local names."""
        self.operations.append(("list", prefix))
        return [
            NS(object_name=n)
            for n in sorted(self.objects | set(self.markers))
            if n.startswith(prefix)
        ]

    def remove_object(self, *, bucket_name, object_name):
        """remove_object records local deletion without contacting object storage."""
        self.operations.append(("delete", object_name))
        self.objects.discard(object_name)

    def get_object(self, *, bucket_name, object_name):
        """get_object supplies local marker proof or the precise missing-object error."""
        if object_name not in self.markers:
            error = KeyError("missing synthetic marker")
            error.code = "NoSuchKey"
            raise error
        data = self.markers[object_name]
        return NS(
            read=lambda limit: data[:limit],
            close=lambda: None,
            release_conn=lambda: None,
        )

    def put_object(self, *, bucket_name, object_name, data, length, **kwargs):
        """put_object persists only synthetic new-namespace proof bytes."""
        self.markers[object_name] = data.read(length)


class OwnershipFixture(unittest.IsolatedAsyncioTestCase):
    """OwnershipFixture binds actual source functions to strictly synthetic dependencies."""

    "OwnershipBoundaryTests excludes forged scopes and preserves same-key access."

    def setUp(self):
        """setUp loads real request functions with inert constructor, storage and scheduler."""
        credentials = public_module("ramjet/tasks/gptchat/credentials.py")
        self.storage = MemoryObjects()
        self.buckets = {}
        self.permissions = []
        self.scheduled = []

        def permission(**fields):
            """permission retains actual public constructor arguments for assertions."""
            self.permissions.append(fields)
            return NS(**fields)

        def schedule(pool, fn):
            """schedule executes only synthetic deletion and records admission."""
            self.scheduled.append(fn)
            return fn()

        self.env = {
            "aiohttp": NS(web=web),
            "functools": functools,
            "hashlib": hashlib,
            "resolve_request_credentials": credentials.resolve_request_credentials,
            "prd": NS(UserPermission=permission),
            "settings": NS(
                OPENAI_S3_EMBEDDINGS_PREFIX=PREFIX,
                OPENAI_S3_CHUNK_CACHE_BUCKET="synthetic",
            ),
            "logger": logging.getLogger("ownership-offline"),
            "s3cli": self.storage,
            "threading": threading,
            "asyncio": NS(get_event_loop=lambda: NS(run_in_executor=schedule)),
            "thread_executor": object(),
            "partial": functools.partial,
            "os": __import__("os"),
            "user_prcess_file_sema": self.buckets,
            "user_prcess_file_sema_lock": threading.RLock(),
            "user_processing_files": {},
            "user_processing_files_lock": threading.RLock(),
            "private_user_chain_cache": NS(get=lambda uid, password: None),
            "restore_user_chain": lambda *args, **kwargs: NS(datasets=[]),
            "re": re,
            "dataset_name_regex": re.compile("^[a-zA-Z0-9_-]+$"),
        }
        namespace_path = ROOT / "ramjet/tasks/gptchat/namespace.py"
        if namespace_path.exists():
            namespace = public_module("ramjet/tasks/gptchat/namespace.py")
            self.env.update(
                credential_scope=namespace.credential_scope,
                require_dataset_names=namespace.require_dataset_names,
            )
            ownership = public_module("ramjet/tasks/gptchat/ownership.py")
            self.env["_require_private_namespace"] = ownership.require_private_namespace
            definition(
                "ramjet/tasks/gptchat/router.py", "require_private_namespace", self.env
            )
        for name in (
            "get_user_by_appkey",
            "authenticate_by_appkey",
            "authenticate_by_appkey_sync",
            "recover",
        ):
            definition("ramjet/tasks/gptchat/utils.py", name, self.env)
        self.env["authenticate"] = self.env["authenticate_by_appkey"]
        self.env["authenticate_sync"] = self.env["authenticate_by_appkey_sync"]
        self.acquire = definition(
            "ramjet/tasks/gptchat/router.py", "uid_ratelimiter", self.env
        )
        self.get = definition(
            "ramjet/tasks/gptchat/router.py", "get", self.env, "UploadedFiles"
        )
        self.delete = definition(
            "ramjet/tasks/gptchat/router.py", "delete", self.env, "UploadedFiles"
        )
        self.delete_files = definition(
            "ramjet/tasks/gptchat/router.py", "delete_files", self.env, "UploadedFiles"
        )

    def view(
        self,
        key,
        uid="forged-victim",
        free="false",
        datasets=None,
        base="http://provider.fixture",
    ):
        """view supplies synthetic wire credentials and intentionally forged identity fields."""

        async def body():
            """body returns inert request JSON for deletion."""
            return {"datasets": datasets or ["private"]}

        headers = CIMultiDict(
            {
                "Authorization": key,
                "X-Laisky-User-Id": uid,
                "X-Laisky-User-Is-Free": free,
                "X-Laisky-Openai-Api-Base": base,
            }
        )
        request = NS(headers=headers, query={}, json=body)
        view = NS(request=request)
        view.delete_files = lambda **kw: self.delete_files(view, **kw)
        return view

    def user(self, key=KEY_A, **kwargs):
        """user selects actual request permission fields without private settings."""
        return self.env["get_user_by_appkey"](self.view(key, **kwargs).request)
