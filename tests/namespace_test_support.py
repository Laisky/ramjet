"""Synthetic SSO/registry ownership fixtures with actual request boundaries."""

import ast
import asyncio
import functools
import hashlib
import importlib
import json
import logging
from pathlib import Path
import re
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from aiohttp import web
from multidict import CIMultiDict

from ramjet.sso import WhoAmIVerifier
from ramjet.tasks.gptchat import private_auth, ownership
from ramjet.tasks.gptchat.private_registry import initialize_registry
from tests.test_sso_identity import FakeIssuer, SUBJECT, OTHER, session_token

ROOT = Path(__file__).resolve().parents[1]
KEY_A = "sk-SYNTHETIC-01-A"
KEY_B = "sk-SYNTHETIC-01-B"
PREFIX = "synthetic-embeddings"
BUCKET = "synthetic"


def public_module(relative):
    """public_module returns canonical source types under the public synthetic settings fixture."""
    return importlib.import_module(relative.removesuffix(".py").replace("/", "."))


def definition(relative, name, env, owner=None):
    """definition loads actual source and decorators while replacing only inert external dependencies."""
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


class SyntheticRequest(dict):
    """SyntheticRequest supports real request-local AppKeys and explicit wire fields."""

    def __init__(self, headers, *, query=None, datasets=None, events=None):
        """__init__ installs synthetic headers without any production session or request."""
        super().__init__()
        self.headers = headers
        self.query = {} if query is None else query
        self.match_info = {"op": ""}
        self._datasets = ["private"] if datasets is None else datasets
        self.events = [] if events is None else events

    async def json(self):
        """json records body parsing so denial can be asserted before request effects."""
        self.events.append("json")
        return {"datasets": self._datasets}


class MemoryObjects:
    """MemoryObjects records exact in-memory storage access, including inert historical markers."""

    def __init__(self):
        """__init__ starts with no private records and no external storage connection."""
        self.objects = set()
        self.markers = {}
        self.operations = []

    def list_objects(self, *, bucket_name, prefix, recursive):
        """list_objects records the selected prefix and returns only synthetic matches."""
        self.operations.append(("list", prefix))
        return [
            NS(object_name=name)
            for name in sorted(self.objects | set(self.markers))
            if name.startswith(prefix)
        ]

    def remove_object(self, *, bucket_name, object_name):
        """remove_object records deletion only in the disposable in-memory object set."""
        self.operations.append(("delete", object_name))
        self.objects.discard(object_name)

    def get_object(self, *, bucket_name, object_name):
        """get_object records local reads and never supplies ownership authority."""
        self.operations.append(("get", object_name))
        if object_name not in self.markers:
            error = KeyError("missing synthetic object")
            error.code = "NoSuchKey"
            raise error
        data = self.markers[object_name]
        return NS(
            data=data,
            read=lambda limit: data[:limit],
            close=lambda: None,
            release_conn=lambda: None,
        )

    def put_object(self, *, bucket_name, object_name, data, length, **kwargs):
        """put_object records local bytes, never a trusted registry grant."""
        self.operations.append(("put", object_name))
        self.markers[object_name] = data.read(length)


class OwnershipFixture(unittest.IsolatedAsyncioTestCase):
    """OwnershipFixture uses real issuer verification, registry, decorators and ownership guards."""

    def setUp(self):
        """setUp prepares disposable authority and strictly synthetic service dependencies."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.registry = initialize_registry(
            Path(self.temporary.name) / "operator.sqlite3",
            bucket=BUCKET,
            root_prefix=PREFIX,
            allow_fresh_initialization=True,
        )
        self.issuer = FakeIssuer()
        self.transport_calls = []
        self.response_override = None
        self.transport_error = None

        async def transport(token):
            """transport authenticates public mock tokens or returns selected bounded issuer failures."""
            self.transport_calls.append(token)
            if self.transport_error is not None:
                raise self.transport_error
            if self.response_override is not None:
                return self.response_override
            return await self.issuer.request(token)

        for replacement in (
            patch.object(private_auth, "configured_registry", lambda: self.registry),
            patch.object(private_auth, "SESSION_VERIFIER", WhoAmIVerifier(transport)),
        ):
            replacement.start()
            self.addCleanup(replacement.stop)
        self.storage = MemoryObjects()
        self.buckets = {}
        self.permissions = []
        self.scheduled = []
        self.events = []
        credentials = public_module("ramjet/tasks/gptchat/credentials.py")
        namespace = public_module("ramjet/tasks/gptchat/namespace.py")
        cache = public_module("ramjet/tasks/gptchat/llm/private_cache.py")

        def permission(**fields):
            """permission captures only model-option constructor fields without private authority."""
            self.permissions.append(fields)
            return NS(**fields)

        def schedule(pool, fn):
            """schedule executes inert workers and returns an awaitable local executor receipt."""
            self.scheduled.append(fn)
            value = fn()
            future = asyncio.get_running_loop().create_future()
            future.set_result(value)
            return future

        self.env = {
            "aiohttp": NS(web=web),
            "functools": functools,
            "hashlib": hashlib,
            "resolve_request_credentials": credentials.resolve_request_credentials,
            "prd": NS(UserPermission=permission),
            "settings": NS(
                OPENAI_S3_EMBEDDINGS_PREFIX=PREFIX, OPENAI_S3_CHUNK_CACHE_BUCKET=BUCKET
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
            "get_private_user_chain": lambda storage, user, password: NS(datasets=[]),
            "restore_user_chain": lambda *args, **kwargs: NS(datasets=[]),
            "re": re,
            "dataset_name_regex": re.compile(r"^[a-zA-Z0-9_-]+$"),
            "require_dataset_names": namespace.require_dataset_names,
            "require_object_name": namespace.require_object_name,
            "public_share_owner": namespace.public_share_owner,
            "_require_private_namespace": ownership.require_private_namespace,
            "authenticate_private": private_auth.authenticate_private,
            "authenticate_private_sync": private_auth.authenticate_private_sync,
            "authorize_private_request": private_auth.authorize_private_request,
            "private_context_operation": private_auth.private_context_operation,
            "private_chain_key": cache.private_chain_key,
        }
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
        key=KEY_A,
        *,
        subject=SUBJECT,
        token=None,
        session=True,
        uid="forged-victim",
        free="false",
        datasets=None,
        base="http://provider.fixture",
    ):
        """view supplies independent synthetic session and model credentials with forged wire identity."""
        headers = CIMultiDict(
            {
                "Authorization": key,
                "X-Laisky-User-Id": uid,
                "X-Laisky-User-Is-Free": free,
                "X-Laisky-Openai-Api-Base": base,
            }
        )
        if session:
            value = session_token(sub=subject, uid=subject) if token is None else token
            headers["X-Laisky-SSO-Token"] = "Bearer " + value
        request = SyntheticRequest(headers, datasets=datasets, events=self.events)
        view = NS(request=request)
        view.delete_files = lambda **kw: self.delete_files(view, **kw)
        return view

    async def authorize(self, view, operation, *, initialize=False):
        """authorize exercises real online identity and, optionally, actual fresh upload initialization."""
        if (
            initialize
            and self.registry.lookup(
                session_subject(view.request.headers["X-Laisky-SSO-Token"]), "upload"
            )
            is None
        ):
            upload = self.view(
                view.request.headers.get("Authorization", KEY_A),
                token=view.request.headers["X-Laisky-SSO-Token"].split()[1],
            )
            owner = await private_auth.authorize_private_request(
                upload.request, "upload"
            )
            self.env["require_private_namespace"](
                self.storage, owner, create=True, operation="upload"
            )
        return await private_auth.authorize_private_request(view.request, operation)

    async def user(self, key=KEY_A, *, operation="upload", initialize=True, **kwargs):
        """user returns only a real independently verified request-local private user."""
        return await self.authorize(
            self.view(key, **kwargs), operation, initialize=initialize
        )

    def model_user(self, key=KEY_A, **kwargs):
        """model_user resolves model-only options and grants no private identity or namespace."""
        return self.env["get_user_by_appkey"](
            self.view(key, session=False, **kwargs).request
        )

    def clear_effects(self):
        """clear_effects resets only inert receipts after positive fixture provisioning."""
        self.storage.operations.clear()
        self.scheduled.clear()
        self.events.clear()
        self.transport_calls.clear()


def session_subject(header):
    """session_subject reads only a public fixture claim for optional fixture provisioning."""
    import jwt

    return jwt.decode(header.split()[1], options={"verify_signature": False})["sub"]
