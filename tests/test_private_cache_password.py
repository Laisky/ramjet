"""Offline regressions for private chatbot password checks across cache states."""

import ast
import asyncio
import functools
import hashlib
import importlib.util
import io
import json
import logging
import os
from pathlib import Path
import pickle
import sys
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from urllib.parse import quote

import aiohttp
from Crypto.Cipher import AES
from multidict import CIMultiDict

ROOT = Path(__file__).resolve().parents[1]
UID = "synthetic-private-user"
PASSWORD_A = "synthetic-password-a"
PASSWORD_B = "synthetic-password-b"


def execute_definitions(path, names, env):
    """execute_definitions loads unmodified named functions without settings imports."""
    tree = ast.parse(path.read_text())
    nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names
    ]
    exec(
        compile(
            "from __future__ import annotations\n"
            + "\n".join(ast.unparse(n) for n in nodes),
            str(path),
            "exec",
        ),
        env,
    )


def execute_method(path, owner, name, env):
    """execute_method loads the actual selected handler with its authentication decorator."""
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
    node = next(n for n in cls.body if getattr(n, "name", None) == name)
    exec(
        compile(
            "from __future__ import annotations\n" + ast.unparse(node),
            str(path),
            "exec",
        ),
        env,
    )
    return env[name]


class MemoryMinio:
    """MemoryMinio substitutes encrypted synthetic chatbot objects for remote storage."""

    def __init__(self):
        """__init__ initializes in-memory objects and download receipts."""
        self.objects = {}
        self.downloads = []

    def get_object(self, bucket_name, object_name):
        """get_object returns a current-pointer response without sockets."""
        return NS(
            data=self.objects[object_name],
            close=lambda: None,
            release_conn=lambda: None,
        )

    def fget_object(self, bucket_name, object_name, file_path, **kwargs):
        """fget_object writes only synthetic bytes to the test's temporary directory."""
        self.downloads.append(object_name)
        Path(file_path).write_bytes(self.objects[object_name])

    def fput_object(self, bucket_name, object_name, file_path):
        """fput_object retains newly encrypted fixture bytes in memory."""
        self.objects[object_name] = Path(file_path).read_bytes()

    def put_object(self, bucket_name, object_name, data, length):
        """put_object retains a synthetic current pointer."""
        self.objects[object_name] = data.read(length)

    def list_objects(self, bucket_name, prefix, recursive):
        """list_objects lists fixture object names under the legacy prefix."""
        return [
            NS(object_name=name) for name in self.objects if name.startswith(prefix)
        ]


class SyntheticIndex:
    """SyntheticIndex substitutes inert vector data after actual AES authentication."""

    def __init__(self, payload):
        """__init__ retains synthetic decoded data."""
        self.payload = payload

    def serialize(self):
        """serialize returns fixture bytes without a model or vector call."""
        return json.dumps(self.payload).encode()

    @classmethod
    def deserialize(cls, data, **kwargs):
        """deserialize runs only after actual authenticated decrypt has succeeded."""
        return cls(json.loads(data))


class PrivateCachePasswordTests(unittest.IsolatedAsyncioTestCase):
    """PrivateCachePasswordTests exercise actual cold decrypt and warm handlers."""

    def setUp(self):
        """setUp loads real code with explicitly inert storage and model dependencies."""
        self.storage = MemoryMinio()
        self.user = NS(
            uid=UID,
            apikey="sk-synthetic-never-valid",
            api_base="https://provider.fixture/v1",
            is_paid=True,
            n_concurrent=3,
        )
        self.env = dict(
            hashlib=hashlib,
            AES=AES,
            os=os,
            pickle=pickle,
            tempfile=tempfile,
            quote=quote,
            threading=threading,
            functools=functools,
            logger=logging.getLogger("private-cache-password-fixture"),
            aiohttp=aiohttp,
            prd=NS(
                OPENAI_TOKEN="synthetic-old-server-key",
                OPENAI_S3_EMBEDDINGS_PREFIX="fixture",
                OPENAI_S3_CHUNK_CACHE_BUCKET="fixture",
            ),
            settings=NS(
                OPENAI_S3_EMBEDDINGS_PREFIX="fixture",
                OPENAI_S3_CHUNK_CACHE_BUCKET="fixture",
            ),
            user_embeddings_chain={},
            user_embeddings_chain_mu=threading.RLock(),
            user_shared_chain={},
            user_shared_chain_mu=threading.RLock(),
            user_processing_files={},
            user_processing_files_lock=threading.RLock(),
            user_prcess_file_sema={},
            user_prcess_file_sema_lock=threading.RLock(),
            Index=SyntheticIndex,
            s3cli=self.storage,
            io=io,
            build_llm_for_user=lambda user: NS(),
        )

        def build_user_chain(index, datasets):
            """build_user_chain supplies deterministic retrieval after actual decryption."""
            return NS(
                user_index=index,
                datasets=datasets,
                search=lambda query: (index.payload["text"], ["synthetic-reference"]),
                chain=lambda llm, query: (
                    index.payload["text"],
                    ["synthetic-reference"],
                ),
            )

        self.env["build_user_chain"] = build_user_chain
        gpt = ROOT / "ramjet/tasks/gptchat"
        execute_definitions(gpt / "llm/embeddings.py", ["derive_key"], self.env)
        cache_module = gpt / "llm/private_cache.py"
        if cache_module.exists():
            spec = importlib.util.spec_from_file_location(
                "private_cache_under_test", cache_module
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            self.env["private_user_chain_cache"] = module.PasswordProtectedCache(
                self.env["user_embeddings_chain"],
                self.env["user_embeddings_chain_mu"],
                self.env["derive_key"],
            )
        execute_definitions(
            gpt / "llm/embeddings.py",
            [
                "load_encrypt_store",
                "load_plaintext_store",
                "download_chatbot_index",
                "restore_user_chain",
                "get_private_user_chain",
                "save_encrypt_store",
            ],
            self.env,
        )
        execute_definitions(gpt / "router.py", ["uid_ratelimiter"], self.env)

        def authenticate(func):
            """authenticate preserves the handler shape without claiming UID ownership."""

            @functools.wraps(func)
            def wrapper(view, *args, **kwargs):
                return func(view, self.user, *args, **kwargs)

            return wrapper

        self.env["authenticate_sync"] = authenticate
        self.env["authenticate"] = lambda func: func
        self.env["recover"] = lambda func: func
        self.search = execute_method(
            gpt / "router.py", "EmbeddingContext", "chatbot_search", self.env
        )
        self.query = execute_method(
            gpt / "router.py", "EmbeddingContext", "chatbot_query", self.env
        )
        self.list_files = execute_method(
            gpt / "router.py", "UploadedFiles", "get", self.env
        )
        self.build = execute_method(
            gpt / "router.py", "EmbeddingContext", "_build_user_chatbot", self.env
        )
        for name, password in [("alpha", PASSWORD_A), ("beta", PASSWORD_B)]:
            payload = json.dumps({"text": name + " private content"}).encode()
            cipher = AES.new(self.env["derive_key"](password), AES.MODE_EAX)
            ciphertext, tag = cipher.encrypt_and_digest(payload)
            prefix = "fixture/" + UID + "/chatbot-v2/" + name
            self.storage.objects[prefix + ".store"] = cipher.nonce + tag + ciphertext
            self.storage.objects[prefix + ".pkl"] = pickle.dumps([name + " dataset"])
        self.storage.objects["fixture/" + UID + "/chatbot-v2/__CURRENT"] = b"alpha"

    def view(self, password):
        """view creates an inert request with the original password header and query."""
        return NS(
            request=NS(
                headers=CIMultiDict({"X-PDFCHAT-PASSWORD": password}),
                query={"q": "fixture"},
            )
        )

    def restore(self, password=PASSWORD_A, name="alpha"):
        """restore executes actual authenticated cold loading and installs its cache entry."""
        return self.env["restore_user_chain"](self.storage, self.user, password, name)

    def assert_content(self, method, password, text):
        """assert_content checks the existing JSON response shape and retrieval result."""
        response = method(self.view(password))
        self.assertEqual(response.status, 200)
        self.assertEqual(
            json.loads(response.body), {"text": text, "url": ["synthetic-reference"]}
        )

    async def test_cold_wrong_password_fails_before_retrieval(self):
        """test_cold_wrong_password preserves actual AES authentication failure."""
        for method in [self.search, self.query]:
            with self.subTest(method=method.__name__):
                self.env["user_embeddings_chain"].clear()
                with self.assertRaises(ValueError):
                    method(self.view("synthetic-wrong"))

    async def test_warm_wrong_password_matches_cold_denial(self):
        """test_warm_wrong_password denies the same wrong credential after a legitimate cache warm."""
        self.restore()
        for method in [self.search, self.query]:
            with self.subTest(method=method.__name__):
                with self.assertRaises(ValueError):
                    method(self.view("synthetic-wrong"))

    async def test_correct_cold_and_warm_password_preserves_response_and_cache(self):
        """test_correct_cold_and_warm preserves retrieval and avoids repeat object downloads."""
        self.assert_content(self.search, PASSWORD_A, "alpha private content")
        downloads = list(self.storage.downloads)
        self.assert_content(self.search, PASSWORD_A, "alpha private content")
        self.assert_content(self.query, PASSWORD_A, "alpha private content")
        self.assertEqual(self.storage.downloads, downloads)

    async def test_sibling_password_cannot_read_active_private_cache(self):
        """test_sibling_password rejects alpha's password while beta is the cached active bot."""
        self.restore(PASSWORD_A, "alpha")
        self.restore(PASSWORD_B, "beta")
        downloads = list(self.storage.downloads)
        for method in [self.search, self.query]:
            with self.subTest(method=method.__name__):
                with self.assertRaises(ValueError):
                    method(self.view(PASSWORD_A))
        self.assertEqual(self.storage.downloads, downloads)
        self.assert_content(self.search, PASSWORD_B, "beta private content")

    async def test_unproven_legacy_cache_entry_requires_cold_password_proof(self):
        """test_unproven_legacy_cache rejects plaintext entries without successful-decrypt proof."""
        self.env["user_embeddings_chain"][UID] = self.env["build_user_chain"](
            SyntheticIndex({"text": "unproven cached content"}), ["unproven dataset"]
        )
        with self.assertRaises(ValueError):
            self.search(self.view("synthetic-wrong"))
        self.assert_content(self.search, PASSWORD_A, "alpha private content")

    async def test_selected_private_metadata_requires_correct_password_on_warm_cache(
        self,
    ):
        """test_selected_private_metadata applies the cold check to cached selected datasets."""
        self.restore()
        for password in ["", "synthetic-wrong", PASSWORD_B]:
            with self.subTest(password_label="invalid"):
                response = await self.list_files(self.view(password), self.user)
                self.assertEqual(json.loads(response.body)["selected"], [])
        response = await self.list_files(self.view(PASSWORD_A), self.user)
        self.assertEqual(json.loads(response.body)["selected"], ["alpha dataset"])

    async def test_missing_private_password_remains_rejected(self):
        """test_missing_private_password preserves the existing required-header semantics."""
        self.restore()
        for method in [self.search, self.query]:
            with self.assertRaises((AssertionError, ValueError)):
                method(self.view(""))

    async def test_failed_cold_restore_does_not_replace_valid_cached_chain(self):
        """test_failed_cold_restore keeps a successfully decrypted active entry intact."""
        self.restore(PASSWORD_B, "beta")
        before = self.env["user_embeddings_chain"][UID]
        with self.assertRaises(ValueError):
            self.restore("synthetic-wrong", "alpha")
        self.assertIs(self.env["user_embeddings_chain"][UID], before)
        self.assert_content(self.search, PASSWORD_B, "beta private content")

    async def test_newly_built_chain_also_has_password_proof(self):
        """test_newly_built_chain checks that the second production cache-write path is protected."""
        view = NS(
            load_datasets=lambda **kw: SyntheticIndex({"text": "new private content"})
        )
        self.build(view, self.user, PASSWORD_A, ["alpha dataset"], "new")
        self.assert_content(self.search, PASSWORD_A, "new private content")
        with self.assertRaises(ValueError):
            self.search(self.view(PASSWORD_B))

    async def test_public_shared_restore_remains_passwordless(self):
        """test_public_shared_restore leaves deliberately shared plaintext behavior unchanged."""
        prefix = "fixture/" + UID + "/chatbot-share-v2/public"
        self.storage.objects[prefix + ".store"] = json.dumps(
            {"text": "public content"}
        ).encode()
        self.storage.objects[prefix + ".pkl"] = pickle.dumps(["public dataset"])
        self.env["restore_user_chain"](self.storage, self.user, chatbot_name="public")
        chain = self.env["user_shared_chain"][UID + "public"]
        self.assertEqual(chain.search("fixture")[0], "public content")

    async def test_cache_proof_never_enters_serialized_index(self):
        """test_cache_proof confirms existing persisted payload and namespace format stay unchanged."""
        self.restore()
        payload = self.env["user_embeddings_chain"][UID].user_index.serialize()
        self.assertEqual(json.loads(payload), {"text": "alpha private content"})
        self.assertNotIn(PASSWORD_A.encode(), payload)

    async def test_replaced_chain_cannot_reuse_a_sibling_password_proof(self):
        """test_replaced_chain checks that proof for alpha never authorizes substituted plaintext."""
        self.restore()
        self.env["user_embeddings_chain"][UID] = self.env["build_user_chain"](
            SyntheticIndex({"text": "substituted sibling content"}), ["sibling dataset"]
        )
        downloads = len(self.storage.downloads)
        self.assert_content(self.search, PASSWORD_A, "alpha private content")
        self.assertGreater(len(self.storage.downloads), downloads)

    async def test_concurrent_replacement_uses_the_authenticated_local_snapshot(self):
        """test_concurrent_replacement prevents reading another chain after checking alpha's proof."""
        cache = self.env.get("private_user_chain_cache")
        if cache is None:
            self.skipTest("cache proof is introduced by this fix")
        self.restore()
        derive = cache._derive_key
        replaced = False

        def replace_during_password_check(password):
            """replace_during_password_check simulates a sibling writer after proof snapshotting."""
            nonlocal replaced
            if not replaced:
                replaced = True
                self.restore(PASSWORD_B, "beta")
            return derive(password)

        from unittest.mock import patch

        with patch.object(
            cache, "_derive_key", side_effect=replace_during_password_check
        ):
            self.assert_content(self.search, PASSWORD_A, "alpha private content")
        self.assertEqual(
            self.env["user_embeddings_chain"][UID].datasets, ["beta dataset"]
        )
        with self.assertRaises(ValueError):
            self.search(self.view(PASSWORD_A))
        self.assert_content(self.search, PASSWORD_B, "beta private content")
