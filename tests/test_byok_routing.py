"""Offline BYOK contracts using actual request functions and a mocked SDK transport."""

import ast
import base64
import functools
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import pickle
import tarfile
import tempfile
import time
from types import SimpleNamespace as NS
from typing import List, NamedTuple, Set
import unittest
from unittest.mock import patch
import threading
import copy
from aiohttp import web

from Crypto.Cipher import AES
import httpx
from langchain_openai.chat_models.base import ChatOpenAI
from langchain_openai.embeddings.base import OpenAIEmbeddings
from openai import DefaultHttpxClient

ROOT = Path(__file__).resolve().parents[1]
CALLER_KEY = "synthetic-byok-key-never-valid"
SERVER_KEY = "synthetic-server-key-never-valid"
BASE = "https://internal-provider.fixture/v1"


def load_function(path, name, env):
    """load_function executes repository code with explicitly inert dependencies."""
    tree = ast.parse((ROOT / path).read_text())
    node = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name
    )
    exec(
        compile(
            "from __future__ import annotations\n" + ast.unparse(node),
            str(ROOT / path),
            "exec",
        ),
        env,
    )
    return env[name]


def archive():
    """archive creates a disposable index archive containing synthetic metadata."""
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as bundle:
        payload = pickle.dumps(set())
        info = tarfile.TarInfo("index/scaned_files")
        info.size = len(payload)
        bundle.addfile(info, io.BytesIO(payload))
    return output.getvalue()


class ByokRoutingTests(unittest.TestCase):
    """ByokRoutingTests retains credentials and destinations across cache restoration."""

    def setUp(self):
        """setUp installs synthetic defaults and an in-memory provider transport."""
        self.env_patch = patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": SERVER_KEY,
                "OPENAI_API_BASE": "https://server-default.fixture/v1",
            },
            clear=True,
        )
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.requests = []
        self.logs = io.StringIO()
        self.logger = logging.getLogger("byok-routing-fixture")
        self.logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(self.logs)
        self.logger.addHandler(handler)
        self.addCleanup(self.logger.removeHandler, handler)
        self.client = DefaultHttpxClient(transport=httpx.MockTransport(self.respond))
        self.addCleanup(self.client.close)
        self.prd = NS(
            OPENAI_TOKEN=SERVER_KEY,
            OPENAI_S3_CHUNK_CACHE_BUCKET="fixture",
            OPENAI_S3_EMBEDDINGS_PREFIX="fixture",
            UserPermission=lambda **kw: NS(**kw),
        )
        self.env = dict(
            os=os,
            tempfile=tempfile,
            tarfile=tarfile,
            pickle=pickle,
            NamedTuple=NamedTuple,
            Set=Set,
            prd=self.prd,
            logger=self.logger,
            hashlib=hashlib,
        )
        self.env["OpenAIEmbeddings"] = self.embeddings
        self.env["FAISS"] = NS(
            load_local=lambda **kw: NS(embedding_function=kw["embeddings"])
        )
        self.Index = load_function(
            "ramjet/tasks/gptchat/llm/base.py", "Index", self.env
        )

    def respond(self, request):
        """respond captures synthetic outbound credentials without any sockets."""
        payload = json.loads(request.content)
        self.requests.append(
            (
                str(request.url),
                request.headers.get("Authorization"),
                payload.get("model"),
            )
        )
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(
                200,
                json={
                    "id": "fixture",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "fixture",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "fixture"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                },
            )
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.0, 1.0]}],
                "model": "fixture",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    def embeddings(self, **kwargs):
        """embeddings creates the real LangChain client with a socket-free transport."""
        return OpenAIEmbeddings(
            **kwargs,
            http_client=self.client,
            max_retries=0,
            check_embedding_ctx_length=False,
        )

    def assert_byok(self):
        """assert_byok checks the actual request without exposing synthetic key values."""
        url, key, model = self.requests[-1]
        self.assertEqual(url, BASE + "/embeddings")
        self.assertTrue(key == "Bearer " + CALLER_KEY, "caller credential was replaced")
        self.assertEqual(model, "text-embedding-3-small")
        self.assertNotIn(CALLER_KEY, self.logs.getvalue())
        self.assertNotIn(SERVER_KEY, self.logs.getvalue())

    def test_fresh_chat_preserves_caller_key_and_internal_backend(self):
        """test_fresh_chat exercises the parser, constructor and real SDK request."""
        user = load_function(
            "ramjet/tasks/gptchat/utils.py", "get_user_by_appkey", self.env
        )(
            NS(
                headers={
                    "Authorization": "Bearer " + CALLER_KEY,
                    "X-Laisky-Openai-Api-Base": BASE.removesuffix("/v1"),
                },
                query={},
            )
        )
        env = dict(
            self.env,
            re=__import__("re"),
            ChatOpenAI=functools.partial(
                ChatOpenAI, http_client=self.client, max_retries=0
            ),
        )
        llm = load_function(
            "ramjet/tasks/gptchat/llm/query.py", "build_llm_for_user", env
        )(user)
        llm.invoke("synthetic prompt")
        url, key, _ = self.requests[-1]
        self.assertEqual(url, BASE + "/chat/completions")
        self.assertTrue(key == "Bearer " + CALLER_KEY)
        self.assertNotIn(CALLER_KEY, self.logs.getvalue())

    def test_cached_private_chain_uses_current_request_credentials(self):
        """test_cached_private_chain prevents a prior request key from being reused."""
        cached_index = NS(
            store=NS(
                embedding_function=self.embeddings(
                    api_key=SERVER_KEY, base_url="https://previous-provider.fixture/v1"
                )
            ),
            scaned_files=set(),
        )

        def chain_for_index(index, datasets):
            """chain_for_index supplies an inert retriever with the real SDK client."""

            def search(query):
                """search exercises the bound index embedding client."""
                index.store.embedding_function.embed_query(query)
                return "fixture", []

            return NS(user_index=index, datasets=datasets, search=search)

        env = dict(
            self.env,
            Index=self.Index,
            copy=copy,
            build_user_chain=chain_for_index,
            aiohttp=NS(web=web),
            user_embeddings_chain_mu=threading.Lock(),
            user_embeddings_chain={"synthetic-user": chain_for_index(cached_index, [])},
            uid_ratelimiter=lambda **kw: NS(release=lambda: None),
        )
        source = ROOT / "ramjet/tasks/gptchat/llm/embeddings.py"
        if "def bind_user_chain(" in source.read_text():
            load_function(
                "ramjet/tasks/gptchat/llm/embeddings.py", "bind_user_chain", env
            )
        tree = ast.parse((ROOT / "ramjet/tasks/gptchat/router.py").read_text())
        owner = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "EmbeddingContext"
        )
        method = next(
            n
            for n in owner.body
            if isinstance(n, ast.FunctionDef) and n.name == "chatbot_search"
        )
        method.decorator_list = []
        exec(
            compile(
                "from __future__ import annotations\n" + ast.unparse(method),
                "<actual cached-chain request>",
                "exec",
            ),
            env,
        )
        headers = NS(getone=lambda _: "synthetic-password")
        handler = NS(request=NS(query={"q": "synthetic query"}, headers=headers))
        user = NS(uid="synthetic-user", apikey=CALLER_KEY, api_base=BASE)
        env["chatbot_search"](handler, user)
        self.assert_byok()
        self.assertTrue(
            cached_index.store.embedding_function.openai_api_key.get_secret_value()
            == SERVER_KEY,
            "request must not mutate the shared cached client",
        )

    def test_provider_url_credentials_are_absent_from_logs(self):
        """test_provider_url_credentials rejects disclosure from provider diagnostics."""
        provider = "https://fixture:" + CALLER_KEY + "@internal-provider.fixture/v1"
        env = dict(
            self.env, Index=self.Index, FAISS=NS(from_texts=lambda *args, **kw: NS())
        )
        load_function("ramjet/tasks/gptchat/llm/embeddings.py", "new_store", env)(
            CALLER_KEY, provider
        )
        env.update(
            DEFAULT_MAX_CHUNKS_FOR_FREE=600,
            DEFAULT_CHUNK_SIZE=500,
            DEFAULT_CHUNK_OVERLAP=30,
            time=time,
            chunk_file=lambda **kw: [],
            embed_chunks=lambda **kw: NS(),
            _make_embedding_chunk=lambda **kw: (
                NS(store=NS(similarity_search=lambda *args, **kw: [])),
                True,
            ),
            aiohttp=NS(web=web),
        )
        load_function("ramjet/tasks/gptchat/llm/embeddings.py", "embedding_file", env)(
            "synthetic.txt", "fixture", CALLER_KEY, api_base=provider
        )
        load_function("ramjet/tasks/gptchat/router.py", "_chunk_search", env)(
            "fixture-cache",
            "synthetic query",
            "",
            ".txt",
            CALLER_KEY,
            api_base=provider,
        )
        self.assertFalse(
            CALLER_KEY in self.logs.getvalue(),
            "provider diagnostics exposed a credential",
        )

    def test_sdk_cross_origin_redirect_strips_auth_but_forwards_query_body(self):
        """test_sdk_cross_origin_redirect records current behavior without policy changes."""
        observed = []

        def redirected(request):
            """redirected handles both provider origins entirely in memory."""
            observed.append(
                (
                    request.url.host,
                    request.headers.get("Authorization"),
                    bytes(request.content),
                )
            )
            if request.url.host == "internal-provider.fixture":
                return httpx.Response(
                    307,
                    headers={
                        "Location": "https://redirected-provider.fixture/v1/embeddings"
                    },
                )
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {"object": "embedding", "index": 0, "embedding": [0.0, 1.0]}
                    ],
                    "model": "fixture",
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                },
            )

        with DefaultHttpxClient(transport=httpx.MockTransport(redirected)) as client:
            model = OpenAIEmbeddings(
                api_key=CALLER_KEY,
                base_url=BASE,
                http_client=client,
                max_retries=0,
                check_embedding_ctx_length=False,
            )
            model.embed_query("synthetic redirect query")
        self.assertEqual(len(observed), 2)
        self.assertTrue(observed[0][1] == "Bearer " + CALLER_KEY)
        self.assertIsNone(observed[1][1])
        self.assertEqual(observed[0][2], observed[1][2])

    def test_legacy_server_index_keeps_existing_defaults(self):
        """test_legacy_server_index retains defaults when no caller options exist."""
        index = self.Index.deserialize(archive())
        index.store.embedding_function.embed_query("synthetic legacy query")
        url, key, model = self.requests[-1]
        self.assertEqual(url, "https://server-default.fixture/v1/embeddings")
        self.assertTrue(key == "Bearer " + SERVER_KEY)
        self.assertEqual(model, "text-embedding-ada-002")

    def test_cached_chunk_preserves_key_backend_and_embedding_model(self):
        """test_cached_chunk queries a restored cached index through the real SDK."""
        env = dict(
            self.env,
            Index=self.Index,
            _embedding_chunk_cache=NS(get_cache=lambda _: archive()),
        )
        index, cached = load_function(
            "ramjet/tasks/gptchat/router.py", "_make_embedding_chunk", env
        )(
            "fixture-cache",
            base64.b64encode(b"fixture").decode(),
            ".txt",
            CALLER_KEY,
            api_base=BASE,
        )
        self.assertTrue(cached)
        index.store.embedding_function.embed_query("synthetic query")
        self.assert_byok()

    def test_restored_chatbot_preserves_key_backend_and_model(self):
        """test_restored_chatbot covers encrypted and shared index restoration."""
        for encrypted in (False, True):
            with (
                self.subTest(encrypted=encrypted),
                tempfile.TemporaryDirectory() as directory,
            ):

                def download(**kw):
                    """download writes synthetic S3 objects into the disposable directory."""
                    path = Path(kw["file_path"])
                    payload = pickle.dumps([]) if path.suffix == ".pkl" else archive()
                    if encrypted and path.suffix == ".store":
                        cipher = AES.new(b"0" * 32, AES.MODE_EAX)
                        ciphertext, tag = cipher.encrypt_and_digest(payload)
                        payload = cipher.nonce + tag + ciphertext
                    path.write_bytes(payload)

                env = dict(
                    self.env,
                    Index=self.Index,
                    AES=AES,
                    derive_key=lambda _: b"0" * 32,
                    quote=__import__("urllib.parse", fromlist=["quote"]).quote,
                )
                for name in ("load_encrypt_store", "load_plaintext_store"):
                    load_function("ramjet/tasks/gptchat/llm/embeddings.py", name, env)
                user = NS(uid="synthetic-user", apikey=CALLER_KEY, api_base=BASE)
                _, index, _ = load_function(
                    "ramjet/tasks/gptchat/llm/embeddings.py",
                    "download_chatbot_index",
                    env,
                )(
                    directory,
                    NS(fget_object=download),
                    user,
                    chatbot_name="fixture",
                    password="fixture" if encrypted else "",
                )
                index.store.embedding_function.embed_query("synthetic query")
                self.assert_byok()


if __name__ == "__main__":
    unittest.main()
