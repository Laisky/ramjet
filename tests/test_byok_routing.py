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
from openai import DefaultAsyncHttpxClient, DefaultHttpxClient

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

        def sdk_sync_client(**kwargs):
            client = DefaultHttpxClient(
                transport=httpx.MockTransport(self.respond), **kwargs
            )
            self.addCleanup(client.close)
            return client

        def sdk_async_client(**kwargs):
            import asyncio

            client = DefaultAsyncHttpxClient(
                transport=httpx.MockTransport(self.respond), **kwargs
            )
            self.addCleanup(lambda: asyncio.run(client.aclose()))
            return client

        for name, factory in (
            ("DefaultHttpxClient", sdk_sync_client),
            ("DefaultAsyncHttpxClient", sdk_async_client),
        ):
            client_patch = patch("openai." + name, new=factory)
            client_patch.start()
            self.addCleanup(client_patch.stop)
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
        self.env["aiohttp"] = NS(web=web)
        self.env["urlsplit"] = __import__(
            "urllib.parse", fromlist=["urlsplit"]
        ).urlsplit
        self.env["parse_qsl"] = __import__(
            "urllib.parse", fromlist=["parse_qsl"]
        ).parse_qsl
        self.env["urlunsplit"] = __import__(
            "urllib.parse", fromlist=["urlunsplit"]
        ).urlunsplit
        self.env["DEFAULT_API_BASE"] = "https://api.openai.com/v1"
        self.env["logging"] = logging
        load_function(
            "ramjet/tasks/gptchat/credentials.py", "ModelDiagnosticFilter", self.env
        )
        for logger_name in ("httpx", "openai._base_client"):
            diagnostic_logger = logging.getLogger(logger_name)
            original_filters = list(diagnostic_logger.filters)
            self.addCleanup(setattr, diagnostic_logger, "filters", original_filters)
        for name in (
            "require_api_key",
            "resolve_model_credentials",
            "resolve_request_credentials",
            "resolve_sdk_credentials",
        ):
            load_function("ramjet/tasks/gptchat/credentials.py", name, self.env)
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
        kwargs.setdefault("http_client", self.client)
        return OpenAIEmbeddings(
            **kwargs,
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

    def test_index_missing_key_cannot_use_environment_server_key(self):
        """test_index_missing_key requires explicit credentials before index restoration."""
        with self.assertRaises(ValueError):
            self.Index.deserialize(archive())
        self.assertEqual(self.requests, [])

    def test_missing_model_keys_fail_before_any_upstream_action(self):
        """test_missing_model_keys excludes SDK environment fallback in model helpers."""
        for path, name, kwargs in [
            (
                "ramjet/tasks/gptchat/llm/query.py",
                "build_llm_for_user",
                {
                    "user": NS(
                        apikey=None, api_base=BASE, is_paid=False, chat_model="fixture"
                    )
                },
            ),
            (
                "ramjet/tasks/gptchat/llm/scan.py",
                "summary_docu",
                {
                    "docu": NS(page_content="synthetic document"),
                    "apikey": None,
                    "api_base": BASE,
                },
            ),
            (
                "ramjet/tasks/gptchat/llm/embeddings.py",
                "new_store",
                {"apikey": None, "api_base": BASE},
            ),
        ]:
            with self.subTest(helper=name):
                env = dict(
                    self.env,
                    Index=self.Index,
                    re=__import__("re"),
                    dedent=__import__("textwrap", fromlist=["dedent"]).dedent,
                    ChatOpenAI=functools.partial(
                        ChatOpenAI, http_client=self.client, max_retries=0
                    ),
                    FAISS=NS(from_texts=lambda *args, **kw: NS()),
                )
                with self.assertRaises(ValueError):
                    model = load_function(path, name, env)(**kwargs)
                    if name == "build_llm_for_user":
                        model.invoke("synthetic no-key query")
        self.assertEqual(self.requests, [])

    def test_malformed_request_keys_are_unauthorized(self):
        """test_malformed_request_keys rejects malformed input before profile construction."""
        parser = load_function(
            "ramjet/tasks/gptchat/utils.py", "get_user_by_appkey", self.env
        )
        for header in (
            "",
            "Bearer",
            "Bearer   ",
            "Bearer synthetic key",
            "null",
            "undefined",
        ):
            with self.subTest(header_kind=header.split(" ")[0]):
                with self.assertRaises(web.HTTPUnauthorized):
                    parser(NS(headers={"Authorization": header}, query={}))

    def test_user_embedding_helper_preserves_selected_backend(self):
        """test_user_embedding_helper uses the user's backend in its actual SDK request."""
        model = load_function(
            "ramjet/tasks/gptchat/llm/embeddings.py",
            "build_embeddings_llm_for_user",
            self.env,
        )(NS(apikey=CALLER_KEY, api_base=BASE))
        model.embed_query("synthetic helper query")
        self.assert_byok()

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

    def test_prebuilt_queries_use_request_credentials(self):
        """test_prebuilt_queries replaces startup credentials before every retrieval."""
        index = NS(
            store=NS(
                embedding_function=self.embeddings(
                    api_key=SERVER_KEY, base_url="https://startup-provider.fixture/v1"
                )
            ),
            scaned_files=set(),
        )

        def build_chain(index, datasets):
            def search(question):
                index.store.embedding_function.embed_query(question)
                return "fixture", []

            return NS(
                user_index=index,
                datasets=datasets,
                search=search,
                chain=lambda llm, question: search(question),
            )

        env = dict(
            self.env,
            Index=self.Index,
            copy=copy,
            build_user_chain=build_chain,
            prebuild_chains={"fixture": build_chain(index, [])},
            Response=__import__("collections").namedtuple(
                "Response", ["question", "text", "url"]
            ),
        )
        load_function("ramjet/tasks/gptchat/llm/embeddings.py", "bind_user_chain", env)
        user = NS(apikey=CALLER_KEY, api_base=BASE)
        for name in ("search_for_prebuild_qa", "query_for_prebuild_qa"):
            with self.subTest(operation=name):
                function = load_function("ramjet/tasks/gptchat/llm/query.py", name, env)
                # Execute the baseline signature too, so the reproduction reaches the SDK.
                kwargs = dict(project_name="fixture", question="synthetic query")
                if "user" in __import__("inspect").signature(function).parameters:
                    kwargs["user"] = user
                if name == "query_for_prebuild_qa":
                    kwargs["llm"] = NS()
                function(**kwargs)
                self.assert_byok()

    def test_plaintext_store_excludes_serializable_client_credentials(self):
        """test_plaintext_store persists vector data without the client's key."""
        captured = {}
        store = NS(embedding_function=NS(api_key=CALLER_KEY))
        store.save_local = None
        # A minimal serializable store models legacy/custom embeddings; current SDK
        # clients contain locks and are not assumed to be picklable.
        del store.save_local
        index = NS(store=store, serialize=lambda: archive())

        def upload(**kwargs):
            captured[kwargs["object_name"]] = Path(kwargs["file_path"]).read_bytes()

        env = dict(self.env, quote=__import__("urllib.parse", fromlist=["quote"]).quote)
        load_function(
            "ramjet/tasks/gptchat/llm/embeddings.py", "save_plaintext_store", env
        )(NS(fput_object=upload), NS(uid="synthetic-user"), index, "fixture", [])
        payload = next(
            data for path, data in captured.items() if path.endswith(".store")
        )
        self.assertFalse(
            CALLER_KEY.encode() in payload,
            "serialized embedding client exposed its credential",
        )
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as bundle:
            self.assertIn("index/scaned_files", bundle.getnames())

    def test_background_image_error_does_not_persist_or_log_reflected_key(self):
        """test_background_image_error catches synthetic upstream key reflection."""
        persisted = []
        env = dict(
            self.env,
            io=io,
            aiohttp=NS(web=web),
            timer=lambda function: function,
            settings=self.prd,
            s3cli=NS(put_object=lambda **kw: persisted.append(kw["data"].read())),
            image_objkey=lambda **kw: "fixture/image.png",
        )
        owner = ast.parse((ROOT / "ramjet/tasks/gptchat/router.py").read_text())
        method = next(
            node
            for cls in owner.body
            if isinstance(cls, ast.ClassDef) and cls.name == "Image"
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "catch_and_upload_err"
        )
        method.decorator_list = []
        exec(
            compile(
                "from __future__ import annotations\n" + ast.unparse(method),
                "<actual image background error>",
                "exec",
            ),
            env,
        )

        def fail():
            raise RuntimeError("mock upstream echoed " + CALLER_KEY)

        env["catch_and_upload_err"](NS(), "fixture-task", fail)
        self.assertFalse(
            CALLER_KEY in self.logs.getvalue(), "error log reflected a key"
        )
        self.assertFalse(
            any(CALLER_KEY.encode() in value for value in persisted),
            "persisted error reflected a key",
        )

    def test_summary_missing_key_fails_before_dispatch(self):
        """test_summary_missing_key rejects background work before a task is queued."""
        queued = []
        env = dict(
            self.env, thread_executor=NS(submit=lambda *a, **kw: queued.append(kw))
        )
        function = load_function(
            "ramjet/tasks/gptchat/llm/scan.py", "_get_question_tobe_summary", env
        )
        with self.assertRaises(ValueError):
            function(
                [NS(page_content="synthetic document")], apikey=None, api_base=BASE
            )
        self.assertEqual(queued, [])
        self.assertEqual(self.requests, [])

    def test_real_faiss_serialization_has_no_sdk_credential(self):
        """test_real_faiss_serialization stores vectors and documents, not SDK clients."""
        from langchain_community.vectorstores.faiss import FAISS

        store = FAISS.from_embeddings(
            [("synthetic document", [0.0, 1.0])],
            self.embeddings(api_key=CALLER_KEY, base_url=BASE),
        )
        payload = self.Index(store=store, scaned_files=set()).serialize()
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as bundle:
            for member in bundle.getmembers():
                if member.isfile():
                    self.assertNotIn(
                        CALLER_KEY.encode(), bundle.extractfile(member).read()
                    )
        self.assertEqual(self.requests, [])

    def test_prebuilt_startup_has_no_server_model_client(self):
        """test_prebuilt_startup loads cached vectors without a model credential."""
        from langchain_core.embeddings import Embeddings

        env = dict(self.env, Embeddings=Embeddings)
        guard = load_function(
            "ramjet/tasks/gptchat/llm/data.py", "UnboundEmbeddings", env
        )()
        for operation, value in (
            (guard.embed_query, "synthetic"),
            (guard.embed_documents, ["synthetic"]),
        ):
            with self.assertRaises(ValueError):
                operation(value)
        env["prd"] = NS(OPENAI_EMBEDDING_QA={})
        env["OpenAIEmbeddings"] = lambda **kw: self.fail(
            "startup constructed a model client"
        )
        self.assertEqual(
            load_function(
                "ramjet/tasks/gptchat/llm/data.py", "load_all_prebuild_qa", env
            )(),
            {},
        )
        self.assertEqual(self.requests, [])

    def test_request_error_does_not_reflect_sdk_key(self):
        """test_request_error excludes upstream error text from logs and HTTP responses."""
        import asyncio

        env = dict(self.env, functools=functools, aiohttp=NS(web=web))
        decorator = load_function("ramjet/tasks/gptchat/utils.py", "recover", env)

        async def fail(handler):
            raise RuntimeError("synthetic upstream reflected " + CALLER_KEY)

        response = asyncio.run(decorator(fail)(NS()))
        self.assertEqual(response.status, 400)
        self.assertNotIn(CALLER_KEY, response.text)
        self.assertNotIn(CALLER_KEY, self.logs.getvalue())

    def test_shared_resolver_normalizes_once_and_rejects_malformed_credentials(self):
        """test_shared_resolver keeps explicit provider selection and excludes fallback."""
        request_resolver = self.env["resolve_request_credentials"]
        for base, expected in (
            (
                "http://internal-provider.fixture:1234",
                "http://internal-provider.fixture:1234/v1",
            ),
            (BASE, BASE),
            (BASE + "/", BASE),
            (
                "https://internal-provider.fixture/gateway?tenant=fixture",
                "https://internal-provider.fixture/gateway/v1?tenant=fixture",
            ),
        ):
            with self.subTest(provider=base):
                options = request_resolver("Bearer " + CALLER_KEY, base)
                self.assertEqual(options["base_url"], expected)
                self.assertTrue(options["api_key"] == CALLER_KEY)
        for key in (
            None,
            "",
            "FREETIER",
            "DEFAULT_PROXY_TOKEN",
            "null",
            "undefined",
            "synthetic key",
            "synthetic\nkey",
            "synthetic\x7fkey",
        ):
            with self.subTest(key_kind=type(key).__name__):
                with self.assertRaises(ValueError):
                    self.env["resolve_model_credentials"](key, BASE)
        for header in (
            "Bearer\t" + CALLER_KEY,
            "Bearer \x85" + CALLER_KEY,
            "Bearer " + CALLER_KEY + "\n",
        ):
            with self.assertRaises(ValueError):
                request_resolver(header, BASE)
        for base in (
            "not-a-url",
            "ftp://internal-provider.fixture",
            "http://",
            "http://fixture:invalid",
        ):
            with self.assertRaises(ValueError) as caught:
                request_resolver("Bearer " + CALLER_KEY, base)
            self.assertNotIn(base, str(caught.exception))
        self.assertEqual(self.requests, [])


if __name__ == "__main__":
    unittest.main()
