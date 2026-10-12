"""Actual SDK provider-component and diagnostic privacy contracts; no sockets."""

import logging
import unittest
from types import SimpleNamespace as NS

from tests import test_byok_routing as routing


class ProviderTransportTests(unittest.TestCase):
    """Exercise SDK transport options through actual repository model helpers."""

    def setUp(self):
        """Reuse an inert repository/SDK fixture without duplicating its test suite."""
        self.fixture = routing.ByokRoutingTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def model(self, base):
        """Create the actual embedding helper with synthetic request credentials."""
        return routing.load_function(
            "ramjet/tasks/gptchat/llm/embeddings.py",
            "build_embeddings_llm_for_user",
            self.fixture.env,
        )(NS(apikey=routing.CALLER_KEY, api_base=base))

    def test_selected_backend_query_reaches_sync_and_async_requests(self):
        """Keep repeated/blank provider query bytes outside SDK operation paths."""
        import asyncio

        for query in ("tenant=fixture", "tenant=one&tenant=two&flag="):
            with self.subTest(query=query):
                model = self.model(routing.BASE + "?" + query)
                for operation in (
                    lambda: model.embed_query("synthetic query"),
                    lambda: asyncio.run(model.aembed_query("synthetic async query")),
                ):
                    operation()
                    url, key, model_name = self.fixture.requests[-1]
                    self.assertEqual(url, routing.BASE + "/embeddings?" + query)
                    self.assertTrue(key == "Bearer " + routing.CALLER_KEY)
                    self.assertEqual(model_name, "text-embedding-3-small")

    def test_sdk_diagnostics_omit_provider_url_credentials(self):
        """Do not let SDK/httpx diagnostics expose credential-bearing provider URLs."""
        handler = logging.StreamHandler(self.fixture.logs)
        for name in ("httpx", "openai._base_client"):
            logger = logging.getLogger(name)
            previous_level = logger.level
            logger.setLevel(logging.DEBUG)
            logger.addHandler(handler)
            self.addCleanup(logger.removeHandler, handler)
            self.addCleanup(logger.setLevel, previous_level)
        original_respond = self.fixture.respond

        def reflect_header(request):
            response = original_respond(request)
            response.headers["X-Reflected-Key"] = routing.CALLER_KEY
            return response

        self.fixture.respond = reflect_header
        provider = routing.BASE + "?provider_key=" + routing.CALLER_KEY
        self.model(provider).embed_query("synthetic diagnostic query")
        self.assertNotIn(routing.CALLER_KEY, self.fixture.logs.getvalue())
        self.assertIn("status 200", self.fixture.logs.getvalue())

    def test_selected_backend_query_reaches_sync_and_async_chat(self):
        """Exercise actual chat SDK clients with the same selected query options."""
        import asyncio
        import re

        def chat_client(**kwargs):
            kwargs.setdefault("http_client", self.fixture.client)
            return routing.ChatOpenAI(**kwargs, max_retries=0)

        env = dict(self.fixture.env, re=re, ChatOpenAI=chat_client)
        for query in ("tenant=fixture", "tenant=one&tenant=two&flag="):
            with self.subTest(query=query):
                user = NS(
                    apikey=routing.CALLER_KEY,
                    api_base=routing.BASE + "?" + query,
                    is_paid=False,
                    chat_model="fixture",
                )
                model = routing.load_function(
                    "ramjet/tasks/gptchat/llm/query.py", "build_llm_for_user", env
                )(user)
                for operation in (
                    lambda: model.invoke("synthetic query"),
                    lambda: asyncio.run(model.ainvoke("synthetic async query")),
                ):
                    operation()
                    url, key, _ = self.fixture.requests[-1]
                    self.assertEqual(url, routing.BASE + "/chat/completions?" + query)
                    self.assertTrue(key == "Bearer " + routing.CALLER_KEY)
