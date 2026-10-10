"""Offline image generation contracts using actual SDK parsing and caller routes."""

import ast
import asyncio
import base64
import functools
import json
from pathlib import Path
from types import MethodType, SimpleNamespace as NS
import unittest

from aiohttp import web
import httpx
from multidict import CIMultiDict
import openai

from tests import test_byok_routing as routing

IMAGE_BYTES = b"synthetic-png-image"
IMAGE_BASE = "http://127.0.0.1:1337/inference/v1"


class ImageByokTests(unittest.TestCase):
    """ImageByokTests preserve caller credentials, routing and task responses."""

    def setUp(self):
        """setUp binds actual SDK clients to synthetic responses without sockets."""
        self.fixture = routing.ByokRoutingTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.outbound = []
        self.generated = []
        self.downloads = []
        self.image_data = {"b64_json": base64.b64encode(IMAGE_BYTES).decode("ascii")}
        self.fixture.respond = self.respond
        self.env = dict(
            self.fixture.env,
            openai=NS(OpenAI=self.sdk_client, Image=openai.Image),
            b64decode=base64.b64decode,
            httpx=NS(get=self.download),
            requests=NS(get=self.download),
            DEFAULT_API_BASE=self.fixture.env["DEFAULT_API_BASE"],
        )
        self.env["draw_image_by_dalle"] = routing.load_function(
            "ramjet/tasks/gptchat/llm/image.py", "draw_image_by_dalle", self.env
        )

    def sdk_client(self, **kwargs):
        """sdk_client uses the installed SDK with an inert HTTP transport."""
        if "http_client" not in kwargs:
            client = httpx.Client(transport=httpx.MockTransport(self.respond))
            self.addCleanup(client.close)
            kwargs["http_client"] = client
        return openai.OpenAI(**kwargs, max_retries=0)

    def respond(self, request):
        """respond captures the model request and returns a synthetic image."""
        self.outbound.append(
            {
                "url": str(request.url),
                "authorization": request.headers.get("Authorization"),
                "payload": json.loads(request.content),
            }
        )
        return httpx.Response(200, json={"created": 0, "data": [dict(self.image_data)]})

    def download(self, url=None, **kwargs):
        """download models a provider URL without a socket or model credentials."""
        self.downloads.append({"url": url, "options": kwargs})
        return httpx.Response(
            200,
            content=IMAGE_BYTES,
            request=httpx.Request("GET", url),
        )

    def image(self, key=routing.CALLER_KEY, base=IMAGE_BASE):
        """image calls the actual image helper with transient explicit credentials."""
        return self.env["draw_image_by_dalle"](
            prompt="synthetic image prompt", apikey=key, api_base=base
        )

    def method(self, name):
        """method loads a real route method with inert decorators and settings."""
        tree = ast.parse((routing.ROOT / "ramjet/tasks/gptchat/router.py").read_text())
        owner = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Image"
        )
        node = next(
            node
            for node in owner.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        )
        node.decorator_list = []
        exec(
            compile(
                "from __future__ import annotations\n" + ast.unparse(node),
                "<actual image route>",
                "exec",
            ),
            self.env,
        )
        return self.env[name]

    def view(self, header="azure"):
        """view supplies the public image request and captures queued work."""
        queued = []
        self.fixture.prd.S3_SERVER = "https://storage.fixture"
        self.fixture.prd.OPENAI_S3_CHUNK_CACHE_IMAGES = "images"
        self.env.update(
            aiohttp=NS(web=web),
            settings=self.fixture.prd,
            partial=functools.partial,
            uuid1=lambda: "fixture-task",
            image_objkey=lambda **kw: "images/fi/xt/fixture-task.png",
            thread_executor=NS(
                submit=lambda callback, **kw: queued.append((callback, kw))
            ),
            s3cli=NS(),
            upload_image_to_s3=lambda **kw: self.generated.append(kw),
        )
        self.env["draw_image_by_dalle_azure"] = routing.load_function(
            "ramjet/tasks/gptchat/llm/image.py",
            "draw_image_by_dalle_azure",
            self.env,
        )

        async def request_json():
            """request_json returns a synthetic public image payload."""
            return {"prompt": "synthetic image prompt"}

        headers = CIMultiDict(
            {
                "Authorization": "Bearer " + routing.CALLER_KEY,
                "X-Laisky-Openai-Api-Base": IMAGE_BASE + "?tenant=one&flag=",
                "X-Laisky-User-Id": "synthetic-image-owner",
            }
        )
        if header is not None:
            headers["X-Laisky-Image-Token-Type"] = header
        request = NS(
            headers=headers,
            query={},
            match_info={"op": "/dalle"},
            json=request_json,
        )
        view = NS(request=request, catch_and_upload_err=lambda **kw: None)
        view._draw_by_dalle = MethodType(self.method("_draw_by_dalle"), view)
        return view, queued

    def test_supported_sdk_returns_decoded_image_with_explicit_provider(self):
        """test_supported_sdk retains DALL-E2 parameters and image bytes."""
        self.assertEqual(self.image(), IMAGE_BYTES)
        request = self.outbound[-1]
        self.assertEqual(request["url"], IMAGE_BASE + "/images/generations")
        self.assertEqual(request["authorization"], "Bearer " + routing.CALLER_KEY)
        self.assertEqual(
            request["payload"],
            {
                "model": "dall-e-2",
                "prompt": "synthetic image prompt",
                "n": 1,
                "size": "1024x1024",
                "response_format": "b64_json",
            },
        )

    def test_backend_query_survives_sdk_path_construction(self):
        """test_backend_query retains repeated and blank selected-provider values."""
        self.assertEqual(
            self.image(base=IMAGE_BASE + "?tenant=one&tenant=two&flag=#ignored"),
            IMAGE_BYTES,
        )
        self.assertEqual(
            self.outbound[-1]["url"],
            IMAGE_BASE + "/images/generations?tenant=one&tenant=two&flag=",
        )

    def test_missing_key_never_uses_sdk_environment_or_dispatches(self):
        """test_missing_key rejects invalid credentials before constructing SDK work."""
        for key in (None, "", "FREETIER", "default_proxy_token", "bad key", "bad\nkey"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.image(key=key)
        self.assertEqual(self.outbound, [])
        self.assertEqual(self.downloads, [])

    def test_malformed_provider_fails_before_dispatch(self):
        """test_malformed_provider rejects bad syntax without banning private HTTP."""
        for base in ("ftp://fixture", "https://", "https://fixture:invalid"):
            with self.subTest(base=base), self.assertRaises(ValueError):
                self.image(base=base)
        self.assertEqual(self.outbound, [])

    def test_url_only_response_returns_bytes_without_forwarding_model_key(self):
        """test_url_only preserves legacy bytes for compatible URL-only providers."""
        self.image_data = {"url": "https://image-cdn.fixture/image.png"}
        self.assertEqual(self.image(), IMAGE_BYTES)
        self.assertEqual(len(self.downloads), 1)
        self.assertNotIn("headers", self.downloads[0]["options"])
        self.assertNotIn("auth", self.downloads[0]["options"])
        self.assertEqual(self.downloads[0]["options"]["timeout"], 30)

    def test_missing_image_result_fails_without_echoing_credentials(self):
        """test_missing_image produces a generic error for malformed provider output."""
        self.image_data = {}
        with self.assertRaisesRegex(ValueError, "returned no image"):
            self.image()

    def test_caller_keeps_task_response_and_provider_for_all_legacy_headers(self):
        """test_caller routes legacy labels using the authenticated selected backend."""
        for header in (None, "azure", "openai"):
            with self.subTest(header=header):
                view, queued = self.view(header)
                user = routing.load_function(
                    "ramjet/tasks/gptchat/utils.py",
                    "get_user_by_appkey",
                    self.env,
                )(view.request)
                response = asyncio.run(self.method("post")(view, user))
                self.assertEqual(
                    json.loads(response.body),
                    {
                        "task_id": "fixture-task",
                        "image_url": [
                            "https://storage.fixture/fixture/"
                            "images/fi/xt/fixture-task.png"
                        ],
                    },
                )
                self.assertEqual(len(queued), 1)
                queued[0][1]["func"]()
                self.assertEqual(self.generated[-1]["img_content"], IMAGE_BYTES)
                self.assertEqual(
                    self.outbound[-1]["url"],
                    IMAGE_BASE + "/images/generations?tenant=one&flag=",
                )
                self.assertEqual(
                    self.outbound[-1]["authorization"], "Bearer " + routing.CALLER_KEY
                )

    def test_caller_rejects_invalid_credentials_before_background_queue(self):
        """test_caller validates resolved credentials before creating an image job."""
        view, queued = self.view()
        for key, base in (
            ("", IMAGE_BASE),
            (None, IMAGE_BASE),
            (routing.CALLER_KEY, "https://fixture:invalid"),
        ):
            with self.subTest(key=key, base=base), self.assertRaises(ValueError):
                asyncio.run(self.method("post")(view, NS(apikey=key, api_base=base)))
        self.assertEqual(queued, [])
        self.assertEqual(self.outbound, [])

    def test_legacy_azure_helper_requires_an_explicit_backend(self):
        """test_legacy_azure has no hardcoded endpoint or server-key fallback."""
        helper = routing.load_function(
            "ramjet/tasks/gptchat/llm/image.py",
            "draw_image_by_dalle_azure",
            self.env,
        )
        with self.assertRaises(ValueError):
            helper(prompt="synthetic", apikey=routing.CALLER_KEY)
        self.assertEqual(
            helper(prompt="synthetic", apikey=routing.CALLER_KEY, api_base=IMAGE_BASE),
            IMAGE_BYTES,
        )
        self.assertEqual(self.outbound[-1]["url"], IMAGE_BASE + "/images/generations")


if __name__ == "__main__":
    unittest.main()
