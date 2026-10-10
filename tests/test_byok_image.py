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
            re=__import__("re"),
        )
        self.env["resolve_image_parameters"] = routing.load_function(
            "ramjet/tasks/gptchat/llm/image.py", "resolve_image_parameters", self.env
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

    def image(
        self, key=routing.CALLER_KEY, base=IMAGE_BASE, model="dall-e-2", profile=None
    ):
        """image calls the actual image helper with transient explicit credentials."""
        return self.env["draw_image_by_dalle"](
            prompt="synthetic image prompt",
            apikey=key,
            api_base=base,
            model=model,
            image_profile=profile,
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

    def view(self, header="azure", payload=None):
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
            return (
                payload
                if payload is not None
                else {"prompt": "synthetic image prompt", "model": "dall-e-2"}
            )

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
            helper(
                prompt="synthetic",
                apikey=routing.CALLER_KEY,
                api_base=IMAGE_BASE,
                model="dall-e-2",
            ),
            IMAGE_BYTES,
        )
        self.assertEqual(self.outbound[-1]["url"], IMAGE_BASE + "/images/generations")

    def test_standard_openai_default_uses_current_pinned_gpt_image_parameters(self):
        """test_standard_default uses the explicit key and supported PNG profile."""
        self.assertEqual(
            self.image(base="https://api.openai.com/v1", model=None), IMAGE_BYTES
        )
        request = self.outbound[-1]
        self.assertEqual(request["url"], "https://api.openai.com/v1/images/generations")
        self.assertEqual(request["authorization"], "Bearer " + routing.CALLER_KEY)
        self.assertEqual(
            request["payload"],
            {
                "model": "gpt-image-2-2026-04-21",
                "prompt": "synthetic image prompt",
                "n": 1,
                "size": "1024x1024",
                "output_format": "png",
                "quality": "low",
            },
        )
        self.assertNotIn("response_format", request["payload"])

    def test_custom_provider_requires_explicit_model_without_fallback(self):
        """test_custom_missing_model rejects ambiguous capability without dispatch."""
        for base in (
            IMAGE_BASE,
            "https://api.openai.com.evil.fixture/v1",
            "http://api.openai.com/v1",
            "https://api.openai.com:444/v1",
            "https://api.openai.com/alternate/v1",
            "https://api.openai.com/v1?tenant=one",
        ):
            with (
                self.subTest(base=base),
                self.assertRaisesRegex(ValueError, "explicit model"),
            ):
                self.image(base=base, model=None)
        self.assertEqual(self.outbound, [])

    def test_custom_models_use_declared_profiles_and_unchanged_backend(self):
        """test_custom_profiles honors explicit model and capability parameters."""
        for model, profile, response_format in (
            ("custom/image-v4", "legacy", True),
            ("custom/image-v4", "gpt-image", False),
            ("gpt-image-custom-fixture", "gpt-image", False),
            ("gpt-image-2.5-flare", None, False),
            ("dall-e-3", None, True),
        ):
            with self.subTest(model=model, profile=profile):
                self.assertEqual(self.image(model=model, profile=profile), IMAGE_BYTES)
                request = self.outbound[-1]
                self.assertEqual(request["url"], IMAGE_BASE + "/images/generations")
                self.assertEqual(request["payload"]["model"], model)
                self.assertEqual(
                    "response_format" in request["payload"], response_format
                )
                self.assertEqual(
                    "output_format" in request["payload"], not response_format
                )

    def test_invalid_or_conflicting_model_profile_fails_before_dispatch(self):
        """test_model_profile rejects invalid options without echoing input values."""
        cases = [
            ("", None),
            (42, None),
            ("secret model", None),
            ("custom-model", None),
            ("gpt-image-custom-fixture", None),
            ("custom-model", "invalid"),
            ("custom-model", {}),
            ("gpt-image-2", "legacy"),
            ("dall-e-2", "gpt-image"),
        ]
        for model, profile in cases:
            with (
                self.subTest(model=model, profile=profile),
                self.assertRaises(ValueError),
            ):
                self.image(model=model, profile=profile)
        for model in ("dall-e-2", "dall-e-3"):
            with (
                self.subTest(model=model),
                self.assertRaisesRegex(ValueError, "Retired"),
            ):
                self.image(base="https://api.openai.com/v1", model=model)
        self.assertEqual(self.outbound, [])

    def test_caller_configuration_error_precedes_queue_and_is_actionable(self):
        """test_caller_configuration reports missing custom model before submission."""
        view, queued = self.view(payload={"prompt": "synthetic"})
        with self.assertRaises(web.HTTPBadRequest) as raised:
            asyncio.run(
                self.method("post")(
                    view, NS(apikey=routing.CALLER_KEY, api_base=IMAGE_BASE)
                )
            )
        self.assertEqual(raised.exception.status, 400)
        self.assertIn(
            "Custom providers require an explicit model", raised.exception.text
        )
        self.assertEqual(queued, [])
        self.assertEqual(self.outbound, [])

    def test_caller_passes_explicit_custom_model_and_profile_to_actual_sdk(self):
        """test_caller_selection preserves public shape and propagates configuration."""
        view, queued = self.view(
            payload={
                "prompt": "synthetic image prompt",
                "model": "custom/image-v4",
                "image_profile": "gpt-image",
            }
        )
        user = NS(apikey=routing.CALLER_KEY, api_base=IMAGE_BASE, uid="fixture-owner")
        response = asyncio.run(self.method("post")(view, user))
        self.assertIsInstance(json.loads(response.body)["image_url"], list)
        queued[0][1]["func"]()
        self.assertEqual(self.outbound[-1]["payload"]["model"], "custom/image-v4")
        self.assertEqual(self.outbound[-1]["payload"]["output_format"], "png")
        self.assertNotIn("response_format", self.outbound[-1]["payload"])
        self.assertEqual(
            self.outbound[-1]["authorization"], "Bearer " + routing.CALLER_KEY
        )

    def test_standard_prompt_only_caller_keeps_public_array_and_current_model(self):
        """test_standard_caller propagates default selection through the actual route."""
        view, queued = self.view(payload={"prompt": "synthetic image prompt"})
        view.request.headers["X-Laisky-Openai-Api-Base"] = "https://api.openai.com/v1"
        user = routing.load_function(
            "ramjet/tasks/gptchat/utils.py", "get_user_by_appkey", self.env
        )(view.request)
        response = asyncio.run(self.method("post")(view, user))
        self.assertEqual(json.loads(response.body)["task_id"], "fixture-task")
        self.assertIsInstance(json.loads(response.body)["image_url"], list)
        queued[0][1]["func"]()
        self.assertEqual(
            self.outbound[-1]["payload"]["model"], "gpt-image-2-2026-04-21"
        )
        self.assertNotIn("response_format", self.outbound[-1]["payload"])
        self.assertEqual(
            self.outbound[-1]["authorization"], "Bearer " + routing.CALLER_KEY
        )

    def test_url_download_redirect_uses_real_httpx_without_model_auth(self):
        """test_download_redirect keeps model credentials out of both actual requests."""
        requests = []

        def respond(request):
            """respond redirects across synthetic origins and records download headers."""
            requests.append(request)
            if request.url.host == "image-cdn.fixture":
                return httpx.Response(
                    302, headers={"Location": "https://other-cdn.fixture/image.png"}
                )
            return httpx.Response(200, content=IMAGE_BYTES)

        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            self.env["httpx"] = NS(get=client.get)
            self.image_data = {"url": "https://image-cdn.fixture/image.png"}
            self.assertEqual(self.image(), IMAGE_BYTES)
        self.assertEqual(len(requests), 2)
        for request in requests:
            self.assertNotIn("Authorization", request.headers)
            self.assertNotIn(routing.CALLER_KEY, str(request.url))

    def test_configuration_error_never_reflects_exception_details(self):
        """test_configuration_error excludes key, provider URL and trace details."""
        reflected = "synthetic-secret-key https://private.fixture/?token=synthetic-secret-key Traceback"

        def fail(*args, **kwargs):
            """fail simulates exception details from a configuration dependency."""
            raise ValueError(reflected)

        self.env["resolve_image_parameters"] = fail
        view, queued = self.view()
        user = NS(apikey=routing.CALLER_KEY, api_base=IMAGE_BASE)
        with self.assertRaises(web.HTTPBadRequest) as raised:
            asyncio.run(self.method("post")(view, user))
        self.assertNotIn(reflected, raised.exception.text)
        self.assertNotIn("synthetic-secret-key", raised.exception.text)
        self.assertNotIn("Traceback", raised.exception.text)
        self.assertEqual(queued, [])


if __name__ == "__main__":
    unittest.main()
