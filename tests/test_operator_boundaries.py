"""Offline HTTP regressions for operator-only SMTP and stored-account jobs."""

import ast
from functools import partial
import logging
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

ROOT = Path(__file__).resolve().parents[1]


def load_post(path, class_name, env):
    """load_post returns the actual handler method with inert import dependencies."""
    tree = ast.parse((ROOT / path).read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    post = next(
        node
        for node in cls.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "post"
    )
    exec(compile(ast.unparse(post), "<actual operator handler>", "exec"), env)
    return type(class_name, (web.View,), {"post": env["post"]})


class OperatorBoundaryTests(unittest.IsolatedAsyncioTestCase):
    """OperatorBoundaryTests exercises HTTP admission without external side effects."""

    async def asyncSetUp(self):
        """asyncSetUp constructs actual HTTP routes with inert SMTP and crawler effects."""
        self.effects = []
        effects = self.effects

        class Sender:
            """Sender records construction and dispatch without opening SMTP."""

            def __init__(self, **kwargs):
                """__init__ records the configured caller SMTP options."""
                effects.append(("smtp_construct", kwargs))

            def send_email(self, **kwargs):
                """send_email records the accepted local message arguments."""
                effects.append(("smtp_send", kwargs))
                return "local accepted"

        class Loop:
            """Loop dispatches the inert sender inline without background I/O."""

            async def run_in_executor(self, executor, fn):
                """run_in_executor returns the inert callable's local result."""
                return fn()

        class Twitter:
            """Twitter records stored-account construction without loading any account."""

            def __init__(self):
                """__init__ records crawler construction only."""
                effects.append(("twitter_construct", None))

            def run_for_tweet_id(self, tweet_id):
                """run_for_tweet_id records the normalized local tweet ID."""
                effects.append(("twitter_run", tweet_id))

        class Executor:
            """Executor records submission without executing stored-account work."""

            def submit(self, fn, value):
                """submit records the queued callable and argument without running it."""
                effects.append(("twitter_submit", value))

        try:
            from ramjet.operator_auth import require_operator
        except ModuleNotFoundError:

            def require_operator(request):
                """require_operator is unused by the unchanged baseline handler."""
                raise AssertionError("baseline does not call authorization")

        env = dict(
            web=web,
            partial=partial,
            EmailSender=Sender,
            ioloop=Loop(),
            thread_executor=Executor(),
            TwitterAPI=Twitter,
            logger=logging.getLogger("local-operator"),
            require_operator=require_operator,
        )
        email = load_post("ramjet/tasks/email_proxy.py", "EmailProxyHandle", env.copy())
        twitter = load_post("ramjet/tasks/twitter/crawler.py", "FetchView", env.copy())
        app = web.Application()
        app.router.add_view("/smtp", email)
        app.router.add_view("/tweet", twitter)
        self.client = TestClient(TestServer(app))
        await self.client.start_server()
        from ramjet import settings

        self.settings = settings
        self.token_patch = patch.object(
            settings, "OPERATOR_API_TOKEN", "public-local-operator-token", create=True
        )
        self.token_patch.start()

    async def asyncTearDown(self):
        """asyncTearDown releases only the disposable server and synthetic setting."""
        self.token_patch.stop()
        await self.client.close()

    def smtp_payload(self):
        """smtp_payload returns harmless synthetic SMTP arguments for local mocks."""
        return dict(
            host="configured-private.fixture",
            username="fixture-user",
            passwd="fixture-password",
            use_tls=False,
            to_addrs=["nobody.invalid"],
            subject="local fixture",
            content="inert",
        )

    async def test_anonymous_calls_never_construct_or_submit(self):
        """test_anonymous_calls requires authentication before SMTP or stored accounts."""
        for route, kwargs in [
            ("/smtp", {"json": self.smtp_payload()}),
            ("/tweet", {"data": {"tweet_id": "12345"}}),
        ]:
            with self.subTest(route=route):
                response = await self.client.post(route, **kwargs)
                self.assertEqual(response.status, 401)
                await response.release()
        self.assertEqual(self.effects, [])

    async def test_wrong_credentials_are_rejected_before_body_parsing(self):
        """test_wrong_credentials rejects malformed bodies before parsing or effects."""
        for header in [
            "Bearer wrong",
            "Basic public-local-operator-token",
            "Bearer ",
            "Bearer public-local-operator-token extra",
        ]:
            for route in ["/smtp", "/tweet"]:
                response = await self.client.post(
                    route, data="{malformed", headers={"Authorization": header}
                )
                self.assertEqual(response.status, 401)
                await response.release()
        self.assertEqual(self.effects, [])

    async def test_missing_configuration_fails_closed(self):
        """test_missing_configuration returns unavailable without privileged dispatch."""
        for value in ["", None]:
            with patch.object(self.settings, "OPERATOR_API_TOKEN", value):
                for route, kwargs in [
                    ("/smtp", {"json": self.smtp_payload()}),
                    ("/tweet", {"data": {"tweet_id": "12345"}}),
                ]:
                    response = await self.client.post(
                        route,
                        headers={"Authorization": "Bearer public-local-operator-token"},
                        **kwargs,
                    )
                    self.assertEqual(response.status, 503)
                    await response.release()
        self.assertEqual(self.effects, [])

    async def test_operator_preserves_optional_tls_and_custom_smtp(self):
        """test_operator preserves explicitly configured caller SMTP and optional TLS."""
        for tls in [False, True, None]:
            payload = self.smtp_payload()
            if tls is None:
                payload.pop("use_tls")
            else:
                payload["use_tls"] = tls
            response = await self.client.post(
                "/smtp",
                json=payload,
                headers={"Authorization": "Bearer public-local-operator-token"},
            )
            self.assertEqual(response.status, 200)
            self.assertEqual(await response.text(), "local accepted")
            self.assertEqual(self.effects[-2][1]["host"], "configured-private.fixture")
            self.assertIs(self.effects[-2][1]["use_tls"], tls)

    async def test_operator_preserves_tweet_normalization(self):
        """test_operator preserves existing URL normalization and one inert job submission."""
        response = await self.client.post(
            "/tweet",
            data={"tweet_id": "https://x.com/local/status/12345?fixture=1"},
            headers={"Authorization": "Bearer public-local-operator-token"},
        )
        self.assertEqual(response.status, 200)
        self.assertEqual(
            self.effects, [("twitter_construct", None), ("twitter_submit", "12345")]
        )

    async def test_duplicate_authorization_headers_fail_closed(self):
        """test_duplicate_authorization_headers rejects ambiguous operator credentials."""
        response = await self.client.post(
            "/tweet",
            data={"tweet_id": "12345"},
            headers=[
                ("Authorization", "Bearer public-local-operator-token"),
                ("Authorization", "Bearer wrong"),
            ],
        )
        self.assertEqual(response.status, 401)
        self.assertEqual(self.effects, [])


if __name__ == "__main__":
    unittest.main()
