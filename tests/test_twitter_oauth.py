"""Local behavioral regressions for Twitter OAuth state, without service imports."""

import ast
import asyncio
import binascii
import importlib.util
import logging
from pathlib import Path
import pickle
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import urllib.parse

ROOT = Path(__file__).resolve().parents[1]
REDUCER_CALLS = []


def harmless_reducer():
    """harmless_reducer records invocation without I/O or service access."""
    REDUCER_CALLS.append("executed")
    return {"oauth_token": "local-token", "oauth_token_secret": "local-secret"}


class LegacyState:
    """LegacyState serializes an inert reducer sentinel for rejection tests."""

    def __reduce__(self):
        """__reduce__ returns only the local non-I/O marker callable."""
        return harmless_reducer, ()


def load_handlers(session):
    """load_handlers executes actual handler classes with inert service doubles."""
    utils_tree = ast.parse((ROOT / "ramjet/utils/__init__.py").read_text())
    env = {
        "pickle": pickle,
        "binascii": binascii,
        "urllib": urllib,
        "web": NS(
            View=object, Response=lambda **kw: NS(**kw), HTTPFound=lambda url: url
        ),
        "logger": logging.getLogger("oauth-local-tests"),
        "utcnow": lambda: "local-time",
        "generate_token": lambda claims: "local-generated-token",
    }
    helpers = [
        n
        for n in utils_tree.body
        if isinstance(n, ast.FunctionDef) and n.name in ("obj2str", "str2obj")
    ]
    exec(
        compile(ast.Module(body=helpers, type_ignores=[]), "<legacy helpers>", "exec"),
        env,
    )
    state_path = ROOT / "ramjet/oauth_state.py"
    if state_path.exists():
        spec = importlib.util.spec_from_file_location(
            "oauth_state_under_test", state_path
        )
        state = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(state)
        env.update(
            make_request_token_state=state.make_request_token_state,
            consume_request_token=state.consume_request_token,
        )

    async def get_session(request):
        """get_session returns only the fixture's in-memory dictionary."""
        return session

    auth = NS(
        request_token={
            "oauth_token": "local-token",
            "oauth_token_secret": "local-secret",
        },
        get_authorization_url=lambda: "https://provider.example.invalid/authorize",
        get_access_token=unittest.mock.Mock(
            return_value=("local-access", "local-access-secret")
        ),
        set_access_token=unittest.mock.Mock(),
    )
    env.update(
        get_session=get_session,
        get_auth=lambda: auth,
        tweepy=NS(API=lambda value: NS(), TweepyException=RuntimeError),
    )
    tree = ast.parse((ROOT / "ramjet/tasks/twitter/login.py").read_text())
    classes = [
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name in ("LoginHandle", "OAuthHandle")
    ]
    for cls in classes:
        for node in cls.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                node.decorator_list = []
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=classes, type_ignores=[])),
            "<actual OAuth handlers>",
            "exec",
        ),
        env,
    )
    return env, auth


class TwitterOAuthTests(unittest.IsolatedAsyncioTestCase):
    """TwitterOAuthTests checks state validation before any provider/account call."""

    def setUp(self):
        """setUp blocks networking and resets the inert reducer sentinel."""
        REDUCER_CALLS.clear()
        self.network = patch(
            "socket.socket.connect", side_effect=AssertionError("Network forbidden")
        )
        self.network.start()
        self.addCleanup(self.network.stop)

    def fixture(
        self, state, query="oauth_token=local-token&oauth_verifier=local-verifier"
    ):
        """fixture builds a request, real callback class, and provider mocks."""
        session = {"request_token": state}
        env, auth = load_handlers(session)
        handler = NS(
            request=NS(query_string=query),
            get_userinfo=unittest.mock.Mock(
                return_value={"id": 123, "username": "local-user"}
            ),
            save_userinfo=unittest.mock.Mock(),
        )
        return session, env, auth, handler

    def valid_state(self):
        """valid_state returns bounded JSON primitives with a current timestamp."""
        return {
            "oauth_token": "local-token",
            "oauth_token_secret": "local-secret",
            "issued_at": int(time.time()),
        }

    async def test_login_stores_json_primitives(self):
        """test_login_stores_json_primitives rejects pickle/base64 session state."""
        session = {}
        env, auth = load_handlers(session)
        await env["LoginHandle"].get(NS(request=NS()))
        state = session["request_token"]
        self.assertIsInstance(state, dict)
        self.assertEqual(state["oauth_token"], auth.request_token["oauth_token"])
        self.assertEqual(
            state["oauth_token_secret"], auth.request_token["oauth_token_secret"]
        )
        self.assertIsInstance(state["issued_at"], int)

    async def test_legacy_reducer_never_executes(self):
        """test_legacy_reducer_never_executes rejects old state before decoding."""
        legacy = binascii.b2a_base64(pickle.dumps(LegacyState())).decode()
        session, env, auth, handler = self.fixture(legacy, query="")
        result = await env["OAuthHandle"].get(handler)
        self.assertEqual(REDUCER_CALLS, [])
        self.assertEqual(result.text, "OAuth Error")
        self.assertNotIn("request_token", session)
        auth.get_access_token.assert_not_called()
        handler.save_userinfo.assert_not_called()

    async def test_valid_state_exchanges_once(self):
        """test_valid_state_exchanges_once preserves login and consumes callback state."""
        session, env, auth, handler = self.fixture(self.valid_state())
        result = await env["OAuthHandle"].get(handler)
        self.assertEqual(result["token"], "local-generated-token")
        auth.get_access_token.assert_called_once_with("local-verifier")
        self.assertEqual(
            auth.request_token,
            {"oauth_token": "local-token", "oauth_token_secret": "local-secret"},
        )
        handler.save_userinfo.assert_called_once()
        self.assertNotIn("request_token", session)
        await env["OAuthHandle"].get(handler)
        auth.get_access_token.assert_called_once()

    async def test_failed_exchange_does_not_write_account(self):
        """test_failed_exchange_does_not_write_account handles a rejected/replayed provider token."""
        session, env, auth, handler = self.fixture(self.valid_state())
        auth.get_access_token.side_effect = RuntimeError("local-rejected-token")
        result = await env["OAuthHandle"].get(handler)
        self.assertEqual(result.text, "OAuth Error")
        self.assertNotIn("request_token", session)
        handler.get_userinfo.assert_not_called()
        handler.save_userinfo.assert_not_called()

    async def test_invalid_state_never_calls_provider(self):
        """test_invalid_state_never_calls_provider covers schema, expiry, and binding."""
        cases = [
            ({}, "oauth_token=local-token&oauth_verifier=v"),
            ("legacy-string", "oauth_token=local-token&oauth_verifier=v"),
            (
                dict(self.valid_state(), issued_at=0),
                "oauth_token=local-token&oauth_verifier=v",
            ),
            (
                dict(self.valid_state(), issued_at=int(time.time()) + 3600),
                "oauth_token=local-token&oauth_verifier=v",
            ),
            (
                dict(self.valid_state(), oauth_token_secret=[]),
                "oauth_token=local-token&oauth_verifier=v",
            ),
            (self.valid_state(), "oauth_token=other&oauth_verifier=v"),
            (self.valid_state(), "oauth_token=local-token"),
            (
                self.valid_state(),
                "oauth_token=local-token&oauth_verifier=v&oauth_verifier=w",
            ),
            (
                dict(self.valid_state(), extra="unexpected"),
                "oauth_token=local-token&oauth_verifier=v",
            ),
        ]
        for state, query in cases:
            with self.subTest(state_type=type(state).__name__, query=query):
                session, env, auth, handler = self.fixture(state, query)
                result = await env["OAuthHandle"].get(handler)
                self.assertEqual(result.text, "OAuth Error")
                self.assertNotIn("request_token", session)
                auth.get_access_token.assert_not_called()
                handler.save_userinfo.assert_not_called()


class CookieKeyTests(unittest.TestCase):
    """CookieKeyTests checks startup rejection and purpose-separated key derivation."""

    def setUp(self):
        """setUp loads only the pure session helper module."""
        spec = importlib.util.spec_from_file_location(
            "cookie_keys_under_test", ROOT / "ramjet/oauth_state.py"
        )
        self.state = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.state)

    def test_unsafe_secrets_rejected(self):
        """test_unsafe_secrets_rejected rejects absent, public, and short secrets."""
        public = "public-default-for-local-test-" * 3
        for secret in (None, "", "short", public):
            with self.subTest(secret_type=type(secret).__name__):
                with self.assertRaises(ValueError):
                    self.state.create_cookie_key(secret, public)

    def test_setup_checks_key_before_cookie_storage(self):
        """test_setup_checks_key_before_cookie_storage verifies the actual startup boundary."""
        function = next(
            n
            for n in ast.parse((ROOT / "ramjet/app.py").read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "setup_web_handlers"
        )
        public = "public-default-for-startup-test-" * 3
        storage = unittest.mock.Mock()
        setup = unittest.mock.Mock()
        env = {
            "settings": NS(),
            "SECRET_KEY": public,
            "DEFAULT_SECRET_KEY": public,
            "create_cookie_key": self.state.create_cookie_key,
            "EncryptedCookieStorage": storage,
            "setup": setup,
            "web": NS(view=unittest.mock.Mock()),
            "PageNotFound": object,
        }
        exec(
            compile(
                ast.Module(body=[function], type_ignores=[]), "<actual setup>", "exec"
            ),
            env,
        )
        with self.assertRaises(ValueError):
            env["setup_web_handlers"](object())
        storage.assert_not_called()
        setup.assert_not_called()
        env["settings"].SESSION_SECRET_KEY = "local-configured-session-secret-" * 3
        env["setup_web_handlers"](object())
        storage.assert_called_once()
        self.assertEqual(
            storage.call_args.kwargs, {"httponly": True, "samesite": "Lax"}
        )
        setup.assert_called_once()

    def test_cookie_key_is_domain_separated(self):
        """test_cookie_key_is_domain_separated produces a stable distinct Fernet key."""
        import hashlib

        secret = "local-ephemeral-secret-for-cookie-tests-" * 3
        key = self.state.create_cookie_key(secret, "public-default")
        self.assertEqual(len(key), 32)
        self.assertEqual(key, self.state.create_cookie_key(secret, "public-default"))
        self.assertNotEqual(key, hashlib.sha256(secret.encode()).digest())


class CookieStateRoundTripTests(unittest.IsolatedAsyncioTestCase):
    """CookieStateRoundTripTests uses real encrypted local cookies and no external I/O."""

    async def test_encrypted_json_round_trip_and_consumption(self):
        """test_encrypted_json_round_trip_and_consumption exercises the storage boundary."""
        from aiohttp import web, CookieJar
        from aiohttp.test_utils import TestClient, TestServer
        from aiohttp_session import get_session, setup
        from aiohttp_session.cookie_storage import EncryptedCookieStorage

        spec = importlib.util.spec_from_file_location(
            "cookie_state_under_test", ROOT / "ramjet/oauth_state.py"
        )
        state = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(state)
        app = web.Application()
        key = state.create_cookie_key(
            "local-ephemeral-cookie-secret-" * 3, "public-default"
        )
        setup(app, EncryptedCookieStorage(key, httponly=True, samesite="Lax"))

        async def begin(request):
            """begin records only synthetic request-token primitives in the real session."""
            session = await get_session(request)
            session["request_token"] = state.make_request_token_state(
                {"oauth_token": "local-token", "oauth_token_secret": "local-secret"}
            )
            return web.Response(text="started")

        async def callback(request):
            """callback validates the real decoded session and clears its request token."""
            session = await get_session(request)
            try:
                token, verifier = state.consume_request_token(
                    session, request.query_string
                )
            except ValueError:
                return web.Response(status=400)
            self.assertEqual(token["oauth_token"], "local-token")
            self.assertEqual(verifier, "local-verifier")
            return web.Response(text="accepted")

        app.router.add_get("/begin", begin)
        app.router.add_get("/callback", callback)
        client = TestClient(
            TestServer(app, host="127.0.0.1"), cookie_jar=CookieJar(unsafe=True)
        )
        await client.start_server()
        try:
            started = await client.get("/begin")
            self.assertEqual(started.status, 200)
            cookie = started.cookies["AIOHTTP_SESSION"]
            self.assertTrue(cookie["httponly"])
            self.assertEqual(cookie["samesite"], "Lax")
            self.assertNotIn("local-token", cookie.value)
            first = await client.get(
                "/callback?oauth_token=local-token&oauth_verifier=local-verifier"
            )
            self.assertEqual(first.status, 200)
            second = await client.get(
                "/callback?oauth_token=local-token&oauth_verifier=local-verifier"
            )
            self.assertEqual(second.status, 400)
        finally:
            await client.close()


if __name__ == "__main__":
    unittest.main()
