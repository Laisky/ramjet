"""Offline contracts for dependency upgrades without provider or account access."""

import json
import unittest
from unittest.mock import patch
from uuid import UUID
from urllib.parse import unquote
from urllib.request import parse_http_list, parse_keqv_list

import requests
import tweepy
from kipp.decorator import calculate_args_hash, timeout_cache
from langsmith import Client


class DependencyRuntimeContracts(unittest.TestCase):
    """DependencyRuntimeContracts preserves signing, parsing and cache behavior."""

    def setUp(self):
        """setUp rejects real networking before constructing any SDK client."""
        self.network = patch(
            "socket.socket.connect", side_effect=AssertionError("Network forbidden")
        )
        self.network.start()
        self.addCleanup(self.network.stop)

    def response(self, request, body, status=200):
        """response returns a synthetic requests response for a prepared request."""
        response = requests.Response()
        response.status_code = status
        response._content = body
        response.request = request
        response.url = request.url
        return response

    def authorization(self, request):
        """authorization decodes public OAuth fields from the prepared header."""
        value = request.headers["Authorization"]
        if isinstance(value, bytes):
            value = value.decode("ascii")
        self.assertTrue(value.startswith("OAuth "))
        return {
            key: unquote(value)
            for key, value in parse_keqv_list(parse_http_list(value[6:])).items()
        }

    def test_request_token_preserves_existing_oauth1_signature_and_callback(self):
        """test_request_token_preserves_existing_oauth1_signature_and_callback checks a frozen public vector."""
        auth = tweepy.OAuth1UserHandler(
            "public-consumer",
            "public-consumer-secret",
            callback="https://app.example.invalid/oauth/",
        )
        auth.oauth.auth.client.nonce = "public-fixed-nonce"
        auth.oauth.auth.client.timestamp = "1700000000"
        sent = []

        def send(session, request, **kwargs):
            """send records the fully signed request and returns a public fixture."""
            sent.append(request)
            return self.response(
                request,
                b"oauth_token=public-request&oauth_token_secret=public-request-secret"
                b"&oauth_callback_confirmed=true",
            )

        with patch.object(requests.Session, "send", new=send):
            url = auth.get_authorization_url()
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0].method, "POST")
        self.assertEqual(sent[0].url, "https://api.twitter.com/oauth/request_token")
        fields = self.authorization(sent[0])
        self.assertEqual(fields["oauth_signature"], "ulxfEI3iCk0rXRe5Ktsd7Vm9jsA=")
        self.assertEqual(fields["oauth_callback"], "https://app.example.invalid/oauth/")
        self.assertEqual(
            url, "https://api.twitter.com/oauth/authorize?oauth_token=public-request"
        )
        self.assertEqual(auth.request_token["oauth_callback_confirmed"], "true")

    def test_access_token_exchange_keeps_request_token_and_verifier(self):
        """test_access_token_exchange_keeps_request_token_and_verifier exercises real request preparation."""
        auth = tweepy.OAuth1UserHandler("public-consumer", "public-consumer-secret")
        auth.request_token = {
            "oauth_token": "public-request",
            "oauth_token_secret": "public-request-secret",
        }
        sent = []

        def send(session, request, **kwargs):
            """send captures the OAuth exchange without contacting a provider."""
            sent.append(request)
            return self.response(
                request,
                b"oauth_token=public-access&oauth_token_secret=public-access-secret",
            )

        with patch.object(requests.Session, "send", new=send):
            tokens = auth.get_access_token("public-verifier")
        self.assertEqual(tokens, ("public-access", "public-access-secret"))
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0].url, "https://api.twitter.com/oauth/access_token")
        fields = self.authorization(sent[0])
        self.assertEqual(fields["oauth_token"], "public-request")
        self.assertEqual(fields["oauth_verifier"], "public-verifier")

    def test_authenticated_api_response_and_unauthorized_error_stay_compatible(self):
        """test_authenticated_api_response_and_unauthorized_error_stay_compatible checks real Tweepy parsing."""
        auth = tweepy.OAuth1UserHandler(
            "public-consumer",
            "public-consumer-secret",
            "public-access",
            "public-access-secret",
        )
        api = tweepy.API(auth)
        sent = []

        def send(session, request, **kwargs):
            """send supplies success once and an OAuth error on the next request."""
            sent.append(request)
            if len(sent) == 1:
                return self.response(
                    request,
                    b'{"id":123,"name":"Public Example","screen_name":"public_example"}',
                )
            return self.response(
                request, b'{"errors":[{"message":"Unauthorized","code":32}]}', 401
            )

        with patch.object(requests.Session, "send", new=send):
            user = api.verify_credentials()
            self.assertEqual(user.id, 123)
            self.assertEqual(user.screen_name, "public_example")
            with self.assertRaises(tweepy.Unauthorized):
                api.verify_credentials()
        self.assertEqual(len(sent), 2)
        self.assertEqual(self.authorization(sent[0])["oauth_token"], "public-access")

    def test_langsmith_trace_keeps_public_run_payload_and_http_contract(self):
        """test_langsmith_trace_keeps_public_run_payload_and_http_contract checks real SDK serialization."""
        session = requests.Session()
        client = Client(
            api_url="https://trace.example.invalid",
            api_key="public-fixture-key",
            session=session,
            auto_batch_tracing=False,
        )
        self.addCleanup(session.close)
        sent = []

        def send(session, request, **kwargs):
            """send records the serialized trace without contacting any service."""
            sent.append(request)
            return self.response(request, b'{"success":true}')

        run_id = UUID("00000000-0000-4000-8000-000000000001")
        with patch.object(requests.Session, "send", new=send):
            client.create_run(
                id=run_id,
                name="offline-contract",
                inputs={"prompt": "中文"},
                outputs={"answer": "public example"},
                run_type="chain",
                project_name="public-fixture-project",
            )
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0].method, "POST")
        self.assertEqual(sent[0].url, "https://trace.example.invalid/runs")
        body = json.loads(sent[0].body)
        self.assertEqual(body["id"], str(run_id))
        self.assertEqual(body["run_type"], "chain")
        self.assertEqual(body["inputs"], {"prompt": "中文"})
        self.assertEqual(body["outputs"], {"answer": "public example"})
        self.assertEqual(body["session_name"], "public-fixture-project")

    def test_existing_xxh32_cache_keys_remain_identical(self):
        """test_existing_xxh32_cache_keys_remain_identical preserves frozen xxhash 1.4.4 keys."""
        vectors = [
            ((), {}, "a892f6cd"),
            ((1, 2, 3), {}, "e149b888"),
            (("中文",), {}, "6a6de4fa"),
            ((b"bytes", None, True), {}, "1784ebf9"),
            ((1,), {"name": "alpha"}, "2707ef6a"),
            ((1,), {"name": "beta"}, "22a86dc1"),
            ((), {"a": 1, "b": 2}, "dbeaf971"),
        ]
        for args, kwargs, expected in vectors:
            with self.subTest(args=args, kwargs=kwargs):
                self.assertEqual(calculate_args_hash(*args, **kwargs), expected)

    def test_cache_hits_argument_separation_and_expiration_are_preserved(self):
        """test_cache_hits_argument_separation_and_expiration_are_preserved checks actual cached calls."""
        clock = {"now": 0.0}
        calls = []

        @timeout_cache(expires_sec=30)
        def cached(value, name="alpha"):
            """cached returns an invocation marker for each uncached argument set."""
            calls.append((value, name))
            return len(calls)

        with patch("kipp.decorator.time", side_effect=lambda: clock["now"]):
            self.assertEqual(cached("中文", name="alpha"), 1)
            self.assertEqual(cached("中文", name="alpha"), 1)
            self.assertEqual(cached("中文", name="beta"), 2)
            clock["now"] = 31.0
            self.assertEqual(cached("中文", name="alpha"), 3)
        self.assertEqual(
            calls, [("中文", "alpha"), ("中文", "beta"), ("中文", "alpha")]
        )


if __name__ == "__main__":
    unittest.main()
