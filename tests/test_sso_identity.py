"""Bounded offline tests of strict independent issuer and model-token separation."""

import json
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import time

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import jwt
from multidict import CIMultiDict

from ramjet.credentials import require_api_key, resolve_request_credentials
from ramjet.sso import (
    IssuerUnavailable,
    MAX_RESPONSE_BYTES,
    SessionInvalid,
    SSO_ENDPOINT,
    VerifiedPrincipal,
    WhoAmIVerifier,
    _request_whoami,
    claimed_principal,
    require_session_token,
)

SUBJECT = "0194d5f8-19f7-7f7b-a8d3-421a60f8d8ab"
OTHER = "0194d5f8-19f7-7f7b-a8d3-421a60f8d8ac"
# This deterministic public test key is trusted only by the in-process mock.
OFFLINE_KEY = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))


def session_token(**changes):
    """session_token creates only synthetic mock-issuer tokens, never real credentials."""
    now = int(time.time())
    claims = dict(
        iss="laisky-sso",
        sub=SUBJECT,
        uid=SUBJECT,
        iat=now - 1,
        exp=now + 600,
        jti="offline-test-session",
        username="offline@example.invalid",
        display_name="Offline fixture",
    )
    claims.update(changes)
    return jwt.encode(claims, OFFLINE_KEY, algorithm="EdDSA")


class FakeIssuer:
    """FakeIssuer verifies real EdDSA math for an isolated public test issuer."""

    def __init__(self):
        """__init__ keeps calls and revocation state only in test memory."""
        self.calls = []
        self.revoked = False

    async def request(self, token):
        """request rejects forged/revoked fixtures and returns only a stable mock account."""
        self.calls.append(token)
        try:
            claims = jwt.decode(
                token,
                OFFLINE_KEY.public_key(),
                algorithms=["EdDSA"],
                issuer="laisky-sso",
            )
        except jwt.InvalidTokenError:
            return 401, b'{"errors":[{"message":"invalid fixture"}]}'
        if self.revoked:
            return 403, b'{"errors":[{"message":"revoked fixture"}]}'
        return 200, json.dumps({"data": {"WhoAmI": {"id": claims["sub"]}}}).encode()


class StrictClaimTests(unittest.TestCase):
    """StrictClaimTests reject legacy/ambiguous claims before any issuer transmission."""

    def test_documented_shape_and_bearer_header(self):
        """test_documented_shape requires a dedicated header and returns canonical stable identity."""
        token = session_token()
        self.assertEqual(claimed_principal(token).subject, SUBJECT)
        self.assertEqual(
            require_session_token(
                CIMultiDict({"X-Laisky-SSO-Token": "Bearer " + token})
            ),
            token,
        )
        for headers in (
            {"Authorization": "Bearer " + token},
            {"X-Laisky-User-Id": SUBJECT},
            {"X-Laisky-SSO-Token": token},
            CIMultiDict(
                [
                    ("X-Laisky-SSO-Token", "Bearer " + token),
                    ("X-Laisky-SSO-Token", "Bearer " + token),
                ]
            ),
        ):
            with self.subTest(headers_kind=type(headers).__name__):
                with self.assertRaises(SessionInvalid):
                    require_session_token(headers)

    def test_expiry_issuer_identity_and_required_shape_fail_closed(self):
        """test_expiry_issuer_identity rejects unsafe claims without interpreting them as authority."""
        invalid = [
            {"iss": "other"},
            {"exp": 0},
            {"exp": True},
            {"exp": "future"},
            {"iat": int(time.time()) + 120},
            {"uid": OTHER},
            {"sub": "victim"},
            {"uid": "00000000-0000-0000-0000-000000000000"},
            {"jti": ""},
            {"username": None},
            {"display_name": None},
        ]
        for change in invalid:
            with self.subTest(fields=list(change)):
                with self.assertRaises(SessionInvalid):
                    claimed_principal(session_token(**change))
        for token in ("", "not-a-jwt", "x" * 8193, "a.b.c"):
            with self.assertRaises(SessionInvalid):
                claimed_principal(token)

    def test_legacy_algorithm_has_no_fallback(self):
        """test_legacy_algorithm refuses a token even when its subject and issuer look current."""
        claims = jwt.decode(session_token(), options={"verify_signature": False})
        legacy = jwt.encode(
            claims, "public-offline-legacy-test-key-at-least-32", algorithm="HS256"
        )
        with self.assertRaises(SessionInvalid):
            claimed_principal(legacy)

    def test_session_tokens_never_become_model_keys(self):
        """test_session_tokens rejects current/legacy SSO and preserves unrelated provider JWTs."""
        claims = jwt.decode(session_token(), options={"verify_signature": False})
        for token in (
            session_token(),
            jwt.encode(
                claims,
                "public-offline-fixture-legacy-key-at-least-32",
                algorithm="HS256",
            ),
        ):
            for value in (token, "Bearer " + token):
                with self.assertRaises(ValueError):
                    resolve_request_credentials(value)
            with self.assertRaises(ValueError):
                require_api_key(token)
        token = session_token()
        from urllib.parse import quote

        for base in (
            "https://provider.fixture/v1?access_token=" + token,
            "https://provider.fixture/v1?" + token + "=routing-value",
            "https://provider.fixture/v1/" + token,
            "https://" + token + "@provider.fixture/v1",
            "https://provider.fixture/v1#" + token,
            "http://provider.fixture/v1?auth=" + quote("Bearer " + token),
        ):
            with self.assertRaises(ValueError):
                resolve_request_credentials("model-only-key", base)
        claims["iss"] = "fixture-model-provider"
        provider_key = jwt.encode(
            claims, "public-offline-fixture-legacy-key-at-least-32", algorithm="HS256"
        )
        self.assertEqual(require_api_key(provider_key), provider_key)
        unrelated_base = "http://internal.fixture/v1?" + provider_key + "=routing"
        self.assertEqual(
            resolve_request_credentials("model-only-key", unrelated_base)["base_url"],
            unrelated_base,
        )
        self.assertEqual(
            resolve_request_credentials(
                "weak-but-model-only", "http://internal.fixture"
            )["api_key"],
            "weak-but-model-only",
        )


class OnlineIdentityTests(unittest.IsolatedAsyncioTestCase):
    """OnlineIdentityTests require online binding and never cache revocation decisions."""

    async def test_active_identity_then_revocation_is_rechecked(self):
        """test_active_identity checks every request rather than trusting parsed claims or prior success."""
        issuer = FakeIssuer()
        verifier = WhoAmIVerifier(issuer.request)
        token = session_token()
        self.assertEqual((await verifier.verify(token)).subject, SUBJECT)
        issuer.revoked = True
        with self.assertRaises(SessionInvalid):
            await verifier.verify(token)
        self.assertEqual(len(issuer.calls), 2)

    async def test_forged_signature_and_legacy_are_denied(self):
        """test_forged_signature relies on issuer math while legacy tokens never reach that issuer."""
        issuer = FakeIssuer()
        verifier = WhoAmIVerifier(issuer.request)
        token = session_token()
        claims = jwt.decode(token, options={"verify_signature": False})
        forged_key = Ed25519PrivateKey.from_private_bytes(b"z" * 32)
        forged = jwt.encode(claims, forged_key, algorithm="EdDSA")
        with self.assertRaises(SessionInvalid):
            await verifier.verify(forged)
        legacy = jwt.encode(
            claims, "public-offline-fixture-legacy-key-at-least-32", algorithm="HS256"
        )
        with self.assertRaises(SessionInvalid):
            await verifier.verify(legacy)
        self.assertEqual(len(issuer.calls), 1)

    async def test_unavailable_redirect_malformed_and_oversized_responses(self):
        """test_unavailable rejects responses that cannot supply a trustworthy account answer."""
        cases = [
            (302, b""),
            (500, b"reflected secret"),
            (200, b"not-json"),
            (200, b"[]"),
            (200, b"{}"),
            (200, json.dumps({"data": {"WhoAmI": {"id": None}}}).encode()),
            (200, json.dumps({"data": {"WhoAmI": {"id": "not-a-uuid"}}}).encode()),
            (200, b"[" * 2000 + b"]" * 2000),
            (200, b"x" * (MAX_RESPONSE_BYTES + 1)),
        ]
        for status, body in cases:

            async def respond(token):
                """respond returns only a bounded synthetic envelope or protocol failure."""
                return status, body

            with self.subTest(status=status, size=len(body)):
                with self.assertRaises(IssuerUnavailable) as raised:
                    await WhoAmIVerifier(respond).verify(session_token())
                self.assertEqual(str(raised.exception), "SSO service is unavailable")

    async def test_valid_identity_mismatch_and_graphql_errors_deny_session(self):
        """test_valid_identity_mismatch excludes claim-only and response-only authorization."""
        for body in (
            {"data": {"WhoAmI": {"id": OTHER}}},
            {
                "data": {"WhoAmI": {"id": SUBJECT}},
                "errors": [{"message": "secret-reflection"}],
            },
        ):

            async def respond(token):
                """respond supplies the selected mismatched or issuer-rejected account."""
                return 200, json.dumps(body).encode()

            with self.assertRaises(SessionInvalid) as raised:
                await WhoAmIVerifier(respond).verify(session_token())
            self.assertNotIn("secret-reflection", str(raised.exception))

    async def test_expiry_during_round_trip_is_rejected(self):
        """test_expiry_during_round_trip refuses a session that expired while the issuer replied."""
        token = session_token()
        expiry = jwt.decode(token, options={"verify_signature": False})["exp"]

        async def respond(value):
            """respond models an answer arriving after the original session expiry."""
            return 200, json.dumps({"data": {"WhoAmI": {"id": SUBJECT}}}).encode()

        with patch("ramjet.sso.time.time", side_effect=[expiry - 10, expiry + 1]):
            with self.assertRaises(SessionInvalid):
                await WhoAmIVerifier(respond).verify(token)

    async def test_fixed_transport_disables_redirect_and_environment_proxy(self):
        """test_fixed_transport proves the bearer goes only to the pinned issuer endpoint."""
        calls = []
        chunks = [b'{"data":{"WhoAmI":{"id":"', SUBJECT.encode(), b'"}}}', b""]

        class Response:
            """Response supplies a local stream without opening a socket."""

            status = 200

            async def __aenter__(self):
                """__aenter__ returns the synthetic response."""
                return self

            async def __aexit__(self, *args):
                """__aexit__ closes no real resource."""
                return False

            async def read(self, size):
                """read yields only synthetic bounded chunks."""
                return chunks.pop(0)

            @property
            def content(self):
                """content exposes the synthetic stream reader."""
                return self

        class Client:
            """Client records fixed transport options and makes no network request."""

            def __init__(self, **kwargs):
                """__init__ retains explicit timeout and proxy policy."""
                calls.append(("client", kwargs))

            async def __aenter__(self):
                """__aenter__ returns this in-memory client."""
                return self

            async def __aexit__(self, *args):
                """__aexit__ performs no external action."""
                return False

            def post(self, endpoint, **kwargs):
                """post records one issuer-only request."""
                calls.append((endpoint, kwargs))
                return Response()

        token = session_token()
        with patch("ramjet.sso.aiohttp.ClientSession", Client):
            status, body = await _request_whoami(token)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["data"]["WhoAmI"]["id"], SUBJECT)
        self.assertFalse(calls[0][1]["trust_env"])
        self.assertEqual(calls[0][1]["timeout"].total, 5)
        self.assertEqual(calls[1][0], SSO_ENDPOINT)
        self.assertFalse(calls[1][1]["allow_redirects"])
        self.assertEqual(calls[1][1]["headers"]["Authorization"], "Bearer " + token)


if __name__ == "__main__":
    unittest.main()
