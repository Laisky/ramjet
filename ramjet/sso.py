"""Independent online Laisky SSO verification without issuer signing material."""

import base64
from dataclasses import dataclass
import hmac
import json
import math
import re
import time
from uuid import UUID

import aiohttp

SSO_ENDPOINT = "https://sso.laisky.com/query"
SSO_ISSUER = "laisky-sso"
SESSION_HEADER = "X-Laisky-SSO-Token"
MAX_TOKEN_BYTES = 8192
MAX_RESPONSE_BYTES = 64 << 10
WHOAMI_QUERY = "query RamjetPrivateSession { WhoAmI { id } }"
_COMPACT_PART = re.compile(r"[A-Za-z0-9_-]+")


class SessionInvalid(ValueError):
    """SessionInvalid represents missing, malformed or issuer-rejected sessions."""


class IssuerUnavailable(RuntimeError):
    """IssuerUnavailable represents an untrustworthy or unavailable issuer response."""


@dataclass(frozen=True)
class VerifiedPrincipal:
    """VerifiedPrincipal carries only the online-verified stable identity and expiry."""

    subject: str
    expires_at: float
    issuer: str = SSO_ISSUER


def canonical_subject(value: str) -> str:
    """canonical_subject accepts only nonzero UUID identities in canonical text form."""
    if not isinstance(value, str):
        raise SessionInvalid("Invalid SSO session")
    try:
        parsed = UUID(value)
    except (ValueError, AttributeError):
        raise SessionInvalid("Invalid SSO session") from None
    if parsed.int == 0 or value.lower() != str(parsed):
        raise SessionInvalid("Invalid SSO session")
    return str(parsed)


def _json_object(pairs):
    """_json_object rejects duplicate authentication fields and ambiguous envelopes."""
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("duplicate field")
        result[name] = value
    return result


def _decode_part(part: str) -> bytes:
    """_decode_part decodes one bounded unpadded compact JWT component."""
    if not _COMPACT_PART.fullmatch(part):
        raise ValueError("invalid compact component")
    return base64.b64decode(
        part + "=" * (-len(part) % 4), altchars=b"-_", validate=True
    )


def _untrusted_parts(token: str):
    """_untrusted_parts decodes shape only; it never authenticates token claims."""
    if not isinstance(token, str) or not 0 < len(token) <= MAX_TOKEN_BYTES:
        raise ValueError("invalid token size")
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("invalid token shape")
    header = json.loads(_decode_part(parts[0]), object_pairs_hook=_json_object)
    claims = json.loads(_decode_part(parts[1]), object_pairs_hook=_json_object)
    signature = _decode_part(parts[2])
    if not isinstance(header, dict) or not isinstance(claims, dict):
        raise ValueError("invalid token objects")
    return header, claims, signature


def looks_like_sso_token(value: str) -> bool:
    """looks_like_sso_token prevents identifiable SSO sessions becoming provider keys.

    This is a rejection heuristic, not identity verification. Even legacy tokens
    naming the Laisky issuer are refused as model credentials.
    """
    try:
        _, claims, _ = _untrusted_parts(value)
        return claims.get("iss") == SSO_ISSUER
    except (ValueError, TypeError, UnicodeError, RecursionError):
        return False


def require_session_token(headers) -> str:
    """require_session_token extracts one explicit bearer session without UID/query fallback."""
    values = (
        headers.getall(SESSION_HEADER, [])
        if hasattr(headers, "getall")
        else [headers.get(SESSION_HEADER, "")]
    )
    if len(values) != 1 or not isinstance(values[0], str):
        raise SessionInvalid("An SSO session is required")
    parts = values[0].split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise SessionInvalid("An SSO session is required")
    token = parts[1]
    if len(token) > MAX_TOKEN_BYTES:
        raise SessionInvalid("Invalid SSO session")
    return token


def claimed_principal(token: str, *, now: float | None = None) -> VerifiedPrincipal:
    """claimed_principal rejects legacy/expired claims before online authentication.

    The result remains untrusted until WhoAmI verifies the original token and
    resolves the identical stable account. No local signature or algorithm
    fallback is attempted.
    """
    current = time.time() if now is None else now
    try:
        header, claims, signature = _untrusted_parts(token)
        if (
            header.get("alg") != "EdDSA"
            or header.get("typ", "JWT") != "JWT"
            or "crit" in header
            or header.get("b64", True) is not True
            or len(signature) != 64
            or claims.get("iss") != SSO_ISSUER
        ):
            raise ValueError()
        subject = canonical_subject(claims.get("sub"))
        uid = canonical_subject(claims.get("uid"))
        if not hmac.compare_digest(subject, uid):
            raise ValueError()
        expiry = claims.get("exp")
        issued = claims.get("iat")
        for numeric in (expiry, issued):
            if (
                isinstance(numeric, bool)
                or not isinstance(numeric, (int, float))
                or not math.isfinite(numeric)
            ):
                raise ValueError()
        if expiry <= current or issued > current + 60 or issued >= expiry:
            raise ValueError()
        for field in ("jti", "username", "display_name"):
            if not isinstance(claims.get(field), str):
                raise ValueError()
        if not claims["jti"] or not claims["username"]:
            raise ValueError()
        return VerifiedPrincipal(subject, expiry)
    except (ValueError, TypeError, UnicodeError, OverflowError, RecursionError):
        raise SessionInvalid("Invalid or expired SSO session") from None


async def _request_whoami(token: str) -> tuple[int, bytes]:
    """_request_whoami sends the original token only to fixed HTTPS without redirects/proxies."""
    timeout = aiohttp.ClientTimeout(total=5)
    try:
        async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as client:
            async with client.post(
                SSO_ENDPOINT,
                headers={
                    "Authorization": "Bearer " + token,
                    "Accept": "application/json",
                },
                json={"query": WHOAMI_QUERY},
                allow_redirects=False,
            ) as response:
                body = bytearray()
                while True:
                    chunk = await response.content.read(
                        min(8192, MAX_RESPONSE_BYTES + 1 - len(body))
                    )
                    if not chunk:
                        break
                    body.extend(chunk)
                    if len(body) > MAX_RESPONSE_BYTES:
                        raise IssuerUnavailable("SSO service is unavailable")
                return response.status, bytes(body)
    except Exception:
        raise IssuerUnavailable("SSO service is unavailable") from None


class WhoAmIVerifier:
    """WhoAmIVerifier independently verifies each request; positive sessions are not cached."""

    def __init__(self, request=None):
        """__init__ selects fixed production transport or an explicit in-process test mock."""
        self._request = _request_whoami if request is None else request

    async def verify(self, token: str) -> VerifiedPrincipal:
        """verify binds strict current claims to an active online account before returning identity."""
        claimed = claimed_principal(token)
        try:
            status, raw = await self._request(token)
        except Exception:
            raise IssuerUnavailable("SSO service is unavailable") from None
        if status in (401, 403):
            raise SessionInvalid("Invalid or expired SSO session")
        if status != 200 or not isinstance(raw, bytes) or len(raw) > MAX_RESPONSE_BYTES:
            raise IssuerUnavailable("SSO service is unavailable")
        try:
            payload = json.loads(raw, object_pairs_hook=_json_object)
            if not isinstance(payload, dict):
                raise ValueError()
            if payload.get("errors"):
                raise SessionInvalid("Invalid or expired SSO session")
            identity = payload["data"]["WhoAmI"]
            try:
                subject = canonical_subject(identity["id"])
            except SessionInvalid:
                raise IssuerUnavailable("SSO service is unavailable") from None
        except SessionInvalid:
            raise
        except (ValueError, KeyError, TypeError, UnicodeError, RecursionError):
            raise IssuerUnavailable("SSO service is unavailable") from None
        if not hmac.compare_digest(subject, claimed.subject):
            raise SessionInvalid("Invalid SSO session")
        # Expiry may occur during the bounded issuer round trip.
        if claimed.expires_at <= time.time():
            raise SessionInvalid("Invalid or expired SSO session")
        return VerifiedPrincipal(subject, claimed.expires_at)
