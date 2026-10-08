"""Non-executable, bounded state for the local Twitter OAuth cookie flow."""

import hashlib
import hmac
import time
import urllib.parse

TOKEN_LIFETIME_SECONDS = 600
TOKEN_FIELD_LIMIT = 1024


def validate_token_fields(token):
    """validate_token_fields checks two bounded provider token strings and returns them."""
    if not isinstance(token, dict) or set(token) != {
        "oauth_token",
        "oauth_token_secret",
    }:
        raise ValueError("Invalid OAuth request token")
    if any(
        type(value) is not str or not 0 < len(value) <= TOKEN_FIELD_LIMIT
        for value in token.values()
    ):
        raise ValueError("Invalid OAuth request token")
    return dict(token)


def make_request_token_state(token):
    """make_request_token_state returns validated JSON token fields with issuance time."""
    return dict(validate_token_fields(token), issued_at=int(time.time()))


def consume_request_token(session, query_string):
    """consume_request_token removes state and validates its schema, age, and callback binding."""
    state = session.pop("request_token", None)
    if not isinstance(state, dict) or set(state) != {
        "oauth_token",
        "oauth_token_secret",
        "issued_at",
    }:
        raise ValueError("Invalid OAuth state")
    issued_at = state["issued_at"]
    if (
        type(issued_at) is not int
        or not 0 <= time.time() - issued_at <= TOKEN_LIFETIME_SECONDS
    ):
        raise ValueError("Expired OAuth state")
    token = validate_token_fields(
        {name: state[name] for name in ("oauth_token", "oauth_token_secret")}
    )
    if not isinstance(query_string, str) or len(query_string) > 4096:
        raise ValueError("Invalid OAuth callback")
    query = urllib.parse.parse_qs(query_string, max_num_fields=8)
    if any(len(query.get(name, [])) != 1 for name in ("oauth_token", "oauth_verifier")):
        raise ValueError("Invalid OAuth callback")
    callback_token, verifier = query["oauth_token"][0], query["oauth_verifier"][0]
    if (
        not 0 < len(callback_token) <= TOKEN_FIELD_LIMIT
        or not 0 < len(verifier) <= TOKEN_FIELD_LIMIT
        or not hmac.compare_digest(
            callback_token.encode("utf-8"), token["oauth_token"].encode("utf-8")
        )
    ):
        raise ValueError("Invalid OAuth callback")
    return token, verifier


def create_cookie_key(secret, public_default):
    """create_cookie_key rejects missing/public/short keys and derives a cookie-purpose key."""
    if not isinstance(secret, str) or len(secret.encode("utf-8")) < 32:
        raise ValueError(
            "Configure a deployment-specific session secret of at least 32 bytes"
        )
    if hmac.compare_digest(secret.encode("utf-8"), public_default.encode("utf-8")):
        raise ValueError("The public default secret cannot protect session cookies")
    return hmac.digest(
        secret.encode("utf-8"), b"ramjet/session-cookie/v2", hashlib.sha256
    )
