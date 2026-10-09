"""Resolve explicit client model credentials without process-global key fallback."""

from typing import Any
from urllib.parse import parse_qsl, urlsplit, urlunsplit

DEFAULT_API_BASE = "https://api.openai.com/v1"


def require_api_key(api_key: str | None) -> str:
    """require_api_key returns an opaque HTTP-safe key or fails before model work."""
    if (
        not isinstance(api_key, str)
        or not api_key
        or api_key.lower()
        in {"bearer", "null", "undefined", "default_proxy_token", "freetier"}
        or any(ord(char) < 33 or ord(char) > 126 for char in api_key)
    ):
        raise ValueError("A valid client API key is required")
    return api_key


def resolve_model_credentials(
    api_key: str | None, api_base: str | None = DEFAULT_API_BASE
) -> dict[str, str]:
    """resolve_model_credentials supplies explicit SDK key and backend options.

    Keys remain transient request data. Provider syntax is checked without
    restricting internal addresses or introducing a destination allowlist.
    """
    key = require_api_key(api_key)
    base = DEFAULT_API_BASE if api_base is None else api_base
    try:
        parsed = urlsplit(base)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or any(ord(char) < 33 for char in base)
        ):
            raise ValueError()
        parsed.port
    except (TypeError, ValueError):
        raise ValueError("A valid model provider URL is required") from None
    return {"api_key": key, "base_url": base.rstrip("/")}


def resolve_request_credentials(
    authorization: str, api_base: str = ""
) -> dict[str, str]:
    """resolve_request_credentials accepts a bearer or legacy raw key and root URL."""
    if not isinstance(authorization, str) or any(
        ord(char) < 32 or ord(char) > 126 for char in authorization
    ):
        raise ValueError("A valid client API key is required")
    parts = authorization.strip().split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        key = parts[1]
    elif len(parts) == 1:
        key = parts[0]
    else:
        raise ValueError("A valid client API key is required")
    base = api_base or DEFAULT_API_BASE
    options = resolve_model_credentials(key, base)
    parsed = urlsplit(options["base_url"])
    path = parsed.path.rstrip("/")
    if not path.endswith("/v1"):
        path += "/v1"
    options["base_url"] = urlunsplit(parsed._replace(path=path))
    return options


def resolve_sdk_credentials(
    api_key: str | None, api_base: str | None = DEFAULT_API_BASE
) -> dict[str, Any]:
    """Resolve model options without appending SDK operation paths to URL queries.

    The public canonical resolver preserves the selected URL for callers. SDK
    options move its query into explicit request parameters; duplicate values are
    preserved as lists, and URL fragments are omitted from the HTTP destination.
    """
    options: dict[str, Any] = resolve_model_credentials(api_key, api_base)
    parsed = urlsplit(options["base_url"])
    if parsed.query:
        query: dict[str, Any] = {}
        for name, value in parse_qsl(parsed.query, keep_blank_values=True):
            if name in query:
                previous = query[name]
                query[name] = (
                    previous + [value]
                    if isinstance(previous, list)
                    else [previous, value]
                )
            else:
                query[name] = value
        options["default_query"] = query
    options["base_url"] = urlunsplit(parsed._replace(query="", fragment=""))
    return options
