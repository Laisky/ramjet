"""Resolve explicit client model credentials without process-global key fallback."""

import logging
from typing import Any
from urllib.parse import urlsplit, urlunsplit

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


class ModelDiagnosticFilter(logging.Filter):
    """Keep model-library diagnostics free of URLs, payloads and exception bodies."""

    _ramjet_model_diagnostic_filter = True

    def filter(self, record: logging.LogRecord) -> bool:
        """Retain diagnostic severity and HTTP status without credential-bearing data."""
        if record.levelno < logging.INFO:
            return False
        args = record.args
        if record.name == "httpx" and isinstance(args, tuple) and len(args) >= 4:
            method = (
                args[0]
                if isinstance(args[0], str)
                and args[0]
                in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}
                else "request"
            )
            status = args[3] if isinstance(args[3], int) else "unknown"
            record.msg = "HTTP client request: %s (status %s)"
            record.args = (method, status)
        else:
            record.msg = "Model client diagnostic"
            record.args = ()
        record.exc_info = None
        record.exc_text = None
        record.stack_info = None
        return True


def resolve_sdk_credentials(
    api_key: str | None,
    api_base: str | None = DEFAULT_API_BASE,
    *,
    include_async_client: bool = True,
) -> dict[str, Any]:
    """Bind explicit SDK credentials while preserving provider URL query bytes.

    SDK clients otherwise append operation paths after URL queries or rewrite
    repeated/blank parameters. Public HTTPX hooks restore the selected query
    after SDK path construction for both synchronous and asynchronous requests.
    Imports remain local so the canonical resolver needs only the standard library.
    Synchronous callers can omit creation of an unused asynchronous transport.
    """
    options: dict[str, Any] = resolve_model_credentials(api_key, api_base)
    for name in ("httpx", "openai._base_client"):
        diagnostic_logger = logging.getLogger(name)
        if not any(
            getattr(item, "_ramjet_model_diagnostic_filter", False)
            for item in diagnostic_logger.filters
        ):
            diagnostic_logger.addFilter(ModelDiagnosticFilter())
    parsed = urlsplit(options["base_url"])
    options["base_url"] = urlunsplit(parsed._replace(query="", fragment=""))
    if parsed.query:
        import httpx
        from openai import DefaultAsyncHttpxClient, DefaultHttpxClient

        provider_query = httpx.URL(api_base).query

        def attach_query(request: httpx.Request) -> None:
            """Preserve the selected provider query and any operation parameters."""
            operation_query = request.url.query
            query = provider_query
            if operation_query:
                query += b"&" + operation_query
            request.url = request.url.copy_with(query=query)

        async def attach_async_query(request: httpx.Request) -> None:
            """Apply the same selected provider query on asynchronous SDK requests."""
            attach_query(request)

        options["http_client"] = DefaultHttpxClient(
            event_hooks={"request": [attach_query]}
        )
        if include_async_client:
            options["http_async_client"] = DefaultAsyncHttpxClient(
                event_hooks={"request": [attach_async_query]}
            )
    return options
