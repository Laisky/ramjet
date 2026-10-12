"""Resolve explicit request credentials without importing model tasks or settings."""

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
    # Reject identifiable session tokens without importing settings, tasks or SDKs.
    if api_key.count(".") == 2:
        import base64
        import json

        payload = api_key.split(".")[1]
        if len(payload) <= 8192:
            try:
                claims = json.loads(
                    base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))
                )
            except (ValueError, TypeError, UnicodeError, RecursionError):
                claims = None
            if isinstance(claims, dict) and claims.get("iss") == "laisky-sso":
                raise ValueError("A separate client model API key is required")
    return api_key


def resolve_model_credentials(
    api_key: str | None, api_base: str | None = DEFAULT_API_BASE
) -> dict[str, str]:
    """resolve_model_credentials supplies explicit SDK key and backend options.

    Keys remain transient request data. Provider syntax is checked without
    restricting internal addresses or introducing a destination allowlist.
    """
    from urllib.parse import parse_qsl, unquote

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
        # A session is never a provider URL option. Preserve all other URL syntax.
        candidates = [
            parsed.username or "",
            parsed.password or "",
            parsed.fragment,
            *parsed.path.split("/"),
            *(
                component
                for pair in parse_qsl(parsed.query, keep_blank_values=True)
                for component in pair
            ),
        ]
        for candidate in candidates:
            candidate = unquote(candidate)
            parts = candidate.split()
            if len(parts) == 2 and parts[0].lower() == "bearer":
                candidate = parts[1]
            try:
                require_api_key(candidate)
            except ValueError as error:
                if str(error) == "A separate client model API key is required":
                    raise ValueError(
                        "SSO sessions cannot be model provider options"
                    ) from None
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
