"""Resolve explicit request credentials without importing model tasks or settings."""

from urllib.parse import urlsplit, urlunsplit

DEFAULT_API_BASE = "https://api.openai.com/v1"


def require_api_key(api_key: str | None) -> str:
    """require_api_key returns an opaque model key and excludes identifiable Laisky sessions.

    Session recognition is a fail-closed credential-separation policy, not JWT
    authentication or general content inspection. Compound strings can contain
    a complete original session alongside another opaque value or provider JWT.
    Unrelated issuer claims and non-JWT opaque keys remain unchanged.
    """
    if (
        not isinstance(api_key, str)
        or not api_key
        or api_key.lower()
        in {"bearer", "null", "undefined", "default_proxy_token", "freetier"}
    ):
        raise ValueError("A valid client API key is required")
    # Keep these standard-library imports local for isolated source fixtures.
    import base64
    import json
    import re

    # Examine dot-separated compact components in each contiguous token run.
    # Sliding payload positions also cover a JWT preceded/followed by dot values.
    for run in re.finditer(r"[A-Za-z0-9_.-]+", api_key):
        components = run.group().split(".")
        for index in range(1, len(components) - 1):
            payload = components[index]
            if (
                not components[index - 1]
                or not components[index + 1]
                or not 0 < len(payload) <= 8192
            ):
                continue
            try:
                claims = json.loads(
                    base64.b64decode(
                        payload + "=" * (-len(payload) % 4),
                        altchars=b"-_",
                        validate=True,
                    )
                )
            except (ValueError, TypeError, UnicodeError, RecursionError):
                continue
            if isinstance(claims, dict) and claims.get("iss") == "laisky-sso":
                raise ValueError("A separate client model API key is required")
    if any(ord(char) < 33 or ord(char) > 126 for char in api_key):
        raise ValueError("A valid client API key is required")
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
            # netloc retains original JWT case, unlike parsed.hostname.
            *(
                unquote(component)
                for component in (
                    parsed.netloc,
                    parsed.fragment,
                    *parsed.path.split("/"),
                )
            ),
            # parse_qsl already applies the single URL-component decode.
            *(
                component
                for pair in parse_qsl(parsed.query, keep_blank_values=True)
                for component in pair
            ),
        ]
        for candidate in candidates:
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
