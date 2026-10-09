"""Configured authorization for privileged task dispatch."""

import hmac

from aiohttp import web

from ramjet import settings


def require_operator(request):
    """require_operator checks one configured bearer and returns no credential data.

    Parameters:
        request: The aiohttp request whose Authorization header is checked.
    Returns:
        None when authorized; otherwise raises HTTP 401 or configuration HTTP 503.
    """
    expected = getattr(settings, "OPERATOR_API_TOKEN", "")
    if not isinstance(expected, str) or not expected:
        raise web.HTTPServiceUnavailable(text="operator endpoint is not configured")
    headers = request.headers.getall("Authorization", [])
    if len(headers) != 1 or not headers[0].startswith("Bearer "):
        raise web.HTTPUnauthorized(text="operator authentication required")
    supplied = headers[0][len("Bearer ") :]
    if not supplied or not hmac.compare_digest(supplied.encode(), expected.encode()):
        raise web.HTTPUnauthorized(text="operator authentication required")
