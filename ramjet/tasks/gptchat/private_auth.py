"""Separate verified private ownership from transient model-provider credentials."""

from dataclasses import dataclass
import functools

import aiohttp.web

from ramjet import settings
from ramjet.credentials import DEFAULT_API_BASE, resolve_request_credentials
from ramjet.sso import (
    IssuerUnavailable,
    SessionInvalid,
    VerifiedPrincipal,
    WhoAmIVerifier,
    require_session_token,
)

from .private_registry import (
    FrozenGrant,
    PRIVATE_OPERATIONS,
    Registry,
    RegistryDenied,
    RegistryUnavailable,
)

_METADATA_OPERATIONS = frozenset({"list-files", "delete-files", "list-chatbots"})
SESSION_VERIFIER = WhoAmIVerifier()


@dataclass(repr=False)
class PrivateUser:
    """PrivateUser contains an online identity and model options, never a session token."""

    principal: VerifiedPrincipal
    registry: Registry
    operation: str
    grant: FrozenGrant | None
    uid: str
    apikey: str
    api_base: str
    chat_model: str
    is_paid: bool = False
    n_concurrent: int = 3


_PRIVATE_USER_KEY = aiohttp.web.AppKey("ramjet.private_verified_user", PrivateUser)


def configured_registry() -> Registry:
    """configured_registry opens only an operator-created local authority, disabled by default."""
    return Registry(
        getattr(settings, "PRIVATE_NAMESPACE_REGISTRY_PATH", None),
        bucket=settings.OPENAI_S3_CHUNK_CACHE_BUCKET,
        root_prefix=settings.OPENAI_S3_EMBEDDINGS_PREFIX,
    )


def _deny_registry(error):
    """_deny_registry maps authority failures to fixed responses without paths or grant details."""
    if isinstance(error, RegistryUnavailable):
        raise aiohttp.web.HTTPServiceUnavailable(
            text="Private ownership registry is not available"
        ) from None
    raise aiohttp.web.HTTPForbidden(
        text="Private namespace grant is required"
    ) from None


async def authorize_private_request(request, operation: str) -> PrivateUser:
    """authorize_private_request verifies the original session before private side effects.

    All non-upload operations require an existing exact grant. Fresh upload
    initialization is permitted only by the registry's explicit server policy,
    and occurs later after multipart validation and bounded admission.
    """
    if operation not in PRIVATE_OPERATIONS:
        raise aiohttp.web.HTTPNotFound(text="Unknown private operation")
    try:
        token = require_session_token(request.headers)
    except SessionInvalid:
        raise aiohttp.web.HTTPUnauthorized(text="An SSO session is required") from None
    try:
        registry = configured_registry()
        allow_fresh = registry.check_ready()
    except (RegistryDenied, RegistryUnavailable) as error:
        _deny_registry(error)
    try:
        principal = await SESSION_VERIFIER.verify(token)
    except SessionInvalid:
        raise aiohttp.web.HTTPUnauthorized(
            text="Invalid or expired SSO session"
        ) from None
    except IssuerUnavailable:
        raise aiohttp.web.HTTPServiceUnavailable(
            text="SSO service is unavailable"
        ) from None
    try:
        grant = registry.lookup(principal.subject, operation)
    except (RegistryDenied, RegistryUnavailable) as error:
        _deny_registry(error)
    if grant is None and (operation != "upload" or not allow_fresh):
        raise aiohttp.web.HTTPForbidden(text="Private namespace grant is required")
    authorization = request.headers.get("Authorization", "")
    if not authorization and operation in _METADATA_OPERATIONS:
        options = {"api_key": "", "base_url": DEFAULT_API_BASE}
    else:
        try:
            options = resolve_request_credentials(
                authorization, request.headers.get("X-Laisky-Openai-Api-Base", "")
            )
        except ValueError:
            raise aiohttp.web.HTTPUnauthorized(
                text="A separate client model API key is required"
            ) from None
    user = PrivateUser(
        principal=principal,
        registry=registry,
        operation=operation,
        grant=grant,
        uid=grant.namespace if grant else "",
        apikey=options["api_key"],
        api_base=options["base_url"],
        chat_model=request.query.get("model", "") or "gpt-4o-mini",
    )
    request[_PRIVATE_USER_KEY] = user
    return user


def verified_private_user(request, operation: str) -> PrivateUser:
    """verified_private_user consumes only the async entry's exact request-local identity."""
    user = request.get(_PRIVATE_USER_KEY)
    if (
        not isinstance(user, PrivateUser)
        or not isinstance(user.principal, VerifiedPrincipal)
        or user.operation != operation
    ):
        raise aiohttp.web.HTTPUnauthorized(
            text="An independently verified SSO session is required"
        )
    return user


def authenticate_private(operation):
    """authenticate_private verifies one private entry before parsing or scheduling its work."""

    def decorate(func):
        """decorate binds the handler to one explicit operation or server route resolver."""

        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            """wrapper obtains online identity and exact authority before invoking the handler."""
            selected = operation(self.request) if callable(operation) else operation
            user = await authorize_private_request(self.request, selected)
            return await func(self, user, *args, **kwargs)

        return wrapper

    return decorate


def authenticate_private_sync(operation):
    """authenticate_private_sync enforces the verified entry's operation in a worker."""

    def decorate(func):
        """decorate keeps worker authorization separate from model-only decorators."""

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            """wrapper refuses direct worker calls without the async request's exact identity."""
            user = verified_private_user(self.request, operation)
            return func(self, user, *args, **kwargs)

        return wrapper

    return decorate


def private_context_operation(request) -> str:
    """private_context_operation maps only explicitly declared private POST operations."""
    operations = {"/build": "build", "/active": "activate", "/share": "publish-share"}
    selected = operations.get(request.match_info.get("op", ""))
    if selected is None:
        raise aiohttp.web.HTTPNotFound(text="Unknown private operation")
    return selected
