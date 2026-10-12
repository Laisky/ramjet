"""Private ownership authority comes only from the verified local registry."""

import hmac
import time

import aiohttp.web

from .private_auth import PrivateUser
from .private_registry import NamespaceConflict, RegistryDenied, RegistryUnavailable


def require_private_namespace(
    storage,
    user,
    *,
    bucket: str,
    root_prefix: str,
    create: bool = False,
    operation: str | None = None,
) -> bool:
    """require_private_namespace rechecks exact identity/target/operation authority.

    Historical objects and S3 markers never supply ownership. New upload
    initialization is an atomic registry transaction with an empty independently
    chosen namespace check. Stopping old writers remains a rollout prerequisite;
    SQLite and object storage do not form a distributed transaction.
    """
    selected = (
        user.operation
        if operation is None and isinstance(user, PrivateUser)
        else operation
    )
    if (
        not isinstance(user, PrivateUser)
        or selected != user.operation
        or user.principal.expires_at <= time.time()
        or user.principal.issuer != "laisky-sso"
    ):
        raise aiohttp.web.HTTPUnauthorized(
            text="An independently verified SSO session is required"
        )
    try:
        if create:
            if selected != "upload":
                raise RegistryDenied("Initialization is limited to upload")

            def is_empty(namespace):
                """is_empty inspects only the server's independently chosen fresh prefix."""
                return (
                    next(
                        iter(
                            storage.list_objects(
                                bucket_name=bucket,
                                prefix=f"{root_prefix}/{namespace}/",
                                recursive=True,
                            )
                        ),
                        None,
                    )
                    is None
                )

            grant = user.registry.initialize(
                user.principal.subject, operation="upload", is_empty=is_empty
            )
        else:
            grant = user.registry.lookup(user.principal.subject, selected)
        if grant is None:
            return False
        if (
            grant.bucket != bucket
            or grant.root_prefix != root_prefix
            or grant.operation != selected
            or not hmac.compare_digest(grant.principal_id, user.principal.subject)
            or (user.uid and not hmac.compare_digest(user.uid, grant.namespace))
        ):
            raise RegistryDenied("Private authority does not match the request")
        user.grant = grant
        user.uid = grant.namespace
        return True
    except NamespaceConflict:
        raise aiohttp.web.HTTPConflict(
            text="Fresh private namespace conflicts with existing data"
        ) from None
    except RegistryUnavailable:
        raise aiohttp.web.HTTPServiceUnavailable(
            text="Private ownership registry is not available"
        ) from None
    except RegistryDenied:
        raise aiohttp.web.HTTPForbidden(
            text="Private namespace grant is required"
        ) from None
