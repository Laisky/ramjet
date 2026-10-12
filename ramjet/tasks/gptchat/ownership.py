"""Authenticate provenance for newly initialized private credential namespaces."""

import hashlib
import hmac
import io
import json

from .namespace import credential_scope

_MARKER_NAME = "__ownership_v2/marker.json"
_MARKER_DOMAIN = b"ramjet-private-namespace-provenance-v2\x00"
_MARKER_LIMIT = 1024


def _read_marker(storage, bucket: str, object_name: str) -> bytes | None:
    """_read_marker returns bounded marker bytes or only an explicit missing-key result."""
    response = None
    try:
        response = storage.get_object(bucket_name=bucket, object_name=object_name)
        data = response.read(_MARKER_LIMIT + 1)
        if len(data) > _MARKER_LIMIT:
            raise ValueError("Private namespace ownership could not be verified")
        return data
    except Exception as error:
        if getattr(error, "code", None) == "NoSuchKey":
            return None
        raise
    finally:
        if response is not None:
            response.close()
            response.release_conn()


def _verify_marker(data: bytes, scope: str, key: str) -> None:
    """_verify_marker authenticates exact scope provenance with constant-time MAC comparison."""
    try:
        marker = json.loads(data)
        if not isinstance(marker, dict) or set(marker) != {"version", "scope", "proof"}:
            raise ValueError()
        if marker["version"] != 2 or marker["scope"] != scope:
            raise ValueError()
        proof = marker["proof"]
        expected = hmac.new(
            key.encode(), _MARKER_DOMAIN + scope.encode(), hashlib.sha256
        ).hexdigest()
        if not isinstance(proof, str) or not hmac.compare_digest(proof, expected):
            raise ValueError()
    except (ValueError, TypeError, UnicodeError):
        raise ValueError("Private namespace ownership could not be verified") from None


def require_private_namespace(
    storage, user, *, bucket: str, root_prefix: str, create: bool = False
) -> bool:
    """require_private_namespace admits authenticated new provenance or an empty scope.

    True admits a namespace carrying a MAC for the complete caller credential.
    False represents a still-empty, uninitialized namespace on a read. Creation
    requires an empty prefix; occupied unmarked/invalid-marked data is never
    adopted, overwritten or returned. Unknown storage failures propagate.

    All writers must run this boundary before rollout. This marker establishes
    new namespace creation, not factual ownership of any historical records.
    """
    # Recompute rather than trust a constructed user's UID.
    scope = credential_scope(user.apikey)
    if user.uid != scope:
        raise ValueError("Private namespace ownership could not be verified")
    prefix = f"{root_prefix}/{scope}/"
    object_name = prefix + _MARKER_NAME
    data = _read_marker(storage, bucket, object_name)
    if data is not None:
        _verify_marker(data, scope, user.apikey)
        return True

    if (
        next(
            iter(
                storage.list_objects(bucket_name=bucket, prefix=prefix, recursive=True)
            ),
            None,
        )
        is not None
    ):
        # Another same-key initializer may have completed during the first read.
        data = _read_marker(storage, bucket, object_name)
        if data is None:
            raise ValueError("Private namespace ownership could not be verified")
        _verify_marker(data, scope, user.apikey)
        return True
    if not create:
        return False
    marker = {
        "version": 2,
        "scope": scope,
        "proof": hmac.new(
            user.apikey.encode(), _MARKER_DOMAIN + scope.encode(), hashlib.sha256
        ).hexdigest(),
    }
    payload = json.dumps(marker, sort_keys=True, separators=(",", ":")).encode()
    storage.put_object(
        bucket_name=bucket,
        object_name=object_name,
        data=io.BytesIO(payload),
        length=len(payload),
        content_type="application/json",
    )
    # Observe persisted proof before dispatching work; a write error never grants access.
    data = _read_marker(storage, bucket, object_name)
    if data is None:
        raise ValueError("Private namespace ownership could not be verified")
    _verify_marker(data, scope, user.apikey)
    return True
