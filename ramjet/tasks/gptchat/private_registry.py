"""Server-local operator grants for verified private-data principals."""

from dataclasses import dataclass
import os
from pathlib import Path
import re
import sqlite3
import stat
from typing import Callable
from uuid import UUID, uuid4

PRIVATE_OPERATIONS = frozenset(
    {
        "list-files",
        "upload",
        "delete-files",
        "list-chatbots",
        "read-private",
        "build",
        "activate",
        "publish-share",
    }
)
_NAMESPACE = re.compile(r"pns3-[0-9a-f]{32}")
_SCHEMA_VERSION = 3
_OPERATION_SQL = ", ".join("'" + value + "'" for value in sorted(PRIVATE_OPERATIONS))
_SCHEMA = {
    "registry_policy": """CREATE TABLE registry_policy (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        version INTEGER NOT NULL CHECK (version = 3),
        bucket TEXT NOT NULL,
        root_prefix TEXT NOT NULL,
        allow_fresh_initialization INTEGER NOT NULL CHECK (allow_fresh_initialization IN (0, 1))
    )""",
    "namespace_owners": """CREATE TABLE namespace_owners (
        principal_id TEXT PRIMARY KEY NOT NULL,
        namespace TEXT UNIQUE NOT NULL
    )""",
    "operation_grants": """CREATE TABLE operation_grants (
        principal_id TEXT NOT NULL REFERENCES namespace_owners(principal_id),
        operation TEXT NOT NULL CHECK (operation IN ("""
    + _OPERATION_SQL
    + """)),
        PRIMARY KEY (principal_id, operation)
    )""",
}


class RegistryUnavailable(RuntimeError):
    """RegistryUnavailable reports a disabled, inaccessible or invalid registry."""


class RegistryDenied(RuntimeError):
    """RegistryDenied reports malformed input or absence of an exact authorization."""


class NamespaceConflict(RegistryDenied):
    """NamespaceConflict refuses an occupied or already assigned fresh namespace."""


@dataclass(frozen=True)
class FrozenGrant:
    """FrozenGrant binds a verified principal to one exact operation and storage target."""

    principal_id: str
    namespace: str
    operation: str
    bucket: str
    root_prefix: str


def _principal(value: str) -> str:
    """_principal accepts only a nonzero canonical UUID string from the identity verifier."""
    try:
        identifier = UUID(value)
        if str(identifier) != value or identifier.int == 0:
            raise ValueError()
    except (ValueError, TypeError, AttributeError):
        raise RegistryDenied("A canonical private-data principal is required") from None
    return value


def _operation(value: str) -> str:
    """_operation accepts only exact supported operation names without aliases."""
    if not isinstance(value, str) or value not in PRIVATE_OPERATIONS:
        raise RegistryDenied("Unsupported private-data operation")
    return value


def _target(value: str) -> str:
    """_target preserves the exact server-configured storage target without controls."""
    if not isinstance(value, str) or not value or any(ord(char) < 33 for char in value):
        raise RegistryUnavailable("Invalid private registry storage target")
    return value


def _absolute_path(path) -> Path:
    """_absolute_path accepts only an explicit absolute local path, never a SQLite URI."""
    try:
        result = Path(path)
        if not result.is_absolute():
            raise ValueError()
    except (TypeError, ValueError):
        raise RegistryUnavailable(
            "Private-data registry is disabled or unavailable"
        ) from None
    return result


class Registry:
    """Registry reads exact operator grants and atomically initializes permitted fresh scopes.

    Business construction never creates a file or schema. The path and storage target
    are trusted server inputs, never request fields. Operators must protect the local
    file and its parent directory from untrusted writes; SQLite is the authority and
    S3 markers have no role. Historical grant import is deliberately unsupported.
    """

    def __init__(self, path, *, bucket: str, root_prefix: str):
        """__init__ validates the existing schema and its pinned server storage target."""
        self.path = _absolute_path(path)
        self.bucket = _target(bucket)
        self.root_prefix = _target(root_prefix)
        connection = self._connect()
        try:
            self._validate(connection)
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        """_connect opens existing effective-UID-owned authority with protected parent permissions.

        No chmod or ownership changes are performed. Different trusted owner/group
        requirements need a separately approved operator deployment contract.
        """
        try:
            info = self.path.lstat()
            parent = self.path.parent.lstat()
            effective_uid = getattr(os, "geteuid", lambda: None)()
            if (
                not stat.S_ISREG(info.st_mode)
                or not stat.S_ISDIR(parent.st_mode)
                or effective_uid is None
                or info.st_uid != effective_uid
                or parent.st_uid not in (effective_uid, 0)
                or info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
                or parent.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
            ):
                raise ValueError()
            connection = sqlite3.connect(
                self.path.as_uri() + "?mode=rw",
                uri=True,
                timeout=5,
                isolation_level=None,
            )
            connection.execute("PRAGMA foreign_keys = ON")
            return connection
        except (OSError, ValueError, sqlite3.Error):
            raise RegistryUnavailable("Private-data registry is unavailable") from None

    def _validate(self, connection: sqlite3.Connection) -> bool:
        """_validate refuses unknown schemas, triggers, policies or storage-target changes."""
        try:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            objects = connection.execute(
                "SELECT type, name, sql FROM sqlite_master "
                "WHERE name NOT LIKE 'sqlite_%'"
            ).fetchall()
            actual = {}
            for kind, name, sql in objects:
                if kind != "table" or name not in _SCHEMA or sql != _SCHEMA[name]:
                    raise ValueError()
                actual[name] = sql
            if version != _SCHEMA_VERSION or actual != _SCHEMA:
                raise ValueError()
            rows = connection.execute(
                "SELECT id, version, bucket, root_prefix, allow_fresh_initialization "
                "FROM registry_policy"
            ).fetchall()
            if len(rows) != 1:
                raise ValueError()
            row = rows[0]
            if row[:4] != (1, 3, self.bucket, self.root_prefix) or row[4] not in (0, 1):
                raise ValueError()
            if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
                raise ValueError()
            return row[4] == 1
        except (sqlite3.Error, ValueError, TypeError):
            raise RegistryUnavailable(
                "Private-data registry schema or target is invalid"
            ) from None

    def check_ready(self) -> bool:
        """check_ready validates authority before issuer transmission and returns fresh policy."""
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            result = self._validate(connection)
            connection.commit()
            return result
        except sqlite3.Error:
            raise RegistryUnavailable(
                "Private-data registry readiness check failed"
            ) from None
        finally:
            connection.close()

    def _lookup(
        self, connection: sqlite3.Connection, principal_id: str, operation: str
    ) -> FrozenGrant | None:
        """_lookup returns only a validated namespace with an exact explicit operation grant."""
        row = connection.execute(
            "SELECT owner.namespace FROM namespace_owners AS owner "
            "JOIN operation_grants AS grant ON grant.principal_id = owner.principal_id "
            "WHERE owner.principal_id = ? AND grant.operation = ?",
            (principal_id, operation),
        ).fetchone()
        if row is None:
            return None
        namespace = row[0]
        if not isinstance(namespace, str) or not _NAMESPACE.fullmatch(namespace):
            raise RegistryUnavailable(
                "Private-data registry contains an invalid namespace"
            )
        return FrozenGrant(
            principal_id, namespace, operation, self.bucket, self.root_prefix
        )

    def lookup(self, principal_id: str, operation: str) -> FrozenGrant | None:
        """lookup returns an exact grant or None without creating or inferring any ownership."""
        principal_id, operation = _principal(principal_id), _operation(operation)
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            self._validate(connection)
            result = self._lookup(connection, principal_id, operation)
            connection.commit()
            return result
        except sqlite3.Error:
            raise RegistryUnavailable("Private-data registry lookup failed") from None
        finally:
            connection.close()

    def initialize(
        self,
        principal_id: str,
        operation: str = "upload",
        *,
        is_empty: Callable[[str], bool],
    ) -> FrozenGrant:
        """initialize atomically grants a new empty namespace only under explicit fresh policy.

        Only upload can initialize. Existing exact grants are reused; an existing
        owner without upload permission cannot escalate. The empty-prefix callback
        inspects only the generated candidate, and must return exactly True.
        Its check is not a distributed transaction with S3: separately authorized
        rollout must first stop legacy writers and drain their background jobs.
        """
        principal_id, operation = _principal(principal_id), _operation(operation)
        if operation != "upload":
            raise RegistryDenied("Only upload may initialize a fresh private namespace")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            allowed = self._validate(connection)
            existing = self._lookup(connection, principal_id, operation)
            if existing is not None:
                connection.commit()
                return existing
            if (
                connection.execute(
                    "SELECT 1 FROM namespace_owners WHERE principal_id = ?",
                    (principal_id,),
                ).fetchone()
                is not None
            ):
                raise RegistryDenied("Private namespace has no upload grant")
            if not allowed:
                raise RegistryDenied(
                    "Fresh private namespace initialization is disabled"
                )
            namespace = "pns3-" + uuid4().hex
            if (
                connection.execute(
                    "SELECT 1 FROM namespace_owners WHERE namespace = ?", (namespace,)
                ).fetchone()
                is not None
            ):
                raise NamespaceConflict(
                    "Fresh private namespace conflicts with existing data"
                )
            try:
                empty = is_empty(namespace)
            except Exception:
                raise RegistryUnavailable(
                    "Fresh private namespace could not be verified"
                ) from None
            if empty is not True:
                raise NamespaceConflict(
                    "Fresh private namespace conflicts with existing data"
                )
            connection.execute(
                "INSERT INTO namespace_owners(principal_id, namespace) VALUES (?, ?)",
                (principal_id, namespace),
            )
            connection.executemany(
                "INSERT INTO operation_grants(principal_id, operation) VALUES (?, ?)",
                [(principal_id, value) for value in sorted(PRIVATE_OPERATIONS)],
            )
            grant = self._lookup(connection, principal_id, operation)
            if grant is None:
                raise RegistryUnavailable(
                    "Fresh private namespace initialization failed"
                )
            connection.commit()
            return grant
        except sqlite3.IntegrityError:
            raise NamespaceConflict(
                "Fresh private namespace conflicts with existing data"
            ) from None
        except sqlite3.Error:
            raise RegistryUnavailable(
                "Fresh private namespace initialization failed"
            ) from None
        finally:
            connection.close()


def initialize_registry(
    path, *, bucket: str, root_prefix: str, allow_fresh_initialization: bool = False
) -> Registry:
    """initialize_registry explicitly creates an exclusive operator registry, never from an API.

    This operator helper is used only with disposable test paths in this task.
    Existing files are never replaced and historical records are never imported.
    """
    path = _absolute_path(path)
    bucket, root_prefix = _target(bucket), _target(root_prefix)
    if type(allow_fresh_initialization) is not bool:
        raise RegistryUnavailable("Invalid fresh initialization policy")
    connection = None
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
        connection = sqlite3.connect(path.as_uri() + "?mode=rw", uri=True)
        connection.execute("PRAGMA foreign_keys = ON")
        for statement in _SCHEMA.values():
            connection.execute(statement)
        connection.execute("PRAGMA user_version = 3")
        connection.execute(
            "INSERT INTO registry_policy VALUES (?, ?, ?, ?, ?)",
            (1, 3, bucket, root_prefix, int(allow_fresh_initialization)),
        )
        connection.commit()
    except (OSError, sqlite3.Error):
        raise RegistryUnavailable("Operator registry initialization failed") from None
    finally:
        if connection is not None:
            connection.close()
    return Registry(path, bucket=bucket, root_prefix=root_prefix)
