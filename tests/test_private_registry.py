"""Offline contracts for operator-owned SQLite private-data grants."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
import importlib.util
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "ramjet/tasks/gptchat/private_registry.py"
)
SPEC = importlib.util.spec_from_file_location(
    "private_registry_test_module", MODULE_PATH
)
registry = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = registry
SPEC.loader.exec_module(registry)


class PrivateRegistryTests(unittest.TestCase):
    """PrivateRegistryTests validate exact grants without production storage or identities."""

    def setUp(self):
        """setUp creates an owned temporary directory and synthetic principal IDs."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "operator.sqlite3"
        self.bucket = "synthetic-private"
        self.root = "synthetic/root"
        self.subject = str(uuid4())
        self.other = str(uuid4())

    def create(self, allowed=False):
        """create explicitly initializes only a disposable operator fixture."""
        return registry.initialize_registry(
            self.path,
            bucket=self.bucket,
            root_prefix=self.root,
            allow_fresh_initialization=allowed,
        )

    def open(self, **kwargs):
        """open uses the existing-file business interface with exact fixture targets."""
        return registry.Registry(
            kwargs.get("path", self.path),
            bucket=kwargs.get("bucket", self.bucket),
            root_prefix=kwargs.get("root_prefix", self.root),
        )

    def mutate(self, query, parameters=()):
        """mutate acts only as the synthetic operator, never as a business request."""
        with sqlite3.connect(self.path) as connection:
            connection.execute(query, parameters)

    def test_disabled_and_missing_registry_never_create_a_file(self):
        """test_disabled_and_missing_registry refuses unavailable paths without schema creation."""
        for path in (None, "", Path("relative.sqlite"), self.path):
            with (
                self.subTest(path=path),
                self.assertRaises(registry.RegistryUnavailable),
            ):
                self.open(path=path)
        self.assertFalse(self.path.exists())

    def test_business_open_refuses_directory_and_symbolic_link(self):
        """test_business_open rejects nonregular paths and URI-style inputs."""
        self.create()
        link = self.path.parent / "link.sqlite"
        link.symlink_to(self.path)
        for path in (self.path.parent, link, "file:" + str(self.path)):
            with (
                self.subTest(path=path),
                self.assertRaises(registry.RegistryUnavailable),
            ):
                self.open(path=path)

    def test_group_or_other_writable_registry_fails_without_permission_repair(self):
        """test_group_or_other_writable_registry refuses insecure temporary files as found."""
        instance = self.create()
        for mode in (0o620, 0o602, 0o666):
            self.path.chmod(mode)
            with (
                self.subTest(mode=mode),
                self.assertRaises(registry.RegistryUnavailable),
            ):
                instance.check_ready()
            self.assertEqual(self.path.stat().st_mode & 0o777, mode)
        self.path.chmod(0o600)
        self.assertFalse(instance.check_ready())

    def test_group_or_other_writable_parent_fails_without_permission_repair(self):
        """test_group_or_other_writable_parent denies untrusted local authority directories."""
        instance = self.create()
        parent = self.path.parent
        try:
            for mode in (0o720, 0o702, 0o777):
                parent.chmod(mode)
                with (
                    self.subTest(mode=mode),
                    self.assertRaises(registry.RegistryUnavailable),
                ):
                    instance.check_ready()
                self.assertEqual(parent.stat().st_mode & 0o777, mode)
        finally:
            parent.chmod(0o700)
        self.assertFalse(instance.check_ready())

    def test_different_effective_uid_denies_existing_registry(self):
        """test_different_effective_uid treats a differently owned file as unavailable."""
        instance = self.create()
        owner = self.path.stat().st_uid
        with patch.object(registry.os, "geteuid", return_value=owner + 1):
            with self.assertRaises(registry.RegistryUnavailable):
                instance.check_ready()
        self.assertFalse(instance.check_ready())

    def test_immediate_parent_owner_must_be_effective_uid_or_root(self):
        """test_immediate_parent_owner refuses an untrusted directory without changing real ownership."""
        instance = self.create()
        real_lstat = registry.Path.lstat
        parent_owner = [self.path.stat().st_uid + 1000]

        def inspected(path, *args, **kwargs):
            """inspected alters only the immediate parent's synthetic stat ownership."""
            info = real_lstat(path, *args, **kwargs)
            if path == self.path.parent:
                values = list(info)
                values[4] = parent_owner[0]
                return registry.os.stat_result(values)
            return info

        with patch.object(registry.Path, "lstat", autospec=True, side_effect=inspected):
            with self.assertRaises(registry.RegistryUnavailable):
                instance.check_ready()
            parent_owner[0] = 0
            self.assertFalse(instance.check_ready())
            parent_owner[0] = self.path.stat().st_uid
            self.assertFalse(instance.check_ready())
        self.assertFalse(instance.check_ready())

    def test_operator_creation_is_exclusive_and_disabled_by_default(self):
        """test_operator_creation leaves existing registry policy intact."""
        instance = self.create()
        with self.assertRaises(registry.RegistryDenied):
            instance.initialize(self.subject, is_empty=lambda scope: True)
        original = self.path.read_bytes()
        with self.assertRaises(registry.RegistryUnavailable):
            self.create(True)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertIsNone(instance.lookup(self.subject, "upload"))

    def test_readiness_validates_policy_before_any_identity_or_grant(self):
        """test_readiness reads explicit policy without creating identity grants."""
        instance = self.create()
        self.assertFalse(instance.check_ready())
        self.mutate("UPDATE registry_policy SET allow_fresh_initialization = 1")
        self.assertTrue(instance.check_ready())
        self.assertIsNone(instance.lookup(self.subject, "upload"))
        self.mutate("UPDATE registry_policy SET root_prefix = ?", ("other",))
        with self.assertRaises(registry.RegistryUnavailable):
            instance.check_ready()

    def test_empty_database_and_unknown_schema_fail_closed(self):
        """test_empty_database rejects a file that is not the exact operator schema."""
        self.path.touch()
        with self.assertRaises(registry.RegistryUnavailable):
            self.open()
        with sqlite3.connect(self.path) as connection:
            connection.execute("CREATE TABLE arbitrary (x INTEGER)")
        with self.assertRaises(registry.RegistryUnavailable):
            self.open()

    def test_schema_changes_are_rechecked_after_business_construction(self):
        """test_schema_changes detects changed version, extra tables and altered metadata."""
        instance = self.create(True)
        self.mutate("PRAGMA user_version = 99")
        with self.assertRaises(registry.RegistryUnavailable):
            instance.lookup(self.subject, "upload")
        self.mutate("PRAGMA user_version = 3")
        self.mutate("CREATE TABLE unexpected (value TEXT)")
        with self.assertRaises(registry.RegistryUnavailable):
            instance.initialize(self.subject, is_empty=lambda scope: True)

    def test_wrong_bucket_or_root_cannot_reuse_registry_grants(self):
        """test_wrong_bucket_or_root pins authority to the exact server storage target."""
        self.create(True)
        for arguments in (
            {"bucket": "other"},
            {"root_prefix": "other"},
            {"root_prefix": self.root + "/"},
        ):
            with (
                self.subTest(arguments=arguments),
                self.assertRaises(registry.RegistryUnavailable),
            ):
                self.open(**arguments)
        self.mutate("UPDATE registry_policy SET bucket = ?", ("other",))
        with self.assertRaises(registry.RegistryUnavailable):
            self.open()

    def test_missing_principal_or_operation_grant_does_not_infer_authority(self):
        """test_missing_principal_or_operation_grant never grants from namespace resemblance."""
        instance = self.create(True)
        grant = instance.initialize(self.subject, is_empty=lambda scope: True)
        self.assertIsNone(instance.lookup(self.other, "upload"))
        self.mutate(
            "DELETE FROM operation_grants WHERE principal_id = ? AND operation = ?",
            (self.subject, "delete-files"),
        )
        self.assertIsNone(instance.lookup(self.subject, "delete-files"))
        self.assertEqual(
            instance.lookup(self.subject, "upload").namespace, grant.namespace
        )

    def test_exact_operations_and_canonical_principals_only(self):
        """test_exact_operations denies aliases, injection strings and malformed identity values."""
        instance = self.create(True)
        for operation in (
            "UPLOAD",
            " upload",
            "read",
            "../upload",
            "upload' OR 1=1 --",
            "",
            None,
        ):
            with (
                self.subTest(operation=operation),
                self.assertRaises(registry.RegistryDenied),
            ):
                instance.lookup(self.subject, operation)
        for principal in (
            "",
            self.subject.upper(),
            self.subject.replace("-", ""),
            "not-a-uuid",
            None,
            "00000000-0000-0000-0000-000000000000",
        ):
            with (
                self.subTest(principal=principal),
                self.assertRaises(registry.RegistryDenied),
            ):
                instance.lookup(principal, "upload")
        self.assertIsNone(instance.lookup(self.subject, "upload"))

    def test_fresh_initialization_grants_exact_operations_and_frozen_target(self):
        """test_fresh_initialization stores independent scope and explicit immutable operation grants."""
        instance = self.create(True)
        checked = []
        grant = instance.initialize(
            self.subject, is_empty=lambda scope: checked.append(scope) or True
        )
        self.assertEqual(checked, [grant.namespace])
        self.assertRegex(grant.namespace, r"^pns3-[0-9a-f]{32}$")
        self.assertEqual(
            (grant.principal_id, grant.operation, grant.bucket, grant.root_prefix),
            (self.subject, "upload", self.bucket, self.root),
        )
        with self.assertRaises(FrozenInstanceError):
            grant.namespace = "other"
        for operation in registry.PRIVATE_OPERATIONS:
            exact = instance.lookup(self.subject, operation)
            self.assertEqual(exact.namespace, grant.namespace)
            self.assertEqual(exact.operation, operation)
        second = instance.initialize(self.other, is_empty=lambda scope: True)
        self.assertNotEqual(second.namespace, grant.namespace)

    def test_initialization_only_supports_upload(self):
        """test_initialization_only_supports_upload cannot create ownership through a read or delete."""
        instance = self.create(True)
        checked = []
        for operation in registry.PRIVATE_OPERATIONS - {"upload"}:
            with (
                self.subTest(operation=operation),
                self.assertRaises(registry.RegistryDenied),
            ):
                instance.initialize(
                    self.subject,
                    operation,
                    is_empty=lambda scope: checked.append(scope) or True,
                )
        self.assertEqual(checked, [])
        self.assertIsNone(instance.lookup(self.subject, "upload"))

    def test_existing_grant_is_reused_without_prefix_inspection_or_policy_escalation(
        self,
    ):
        """test_existing_grant returns the exact scope even when fresh creation is subsequently disabled."""
        instance = self.create(True)
        original = instance.initialize(self.subject, is_empty=lambda scope: True)
        self.mutate("UPDATE registry_policy SET allow_fresh_initialization = 0")
        repeated = instance.initialize(
            self.subject,
            is_empty=lambda scope: self.fail("existing grant must not allocate"),
        )
        self.assertEqual(repeated, original)
        with self.assertRaises(registry.RegistryDenied):
            instance.initialize(self.other, is_empty=lambda scope: True)

    def test_owner_missing_upload_cannot_escalate_from_other_operation(self):
        """test_owner_missing_upload does not fill removed operator permissions."""
        instance = self.create(True)
        original = instance.initialize(self.subject, is_empty=lambda scope: True)
        self.mutate(
            "DELETE FROM operation_grants WHERE principal_id = ? AND operation = ?",
            (self.subject, "upload"),
        )
        with self.assertRaises(registry.RegistryDenied):
            instance.initialize(
                self.subject,
                is_empty=lambda scope: self.fail("existing owner must not allocate"),
            )
        self.assertIsNone(instance.lookup(self.subject, "upload"))
        self.assertEqual(
            instance.lookup(self.subject, "list-files").namespace, original.namespace
        )

    def test_occupied_prefix_and_unknown_storage_result_leave_no_grants(self):
        """test_occupied_prefix denies any unverified or occupied storage without adoption."""
        instance = self.create(True)
        for result in (False, None, 1, "empty"):
            with (
                self.subTest(result=result),
                self.assertRaises(registry.NamespaceConflict),
            ):
                instance.initialize(self.subject, is_empty=lambda scope: result)
            self.assertIsNone(instance.lookup(self.subject, "upload"))
        self.assertIsNotNone(
            instance.initialize(self.subject, is_empty=lambda scope: True)
        )

    def test_storage_failure_rolls_back_and_keeps_exception_payload_private(self):
        """test_storage_failure exposes no callback contents and persists no owner or grants."""
        instance = self.create(True)

        def fail(scope):
            """fail simulates a storage outage without any network request."""
            raise RuntimeError("synthetic secret must not appear")

        with self.assertRaises(registry.RegistryUnavailable) as error:
            instance.initialize(self.subject, is_empty=fail)
        self.assertNotIn("synthetic secret", str(error.exception))
        self.assertIsNone(instance.lookup(self.subject, "upload"))
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM namespace_owners").fetchone()[
                    0
                ],
                0,
            )
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM operation_grants").fetchone()[
                    0
                ],
                0,
            )

    def test_candidate_namespace_conflict_never_reassigns_existing_owner(self):
        """test_candidate_namespace_conflict refuses duplicate independent candidates."""
        instance = self.create(True)
        candidate = uuid4()
        with patch.object(registry, "uuid4", return_value=candidate):
            original = instance.initialize(self.subject, is_empty=lambda scope: True)
            with self.assertRaises(registry.NamespaceConflict):
                instance.initialize(
                    self.other,
                    is_empty=lambda scope: self.fail(
                        "collision must reject before storage"
                    ),
                )
        self.assertEqual(instance.lookup(self.subject, "upload"), original)
        self.assertIsNone(instance.lookup(self.other, "upload"))

    def test_sql_failure_after_owner_insert_rolls_back_entire_grant_creation(self):
        """test_sql_failure_after_owner_insert verifies ownership and all rights commit together."""
        instance = self.create(True)
        original_lookup = instance._lookup

        def unavailable_after_insert(connection, principal, operation):
            """unavailable_after_insert fails only after the transaction has inserted ownership."""
            result = original_lookup(connection, principal, operation)
            if result is not None:
                raise registry.RegistryUnavailable(
                    "synthetic final verification failure"
                )
            return result

        with patch.object(instance, "_lookup", side_effect=unavailable_after_insert):
            with self.assertRaises(registry.RegistryUnavailable):
                instance.initialize(self.subject, is_empty=lambda scope: True)
        self.assertIsNone(instance.lookup(self.subject, "upload"))
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM namespace_owners").fetchone()[
                    0
                ],
                0,
            )

    def test_concurrent_initializers_for_same_principal_commit_one_scope(self):
        """test_concurrent_initializers uses real SQLite locking to prevent duplicate ownership."""
        instance = self.create(True)
        checked = []

        def initialize(index):
            """initialize uses a separate business connection for each synthetic worker."""
            return self.open().initialize(
                self.subject, is_empty=lambda scope: checked.append(scope) or True
            )

        with ThreadPoolExecutor(max_workers=4) as workers:
            grants = list(workers.map(initialize, range(4)))
        self.assertEqual(len({grant.namespace for grant in grants}), 1)
        self.assertEqual(checked, [grants[0].namespace])
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM namespace_owners").fetchone()[
                    0
                ],
                1,
            )
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM operation_grants").fetchone()[
                    0
                ],
                len(registry.PRIVATE_OPERATIONS),
            )

    def test_grants_persist_across_instances_without_provider_or_api_key_inputs(self):
        """test_grants_persist proves model credentials have no authority in registry selection."""
        original = self.create(True).initialize(
            self.subject, is_empty=lambda scope: True
        )
        reopened = self.open()
        self.assertEqual(reopened.lookup(self.subject, "upload"), original)
        self.assertIsNone(reopened.lookup(self.other, "upload"))
        with sqlite3.connect(self.path) as connection:
            fields = connection.execute(
                "PRAGMA table_info(namespace_owners)"
            ).fetchall()
        self.assertEqual({field[1] for field in fields}, {"principal_id", "namespace"})

    def test_invalid_namespace_record_and_removed_file_fail_closed(self):
        """test_invalid_namespace_record rejects operator corruption and missing durable authority."""
        instance = self.create(True)
        instance.initialize(self.subject, is_empty=lambda scope: True)
        self.mutate(
            "UPDATE namespace_owners SET namespace = ? WHERE principal_id = ?",
            ("../legacy", self.subject),
        )
        with self.assertRaises(registry.RegistryUnavailable):
            instance.lookup(self.subject, "upload")
        self.path.unlink()
        with self.assertRaises(registry.RegistryUnavailable):
            instance.lookup(self.subject, "upload")
        self.assertFalse(self.path.exists())


if __name__ == "__main__":
    unittest.main()
