"""Historical objects and caller-controlled markers never establish SSO ownership."""

import hashlib
import hmac
import json
import sqlite3
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from uuid import UUID

from aiohttp import web

from ramjet.sso import VerifiedPrincipal
from ramjet.tasks.gptchat import private_auth, private_registry
from tests.test_sso_identity import SUBJECT, OTHER
from tests.namespace_test_support import OwnershipFixture, KEY_A, PREFIX


class NamespaceProvenanceTests(OwnershipFixture):
    """NamespaceProvenanceTests retain the exact fail-closed historical compatibility boundary."""

    async def test_forged_markers_and_legacy_names_never_grant_list_or_delete(self):
        """test_forged_markers excludes valid-looking old proofs without consulting their contents."""
        scope = (
            "ckv2-"
            + hashlib.sha256(
                b"ramjet-private-key-capability-v2\0" + KEY_A.encode()
            ).hexdigest()
        )
        proof = hmac.new(
            KEY_A.encode(),
            b"ramjet-private-namespace-provenance-v2\0" + scope.encode(),
            hashlib.sha256,
        ).hexdigest()
        payload = json.dumps({"version": 2, "scope": scope, "proof": proof}).encode()
        owners = (
            scope,
            hashlib.sha1(KEY_A.encode()).hexdigest(),
            KEY_A[:15],
            "forged-victim",
            "pns3-" + "a" * 32,
        )
        records = {f"{PREFIX}/{owner}/legacy.store" for owner in owners}
        self.storage.objects = records.copy()
        self.storage.markers[f"{PREFIX}/{scope}/__ownership_v2/marker.json"] = payload
        self.clear_effects()
        for owner in owners:
            for method in (self.get, self.delete):
                with self.subTest(owner=owner), self.assertRaises(web.HTTPForbidden):
                    await method(self.view(uid=owner))
                self.assertEqual(self.storage.operations, [])
                self.assertEqual(self.scheduled, [])
                self.assertEqual(self.events, [])
                self.assertEqual(self.storage.objects, records)
        self.assertIsNone(self.registry.lookup(SUBJECT, "upload"))

    async def test_model_constructor_or_data_password_is_not_a_private_principal(self):
        """test_model_constructor denies model-only objects despite forged UID and known data password."""
        model = self.model_user(uid="pns3-" + "a" * 32)
        for create in (False, True):
            with self.assertRaises(web.HTTPUnauthorized):
                self.env["require_private_namespace"](
                    self.storage, model, create=create, operation="upload"
                )
        self.assertEqual(self.storage.operations, [])
        self.assertIsNone(self.registry.lookup(SUBJECT, "upload"))

    async def test_fresh_initialization_is_registry_authority_without_marker_writes(
        self,
    ):
        """test_fresh_initialization allocates an independent empty scope and never serializes credentials."""
        self.storage.markers[f"{PREFIX}/forged-victim/__ownership_v2/marker.json"] = (
            b"untrusted-old-marker"
        )
        owner = await self.user(initialize=False)
        self.assertEqual(owner.uid, "")
        self.clear_effects()
        self.storage.put_object = lambda **kw: self.fail(
            "ownership must never write an S3 marker"
        )
        self.assertTrue(
            self.env["require_private_namespace"](
                self.storage, owner, create=True, operation="upload"
            )
        )
        self.assertTrue(owner.uid.startswith("pns3-"))
        self.assertEqual(self.registry.lookup(SUBJECT, "upload").namespace, owner.uid)
        self.assertEqual(self.storage.operations, [("list", f"{PREFIX}/{owner.uid}/")])
        self.assertEqual(len(self.storage.markers), 1)
        with sqlite3.connect(self.registry.path) as connection:
            names = connection.execute(
                "SELECT principal_id, namespace FROM namespace_owners"
            ).fetchall()
        self.assertEqual(names, [(SUBJECT, owner.uid)])
        self.assertNotIn(KEY_A, str(names))

    async def test_occupied_random_candidate_never_adopts_or_overwrites_records(self):
        """test_occupied_random_candidate proves prefix resemblance does not establish historical ownership."""
        candidate = UUID("123e4567-e89b-42d3-a456-426614174000")
        namespace = "pns3-" + candidate.hex
        legacy = f"{PREFIX}/{namespace}/legacy.store"
        self.storage.objects.add(legacy)
        owner = await self.user(initialize=False)
        with patch.object(private_registry, "uuid4", return_value=candidate):
            with self.assertRaises(web.HTTPConflict):
                self.env["require_private_namespace"](
                    self.storage, owner, create=True, operation="upload"
                )
        self.assertEqual(self.storage.objects, {legacy})
        self.assertEqual(owner.uid, "")
        self.assertIsNone(self.registry.lookup(SUBJECT, "upload"))

    async def test_storage_failure_cannot_commit_fresh_grants_or_fabricate_absence(
        self,
    ):
        """test_storage_failure preserves missing authority when the actual empty check fails."""
        owner = await self.user(initialize=False)

        def denied(**kwargs):
            """denied simulates an outage without touching external storage."""
            raise PermissionError("synthetic storage denied")

        self.storage.list_objects = denied
        with self.assertRaises(web.HTTPServiceUnavailable):
            self.env["require_private_namespace"](
                self.storage, owner, create=True, operation="upload"
            )
        self.assertIsNone(self.registry.lookup(SUBJECT, "upload"))
        self.assertEqual(owner.uid, "")
        self.assertEqual(self.storage.markers, {})

    async def test_expired_request_context_and_wrong_operation_cannot_reuse_grants(
        self,
    ):
        """test_expired_request_context rechecks expiry and exact operation before worker effects."""
        owner = await self.user()
        self.clear_effects()
        with self.assertRaises(web.HTTPUnauthorized):
            self.env["require_private_namespace"](
                self.storage, owner, operation="delete-files"
            )
        owner.principal = VerifiedPrincipal(owner.principal.subject, 0)
        with self.assertRaises(web.HTTPUnauthorized):
            self.env["require_private_namespace"](
                self.storage, owner, operation="upload"
            )
        self.assertEqual(self.storage.operations, [])

    async def test_registry_revocation_after_entry_stops_delete_worker(self):
        """test_registry_revocation rechecks an exact grant immediately before synthetic mutation."""
        owner = await self.user(operation="delete-files")
        record = f"{PREFIX}/{owner.uid}/private.store"
        self.storage.objects.add(record)
        with sqlite3.connect(self.registry.path) as connection:
            connection.execute(
                "DELETE FROM operation_grants WHERE principal_id = ? AND operation = ?",
                (SUBJECT, "delete-files"),
            )
        self.clear_effects()
        with self.assertRaises(web.HTTPForbidden):
            self.delete_files(NS(), owner, ["private"])
        self.assertEqual(self.storage.objects, {record})
        self.assertEqual(self.storage.operations, [])

    async def test_uid_mutation_cannot_rebind_an_existing_verified_grant(self):
        """test_uid_mutation refuses a modified request context rather than trusting its namespace field."""
        owner = await self.user()
        other = await self.user(subject=OTHER)
        owner.uid = other.uid
        self.clear_effects()
        with self.assertRaises(web.HTTPForbidden):
            self.env["require_private_namespace"](
                self.storage, owner, operation="upload"
            )
        self.assertEqual(self.storage.operations, [])

    async def test_sync_private_context_cannot_change_authorized_operation(self):
        """test_sync_private_context allows only the asynchronous entry's exact operation."""
        reader = self.view()
        await self.authorize(reader, "upload")
        with self.assertRaises(web.HTTPUnauthorized):
            private_auth.verified_private_user(reader.request, "delete-files")
        self.assertEqual(self.storage.operations, [])


if __name__ == "__main__":
    unittest.main()
