"""Offline synthetic regressions for credential-scoped private and public data."""

import hashlib
import importlib.util
import json
from pathlib import Path
import threading
from types import SimpleNamespace as NS
import unittest

from aiohttp import web

from tests.namespace_test_support import (
    OwnershipFixture,
    ROOT,
    KEY_A,
    KEY_B,
    PREFIX,
    public_module,
    definition,
)


class NamespaceProvenanceTests(OwnershipFixture):
    """NamespaceProvenanceTests exercises actual source with inert storage and model dependencies."""

    def test_occupied_unmarked_new_looking_scope_is_never_adopted(self):
        """test_occupied_unmarked proves a ckv2 name alone attributes no historical records."""
        user = self.user()
        legacy = f"{PREFIX}/{user.uid}/legacy.store"
        self.storage.objects.add(legacy)
        for create in (False, True):
            with self.subTest(create=create):
                with self.assertRaises(ValueError):
                    self.env["require_private_namespace"](
                        self.storage, user, create=create
                    )
                self.assertEqual(self.storage.objects, {legacy})
                self.assertEqual(self.storage.markers, {})

    def test_new_marker_authenticates_exact_complete_key_and_persists_no_secret(self):
        """test_new_marker admits only new creation and rejects copied/invalid owner claims."""
        first = self.user(KEY_A)
        second = self.user(KEY_B)
        self.assertFalse(self.env["require_private_namespace"](self.storage, first))
        self.assertTrue(
            self.env["require_private_namespace"](self.storage, first, create=True)
        )
        self.assertTrue(self.env["require_private_namespace"](self.storage, first))
        marker_a = f"{PREFIX}/{first.uid}/__ownership_v2/marker.json"
        marker_b = f"{PREFIX}/{second.uid}/__ownership_v2/marker.json"
        payload = self.storage.markers[marker_a]
        self.assertNotIn(KEY_A.encode(), payload)
        self.storage.markers[marker_b] = payload
        with self.assertRaises(ValueError):
            self.env["require_private_namespace"](self.storage, second, create=True)
        for malformed in (b"{}", b"not-json", b"x" * 1025):
            self.storage.markers[marker_a] = malformed
            with self.assertRaises(ValueError):
                self.env["require_private_namespace"](self.storage, first)
        self.storage.markers[marker_a] = payload
        with self.assertRaises(ValueError):
            self.env["require_private_namespace"](
                self.storage, NS(uid=second.uid, apikey=KEY_A)
            )
        self.assertEqual(self.storage.objects, set())

    async def test_unmarked_namespace_rejects_list_delete_before_mutation(self):
        """test_unmarked_namespace protects historical objects through actual decorated routes."""
        user = self.user()
        record = f"{PREFIX}/{user.uid}/private.store"
        self.storage.objects.add(record)
        for method in (self.get, self.delete):
            response = await method(self.view(KEY_A))
            self.assertEqual(response.status, 400)
        self.assertEqual(self.storage.objects, {record})
        self.assertEqual(self.storage.markers, {})
        self.assertEqual(self.scheduled, [])

    def test_unknown_marker_storage_failure_never_grants_access_or_creates_proof(self):
        """test_unknown_marker_storage_failure distinguishes missing data from outage/denial."""

        def denied(**kwargs):
            """denied simulates a storage authorization error without network work."""
            error = PermissionError("synthetic denial")
            error.code = "AccessDenied"
            raise error

        self.storage.get_object = denied
        with self.assertRaises(PermissionError):
            self.env["require_private_namespace"](
                self.storage, self.user(), create=True
            )
        self.assertEqual(self.storage.markers, {})
        self.assertEqual(self.storage.operations, [])

    def test_marker_write_or_readback_failure_never_initializes_access(self):
        """test_marker_write_or_readback_failure denies proof-less state after any write failure."""
        original_put = self.storage.put_object

        def failed_put(**kwargs):
            """failed_put simulates a bounded synthetic storage write failure."""
            raise OSError("synthetic write failure")

        for put in (
            failed_put,
            lambda **kwargs: None,
            lambda **kwargs: self.storage.markers.update(
                {kwargs["object_name"]: b"invalid"}
            ),
        ):
            self.storage.markers.clear()
            self.storage.put_object = put
            with self.assertRaises((OSError, ValueError)):
                self.env["require_private_namespace"](
                    self.storage, self.user(), create=True
                )
        self.storage.put_object = original_put
        self.storage.markers.clear()
        self.assertFalse(
            self.env["require_private_namespace"](self.storage, self.user())
        )

    def test_same_key_initializer_race_requires_valid_persisted_marker(self):
        """test_same_key_initializer_race accepts only the same authenticated namespace proof."""
        user = self.user()
        self.env["require_private_namespace"](self.storage, user, create=True)
        name, payload = next(iter(self.storage.markers.items()))
        self.storage.markers.clear()

        def concurrent_list(**kwargs):
            """concurrent_list models a completed same-key initializer during the empty check."""
            self.storage.markers[name] = payload
            return [NS(object_name=name)]

        self.storage.list_objects = concurrent_list
        self.storage.put_object = lambda **kwargs: self.fail(
            "race must reread valid existing proof"
        )
        self.assertTrue(
            self.env["require_private_namespace"](self.storage, user, create=True)
        )


if __name__ == "__main__":
    unittest.main()
