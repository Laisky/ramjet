"""Opt-in characterization of the unresolved weak credential capability contract.

This is known-risk evidence, not a secure acceptance assertion. It uses only
synthetic keys, local in-memory storage and actual request selection/list/delete.
"""

import json
import os
import unittest

from tests.namespace_test_support import OwnershipFixture, PREFIX, public_module


@unittest.skipUnless(
    os.environ.get("RAMJET_WEAK_CAPABILITY_CHARACTERIZATION") == "1",
    "known weak-capability characterization is opt-in",
)
class WeakCredentialCapabilityTests(OwnershipFixture):
    """WeakCredentialCapabilityTests preserves the required ownership-design decision."""

    async def test_public_scope_allows_guessing_a_weak_private_capability(self):
        """test_public_scope shows key recovery permits metadata and deletion without data password."""
        weak = "sk-weak-synthetic"
        owner = self.user(weak)
        self.env["require_private_namespace"](self.storage, owner, create=True)
        record = f"{PREFIX}/{owner.uid}/private.store"
        self.storage.objects.add(record)
        namespace = public_module("ramjet/tasks/gptchat/namespace.py")
        candidates = ["sk-wrong-synthetic", weak, "sk-other-synthetic"]
        matches = [
            key for key in candidates if namespace.credential_scope(key) == owner.uid
        ]
        self.assertEqual(matches, [weak])
        guessed = matches[0]
        response = await self.get(self.view(guessed))
        self.assertEqual(
            json.loads(response.text)["datasets"],
            [{"name": "private", "status": "done"}],
        )
        await self.delete(self.view(guessed))
        self.assertNotIn(record, self.storage.objects)
        self.assertEqual(
            self.view(guessed).request.headers.get("X-PDFCHAT-PASSWORD"), None
        )


if __name__ == "__main__":
    unittest.main()
