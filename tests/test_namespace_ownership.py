"""Offline acceptance of independent SSO ownership and registry grants."""

import base64
import hashlib
import json
import threading
from types import SimpleNamespace as NS
import unittest

from aiohttp import web
import jwt
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ramjet.tasks.gptchat import private_auth
from tests.test_sso_identity import SUBJECT, OTHER, session_token
from tests.namespace_test_support import (
    OwnershipFixture,
    KEY_A,
    KEY_B,
    PREFIX,
    public_module,
    definition,
)


class OwnershipBoundaryTests(OwnershipFixture):
    """OwnershipBoundaryTests exercise real private auth before synthetic handler effects."""

    async def test_same_provider_key_cannot_merge_distinct_sso_owners(self):
        """test_same_provider_key isolates two verified principals sharing identical model credentials."""
        first = await self.user(KEY_A, subject=SUBJECT)
        second = await self.user(KEY_A, subject=OTHER)
        self.assertNotEqual(first.uid, second.uid)
        self.assertTrue(first.uid.startswith("pns3-"))
        self.storage.objects = {f"{PREFIX}/{first.uid}/private.store"}
        self.clear_effects()
        response = await self.get(self.view(KEY_A, subject=OTHER, uid=first.uid))
        self.assertEqual(json.loads(response.text)["datasets"], [])
        self.assertEqual(self.storage.operations, [("list", f"{PREFIX}/{second.uid}/")])
        own = await self.get(self.view(KEY_A, subject=SUBJECT))
        self.assertEqual(
            json.loads(own.text)["datasets"], [{"name": "private", "status": "done"}]
        )

    async def test_provider_rotation_preserves_verified_principal_namespace(self):
        """test_provider_rotation never transfers ownership through a model credential change."""
        first = await self.user(KEY_A)
        rotated = await self.user(
            KEY_B, uid="forged-other", base="http://other-provider.fixture"
        )
        self.assertEqual(first.uid, rotated.uid)
        self.assertEqual(rotated.apikey, KEY_B)
        self.assertEqual(rotated.api_base, "http://other-provider.fixture/v1")
        self.assertNotEqual(first.uid, hashlib.sha1(KEY_A.encode()).hexdigest())
        self.assertNotEqual(first.uid, KEY_A[:15])
        self.assertNotIn(SUBJECT, first.uid)

    async def test_forged_uid_delete_cannot_remove_other_principal_or_legacy_objects(
        self,
    ):
        """test_forged_uid_delete allows only the independently granted namespace."""
        first = await self.user()
        second = await self.user(KEY_A, subject=OTHER)
        victim = {
            f"{PREFIX}/{first.uid}/private.{ext}" for ext in ("store", "index", "pdf")
        }
        own = {
            f"{PREFIX}/{second.uid}/private.{ext}" for ext in ("store", "index", "pdf")
        }
        legacy = {
            f"{PREFIX}/forged-victim/private.store",
            f"{PREFIX}/{KEY_A[:15]}/private.store",
        }
        self.storage.objects = victim | own | legacy
        self.clear_effects()
        response = await self.delete(self.view(KEY_A, subject=OTHER, uid=first.uid))
        self.assertEqual(response.status, 200)
        self.assertEqual(self.storage.objects, victim | legacy)
        self.assertTrue(
            all(
                name.startswith(f"{PREFIX}/{second.uid}/")
                for _, name in self.storage.operations
            )
        )
        await self.delete(self.view(KEY_B))
        self.assertEqual(self.storage.objects, legacy)

    async def test_wire_uid_free_and_backend_cannot_grant_scope_or_capacity(self):
        """test_wire_uid_free holds ownership fixed across untrusted routing and entitlement assertions."""
        owner = await self.user()
        for uid, free in (
            ("public", "false"),
            ("../victim", "true"),
            (KEY_A[:15], "junk"),
            (owner.uid, "TrUe"),
        ):
            user = await self.user(
                KEY_B, uid=uid, free=free, base="http://internal-model.fixture"
            )
            self.assertEqual(user.uid, owner.uid)
            self.assertFalse(user.is_paid)
            self.assertEqual(user.n_concurrent, 3)
            self.assertEqual(user.apikey, KEY_B)

    async def test_missing_session_and_ungranted_other_subject_deny_before_effects(
        self,
    ):
        """test_missing_session refuses model-only credentials and valid but ungranted SSO identities."""
        owner = await self.user()
        self.clear_effects()
        for view, expected in (
            (self.view(KEY_A, session=False, uid=owner.uid), web.HTTPUnauthorized),
            (self.view(KEY_A, subject=OTHER, uid=owner.uid), web.HTTPForbidden),
        ):
            with self.assertRaises(expected):
                await self.delete(view)
            self.assertEqual(self.storage.operations, [])
            self.assertEqual(self.scheduled, [])
            self.assertEqual(self.events, [])

    async def test_legacy_expired_forged_revoked_and_mismatched_sessions_precede_effects(
        self,
    ):
        """test_legacy_expired proves actual private decorators fail before JSON, storage or scheduling."""
        await self.user()
        claims = jwt.decode(session_token(), options={"verify_signature": False})
        legacy = jwt.encode(claims, "public-offline-legacy", algorithm="HS256")
        forged = jwt.encode(
            claims, Ed25519PrivateKey.from_private_bytes(b"z" * 32), algorithm="EdDSA"
        )
        tokens = (legacy, session_token(exp=0), forged, session_token(uid=OTHER))
        for token in tokens:
            self.clear_effects()
            with self.assertRaises(web.HTTPUnauthorized):
                await self.delete(self.view(token=token))
            self.assertEqual(self.events, [])
            self.assertEqual(self.storage.operations, [])
            self.assertEqual(self.scheduled, [])
        self.issuer.revoked = True
        self.clear_effects()
        with self.assertRaises(web.HTTPUnauthorized):
            await self.delete(self.view())
        self.assertEqual(self.storage.operations, [])
        self.assertEqual(self.events, [])
        self.issuer.revoked = False
        self.response_override = (
            200,
            json.dumps({"data": {"WhoAmI": {"id": OTHER}}}).encode(),
        )
        with self.assertRaises(web.HTTPUnauthorized):
            await self.delete(self.view())
        self.assertEqual(self.scheduled, [])

    async def test_unavailable_issuer_and_registry_deny_before_token_or_data_effects(
        self,
    ):
        """test_unavailable distinguishes failed authority from a permission grant."""
        await self.user()
        self.clear_effects()
        self.response_override = (503, b"synthetic outage")
        with self.assertRaises(web.HTTPServiceUnavailable):
            await self.delete(self.view())
        self.assertEqual(self.events, [])
        self.assertEqual(self.storage.operations, [])
        self.response_override = None
        self.registry.path.unlink()
        self.clear_effects()
        with self.assertRaises(web.HTTPServiceUnavailable):
            await self.delete(self.view())
        self.assertEqual(self.transport_calls, [])
        self.assertEqual(self.scheduled, [])

    async def test_metadata_ownership_does_not_require_or_send_provider_key(self):
        """test_metadata_ownership permits the exact SSO grant without unrelated model credentials."""
        owner = await self.user()
        self.storage.objects.add(f"{PREFIX}/{owner.uid}/private.store")
        response = await self.get(self.view(""))
        self.assertEqual(
            json.loads(response.text)["datasets"],
            [{"name": "private", "status": "done"}],
        )
        self.assertEqual(self.permissions, [])

    async def test_outer_private_context_denies_before_worker_queue(self):
        """test_outer_private_context prevents a directly invoked sync worker from doing issuer I/O."""
        await self.user()
        context = definition(
            "ramjet/tasks/gptchat/router.py", "get", self.env, "EmbeddingContext"
        )
        for op in ("/chat", "/search", "/list"):
            view = self.view(session=False)
            view.request.match_info = {"op": op}
            view.chatbot_query = view.chatbot_search = view.list_chatbots = (
                lambda: self.fail("unauthorized worker queued")
            )
            self.clear_effects()
            with self.assertRaises(web.HTTPUnauthorized):
                await context(view)
            self.assertEqual(self.scheduled, [])
        query = definition(
            "ramjet/tasks/gptchat/router.py",
            "chatbot_query",
            self.env,
            "EmbeddingContext",
        )
        with self.assertRaises(web.HTTPUnauthorized):
            query(self.view())

    async def test_delete_names_cannot_escape_or_queue_work(self):
        """test_delete_names retains complete path-component rejection after valid identity."""
        await self.user()
        self.clear_effects()
        for datasets in (
            ["../victim/private"],
            ["private/../../victim"],
            ["private\n"],
            ["private", ".."],
            "private",
            [1],
            [""],
        ):
            with self.subTest(datasets=datasets):
                response = await self.delete(self.view(datasets=datasets))
                self.assertEqual(response.status, 400)
                self.assertEqual(self.scheduled, [])
                self.assertEqual(self.storage.operations, [])

    async def test_warm_cache_requires_registry_scope_and_independent_password(self):
        """test_warm_cache excludes another granted owner even with its model key and data password."""
        module = public_module("ramjet/tasks/gptchat/llm/private_cache.py")
        cache = module.PasswordProtectedCache(
            {}, threading.RLock(), lambda pwd: hashlib.sha256(pwd.encode()).digest()
        )
        owner = await self.user()
        other = await self.user(subject=OTHER)
        password = "synthetic-password"
        chain = NS(datasets=["private"], chain=lambda llm, q: ("private-answer", []))
        cache.save(module.private_chain_key(owner.uid), chain, password)
        restores, bindings = [], []

        def cold(storage, user, password):
            """cold records only the current registry namespace before denying absent synthetic data."""
            restores.append(user.uid)
            raise KeyError("no new object for this independently granted scope")

        def bind(chain, user):
            """bind records only reader model credentials without an SDK transport."""
            bindings.append((user.uid, user.apikey))
            return chain

        self.env.update(
            private_user_chain_cache=cache,
            restore_user_chain=cold,
            bind_user_chain=bind,
            build_llm_for_user=lambda user: object(),
        )
        self.env["get_private_user_chain"] = definition(
            "ramjet/tasks/gptchat/llm/embeddings.py", "get_private_user_chain", self.env
        )
        query = definition(
            "ramjet/tasks/gptchat/router.py",
            "chatbot_query",
            self.env,
            "EmbeddingContext",
        )
        reader = self.view(KEY_B)
        reader.request.query = {"q": "synthetic"}
        reader.request.headers["X-PDFCHAT-PASSWORD"] = password
        await self.authorize(reader, "read-private")
        self.assertEqual(json.loads(query(reader).text)["text"], "private-answer")
        reader.request.headers["X-PDFCHAT-PASSWORD"] = "wrong-data-password"
        with self.assertRaises(ValueError):
            query(reader)
        attacker = self.view(KEY_A, subject=OTHER, uid=owner.uid)
        attacker.request.query = {"q": "synthetic"}
        attacker.request.headers["X-PDFCHAT-PASSWORD"] = password
        await self.authorize(attacker, "read-private")
        with self.assertRaises(KeyError):
            query(attacker)
        self.assertEqual(restores, [other.uid])
        self.assertEqual(bindings, [(owner.uid, KEY_B)])

    async def test_proven_legacy_string_and_v2_cache_grants_never_enter_sso_v3(self):
        """test_proven_legacy isolates prior password-proven entries even with identical namespace text."""
        module = public_module("ramjet/tasks/gptchat/llm/private_cache.py")
        cache = module.PasswordProtectedCache(
            {}, threading.RLock(), lambda pwd: hashlib.sha256(pwd.encode()).digest()
        )
        user = await self.user(operation="read-private")
        password = "synthetic-password"
        legacy = NS(datasets=["legacy-private"])
        for key in (user.uid, ("private-credential-v2", user.uid)):
            cache.save(key, legacy, password)
        self.assertEqual(
            module.private_chain_key(user.uid), ("sso-private-v3", user.uid)
        )

        def cold(storage, reader, pwd):
            """cold rejects absent new objects rather than adopting prior decrypted entries."""
            raise KeyError("no current private chain")

        self.env.update(private_user_chain_cache=cache, restore_user_chain=cold)
        get_chain = definition(
            "ramjet/tasks/gptchat/llm/embeddings.py", "get_private_user_chain", self.env
        )
        with self.assertRaises(KeyError):
            get_chain(self.storage, user, password)
        current = NS(datasets=["new-private"])
        cache.save(module.private_chain_key(user.uid), current, password)
        self.assertIs(get_chain(self.storage, user, password), current)
        self.assertIs(cache.get(user.uid, password), legacy)

    async def test_legacy_processing_metadata_never_enters_current_scope(self):
        """test_legacy_processing separates both string and old tuple metadata from current SSO state."""
        user = await self.user()
        legacy = {"legacy-private-filename"}
        state = self.env["user_processing_files"]
        state[user.uid] = state[("private-processing-v2", user.uid)] = legacy
        response = await self.get(self.view())
        self.assertEqual(json.loads(response.text)["datasets"], [])
        state[("sso-private-processing-v3", user.uid)] = {"current"}
        response = await self.get(self.view())
        self.assertEqual(
            json.loads(response.text)["datasets"],
            [{"name": "current", "status": "processing", "progress": 75}],
        )
        self.assertEqual(legacy, {"legacy-private-filename"})

    async def test_current_processing_worker_preserves_old_state_and_cleans_success_failure(
        self,
    ):
        """test_current_processing exercises actual worker bookkeeping and exact upload authority."""
        user = await self.user()
        legacy = {"legacy-private-filename"}
        self.env["user_processing_files"][user.uid] = legacy
        self.env["timer"] = lambda fn: fn
        process = definition(
            "ramjet/tasks/gptchat/router.py", "process_file", self.env, "UploadedFiles"
        )
        state_key = ("sso-private-processing-v3", user.uid)

        def worker(reader, data):
            """worker checks current state while leaving legacy filenames untouched."""
            self.assertEqual(self.env["user_processing_files"][state_key], {"current"})
            if data.get("fail"):
                raise ValueError("synthetic worker failure")
            return ["current"]

        view = NS(_process_file=worker)
        self.assertEqual(process(view, user, {"file_key": "current"}), ["current"])
        with self.assertRaises(ValueError):
            process(view, user, {"file_key": "current", "fail": True})
        self.assertEqual(self.env["user_processing_files"][state_key], set())
        self.assertEqual(legacy, {"legacy-private-filename"})

    async def test_public_admission_stays_bounded_despite_wire_paid_and_legacy_capacity(
        self,
    ):
        """test_public_admission denies inherited legacy paid capacity and model-key churn."""
        user = await self.user()
        other = await self.user(KEY_B, subject=OTHER, free="false")
        legacy = threading.Semaphore(100)
        self.buckets["public"] = legacy
        permits = [self.acquire(user) for _ in range(3)]
        try:
            with self.assertRaises(web.HTTPTooManyRequests):
                self.acquire(other)
            self.assertIs(self.buckets["public"], legacy)
            self.assertIsNot(permits[0], legacy)
        finally:
            for permit in permits:
                permit.release()
        self.acquire(other).release()

    async def test_private_mutation_names_reject_before_storage_or_provider_work(self):
        """test_private_mutation_names retains boundary validation for every private path builder."""
        work = []
        self.env.update(
            download_chatbot_index=lambda **kwargs: work.append(kwargs),
            restore_user_chain=lambda *args, **kwargs: work.append(kwargs),
            new_store=lambda **kwargs: work.append(kwargs),
        )
        for name, operation in (
            ("build_chatbot", "build"),
            ("active_chatbot", "activate"),
            ("share_chatbot", "publish-share"),
            ("load_datasets", "build"),
        ):
            user = await self.user(operation=operation)
            method = definition(
                "ramjet/tasks/gptchat/router.py", name, self.env, "EmbeddingContext"
            )
            for invalid in ("../victim", "valid\n", "", 1):
                with self.assertRaises((ValueError, AssertionError)):
                    if name == "load_datasets":
                        method(
                            NS(),
                            user=user,
                            datasets=[invalid],
                            password="synthetic-password",
                        )
                    else:
                        method(
                            NS(),
                            user=user,
                            data={
                                "chatbot_name": invalid,
                                "datasets": ["valid"],
                                "data_key": "synthetic-password",
                            },
                        )
                self.assertEqual(work, [])

    def test_chunk_worker_cannot_grant_paid_budget_from_wire_values(self):
        """test_chunk_worker preserves conservative model-only capacity without private SSO authority."""
        calls = []
        self.env.update(
            base64=base64,
            DEFAULT_MAX_CHUNKS_FOR_FREE=600,
            DEFAULT_MAX_CHUNKS_FOR_PAID=10000,
            classificate_query_type=lambda **kwargs: "search",
            _chunk_search=lambda **kwargs: calls.append(kwargs)
            or web.json_response({"ok": True}),
        )
        worker = definition(
            "ramjet/tasks/gptchat/router.py", "_embedding_chunk_worker", self.env
        )
        user = self.model_user(free="false")
        user.info = lambda: "synthetic-user"
        for requested, expected in ((0, 600), (1, 1), (600, 600), (10000, 600)):
            data = {
                "content": base64.b64encode(b"synthetic").decode(),
                "query": "synthetic",
                "ext": ".txt",
                "max_chunks": requested,
            }
            worker(NS(), data, user)
            self.assertEqual(calls[-1]["max_chunks"], expected)
            self.assertEqual(calls[-1]["apikey"], KEY_A)
        count = len(calls)
        for requested in (-1, "10000", True):
            with self.assertRaises(ValueError):
                worker(NS(), dict(data, max_chunks=requested), user)
        self.assertEqual(len(calls), count)


if __name__ == "__main__":
    unittest.main()
