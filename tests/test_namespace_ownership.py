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


class OwnershipBoundaryTests(OwnershipFixture):
    """OwnershipBoundaryTests exercises actual source with inert storage and model dependencies."""

    async def test_distinct_complete_keys_cannot_list_same_prefix_or_legacy(self):
        """test_distinct_complete_keys exercises decorated list against actual selected prefixes."""
        self.assertEqual(KEY_A[:15], KEY_B[:15])
        first = self.user(KEY_A)
        second = self.user(KEY_B)
        self.assertNotEqual(first.uid, second.uid)
        self.env["require_private_namespace"](self.storage, first, create=True)
        self.storage.operations.clear()
        self.storage.objects = {
            f"{PREFIX}/{first.uid}/private.store",
            f"{PREFIX}/forged-victim/legacy.store",
            f"{PREFIX}/{KEY_A[:15]}/old.store",
        }
        first_response = await self.get(self.view("Bearer " + KEY_A))
        second_response = await self.get(self.view("Bearer " + KEY_B, uid=first.uid))
        self.assertEqual(
            json.loads(first_response.text)["datasets"],
            [{"name": "private", "status": "done"}],
        )
        self.assertEqual(json.loads(second_response.text)["datasets"], [])
        self.assertEqual(
            self.storage.operations,
            [("list", f"{PREFIX}/{first.uid}/"), ("list", f"{PREFIX}/{second.uid}/")],
        )

    async def test_forged_uid_delete_cannot_remove_other_key_or_legacy_objects(self):
        """test_forged_uid_delete retains victim records while allowing own-scope deletion."""
        first = self.user(KEY_A)
        second = self.user(KEY_B)
        self.env["require_private_namespace"](self.storage, first, create=True)
        self.storage.operations.clear()
        objects = {
            f"{PREFIX}/{first.uid}/private.{ext}" for ext in ("store", "index", "pdf")
        }
        legacy = {
            f"{PREFIX}/forged-victim/private.store",
            f"{PREFIX}/{KEY_A[:15]}/private.store",
        }
        self.storage.objects = objects | legacy
        response = await self.delete(self.view(KEY_B, uid=first.uid))
        self.assertEqual(response.status, 200)
        self.assertEqual(self.storage.objects, objects | legacy)
        self.assertTrue(
            all((f"/{second.uid}/" in name for _, name in self.storage.operations))
        )
        await self.delete(self.view(KEY_A))
        self.assertEqual(self.storage.objects, legacy)

    def test_uid_free_headers_and_backend_do_not_grant_identity_or_capacity(self):
        """test_uid_free_headers rejects identity/paid grants from arbitrary wire assertions."""
        original = self.user()
        self.assertTrue(original.uid.startswith("ckv2-"))
        self.assertEqual(len(original.uid), 69)
        for uid in ("public", "../victim", KEY_A[:15], "", original.uid):
            for free in ("true", "false", "", "TrUe", "junk"):
                with self.subTest(uid=uid, free=free):
                    user = self.user(
                        "Bearer " + KEY_A,
                        uid=uid,
                        free=free,
                        base="http://other-provider.fixture/v1",
                    )
                    self.assertEqual(user.uid, original.uid)
                    self.assertFalse(user.is_paid)
                    self.assertEqual(user.n_concurrent, 3)
                    self.assertEqual(user.apikey, KEY_A)
                    self.assertEqual(user.api_base, "http://other-provider.fixture/v1")

    def test_rotation_and_known_prefix_do_not_authorize_old_scope(self):
        """test_rotation makes no inferred relationship between complete credentials."""
        first = self.user(KEY_A)
        rotated = self.user(KEY_B, uid=first.uid)
        self.assertNotEqual(first.uid, rotated.uid)
        self.assertEqual(self.user("Bearer " + KEY_A, uid=rotated.uid).uid, first.uid)
        self.assertNotEqual(first.uid, hashlib.sha1(KEY_A.encode()).hexdigest())
        self.assertNotEqual(first.uid, KEY_A[:15])

    def test_missing_key_denied_before_permission_or_storage(self):
        """test_missing_key proves the local constructor and storage are not indiscriminately permissive."""
        with self.assertRaises(web.HTTPUnauthorized):
            self.env["get_user_by_appkey"](
                NS(headers={"X-Laisky-User-Id": "victim"}, query={})
            )
        self.assertEqual(self.permissions, [])
        self.assertEqual(self.storage.operations, [])

    def test_public_admission_is_shared_even_without_preseed(self):
        """test_public_admission prevents UID/key churn from resetting conservative process capacity."""
        permits = [self.acquire(self.user(KEY_A)) for _ in range(3)]
        try:
            with self.assertRaises(web.HTTPTooManyRequests):
                self.acquire(self.user(KEY_B, free="false"))
            self.assertEqual(set(self.buckets), {("private-admission-v2", "public")})
        finally:
            for permit in permits:
                permit.release()
        self.acquire(self.user(KEY_B)).release()

    async def test_delete_names_cannot_escape_or_queue_work(self):
        """test_delete_names denies traversal, separators and non-list input before worker submission."""
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
                response = await self.delete(self.view(KEY_A, datasets=datasets))
                self.assertEqual(response.status, 400)
                self.assertEqual(self.scheduled, [])
                self.assertEqual(self.storage.operations, [])

    def test_warm_private_cache_cannot_be_selected_by_forged_uid_or_password(self):
        """test_warm_private_cache requires the reader's scope even with a known victim password."""
        cache_spec = importlib.util.spec_from_file_location(
            "ownership_password_cache",
            ROOT / "ramjet/tasks/gptchat/llm/private_cache.py",
        )
        cache_module = importlib.util.module_from_spec(cache_spec)
        __import__("sys").modules[cache_spec.name] = cache_module
        cache_spec.loader.exec_module(cache_module)
        chains = {}
        cache = cache_module.PasswordProtectedCache(
            chains, threading.RLock(), lambda pwd: hashlib.sha256(pwd.encode()).digest()
        )
        first = self.user(KEY_A)
        password = "synthetic-password"
        chain = NS(datasets=["private"], chain=lambda llm, q: ("private-answer", []))
        self.env["require_private_namespace"](self.storage, first, create=True)
        cache.save(self.env["private_chain_key"](first.uid), chain, password)
        restores = []
        bindings = []

        def cold_restore(storage, user, password):
            """cold_restore records the caller's new scope and denies absent synthetic data."""
            restores.append(user.uid)
            raise KeyError("no synthetic object for this scope")

        def bind(chain, user):
            """bind records reader credentials without creating an embedding client."""
            bindings.append((user.uid, user.apikey))
            return chain

        self.env.update(
            private_user_chain_cache=cache,
            restore_user_chain=cold_restore,
            bind_user_chain=bind,
            build_llm_for_user=lambda user: object(),
        )
        get_chain = definition(
            "ramjet/tasks/gptchat/llm/embeddings.py", "get_private_user_chain", self.env
        )
        self.env["get_private_user_chain"] = get_chain
        query = definition(
            "ramjet/tasks/gptchat/router.py",
            "chatbot_query",
            self.env,
            "EmbeddingContext",
        )
        victim = self.view(KEY_A)
        victim.request.query = {"q": "synthetic"}
        victim.request.headers["X-PDFCHAT-PASSWORD"] = password
        self.assertEqual(json.loads(query(victim).text)["text"], "private-answer")
        attacker = self.view(KEY_B, uid=first.uid)
        attacker.request.query = {"q": "synthetic"}
        attacker.request.headers["X-PDFCHAT-PASSWORD"] = password
        with self.assertRaises(web.HTTPNotFound):
            query(attacker)
        self.assertEqual(bindings, [(first.uid, KEY_A)])
        self.assertEqual(restores, [])
        self.assertEqual(set(chains), {self.env["private_chain_key"](first.uid)})

    def test_proven_legacy_string_cache_cannot_enter_new_private_scope(self):
        """test_proven_legacy_string_cache excludes old grants even with valid AES proof."""
        import sys

        spec = importlib.util.spec_from_file_location(
            "ownership_legacy_cache", ROOT / "ramjet/tasks/gptchat/llm/private_cache.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        cache = module.PasswordProtectedCache(
            {}, threading.RLock(), lambda pwd: hashlib.sha256(pwd.encode()).digest()
        )
        user = self.user()
        password = "synthetic-password"
        legacy = NS(datasets=["legacy-private"])
        cache.save(user.uid, legacy, password)
        self.assertIs(cache.get(user.uid, password), legacy)
        self.env["require_private_namespace"](self.storage, user, create=True)
        restores = []

        def cold_restore(storage, reader, pwd):
            """cold_restore refuses absent new objects and records the attempted scope."""
            restores.append(reader.uid)
            raise KeyError("new namespace contains no private chain")

        key = getattr(
            module, "private_chain_key", lambda uid: ("private-credential-v2", uid)
        )
        self.env.update(
            private_user_chain_cache=cache,
            restore_user_chain=cold_restore,
            private_chain_key=key,
        )
        get_chain = definition(
            "ramjet/tasks/gptchat/llm/embeddings.py", "get_private_user_chain", self.env
        )
        with self.assertRaises(KeyError):
            get_chain(self.storage, user, password)
        self.assertEqual(restores, [user.uid])
        current = NS(datasets=["new-private"])
        cache.save(key(user.uid), current, password)
        self.assertIs(get_chain(self.storage, user, password), current)
        self.assertIs(cache.get(user.uid, password), legacy)

    async def test_legacy_processing_names_do_not_enter_initialized_scope(self):
        """test_legacy_processing_names separates old UID strings from current worker state."""
        user = self.user()
        self.env["user_processing_files"][user.uid] = {"legacy-private-filename"}
        self.env["require_private_namespace"](self.storage, user, create=True)
        response = await self.get(self.view(KEY_A))
        self.assertEqual(json.loads(response.text)["datasets"], [])
        self.env["user_processing_files"][("private-processing-v2", user.uid)] = {
            "current"
        }
        response = await self.get(self.view(KEY_A))
        self.assertEqual(
            json.loads(response.text)["datasets"],
            [{"name": "current", "status": "processing", "progress": 75}],
        )
        self.assertEqual(
            self.env["user_processing_files"][user.uid], {"legacy-private-filename"}
        )

    def test_current_processing_worker_uses_and_cleans_only_typed_state(self):
        """test_current_processing_worker preserves legacy state on success and failure."""
        user = self.user()
        legacy = {"legacy-private-filename"}
        self.env["user_processing_files"][user.uid] = legacy
        self.env.update(
            timer=lambda fn: fn,
            require_object_name=public_module(
                "ramjet/tasks/gptchat/namespace.py"
            ).require_object_name,
        )
        process = definition(
            "ramjet/tasks/gptchat/router.py", "process_file", self.env, "UploadedFiles"
        )
        state_key = ("private-processing-v2", user.uid)

        def worker(reader, data):
            """worker checks current pending state while preserving old filenames."""
            self.assertEqual(self.env["user_processing_files"][state_key], {"current"})
            self.assertIs(self.env["user_processing_files"][user.uid], legacy)
            if data.get("fail"):
                raise ValueError("synthetic worker failure")
            return ["current"]

        view = NS(_process_file=worker)
        self.assertEqual(process(view, user, {"file_key": "current"}), ["current"])
        self.assertEqual(self.env["user_processing_files"][state_key], set())
        with self.assertRaises(ValueError):
            process(view, user, {"file_key": "current", "fail": True})
        self.assertEqual(self.env["user_processing_files"][state_key], set())
        self.assertEqual(legacy, {"legacy-private-filename"})

    def test_legacy_public_semaphore_cannot_supply_new_admission_capacity(self):
        """test_legacy_public_semaphore retains old state without adopting its paid capacity."""
        legacy = threading.Semaphore(100)
        self.buckets["public"] = legacy
        permits = [self.acquire(self.user()) for _ in range(3)]
        try:
            with self.assertRaises(web.HTTPTooManyRequests):
                self.acquire(self.user(KEY_B))
            self.assertIs(self.buckets["public"], legacy)
            self.assertIsNot(permits[0], legacy)
        finally:
            for permit in permits:
                permit.release()
        self.acquire(self.user(KEY_B)).release()

    def test_private_mutation_names_reject_before_storage_or_provider_work(self):
        """test_private_mutation_names checks every selected private path builder."""
        namespace = public_module("ramjet/tasks/gptchat/namespace.py")
        work = []
        self.env.update(
            require_object_name=namespace.require_object_name,
            require_dataset_names=namespace.require_dataset_names,
            download_chatbot_index=lambda **kwargs: work.append(kwargs),
            restore_user_chain=lambda *args, **kwargs: work.append(kwargs),
            new_store=lambda **kwargs: work.append(kwargs),
        )
        user = self.user()
        for name in (
            "build_chatbot",
            "active_chatbot",
            "share_chatbot",
            "load_datasets",
        ):
            method = definition(
                "ramjet/tasks/gptchat/router.py", name, self.env, "EmbeddingContext"
            )
            for invalid in ("../victim", "valid\n", "", 1):
                with self.subTest(method=name, name=invalid):
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
        """test_chunk_worker bounds unverified capacity before classification/provider work."""
        import base64

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
        user = self.user(free="false")
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
