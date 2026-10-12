"""Published public links retain model-only reader credentials without private SSO grants."""

import hashlib
import importlib.util
import json
from pathlib import Path
import threading
from types import SimpleNamespace as NS
import unittest

from aiohttp import web

from tests.test_sso_identity import SUBJECT

from tests.namespace_test_support import (
    OwnershipFixture,
    ROOT,
    KEY_A,
    KEY_B,
    PREFIX,
    public_module,
    definition,
)


class NamespacePublicTests(OwnershipFixture):
    """NamespacePublicTests exercises actual source with inert storage and model dependencies."""

    def test_public_link_selects_only_public_owner_and_retains_reader(self):
        """test_public_link honors published query UID while preserving reader credentials/admission."""
        import copy

        cache = {}
        restores = []
        bindings = []
        readers = []

        def public_restore(*, s3cli, user, chatbot_name):
            """public_restore captures only the selected share owner and returns inert plaintext."""
            restores.append((user.uid, chatbot_name, user.apikey))
            chain = NS(chain=lambda llm, q: ("published-answer", []))
            cache[user.uid, chatbot_name] = chain
            return chain

        def bind(chain, user):
            """bind records the reader's credential capability rather than public ownership."""
            bindings.append((user.uid, user.apikey))
            return chain

        def admission(*, user):
            """admission records the reader's scope independently of the public selector."""
            readers.append(user.uid)
            return NS(release=lambda: None)

        self.env.update(
            copy=copy,
            user_shared_chain=cache,
            user_shared_chain_mu=threading.RLock(),
            restore_user_chain=public_restore,
            bind_user_chain=bind,
            build_llm_for_user=lambda user: object(),
            uid_ratelimiter=admission,
        )
        namespace = public_module("ramjet/tasks/gptchat/namespace.py")
        self.env.update(
            require_object_name=namespace.require_object_name,
            public_share_owner=namespace.public_share_owner,
        )
        share = definition(
            "ramjet/tasks/gptchat/router.py",
            "share_chatbot_query",
            self.env,
            "EmbeddingContext",
        )
        first = self.view(KEY_B, session=False, uid="forged-private-victim")
        first.request.query = {
            "uid": "legacy-published-owner",
            "chatbot_name": "public_bot",
            "q": "synthetic",
        }
        self.assertEqual(json.loads(share(first).text)["text"], "published-answer")
        self.assertEqual(json.loads(share(first).text)["text"], "published-answer")
        self.assertEqual(restores, [("legacy-published-owner", "public_bot", KEY_B)])
        reader = self.model_user(KEY_B).uid
        self.assertEqual(bindings, [(reader, KEY_B), (reader, KEY_B)])
        self.assertEqual(readers, [reader, reader])
        self.assertEqual(self.transport_calls, [])
        self.assertIsNone(self.registry.lookup(SUBJECT, "upload"))
        for owner, bot in (("ab", "c"), ("a", "bc")):
            first.request.query = {"uid": owner, "chatbot_name": bot, "q": "synthetic"}
            share(first)
        self.assertIn(("ab", "c"), cache)
        self.assertIn(("a", "bc"), cache)
        self.assertIsNot(cache["ab", "c"], cache["a", "bc"])
        count = len(restores)
        for uid, name in (
            ("../private-victim", "bot"),
            ("owner", "../bot"),
            ("owner", "bot\n"),
        ):
            first.request.query = {"uid": uid, "chatbot_name": name, "q": "synthetic"}
            with self.assertRaises(ValueError):
                share(first)
        self.assertEqual(len(restores), count)

    def test_current_pointer_names_reject_before_private_or_public_store_download(self):
        """test_current_pointer_names confines storage even when persisted metadata is malformed."""
        namespace = public_module("ramjet/tasks/gptchat/namespace.py")
        downloads = []
        gets = []
        current = ["../victim"]

        def get_object(**kwargs):
            """get_object supplies a synthetic malicious current pointer."""
            gets.append(kwargs["object_name"])
            return NS(
                data=current[0].encode(), close=lambda: None, release_conn=lambda: None
            )

        storage = NS(
            get_object=get_object, fget_object=lambda **kwargs: downloads.append(kwargs)
        )
        self.env.update(
            require_object_name=namespace.require_object_name,
            quote=__import__("urllib.parse", fromlist=["quote"]).quote,
        )
        actual_user = self.model_user()
        self.env["prd"] = self.env["settings"]
        download = definition(
            "ramjet/tasks/gptchat/llm/embeddings.py", "download_chatbot_index", self.env
        )
        for password in ("synthetic-password", ""):
            for name in ("../victim", "valid\n"):
                with self.assertRaises(ValueError):
                    download(
                        "/tmp/unused-synthetic",
                        storage,
                        actual_user,
                        chatbot_name=name,
                        password=password,
                    )
        self.assertEqual(gets, [])
        for pointer in ("../victim", "valid\n", ""):
            current[0] = pointer
            with self.assertRaises(ValueError):
                download(
                    "/tmp/unused-synthetic",
                    storage,
                    NS(uid="synthetic-owner"),
                    password="synthetic-password",
                )
        self.assertEqual(downloads, [])
        self.assertTrue(
            all(("/synthetic-owner/chatbot-v2/__CURRENT" in value for value in gets))
        )

    def test_actual_public_archive_reader_never_selects_private_pointer_or_store(self):
        """test_actual_public_archive_reader pins published object paths using the real downloader."""
        import pickle
        import tempfile

        namespace = public_module("ramjet/tasks/gptchat/namespace.py")
        paths = []
        client_bindings = []

        def fetch(**kwargs):
            """fetch records actual selected public archive paths and supplies inert bytes."""
            paths.append(kwargs["object_name"])
            path = Path(kwargs["file_path"])
            path.write_bytes(
                pickle.dumps(["published-dataset"])
                if path.suffix == ".pkl"
                else b"synthetic-store"
            )

        def plaintext(**kwargs):
            """plaintext records only reader model credentials with no provider execution."""
            client_bindings.append((kwargs["api_key"], kwargs["api_base"]))
            return NS()

        def forbidden(**kwargs):
            """forbidden fails on private current-pointer reads or private decryption."""
            self.fail("Public share must never choose a private storage operation")

        self.env.update(
            require_object_name=namespace.require_object_name,
            quote=__import__("urllib.parse", fromlist=["quote"]).quote,
            pickle=pickle,
            load_plaintext_store=plaintext,
            load_encrypt_store=forbidden,
        )
        self.env["prd"] = self.env["settings"]
        download = definition(
            "ramjet/tasks/gptchat/llm/embeddings.py", "download_chatbot_index", self.env
        )
        storage = NS(fget_object=fetch, get_object=forbidden)
        for owner in ("legacy-published-owner", "ckv2-" + "a" * 64, "pns3-" + "b" * 32):
            reader = NS(
                uid=owner, apikey=KEY_B, api_base="http://reader-provider.fixture/v1"
            )
            with tempfile.TemporaryDirectory() as directory:
                name, _, datasets = download(
                    directory, storage, reader, chatbot_name="public_bot"
                )
            self.assertEqual(name, "public_bot")
            self.assertEqual(datasets, ["published-dataset"])
        self.assertEqual(
            paths,
            [
                f"{PREFIX}/legacy-published-owner/chatbot-share-v2/public_bot.store",
                f"{PREFIX}/legacy-published-owner/chatbot-share-v2/public_bot.pkl",
                f"{PREFIX}/ckv2-" + "a" * 64 + "/chatbot-share-v2/public_bot.store",
                f"{PREFIX}/ckv2-" + "a" * 64 + "/chatbot-share-v2/public_bot.pkl",
                f"{PREFIX}/pns3-" + "b" * 32 + "/chatbot-share-v2/public_bot.store",
                f"{PREFIX}/pns3-" + "b" * 32 + "/chatbot-share-v2/public_bot.pkl",
            ],
        )
        self.assertEqual(
            client_bindings, [(KEY_B, "http://reader-provider.fixture/v1")] * 3
        )


if __name__ == "__main__":
    unittest.main()
