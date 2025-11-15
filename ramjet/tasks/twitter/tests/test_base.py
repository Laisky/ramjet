from __future__ import annotations

import os
import pickle
import tarfile
import tempfile
from collections import namedtuple
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from ramjet.tasks.gptchat.llm.base import deserialize
from ramjet.tasks.twitter.base import twitter_surl_regex


class TestTwitterBase(TestCase):
    def test_twitter_surl_regex(self):
        Case = namedtuple("Case", ["input", "output"])
        cases = [
            Case("123", []),
            Case("https://t.co/123 ", ["https://t.co/123"]),
            Case("abchttps://t.co/123 ", ["https://t.co/123"]),
            Case("abchttps://t.co/123-+", ["https://t.co/123"]),
        ]
        for c in cases:
            self.assertEqual(c.output, twitter_surl_regex.findall(c.input))


class TestGPTChatBase(TestCase):
    def test_deserialize(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            index_dir = Path(tmpdir, "index")
            index_dir.mkdir()
            scan_file = index_dir / "scaned_files"
            with open(scan_file, "wb") as f:
                pickle.dump({"foo.pdf"}, f)

            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as temp_archive:
                archive_path = temp_archive.name
                with tarfile.open(fileobj=temp_archive, mode="w:gz") as tar:
                    tar.add(index_dir, arcname="index")
                temp_archive.flush()
                temp_archive.seek(0)
                archive_bytes = temp_archive.read()

        os.remove(archive_path)

        with patch("ramjet.tasks.gptchat.llm.base.FAISS.load_local") as mock_load_local:
            mock_load_local.return_value = "store"
            result = deserialize(archive_bytes, api_key="dummy")

        self.assertEqual(result.store, "store")
        self.assertEqual(result.scaned_files, {"foo.pdf"})
        mock_load_local.assert_called_once()

        # Temporary file should be cleaned up
        self.assertFalse(os.path.exists(archive_path))
