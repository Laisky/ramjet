from collections import namedtuple
from unittest import TestCase

from ramjet.tasks.twitter.base import twitter_surl_regex


class TestBase(TestCase):
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
import os
import tempfile
import tarfile
from unittest import TestCase
from unittest.mock import patch

from ramjet.tasks.gptchat.llm.base import deserialize


class TestBase(TestCase):
    def test_deserialize(self):
        # Create a temporary tar file
        with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tempf:
            # Write some data to the temporary file
            tempf.write(b"test data")
            tempf.seek(0)

            # Mock the FAISS.load_local method
            with patch("ramjet.tasks.gptchat.llm.base.FAISS.load_local") as mock_load_local:
                # Call the deserialize method with the temporary file
                result = deserialize(tempf.read())

                # Assert that the FAISS.load_local method was called with the correct arguments
                mock_load_local.assert_called_with(
                    mock_load_local.return_value,
                    "index"
                )

                # Assert that the result is not None
                self.assertIsNotNone(result)

        # Assert that the temporary file is deleted
        self.assertFalse(os.path.exists(tempf.name))
