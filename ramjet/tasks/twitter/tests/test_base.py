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
