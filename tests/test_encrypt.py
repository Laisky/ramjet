"""Offline behavioral coverage for password and JWT helpers."""

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import jwt


def load_encrypt_helpers():
    """load_encrypt_helpers imports real helpers with a public test signing key."""
    settings = types.ModuleType("ramjet.settings")
    settings.SECRET_KEY = "public-test-default-signing-key-" * 3
    source = Path(__file__).resolve().parents[1] / "ramjet" / "utils" / "encrypt.py"
    spec = importlib.util.spec_from_file_location("ramjet_encrypt_under_test", source)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"ramjet.settings": settings}):
        spec.loader.exec_module(module)
    return module, settings.SECRET_KEY


class EncryptHelperTests(unittest.TestCase):
    def setUp(self):
        """setUp loads helpers without importing production settings or services."""
        self.helpers, self.default_key = load_encrypt_helpers()
        self.custom_key = "public-test-custom-signing-key-" * 3
        self.claims = {"sub": "user-1", "scope": ["read", "write"], "label": "hello"}

    def test_generate_token_returns_text(self):
        """test_generate_token_returns_text checks both configured and explicit keys."""
        for secret in (None, self.custom_key):
            with self.subTest(explicit_key=secret is not None):
                token = (
                    self.helpers.generate_token(self.claims)
                    if secret is None
                    else self.helpers.generate_token(self.claims, secret=secret)
                )
                self.assertIsInstance(token, str)
                self.assertEqual(jwt.get_unverified_header(token)["alg"], "HS512")
                key = self.default_key if secret is None else secret
                self.assertEqual(jwt.decode(token, key, algorithms=["HS512"]), self.claims)

    def test_validate_token_accepts_signed_claims(self):
        """test_validate_token_accepts_signed_claims verifies the signing contract."""
        for secret in (None, self.custom_key):
            with self.subTest(explicit_key=secret is not None):
                key = self.default_key if secret is None else secret
                token = jwt.encode(self.claims, key, algorithm="HS512")
                claims = (
                    self.helpers.validate_token(token)
                    if secret is None
                    else self.helpers.validate_token(token, secret=secret)
                )
                self.assertEqual(claims, self.claims)

    def test_validate_token_rejects_wrong_signing_key(self):
        """test_validate_token_rejects_wrong_signing_key preserves signature checks."""
        token = jwt.encode(self.claims, self.custom_key, algorithm="HS512")
        with self.assertRaises(jwt.InvalidSignatureError):
            self.helpers.validate_token(token, secret=self.default_key)

    def test_validate_token_rejects_expired_claims(self):
        """test_validate_token_rejects_expired_claims preserves expiration checks."""
        token = jwt.encode(dict(self.claims, exp=0), self.custom_key, algorithm="HS512")
        with self.assertRaises(jwt.ExpiredSignatureError):
            self.helpers.validate_token(token, secret=self.custom_key)

    def test_validate_token_rejects_other_algorithms(self):
        """test_validate_token_rejects_other_algorithms excludes HS256 and unsigned JWTs."""
        for algorithm, key in (("HS256", self.custom_key), ("none", "")):
            with self.subTest(algorithm=algorithm):
                token = jwt.encode(self.claims, key, algorithm=algorithm)
                with self.assertRaises(jwt.InvalidAlgorithmError):
                    self.helpers.validate_token(token, secret=self.custom_key)

    def test_password_hash_round_trip(self):
        """test_password_hash_round_trip checks unchanged valid and invalid passwords."""
        hashed = self.helpers.generate_passwd(b"correct-password")
        self.assertTrue(self.helpers.validate_passwd(b"correct-password", hashed))
        self.assertFalse(self.helpers.validate_passwd(b"wrong-password", hashed))
