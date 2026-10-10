"""Password proof for an already decrypted private chatbot cache entry."""

from dataclasses import dataclass
from typing import Any, Callable, ContextManager, MutableMapping

from Crypto.Cipher import AES


@dataclass(frozen=True, repr=False)
class _PasswordProof:
    """_PasswordProof binds an authenticated marker to one exact cached chain."""

    chain: Any
    nonce: bytes
    tag: bytes
    ciphertext: bytes


class PasswordProtectedCache:
    """PasswordProtectedCache retains password proof without passwords or AES keys.

    The caller installs a chain only after its private data was successfully
    decrypted or built from successfully decrypted datasets. Proof remains local
    to this process and never enters serialized indexes or object storage.
    """

    def __init__(
        self,
        chains: MutableMapping[str, Any],
        lock: ContextManager,
        derive_key: Callable[[str], bytes],
    ):
        """__init__ shares existing chain storage/lock and its unchanged AES key derivation."""
        self._chains = chains
        self._lock = lock
        self._derive_key = derive_key
        self._proofs: dict[str, _PasswordProof] = {}

    def save(self, uid: str, chain: Any, password: str) -> None:
        """save binds the checked password to a chain and installs both atomically."""
        if not password:
            raise ValueError("X-PDFCHAT-PASSWORD is required")
        cipher = AES.new(self._derive_key(password), AES.MODE_EAX)
        ciphertext, tag = cipher.encrypt_and_digest(b"private-chain-password-proof-v1")
        proof = _PasswordProof(chain, cipher.nonce, tag, ciphertext)
        with self._lock:
            self._chains[uid] = chain
            self._proofs[uid] = proof

    def get(self, uid: str, password: str) -> Any | None:
        """get authenticates the exact cached chain or requests a checked cold restore.

        Wrong passwords raise the same AES authentication error as cold decrypt.
        Missing or replaced proof never authorizes an existing plaintext entry.
        The returned local snapshot cannot be swapped by a sibling cache writer.
        """
        if not password:
            raise ValueError("X-PDFCHAT-PASSWORD is required")
        with self._lock:
            chain = self._chains.get(uid)
            proof = self._proofs.get(uid)
            if chain is None or proof is None or proof.chain is not chain:
                return None
        cipher = AES.new(self._derive_key(password), AES.MODE_EAX, nonce=proof.nonce)
        cipher.decrypt_and_verify(proof.ciphertext, proof.tag)
        return chain
