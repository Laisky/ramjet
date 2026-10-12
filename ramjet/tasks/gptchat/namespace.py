"""Validate storage components; only the trusted registry grants private ownership."""

import re

_OBJECT_NAME = re.compile(r"[a-zA-Z0-9_-]+")
_PUBLIC_OWNER = re.compile(r"[a-zA-Z0-9_-]{1,128}")


def require_object_name(name: str) -> str:
    """require_object_name returns a single safe dataset/chatbot path component."""
    if not isinstance(name, str) or not _OBJECT_NAME.fullmatch(name):
        raise ValueError("Invalid dataset or chatbot name")
    return name


def require_dataset_names(names: list[str]) -> list[str]:
    """require_dataset_names validates every component before storage or scheduling."""
    if not isinstance(names, list) or not names:
        raise ValueError("datasets must be a non-empty array of names")
    for name in names:
        require_object_name(name)
    return names


def public_share_owner(owner: str) -> str:
    """public_share_owner validates only a published plaintext-share selector.

    Legacy public links remain usable. This value never authorizes private data,
    current pointers, mutation, or the reader's quota/model credentials.
    """
    if not isinstance(owner, str) or not _PUBLIC_OWNER.fullmatch(owner):
        raise ValueError("Invalid public share owner")
    return owner
