"""Synthetic settings and deterministic dependencies for the offline test suite."""

from dataclasses import dataclass
import hashlib
import logging
import socket
import sys
import types

from aiohttp.test_utils import TestServer
from langchain_core.embeddings import Embeddings
import langchain_openai


@dataclass
class UserPermission:
    """UserPermission supplies the public request-handler fields for test users."""

    is_paid: bool
    uid: str
    n_concurrent: int
    chat_model: str
    apikey: str
    api_base: str


class OfflineEmbeddings(Embeddings):
    """OfflineEmbeddings provides deterministic vectors without an API client."""

    def __init__(self, **kwargs):
        """__init__ accepts production constructor options without creating clients."""

    def embed_query(self, text):
        """embed_query returns a stable eight-dimensional vector for text."""
        return [value / 255 for value in hashlib.sha256(text.encode()).digest()[:8]]

    def embed_documents(self, texts):
        """embed_documents returns a deterministic vector for every input text."""
        return [self.embed_query(text) for text in texts]


# Install the synthetic production-module replacement before application imports.
prd = types.ModuleType("ramjet.settings.prd")
for name, value in {
    "UserPermission": UserPermission,
    "SECRET_KEY": "public-offline-signing-key-" * 4,
    "OPENAI_TOKEN": "public-offline-api-key",
    "OPENAI_API": "http://127.0.0.1:9",
    "OPENAI_EMBEDDING_QA": {},
    "OPENAI_INDEX_DIR": "/tmp/ramjet-offline-unused-index",
    "S3_MINIO_ADDR": "127.0.0.1:9",
    "S3_KEY": "public-offline-key",
    "S3_SECRET": "public-offline-secret",
    "S3_SERVER": "http://127.0.0.1:9",
    "OPENAI_S3_CHUNK_CACHE_BUCKET": "offline-test-chunks",
    "OPENAI_S3_EMBEDDINGS_PREFIX": "offline-test-embeddings",
    "OPENAI_S3_CHUNK_CACHE_IMAGES": "offline-test-images",
    "OPENAI_EMBEDDING_FILE_SIZE_LIMIT": 10 * 1024 * 1024,
    "OPENAI_EMBEDDING_REF_URL_PREFIX": "http://127.0.0.1:9/",
}.items():
    setattr(prd, name, value)
sys.modules[prd.__name__] = prd

# Keep imports from creating the production background logging dispatcher.
log_module = types.ModuleType("ramjet.utils.log")
log_module.logger = logging.getLogger("ramjet.offline-tests")
sys.modules[log_module.__name__] = log_module
langchain_openai.OpenAIEmbeddings = OfflineEmbeddings

# Allow only sockets belonging to aiohttp's disposable test servers.
_test_ports = set()
_original_start_server = TestServer.start_server
_original_connect = socket.socket.connect
_original_connect_ex = socket.socket.connect_ex
_original_getaddrinfo = socket.getaddrinfo


async def start_test_server(self, *args, **kwargs):
    """start_test_server records the disposable server port after it starts."""
    result = await _original_start_server(self, *args, **kwargs)
    _test_ports.add(self.port)
    return result


def check_test_address(address):
    """check_test_address rejects every connection except registered local servers."""
    if (
        not isinstance(address, tuple)
        or address[0] not in ("127.0.0.1", "::1", "localhost")
        or address[1] not in _test_ports
    ):
        raise AssertionError("Offline tests prohibit external or service connections")


def connect_test_socket(self, address):
    """connect_test_socket checks the address before opening a test connection."""
    check_test_address(address)
    return _original_connect(self, address)


def connect_ex_test_socket(self, address):
    """connect_ex_test_socket checks the address before opening a test connection."""
    check_test_address(address)
    return _original_connect_ex(self, address)


def resolve_test_address(host, port, *args, **kwargs):
    """resolve_test_address prevents DNS requests for external hostnames."""
    if host not in (None, "127.0.0.1", "::1", "localhost"):
        raise AssertionError("Offline tests prohibit external DNS resolution")
    return _original_getaddrinfo(host, port, *args, **kwargs)


TestServer.start_server = start_test_server
socket.socket.connect = connect_test_socket
socket.socket.connect_ex = connect_ex_test_socket
socket.getaddrinfo = resolve_test_address


def pytest_sessionfinish(session, exitstatus):
    """pytest_sessionfinish stops any executor used by the disposable handlers."""
    engines = sys.modules.get("ramjet.engines")
    if engines:
        for name in ("thread_executor", "process_executor"):
            executor = getattr(engines, name).executor
            if executor:
                executor.shutdown(wait=True)
