import tempfile
import tarfile
from typing import Callable, List, NamedTuple, Set, Tuple

from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

from ramjet.settings import prd


class Index(NamedTuple):
    """embeddings index"""

    store: FAISS
    scaned_files: Set[str]

    def serialize(self) -> bytes:
        """serialize index to bytes"""
        with tempfile.TemporaryDirectory() as tempdir:
            self.store.save_local(tempdir)

            # compress dir
            with tempfile.TemporaryFile() as tempf:
                with tarfile.open(fileobj=tempf, mode="w:gz") as tar:
                    tar.add(tempdir, arcname="index")
                tempf.seek(0)
                return tempf.read()

    @classmethod
    def deserialize(cls, data: bytes, api_key: str = prd.OPENAI_TOKEN) -> "Index":
        """deserialize index from bytes"""
        with tempfile.TemporaryDirectory() as tempdir:
            with tempfile.TemporaryFile() as tempf:
                tempf.write(data)
                tempf.seek(0)
                with tarfile.open(fileobj=tempf, mode="r:gz") as tar:
                    tar.extractall(tempdir)

            store = FAISS.load_local(
                tempdir, OpenAIEmbeddings(api_key=SecretStr(api_key)), "index"
            )
            return cls(store, set())


class UserChain(NamedTuple):
    """user chatbot"""

    user_index: Index
    datasets: List[str]
    chain: Callable[[ChatOpenAI, str], Tuple[str, List[str]]]
    search: Callable[[str], Tuple[str, List[str]]]
