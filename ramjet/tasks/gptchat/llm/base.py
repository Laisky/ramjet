import os
import tempfile
import tarfile
from typing import Callable, List, NamedTuple, Set, Tuple
import pickle

from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ramjet.settings import prd
from ramjet.utils import logger


logger = logger.getChild("tasks.gptchat.llm.base")


class Index(NamedTuple):
    """embeddings index"""

    store: FAISS
    scaned_files: Set[str]

    def serialize(self) -> bytes:
        """serialize index to bytes"""
        with tempfile.TemporaryDirectory() as tempdir:
            self.store.save_local(tempdir)
            with open(os.path.join(tempdir, "scaned_files"), "wb") as f:
                f.write(pickle.dumps(self.scaned_files))

            # compress dir
            with tempfile.TemporaryFile() as tempf:
                with tarfile.open(fileobj=tempf, mode="w:gz") as tar:
                    tar.add(tempdir, arcname="index")
                tempf.seek(0)
                return tempf.read()

    @classmethod
    def deserialize(cls, data: bytes, api_key: str = prd.OPENAI_TOKEN) -> "Index":
        """deserialize index from bytes"""
        assert data, "data should not be empty"
        with tempfile.TemporaryDirectory() as tempdir:
            with tempfile.TemporaryFile() as tempf:
                tempf.write(data)
                tempf.seek(0)
                with tarfile.open(fileobj=tempf, mode="r:gz") as tar:
                    tar.extractall(tempdir)

            tempdir = os.path.join(tempdir, "index")
            store = FAISS.load_local(
                folder_path=tempdir, embeddings=OpenAIEmbeddings(api_key=api_key),
                allow_dangerous_deserialization=True,
            )
            with open(os.path.join(tempdir, "scaned_files"), "rb") as f:
                scaned_files = pickle.load(f)

        return cls(store=store, scaned_files=scaned_files)


class UserChain(NamedTuple):
    """user chatbot"""

    user_index: Index
    datasets: List[str]
    chain: Callable[[ChatOpenAI, str], Tuple[str, List[str]]]
    search: Callable[[str], Tuple[str, List[str]]]
