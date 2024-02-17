from typing import Callable, List, NamedTuple, Set, Tuple

from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores.faiss import FAISS


class Index(NamedTuple):
    """embeddings index"""

    store: FAISS
    scaned_files: Set[str]


class UserChain(NamedTuple):
    """user chatbot"""

    user_index: Index
    datasets: List[str]
    chain: Callable[[ChatOpenAI, str], Tuple[str, List[str]]]
    search: Callable[[str], Tuple[str, List[str]]]
