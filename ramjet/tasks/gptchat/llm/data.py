import asyncio
import os
import pickle
from typing import Coroutine, Dict, List

import aiohttp
import faiss
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.embeddings import Embeddings

from ramjet.settings import prd

from ..base import logger


class UnboundEmbeddings(Embeddings):
    """Keep loaded vector data unusable until explicit request credentials are bound."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Reject model operations without an explicit request credential."""
        raise ValueError("A valid client API key is required")

    def embed_query(self, text: str) -> List[float]:
        """Reject model operations without an explicit request credential."""
        raise ValueError("A valid client API key is required")


def load_all_prebuild_qa() -> Dict[str, FAISS]:
    """load all prebuild qa embeddings stores"""
    stores = {}
    # Startup only loads vector data; request handlers bind their own credentials.
    for project_name in prd.OPENAI_EMBEDDING_QA:
        try:
            fname = os.path.join(prd.OPENAI_INDEX_DIR, project_name)
            with open(fname + ".store", "rb") as f:
                store = pickle.load(f)

            store.embedding_function = UnboundEmbeddings()
            store.index = faiss.read_index(fname + ".index")
            stores[project_name] = store
        except Exception as err:
            logger.warn(f"cannot load embedding index for {project_name=}, {err=}")

    return stores


def prepare_data():
    tasks: List[Coroutine] = []
    for name, project in prd.OPENAI_EMBEDDING_QA.items():
        logger.info(
            f"download vector datasets to {prd.OPENAI_INDEX_DIR} for {name} ..."
        )
        tasks.append(_download_index_data(project))

    if not tasks:
        return

    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.wait(tasks))


async def _download_index_data(project: Dict[str, str]):
    # download store
    url: str = project["store"]
    fname = url.rsplit("/")[-1]
    fpath = os.path.join(prd.OPENAI_INDEX_DIR, fname)
    if os.path.exists(fpath):
        logger.info(f"skip download {fname}")
    else:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                assert (
                    resp.status == 200
                ), f"download vector store failed: {resp.status}"
                with open(fpath, "wb") as f:
                    while True:
                        chunk = await resp.content.read(4096)
                        if not chunk:
                            break

                        f.write(chunk)

    # download index
    url = project["index"]
    fname = url.rsplit("/")[-1]
    fpath = os.path.join(prd.OPENAI_INDEX_DIR, fname)
    if os.path.exists(fpath):
        logger.info(f"skip download {fname}")
    else:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                assert (
                    resp.status == 200
                ), f"download vector store failed: {resp.status}"
                with open(fpath, "wb") as f:
                    while True:
                        chunk = await resp.content.read(4096)
                        if not chunk:
                            break
                        f.write(chunk)
