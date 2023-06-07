import os
import pickle
import tempfile
import requests
import aiohttp
import asyncio
from typing import Dict

import faiss
from kipp import options as opt
from langchain.embeddings import OpenAIEmbeddings
# from langchain.text_splitter import CharacterTextSplitter, MarkdownTextSplitter
from langchain.vectorstores import FAISS

from ramjet.settings import prd
# from ramjet.utils import get_db
from ..base import logger


def load_all_stores() -> Dict[str, FAISS]:
    stores = {}
    for project_name in prd.OPENAI_EMBEDDING_QA:
        fname = os.path.join(prd.OPENAI_INDEX_DIR, project_name)
        with open(fname+".store", "rb") as f:
            store = pickle.load(f)
        store.index = faiss.read_index(fname+".index")
        stores[project_name] = store

    return stores


def prepare_data():
    tasks = []
    for name, project in prd.OPENAI_EMBEDDING_QA.items():
        logger.info(f"download vector datasets to {prd.OPENAI_INDEX_DIR} for {name} ...")
        tasks.append(_download_index_data(project))

    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.wait(tasks))


async def _download_index_data(project: Dict):
    # download store
    url = project["store"]
    fname = url.rsplit("/")[-1]
    fpath = os.path.join(prd.OPENAI_INDEX_DIR, fname)
    if os.path.exists(fpath):
        logger.info(f"skip download {fname}")
    else:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                assert resp.status == 200, f"download vector store failed: {resp.status}"
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
                assert resp.status == 200, f"download vector store failed: {resp.status}"
                with open(fpath, "wb") as f:
                    while True:
                        chunk = await resp.content.read(4096)
                        if not chunk:
                            break
                        f.write(chunk)
