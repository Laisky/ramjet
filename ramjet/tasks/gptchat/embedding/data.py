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
from langchain.text_splitter import CharacterTextSplitter, MarkdownTextSplitter
from langchain.vectorstores import FAISS

from ramjet.settings import prd
from ramjet.utils import get_db
from ..base import logger


def load_all_stores() -> Dict[str, FAISS]:
    stores = {}
    for project_name in prd.OPENAI_EMBEDDING_QA:
        fname = os.path.join(prd.OPENAI_TMP_DIR, project_name)
        with open(fname+".store", "rb") as f:
            store = pickle.load(f)
        store.index = faiss.read_index(fname+".index")
        stores[project_name] = store

    return stores


def prepare_data():
    tasks = []
    for name, project in prd.OPENAI_EMBEDDING_QA.items():
        logger.info(f"download vector datasets for {name} ...")
        tasks.append(_download_index_data(project))

    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.wait(tasks))


async def _download_index_data(project: Dict):
    # download store
    url = project["store"]
    fname = url.rsplit("/")[-1]
    fpath = os.path.join(prd.OPENAI_TMP_DIR, fname)
    if not os.path.exists(fpath):
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                assert resp.status == 200, f"download vector store failed: {resp.status}"
                with open(fpath, "wb") as f:
                    f.write(await resp.read())

    # download index
    url = project["index"]
    fname = url.rsplit("/")[-1]
    fpath = os.path.join(prd.OPENAI_TMP_DIR, fname)
    if not os.path.exists(fpath):
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                assert resp.status == 200, f"download vector store failed: {resp.status}"
                with open(fpath, "wb") as f:
                    f.write(await resp.read())
