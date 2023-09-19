import asyncio
import os
import pickle
from typing import Coroutine, Dict, List

import aiohttp
import faiss
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from ramjet.settings import prd

from ..base import logger


def load_all_prebuild_qa() -> Dict[str, FAISS]:
    """load all prebuild qa embeddings stores
    """
    stores = {}
    # if os.environ.get("OPENAI_API_TYPE") == "azure":
    #     azure_embeddings_deploymentid = prd.OPENAI_AZURE_DEPLOYMENTS[
    #         "embeddings"
    #     ].deployment_id
    #     # azure_gpt_deploymentid = prd.OPENAI_AZURE_DEPLOYMENTS["chat"].deployment_id

    #     embedding_model = OpenAIEmbeddings(
    #         client=None,
    #         model="text-embedding-ada-002",
    #         deployment=azure_embeddings_deploymentid,
    #     )
    # else:

    # warning: according to compatible with legacy langchain FAISS store,
    # replace embedding_funcion by my own apikey
    embedding_model = OpenAIEmbeddings(
        client=None,
        openai_api_key=prd.OPENAI_TOKEN,
        model="text-embedding-ada-002",
    )

    for project_name in prd.OPENAI_EMBEDDING_QA:
        try:
            fname = os.path.join(prd.OPENAI_INDEX_DIR, project_name)
            with open(fname + ".store", "rb") as f:
                store = pickle.load(f)

            store.embedding_function = embedding_model.embed_query
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
