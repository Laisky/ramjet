import os
import pickle
import tempfile
import requests

import faiss
from kipp import options as opt
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, MarkdownTextSplitter
from langchain.vectorstores import FAISS

from ramjet.settings import prd
from ramjet.utils import get_db
from ..base import logger

tmpdirname = tempfile.mkdtemp()
FILE_VECTOR_STORE = os.path.join(tmpdirname, "docs.store")
FILE_VECTOR_INDEX = os.path.join(tmpdirname, "docs.index")


def load_store() -> FAISS:
    with open(FILE_VECTOR_STORE, "rb") as f:
        store = pickle.load(f)
    store.index = faiss.read_index(FILE_VECTOR_INDEX)
    return store


def prepare_data():
    logger.info("download vector datasets...")
    resp = requests.get(prd.OPENAI_EMBEDDING_QA[prd.OPENAI_EMBEDDING_QA_ENABLE]["index"])
    assert resp.status_code == 200, f"download vector index failed: {resp.status_code}"
    with open(FILE_VECTOR_INDEX, "wb") as f:
        f.write(resp.content)

    resp = requests.get(prd.OPENAI_EMBEDDING_QA[prd.OPENAI_EMBEDDING_QA_ENABLE]["store"])
    assert resp.status_code == 200, f"download vector store failed: {resp.status_code}"
    with open(FILE_VECTOR_STORE, "wb") as f:
        f.write(resp.content)

    # os.makedirs(os.path.dirname(FILE_VECTOR_STORE), exist_ok=True)
    # store = _load_docus_from_db()
    # _save_vector_stores(store)
    # logger.info("basebit vector data ok")


# def _load_docus_from_db():
#     logger.debug("load data from db...")
#     db = get_db()
#     db_docus = db["basebit"]["docus"]
#     text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")

#     docs = []
#     metadatas = []
#     for doc in db_docus.find():
#         splits = text_splitter.split(doc["text"])
#         docs.extend(splits)
#         metadatas.extend([doc["url"]] * len(splits))

#     store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)
#     logger.info("succeed load data from db")
#     return store

# def _save_vector_stores(store: FAISS):
#     # save to files
#     faiss.write_index(store.index, FILE_VECTOR_INDEX)
#     store.index = None
#     with open(os.path.join(dir, FILE_VECTOR_STORE), "wb") as f:
#         pickle.dump(store, f)
