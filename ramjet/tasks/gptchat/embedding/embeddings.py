import codecs
import glob
import hashlib
import os
import pickle
import re
import textwrap
from collections import namedtuple
from sys import path
from typing import List

import faiss
import openai
from Crypto.Cipher import AES
from kipp.utils import setup_logger
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, MarkdownTextSplitter
from langchain.vectorstores import FAISS
from pymongo import MongoClient

from ramjet.settings import prd

Index = namedtuple("index", ["store", "scaned_files"])

logger = setup_logger("security")


def pretty_print(text: str) -> str:
    text = text.strip()
    return textwrap.fill(text, width=60, subsequent_indent="    ")


# =============================
# 定义文件路径
# =============================

index_dirpath = prd.OPENAI_INDEX_DIR

# ==============================================================
# prepare pdf documents docs.index & docs.store
#
# https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/pdf.html#retain-elements
#
# 通用的函数定义
# ==============================================================

from urllib.parse import quote

from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, MarkdownTextSplitter
# from langchain.document_loaders import UnstructuredPDFLoader
from langchain.vectorstores import FAISS

text_splitter = CharacterTextSplitter(chunk_size=500, separator="\n")
markdown_splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=50)

N_BACTCH_FILES = 5


def is_file_scaned(index: Index, fpath):
    return os.path.split(fpath)[1] in index.scaned_files


def embedding_pdf(fpath: str, metadata_name: str) -> Index:
    """embedding pdf file

    Args:
        fpath (str): file path
        fkey (str): file key

    Returns:
        Index: index
    """
    logger.info(f"call embedding_pdf {fpath=}, {metadata_name=}")
    index = new_store()
    docs = []
    metadatas = []
    loader = PyPDFLoader(fpath)
    for page, data in enumerate(loader.load_and_split()):
        splits = text_splitter.split_text(data.page_content)
        docs.extend(splits)
        logger.debug(f"embedding {fpath} page {page+1} with {len(splits)} chunks")
        for ichunk, _ in enumerate(splits):
            # furl = prd.OPENAI_EMBEDDING_REF_URL_PREFIX.format(uid, password) + quote(fkey, safe="")
            metadatas.append({"source": f"{metadata_name}#page={page+1}"})

    logger.info(f"succeed embedding {fpath} with {len(docs)} chunks")
    index.store.add_texts(docs, metadatas=metadatas)
    return index


def embedding_markdowns(index: Index, fpaths, url, replace_by_url):
    i = 0
    docs = []
    metadatas = []
    for fpath in fpaths:
        fname = os.path.split(fpath)[1]
        if is_file_scaned(index, fpath):
            continue

        with codecs.open(fpath, "rb", "utf8") as fp:
            docus = markdown_splitter.create_documents([fp.read()])
            for ichunk, docu in enumerate(docus):
                docs.append(docu.page_content)
                title = quote(docu.page_content.strip().split("\n", maxsplit=1)[0])
                if url:
                    fnameurl = quote(fpath.removeprefix(replace_by_url), safe="")
                    furl = url + fnameurl
                    metadatas.append({"source": f"{furl}#{title}"})
                else:
                    metadatas.append({"source": f"{fname}#{title}"})

        index.scaned_files.add(fname)
        print(f"scaned {fpath}")
        i += 1
        if i > N_BACTCH_FILES:
            break

    if i != 0:
        index.store.add_texts(docs, metadatas=metadatas)

    return i


def new_store() -> Index:
    if os.environ.get("OPENAI_API_TYPE") == "azure":
        azure_embeddings_deploymentid = prd.OPENAI_AZURE_DEPLOYMENTS[
            "embeddings"
        ].deployment_id
        azure_gpt_deploymentid = prd.OPENAI_AZURE_DEPLOYMENTS["chat"].deployment_id

        embedding_model = OpenAIEmbeddings(
            client=None,
            model="text-embedding-ada-002",
            deployment=azure_embeddings_deploymentid,
        )
    else:
        embedding_model = OpenAIEmbeddings(
            client=None,
            model="text-embedding-ada-002",
        )

    store = FAISS.from_texts(
        ["world"], embedding_model, metadatas=[{"source": "hello"}]
    )
    return Index(
        store=store,
        scaned_files=set([]),
    )


def derive_key(password):
    key = hashlib.pbkdf2_hmac(
        "sha256",
        bytes(password, "utf-8"),
        b"123456",
        100000,  # iteration count
        dklen=32,  # length of the derived key
    )
    return key


def save_encrypt_store(index: Index, dirpath, name, password) -> List[str]:
    """save encrypted store

    Args:
        index (Index): index
        dirpath (str): dirpath
        name (str): name of file, without ext
        password (str): password

    Returns:
        List[str]: saved filepaths
    """
    key = derive_key(password)
    store_index = index.store.index

    # do not encrypt index
    fpath_prefix = os.path.join(dirpath, name)
    logger.debug(f"save index to {fpath_prefix}.index")
    faiss.write_index(store_index, f"{fpath_prefix}.index")
    index.store.index = None

    with open(f"{fpath_prefix}.store", "wb") as f:
        cipher = AES.new(key, AES.MODE_EAX)
        ciphertext, tag = cipher.encrypt_and_digest(pickle.dumps(index.store))
        [f.write(x) for x in (cipher.nonce, tag, ciphertext)]
    index.store.index = store_index

    # with open(f"{fpath_prefix}.scanedfile", "wb") as f:
    #     cipher = AES.new(key, AES.MODE_EAX)
    #     ciphertext, tag = cipher.encrypt_and_digest(pickle.dumps(index.scaned_files))
    #     [f.write(x) for x in (cipher.nonce, tag, ciphertext)]

    # test
    load_encrypt_store(dirpath, name, password)

    return [
        f"{fpath_prefix}.index",
        f"{fpath_prefix}.store",
        # f"{fpath_prefix}.scanedfile",
    ]


def load_encrypt_store(dirpath, name, password) -> Index:
    """
    Args:
        dirpath: dirpath to store index files
        name: project/file name
        key: AES256 key used to decrypt the index files
    """
    key = derive_key(password)

    fpath_prefix = os.path.join(dirpath, name)
    with open(f"{fpath_prefix}.store", "rb") as f:
        nonce, tag, ciphertext = [f.read(x) for x in (16, 16, -1)]
    cipher = AES.new(key, AES.MODE_EAX, nonce)
    store = pickle.loads(cipher.decrypt_and_verify(ciphertext, tag))

    # index not encrypted
    index = faiss.read_index(f"{os.path.join(dirpath, name)}.index")
    store.index = index

    # with open(f"{fpath_prefix}.scanedfile", "rb") as f:
    #     nonce, tag, ciphertext = [f.read(x) for x in (16, 16, -1)]
    # cipher = AES.new(key, AES.MODE_EAX, nonce)
    # scaned_files = pickle.loads(cipher.decrypt_and_verify(ciphertext, tag))

    return Index(
        store=store,
        scaned_files=set([]),
    )
