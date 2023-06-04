import os
import glob
import codecs
import pickle
import re
import textwrap
from collections import namedtuple

import openai
import faiss
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, MarkdownTextSplitter
from langchain.vectorstores import FAISS
from pymongo import MongoClient
from kipp.utils import setup_logger

from sys import path

path.append("/opt/configs/ramjet")
import prd

# ----------------------------------------------
# Azure
# ----------------------------------------------
os.environ['OPENAI_API_TYPE'] = "azure"
os.environ['OPENAI_API_VERSION'] = "2023-05-15"
os.environ['OPENAI_API_BASE'] = prd.OPENAI_AZURE_API + "/"
os.environ['OPENAI_API_KEY'] = prd.OPENAI_AZURE_TOKEN

azure_embeddings_deploymentid = "embedding"
azure_gpt_deploymentid = "gpt35"
# ----------------------------------------------

# ----------------------------------------------
# OpenAI
# ----------------------------------------------
# os.environ["OPENAI_API_KEY"] = prd.OPENAI_TOKEN
# ----------------------------------------------

Index = namedtuple("index", ["store", "scaned_files"])

logger = setup_logger("security")

def pretty_print(text: str) -> str:
    text = text.strip()
    return textwrap.fill(text, width=60, subsequent_indent="    ")

# ==============================================================
# prepare pdf documents docs.index & docs.store
#
# https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/pdf.html#retain-elements
#
# 通用的函数定义
# ==============================================================

from urllib.parse import quote

from langchain.document_loaders import PyPDFLoader

# from langchain.document_loaders import UnstructuredPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, MarkdownTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=500, separator="\n")
markdown_splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=50)

N_BACTCH_FILES = 5


def is_file_scaned(index: Index, fpath):
    return os.path.split(fpath)[1] in index.scaned_files


def embedding_pdfs(index: Index, fpaths, url, replace_by_url):
    i = 0
    docs = []
    metadatas = []
    for fpath in fpaths:
        fname = os.path.split(fpath)[1]
        if is_file_scaned(index, fname):
            continue

        try:
            loader = PyPDFLoader(fpath)
            for page, data in enumerate(loader.load_and_split()):
                splits = text_splitter.split_text(data.page_content)
                docs.extend(splits)
                for ichunk, _ in enumerate(splits):
                    fnameurl = quote(fpath.removeprefix(replace_by_url), safe="")
                    furl = url + fnameurl
                    metadatas.append({"source": f"{furl}#page={page+1}"})
        except Exception as err:
            logger.error(f"skip file {fpath}: {err}")
            continue

        index.scaned_files.add(fname)
        print(f"scaned {fpath}")
        i += 1
        if i > N_BACTCH_FILES:
            break

    if i != 0:
        index.store.add_texts(docs, metadatas=metadatas)

    return i


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


def load_store(dirpath, name) -> Index:
    """
    Args:
        dirpath: dirpath to store index files
        name: project/file name
    """
    index = faiss.read_index(f"{os.path.join(dirpath, name)}.index")
    with open(f"{os.path.join(dirpath, name)}.store", "rb") as f:
        store = pickle.load(f)
    store.index = index

    with open(f"{os.path.join(dirpath, name)}.scanedfile", "rb") as f:
        scaned_files = pickle.load(f)

    return Index(
        store=store,
        scaned_files=scaned_files,
    )


def new_store() -> Index:
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        deployment=azure_embeddings_deploymentid,
    )
    store = FAISS.from_texts(["world"], OpenAIEmbeddings(), metadatas=[{"source": "hello"}])
    return Index(
        store=store,
        scaned_files=set([]),
    )


def save_store(index: Index, dirpath, name):
    store_index = index.store.index
    fpath_prefix = os.path.join(dirpath, name)
    print(f"save store to {fpath_prefix}")
    faiss.write_index(store_index, f"{fpath_prefix}.index")
    index.store.index = None
    with open(f"{fpath_prefix}.store", "wb") as f:
        pickle.dump(index.store, f)
    index.store.index = store_index

    with open(f"{fpath_prefix}.scanedfile", "wb") as f:
        pickle.dump(index.scaned_files, f)


from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.schema import HumanMessage

def chat():
    llm = AzureChatOpenAI(
        deployment_name=azure_gpt_deploymentid,
        model_name="gpt-3.5-turbo",
    )

    print(llm([HumanMessage(content='Hello')]))

def embedding():



if __name__ == "__main__":
    main()
