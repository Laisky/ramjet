import tempfile
import codecs
import glob
import hashlib
import os
import pickle
import textwrap
from collections import namedtuple
from sys import path
from typing import List, Tuple, Dict

import faiss
from Crypto.Cipher import AES
from kipp.utils import setup_logger
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, MarkdownTextSplitter
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

from ramjet.settings import prd

logger = setup_logger("security")
UserChain = namedtuple("UserChain", ["chain", "index", "datasets"])

Index = namedtuple("index", ["store", "scaned_files"])
user_embeddings_chain: Dict[str, UserChain] = {}  # uid -> UserChain


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
N_NEAREST_CHUNKS = 5


def is_file_scaned(index: Index, fpath):
    return os.path.split(fpath)[1] in index.scaned_files


def build_chain(llm, store: FAISS, nearest_k=N_NEAREST_CHUNKS):
    def chain(query):
        related_docs = store.similarity_search(
            query=query["question"],
            k=nearest_k,
        )
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(
            input_documents=related_docs,
            question=query["question"],
        )
        refs = [d.metadata["source"] for d in related_docs]
        return response, refs

    return chain


def build_user_chain(user: prd.UserPermission, index: Index, datasets: List[str]):
    """build user's embedding index and save in memory"""
    uid = user.uid
    n_chunks = N_NEAREST_CHUNKS
    model_name = user.chat_model or "gpt-3.5-turbo"
    max_tokens = 1000
    if os.environ.get("OPENAI_API_TYPE", "") == "azure":
        llm = AzureChatOpenAI(
            client=None,
            deployment_name="gpt35",
            # model_name="gpt-3.5-turbo",
            temperature=0,
            max_tokens=1000,
            streaming=False,
        )
    else:
        if "16k" in model_name:
            max_tokens = 8000
            n_chunks = max(n_chunks, 10)

        llm = ChatOpenAI(
            client=None,
            model=model_name,
            temperature=0,
            max_tokens=max_tokens,
            streaming=False,
        )

    user_embeddings_chain[uid] = UserChain(
        chain=build_chain(llm, index.store, n_chunks),
        index=index,
        datasets=datasets,
    )
    logger.info(
        f"succeed to build user chain {uid=}, {model_name=}, {max_tokens=}, {n_chunks=}"
    )


def embedding_pdf(fpath: str, metadata_name: str, max_chunks=1500) -> Index:
    """embedding pdf file

    pricing: https://openai.com/pricing

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

    assert len(docs) < max_chunks, f"too many chunks {len(docs)} > {max_chunks}"

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


def restore_user_chain(s3cli, user: prd.UserPermission, password: str):
    uid = user.uid
    with tempfile.TemporaryDirectory() as tmpdir:
        # encrypt and upload origin pdf file
        objkeys = [
            quote(
                f"{prd.OPENAI_S3_EMBEDDINGS_prefix}/{uid}/chatbot/default/qachat.index"
            ),
            quote(
                f"{prd.OPENAI_S3_EMBEDDINGS_prefix}/{uid}/chatbot/default/qachat.store"
            ),
            quote(
                f"{prd.OPENAI_S3_EMBEDDINGS_prefix}/{uid}/chatbot/default/datasets.pkl"
            ),
        ]

        for objkey in objkeys:
            s3cli.fget_object(
                bucket_name=prd.OPENAI_S3_EMBEDDINGS_BUCKET,
                object_name=objkey,
                file_path=os.path.join(tmpdir, os.path.basename(objkey)),
            )

        # read index file
        index = load_encrypt_store(
            dirpath=tmpdir,
            name="qachat",
            password=password,
        )

        # read datasets file
        with open(os.path.join(tmpdir, "datasets.pkl"), "rb") as fp:
            datasets = pickle.load(fp)

    logger.info(f"succeed restore user {uid=} chain from s3")
    build_user_chain(user, index, datasets)


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


def load_encrypt_store(dirpath: str, name: str, password: str) -> Index:
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
