import codecs
import glob
import hashlib
import os
import pickle
import tempfile
import textwrap
import threading
from collections import namedtuple
from sys import path
from typing import Dict, List, Tuple, Callable
from urllib.parse import quote

import faiss
from Crypto.Cipher import AES
from kipp.utils import setup_logger
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, MarkdownTextSplitter
from langchain.vectorstores import FAISS
from minio import Minio
from langchain.document_loaders import Docx2txtLoader, UnstructuredPowerPointLoader, UnstructuredWordDocumentLoader

from ramjet.settings import prd
from ..base import logger

UserChain = namedtuple("UserChain", ["chain", "index", "datasets"])

Index = namedtuple("index", ["store", "scaned_files"])
user_embeddings_chain_mu = threading.RLock()
user_embeddings_chain: Dict[str, UserChain] = {}  # uid -> UserChain
user_shared_chain_mu = threading.RLock()
user_shared_chain: Dict[str, UserChain] = {}  # uid-botname -> UserChain

text_splitter = CharacterTextSplitter(chunk_size=500, separator="\n")
markdown_splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=50)

N_BACTCH_FILES = 5
N_NEAREST_CHUNKS = 5


def is_file_scaned(index: Index, fpath: str) -> bool:
    """
    check if a file is already scaned by the index, if so, skip it

    Args:
        index (Index): index
        fpath (str): file path

    Returns:
        bool: True if the file is already scaned
    """
    return os.path.split(fpath)[1] in index.scaned_files


def build_chain(
    llm, store: FAISS, nearest_k=N_NEAREST_CHUNKS
) -> Callable[[Dict], Tuple[str, List[str]]]:
    """
    build a chain for a given store

    Args:
        llm: langchain chat model
        store: vector store
        nearest_k: number of nearest chunks to search,
            these chunks will be used as context for the chat model
    """

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


def build_user_chain(
    user: prd.UserPermission, index: Index, datasets: List[str]
) -> UserChain:
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

    logger.info(
        f"succeed to build user chain {uid=}, {model_name=}, {max_tokens=}, {n_chunks=}"
    )
    return UserChain(
        chain=build_chain(llm, index.store, n_chunks),
        index=index,
        datasets=datasets,
    )


def embedding_file(fpath: str, metadata_name: str, max_chunks=1500) -> Index:
    file_ext = os.path.splitext(fpath)[1].lower()
    if file_ext == ".pdf":
        return _embedding_pdf(fpath, metadata_name, max_chunks)
    elif file_ext == ".md":
        return _embedding_markdown(fpath, metadata_name, max_chunks)
    elif file_ext == ".docx" or file_ext == ".doc":
        return _embedding_word(fpath, metadata_name, max_chunks)
    elif file_ext == ".pptx" or file_ext == ".ppt":
        return _embedding_ppt(fpath, metadata_name, max_chunks)
    else:
        raise ValueError(f"unsupported file type {file_ext}")


PDF_EOF_MARKER = b"%%EOF"


def reset_eof_of_pdf(fpath: str) -> None:
    """
    reset the EOF marker of a pdf file.
    fix `EOF marker not found`. see https://github.com/py-pdf/pypdf/issues/480

    will overwrite the original file.

    Args:
        fpath (str): file path
    """
    with open(fpath, "rb") as f:
        content = f.read()

    # replace the EOF marker to the end of the file
    if PDF_EOF_MARKER in content:
        content = content.replace(PDF_EOF_MARKER, b"")

    content = content + PDF_EOF_MARKER
    with open(fpath, "wb") as f:
        f.write(content)


def _embedding_pdf(fpath: str, metadata_name: str, max_chunks=1500) -> Index:
    """embedding pdf file

    pricing: https://openai.com/pricing

    Args:
        fpath (str): file path
        metadata_name (str): file key

    Returns:
        Index: index
    """
    logger.info(f"call embedding_pdf {fpath=}, {metadata_name=}")

    reset_eof_of_pdf(fpath)
    loader = PyPDFLoader(fpath)

    index = new_store()
    docs = []
    metadatas = []
    for page, data in enumerate(loader.load_and_split()):
        splits = text_splitter.split_text(data.page_content)
        docs.extend(splits)
        logger.debug(f"embedding {fpath} page {page+1} with {len(splits)} chunks")
        for ichunk, _ in enumerate(splits):
            metadatas.append({"source": f"{metadata_name}#page={page+1}"})

    assert len(docs) <= max_chunks, f"too many chunks {len(docs)} > {max_chunks}"

    logger.info(f"succeed embedding pdf {fpath} with {len(docs)} chunks")
    index.store.add_texts(docs, metadatas=metadatas)
    return index


def _embedding_markdown(fpath: str, metadata_name: str, max_chunks=1500) -> Index:
    """embedding markdown file

    Args:
        fpath (str): file path
        metadata_name (str): file key

    Returns:
        Index: index
    """
    logger.info(f"call embedding_markdown {fpath=}, {metadata_name=}")
    index = new_store()
    docs = []
    metadatas = []
    with codecs.open(fpath, "rb", "utf-8") as fp:
        docus = markdown_splitter.create_documents([fp.read()])
        for ichunk, docu in enumerate(docus):
            docs.append(docu.page_content)

            title = quote(docu.page_content.strip().split("\n", maxsplit=1)[0])
            metadatas.append({"source": f"{metadata_name}#{title}"})

    assert len(docs) <= max_chunks, f"too many chunks {len(docs)} > {max_chunks}"

    logger.info(f"succeed embedding markdown {fpath} with {len(docs)} chunks")
    index.store.add_texts(docs, metadatas=metadatas)
    return index


def _embedding_word(fpath: str, metadata_name: str, max_chunks=1500) -> Index:
    """embedding word file

    Args:
        fpath (str): file path
        metadata_name (str): file key

    Returns:
        Index: index
    """
    logger.info(f"call embeddings_word {fpath=}, {metadata_name=}")
    index = new_store()
    docs = []
    metadatas = []

    fileext = os.path.splitext(fpath)[1].lower()
    if fileext == ".docx":
        loader = Docx2txtLoader(fpath)
    elif fileext == ".doc":
        loader = UnstructuredWordDocumentLoader(fpath)
    else:
        raise ValueError(f"unsupported file type {fileext}")

    for page, data in enumerate(loader.load_and_split()):
        splits = text_splitter.split_text(data.page_content)
        docs.extend(splits)
        logger.debug(f"embedding {fpath} page {page+1} with {len(splits)} chunks")
        for ichunk, _ in enumerate(splits):
            metadatas.append({"source": f"{metadata_name}#page={page+1}"})

    assert len(docs) <= max_chunks, f"too many chunks {len(docs)} > {max_chunks}"

    logger.info(f"succeed embedding word {fpath} with {len(docs)} chunks")
    index.store.add_texts(docs, metadatas=metadatas)
    return index


def _embedding_ppt(fpath: str, metadata_name: str, max_chunks=1500) -> Index:
    logger.info(f"call embeddings_word {fpath=}, {metadata_name=}")
    index = new_store()
    docs = []
    metadatas = []
    loader = UnstructuredPowerPointLoader(fpath)
    for page, data in enumerate(loader.load_and_split()):
        splits = text_splitter.split_text(data.page_content)
        docs.extend(splits)
        logger.debug(f"embedding {fpath} page {page+1} with {len(splits)} chunks")
        for ichunk, _ in enumerate(splits):
            metadatas.append({"source": f"{metadata_name}#page={page+1}"})

    assert len(docs) <= max_chunks, f"too many chunks {len(docs)} > {max_chunks}"

    logger.info(f"succeed embedding powerpoint {fpath} with {len(docs)} chunks")
    index.store.add_texts(docs, metadatas=metadatas)
    return index


def new_store() -> Index:
    """
    new FAISS store

    Returns:
        Index: FAISS index
    """
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


def derive_key(password: str) -> bytes:
    """
    derive aes key in 32 bytes from password
    """
    key = hashlib.pbkdf2_hmac(
        "sha256",
        bytes(password, "utf-8"),
        b"123456",
        100000,  # iteration count
        dklen=32,  # length of the derived key
    )
    return key


def download_chatbot_index(
    dirpath: str,
    s3cli: Minio,
    user: prd.UserPermission,
    chatbot_name: str = "",
    password: str = "",
) -> Tuple[Index, List[str]]:
    """
    download chatbot index from s3

    Args:
        s3cli (Minio): s3 client
        user (prd.UserPermission): user
        chatbot_name (str, optional): chatbot name. Defaults to "".
            If empty, download __CURRENT chatbot name from s3.
        password (str, optional): password. Defaults to "".
            if empty, download shared chatbot.

    Returns:
        Tuple[Index, List[str]]: index, datasets
    """
    uid = user.uid
    logger.debug(f"call download_chatbot_index {uid=}, {dirpath=}, {chatbot_name=}")

    if chatbot_name == "":
        # download current chatbot name
        response = s3cli.get_object(
            bucket_name=prd.OPENAI_S3_EMBEDDINGS_BUCKET,
            object_name=f"{prd.OPENAI_S3_EMBEDDINGS_PREFIX}/{uid}/chatbot/__CURRENT",
        )
        chatbot_name = response.data.decode("utf-8")
        response.close()
        response.release_conn()

    if password:
        objkeys = [
            quote(
                f"{prd.OPENAI_S3_EMBEDDINGS_PREFIX}/{uid}/chatbot/{chatbot_name}.index"
            ),
            quote(
                f"{prd.OPENAI_S3_EMBEDDINGS_PREFIX}/{uid}/chatbot/{chatbot_name}.store"
            ),
            quote(
                f"{prd.OPENAI_S3_EMBEDDINGS_PREFIX}/{uid}/chatbot/{chatbot_name}.pkl"
            ),
        ]
    else:
        objkeys = [
            quote(
                f"{prd.OPENAI_S3_EMBEDDINGS_PREFIX}/{uid}/chatbot-share/{chatbot_name}.index"
            ),
            quote(
                f"{prd.OPENAI_S3_EMBEDDINGS_PREFIX}/{uid}/chatbot-share/{chatbot_name}.store"
            ),
            quote(
                f"{prd.OPENAI_S3_EMBEDDINGS_PREFIX}/{uid}/chatbot-share/{chatbot_name}.pkl"
            ),
        ]

    for objkey in objkeys:
        s3cli.fget_object(
            bucket_name=prd.OPENAI_S3_EMBEDDINGS_BUCKET,
            object_name=objkey,
            file_path=os.path.join(dirpath, os.path.basename(objkey)),
        )

    # read index file
    if password:
        index = load_encrypt_store(
            dirpath=dirpath,
            name=chatbot_name,
            password=password,
        )
    else:
        index = load_plaintext_store(
            dirpath=dirpath,
            name=chatbot_name,
        )

    # read datasets file
    with open(os.path.join(dirpath, f"{chatbot_name}.pkl"), "rb") as fp:
        datasets = pickle.load(fp)

    return index, datasets


def restore_user_chain(
    s3cli: Minio, user: prd.UserPermission, password: str = "", chatbot_name: str = ""
) -> None:
    """
    load and restore user chain from s3

    Args:
        s3cli (Minio): s3 client
        user (prd.UserPermission): user
        password (str): password.
            if empty, build chain from shared datasets
        chatbot_name (str, optional): chatbot name. Defaults to "".
            If empty, download __CURRENT chatbot name from s3.
    """
    uid = user.uid
    with tempfile.TemporaryDirectory() as tmpdir:
        index, datasets = download_chatbot_index(
            s3cli=s3cli,
            user=user,
            dirpath=tmpdir,
            chatbot_name=chatbot_name,
            password=password,
        )

    logger.info(f"succeed restore user {uid=} chain from s3")
    chain = build_user_chain(user, index, datasets)

    if password:
        logger.info(f"load encrypted user {uid=} chain {chatbot_name=}")
        with user_embeddings_chain_mu:
            user_embeddings_chain[uid] = chain
    else:
        logger.info(f"load shared user {uid=} chain {chatbot_name=}")
        with user_shared_chain_mu:
            user_shared_chain[uid + chatbot_name] = chain


def save_encrypt_store(
    s3cli: Minio,
    user: prd.UserPermission,
    index: Index,
    name: str,
    password: str,
    datasets: List[str] = [],
) -> None:
    """save encrypted store

    Args:
        index (Index): index
        dirpath (str): dirpath
        name (str): name of file, without ext
        password (str): password
        datasets (List[str]): datasets, if empty, save as user's datasets,
            if not empty, save as user's chatbot.

    Returns:
        List[str]: saved filepaths
    """
    key = derive_key(password)
    store_index = index.store.index
    uid = user.uid

    with tempfile.TemporaryDirectory() as tmpdir:
        # do not encrypt index
        fpath_prefix = os.path.join(tmpdir, name)
        logger.debug(f"save index to {fpath_prefix}.index")
        faiss.write_index(store_index, f"{fpath_prefix}.index")
        index.store.index = None

        with open(f"{fpath_prefix}.store", "wb") as f:
            cipher = AES.new(key, AES.MODE_EAX)
            ciphertext, tag = cipher.encrypt_and_digest(pickle.dumps(index.store))
            [f.write(x) for x in (cipher.nonce, tag, ciphertext)]
        index.store.index = store_index

        fs = [
            f"{fpath_prefix}.index",
            f"{fpath_prefix}.store",
        ]

        if datasets:
            # save user built chatbot, dump selected datasets
            datasets_fname = os.path.join(tmpdir, f"{name}.pkl")
            fs.append(datasets_fname)
            with open(datasets_fname, "wb") as datasets_fp:
                pickle.dump(datasets, datasets_fp)

        logger.debug(f"try to upload embedding chat store {uid=}")
        for fpath in fs:
            if datasets:
                objkey = quote(
                    f"{prd.OPENAI_S3_EMBEDDINGS_PREFIX}/{uid}/chatbot/{os.path.basename(fpath)}"
                )
            else:
                objkey = quote(
                    f"{prd.OPENAI_S3_EMBEDDINGS_PREFIX}/{uid}/{os.path.basename(fpath)}"
                )

            s3cli.fput_object(
                bucket_name=prd.OPENAI_S3_EMBEDDINGS_BUCKET,
                object_name=objkey,
                file_path=fpath,
            )
            logger.debug(f"upload {objkey} to s3")


def save_plaintext_store(
    s3cli: Minio, user: prd.UserPermission, index: Index, name: str, datasets: List[str]
) -> None:
    """save plaintext store

    Args:
        index (Index): index
        name (str): name of file, without ext
        datasets (List[str]): datasets, if empty, save as user's datasets,
            if not empty, save as user's chatbot.
    """

    uid = user.uid

    with tempfile.TemporaryDirectory() as tmpdir:
        # do not encrypt index
        fpath_prefix = os.path.join(tmpdir, name)
        logger.debug(f"save index to {fpath_prefix}.index")
        faiss.write_index(index.store.index, f"{fpath_prefix}.index")

        idx = index.store.index
        index.store.index = None
        with open(f"{fpath_prefix}.store", "wb") as f:
            pickle.dump(index.store, f)
        index.store.index = idx

        fs = [
            f"{fpath_prefix}.index",
            f"{fpath_prefix}.store",
        ]

        # save user built chatbot, dump selected datasets
        datasets_fname = os.path.join(tmpdir, f"{name}.pkl")
        fs.append(datasets_fname)
        with open(datasets_fname, "wb") as datasets_fp:
            pickle.dump(datasets, datasets_fp)

        logger.debug(f"try to upload embedding chat store {uid=}")
        for fpath in fs:
            objkey = quote(
                f"{prd.OPENAI_S3_EMBEDDINGS_PREFIX}/{uid}/chatbot-share/{os.path.basename(fpath)}"
            )

            s3cli.fput_object(
                bucket_name=prd.OPENAI_S3_EMBEDDINGS_BUCKET,
                object_name=objkey,
                file_path=fpath,
            )
            logger.info(f"upload plaintext {objkey} to s3")


def load_plaintext_store(dirpath: str, name: str) -> Index:
    """
    Args:
        dirpath: dirpath to store index files
        name: project/file name
    """
    fpath_prefix = os.path.join(dirpath, name)
    with open(f"{fpath_prefix}.store", "rb") as f:
        store = pickle.load(f)

    # index not encrypted
    index = faiss.read_index(f"{os.path.join(dirpath, name)}.index")
    store.index = index

    # with open(f"{fpath_prefix}.scanedfile", "rb") as f:
    #     index.scaned_files = pickle.load(f)

    return Index(
        store=store,
        scaned_files=set([]),
    )


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
