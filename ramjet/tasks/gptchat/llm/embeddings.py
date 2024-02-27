import codecs
import hashlib
import os
import pickle
import re
import tempfile
import threading
import time
from typing import Callable, Dict, List, Tuple, Optional, Any
from urllib.parse import quote
from concurrent.futures import Future

import faiss
from kipp.utils import timer
from Crypto.Cipher import AES
from langchain.chains import LLMChain
from langchain_community.document_loaders import (
    BSHTMLLoader,
    Docx2txtLoader,
    PyPDFLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders.base import BaseLoader
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema.document import Document
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores.faiss import FAISS
from minio import Minio

from ramjet.engines import thread_executor
from ramjet.settings import prd

from ..base import logger
from .base import Index, UserChain


user_embeddings_chain_mu = threading.RLock()
user_embeddings_chain: Dict[str, UserChain] = {}  # uid -> UserChain
user_shared_chain_mu = threading.RLock()
user_shared_chain: Dict[str, UserChain] = {}  # uid-botname -> UserChain


N_BACTCH_FILES: int = 5
N_NEAREST_CHUNKS: int = 5
DEFAULT_MAX_CHUNKS: int = 300


def build_embeddings_llm_for_user(user: prd.UserPermission) -> OpenAIEmbeddings:
    """build llm for user

    Args:
        user (UserPermission): user info

    Returns:
        ChatOpenAI: llm
    """
    return OpenAIEmbeddings(
        client=None,
        openai_api_key=user.apikey,
        model="text-embedding-3-small",
    )


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


def build_search(
    store: FAISS, nearest_k=N_NEAREST_CHUNKS
) -> Callable[[str], Tuple[str, List[str]]]:
    """build search function for a given store

    Args:
        store (FAISS): store
        nearest_k (int, optional): number of nearest chunks to search.
            Defaults to N_NEAREST_CHUNKS.

    Returns:
        Callable[[str], Tuple[str, List[str]]]:
            search function, return context and references
    """

    @timer
    def query_for_more_info(query: str) -> Tuple[str, List[str]]:
        """query more information from embeddings store

        Args:
            query (str): query

        Returns:
            Tuple[str, List[str]]: context and references
        """
        logger.debug(f"query for more info: {query}")
        related_docs = store.similarity_search(
            query=query,
            k=nearest_k,
        )

        ctx = "; ".join([d.page_content for d in related_docs if d.page_content])
        refs = [d.metadata["source"] for d in related_docs]

        return ctx, refs

    return query_for_more_info


def build_chain(
    store: FAISS, nearest_k=N_NEAREST_CHUNKS
) -> Callable[[ChatOpenAI, str], Tuple[str, List[str]]]:
    """
    build a chain for a given store

    Args:
        store: vector store
        nearest_k: number of nearest chunks to search,
            these chunks will be used as context for the chat model

    Returns:
        Callable[[ChatOpenAI, str], Tuple[str, List[str]]]: a chain function
    """

    system_template = """Use the following pieces of context to answer the users question.
    If you don't know the answer, or you think more information is needed to provide a better answer,
    just say in this strict format: "I need more informations about: [list keywords that will be used to search more informations]" to ask more informations,
    don't try to make up an answer.
    ----------------
    context: {summaries}"""
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)

    # fix stupid compatability issue in langchain faiss
    if not getattr(store, "_normalize_L2", None):
        store._normalize_L2 = False  # pyliny: disable=protected-access

    query_for_more_info = build_search(store=store, nearest_k=nearest_k)

    def chain(llm: ChatOpenAI, query: str) -> Tuple[str, List[str]]:
        """chain function

        Args:
            query (str): query

        Returns:
            Tuple[str, List[str]]: response and references
        """
        query_chain = LLMChain(llm=llm, prompt=prompt)
        n = 0
        last_sub_query = ""
        regexp = re.compile(r"I need more information about: .*")
        all_refs: List[str] = []
        ctx, refs = query_for_more_info(query)
        resp = ""
        while n < 3:
            n += 1
            all_refs.extend(refs)
            resp = query_chain.run(
                {
                    "summaries": ctx,
                    "question": query,
                }
            )
            matched = regexp.findall(resp)
            if len(matched) == 0:
                break

            # load more context by new sub_query
            sub_query = matched[0]
            if sub_query == last_sub_query:
                break
            last_sub_query = sub_query

            logger.debug(f"require more informations about: {sub_query}")
            new_ctx, refs = query_for_more_info(sub_query)
            ctx += f"; {new_ctx}"

        all_refs = [r for r in all_refs if r != "hello"]
        return resp, all_refs

    return chain


def build_user_chain(
    index: Index, datasets: List[str], nearest_k: int = N_NEAREST_CHUNKS
) -> UserChain:
    """build user's embedding index and save in memory

    Args:
        index (Index): index
        datasets (List[str]): datasets
        nearest_k (int, optional): number of nearest chunks to search.

    Returns:
        UserChain: user chain
    """
    return UserChain(
        chain=build_chain(store=index.store, nearest_k=nearest_k),
        user_index=index,
        datasets=datasets,
        search=build_search(store=index.store, nearest_k=nearest_k),
    )


def embedding_file(
    fpath: str,
    metadata_name: str,
    apikey: str,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
    api_base: str = "https://api.openai.com/v1/",
) -> Index:
    """read and parse file content, then embedding it to FAISS index

    this function is thread safe

    Args:
        fpath (str): file path
        metadata_name (str): file key
        apikey (str): openai api key
        max_chunks (int, optional): max chunks

    Returns:
        Index: index
    """
    start_at: float = time.time()
    file_ext: str = os.path.splitext(fpath)[1].lower()
    if file_ext == ".pdf":
        idx = embedding_pdf(
            fpath=fpath,
            metadata_name=metadata_name,
            max_chunks=max_chunks,
            apikey=apikey,
            api_base=api_base,
        )
    elif file_ext in [".md", ".txt"]:
        idx = embedding_markdown(
            fpath=fpath,
            metadata_name=metadata_name,
            max_chunks=max_chunks,
            apikey=apikey,
            api_base=api_base,
        )
    elif file_ext in [".docx", ".doc"]:
        idx = embedding_msword(
            fpath=fpath,
            metadata_name=metadata_name,
            max_chunks=max_chunks,
            apikey=apikey,
            api_base=api_base,
        )
    elif file_ext in [".pptx", ".ppt"]:
        idx = embedding_msppt(
            fpath=fpath,
            metadata_name=metadata_name,
            max_chunks=max_chunks,
            apikey=apikey,
            api_base=api_base,
        )
    elif file_ext == ".html":
        idx = embedding_html(
            fpath=fpath,
            metadata_name=metadata_name,
            max_chunks=max_chunks,
            apikey=apikey,
            api_base=api_base,
        )
    else:
        raise ValueError(f"unsupported file type {file_ext}")

    logger.info(
        f"embedding {fpath} done, {api_base=}, cost {time.time() - start_at:.2f}s"
    )
    return idx


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

    if PDF_EOF_MARKER in content:
        # content = content.replace(PDF_EOF_MARKER, b"")
        return

    # replace the EOF marker to the end of the file
    content = content + PDF_EOF_MARKER
    with open(fpath, "wb") as f:
        f.write(content)


def embedding_pdf(
    fpath: str,
    metadata_name: str,
    apikey: str,
    api_base: str = "https://api.openai.com/v1",
    max_chunks: int = DEFAULT_MAX_CHUNKS,
) -> Index:
    """embedding pdf file

    pricing: https://openai.com/pricing

    Args:
        fpath (str): file path
        metadata_name (str): file key
        apikey (str): openai api key
        max_chunks (int, optional): max chunks

    Returns:
        Index: index
    """
    logger.info(f"call embedding_pdf {fpath=}, {metadata_name=}")

    reset_eof_of_pdf(fpath)
    loader = PyPDFLoader(fpath)

    index = new_store(apikey=apikey, api_base=api_base)
    docs: List[str] = []
    metadatas: List[dict] = []
    text_splitter = TokenTextSplitter(
        chunk_size=500,
        chunk_overlap=30,
    )
    for page, data in enumerate(loader.load_and_split()):
        splits = text_splitter.split_text(data.page_content)
        docs.extend(splits)
        logger.debug(f"embedding {fpath} page {page+1} with {len(splits)} chunks")
        for i, _ in enumerate(splits):
            metadatas.append({"source": f"{metadata_name}#page={page+1}?chunk={i+1}"})

    assert len(docs) <= max_chunks, (
        f"your account limit to parse at most {max_chunks} chunks, ",
        f"but you submit {len(docs)}, consider upgrade your account",
    )

    logger.debug(f"send chunk to LLM embeddings, {fpath=}, {len(docs)} chunks")
    futures: List[Future] = []
    start_idx: int = 0
    n_batch: int = 5
    while True:
        if start_idx >= len(docs):
            break

        end_at = min(start_idx + n_batch, len(docs))
        f = thread_executor.submit(
            _embeddings_worker,
            texts=docs[start_idx:end_at],
            metadatas=metadatas[start_idx:end_at],
            apikey=apikey,
            api_base=api_base,
        )
        futures.append(f)
        start_idx = end_at

    index = new_store(apikey=apikey, api_base=api_base)
    for f in futures:
        index.store.merge_from(f.result())

    return index


def _embeddings_worker(
    texts: List[str],
    metadatas: List[dict],
    apikey: str,
    api_base: str = "https://api.openai.com/v1",
) -> FAISS:
    index = new_store(apikey=apikey, api_base=api_base)
    index.store.add_texts(texts=texts, metadatas=metadatas)
    return index.store


def embedding_markdown(
    fpath: str,
    metadata_name: str,
    apikey: str,
    api_base: str = "https://api.openai.com/v1",
    max_chunks=DEFAULT_MAX_CHUNKS,
) -> Index:
    """embedding markdown file

    Args:
        fpath (str): file path
        metadata_name (str): file key
        apikey (str): openai api key

    Returns:
        Index: index
    """
    logger.info(f"call embedding_markdown {fpath=}, {metadata_name=}")
    # splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=50)
    splitter = TokenTextSplitter(
        chunk_size=500,
        chunk_overlap=30,
    )
    index = new_store(apikey=apikey, api_base=api_base)
    docs = []
    metadatas = []
    docus: Optional[List[Document]] = None
    err: Exception
    for charset in ["utf-8", "gbk", "gb2312"]:
        try:
            fp = codecs.open(fpath, "rb", charset)
            docus = splitter.create_documents([fp.read()])
            fp.close()
            break
        except UnicodeDecodeError as e:
            err = e
            continue

    if not docus:
        raise err

    for ichunk, docu in enumerate(docus):
        title = quote(docu.page_content.strip().split("\n", maxsplit=1)[0])
        docs.append(docu.page_content)
        metadatas.append({"source": f"{metadata_name}#{title}?chunk={ichunk+1}"})

    assert len(docs) <= max_chunks, (
        f"your account limit to parse at most {max_chunks} chunks, ",
        f"but you submit {len(docs)}, consider upgrade your account",
    )

    logger.debug(f"send chunk to LLM embeddings, {fpath=}, {len(docs)} chunks")
    futures: List[Future] = []
    start_at = 0
    n_batch = 5
    while True:
        if start_at >= len(docs):
            break

        end_at = min(start_at + n_batch, len(docs))
        f = thread_executor.submit(
            _embeddings_worker,
            texts=docs[start_at:end_at],
            metadatas=metadatas[start_at:end_at],
            apikey=apikey,
            api_base=api_base,
        )
        futures.append(f)
        start_at = end_at

    index = new_store(apikey=apikey, api_base=api_base)
    for f in futures:
        index.store.merge_from(f.result())

    return index


def embedding_msword(
    fpath: str,
    metadata_name: str,
    apikey: str,
    api_base: str = "https://api.openai.com/v1",
    max_chunks=DEFAULT_MAX_CHUNKS,
) -> Index:
    """embedding word file

    Args:
        fpath (str): file path
        metadata_name (str): file key
        apikey (str): openai api key
        max_chunks (int, optional): max chunks

    Returns:
        Index: index
    """
    logger.info(f"call embeddings_word {fpath=}, {metadata_name=}")
    index = new_store(apikey=apikey, api_base=api_base)
    docs = []
    metadatas = []
    text_splitter = TokenTextSplitter(
        chunk_size=500,
        chunk_overlap=30,
    )

    fileext = os.path.splitext(fpath)[1].lower()
    loader: BaseLoader
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
            metadatas.append(
                {"source": f"{metadata_name}#page={page+1}?chunk={ichunk+1}"}
            )

    assert len(docs) <= max_chunks, (
        f"your account limit to parse at most {max_chunks} chunks, ",
        f"but you submit {len(docs)}, consider upgrade your account",
    )

    logger.debug(f"send chunk to LLM embeddings, {fpath=}, {len(docs)} chunks")
    futures: List[Future] = []
    start_at = 0
    n_batch = 5
    while True:
        if start_at >= len(docs):
            break

        end_at = min(start_at + n_batch, len(docs))
        f = thread_executor.submit(
            _embeddings_worker,
            texts=docs[start_at:end_at],
            metadatas=metadatas[start_at:end_at],
            apikey=apikey,
            api_base=api_base,
        )
        futures.append(f)
        start_at = end_at

    index = new_store(apikey=apikey, api_base=api_base)
    for f in futures:
        index.store.merge_from(f.result())

    return index


def embedding_msppt(
    fpath: str,
    metadata_name: str,
    apikey: str,
    api_base: str = "https://api.openai.com/v1",
    max_chunks=DEFAULT_MAX_CHUNKS,
) -> Index:
    """embedding office powerpoint file

    Args:
        fpath (str): file path
        metadata_name (str): file key
        apikey (str): openai api key
        max_chunks (int, optional): max chunks.

    Returns:
        Index: index
    """
    logger.info(f"call embeddings_word {fpath=}, {metadata_name=}")
    text_splitter = TokenTextSplitter(
        chunk_size=500,
        chunk_overlap=30,
    )
    index = new_store(apikey=apikey, api_base=api_base)
    docs = []
    metadatas = []
    loader = UnstructuredPowerPointLoader(fpath)
    for page, data in enumerate(loader.load_and_split()):
        splits = text_splitter.split_text(data.page_content)
        docs.extend(splits)
        logger.debug(f"embedding {fpath} page {page+1} with {len(splits)} chunks")
        for ichunk, _ in enumerate(splits):
            metadatas.append(
                {"source": f"{metadata_name}#page={page+1}?chunk={ichunk+1}"}
            )

    assert len(docs) <= max_chunks, (
        f"your account limit to parse at most {max_chunks} chunks, ",
        f"but you submit {len(docs)}, consider upgrade your account",
    )

    logger.debug(f"send chunk to LLM embeddings, {fpath=}, {len(docs)} chunks")
    futures: List[Future] = []
    start_at = 0
    n_batch = 5
    while True:
        if start_at >= len(docs):
            break

        end_at = min(start_at + n_batch, len(docs))
        f = thread_executor.submit(
            _embeddings_worker,
            texts=docs[start_at:end_at],
            metadatas=metadatas[start_at:end_at],
            apikey=apikey,
            api_base=api_base,
        )
        futures.append(f)
        start_at = end_at

    index = new_store(apikey=apikey, api_base=api_base)
    for f in futures:
        index.store.merge_from(f.result())

    return index


def embedding_html(
    fpath: str,
    metadata_name: str,
    apikey: str,
    api_base: str = "https://api.openai.com/v1",
    max_chunks=DEFAULT_MAX_CHUNKS,
) -> Index:
    """embedding html file

    Args:
        fpath (str): file path
        metadata_name (str): file key
        apikey (str): openai api key
        max_chunks (int, optional): max chunks.

    Returns:
        Index: index
    """
    logger.debug(f"call embeddings_html {fpath=}, {metadata_name=}")

    loader = BSHTMLLoader(fpath)
    page_data = loader.load()[0]

    text_splitter = TokenTextSplitter(
        chunk_size=500,
        chunk_overlap=30,
    )
    splits = text_splitter.split_text(page_data.page_content)
    assert len(splits) <= max_chunks, f"too many chunks {len(splits)} > {max_chunks}"

    logger.debug(f"send chunk to LLM embeddings, {fpath=}, {len(splits)} chunks")
    futures: List[Future] = []
    start_idx: int = 0
    n_batch = 5
    while True:
        if start_idx >= len(splits):
            break

        end_at = min(start_idx + n_batch, len(splits))
        f = thread_executor.submit(
            _embeddings_worker,
            texts=splits[start_idx:end_at],
            metadatas=[{"key_holder": "val_holder"}] * (end_at - start_idx),
            apikey=apikey,
            api_base=api_base,
        )
        futures.append(f)
        start_idx = end_at

    index = new_store(apikey=apikey, api_base=api_base)
    for f in futures:
        index.store.merge_from(f.result())

    return index


def new_store(apikey: str, api_base: str = "https://api.openai.com/v1") -> Index:
    """
    new FAISS store

    Args:
        apikey (str): openai api key
        api_base (str, optional): openai api base url. Defaults to "https://api.openai.com/v1".

    Returns:
        Index: FAISS index
    """
    # if os.environ.get("OPENAI_API_TYPE") == "azure":
    #     azure_embeddings_deploymentid = prd.OPENAI_AZURE_DEPLOYMENTS[
    #         "embeddings"
    #     ].deployment_id
    #     azure_gpt_deploymentid = prd.OPENAI_AZURE_DEPLOYMENTS["chat"].deployment_id

    #     embedding_model = OpenAIEmbeddings(
    #         openai_api_key=apikey,
    #         client=None,
    #         model="text-embedding-3-small",
    #         deployment=azure_embeddings_deploymentid,
    #     )
    # else:
    logger.debug(f"new faiss store {api_base=}")
    embedding_model = OpenAIEmbeddings(
        api_key=apikey,
        base_url=api_base,
        model="text-embedding-3-small",
    )

    store = FAISS.from_texts([""], embedding_model, metadatas=[{"source": ""}])
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
) -> Tuple[str, Index, List[str]]:
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
        Tuple[str, Index, List[str]]: chatbot_name, index, datasets
    """
    uid = user.uid
    logger.debug(f"call download_chatbot_index {uid=}, {dirpath=}, {chatbot_name=}")

    if chatbot_name == "":
        # download current chatbot name
        resp: Any = None
        try:
            resp = s3cli.get_object(
                bucket_name=prd.OPENAI_S3_CHUNK_CACHE_BUCKET,
                object_name=f"{prd.OPENAI_S3_EMBEDDINGS_PREFIX}/{uid}/chatbot-v2/__CURRENT",
            )
            chatbot_name = resp.data.decode("utf-8")
            logger.debug(f"download current chatbot name {chatbot_name=}")
        finally:
            if resp:
                resp.close()
                resp.release_conn()

    if password:
        objkeys = [
            quote(
                f"{prd.OPENAI_S3_EMBEDDINGS_PREFIX}/{uid}/chatbot-v2/{chatbot_name}.store"
            ),
            quote(
                f"{prd.OPENAI_S3_EMBEDDINGS_PREFIX}/{uid}/chatbot-v2/{chatbot_name}.pkl"
            ),
        ]
    else:
        objkeys = [
            quote(
                f"{prd.OPENAI_S3_EMBEDDINGS_PREFIX}/{uid}/chatbot-share-v2/{chatbot_name}.store"
            ),
            quote(
                f"{prd.OPENAI_S3_EMBEDDINGS_PREFIX}/{uid}/chatbot-share-v2/{chatbot_name}.pkl"
            ),
        ]

    # fix bug for https://github.com/minio/minio-py/pull/1309
    tmp_file = os.path.join(dirpath, "tmp.part.minio")

    for objkey in objkeys:
        s3cli.fget_object(
            bucket_name=prd.OPENAI_S3_CHUNK_CACHE_BUCKET,
            object_name=objkey,
            file_path=os.path.join(dirpath, os.path.basename(objkey)),
            tmp_file_path=tmp_file,
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

    return chatbot_name, index, datasets


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
        chatbot_name, index, datasets = download_chatbot_index(
            dirpath=tmpdir,
            s3cli=s3cli,
            user=user,
            chatbot_name=chatbot_name,
            password=password,
        )

    logger.info(f"succeed restore user {uid=} chain from s3")
    chain = build_user_chain(index, datasets)

    if password:
        logger.info(f"load encrypted user chain, {uid=}, {chatbot_name=}")
        with user_embeddings_chain_mu:
            user_embeddings_chain[uid] = chain
    else:
        logger.info(f"load shared user chain, {uid=}, {chatbot_name=}")
        with user_shared_chain_mu:
            user_shared_chain[uid + chatbot_name] = chain


def save_encrypt_store(
    s3cli: Minio,
    user: prd.UserPermission,
    index: Index,
    name: str,
    password: str,
    datasets: Optional[List[str]] = None,
) -> None:
    """save encrypted store

    Args:
        index (Index): index
        dirpath (str): dirpath
        name (str): name of file, without ext
        password (str): password
        datasets (Optional[List[str]]): datasets, if empty, save as user's datasets,
            if not empty, save as user's chatbot.

    Returns:
        List[str]: saved filepaths
    """
    key = derive_key(password)
    uid = user.uid

    with tempfile.TemporaryDirectory() as tmpdir:
        fpath_prefix = os.path.join(tmpdir, name)
        data = index.serialize()
        with open(f"{fpath_prefix}.store", "wb") as f:
            cipher = AES.new(key, AES.MODE_EAX)
            ciphertext, tag = cipher.encrypt_and_digest(data)
            for x in (cipher.nonce, tag, ciphertext):
                f.write(x)

        fs = [
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
                    f"{prd.OPENAI_S3_EMBEDDINGS_PREFIX}/{uid}/chatbot-v2/{os.path.basename(fpath)}"
                )
            else:
                objkey = quote(
                    f"{prd.OPENAI_S3_EMBEDDINGS_PREFIX}/{uid}/{os.path.basename(fpath)}"
                )

            s3cli.fput_object(
                bucket_name=prd.OPENAI_S3_CHUNK_CACHE_BUCKET,
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
        fpath_prefix = os.path.join(tmpdir, name)
        with open(f"{fpath_prefix}.store", "wb") as f:
            pickle.dump(index.store, f)

        fs = [
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
                f"{prd.OPENAI_S3_EMBEDDINGS_PREFIX}/{uid}/chatbot-share-v2/{os.path.basename(fpath)}"
            )

            s3cli.fput_object(
                bucket_name=prd.OPENAI_S3_CHUNK_CACHE_BUCKET,
                object_name=objkey,
                file_path=fpath,
            )
            logger.info(f"upload plaintext {objkey} to s3")


def load_plaintext_store(dirpath: str, name: str) -> Index:
    """load plaintext store

    Args:
        dirpath: dirpath to store index files
        name: project/file name
    """
    fpath_prefix = os.path.join(dirpath, name)
    with open(f"{fpath_prefix}.store", "rb") as f:
        data = f.read()
        return Index.deserialize(data, api_key=prd.OPENAI_TOKEN)


def load_encrypt_store(dirpath: str, name: str, password: str) -> Index:
    """load encrypted store

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
    data = cipher.decrypt_and_verify(ciphertext, tag)

    return Index.deserialize(data, api_key=prd.OPENAI_TOKEN)
