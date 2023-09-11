import codecs
import hashlib
import os
import pickle
import re
import tempfile
import threading
import time
import base64
import tempfile
from typing import List
from textwrap import dedent
from concurrent.futures import Future

from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.document_loaders import (
    BSHTMLLoader,
    Docx2txtLoader,
    PyPDFLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.schema.document import Document
from langchain.text_splitter import (
    CharacterTextSplitter,
    MarkdownTextSplitter,
    TokenTextSplitter,
)

from ramjet.engines import thread_executor


def summary_content(b64content: str, ext: str, apikey: str = None) -> str:
    """Summarize the content of a document.

    Args:
        b64content: The base64 encoded content of the document.
        ext: The extension of the document.
            should be one of: .docx, .pptx, .pdf, .html, .md, .txt

    Returns:
        The summary of the document.
    """
    # text_splitter = CharacterTextSplitter(
    #     chunk_size=500, chunk_overlap=30, separator="\n"
    # )
    text_splitter = TokenTextSplitter(
        chunk_size=3000,
        chunk_overlap=30,
    )

    # write to file
    file_content = base64.b64decode(b64content)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfile = os.path.join(tmpdir, "tmpfile")
        with open(tmpfile, "wb") as f:
            f.write(file_content)

        docus: List[Document]
        if ext == ".docx":
            loader = UnstructuredWordDocumentLoader(tmpfile)
            docus = loader.load_and_split(text_splitter=text_splitter)
        elif ext == ".pptx":
            loader = UnstructuredPowerPointLoader(tmpfile)
            docus = loader.load_and_split(text_splitter=text_splitter)
        elif ext == ".pdf":
            loader = PyPDFLoader(tmpfile)
            docus = loader.load_and_split(text_splitter=text_splitter)
        elif ext == ".html":
            loader = BSHTMLLoader(tmpfile)
            docus = loader.load_and_split(text_splitter=text_splitter)
        elif ext == ".md":
            docus = MarkdownTextSplitter(
                chunk_size=500, chunk_overlap=50
            ).create_documents([file_content.decode("utf-8")])
        elif ext == ".txt":
            docus = text_splitter.create_documents([file_content.decode("utf-8")])
        else:
            raise ValueError(f"Unsupported extension: {ext}")

    return _summary_by_mapreduce(docus, apikey=apikey)


def _summary_by_mapreduce(docus: List[Document], apikey: str = None) -> str:
    """Summarize a list of documents using mapreduce.

    Args:
        docus: A list of documents.

    Returns:
        The summary of the documents.
    """

    summary: str = ""
    # map
    fs: List[Future] = []
    for docu in docus:
        fs.append(thread_executor.submit(summary_docu, docu, apikey=apikey))
    for f in fs:
        summary += f"* {f.result()}\n"

    # reduce
    query = dedent(
        f"""
        The following is set of summaries:

        {summary}

        Take these and distill it into a final, consolidated summary of the main themes.
        Helpful Answer:
        """
    )
    return query

    # reduce by go-ramjet, do not use it for now

    # apikey = apikey or os.environ["OPENAI_API_KEY"]
    # llm = ChatOpenAI(
    #     client=None,
    #     openai_api_key=apikey,
    #     model="gpt-3.5-turbo",
    #     temperature=0,
    #     max_tokens=1000,
    #     streaming=False,
    # )

    # return llm.predict(query)


def summary_docu(docu: Document, apikey: str = None) -> str:
    """Summarize a document.

    Args:
        docu: A document.
        apikey: The openai api key.

    Returns:
        The summary of the document.
    """
    apikey = apikey or os.environ["OPENAI_API_KEY"]
    llm = ChatOpenAI(
        client=None,
        openai_api_key=apikey,
        model="gpt-3.5-turbo",
        temperature=0,
        max_tokens=500,
        streaming=False,
    )

    query = dedent(
        f"""
        Write a concise summary of the following content between "@>>>>>" and "@<<<<<",
        just response the summary text in a single short line, just contains necessary key informations,
        do not contains any other words:

        @>>>>>
        {docu.page_content}
        @<<<<<

        CONCISE SUMMARY:
    """
    )
    return llm.predict(query)
