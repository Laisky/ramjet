import asyncio
import threading
import base64
import os
import pickle
import re
import tempfile
import urllib.parse
from collections import namedtuple
from typing import Dict, List, Tuple
from urllib.parse import quote

import aiohttp.web
from aiohttp.web_request import FileField
from Crypto.Cipher import AES
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from minio import Minio
from minio.error import S3Error

from ramjet.engines import thread_executor
from ramjet.settings import prd

from .auth import authenticate_by_appkey as authenticate
from .base import logger
from .embedding.embeddings import (
    Index,
    derive_key,
    embedding_pdf,
    load_encrypt_store,
    new_store,
    save_encrypt_store,
)
from .embedding.query import build_chain, query, setup

s3cli: Minio = Minio(
    endpoint=prd.S3_MINIO_ADDR,
    access_key=prd.S3_KEY,
    secret_key=prd.S3_SECRET,
    secure=False,
)


dataset_name_regex = re.compile(r"^[a-zA-Z0-9_-]+$")


def bind_handle(add_route):
    logger.info("bind gpt web handlers")
    setup()

    add_route("", LandingPage)
    add_route("query", Query)
    add_route("files", PDFFiles)
    add_route("ctx", EmbeddingContext)
    add_route("encrypted-files/{filekey:.*}", EncryptedFiles)


class LandingPage(aiohttp.web.View):
    async def get(self):
        return aiohttp.web.Response(text="welcome to gptchat")


class Query(aiohttp.web.View):
    """query by pre-embedded pdf files"""

    async def get(self):
        project = self.request.query.get("p")
        question = urllib.parse.unquote(self.request.query.get("q", ""))

        if not project:
            return aiohttp.web.Response(text="p is required", status=400)
        if not question:
            return aiohttp.web.Response(text="q is required", status=400)

        resp = await query(project, question)
        return aiohttp.web.json_response(resp._asdict())


class EncryptedFiles(aiohttp.web.View):

    async def get(self):
        # https://uid:password@fikekey.pdf
        # get uid and password from request basic auth
        auth_header = self.request.headers.get("Authorization")
        if auth_header:
            encoded_credentials = auth_header.split(" ")[1]
            decoded_credentials = base64.b64decode(encoded_credentials).decode("utf-8")
            _, password = decoded_credentials.split(":")
        else:
            response = aiohttp.web.Response(status=401)
            response.headers['WWW-Authenticate'] = 'Basic realm="My Realm"'
            return response

        # download pdf file to temp dir
        filekey =  self.request.match_info["filekey"]
        filename = os.path.basename(filekey)
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, filename)
            s3cli.fget_object(
                bucket_name=prd.OPENAI_S3_EMBEDDINGS_BUCKET,
                object_name=f"{filekey}",
                file_path=fpath,
                request_headers={"Cache-Control": "no-cache"},
            )

            # decrypt pdf file
            with open(fpath, "rb") as f:
                try:
                    key = derive_key(password)
                    nonce, tag, ciphertext = [f.read(x) for x in (16, 16, -1)]
                    cipher = AES.new(key, AES.MODE_EAX, nonce)
                    data = cipher.decrypt_and_verify(ciphertext, tag)
                except Exception:
                    logger.exception(f"failed to decrypt file {filekey}, ask to retry")
                    response = aiohttp.web.Response(status=401)
                    response.headers['WWW-Authenticate'] = 'Basic realm="My Realm"'
                    return response


        # return decrypted pdf file
        return aiohttp.web.Response(body=data, content_type="application/pdf")


user_prcess_file_sema_lock = threading.RLock()
user_prcess_file_sema: Dict[str, threading.Semaphore] = {}

def uid_limitrate(concurrent=1):
    """rate limit by uid

    the first argument of the decorated function must be uid
    """
    def decorator(func):
        def wrapper(uid: str, *args, **kwargs):
            limit = prd.OPENAI_PRIVATE_EMBEDDINGS_USER_LIMIT.get(uid, concurrent)

            sema = user_prcess_file_sema.get(uid)
            if not sema:
                with user_prcess_file_sema_lock:
                    sema = user_prcess_file_sema.get(uid)
                    if not sema:
                        sema = threading.Semaphore(concurrent)
                        user_prcess_file_sema[uid] = sema

            if not sema.acquire(blocking=False):
                raise aiohttp.web.HTTPTooManyRequests(
                    reason=f"current user {uid} can only process {limit} files concurrently"
                )

            try:
                return func(uid, *args, **kwargs)
            finally:
                sema.release()

        return wrapper
    return decorator

class PDFFiles(aiohttp.web.View):
    """build private dataset by embedding pdf files"""

    @authenticate
    async def get(self, uid):
        """list s3 files"""
        objs = []
        for obj in s3cli.list_objects(
            bucket_name=prd.OPENAI_S3_EMBEDDINGS_BUCKET,
            prefix=f"{prd.OPENAI_S3_EMBEDDINGS_prefix}/{uid}",
            recursive=True,
        ):
            if obj.object_name.endswith(".store"):
                objs.append(os.path.basename(obj.object_name.removesuffix(".store")))

        return aiohttp.web.json_response(
            {
                "files": objs,
            }
        )

    @authenticate
    async def post(self, uid):
        """Upload pdf file by form"""
        data = await self.request.post()

        try:
            ioloop = asyncio.get_event_loop()
            objs = await ioloop.run_in_executor(
                thread_executor, self.process_file, uid, data
            )
        except Exception as e:
            logger.exception(f"failed to process file {data.get('file', '')}")
            return aiohttp.web.json_response({"error": str(e)}, status=400)

        return aiohttp.web.json_response(
            {
                "file": objs,
            }
        )

    @uid_limitrate(concurrent=1)
    def process_file(self, uid, data) -> List[str]:
        # check locks
        sema = user_prcess_file_sema.get(uid, threading.Semaphore(1))

        file = data.get("file", "")
        assert type(file) == FileField, f"file must be FileField, got {type(file)}"

        dataset_name = data.get("file_key", "")
        assert type(dataset_name) == str, "file_key must be string"
        assert dataset_name, "file_key is required"
        assert dataset_name_regex.match(
            dataset_name
        ), "file_key should only contain [a-zA-Z0-9_-]"

        password = data.get("data_key", "")
        assert type(password) == str, "data_key must be string"
        assert password, "data_key is required"

        objs = []
        encrypted_file_key = f"{prd.OPENAI_S3_EMBEDDINGS_prefix}/{uid}/{dataset_name}.pdf"

        logger.info(f"process file {file.filename} for {uid}")
        with tempfile.TemporaryDirectory() as tmpdir:
            # copy uploaded file to temp dir
            source_fpath = os.path.join(tmpdir, file.filename)
            with open(source_fpath, "wb") as fp:
                total_size = 0
                chunk_size = 1024 * 1024 # 1MB
                while True:
                    chunk = file.file.read(chunk_size)
                    if not chunk:
                        break

                    total_size += len(chunk)
                    if total_size > prd.OPENAI_EMBEDDING_FILE_SIZE_LIMIT:
                        raise Exception(f"file size should not exceed {prd.OPENAI_EMBEDDING_FILE_SIZE_LIMIT} bytes")

                    fp.write(chunk)

            metadata_name = f"{prd.OPENAI_EMBEDDING_REF_URL_PREFIX}{encrypted_file_key}"
            index = embedding_pdf(fp.name, metadata_name)

            # encrypt and upload origin pdf file
            encrypted_file_path = os.path.join(tmpdir, dataset_name + ".pdf")
            logger.debug(f"try to upload {encrypted_file_path}")
            with open(source_fpath, "rb") as src_fp:
                with open(encrypted_file_path, "wb") as encrypted_fp:
                    key = derive_key(password)
                    cipher = AES.new(key, AES.MODE_EAX)
                    ciphertext, tag = cipher.encrypt_and_digest(pickle.dumps(src_fp.read()))
                    [encrypted_fp.write(x) for x in (cipher.nonce, tag, ciphertext)]
                    encrypted_fp.flush()

                    objs.append(encrypted_file_key)
                    s3cli.fput_object(
                        bucket_name=prd.OPENAI_S3_EMBEDDINGS_BUCKET,
                        object_name=encrypted_file_key,
                        file_path=encrypted_file_path,
                    )
            logger.info(f"succeed upload encrypted source file {encrypted_file_key} to s3")

            # encrypt and upload index
            files = save_encrypt_store(
                index,
                dirpath=tmpdir,
                name=dataset_name,
                password=password,
            )
            for fpath in files:
                objkey = quote(
                    f"{prd.OPENAI_S3_EMBEDDINGS_prefix}/"
                    + uid
                    + fpath.removeprefix(tmpdir)
                )
                logger.debug(f"try upload {objkey} to s3")
                objs.append(objkey)
                with open(fpath, "rb") as encrypted_fp:
                    s3cli.fput_object(
                        bucket_name=prd.OPENAI_S3_EMBEDDINGS_BUCKET,
                        object_name=objkey,
                        file_path=fpath,
                    )
                logger.info(f"succeed upload {objkey} to s3")

        return objs


UserChain = namedtuple("UserChain", ["chain", "index"])
user_embeddings_chain: Dict[str, UserChain] = {}  # uid -> UserChain


class EmbeddingContext(aiohttp.web.View):
    """build private knowledge base by consisit of embedding indices"""

    @authenticate
    async def get(self, uid):
        """talk with user's private knowledge base"""
        try:
            assert user_embeddings_chain.get(uid), "there is no context for this user"

            query = self.request.query.get("q", "").strip()
            assert query, "q is required"
        except Exception as e:
            logger.exception(f"failed to get context for {uid}")
            return aiohttp.web.json_response({"error": str(e)}, status=400)

        ioloop = asyncio.get_event_loop()
        resp, refs = await ioloop.run_in_executor(
            thread_executor, self.query, uid, query
        )

        return aiohttp.web.json_response(
            {
                "text": resp,
                "url": refs,
            }
        )

    def query(self, uid: str, query: str) -> Tuple[str, List[str]]:
        resp, refs = user_embeddings_chain[uid].chain({"question": query})
        return resp, list(set(refs))

    @authenticate
    async def post(self, uid):
        """build context by selected datasets"""
        try:
            data = await self.request.json()
            datasets = data.get("datasets", [])
            assert type(datasets) == list, "datasets must be list"
            assert datasets, "datasets is required"

            data_key = data.get("data_key", "")
            assert type(data_key) == str, "data_key must be string"
            assert data_key, "data_key is required"

            ioloop = asyncio.get_event_loop()
            index = await ioloop.run_in_executor(
                thread_executor, self.load_datasets, uid, datasets, data_key
            )
            await ioloop.run_in_executor(
                thread_executor, self.build_user_chain, uid, index
            )
        except Exception as e:
            logger.exception(f"failed to parse request body")
            return aiohttp.web.json_response({"error": str(e)}, status=400)

        return aiohttp.web.json_response(
            {"msg": "ok"},
        )

    def build_user_chain(self, uid: str, index: Index):
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
            llm = ChatOpenAI(
                client=None,
                # model_name="gpt-3.5-turbo",
                temperature=0,
                max_tokens=1000,
                streaming=False,
            )

        user_embeddings_chain[uid] = UserChain(
            chain=build_chain(llm, index.store),
            index=index,
        )

    def load_datasets(self, uid: str, datasets: List[str], password: str) -> Index:
        """load datasets from s3"""
        store = new_store()
        for dataset in datasets:
            idx_key = f"{prd.OPENAI_S3_EMBEDDINGS_prefix}/{uid}/{dataset}.index"
            store_key = f"{prd.OPENAI_S3_EMBEDDINGS_prefix}/{uid}/{dataset}.store"

            with tempfile.TemporaryDirectory() as tmpdir:
                idx_path = os.path.join(tmpdir, idx_key)
                store_path = os.path.join(tmpdir, store_key)

                logger.debug(f"load dataset {dataset} from {tmpdir}/{uid}")
                s3cli.fget_object(
                    bucket_name=prd.OPENAI_S3_EMBEDDINGS_BUCKET,
                    object_name=idx_key,
                    file_path=idx_path,
                    # request_headers={"Cache-Control": "no-cache"},

                )
                s3cli.fget_object(
                    bucket_name=prd.OPENAI_S3_EMBEDDINGS_BUCKET,
                    object_name=store_key,
                    file_path=store_path,
                    # request_headers={"Cache-Control": "no-cache"},

                )

                store_part = load_encrypt_store(
                    dirpath=os.path.dirname(idx_path),
                    name=dataset,
                    password=password,
                )

            store.store.merge_from(store_part.store)

        logger.info(f"build private knowledge base for {uid}")
        return store
