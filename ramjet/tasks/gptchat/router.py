import asyncio
import base64
import os
import pickle
import re
import tempfile
import threading
import urllib.parse
from typing import Dict, List, Tuple, Set
from urllib.parse import quote

import aiohttp.web
from aiohttp.web_request import FileField
from Crypto.Cipher import AES
from minio import Minio
from kipp.decorator import timer

from ramjet.engines import thread_executor
from ramjet.settings import prd
from .utils import authenticate_by_appkey as authenticate, recover
from .base import logger
from .embedding.embeddings import (
    Index,
    derive_key,
    embedding_pdf,
    load_encrypt_store,
    new_store,
    save_encrypt_store,
    restore_user_chain,
    user_embeddings_chain,
    build_user_chain,
)
from .embedding.query import build_chain, query, setup


# limit concurrent process files by uid
user_prcess_file_sema_lock = threading.RLock()
user_prcess_file_sema: Dict[str, threading.Semaphore] = {}

# track processing files by uid
# {uid: [filekeys]}
user_processing_files_lock = threading.RLock()
user_processing_files: Dict[str, Set[str]] = {}

# valid dataset name
dataset_name_regex = re.compile(r"^[a-zA-Z0-9_-]+$")

s3cli: Minio = Minio(
    endpoint=prd.S3_MINIO_ADDR,
    access_key=prd.S3_KEY,
    secret_key=prd.S3_SECRET,
    secure=False,
)


def uid_ratelimiter(user: prd.UserPermission, concurrent=3) -> threading.Semaphore:
    uid = user.uid
    limit = user.n_concurrent or concurrent
    sema = user_prcess_file_sema.get(uid)
    if not sema:
        with user_prcess_file_sema_lock:
            sema = user_prcess_file_sema.get(uid)
            if not sema:
                sema = threading.Semaphore(limit)
                user_prcess_file_sema[uid] = sema

    if not sema.acquire(blocking=False):
        raise aiohttp.web.HTTPTooManyRequests(
            reason=f"current user {uid} can only process {limit} files concurrently"
        )

    return sema


def uid_method_ratelimiter(concurrent=3):
    """rate limit by uid

    the first argument of the decorated function must be uid
    """

    def decorator(func):
        def wrapper(self, user: prd.UserPermission, *args, **kwargs):
            sema = uid_ratelimiter(user, concurrent)

            try:
                return func(self, user, *args, **kwargs)
            finally:
                sema.release()

        return wrapper

    return decorator


def bind_handle(add_route):
    logger.info("bind gpt web handlers")

    # thread_executor.submit(setup)
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

    @recover
    @authenticate
    async def get(self, user: prd.UserPermission):
        project = self.request.query.get("p")
        question = urllib.parse.unquote(self.request.query.get("q", ""))

        if not project:
            return aiohttp.web.Response(text="p is required", status=400)
        if not question:
            return aiohttp.web.Response(text="q is required", status=400)

        ioloop = asyncio.get_running_loop()
        resp = await ioloop.run_in_executor(
            thread_executor, self.query, user, project, question
        )

        return aiohttp.web.json_response(resp._asdict())

    @uid_method_ratelimiter()
    def query(self, user: prd.UserPermission, project: str, question: str):
        return query(project, question)


class EncryptedFiles(aiohttp.web.View):
    @recover
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
            response.headers["WWW-Authenticate"] = 'Basic realm="My Realm"'
            return response

        # download pdf file to temp dir
        filekey = self.request.match_info["filekey"]
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
                    response.headers["WWW-Authenticate"] = 'Basic realm="My Realm"'
                    return response

        # return decrypted pdf file
        return aiohttp.web.Response(body=data, content_type="application/pdf")


class PDFFiles(aiohttp.web.View):
    """build private dataset by embedding pdf files

    Returns:
        aiohttp.web.Response -- json response, contains file status
        ::
            [
                {"name": "file1", "status": "done"},
                {"name": "file2", "status": "processing", "progress": 75},
            ]
    """

    @recover
    @authenticate
    async def get(self, user: prd.UserPermission):
        """list s3 files"""
        uid = user.uid
        files: List[Dict] = []
        password = self.request.headers.get("X-PDFCHAT-PASSWORD", "")

        # for test
        # files.append({"name": "test.pdf", "status": "processing", "progress": 75})

        for obj in s3cli.list_objects(
            bucket_name=prd.OPENAI_S3_EMBEDDINGS_BUCKET,
            prefix=f"{prd.OPENAI_S3_EMBEDDINGS_prefix}/{uid}/",
            recursive=False,
        ):
            if obj.object_name.endswith(".store"):
                files.append(
                    {
                        "name": os.path.basename(
                            obj.object_name.removesuffix(".store")
                        ),
                        "status": "done",
                    }
                )

        with user_processing_files_lock:
            files.extend(
                [
                    {"name": f, "status": "processing", "progress": 75}
                    for f in user_processing_files.get(uid, [])
                ]
            )

        if uid not in user_embeddings_chain:
            restore_user_chain(s3cli, user, password)

        selected = []
        if uid in user_embeddings_chain:
            selected = user_embeddings_chain[uid].datasets

        return aiohttp.web.json_response(
            {
                "datasets": files,
                "selected": selected,
            }
        )

    @recover
    @authenticate
    async def post(self, user: prd.UserPermission):
        """Upload pdf file by form"""
        uid = user.uid
        data = await self.request.post()

        sema = uid_ratelimiter(user, 3)
        try:
            ioloop = asyncio.get_event_loop()

            # do not wait task done
            ioloop.run_in_executor(thread_executor, self.process_file, uid, data)
        except Exception as e:
            logger.exception(f"failed to process file {data.get('file', '')}")
            return aiohttp.web.json_response({"error": str(e)}, status=400)
        finally:
            sema.release()

        return aiohttp.web.json_response({"status": "ok"})

    @timer
    def process_file(self, uid, data) -> List[str]:
        dataset_name = data.get("file_key", "")
        assert type(dataset_name) == str, "file_key must be string"
        assert dataset_name, "file_key is required"
        assert dataset_name_regex.match(
            dataset_name
        ), "file_key should only contain [a-zA-Z0-9_-]"

        try:
            with user_processing_files_lock:
                fset = user_processing_files.get(uid, set())
                fset.add(dataset_name)
                user_processing_files[uid] = fset

            logger.info(f"process file {dataset_name=} for user {uid=}")
            return self._process_file(uid, data)
        finally:
            with user_processing_files_lock:
                user_processing_files[uid].remove(dataset_name)

    def _process_file(self, uid, data) -> List[str]:
        """process user uploaded file by openai embeddings,
        then encrypt and upload to s3
        """
        file = data.get("file", "")
        assert type(file) == FileField, f"file must be FileField, got {type(file)}"

        dataset_name = data.get("file_key", "")
        password = data.get("data_key", "")
        assert type(password) == str, "data_key must be string"
        assert password, "data_key is required"

        objs = []
        encrypted_file_key = (
            f"{prd.OPENAI_S3_EMBEDDINGS_prefix}/{uid}/{dataset_name}.pdf"
        )

        logger.info(f"process file {file.filename} for {uid}")
        with tempfile.TemporaryDirectory() as tmpdir:
            # copy uploaded file to temp dir
            source_fpath = os.path.join(tmpdir, file.filename)
            with open(source_fpath, "wb") as fp:
                total_size = 0
                chunk_size = 1024 * 1024  # 1MB
                while True:
                    chunk = file.file.read(chunk_size)
                    if not chunk:
                        break

                    total_size += len(chunk)
                    if total_size > prd.OPENAI_EMBEDDING_FILE_SIZE_LIMIT:
                        raise Exception(
                            f"file size should not exceed {prd.OPENAI_EMBEDDING_FILE_SIZE_LIMIT} bytes"
                        )

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
                    ciphertext, tag = cipher.encrypt_and_digest(
                        pickle.dumps(src_fp.read())
                    )
                    [encrypted_fp.write(x) for x in (cipher.nonce, tag, ciphertext)]
                    encrypted_fp.flush()

                    objs.append(encrypted_file_key)
                    s3cli.fput_object(
                        bucket_name=prd.OPENAI_S3_EMBEDDINGS_BUCKET,
                        object_name=encrypted_file_key,
                        file_path=encrypted_file_path,
                    )
            logger.info(
                f"succeed upload encrypted source file {encrypted_file_key} to s3"
            )

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


class EmbeddingContext(aiohttp.web.View):
    """build private knowledge base by consisit of embedding indices"""

    @recover
    @authenticate
    async def get(self, user: prd.UserPermission):
        """talk with user's private knowledge base"""
        uid = user.uid
        ioloop = asyncio.get_event_loop()
        try:
            query = self.request.query.get("q", "").strip()
            assert query, "q is required"

            password = self.request.headers.getone("X-PDFCHAT-PASSWORD")
            if uid not in user_embeddings_chain:
                logger.debug(f"try restore user chain from s3 for {uid=}")
                await ioloop.run_in_executor(
                    thread_executor, restore_user_chain, s3cli, user, password
                )
        except Exception as e:
            logger.exception(f"failed to get context for {uid}")
            return aiohttp.web.json_response({"error": str(e)}, status=400)

        resp, refs = await ioloop.run_in_executor(
            thread_executor, self.query, user, query
        )

        return aiohttp.web.json_response(
            {
                "text": resp,
                "url": refs,
            }
        )

    @uid_method_ratelimiter(concurrent=1)
    def query(self, user: prd.UserPermission, query: str) -> Tuple[str, List[str]]:
        uid = user.uid
        resp, refs = user_embeddings_chain[uid].chain({"question": query})
        return resp, list(set(refs))

    @recover
    @authenticate
    async def post(self, user: prd.UserPermission):
        """build context by selected datasets"""
        try:
            uid = user.uid
            data = await self.request.json()
            datasets = data.get("datasets", [])
            assert type(datasets) == list, "datasets must be list"
            assert datasets, "datasets is required"

            password = data.get("data_key", "")
            assert type(password) == str, "data_key must be string"
            assert password, "data_key is required"

            ioloop = asyncio.get_event_loop()
            await ioloop.run_in_executor(
                thread_executor, self._build_user_chatbot, user, password, datasets
            )
        except Exception as e:
            logger.exception(f"failed to parse request body")
            return aiohttp.web.json_response({"error": str(e)}, status=400)

        return aiohttp.web.json_response(
            {"msg": "ok"},
        )

    def _build_user_chatbot(
        self, user: prd.UserPermission, password: str, datasets: List[str]
    ):
        uid = user.uid
        index = self.load_datasets(uid, datasets, password)
        build_user_chain(user, index, datasets)
        self.save_user_chain(index, uid, password, datasets)

    def save_user_chain(
        self, index: Index, uid: str, password: str, datasets: List[str]
    ):
        """save user's embedding index to s3"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # encrypt and upload origin pdf file
            fs = save_encrypt_store(
                index=index,
                dirpath=tmpdir,
                name="qachat",
                password=password,
            )

            # dump selected datasets
            datasets_fname = os.path.join(tmpdir, "datasets.pkl")
            fs.append(datasets_fname)
            with open(datasets_fname, "wb") as datasets_fp:
                pickle.dump(datasets, datasets_fp)

            logger.debug(f"try to upload embedding chat store {uid=}")
            for fpath in fs:
                objkey = quote(
                    f"{prd.OPENAI_S3_EMBEDDINGS_prefix}/{uid}/chatbot/default/{os.path.basename(fpath)}"
                )
                s3cli.fput_object(
                    bucket_name=prd.OPENAI_S3_EMBEDDINGS_BUCKET,
                    object_name=objkey,
                    file_path=fpath,
                )
                logger.debug(f"upload {objkey} to s3")

        logger.info(f"succeed to upload qa chat store {uid=}")

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
