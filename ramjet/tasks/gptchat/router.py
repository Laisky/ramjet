import asyncio
import base64
import io
import os
import pickle
import re
import tempfile
import threading
import urllib.parse
from typing import Dict, List, Set, Tuple
from urllib.parse import quote

import aiohttp.web
from aiohttp.web_request import FileField
from Crypto.Cipher import AES
from kipp.decorator import timer
from minio import Minio

from ramjet.engines import thread_executor
from ramjet.settings import prd

from .base import logger
from .embedding.embeddings import (
    Index,
    build_user_chain,
    derive_key,
    embedding_pdf,
    load_encrypt_store,
    new_store,
    restore_user_chain,
    save_encrypt_store,
    user_embeddings_chain,
    user_embeddings_chain_mu,
    user_shared_chain_mu,
    user_shared_chain,
    save_plaintext_store,
    download_chatbot_index,
)
from .embedding.query import build_chain, query, setup
from .utils import (
    authenticate_by_appkey as authenticate,
    authenticate_by_appkey_sync as authenticate_sync,
    get_user_by_uid,
)
from .utils import recover

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


def bind_handle(add_route):
    logger.info("bind gpt web handlers")

    # thread_executor.submit(setup)
    setup()

    add_route("", LandingPage)
    add_route("query", Query)
    add_route("files", PDFFiles)
    add_route("ctx{op:(/.*)?}", EmbeddingContext)
    add_route("encrypted-files/{filekey:.*}", EncryptedFiles)


def uid_ratelimiter(user: prd.UserPermission, concurrent=3) -> threading.Semaphore:
    logger.debug(f"uid_ratelimiter: {user.uid=}")
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
            prefix=f"{prd.OPENAI_S3_EMBEDDINGS_PREFIX}/{uid}/",
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

        selected = []
        try:
            if uid not in user_embeddings_chain:
                restore_user_chain(s3cli, user, password)

            if uid in user_embeddings_chain:
                selected = user_embeddings_chain[uid].datasets
        except Exception as err:
            logger.debug(f"failed to restore user chain {uid}, {err=}")

        return aiohttp.web.json_response(
            {
                "datasets": files,
                "selected": selected,
            }
        )

    @recover
    @authenticate
    async def delete(self, user: prd.UserPermission):
        """delete s3 files"""
        data = await self.request.json()
        datasets = data.get("datasets", [])
        if not datasets:
            return aiohttp.web.json_response(
                {"error": "datasets is required"}, status=400
            )

        assert isinstance(datasets, list), "datasets must be a array"

        try:
            ioloop = asyncio.get_event_loop()

            # do not wait task done
            ioloop.run_in_executor(thread_executor, self.delete_files, user, datasets)
        except Exception as e:
            logger.exception(f"failed to delete files {data.get('files', [])}")
            return aiohttp.web.json_response({"error": str(e)}, status=400)

        return aiohttp.web.json_response({"status": "ok"})

    def delete_files(self, user: prd.UserPermission, dataset_names: List[str]):
        uid = user.uid
        for ext in ["store", "index", "pdf"]:
            for dataset_name in dataset_names:
                objkey = f"{prd.OPENAI_S3_EMBEDDINGS_PREFIX}/{uid}/{dataset_name}.{ext}"
                try:
                    s3cli.remove_object(
                        bucket_name=prd.OPENAI_S3_EMBEDDINGS_BUCKET,
                        object_name=objkey,
                    )
                    logger.info(f"deleted file on s3, {objkey=}")
                except Exception as e:
                    logger.exception(f"failed to delete file on s3, {objkey=}")

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
            ioloop.run_in_executor(thread_executor, self.process_file, user, data)
        except Exception as e:
            logger.exception(f"failed to process file {data.get('file', '')}")
            return aiohttp.web.json_response({"error": str(e)}, status=400)
        finally:
            sema.release()

        return aiohttp.web.json_response({"status": "ok"})

    @timer
    def process_file(self, user: prd.UserPermission, data) -> List[str]:
        dataset_name = data.get("file_key", "")
        assert type(dataset_name) == str, "file_key must be string"
        assert dataset_name, "file_key is required"
        assert dataset_name_regex.match(
            dataset_name
        ), "file_key should only contain [a-zA-Z0-9_-]"

        uid = user.uid
        try:
            with user_processing_files_lock:
                fset = user_processing_files.get(uid, set())
                fset.add(dataset_name)
                user_processing_files[uid] = fset

            logger.info(f"process file {dataset_name=} for user {uid=}")
            return self._process_file(user, data)
        finally:
            with user_processing_files_lock:
                user_processing_files[uid].remove(dataset_name)

    def _process_file(self, user: prd.UserPermission, data: Dict) -> List[str]:
        """process user uploaded file by openai embeddings,
        then encrypt and upload to s3
        """
        file = data.get("file", "")
        assert type(file) == FileField, f"file must be FileField, got {type(file)}"

        dataset_name = data.get("file_key", "")
        password = data.get("data_key", "")
        assert type(password) == str, "data_key must be string"
        assert password, "data_key is required"

        uid = user.uid
        objs = []
        encrypted_file_key = (
            f"{prd.OPENAI_S3_EMBEDDINGS_PREFIX}/{uid}/{dataset_name}.pdf"
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
            save_encrypt_store(
                s3cli=s3cli,
                user=user,
                index=index,
                name=dataset_name,
                password=password,
            )

        return objs


class EmbeddingContext(aiohttp.web.View):
    """build private knowledge base by consisit of embedding indices"""

    @recover
    # @authenticate
    async def get(self):
        """
        dispatch request to different handlers by url
        """
        op = self.request.match_info["op"]
        ioloop = asyncio.get_event_loop()
        if op == "/chat":
            return await ioloop.run_in_executor(thread_executor, self.chatbot_query)
        elif op == "/list":
            return await ioloop.run_in_executor(thread_executor, self.list_chatbots)
        elif op == "/share":
            return await ioloop.run_in_executor(
                thread_executor, self.share_chatbot_query
            )
        else:
            raise Exception(f"unknown op {op}")

    def share_chatbot_query(self) -> aiohttp.web.Response:
        """talk with somebody shared private knowledge base"""
        # get uid and chatbot_name from query args
        uid = self.request.query.get("uid", "")
        assert re.match(r"^[a-zA-Z0-9\-_]+$", uid), "uid is required"
        user = get_user_by_uid(uid)

        chatbot_name = self.request.query.get("chatbot_name", "")
        assert re.match(r"^[a-zA-Z0-9\-_]+$", chatbot_name), "chatbot_name is required"

        query = self.request.query.get("q", "").strip()
        assert query, "q is required"

        if uid not in user_shared_chain:
            logger.debug(f"try restore user shared chain from s3 for {uid=}")
            restore_user_chain(s3cli=s3cli, user=user, chatbot_name=chatbot_name)

        sema = uid_ratelimiter(user=user, concurrent=user.n_concurrent)
        try:
            resp, refs = user_shared_chain[uid + chatbot_name].chain(
                {"question": query}
            )
            refs = list(set(refs))
        finally:
            sema.release()

        return aiohttp.web.json_response(
            {
                "text": resp,
                "url": refs,
            }
        )

    @authenticate_sync
    def chatbot_query(self, user: prd.UserPermission) -> aiohttp.web.Response:
        """talk with user's private knowledge base"""
        uid = user.uid
        query = self.request.query.get("q", "").strip()
        assert query, "q is required"

        password = self.request.headers.getone("X-PDFCHAT-PASSWORD")
        assert password, "X-PDFCHAT-PASSWORD is required"

        if uid not in user_embeddings_chain:
            logger.debug(f"try restore user chain from s3 for {uid=}")
            restore_user_chain(s3cli, user, password)

        sema = uid_ratelimiter(user=user, concurrent=user.n_concurrent)
        try:
            resp, refs = user_embeddings_chain[uid].chain({"question": query})
            refs = list(set(refs))
        finally:
            sema.release()

        return aiohttp.web.json_response(
            {
                "text": resp,
                "url": refs,
            }
        )

    @authenticate_sync
    def list_chatbots(self, user: prd.UserPermission) -> aiohttp.web.Response:
        chatbot_names = []
        for obj in s3cli.list_objects(
            bucket_name=prd.OPENAI_S3_EMBEDDINGS_BUCKET,
            prefix=f"{prd.OPENAI_S3_EMBEDDINGS_PREFIX}/{user.uid}/chatbot/",
            recursive=False,
        ):
            if not obj.object_name.endswith(".store"):
                continue

            chatbot_names.append(
                os.path.basename(obj.object_name).removesuffix(".store")
            )

        # get current
        resp = s3cli.get_object(
            bucket_name=prd.OPENAI_S3_EMBEDDINGS_BUCKET,
            object_name=f"{prd.OPENAI_S3_EMBEDDINGS_PREFIX}/{user.uid}/chatbot/__CURRENT",
        )
        current_chatbot = resp.data.decode("utf-8")
        resp.close()
        resp.release_conn()

        return aiohttp.web.json_response(
            {
                "current": current_chatbot,
                "chatbots": chatbot_names,
            }
        )

    @recover
    @authenticate
    async def post(self, user: prd.UserPermission):
        data = await self.request.json()

        ioloop = asyncio.get_event_loop()
        op = self.request.match_info["op"]
        logger.debug(f"post to EmbeddingContext with {op=}")
        if op == "/build":
            return await ioloop.run_in_executor(
                thread_executor,
                self.build_chatbot,
                user,
                data,
            )
        elif op == "/active":
            return await ioloop.run_in_executor(
                thread_executor,
                self.active_chatbot,
                user,
                data,
            )
        elif op == "/share":
            return await ioloop.run_in_executor(
                thread_executor,
                self.share_chatbot,
                user,
                data,
            )
        else:
            raise NotImplementedError(f"unknown op {op}")

    def share_chatbot(
        self, user: prd.UserPermission, data: Dict
    ) -> aiohttp.web.Response:
        """
        share chatbot to other users

        will decrypt user's embeddings store and save it in plain text
        """
        password = data.get("data_key", "")
        assert type(password) == str, "data_key must be string"
        assert password, "data_key is required"

        chatbot_name = data.get("chatbot_name", "")
        assert re.match(r"^[a-zA-Z0-9_-]+$", chatbot_name), "chatbot_name is invalid"

        with tempfile.TemporaryDirectory() as tmpdir:
            index, datasets = download_chatbot_index(
                s3cli=s3cli,
                dirpath=tmpdir,
                user=user,
                chatbot_name=chatbot_name,
                password=password,
            )
            save_plaintext_store(
                s3cli=s3cli,
                user=user,
                index=index,
                datasets=datasets,
                name=chatbot_name,
            )

        logger.info(f"succeed share {user.uid}'s chatbot {chatbot_name}")
        return aiohttp.web.json_response(
            {
                "uid": user.uid,
                "chatbot_name": chatbot_name,
            }
        )

    def active_chatbot(
        self,
        user: prd.UserPermission,
        data: Dict,
    ) -> aiohttp.web.Response:
        """active existed chatbot by name"""
        password = data.get("data_key", "")
        assert type(password) == str, "data_key must be string"
        assert password, "data_key is required"

        chatbot_name = data.get("chatbot_name", "")
        assert re.match(r"^[a-zA-Z0-9_-]+$", chatbot_name), "chatbot_name is invalid"

        restore_user_chain(s3cli, user, password, chatbot_name)

        return aiohttp.web.json_response(
            {"msg": "ok"},
        )

    def build_chatbot(
        self, user: prd.UserPermission, data: Dict
    ) -> aiohttp.web.Response:
        """build chatbot by selected datasets"""
        datasets = data.get("datasets", [])
        assert type(datasets) == list, "datasets must be list"
        assert datasets, "datasets is required"

        chatbot_name = data.get("chatbot_name", "")
        assert re.match(r"^[a-zA-Z0-9_-]+$", chatbot_name), "chatbot_name is invalid"

        password = data.get("data_key", "")
        assert type(password) == str, "data_key must be string"
        assert password, "data_key is required"

        self._build_user_chatbot(
            user=user, password=password, datasets=datasets, chatbot_name=chatbot_name
        )
        return aiohttp.web.json_response(
            {"msg": f"{chatbot_name} build ok"},
        )

    def _build_user_chatbot(
        self,
        user: prd.UserPermission,
        password: str,
        datasets: List[str],
        chatbot_name: str = "default",
    ):
        uid = user.uid
        index = self.load_datasets(uid, datasets, password)
        chain = build_user_chain(user, index, datasets)
        with user_embeddings_chain_mu:
            user_embeddings_chain[uid] = chain

        save_encrypt_store(
            s3cli=s3cli,
            user=user,
            index=index,
            name=chatbot_name,
            password=password,
            datasets=datasets,
        )

    # def save_user_chain(
    #     self,
    #     index: Index,
    #     user: prd.UserPermission,
    #     password: str,
    #     datasets: List[str],
    #     chatbot_name: str = "default",
    # ):
    #     """save user's embedding index to s3

    #     will save three files:
    #         - {chatbot_name}.index: embedding index
    #         - {chatbot_name}.store: embedding store
    #         - {chatbot_name}.pkl: selected datasets
    #     """
    #     with tempfile.TemporaryDirectory() as tmpdir:
    #         # encrypt and upload origin pdf file
    #         fs = save_encrypt_store(
    #             s3cli=s3cli,
    #             user=user,
    #             index=index,
    #             chatbot_name=chatbot_name,
    #             password=password,
    #             datasets=datasets,
    #         )

    #     logger.info(f"succeed to upload qa chat store {uid=}")

    def load_datasets(self, uid: str, datasets: List[str], password: str) -> Index:
        """load datasets from s3"""
        store = new_store()
        for dataset in datasets:
            idx_key = f"{prd.OPENAI_S3_EMBEDDINGS_PREFIX}/{uid}/{dataset}.index"
            store_key = f"{prd.OPENAI_S3_EMBEDDINGS_PREFIX}/{uid}/{dataset}.store"

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
