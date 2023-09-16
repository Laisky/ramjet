# import datetime
import io
import pickle
import time
import gzip
from dataclasses import dataclass
from threading import RLock, Thread
from typing import Any, Dict, Optional

from minio import Minio
# from minio.retention import GOVERNANCE, Retention

from ramjet.settings import prd

from .log import logger


@dataclass
class CacheItem:
    key: str
    value: Any
    expire_at_epoch: float


logger = logger.getChild("cache")


class Cache:
    def __init__(self, s3cli: Minio):
        self._cache_lock = RLock()
        self._local_cache_store: Dict[str, CacheItem] = dict()
        self._remote_cache_store = s3cli
        self.__closed = False

        Thread(target=self._cache_cleaner, daemon=True).start()

    def close(self):
        logger.debug("close")
        with self._cache_lock:
            self.__closed = True

    def save_cache(self, key: str, value: Any, expire_at: float = 0):
        """
        Args:
            key (str): The key to save the cache under
            value (Any): The value to save
            expire_at (int, optional): The epoch time to expire the cache. Defaults to 0.
        """
        if not expire_at:
            expire_at = time.time() + 3600 * 24

        item = CacheItem(key, value, expire_at)

        # save to local cache first
        with self._cache_lock:
            self._local_cache_store[key] = item

        # then save to remote cache
        s3key = f"{prd.OPENAI_S3_CHUNK_CACHE_PREFIX}/{key}"
        data = gzip.compress(pickle.dumps(item))
        self._remote_cache_store.put_object(
            bucket_name=prd.OPENAI_S3_CHUNK_CACHE_BUCKET,
            object_name=s3key,
            data=io.BytesIO(data),
            length=len(data),
        )

        logger.debug(f"save cache {key=}, ttl={(expire_at-time.time())/3600:.2f}hr")

    def get_cache(self, key: str) -> Any:
        """
        Args:
            key (str): The key to get the cache for

        Returns:
            Any: The value of the cache if it exists and is not expired, otherwise None
        """
        # load from local cache first
        with self._cache_lock:
            if key in self._local_cache_store:
                cache_item = self._local_cache_store[key]
                if cache_item.expire_at_epoch > time.time():
                    logger.debug(f"hit local cache {key=}")
                    return cache_item.value
                else:
                    del self._local_cache_store[key]

        # then load from remote cache
        data: Optional[CacheItem] = None
        s3key = f"{prd.OPENAI_S3_CHUNK_CACHE_PREFIX}/{key}"
        response: Any = None
        try:
            response = self._remote_cache_store.get_object(
                bucket_name=prd.OPENAI_S3_CHUNK_CACHE_BUCKET,
                object_name=s3key,
            )
            data = pickle.loads(gzip.decompress(response.read()))
        except Exception:
            return
        finally:
            if response:
                response.close()
                response.release_conn()

        if data and data.expire_at_epoch < time.time():
            data = None
            self._remote_cache_store.remove_object(
                bucket_name=prd.OPENAI_S3_CHUNK_CACHE_BUCKET,
                object_name=s3key,
            )

        if not data:
            logger.debug(f"miss cache {key=}")
            return None

        with self._cache_lock:
            logger.debug(f"hit remote cache {s3key=}")
            self._local_cache_store[key] = data
            return data.value

    def _cache_cleaner(self):
        while True:
            time.sleep(60)
            with self._cache_lock:
                if self.__closed:
                    return

                for key, cache_item in self._local_cache_store.items():
                    if cache_item.expire_at_epoch < time.time():
                        logger.debug(f"remove expired cache {key=}")
                        del self._local_cache_store[key]
