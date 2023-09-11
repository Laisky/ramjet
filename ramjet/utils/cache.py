import time
from typing import Any
from dataclasses import dataclass
from threading import Thread, RLock

from .log import logger

@dataclass
class CacheItem:
    key: str
    value: Any
    expire_at_epoch: float

logger = logger.getChild("cache")

class Cache:
    def __init__(self):
        self.cache_lock = RLock()
        self.cache_store = dict()
        self.__closed = False
        self._cleaner = Thread(target=self._cache_cleaner, daemon=True).start()

    def close(self):
        logger.debug("close")
        with self.cache_lock:
            self.__closed = True

    def save_cache(self, key: str, value: Any, expire_at: float = 0):
        """
        Args:
            key (str): The key to save the cache under
            value (Any): The value to save
            expire_at (int, optional): The epoch time to expire the cache. Defaults to 0.
        """
        if not expire_at:
            expire_at = time.time() + 60

        with self.cache_lock:
            logger.debug(f"save cache {key=}, expire_at={expire_at-time.time():.2f}")
            self.cache_store[key] = CacheItem(key, value, expire_at)

    def get_cache(self, key: str) -> Any:
        """
        Args:
            key (str): The key to get the cache for

        Returns:
            Any: The value of the cache if it exists and is not expired, otherwise None
        """
        with self.cache_lock:
            if key in self.cache_store:
                cache_item = self.cache_store[key]
                if cache_item.expire_at_epoch > time.time():
                    logger.debug(f"hit cache {key=}")
                    return cache_item.value
                else:
                    del self.cache_store[key]

        logger.debug(f"miss cache {key=}")
        return None

    def _cache_cleaner(self):
        while True:
            time.sleep(60)
            with self.cache_lock:
                if self.__closed:
                    return

                for key, cache_item in self.cache_store.items():
                    if cache_item.expire_at_epoch < time.time():
                        logger.debug(f"remove expired cache {key=}")
                        del self.cache_store[key]
