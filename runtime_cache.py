import asyncio
import time
from collections import OrderedDict
from copy import deepcopy
from typing import Awaitable, Callable, Generic, TypeVar

T = TypeVar("T")


class TTLCache(Generic[T]):
    """Small in-process TTL cache for transient repeated work."""

    def __init__(self, ttl_seconds: float, max_entries: int = 128, time_func: Callable[[], float] | None = None):
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._time_func = time_func or time.monotonic
        self._store: OrderedDict[object, tuple[float, T]] = OrderedDict()
        self._inflight: dict[object, asyncio.Task[T]] = {}
        self._lock = asyncio.Lock()

    def _evict_expired(self) -> None:
        now = self._time_func()
        expired_keys = [key for key, (expires_at, _) in self._store.items() if expires_at <= now]
        for key in expired_keys:
            self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()
        self._inflight.clear()

    def get(self, key: object) -> T | None:
        self._evict_expired()
        entry = self._store.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if expires_at <= self._time_func():
            self._store.pop(key, None)
            return None
        self._store.move_to_end(key)
        return deepcopy(value)

    def set(self, key: object, value: T) -> T:
        self._evict_expired()
        self._store[key] = (self._time_func() + self.ttl_seconds, deepcopy(value))
        self._store.move_to_end(key)
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)
        return deepcopy(value)

    async def get_or_set(self, key: object, producer: Callable[[], Awaitable[T]]) -> T:
        cached = self.get(key)
        if cached is not None:
            return cached

        owner = False
        async with self._lock:
            cached = self.get(key)
            if cached is not None:
                return cached

            inflight = self._inflight.get(key)
            if inflight is None:
                inflight = asyncio.create_task(producer())
                self._inflight[key] = inflight
                owner = True

        try:
            value = await inflight
        except Exception:
            if owner:
                async with self._lock:
                    self._inflight.pop(key, None)
            raise

        if owner:
            self.set(key, value)
            async with self._lock:
                self._inflight.pop(key, None)

        return deepcopy(value)
