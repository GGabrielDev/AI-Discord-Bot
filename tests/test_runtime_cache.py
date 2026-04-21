import asyncio
import unittest

from runtime_cache import TTLCache


class RuntimeCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_or_set_deduplicates_concurrent_producers(self):
        cache = TTLCache(ttl_seconds=60, max_entries=8)
        calls = 0

        async def produce():
            nonlocal calls
            calls += 1
            await asyncio.sleep(0)
            return {"value": 7}

        first, second = await asyncio.gather(
            cache.get_or_set("same-key", produce),
            cache.get_or_set("same-key", produce),
        )

        self.assertEqual(calls, 1)
        self.assertEqual(first, {"value": 7})
        self.assertEqual(second, {"value": 7})

    async def test_entries_expire_and_return_defensive_copies(self):
        fake_time = [100.0]
        cache = TTLCache(ttl_seconds=5, max_entries=8, time_func=lambda: fake_time[0])

        async def produce():
            return {"items": [1, 2]}

        result = await cache.get_or_set("expiring", produce)
        result["items"].append(3)

        cached = cache.get("expiring")
        self.assertEqual(cached, {"items": [1, 2]})

        fake_time[0] += 6
        self.assertIsNone(cache.get("expiring"))


if __name__ == "__main__":
    unittest.main()
