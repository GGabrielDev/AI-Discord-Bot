import os
import sys
import types
import unittest

os.environ.setdefault("DISCORD_TOKEN", "test-token")

if "httpx" not in sys.modules:
    fake_httpx = types.ModuleType("httpx")

    class _DummyAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

    class _DummyLimits:
        def __init__(self, *args, **kwargs):
            pass

    fake_httpx.AsyncClient = _DummyAsyncClient
    fake_httpx.Limits = _DummyLimits
    sys.modules["httpx"] = fake_httpx

if "dotenv" not in sys.modules:
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = fake_dotenv

from tools import search


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self):
        self.calls = []

    async def get(self, url, params):
        self.calls.append((url, params))
        return _FakeResponse({
            "results": [
                {"title": "Result A", "url": "https://example.com/a", "content": "Alpha"},
                {"title": "Result B", "url": "https://example.com/b", "content": "Beta"},
                {"title": "Result C", "url": "https://example.com/c", "content": "Gamma"},
            ]
        })


class SearchCacheTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.original_client = search._search_client
        search._search_client = _FakeClient()
        search._search_response_cache.clear()

    def tearDown(self):
        search._search_client = self.original_client
        search._search_response_cache.clear()

    async def test_reuses_cached_payload_for_repeated_query(self):
        first = await search.get_search_results("python asyncio", max_results=2)
        first[0]["title"] = "mutated"
        second = await search.get_search_results("python asyncio", max_results=1)

        self.assertEqual(len(search._search_client.calls), 1)
        self.assertEqual(len(first), 2)
        self.assertEqual(len(second), 1)
        self.assertEqual(second[0]["title"], "Result A")


if __name__ == "__main__":
    unittest.main()
