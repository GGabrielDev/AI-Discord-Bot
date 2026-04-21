import os
import sys
import types
import unittest

os.environ.setdefault("DISCORD_TOKEN", "test-token")

if "openai" not in sys.modules:
    fake_openai = types.ModuleType("openai")

    class _DummyAsyncOpenAI:
        def __init__(self, *args, **kwargs):
            pass

    fake_openai.AsyncOpenAI = _DummyAsyncOpenAI
    sys.modules["openai"] = fake_openai

if "dotenv" not in sys.modules:
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = fake_dotenv

from agent import planner


class _FakeLLM:
    calls = 0

    async def generate_json(self, system_prompt: str, user_prompt: str):
        type(self).calls += 1
        if "Generate 2 specific search queries" in user_prompt:
            return {"queries": ["alpha", "beta"]}
        if "Generate 2 NEW search queries" in user_prompt:
            return {"gap_analysis": "Need more detail.", "queries": ["gamma", "delta"]}
        return {"sub_topics": ["topic a", "topic b"]}


class PlannerCacheTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.original_llm = planner.LocalLLM
        planner.LocalLLM = _FakeLLM
        _FakeLLM.calls = 0
        planner._search_query_cache.clear()
        planner._replan_cache.clear()
        planner._decompose_cache.clear()

    def tearDown(self):
        planner.LocalLLM = self.original_llm
        planner._search_query_cache.clear()
        planner._replan_cache.clear()
        planner._decompose_cache.clear()

    async def test_generate_search_queries_uses_cache(self):
        first = await planner.generate_search_queries("cache me", num_queries=2)
        second = await planner.generate_search_queries("cache me", num_queries=2)

        self.assertEqual(first, ["alpha", "beta"])
        self.assertEqual(second, ["alpha", "beta"])
        self.assertEqual(_FakeLLM.calls, 1)

    async def test_evaluate_and_replan_cache_key_includes_inputs(self):
        stats = {"total_chunks": 3, "unique_sources": 2}
        first = await planner.evaluate_and_replan("battery topic", ["one"], stats, num_queries=2)
        second = await planner.evaluate_and_replan("battery topic", ["one"], stats, num_queries=2)
        third = await planner.evaluate_and_replan("battery topic", ["two"], stats, num_queries=2)

        self.assertEqual(first, (["gamma", "delta"], "Need more detail."))
        self.assertEqual(second, first)
        self.assertEqual(third, first)
        self.assertEqual(_FakeLLM.calls, 2)


if __name__ == "__main__":
    unittest.main()
