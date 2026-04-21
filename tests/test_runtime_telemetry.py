import os
import sys
import types
import unittest

os.environ.setdefault("DISCORD_TOKEN", "test-token")

if "dotenv" not in sys.modules:
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv_stub

from runtime_telemetry import add, bump, note_source, snapshot, telemetry_session


class RuntimeTelemetryTests(unittest.TestCase):
    def test_session_collects_compact_summary_data(self):
        with telemetry_session("ask:test question") as session:
            bump("llm.calls", 2)
            add("llm.prompt_tokens", 1200)
            add("llm.completion_tokens", 300)
            bump("cache.hits", 3)
            bump("cache.misses", 1)
            bump("gap.route.partial_local")
            note_source("https://example.com/docs/spec", summary_chunks=4, raw_chunks=6, raw_chunks_kept=2)
            data = snapshot()

        self.assertIsNotNone(data)
        self.assertEqual(data["counters"]["llm.calls"], 2)
        self.assertEqual(data["totals"]["llm.prompt_tokens"], 1200)
        self.assertIn("https://example.com/docs/spec", data["sources"])
        self.assertIn("llm=2", session.format_summary())
        self.assertIn("cache=3h/1m", session.format_summary())


if __name__ == "__main__":
    unittest.main()
