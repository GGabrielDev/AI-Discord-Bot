import os
import sys
import types
import unittest
from unittest.mock import AsyncMock, Mock, patch

os.environ.setdefault("DISCORD_TOKEN", "test-token")

if "dotenv" not in sys.modules:
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv_stub

if "chromadb" not in sys.modules:
    chromadb_stub = types.ModuleType("chromadb")

    class _PersistentClient:
        def __init__(self, *args, **kwargs):
            pass

        def get_or_create_collection(self, *args, **kwargs):
            return Mock()

    chromadb_stub.PersistentClient = _PersistentClient
    sys.modules["chromadb"] = chromadb_stub

if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")

    class _AsyncOpenAI:
        def __init__(self, *args, **kwargs):
            pass

        async def close(self):
            return None

    openai_stub.AsyncOpenAI = _AsyncOpenAI
    sys.modules["openai"] = openai_stub

if "agent.loop" not in sys.modules:
    agent_loop_stub = types.ModuleType("agent.loop")

    async def _run_autonomous_loop(*args, **kwargs):
        return 0

    agent_loop_stub.run_autonomous_loop = _run_autonomous_loop
    sys.modules["agent.loop"] = agent_loop_stub

import query
from agent.ask_state import ensure_gap_state


class AskGapBatchRegressionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.llm = Mock()
        self.llm.generate_text = AsyncMock(side_effect=["initial draft", "refined draft"])
        self.llm.generate_json = AsyncMock(return_value={"queries": []})

        self.db = Mock()
        self.db.search_with_metadata = AsyncMock(return_value=[
            ("summary chunk", {"source": "source-a", "chunk_type": "summary"})
        ])

        self.common_patches = [
            patch.object(query, "LocalLLM", return_value=self.llm),
            patch.object(query, "VectorDB", return_value=self.db),
            patch.object(query, "_expand_query", AsyncMock(side_effect=lambda llm, question, num_variations: [question])),
            patch.object(query, "deep_internal_probe", AsyncMock(return_value={
                "answer": "Resolved locally.",
                "sources": ["source-a"],
                "resolved": True,
                "llm_confidence": 0.9,
                "local_score": 0.95,
                "source_count": 1,
                "raw_hits": 1,
                "summary_hits": 1,
                "avg_quality": 0.9,
                "avg_distance": 0.1,
                "total_hits": 2,
                "has_partial_answer": True,
            })),
            patch.object(query, "load_gap_memory", return_value=None),
            patch.object(query, "save_gap_memory"),
            patch.object(query, "load_ask_checkpoint", return_value=None),
            patch.object(query, "save_ask_checkpoint"),
            patch.object(query, "delete_ask_checkpoint"),
            patch.object(query, "check_soft_stop", return_value=False),
            patch.object(query, "finalize_report", AsyncMock(side_effect=lambda draft, topic: {"english": draft})),
        ]

        self.patchers = []
        for patcher in self.common_patches:
            self.patchers.append(patcher)
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self.patchers):
            patcher.stop()

    async def test_normal_gap_batch_refines_once_after_batch(self):
        with patch.object(query, "extract_gap_queries", AsyncMock(return_value=["gap one", "gap two", "gap three"])):
            result = await query.answer_question(
                topic="energy",
                question="What changed?",
                mode="Balanced",
                style="Concise",
            )

        self.assertEqual(result["english"], "refined draft")
        self.assertNotIn("translated", result)
        self.assertEqual(self.llm.generate_text.await_count, 2)
        self.assertEqual(query.deep_internal_probe.await_count, 3)

    async def test_resumed_draft_batch_refines_once_after_batch(self):
        self.llm.generate_text = AsyncMock(return_value="refined draft")
        with patch.object(query, "extract_gap_queries", AsyncMock(return_value=["gap one", "gap two", "gap three"])):
            result = await query.answer_question(
                topic="energy",
                question="What changed?",
                mode="Balanced",
                style="Concise",
                _draft="existing draft",
                _gap_state=ensure_gap_state(None),
            )

        self.assertEqual(result["english"], "refined draft")
        self.assertEqual(self.llm.generate_text.await_count, 1)
        self.assertEqual(query.deep_internal_probe.await_count, 3)

    async def test_normal_gap_batch_keeps_checkpoint_when_deferred_gaps_remain(self):
        with patch.object(query, "extract_gap_queries", AsyncMock(return_value=["gap one", "gap two", "gap three", "gap four"])):
            result = await query.answer_question(
                topic="energy",
                question="What changed?",
                mode="Balanced",
                style="Concise",
            )

        self.assertEqual(result["english"], "refined draft")
        query.delete_ask_checkpoint.assert_not_called()
        final_gap_state = query.save_ask_checkpoint.call_args_list[-1].kwargs["gap_state"]
        self.assertEqual(final_gap_state["order"], ["gap four"])

    async def test_resumed_batch_keeps_checkpoint_when_deferred_gaps_remain(self):
        resumed_gap_state = ensure_gap_state(None)
        query.queue_gap_queries(resumed_gap_state, ["gap one", "gap two", "gap three", "gap four"])
        self.llm.generate_text = AsyncMock(return_value="refined draft")

        result = await query.answer_question(
            topic="energy",
            question="What changed?",
            mode="Balanced",
            style="Concise",
            _draft="existing draft",
            _gap_state=resumed_gap_state,
        )

        self.assertEqual(result["english"], "refined draft")
        query.delete_ask_checkpoint.assert_not_called()
        final_gap_state = query.save_ask_checkpoint.call_args_list[-1].kwargs["gap_state"]
        self.assertEqual(final_gap_state["order"], ["gap four"])


class AskFinalizeReportTests(unittest.IsolatedAsyncioTestCase):
    async def test_finalize_report_returns_only_english_and_archives_once(self):
        with patch.object(query, "store_final_report") as store_final_report:
            result = await query.finalize_report("final english draft", "energy")

        self.assertEqual(result, {"english": "final english draft"})
        store_final_report.assert_called_once_with("energy", "final english draft")


class TranslateReportHelperTests(unittest.IsolatedAsyncioTestCase):
    async def test_translate_markdown_report_skips_llm_for_english(self):
        llm = Mock()
        llm.generate_text = AsyncMock()

        result = await query.translate_markdown_report("# Title", "English", llm=llm)

        self.assertEqual(result, "# Title")
        llm.generate_text.assert_not_awaited()

    async def test_translate_and_archive_report_uses_llm_and_archives_target_language(self):
        llm = Mock()
        llm.generate_text = AsyncMock(return_value="# Título\n\nContenido")

        with patch.object(query, "store_final_report") as store_final_report:
            result = await query.translate_and_archive_report("# Title\n\nContent", "energy", "Spanish", llm=llm)

        self.assertEqual(result, "# Título\n\nContenido")
        llm.generate_text.assert_awaited_once()
        store_final_report.assert_called_once_with("energy", "# Título\n\nContenido", "Spanish")


class TranslateReportFilenameTests(unittest.TestCase):
    def test_build_translated_report_filename_appends_language_tag(self):
        self.assertEqual(
            query.build_translated_report_filename("Report_energy.md", "Portuguese (Brazil)"),
            "Report_energy_PORTUGUESE_BRAZIL.md",
        )

    def test_infer_report_topic_from_filename_uses_report_stem(self):
        self.assertEqual(query.infer_report_topic_from_filename("Report_energy.md"), "energy")


if __name__ == "__main__":
    unittest.main()
