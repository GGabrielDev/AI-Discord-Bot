import unittest

from agent.ask_state import (
    advance_gap_cycle,
    dequeue_gap_batch,
    ensure_gap_state,
    explain_gap_route,
    merge_gap_memory,
    quality_from_meta,
    queue_gap_queries,
    record_gap_probe,
    restore_gap_batch,
    select_gap_route,
    set_gap_route,
)


class AskStateTests(unittest.TestCase):
    def test_queue_semantic_duplicates_merge_and_sum_weights(self):
        state = ensure_gap_state(None)
        queue_gap_queries(state, [
            "deployment status",
            "latest deployment status",
            "What is the latest deployment status?",
            "battery chemistry note",
        ])

        self.assertEqual(state["order"], ["deployment status", "battery chemistry note"])
        self.assertEqual(state["repeat_counts"]["deployment status"], 3)
        self.assertEqual(state["pending"]["deployment status"], "What is the latest deployment status?")
        self.assertEqual(state["details"]["deployment status"]["repeat_count"], 3)

        batch, deferred = dequeue_gap_batch(state, limit=1)
        self.assertEqual(batch, ["What is the latest deployment status?"])
        self.assertEqual(deferred, 1)

    def test_queue_prioritizes_needs_web_then_repeats(self):
        state = advance_gap_cycle(ensure_gap_state(None), 3)
        queue_gap_queries(state, [
            "fresh regulation status",
            "legacy chemistry detail",
            "legacy chemistry detail",
            "internal voltage spec",
        ])

        legacy_meta = state["details"]["legacy chemistry detail"]
        set_gap_route(legacy_meta, "partial_local", 3)

        regulation_meta = state["details"]["fresh regulation status"]
        set_gap_route(regulation_meta, "needs_web", 3)

        batch, deferred = dequeue_gap_batch(state, limit=2)
        self.assertEqual(batch, ["fresh regulation status", "legacy chemistry detail"])
        self.assertEqual(deferred, 1)

    def test_select_gap_route_prefers_partial_and_web_paths(self):
        state = ensure_gap_state(None)
        queue_gap_queries(state, ["battery chemistry note", "latest deployment status"])

        partial_probe = {
            "answer": "Some local evidence exists.",
            "resolved": False,
            "llm_confidence": 0.4,
            "local_score": 0.45,
            "source_count": 2,
            "raw_hits": 2,
            "summary_hits": 1,
            "total_hits": 3,
            "has_partial_answer": True,
        }
        partial_meta = record_gap_probe(state, "battery chemistry note", partial_probe, 0)
        self.assertEqual(select_gap_route(partial_meta, partial_probe, False), "partial_local")

        web_probe = {
            "answer": "",
            "resolved": False,
            "llm_confidence": 0.0,
            "local_score": 0.12,
            "source_count": 0,
            "raw_hits": 0,
            "summary_hits": 0,
            "total_hits": 0,
            "has_partial_answer": False,
        }
        web_meta = record_gap_probe(state, "latest deployment status", web_probe, 0)
        self.assertEqual(select_gap_route(web_meta, web_probe, False), "needs_web")

    def test_merge_gap_memory_restores_repeat_counts(self):
        state = ensure_gap_state(None)
        memory = {
            "repeat_counts": {"legacy chemistry detail": 4},
            "details": {
                "legacy chemistry detail": {
                    "query": "legacy chemistry detail",
                    "repeat_count": 4,
                    "last_route": "partial_local",
                    "last_resolution": "partial_local",
                }
            },
        }
        merged = merge_gap_memory(state, memory)
        self.assertEqual(merged["repeat_counts"]["legacy chemistry detail"], 4)
        self.assertEqual(merged["details"]["legacy chemistry detail"]["last_route"], "partial_local")

    def test_merge_gap_memory_semantic_duplicates_sum_repeat_counts(self):
        merged = merge_gap_memory(ensure_gap_state(None), {
            "pending": {
                "deployment status": "deployment status",
                "latest deployment status": "latest deployment status",
            },
            "order": ["deployment status", "latest deployment status"],
            "repeat_counts": {
                "deployment status": 2,
                "latest deployment status": 3,
            },
            "details": {
                "deployment status": {
                    "query": "deployment status",
                    "repeat_count": 2,
                    "last_route": "partial_local",
                    "last_resolution": "partial_local",
                },
                "latest deployment status": {
                    "query": "What is the latest deployment status?",
                    "repeat_count": 3,
                    "last_route": "needs_web",
                    "last_resolution": "needs_web",
                },
            },
        })

        self.assertEqual(merged["order"], ["deployment status"])
        self.assertEqual(merged["repeat_counts"]["deployment status"], 5)
        self.assertEqual(merged["pending"]["deployment status"], "What is the latest deployment status?")
        self.assertEqual(merged["details"]["deployment status"]["query"], "What is the latest deployment status?")
        self.assertEqual(merged["details"]["deployment status"]["repeat_count"], 5)
        self.assertEqual(merged["details"]["deployment status"]["last_route"], "needs_web")

    def test_restore_gap_batch_requeues_without_bumping_repeat_counts(self):
        state = ensure_gap_state(None)
        queue_gap_queries(state, ["gap one", "gap two", "gap three"])

        batch, deferred = dequeue_gap_batch(state, limit=2)
        self.assertEqual(batch, ["gap one", "gap two"])
        self.assertEqual(deferred, 1)

        restore_gap_batch(state, ["gap two"])
        self.assertEqual(state["order"], ["gap two", "gap three"])
        self.assertEqual(state["repeat_counts"]["gap two"], 1)

    def test_record_gap_probe_reuses_semantic_gap_key(self):
        state = ensure_gap_state(None)
        queue_gap_queries(state, ["deployment status"])
        probe = {
            "answer": "",
            "resolved": False,
            "llm_confidence": 0.2,
            "local_score": 0.3,
            "source_count": 1,
            "raw_hits": 1,
            "summary_hits": 0,
            "total_hits": 1,
            "has_partial_answer": False,
        }

        gap_meta = record_gap_probe(state, "What is the latest deployment status?", probe, 2)

        self.assertEqual(list(state["details"].keys()), ["deployment status"])
        self.assertEqual(gap_meta["query"], "What is the latest deployment status?")
        self.assertEqual(gap_meta["local_attempts"], 1)

    def test_quality_penalizes_stale_sources_and_explains_route(self):
        fresh_score = quality_from_meta({
            "source_quality_score": 0.8,
            "source_age_days": 30,
        })
        stale_score = quality_from_meta({
            "source_quality_score": 0.8,
            "source_age_days": 365,
        })
        self.assertLess(stale_score, fresh_score)

        state = ensure_gap_state(None)
        queue_gap_queries(state, ["latest deployment status"])
        probe = {
            "answer": "",
            "resolved": False,
            "llm_confidence": 0.0,
            "local_score": 0.15,
            "source_count": 0,
            "raw_hits": 0,
            "summary_hits": 0,
            "total_hits": 0,
            "has_partial_answer": False,
        }
        gap_meta = record_gap_probe(state, "latest deployment status", probe, 0)
        route = select_gap_route(gap_meta, probe, False)
        explanation = explain_gap_route(gap_meta, probe, route, False)
        self.assertEqual(route, "needs_web")
        self.assertTrue(explanation)


if __name__ == "__main__":
    unittest.main()
