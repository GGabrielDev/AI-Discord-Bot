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
    select_gap_route,
    set_gap_route,
)


class AskStateTests(unittest.TestCase):
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
