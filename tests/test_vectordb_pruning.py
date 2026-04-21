import os
import sys
import types
import unittest

os.environ.setdefault("DISCORD_TOKEN", "test-token")

if "chromadb" not in sys.modules:
    sys.modules["chromadb"] = types.SimpleNamespace(PersistentClient=object)

from storage.vectordb import (
    RAW_CHUNK_SOFT_CAP,
    _is_raw_worthy_source,
    _plan_raw_chunk_retention,
    _select_representative_chunks,
)


class VectorDBPruningTests(unittest.TestCase):
    def test_raw_worthy_detection_prefers_reference_like_sources(self):
        self.assertTrue(_is_raw_worthy_source("https://example.com/docs/api/reference"))
        self.assertTrue(_is_raw_worthy_source("https://example.com/files/spec.pdf"))
        self.assertFalse(_is_raw_worthy_source("https://example.com/blog/company-update"))

    def test_representative_selection_keeps_edges_and_middle(self):
        chunks = [f"chunk-{i}" for i in range(8)]
        kept = _select_representative_chunks(chunks, 3)
        self.assertEqual(kept[0], "chunk-0")
        self.assertEqual(kept[-1], "chunk-7")
        self.assertEqual(len(kept), 3)
        self.assertIn("chunk-4", kept)

    def test_low_value_raw_becomes_summary_only_under_pressure(self):
        policy = _plan_raw_chunk_retention(
            "https://example.com/blog/company-update",
            raw_chunk_count=10,
            existing_raw_chunks=RAW_CHUNK_SOFT_CAP,
        )
        self.assertEqual(policy["raw_storage_tier"], "summary-only")
        self.assertEqual(policy["source_raw_chunks_stored"], 0)
        self.assertEqual(policy["source_has_raw"], 0)

    def test_high_value_raw_is_capped_not_dropped_under_pressure(self):
        policy = _plan_raw_chunk_retention(
            "https://example.com/docs/api/reference",
            raw_chunk_count=20,
            existing_raw_chunks=RAW_CHUNK_SOFT_CAP,
        )
        self.assertEqual(policy["raw_storage_tier"], "capped")
        self.assertGreater(policy["source_raw_chunks_stored"], 0)
        self.assertEqual(policy["source_raw_worthy"], 1)


if __name__ == "__main__":
    unittest.main()
