import json
import os
import shutil
import tempfile
import unittest

import agent.checkpoint as checkpoint


class AskCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.mkdtemp(prefix="ask-checkpoint-tests-")
        self.orig_dir = checkpoint.CHECKPOINT_DIR
        self.orig_flag = checkpoint.SOFT_STOP_FLAG
        checkpoint.CHECKPOINT_DIR = self.tempdir
        checkpoint.SOFT_STOP_FLAG = os.path.join(self.tempdir, "SOFT_STOP.flag")

    def tearDown(self):
        checkpoint.CHECKPOINT_DIR = self.orig_dir
        checkpoint.SOFT_STOP_FLAG = self.orig_flag
        shutil.rmtree(self.tempdir, ignore_errors=True)

    def test_save_and_load_ask_checkpoint(self):
        gap_state = {"pending": {"gap": "gap"}, "order": ["gap"], "repeat_counts": {"gap": 2}, "details": {}, "loop_index": 1}
        checkpoint.save_ask_checkpoint(
            topic="energy",
            question="What changed?",
            mode="Balanced",
            style="Concise",
            language="English",
            no_web=False,
            current_auto_loop=1,
            draft="draft text",
            gap_state=gap_state,
            extra_context="extra",
        )

        loaded = checkpoint.load_ask_checkpoint("energy", "What changed?", "Balanced", "Concise", "English", False)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["draft"], "draft text")
        self.assertEqual(loaded["gap_state"]["order"], ["gap"])

        checkpoint.delete_ask_checkpoint("energy", "What changed?", "Balanced", "Concise", "English", False)
        self.assertIsNone(checkpoint.load_ask_checkpoint("energy", "What changed?", "Balanced", "Concise", "English", False))

    def test_load_ask_checkpoint_recovers_tmp_file(self):
        filepath = checkpoint._ask_checkpoint_path("energy", "What changed?", "Balanced", "Concise", "English", False)
        os.makedirs(checkpoint.CHECKPOINT_DIR, exist_ok=True)
        state = {
            "topic": "energy",
            "question": "What changed?",
            "mode": "Balanced",
            "style": "Concise",
            "language": "English",
            "no_web": False,
            "current_auto_loop": 2,
            "draft": "draft text",
            "gap_state": {"pending": {}, "order": [], "repeat_counts": {}, "details": {}, "loop_index": 2},
            "extra_context": None,
            "status": "in_progress",
        }
        with open(filepath + ".tmp", "w", encoding="utf-8") as f:
            json.dump(state, f)

        loaded = checkpoint.load_ask_checkpoint("energy", "What changed?", "Balanced", "Concise", "English", False)
        self.assertIsNotNone(loaded)
        self.assertTrue(os.path.exists(filepath))
        self.assertEqual(loaded["current_auto_loop"], 2)

    def test_gap_memory_roundtrip(self):
        memory = {
            "repeat_counts": {"gap": 3},
            "details": {"gap": {"query": "gap", "last_route": "needs_web"}},
        }
        checkpoint.save_gap_memory("energy", memory)
        loaded = checkpoint.load_gap_memory("energy")
        self.assertEqual(loaded["repeat_counts"]["gap"], 3)
        self.assertEqual(loaded["details"]["gap"]["last_route"], "needs_web")


if __name__ == "__main__":
    unittest.main()
