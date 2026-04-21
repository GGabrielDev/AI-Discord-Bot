import unittest

from agent.source_ranker import prefilter_search_results, score_search_result


class SourceRankerTests(unittest.TestCase):
    def test_scores_reference_docs_above_low_signal_pages(self):
        good_result = {
            "title": "Python asyncio reference",
            "url": "https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API",
            "snippet": "Official asyncio documentation with APIs, examples, and event loop details.",
        }
        weak_result = {
            "title": "Python category page",
            "url": "https://example.com/category/python",
            "snippet": "Miscellaneous posts.",
        }

        good_score, good_reasons = score_search_result(good_result, "python asyncio event loop")
        weak_score, weak_reasons = score_search_result(weak_result, "python asyncio event loop")

        self.assertGreater(good_score, weak_score)
        self.assertIn("known-source", good_reasons)
        self.assertIn("low-signal-path", weak_reasons)

    def test_prefilter_keeps_best_results_with_floor(self):
        results = [
            {
                "title": "Official API guide",
                "url": "https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API",
                "snippet": "Reference documentation for the Fetch API and usage examples.",
            },
            {
                "title": "Forum thread",
                "url": "https://example.com/search?q=fetch+api",
                "snippet": "Search results page.",
            },
            {
                "title": "Wikipedia article",
                "url": "https://en.wikipedia.org/wiki/Fetch",
                "snippet": "Encyclopedic overview with terminology and history.",
            },
        ]

        accepted, rejected = prefilter_search_results(
            results,
            query="fetch api reference",
            max_results=2,
            min_score=1.0,
        )

        self.assertEqual(len(accepted), 2)
        self.assertGreaterEqual(accepted[0]["prefilter_score"], accepted[1]["prefilter_score"])
        self.assertTrue(any(item["url"].startswith("https://example.com/search") for item in rejected))


if __name__ == "__main__":
    unittest.main()
