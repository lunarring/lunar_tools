import unittest
from lunar_tools.web_search import search

class TestWebSearch(unittest.TestCase):
    def test_search_returns_results(self):
        results = search("openai")
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        for item in results:
            self.assertIsInstance(item, dict)
            self.assertIn("title", item)
            self.assertIn("url", item)

if __name__ == '__main__':
    unittest.main()
