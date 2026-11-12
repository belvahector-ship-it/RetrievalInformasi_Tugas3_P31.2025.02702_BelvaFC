"""Uji Reciprocal Rank Fusion."""
import unittest

from src.hybrid import rrf_fuse, rrf_runs


class TestRRF(unittest.TestCase):
    def test_scores_hand_computed(self):
        fused = dict(rrf_fuse([["a", "b", "c"], ["b", "c", "d"]], k=60))
        self.assertAlmostEqual(fused["a"], 1 / 61)
        self.assertAlmostEqual(fused["b"], 1 / 62 + 1 / 61)
        self.assertAlmostEqual(fused["c"], 1 / 63 + 1 / 62)
        self.assertAlmostEqual(fused["d"], 1 / 63)

    def test_order(self):
        fused = [d for d, _ in rrf_fuse([["a", "b", "c"], ["b", "c", "d"]])]
        self.assertEqual(fused, ["b", "c", "a", "d"])

    def test_tie_break_deterministic(self):
        # a dan x sama-sama rank 1 di satu daftar -> seri, dipecah dengan doc_id
        fused = [d for d, _ in rrf_fuse([["x"], ["a"]])]
        self.assertEqual(fused, ["a", "x"])

    def test_truncation(self):
        self.assertEqual(len(rrf_fuse([list("abcdef")], n=3)), 3)

    def test_rrf_runs(self):
        r1 = {"q": [("a", 9.0), ("b", 5.0)]}
        r2 = {"q": [("b", 0.9), ("a", 0.1)]}
        out = rrf_runs([r1, r2], n=10)
        self.assertEqual(set(out), {"q"})
        self.assertAlmostEqual(out["q"][0][1], 1 / 61 + 1 / 62)


if __name__ == "__main__":
    unittest.main()
