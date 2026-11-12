"""Uji metrik evaluasi dengan contoh mainan yang dihitung manual."""
import importlib.util
import math
import unittest

from src.evaluate import (cross_check_ir_measures, evaluate_run, overlap_at_k,
                          query_metrics, summarize)


def ids(*xs):
    return [f"d{x}" for x in xs]


class TestQueryMetrics(unittest.TestCase):
    def test_single_relevant_at_rank_3(self):
        # 1 dokumen relevan di rank 3
        m = query_metrics(ids(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), {"d3"})
        self.assertAlmostEqual(m["P@10"], 0.1)
        self.assertAlmostEqual(m["R@10"], 1.0)
        self.assertAlmostEqual(m["MRR@10"], 1 / 3)
        self.assertAlmostEqual(m["nDCG@10"], 1 / math.log2(4))  # = 0.5
        self.assertAlmostEqual(m["nDCG@10"], 0.5)
        self.assertAlmostEqual(m["MAP@10"], 1 / 3)

    def test_perfect_ranking(self):
        m = query_metrics(ids(1, 2, 3), {"d1", "d2"})
        self.assertAlmostEqual(m["nDCG@10"], 1.0)
        self.assertAlmostEqual(m["MAP@10"], 1.0)
        self.assertAlmostEqual(m["MRR@10"], 1.0)
        self.assertAlmostEqual(m["R@10"], 1.0)
        self.assertAlmostEqual(m["P@10"], 0.2)

    def test_two_relevant_ranks_1_and_4(self):
        # relevan di rank 1 dan 4; R = 2
        m = query_metrics(ids(1, 2, 3, 4, 5), {"d1", "d4"})
        dcg = 1 / math.log2(2) + 1 / math.log2(5)
        idcg = 1 / math.log2(2) + 1 / math.log2(3)
        self.assertAlmostEqual(m["nDCG@10"], dcg / idcg)
        self.assertAlmostEqual(m["MAP@10"], (1 / 1 + 2 / 4) / 2)  # 0.75
        self.assertAlmostEqual(m["MRR@10"], 1.0)
        self.assertAlmostEqual(m["P@10"], 0.2)

    def test_relevant_beyond_cutoff_is_ignored(self):
        ranked = ids(*range(1, 16))  # 15 dokumen
        m = query_metrics(ranked, {"d12"})
        for v in m.values():
            self.assertEqual(v, 0.0)

    def test_ap_normalized_by_total_relevant(self):
        # 12 dokumen relevan, semua 10 teratas relevan -> AP@10 = 10/12 (gaya trec_eval)
        rel = set(ids(*range(1, 13)))
        m = query_metrics(ids(*range(1, 11)), rel)
        self.assertAlmostEqual(m["MAP@10"], 10 / 12)
        self.assertAlmostEqual(m["R@10"], 10 / 12)
        self.assertAlmostEqual(m["nDCG@10"], 1.0)  # IDCG memakai min(R, k)

    def test_no_hits(self):
        m = query_metrics(ids(1, 2), {"d9"})
        self.assertEqual(m, {"nDCG@10": 0.0, "P@10": 0.0, "R@10": 0.0,
                             "MRR@10": 0.0, "MAP@10": 0.0})

    def test_empty_relevant_raises(self):
        with self.assertRaises(ValueError):
            query_metrics(ids(1), set())


class TestRunLevel(unittest.TestCase):
    def setUp(self):
        self.qrels = {"q1": {"d3": 1}, "q2": {"d1": 1, "d2": 0}, "q3": {}}
        self.run = {"q1": [(d, 1.0) for d in ids(1, 2, 3)],
                    "q2": [(d, 1.0) for d in ids(1, 2)]}

    def test_evaluate_and_summarize(self):
        df = evaluate_run(self.run, self.qrels, ["q1", "q2", "q3"])
        self.assertEqual(list(df.index), ["q1", "q2"])  # q3 tanpa relevan dilewati
        self.assertEqual(df.loc["q2", "n_relevant"], 1)  # skor 0 = tidak relevan
        s = summarize({"X": df})
        self.assertAlmostEqual(s.loc["X", "MRR@10"], (1 / 3 + 1) / 2)
        self.assertEqual(s.loc["X", "n_queries"], 2)

    def test_missing_query_in_run_scores_zero(self):
        df = evaluate_run({}, self.qrels, ["q1"])
        self.assertEqual(df.loc["q1", "nDCG@10"], 0.0)

    def test_overlap(self):
        a = {"q": [(d, 0) for d in ids(1, 2, 3)]}
        b = {"q": [(d, 0) for d in ids(3, 4, 1)]}
        o = overlap_at_k(a, b, ["q"])
        self.assertEqual(o.loc["q", "overlap@10"], 2)
        self.assertAlmostEqual(o.loc["q", "jaccard"], 2 / 4)


@unittest.skipUnless(importlib.util.find_spec("ir_measures"), "ir-measures tidak terpasang")
class TestCrossCheck(unittest.TestCase):
    def test_rank_based_matches_and_tie_difference_visible(self):
        qrels = {"q1": {"d2": 1}}
        # d1 dan d2 seri; urutan kita: d1, d2 (urutan korpus). trec_eval (pytrec_eval,
        # dipakai ir-measures untuk nDCG/AP) memecah seri dengan doc_id menurun -> d2
        # di rank 1, sehingga nDCG kita (d2 di rank 2) berbeda dari skor asli.
        run = {"q1": [("d1", 1.0), ("d2", 1.0), ("d3", 0.5)]}
        chk = cross_check_ir_measures(run, qrels, ["q1"])
        self.assertTrue(chk["match_rank"].all())  # urutan identik -> sama persis
        self.assertAlmostEqual(chk.loc["nDCG@10", "ours"], 1 / math.log2(3))
        self.assertAlmostEqual(chk.loc["nDCG@10", "ir_measures_score"], 1.0)
        self.assertGreater(chk.loc["nDCG@10", "max_abs_diff_score"], 0.3)


if __name__ == "__main__":
    unittest.main()
