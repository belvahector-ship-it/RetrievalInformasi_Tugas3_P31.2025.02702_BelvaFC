"""Hybrid retrieval dengan Reciprocal Rank Fusion (RRF).

    score_RRF(d) = sum_{r in runs} 1 / (k + rank_r(d)),   k = 60, rank mulai dari 1

Dokumen yang tidak muncul di suatu daftar tidak mendapat kontribusi dari daftar itu.
Pemecahan skor seri: rank terbaik (terkecil) di daftar mana pun, lalu doc_id menaik.
Referensi: G. V. Cormack, C. L. A. Clarke, S. Büttcher (2009). "Reciprocal rank
fusion outperforms Condorcet and individual rank learning methods". SIGIR '09,
hlm. 758-759. DOI: 10.1145/1571941.1572114
"""
from __future__ import annotations

from . import config


def rrf_fuse(rankings: list[list[str]], k: int = config.RRF_K,
             n: int | None = None) -> list[tuple[str, float]]:
    """Gabungkan beberapa daftar peringkat (doc_id terurut) menjadi satu daftar RRF.

    Seri dipecah secara deterministik: rank terbaik (terkecil) di daftar mana pun,
    lalu doc_id.
    """
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            best_rank[doc_id] = min(best_rank.get(doc_id, rank), rank)
    fused = sorted(scores.items(), key=lambda x: (-x[1], best_rank[x[0]], x[0]))
    return fused[:n] if n is not None else fused


def rrf_runs(runs: list[dict[str, list[tuple[str, float]]]], k: int = config.RRF_K,
             n: int = config.TOP_N) -> dict[str, list[tuple[str, float]]]:
    """RRF per kueri atas beberapa run {qid: [(doc_id, skor), ...]}."""
    qids = runs[0].keys()
    return {
        qid: rrf_fuse([[d for d, _ in run[qid]] for run in runs], k=k, n=n)
        for qid in qids
    }
