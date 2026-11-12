"""Pencarian leksikal BM25 (rank_bm25.BM25Okapi, parameter default pustaka).

Praproses: lowercase -> token regex [a-z0-9]+ -> buang ENGLISH_STOP_WORDS
scikit-learn. Tanpa stemming.

Pemecahan skor seri: dokumen dengan skor sama diurutkan menurut posisinya di
corpus.jsonl (argsort stabil). Aturan yang sama dipakai dense_search.
"""
from __future__ import annotations

import inspect
import re
import time

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from . import config
from .data import doc_text

_TOKEN_RE = re.compile(config.TOKEN_PATTERN)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in ENGLISH_STOP_WORDS]


def bm25_params() -> dict:
    """Baca nilai default k1, b, epsilon langsung dari signature BM25Okapi."""
    sig = inspect.signature(BM25Okapi.__init__)
    return {name: sig.parameters[name].default for name in ("k1", "b", "epsilon")}


def top_n(scores: np.ndarray, doc_ids: list[str], n: int) -> list[tuple[str, float]]:
    """Urutkan skor menurun; seri dipecah menurut urutan dokumen di korpus (stabil)."""
    order = np.argsort(-scores, kind="stable")[:n]
    return [(doc_ids[i], float(scores[i])) for i in order]


class BM25Searcher:
    def __init__(self, corpus: dict[str, dict]):
        t0 = time.perf_counter()
        self.doc_ids = list(corpus.keys())
        tokenized = [tokenize(doc_text(corpus[d])) for d in self.doc_ids]
        self.params = bm25_params()
        # config.BM25_PARAMS mencatat nilai default; pastikan sama dengan pustaka terpasang.
        assert self.params == config.BM25_PARAMS, (
            f"Default BM25Okapi {self.params} berbeda dari config {config.BM25_PARAMS}")
        self.bm25 = BM25Okapi(tokenized)  # nilai default pustaka
        self.index_seconds = time.perf_counter() - t0

    def search(self, queries: dict[str, str], n: int = config.TOP_N
               ) -> dict[str, list[tuple[str, float]]]:
        """Skor seluruh korpus untuk tiap kueri, kembalikan top-n {qid: [(doc_id, skor)]}."""
        t0 = time.perf_counter()
        run = {}
        for qid, text in queries.items():
            scores = np.asarray(self.bm25.get_scores(tokenize(text)), dtype=np.float64)
            run[qid] = top_n(scores, self.doc_ids, n)
        self.search_seconds = time.perf_counter() - t0
        return run
