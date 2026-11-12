"""Pencarian dense/semantik dengan sentence-transformers.

- all-MiniLM-L6-v2      : cosine similarity (embedding dinormalisasi L2, lalu dot).
- msmarco-distilbert-base-tas-b : dot product TANPA normalisasi, sesuai model card
  (contoh kode memakai util.dot_score) dan konfigurasi evaluasi BEIR.

Pemecahan skor seri sama dengan BM25: urutan dokumen di korpus (argsort stabil).
Panjang input maksimum (pemotongan token) mengikuti konfigurasi model:
MiniLM max_seq_length = 256, TAS-B = 512.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from . import config
from .bm25_search import top_n
from .data import doc_text


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class DenseSearcher:
    def __init__(self, key: str, device: str | None = None,
                 batch_size: int = config.ENCODE_BATCH_SIZE):
        spec = config.DENSE_MODELS[key]
        self.key = key
        self.model_name = spec["name"]
        self.revision = spec["revision"]
        self.similarity = spec["similarity"]
        # all-MiniLM-L6-v2 sudah memiliki modul Normalize di pipeline-nya;
        # normalize_embeddings=True tetap diberikan agar eksplisit (idempoten).
        self.normalize = self.similarity == "cosine"
        self.device = device or get_device()
        self.batch_size = batch_size

        t0 = time.perf_counter()
        self.model = SentenceTransformer(self.model_name, revision=self.revision,
                                         device=self.device)
        self.load_seconds = time.perf_counter() - t0

    def _encode(self, texts: list[str], progress: bool) -> torch.Tensor:
        emb = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=progress,
            convert_to_tensor=True,
            normalize_embeddings=self.normalize,
            device=self.device,
        )
        return emb.float()

    def index(self, corpus: dict[str, dict]) -> None:
        """Encode korpus satu kali (title + " " + text)."""
        t0 = time.perf_counter()
        self.doc_ids = list(corpus.keys())
        self.doc_emb = self._encode([doc_text(corpus[d]) for d in self.doc_ids], progress=True)
        self.index_seconds = time.perf_counter() - t0

    def search(self, queries: dict[str, str], n: int = config.TOP_N
               ) -> dict[str, list[tuple[str, float]]]:
        t0 = time.perf_counter()
        qids = list(queries.keys())
        q_emb = self._encode([queries[q] for q in qids], progress=False)
        scores = (q_emb @ self.doc_emb.T).cpu().numpy().astype(np.float64)
        run = {qid: top_n(scores[i], self.doc_ids, n) for i, qid in enumerate(qids)}
        self.search_seconds = time.perf_counter() - t0
        return run
