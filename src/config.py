"""Konfigurasi terpusat untuk eksperimen SciFact (BM25 vs dense retrieval).

Semua nilai yang memengaruhi reprodusibilitas (seed, k, URL dataset, sha256,
nama + revisi model) didefinisikan di sini agar mudah diperiksa.
"""
from pathlib import Path

# --- Reprodusibilitas -------------------------------------------------------
SEED = 42
N_RANDOM_QUERIES = 5
K = 10            # cutoff evaluasi (metrik @10)
TOP_N = 100       # panjang daftar hasil yang disimpan (dipakai untuk fusi RRF)

# --- Path -------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

# --- Dataset (BEIR SciFact, zip resmi) --------------------------------------
DATASET_NAME = "scifact"
DATASET_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
DATASET_SHA256 = "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165"
QRELS_SPLIT = "test"

# --- BM25 -------------------------------------------------------------------
# rank_bm25.BM25Okapi dipakai dengan nilai default pustaka.
# Nilai ini disalin dari signature BM25Okapi.__init__ (rank-bm25 0.2.2) dan
# diverifikasi ulang saat runtime oleh bm25_search.bm25_params().
BM25_PARAMS = {"k1": 1.5, "b": 0.75, "epsilon": 0.25}
TOKEN_PATTERN = r"[a-z0-9]+"

# --- Model dense ------------------------------------------------------------
# "similarity": "cosine" -> embedding dinormalisasi L2 lalu dot product.
# "similarity": "dot"    -> dot product tanpa normalisasi.
DENSE_MODELS = {
    "MiniLM": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "similarity": "cosine",
    },
    # Model card (revisi di bawah) memakai util.dot_score untuk skor
    # kueri-dokumen; BEIR juga mengevaluasi TAS-B dengan dot product.
    "TAS-B": {
        "name": "sentence-transformers/msmarco-distilbert-base-tas-b",
        "revision": "b12d9352e776979147078a8975a4885042984fd1",
        "similarity": "dot",
    },
}
ENCODE_BATCH_SIZE = 64

# --- Hybrid (Reciprocal Rank Fusion) ----------------------------------------
RRF_K = 60

# --- Mode smoke test --------------------------------------------------------
SMOKE_CORPUS_SIZE = 200

# Urutan metode di tabel ringkasan
METHOD_ORDER = ["BM25", "MiniLM", "TAS-B", "RRF-BM25+MiniLM", "RRF-BM25+TAS-B"]
