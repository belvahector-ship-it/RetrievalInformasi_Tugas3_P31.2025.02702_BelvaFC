"""Unduh, verifikasi, dan baca dataset BEIR SciFact (tanpa paket `beir`).

Empat file yang dibaca:
    scifact/corpus.jsonl      -> dokumen (_id, title, text)
    scifact/queries.jsonl     -> kueri (_id, text)
    scifact/qrels/train.tsv   -> penilaian relevansi split train
    scifact/qrels/test.tsv    -> penilaian relevansi split test
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import urllib.request
import zipfile
from pathlib import Path

from . import config


# ---------------------------------------------------------------------------
# Unduh + verifikasi
# ---------------------------------------------------------------------------
def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _safe_extract(zip_path: Path, dest: Path) -> None:
    """Ekstrak zip sambil menolak entri yang keluar dari folder tujuan."""
    dest = dest.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            target = (dest / member).resolve()
            if dest not in target.parents and target != dest:
                raise ValueError(f"Entri zip tidak aman: {member}")
        zf.extractall(dest)


def download_scifact(data_dir: Path = config.DATA_DIR) -> Path:
    """Unduh scifact.zip (bila belum ada), cek sha256, ekstrak. Kembalikan folder dataset."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "scifact.zip"
    ds_dir = data_dir / config.DATASET_NAME

    if not zip_path.exists():
        print(f"Mengunduh {config.DATASET_URL} ...")
        tmp = zip_path.with_suffix(".part")
        urllib.request.urlretrieve(config.DATASET_URL, tmp)
        tmp.replace(zip_path)

    digest = sha256_of(zip_path)
    if digest != config.DATASET_SHA256:
        raise ValueError(
            f"sha256 tidak cocok: {digest} (diharapkan {config.DATASET_SHA256}). "
            "Hapus file zip lalu unduh ulang."
        )
    print(f"sha256 terverifikasi: {digest}")

    if not (ds_dir / "corpus.jsonl").exists():
        _safe_extract(zip_path, data_dir)
    return ds_dir


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def load_corpus(ds_dir: Path) -> dict[str, dict]:
    """Kembalikan {doc_id: {"title": str, "text": str}} dengan urutan file."""
    corpus = {}
    with open(Path(ds_dir) / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                corpus[str(row["_id"])] = {
                    "title": row.get("title", "") or "",
                    "text": row.get("text", "") or "",
                }
    return corpus


def load_queries(ds_dir: Path) -> dict[str, str]:
    """Kembalikan {query_id: teks kueri}."""
    queries = {}
    with open(Path(ds_dir) / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                queries[str(row["_id"])] = row["text"]
    return queries


def load_qrels(ds_dir: Path, split: str) -> dict[str, dict[str, int]]:
    """Baca qrels/<split>.tsv (header: query-id, corpus-id, score).

    Kembalikan {query_id: {doc_id: score}}.
    """
    qrels: dict[str, dict[str, int]] = {}
    with open(Path(ds_dir) / "qrels" / f"{split}.tsv", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        assert [h.strip() for h in header] == ["query-id", "corpus-id", "score"], header
        for qid, did, score in reader:
            qrels.setdefault(qid, {})[did] = int(score)
    return qrels


def count_qrels_rows(ds_dir: Path, split: str) -> int:
    """Jumlah baris data (tanpa header) pada file qrels."""
    with open(Path(ds_dir) / "qrels" / f"{split}.tsv", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip()) - 1


def load_all(ds_dir: Path) -> dict:
    """Baca keempat file sekaligus."""
    return {
        "corpus": load_corpus(ds_dir),
        "queries": load_queries(ds_dir),
        "qrels_train": load_qrels(ds_dir, "train"),
        "qrels_test": load_qrels(ds_dir, "test"),
    }


def doc_text(doc: dict) -> str:
    """Teks dokumen yang diindeks = title + " " + text."""
    return (doc["title"] + " " + doc["text"]).strip()


# ---------------------------------------------------------------------------
# Sampling kueri + subset smoke test
# ---------------------------------------------------------------------------
def sample_queries(qrels_test: dict, n: int = config.N_RANDOM_QUERIES,
                   seed: int = config.SEED) -> list[str]:
    """Ambil n kueri acak HANYA dari kueri yang punya qrels test."""
    return random.Random(seed).sample(sorted(qrels_test.keys()), n)


def smoke_subset(corpus: dict, qrels: dict, query_ids: list[str],
                 size: int = config.SMOKE_CORPUS_SIZE, seed: int = config.SEED) -> dict:
    """Subset korpus kecil: semua dokumen relevan untuk query_ids + dokumen acak lain."""
    relevant = {d for q in query_ids for d in qrels.get(q, {}) if d in corpus}
    others = sorted(d for d in corpus if d not in relevant)
    n_extra = max(0, size - len(relevant))
    extra = set(random.Random(seed).sample(others, min(n_extra, len(others))))
    keep = relevant | extra
    # pertahankan urutan asli korpus
    return {d: corpus[d] for d in corpus if d in keep}
