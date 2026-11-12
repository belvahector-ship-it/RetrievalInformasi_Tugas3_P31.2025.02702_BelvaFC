"""Pipeline lengkap: BM25 vs dense retrieval (MiniLM, TAS-B) + RRF pada BEIR SciFact.

Pemakaian:
    python -m src.run_pipeline                 # run penuh -> results/
    python -m src.run_pipeline --smoke         # uji cepat -> results/smoke/
    python -m src.run_pipeline --out hasil/    # folder keluaran lain

Fungsi-fungsi di modul ini juga dipanggil dari notebook Colab per tahap.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from importlib import metadata
from pathlib import Path

import pandas as pd

from . import config
from .bm25_search import BM25Searcher, bm25_params
from .data import (count_qrels_rows, download_scifact, load_all, sample_queries,
                   smoke_subset)
from .dense_search import DenseSearcher, get_device
from .evaluate import (METRIC_NAMES, cross_check_ir_measures, evaluate_run,
                       overlap_at_k, relevant_set, summarize)
from .hybrid import rrf_runs

SMOKE_LABEL = "SMOKE TEST — bukan hasil"
PACKAGES = ["torch", "sentence-transformers", "transformers", "rank-bm25",
            "scikit-learn", "pandas", "numpy", "tqdm", "ir-measures"]


# ---------------------------------------------------------------------------
# Tahap 2: membaca data
# ---------------------------------------------------------------------------
def prepare_data(smoke: bool = False, data_dir: Path = config.DATA_DIR) -> dict:
    """Unduh + baca 4 file, pilih 5 kueri acak, (opsional) buat subset smoke.

    Kembalikan dict berisi corpus, queries, qrels_train, qrels_test, ds_dir,
    random_qids (5 kueri), eval_qids (semua kueri test atau 5 kueri saat smoke),
    dan full_corpus_size.
    """
    ds_dir = download_scifact(data_dir)
    data = load_all(ds_dir)
    data["ds_dir"] = ds_dir
    data["full_corpus_size"] = len(data["corpus"])
    data["random_qids"] = sample_queries(data["qrels_test"])
    if smoke:
        data["corpus"] = smoke_subset(data["corpus"], data["qrels_test"], data["random_qids"])
        data["eval_qids"] = list(data["random_qids"])
    else:
        data["eval_qids"] = sorted(data["qrels_test"].keys())
    data["smoke"] = smoke
    return data


def dataset_overview(data: dict) -> pd.DataFrame:
    """Ringkasan jumlah isi keempat file."""
    ds = data["ds_dir"]
    rows = [
        {"file": "corpus.jsonl", "jumlah": data["full_corpus_size"], "satuan": "dokumen"},
        {"file": "queries.jsonl", "jumlah": len(data["queries"]), "satuan": "kueri"},
        {"file": "qrels/train.tsv", "jumlah": count_qrels_rows(ds, "train"),
         "satuan": f"baris ({len(data['qrels_train'])} kueri unik)"},
        {"file": "qrels/test.tsv", "jumlah": count_qrels_rows(ds, "test"),
         "satuan": f"baris ({len(data['qrels_test'])} kueri unik)"},
    ]
    return pd.DataFrame(rows)


def file_examples(data: dict, n: int = 3) -> dict[str, pd.DataFrame]:
    """Contoh n baris pertama dari tiap file."""
    corpus_ex = pd.DataFrame([{"_id": d, **v} for d, v in list(data["corpus"].items())[:n]])
    queries_ex = pd.DataFrame([{"_id": q, "text": t} for q, t in list(data["queries"].items())[:n]])

    def qrels_ex(qrels):
        rows = [{"query-id": q, "corpus-id": d, "score": s}
                for q, docs in qrels.items() for d, s in docs.items()]
        return pd.DataFrame(rows[:n])

    return {"corpus.jsonl": corpus_ex, "queries.jsonl": queries_ex,
            "qrels/train.tsv": qrels_ex(data["qrels_train"]),
            "qrels/test.tsv": qrels_ex(data["qrels_test"])}


# ---------------------------------------------------------------------------
# Tahap 3: 5 kueri acak
# ---------------------------------------------------------------------------
def random_queries_table(data: dict) -> pd.DataFrame:
    rows = []
    for qid in data["random_qids"]:
        rel = relevant_set(data["qrels_test"][qid])
        rows.append({"query_id": qid, "query": data["queries"][qid],
                     "n_relevant": len(rel), "relevant_doc_ids": ", ".join(sorted(rel))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tahap 4-5 + bonus: menjalankan metode
# ---------------------------------------------------------------------------
def run_bm25(data: dict) -> tuple[dict, dict]:
    queries = {q: data["queries"][q] for q in data["eval_qids"]}
    s = BM25Searcher(data["corpus"])
    run = s.search(queries)
    timing = {"method": "BM25", "index_seconds": s.index_seconds, "load_seconds": 0.0,
              "search_seconds": s.search_seconds, "n_queries": len(queries),
              "n_docs": len(data["corpus"])}
    timing.update({f"bm25_{k}": v for k, v in s.params.items()})
    return run, timing


def run_dense(data: dict, key: str) -> tuple[dict, dict]:
    queries = {q: data["queries"][q] for q in data["eval_qids"]}
    s = DenseSearcher(key)
    s.index(data["corpus"])
    run = s.search(queries)
    timing = {"method": key, "index_seconds": s.index_seconds, "load_seconds": s.load_seconds,
              "search_seconds": s.search_seconds, "n_queries": len(queries),
              "n_docs": len(data["corpus"]), "device": s.device, "similarity": s.similarity}
    return run, timing


def run_hybrid(runs: dict) -> tuple[dict, list[dict]]:
    out, timings = {}, []
    for dense in ("MiniLM", "TAS-B"):
        name = f"RRF-BM25+{dense}"
        t0 = time.perf_counter()
        out[name] = rrf_runs([runs["BM25"], runs[dense]])
        timings.append({"method": name, "index_seconds": 0.0, "load_seconds": 0.0,
                        "search_seconds": time.perf_counter() - t0,
                        "n_queries": len(out[name]), "rrf_k": config.RRF_K})
    return out, timings


def top10_table(run: dict, data: dict, qid: str, k: int = config.K,
                snippet: int = 80) -> pd.DataFrame:
    """Tabel top-k untuk satu kueri: rank, doc_id, cuplikan judul, skor, relevan."""
    rel = relevant_set(data["qrels_test"].get(qid, {}))
    rows = []
    for rank, (doc_id, score) in enumerate(run[qid][:k], start=1):
        title = data["corpus"][doc_id]["title"]
        rows.append({"rank": rank, "doc_id": doc_id,
                     "title": title if len(title) <= snippet else title[:snippet] + "...",
                     "score": round(score, 4), "relevant": "ya" if doc_id in rel else "tidak"})
    return pd.DataFrame(rows)


def all_top10(runs: dict, data: dict) -> pd.DataFrame:
    frames = []
    for method, run in runs.items():
        for qid in data["random_qids"]:
            df = top10_table(run, data, qid)
            df.insert(0, "query", data["queries"][qid])
            df.insert(0, "query_id", qid)
            df.insert(0, "method", method)
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Tahap 6-7: evaluasi + analisis
# ---------------------------------------------------------------------------
def evaluate_all(runs: dict, data: dict, qids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Kembalikan (summary per metode, per-query long format)."""
    per_query = {m: evaluate_run(runs[m], data["qrels_test"], qids, config.K)
                 for m in config.METHOD_ORDER if m in runs}
    summary = summarize(per_query)
    long = pd.concat([df.assign(method=m) for m, df in per_query.items()]).reset_index()
    long = long[["method", "query_id", "n_relevant"] + METRIC_NAMES]
    return summary, long


def per_query_comparison(per_query_long: pd.DataFrame, metric: str = "nDCG@10") -> pd.DataFrame:
    """Tabel lebar: baris = kueri, kolom = metode, nilai = metrik terpilih."""
    return per_query_long.pivot(index="query_id", columns="method", values=metric)[
        [m for m in config.METHOD_ORDER if m in per_query_long["method"].unique()]]


def overlap_table(runs: dict, qids: list[str]) -> pd.DataFrame:
    frames = []
    for dense in ("MiniLM", "TAS-B"):
        df = overlap_at_k(runs["BM25"], runs[dense], qids, config.K)
        frames.append(df.assign(pair=f"BM25 vs {dense}"))
    return pd.concat(frames).reset_index()[["pair", "query_id", f"overlap@{config.K}", "jaccard"]]


def environment_info() -> dict:
    versions = {}
    for pkg in PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "device": get_device(),
        "packages": versions,
        "seed": config.SEED,
        "k": config.K,
        "top_n": config.TOP_N,
        "dataset_url": config.DATASET_URL,
        "dataset_sha256": config.DATASET_SHA256,
        "bm25_params": config.BM25_PARAMS,
        "bm25_params_runtime": bm25_params(),
        "rrf_k": config.RRF_K,
        "dense_models": config.DENSE_MODELS,
    }
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


# ---------------------------------------------------------------------------
# Penyimpanan keluaran (dipakai bersama oleh CLI dan notebook)
# ---------------------------------------------------------------------------
def default_out_dir(smoke: bool) -> Path:
    return config.RESULTS_DIR / "smoke" if smoke else config.RESULTS_DIR


def run_crosscheck(runs: dict, data: dict) -> pd.DataFrame | None:
    """Validasi silang semua metode terhadap ir-measures; None bila tidak tersedia."""
    checks = []
    for m in config.METHOD_ORDER:
        chk = cross_check_ir_measures(runs[m], data["qrels_test"], data["eval_qids"], config.K)
        if chk is None:
            print("PERINGATAN: validasi silang dengan ir-measures TIDAK dijalankan "
                  "(pasang dengan: pip install ir-measures==0.4.3). "
                  "File ir_measures_crosscheck.csv tidak dibuat.")
            return None
        checks.append(chk.assign(method=m).reset_index())
    return pd.concat(checks, ignore_index=True)


def save_outputs(runs: dict, timings: list[dict], data: dict,
                 out_dir: Path | None = None, verbose: bool = True) -> dict:
    """Tulis seluruh keluaran ke out_dir. Dipanggil oleh CLI maupun notebook.

    File: random_queries.csv, top10_5queries.csv, timing.csv,
    summary_<scope>.csv, per_query_<scope>.csv, overlap_top10_<scope>.csv
    (scope = 5queries, ditambah 300queries bila bukan smoke),
    ir_measures_crosscheck.csv (bila ir-measures tersedia), environment.json,
    serta SMOKE_TEST.txt pada mode smoke.
    Mengembalikan {"summaries": {scope: df}, "per_query": {scope: df}, "crosscheck": df | None}.
    """
    smoke = data["smoke"]
    out_dir = Path(out_dir) if out_dir else default_out_dir(smoke)
    out_dir.mkdir(parents=True, exist_ok=True)
    if smoke:
        (out_dir / "SMOKE_TEST.txt").write_text(
            f"{SMOKE_LABEL}\nKorpus subset {len(data['corpus'])} dokumen; "
            "angka di folder ini hanya untuk menguji pipeline.\n", encoding="utf-8")

    random_queries_table(data).to_csv(out_dir / "random_queries.csv", index=False)
    all_top10(runs, data).to_csv(out_dir / "top10_5queries.csv", index=False)
    pd.DataFrame(timings).to_csv(out_dir / "timing.csv", index=False)

    scopes = {"5queries": data["random_qids"]}
    if not smoke:
        scopes["300queries"] = data["eval_qids"]
    summaries, per_query = {}, {}
    for scope, qids in scopes.items():
        summary, long = evaluate_all(runs, data, qids)
        if smoke:
            summary.insert(0, "label", SMOKE_LABEL)
        summary.to_csv(out_dir / f"summary_{scope}.csv")
        long.to_csv(out_dir / f"per_query_{scope}.csv", index=False)
        overlap_table(runs, qids).to_csv(out_dir / f"overlap_top10_{scope}.csv", index=False)
        summaries[scope], per_query[scope] = summary, long
        if verbose:
            print(f"\nRingkasan metrik ({scope}){' — ' + SMOKE_LABEL if smoke else ''}:")
            print(summary.round(4).to_string())

    chk = run_crosscheck(runs, data)
    if chk is not None:
        chk.to_csv(out_dir / "ir_measures_crosscheck.csv", index=False)
        if verbose:
            print(f"\nValidasi silang ir-measures (skor berbasis peringkat): "
                  f"{int(chk['match_rank'].sum())}/{len(chk)} cocok; "
                  f"selisih absolut maksimum per kueri dengan skor asli: "
                  f"{chk['max_abs_diff_score'].max():.6f}")

    env = environment_info()
    env.update({"smoke": smoke, "n_docs_indexed": len(data["corpus"]),
                "n_queries_evaluated": len(data["eval_qids"]),
                "random_query_ids": data["random_qids"]})
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2, ensure_ascii=False),
                                              encoding="utf-8")
    if verbose:
        print(f"\nSemua keluaran ditulis ke: {out_dir}")
    return {"summaries": summaries, "per_query": per_query, "crosscheck": chk}


# ---------------------------------------------------------------------------
# Orkestrasi CLI
# ---------------------------------------------------------------------------
def run_all(smoke: bool = False, out_dir: Path | None = None) -> dict:
    if smoke:
        print(f"===== {SMOKE_LABEL} =====")

    # Tahap 2-3
    data = prepare_data(smoke=smoke)
    print(dataset_overview(data).to_string(index=False))
    print(f"\nKorpus yang diindeks: {len(data['corpus'])} dokumen; "
          f"kueri dievaluasi: {len(data['eval_qids'])}")
    print("\n5 kueri acak (seed 42):")
    print(random_queries_table(data).to_string(index=False))

    # Tahap 4-5 + bonus
    runs, timings = {}, []
    print("\n[BM25] indeks + pencarian ...")
    runs["BM25"], t = run_bm25(data)
    timings.append(t)
    for key in config.DENSE_MODELS:
        print(f"\n[{key}] memuat model, encode korpus, pencarian ...")
        runs[key], t = run_dense(data, key)
        timings.append(t)
    hyb, t = run_hybrid(runs)
    runs.update(hyb)
    timings.extend(t)

    # Tahap 6 + simpan
    saved = save_outputs(runs, timings, data, out_dir)
    if smoke:
        print(f"===== {SMOKE_LABEL} =====")
    return {"data": data, "runs": runs, "timings": timings, **saved}


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="SciFact: BM25 vs dense retrieval + RRF")
    ap.add_argument("--smoke", action="store_true",
                    help="uji cepat: 5 kueri, subset korpus 200 dokumen")
    ap.add_argument("--out", type=Path, default=None, help="folder keluaran")
    args = ap.parse_args(argv)
    run_all(smoke=args.smoke, out_dir=args.out)


if __name__ == "__main__":
    main()
