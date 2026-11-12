"""Metrik evaluasi retrieval, diimplementasikan sendiri (relevansi biner).

Dokumen dianggap relevan bila skor qrels > 0. Untuk SciFact semua skor qrels
bernilai 1, sehingga nDCG biner identik dengan nDCG bergradasi.

Definisi (k = 10, R = jumlah dokumen relevan untuk kueri, rel_i in {0,1} pada rank i):
    Precision@k = (sum_{i<=k} rel_i) / k
    Recall@k    = (sum_{i<=k} rel_i) / R
    MRR@k       = 1 / rank dokumen relevan pertama bila rank <= k, selain itu 0
    nDCG@k      = DCG@k / IDCG@k,  DCG@k = sum_{i<=k} rel_i / log2(i + 1),
                  IDCG@k = sum_{i=1}^{min(R,k)} 1 / log2(i + 1)
    AP@k (MAP@k bila dirata-rata) = (sum_{i<=k} rel_i * Precision@i) / R
        -> gaya trec_eval `map_cut.10`: dipotong di k tetapi dinormalisasi dengan
           TOTAL dokumen relevan R (bukan min(R, k)). Pilihan ini membuat AP@k
           bisa < 1 walaupun top-k sempurna bila R > k; untuk SciFact R kecil
           (umumnya 1-2), jadi dampaknya minim.

Kueri tanpa dokumen relevan dilewati (tidak ikut dirata-rata).

Pemecahan skor seri: metrik di sini dihitung dari URUTAN run apa adanya.
Urutan itu dibentuk oleh modul pencarian: BM25 dan dense memecah seri menurut
urutan dokumen di korpus; RRF memecah seri menurut rank terbaik lalu doc_id.
trec_eval (pytrec_eval; dipakai ir-measures untuk nDCG/P/R/AP) memecah seri
menurut doc_id menurun, sedangkan RR@10 dihitung ir-measures lewat provider lain
(trec_eval tidak punya recip_rank dengan cutoff) yang memakai aturan sendiri.
Pada kasus seri, nilai MRR/nDCG/MAP bisa berbeda sedikit (lihat
cross_check_ir_measures).
"""
from __future__ import annotations

import math

import pandas as pd

METRIC_NAMES = ["nDCG@10", "P@10", "R@10", "MRR@10", "MAP@10"]


def query_metrics(ranked_ids: list[str], relevant: set[str], k: int = 10) -> dict[str, float]:
    """Hitung semua metrik untuk satu kueri."""
    R = len(relevant)
    if R == 0:
        raise ValueError("Kueri tanpa dokumen relevan tidak dapat dievaluasi.")
    top = ranked_ids[:k]
    hits = 0
    dcg = 0.0
    ap_sum = 0.0
    rr = 0.0
    for i, doc_id in enumerate(top, start=1):
        if doc_id in relevant:
            hits += 1
            dcg += 1.0 / math.log2(i + 1)
            ap_sum += hits / i
            if rr == 0.0:
                rr = 1.0 / i
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(R, k) + 1))
    return {
        f"nDCG@{k}": dcg / idcg,
        f"P@{k}": hits / k,
        f"R@{k}": hits / R,
        f"MRR@{k}": rr,
        f"MAP@{k}": ap_sum / R,
    }


def relevant_set(qrels_q: dict[str, int]) -> set[str]:
    return {d for d, s in qrels_q.items() if s > 0}


def evaluate_run(run: dict[str, list[tuple[str, float]]], qrels: dict[str, dict[str, int]],
                 query_ids: list[str], k: int = 10) -> pd.DataFrame:
    """Metrik per kueri sebagai DataFrame (index = query_id)."""
    rows = []
    for qid in query_ids:
        rel = relevant_set(qrels.get(qid, {}))
        if not rel:
            continue
        ranked = [d for d, _ in run.get(qid, [])]
        rows.append({"query_id": qid, "n_relevant": len(rel), **query_metrics(ranked, rel, k)})
    return pd.DataFrame(rows).set_index("query_id")


def summarize(per_query: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Rata-rata metrik per metode: {metode: per_query_df} -> DataFrame (baris = metode)."""
    out = {m: df[METRIC_NAMES].mean() for m, df in per_query.items()}
    summary = pd.DataFrame(out).T
    summary.index.name = "method"
    summary.insert(0, "n_queries", [len(df) for df in per_query.values()])
    return summary


def overlap_at_k(run_a: dict, run_b: dict, query_ids: list[str], k: int = 10) -> pd.DataFrame:
    """Jumlah dokumen yang sama di top-k dua run, per kueri (plus Jaccard)."""
    rows = []
    for qid in query_ids:
        a = {d for d, _ in run_a[qid][:k]}
        b = {d for d, _ in run_b[qid][:k]}
        inter = len(a & b)
        rows.append({"query_id": qid, f"overlap@{k}": inter,
                     "jaccard": inter / len(a | b) if (a | b) else 0.0})
    return pd.DataFrame(rows).set_index("query_id")


def cross_check_ir_measures(run: dict, qrels: dict, query_ids: list[str],
                            k: int = 10, tol: float = 1e-6):
    """OPSIONAL: bandingkan metrik buatan sendiri dengan pustaka `ir-measures`.

    Dua perbandingan dilakukan per kueri:
      1. Skor berbasis peringkat (len - posisi): urutan dokumen yang dilihat
         ir-measures identik dengan urutan kita, sehingga selisih seharusnya 0.
         Ini menguji kebenaran rumus metrik.
      2. Skor asli dari run: ir-measures mengurutkan ulang dokumen menurut skor.
         Untuk nDCG/P/R/AP (via pytrec_eval/trec_eval) skor seri dipecah menurut
         doc_id menurun; RR@10 dihitung provider lain dengan aturan seri sendiri.
         Pipeline ini memecah seri menurut urutan korpus (BM25/dense) atau rank
         terbaik lalu doc_id (RRF). Bila ada skor seri di sekitar dokumen relevan,
         MRR/nDCG/MAP bisa berbeda sedikit; selisih absolut maksimumnya
         dilaporkan agar terlihat.

    Mengembalikan DataFrame (index = metrik) berisi rata-rata kita, rata-rata
    ir-measures untuk kedua varian, dan selisih absolut maksimum per kueri;
    atau None bila ir-measures tidak tersedia.
    """
    try:
        import ir_measures
        from ir_measures import AP, P, R, RR, nDCG
    except Exception as exc:  # pustaka tidak terpasang / gagal build
        print(f"[cross-check] ir-measures tidak tersedia: {exc}")
        return None

    qids = [q for q in query_ids if relevant_set(qrels.get(q, {}))]
    sub_qrels = {q: {d: int(s) for d, s in qrels[q].items()} for q in qids}
    rank_run = {q: {d: float(len(run.get(q, [])) - i) for i, (d, _) in enumerate(run.get(q, []))}
                for q in qids}
    score_run = {q: {d: float(s) for d, s in run.get(q, [])} for q in qids}
    measures = {f"nDCG@{k}": nDCG @ k, f"P@{k}": P @ k, f"R@{k}": R @ k,
                f"MRR@{k}": RR @ k, f"MAP@{k}": AP @ k}
    name_of = {str(m): n for n, m in measures.items()}

    def per_query(r):
        df = pd.DataFrame(0.0, index=qids, columns=list(measures))
        for res in ir_measures.iter_calc(list(measures.values()), sub_qrels, r):
            df.loc[res.query_id, name_of[str(res.measure)]] = float(res.value)
        return df

    try:
        theirs_rank, theirs_score = per_query(rank_run), per_query(score_run)
    except Exception as exc:
        print(f"[cross-check] ir-measures gagal dijalankan: {exc}")
        return None
    ours = evaluate_run(run, qrels, qids, k)[list(measures)].reindex(qids)

    rows = []
    for name in measures:
        d_rank = (ours[name] - theirs_rank[name]).abs().max()
        d_score = (ours[name] - theirs_score[name]).abs().max()
        rows.append({"metric": name, "ours": ours[name].mean(),
                     "ir_measures_rank": theirs_rank[name].mean(),
                     "ir_measures_score": theirs_score[name].mean(),
                     "max_abs_diff_rank": float(d_rank), "max_abs_diff_score": float(d_score),
                     "match_rank": bool(d_rank <= tol)})
    return pd.DataFrame(rows).set_index("metric")
