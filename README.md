# Pencarian Dokumen pada BEIR SciFact: BM25 vs. *Dense Retrieval*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/belvahector-ship-it/RetrievalInformasi_Tugas3_P31.2025.02702_BelvaFC/blob/main/notebooks/RI_Tugas3_SciFact_BM25_vs_Dense.ipynb)

Tugas 3 mata kuliah **Pengambilan Informasi (Retrieval Informasi)**, Program Studi S2 Magister Teknik Informatika, Universitas Dian Nuswantoro (UDINUS).
Penulis: **Belva Fahrozi Chiangmaitri**.

Repositori ini membangun sistem pencarian dokumen pada dataset BEIR SciFact dan membandingkan pencarian leksikal (BM25) dengan pencarian semantik (*dense retrieval* berbasis *sentence-transformers*), ditambah *hybrid retrieval* dengan Reciprocal Rank Fusion sebagai bonus. Tahapan mengikuti instruksi tugas:

1. Menyiapkan pustaka (`rank_bm25`, `sentence-transformers`, dst.)
2. Membaca empat file dataset (termasuk `corpus.jsonl`)
3. Mengambil 5 kueri secara acak
4. Pencarian BM25 (leksikal)
5. Pencarian *sentence-transformers* (dense/semantik)
6. Menghitung kinerja leksikal dan semantik
7. Analisis

Hasil eksperimen dan analisis disajikan di laporan, bukan di README ini.

## Dataset

BEIR SciFact (Wadden dkk., 2020; Thakur dkk., 2021), diunduh dari zip resmi BEIR:
`https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip`
(sha256 `536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165`, diverifikasi otomatis).

| File | Isi |
|---|---|
| `scifact/corpus.jsonl` | 5.183 dokumen (`_id`, `title`, `text`) |
| `scifact/queries.jsonl` | 1.109 kueri (`_id`, `text`) |
| `scifact/qrels/train.tsv` | penilaian relevansi split *train* |
| `scifact/qrels/test.tsv` | 339 baris, 300 kueri unik (`query-id`, `corpus-id`, `score`) |

Paket `beir` sengaja tidak dipakai; *loader* ditulis sendiri di `src/data.py`.

## Metode

| Metode | Keterangan |
|---|---|
| **BM25** | `rank_bm25.BM25Okapi` dengan parameter *default* pustaka (k1 = 1,5; b = 0,75; epsilon = 0,25). Praproses: *lowercase*, token regex `[a-z0-9]+`, hapus `ENGLISH_STOP_WORDS` scikit-learn, tanpa *stemming*. |
| **MiniLM** | `sentence-transformers/all-MiniLM-L6-v2`, *cosine similarity* (embedding dinormalisasi). |
| **TAS-B** | `sentence-transformers/msmarco-distilbert-base-tas-b`, *dot product* tanpa normalisasi sesuai *model card*. |
| **RRF-BM25+MiniLM**, **RRF-BM25+TAS-B** | Reciprocal Rank Fusion: `score(d) = Σ 1/(60 + rank)`, rank mulai 1, atas daftar top-100. |

## Keputusan desain yang dikunci

| Aspek | Keputusan |
|---|---|
| Teks dokumen | `title + " " + text` |
| *Seed* | 42 |
| 5 kueri acak | `random.Random(42).sample(sorted(test_query_ids), 5)`, hanya dari kueri yang memiliki *qrels* test |
| Cakupan evaluasi | (a) 5 kueri acak (sesuai instruksi); (b) seluruh 300 kueri test (validasi) |
| Cutoff | k = 10; daftar top-100 disimpan untuk fusi |
| Metrik | nDCG@10, Precision@10, Recall@10, MRR@10, MAP@10; relevansi biner (skor *qrels* > 0); diimplementasikan sendiri di `src/evaluate.py` |
| MAP@10 | gaya trec_eval `map_cut.10`: dipotong di 10, dinormalisasi dengan total dokumen relevan |
| Validasi metrik | *unit test* dengan contoh hitung manual; validasi silang dengan pustaka `ir-measures` (dipasang otomatis di Colab, opsional di lokal) |
| Pemecahan skor seri | BM25 dan *dense*: urutan dokumen di `corpus.jsonl`; RRF: rank terbaik, lalu doc_id menaik |
| Revisi model | MiniLM `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; TAS-B `b12d9352e776979147078a8975a4885042984fd1` |

## Cara menjalankan

### Google Colab (disarankan)

Klik lencana **Open in Colab** di atas, pilih *runtime* GPU, lalu jalankan semua sel secara berurutan. Sel pertama melakukan `git clone` repositori ke `/content` (atau `git pull` bila sudah ada) dan memasang dependensi dari `requirements-colab.txt`. Jika Colab meminta *restart runtime*, lakukan *restart* lalu jalankan ulang dari Tahap 1. Sel terakhir mengemas folder `results/` menjadi zip dan mengunduhnya.

### Lokal

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # atau torch versi CUDA
pip install -r requirements.txt
pip install ir-measures==0.4.3                                       # opsional, untuk validasi silang

python -m unittest discover -s tests -v   # uji metrik dan RRF
python -m src.run_pipeline --smoke        # uji cepat (subset 200 dokumen) -> results/smoke/
python -m src.run_pipeline                # eksperimen penuh -> results/
```

Mode `--smoke` memakai 5 kueri yang sama dan subset korpus (semua dokumen relevan kelima kueri + dokumen acak hingga 200). Angka dari mode ini **bukan hasil eksperimen** dan diberi label "SMOKE TEST — bukan hasil".

### Keluaran (`results/`)

| File | Isi |
|---|---|
| `summary_5queries.csv`, `summary_300queries.csv` | rata-rata metrik per metode |
| `per_query_5queries.csv`, `per_query_300queries.csv` | metrik per kueri |
| `top10_5queries.csv` | top-10 tiap metode untuk 5 kueri (doc id, cuplikan judul, skor, relevan ya/tidak) |
| `overlap_top10_*.csv` | irisan top-10 BM25 dengan tiap model *dense* |
| `timing.csv` | waktu indeks/encode/pencarian per metode |
| `environment.json` | versi Python dan pustaka, *device*, revisi model, parameter |
| `ir_measures_crosscheck.csv` | validasi silang dengan `ir-measures` per metode dan metrik: rata-rata, selisih absolut maksimum per kueri untuk skor berbasis peringkat dan skor asli (dibuat bila `ir-measures` terpasang; bila tidak, pipeline mencetak peringatan) |

CLI dan *notebook* memakai fungsi penyimpanan yang sama (`save_outputs` di `src/run_pipeline.py`), sehingga berkas yang dihasilkan identik.

## Catatan reprodusibilitas

- *Seed* tetap (42) untuk pemilihan kueri dan subset *smoke*.
- Ada dua berkas dependensi:
  - `requirements.txt` — *pin* lengkap untuk lingkungan lokal (sentence-transformers, rank-bm25, scikit-learn, pandas, numpy, tqdm).
  - `requirements-colab.txt` — hanya paket yang belum ada di Colab (sentence-transformers, rank-bm25, ir-measures). numpy, pandas, scikit-learn, dan tqdm memakai versi bawaan Colab agar tidak bentrok dengan paket Colab lain.
- **`torch` sengaja tidak di-*pin*** di kedua berkas: Colab sudah menyediakan `torch` dengan dukungan CUDA yang sesuai dengan GPU-nya, dan memasang ulang versi lain berisiko merusak kecocokan CUDA.
- Versi semua pustaka yang benar-benar terpakai (termasuk torch dan paket bawaan Colab) dicatat otomatis di `results/environment.json`.
- Model dimuat dengan revisi (*commit hash*) Hugging Face yang terkunci.
- Dataset diverifikasi dengan sha256 sebelum dibaca.
- Hasil *dense retrieval* pada GPU dan CPU dapat berbeda sangat kecil akibat aritmetika *floating point*.
- Pemecahan skor seri ditetapkan secara deterministik (lihat tabel keputusan desain). trec_eval (dipakai `ir-measures` untuk nDCG/P/R/MAP lewat pytrec_eval) memecah seri berdasarkan doc_id menurun, sedangkan MRR@10 dihitung `ir-measures` lewat *provider* lain dengan aturan seri sendiri. Akibatnya, bila ada skor seri di sekitar dokumen relevan, MRR/nDCG/MAP dapat berbeda sangat kecil. Karena itu validasi silang melaporkan dua perbandingan: dengan skor berbasis peringkat (urutan identik, selisih harus 0) dan dengan skor asli (selisih absolut maksimum ditampilkan apa adanya).

## Catatan keterbatasan

- Panjang input maksimum kedua model berbeda: `all-MiniLM-L6-v2` memotong teks pada 256 token, sedangkan `msmarco-distilbert-base-tas-b` pada 512 token. Dokumen SciFact (judul + abstrak) dapat melebihi 256 token, sehingga perbedaan ini menjadi faktor pengganggu (*confounder*) saat membandingkan kedua model *dense*.
- Embedding MiniLM sudah dinormalisasi oleh modul `Normalize` bawaan model; `normalize_embeddings=True` tetap diberikan agar eksplisit.

## Struktur proyek

```
src/
  config.py         # seed, k, URL, sha256, model + revisi + similarity, path
  data.py           # unduh + verifikasi zip, loader 4 file, sampling kueri, subset smoke
  bm25_search.py    # BM25 (rank_bm25)
  dense_search.py   # sentence-transformers (MiniLM, TAS-B)
  hybrid.py         # Reciprocal Rank Fusion
  evaluate.py       # nDCG/P/R/MRR/MAP @10, overlap, cross-check opsional
  run_pipeline.py   # CLI: python -m src.run_pipeline [--smoke] [--out DIR]
tests/
  test_evaluate.py
  test_hybrid.py
notebooks/
  RI_Tugas3_SciFact_BM25_vs_Dense.ipynb
requirements.txt        # pin lengkap (lokal)
requirements-colab.txt  # paket tambahan untuk Colab
LICENSE
```

## Referensi

- Wadden, D., Lin, S., Lo, K., Wang, L. L., van Zuylen, M., Cohan, A., & Hajishirzi, H. (2020). Fact or Fiction: Verifying Scientific Claims. *EMNLP 2020*. https://doi.org/10.18653/v1/2020.emnlp-main.609
- Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., & Gurevych, I. (2021). BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. *NeurIPS 2021 Datasets and Benchmarks*. https://doi.org/10.48550/arXiv.2104.08663
- Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in Information Retrieval*, 3(4), 333–389. https://doi.org/10.1561/1500000019
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP-IJCNLP 2019*. https://doi.org/10.18653/v1/D19-1410
- Hofstätter, S., Lin, S.-C., Yang, J.-H., Lin, J., & Hanbury, A. (2021). Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling. *SIGIR '21*. https://doi.org/10.1145/3404835.3462891
- Cormack, G. V., Clarke, C. L. A., & Büttcher, S. (2009). Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. *SIGIR '09*. https://doi.org/10.1145/1571941.1572114

## Kredit pustaka

- [rank_bm25](https://github.com/dorianbrown/rank_bm25) — Apache-2.0
- [sentence-transformers](https://github.com/UKPLab/sentence-transformers) — Apache-2.0
- scikit-learn (BSD-3-Clause), pandas (BSD-3-Clause), NumPy (BSD-3-Clause), PyTorch (BSD-3-Clause), tqdm (MPL-2.0/MIT)
- Model: `all-MiniLM-L6-v2` dan `msmarco-distilbert-base-tas-b` dari Hugging Face (lisensi Apache-2.0 sesuai *model card* masing-masing)

## Lisensi

Kode dirilis di bawah lisensi MIT — lihat [LICENSE](LICENSE). Dataset SciFact dan model mengikuti lisensinya masing-masing.
