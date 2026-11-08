# Student Social Profile Segmentation Using Clustering

Proyek UTS Machine Learning (Milestone 1). Proyek ini mengelompokkan profil siswa di jaringan sosial berdasarkan minat yang tercatat, tanpa label (unsupervised learning).

## Anggota

| Nama | NIM |
|---|---|
| Michael Andreas Tjendra | 231111210 |
| Diky Diwa Suwanto | 231110760 |
| Enjel Boyzest Saragih | 231111169 |

## Ringkasan masalah

Dataset berisi 15.000 profil siswa dengan 36 fitur minat (jumlah kemunculan kata kunci) dan 4 atribut profil (`gradyear`, `gender`, `age`, `NumberOffriends`). Fitur minat dipakai untuk membentuk klaster. Atribut profil hanya dipakai untuk menjelaskan hasil, bukan sebagai input clustering.

Tujuan Milestone 1:

1. Memahami struktur, kualitas, dan sparsity data.
2. Menyiapkan dua representasi fitur: `log1p` + standardisasi, dan biner.
3. Menjalankan baseline dan beberapa kandidat model, lalu membandingkannya secara konsisten.
4. Menafsirkan karakteristik klaster dengan atribut profil.

## Dataset

Sumber: [Students Social Network Profile Clustering (Kaggle)](https://www.kaggle.com/datasets/zabihullah18/students-social-network-profile-clustering)

File yang dipakai: `data/Clustering_Marketing.csv`. Di Kaggle, file ini bernama `03_Clustering_Marketing.csv`, dan notebook mengenali kedua nama itu. Periksa lisensi dan ketentuan berbagi dataset di halaman Kaggle sebelum menyebarkan ulang CSV.

## Struktur folder

```text
.
├── data/
│   └── Clustering_Marketing.csv
├── notebooks/
│   ├── eda_data_understanding.ipynb   # EDA dan pemahaman data
│   ├── data_preparation.ipynb         # preprocessing dan feature engineering
│   └── model_development.ipynb        # baseline, kandidat model, evaluasi, profil klaster
├── outputs/
│   ├── EDA/                           # tabel EDA (CSV)
│   │   └── figures/                   # gambar EDA
│   ├── preprocessing/                 # data hasil preprocessing dan scaler
│   ├── figures/                       # gambar preprocessing dan modeling
│   ├── tables/                        # tabel hasil eksperimen (CSV)
│   └── models/                        # model tersimpan (.joblib)
├── reports/                           # laporan UTS (PDF)
├── requirements.txt
├── requirements-lock.txt
└── README.md
```

## Cara menjalankan

1. Buat dan aktifkan virtual environment, lalu pasang paket:

   ```bash
   python -m venv .venv
   # Windows (PowerShell): .\.venv\Scripts\Activate.ps1
   # macOS / Linux:        source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

   `requirements.txt` memakai rentang versi. Untuk lingkungan yang sama persis dengan pengujian, pakai `requirements-lock.txt`.

2. Letakkan dataset di `data/Clustering_Marketing.csv`.
3. Buka notebook dari folder `notebooks/` (pilih kernel dari `.venv`), lalu jalankan **Restart Kernel and Run All** secara berurutan:
   1. `eda_data_understanding.ipynb`
   2. `data_preparation.ipynb`
   3. `model_development.ipynb`

   Urutan ini wajib karena `model_development` membaca hasil `data_preparation` dari `outputs/preprocessing/`.

Semua keluaran tersimpan di folder `outputs/`. Notebook mencari folder proyek dan dataset secara otomatis, jadi bisa dijalankan dari folder `notebooks/`, dari folder induknya, atau di Colab dan Kaggle (taruh dataset di folder kerja atau tambahkan sebagai Input di Kaggle).

## Hasil awal

Baseline adalah K-Means pada representasi `log1p`-scaled dengan k=2. Profil tanpa minat tercatat (1.238 profil, 8,25%) dipisah sebagai Segmen 0 sebelum clustering. Perbandingan dilakukan pada sampel 2.500 profil aktif:

| Model | Peran | Klaster | Noise | Silhouette | Davies-Bouldin | Calinski-Harabasz | Klaster terkecil |
|---|---|---:|---:|---:|---:|---:|---:|
| K-Means (log1p-scaled) | Baseline | 2 | 0% | 0,3025 | 3,2524 | 157,2216 | 14,48% |
| K-Means (biner) | Kandidat | 2 | 0% | 0,1652 | 4,0514 | 127,1706 | 29,64% |
| Agglomerative Ward | Kandidat | 2 | 0% | 0,4493 | 1,9042 | 99,9252 | 2,12% |
| DBSCAN (eps=1,37) | Kandidat | 14 | 88,00% | 0,6268* | 0,5828* | 109,0548* | 3,67%* |
| Gaussian Mixture (diagonal) | Kandidat | 2 | 0% | 0,2402 | 3,9331 | 117,9022 | 18,72% |

*DBSCAN dihitung hanya pada 300 inlier dari 2.500 profil, sehingga skornya tidak sebanding dengan model lain.

K-Means (log1p-scaled, k=2) menjadi kandidat terbaik sementara, bukan model final. Pembagian k=2 terutama memisahkan profil dengan minat sedikit dari profil dengan minat banyak (korelasi 0,63 dengan total minat). Analisis k=3 sampai 5 menunjukkan kelompok topik kecil (3% sampai 6% profil) yang belum diuji kestabilannya.

## Artefak penting

| Berkas | Isi |
|---|---|
| `outputs/tables/model_comparison.csv` | Perbandingan baseline dan kandidat |
| `outputs/tables/k_selection_metrics.csv` | Pemilihan k untuk K-Means |
| `outputs/tables/k_sensitivity_profiles.csv` | Profil klaster pada k=3 sampai 5 |
| `outputs/tables/cluster_profile_summary.csv` | Profil segmen model sementara |
| `outputs/tables/cluster_gender_profile.csv` | Komposisi gender per segmen |
| `outputs/tables/clustered_student_profiles.csv` | Label klaster tiap profil |
| `outputs/tables/preprocessing_summary.csv` | Ringkasan preprocessing |
| `outputs/models/best_clustering_model.joblib` | Model K-Means sementara beserta metadata |

## Catatan dan batasan

- Dataset tidak punya ID siswa. Pola minat yang sama belum tentu berarti data ganda, sehingga baris duplikat tidak dihapus.
- Profil dengan semua fitur minat nol berarti tidak ada minat yang tercatat pada 36 fitur itu, bukan bukti bahwa siswa tidak punya minat.
- Sekitar 19,5% usia diimputasi dengan median angkatan, jadi perbedaan usia antar klaster tidak boleh dianggap temuan kuat.
- Metrik masih internal dan dihitung pada subsampel. Uji kestabilan segmen belum dilakukan.
- Segmen bersifat agregat dan tidak boleh dipakai untuk keputusan tentang individu.

## Rencana setelah UTS

1. Uji kestabilan segmen dengan Adjusted Rand Index antar seed dan antar subsampel.
2. Laporkan metrik model final pada seluruh 13.762 profil aktif.
3. Perdalam analisis k=3 sampai 5 (kestabilan kelompok kecil dan penamaan segmen).
4. Tuning terbatas: jumlah inisialisasi K-Means, `eps` DBSCAN, jenis kovarians GMM, linkage hierarkis.
5. Bandingkan hasil dengan dan tanpa baris duplikat.
6. Uji kepekaan terhadap imputasi usia.
7. Validasi domain atas profil segmen dan analisis kesalahan.
8. Uji representasi yang menghilangkan efek total minat per profil (misalnya normalisasi baris atau jarak cosine pada data biner).
9. Pilih model final, lalu siapkan presentasi dan tanya jawab.