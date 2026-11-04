# Laporan Pengerjaan Data Preparation & Feature Engineering

**Proyek:** Student Social Profile Segmentation Using Clustering  
**Dataset:** 15.000 Profil Siswa (*03_Clustering_Marketing.csv*)  
**Notebook Terkait:** [`notebooks/data_preparation.ipynb`](file:///c:/Users/boyze/OneDrive/Desktop/ML/Student-Social-Profile-Segmentation-Using-Clustering/notebooks/data_preparation.ipynb)  
**Status Tahapan:** Selesai dan Tervalidasi  

---

## 1. Pendahuluan & Latar Belakang

Tahap *Data Preparation* ini merupakan kelanjutan langsung dari temuan dan rekomendasi pada tahap *Exploratory Data Analysis* (EDA) di [`notebooks/eda_data_understanding.ipynb`](file:///c:/Users/boyze/OneDrive/Desktop/ML/Student-Social-Profile-Segmentation-Using-Clustering/notebooks/eda_data_understanding.ipynb).

Tujuan utama dari tahap ini adalah:
1. Membersihkan anomali kualitas data pada atribut profil siswa.
2. Memisahkan secara metodologis antara fitur profil (variabel interpretasi) dan 36 fitur minat (variabel input clustering).
3. Menyelesaikan ambiguitas data duplikat dan profil tanpa minat (*all-zero profiles*).
4. Merekayasa dua pendekatan representasi fitur minat (**Representasi A: Count Log1p + Scaled** dan **Representasi B: Binary Presence/Absence**) untuk diuji pada tahap pemodelan clustering ([`03_model_development.ipynb`](file:///c:/Users/boyze/OneDrive/Desktop/ML/Student-Social-Profile-Segmentation-Using-Clustering/notebooks/model_development.ipynb)).
5. Menyimpan seluruh artefak data bersih dan model transformer ke folder `outputs/preprocessing/`.

---

## 2. Pemisahan Peran Fitur (Clustering Space vs External Profiling)

Dalam analisis klaster tanpa supervisi (*unsupervised clustering*), pencampuran seluruh kolom ke dalam satu metrik jarak sering kali menimbulkan masalah *scale mismatch* dan mengaburkan esensi segmentasi. Oleh karena itu, fitur dipisahkan menjadi dua kelompok:

| Kelompok Fitur | Nama Kolom | Jumlah | Peran Metodologis |
|---|---|:---:|---|
| **Fitur Minat** (*Interest Features*) | `basketball`, `football`, `music`, `rock`, `god`, `shopping`, `dance`, ..., `drugs` | 36 | **Input Utama Clustering (Clustering Space):** Mengelompokkan siswa berdasarkan pola kesamaan preferensi minat dan topik. |
| **Fitur Profil** (*Profile Attributes*) | `gradyear`, `gender`, `age`, `NumberOffriends` | 4 | **Variabel Eksternal (Profiling & Context):** Digunakan pasca-clustering untuk mendeskripsikan karakteristik demografis dan sosial tiap klaster yang terbentuk. |

---

## 3. Pembersihan & Imputasi Fitur Profil Siswa

Berdasarkan investigasi kualitas data pada EDA, atribut profil memiliki beberapa masalah yang telah ditangani:

### 3.1. Penanganan Nilai Hilang pada `gender`
- **Kondisi Awal:** Sebanyak 1.337 baris (8,91%) memiliki nilai kosong (*missing*).
- **Tindakan:** Missing value diimputasi dengan kategori `"Unknown"` pada kolom `gender_clean`. Hal ini memastikan seluruh observasi tetap dapat diklasifikasikan saat analisis profil klaster tanpa membuang baris data.
- **Distribusi Akhir:**
  - `F` (Female): 11.054 (73,69%)
  - `M` (Male): 2.609 (17,39%)
  - `Unknown`: 1.337 (8,91%)

### 3.2. Pembersihan & Imputasi Presisi pada `age`
- **Kondisi Awal:**
  - 2.496 baris hilang (*null*).
  - 276 baris berisi teks lokal Jerman tanpa tahun (seperti `"16. Jun"`, `"17. Mrz"`, `"16. Dez"`).
  - 153 nilai numerik tidak masuk akal untuk populasi siswa SMA (usia < 13 tahun atau > 25 tahun, termasuk nilai ekstrem hingga 106,9 tahun).
- **Tindakan:**
  1. Konversi numerik otomatis dengan `pd.to_numeric(..., errors='coerce')` untuk mengubah teks invalid menjadi `NaN`.
  2. Menyaring nilai usia di luar interval wajar $[13, 25]$ menjadi `NaN` pada kolom `age_clean`.
  3. **Imputasi Berbasis Angkatan (`age_imputed`):** Dari data valid, ditemukan bahwa median usia siswa sangat konsisten terpaut tepat ~1 tahun per tahun kelulusan:
     - Angkatan 2006: Median 18,68 tahun
     - Angkatan 2007: Median 17,70 tahun
     - Angkatan 2008: Median 16,74 tahun
     - Angkatan 2009: Median 15,79 tahun  
     Nilai `NaN` diimputasi menggunakan median angkatan masing-masing, menghasilkan kolom `age_imputed` yang siap pakai untuk evaluasi demografi klaster.

### 3.3. Validasi `gradyear` & `NumberOffriends`
- `gradyear` divalidasi bertipe integer (2006–2009) tanpa ada data hilang.
- `NumberOffriends` divalidasi non-negatif dengan nilai median 20 teman dan nilai maksimum 605 teman.

---

## 4. Investigasi Duplikasi Data & Penanganan Profil All-Zero

### 4.1. Fakta Kunci Baris Duplikat Penuh
Pada EDA tercatat 266 baris duplikat penuh (melibatkan total 318 baris, atau 1,77% dataset). Investigasi lanjutan pada tahap ini menemukan temuan penting:
- **91,8% dari baris duplikat (292 baris) adalah profil "kosong"**: memiliki 0 teman, gender/umur tidak tercatat, dan nilai 0 pada seluruh 36 fitur minat.
- Hanya tersisa 26 baris (13 pasang) yang memiliki minimal 1 minat tercatat.
- Mengingat tidak adanya kolom `student_id` unik, kesamaan nilai ini adalah konsekuensi dari profil siswa yang tidak aktif mencatat minat, bukan kesalahan duplikasi input sistem.

### 4.2. Isolasi Profil All-Zero (*Segment 0: No Recorded Interests*)
Sebanyak **1.238 profil (8,25%)** memiliki nilai 0 di seluruh 36 fitur minat.

**Keputusan Metodologis:**
1. **Mencegah Distorsi Jarak:** Pada ruang berdimensi 36, profil all-zero berada tepat di titik origin $(0, 0, \dots, 0)$. Memasukkan profil ini ke dalam clustering K-Means atau Hierarchical akan menarik centroid secara artifisial dan berpotensi membentuk klaster semu (*artificial cluster*).
2. **Segmentasi Alami:** Profil tanpa catatan minat dikelompokkan secara terpisah sebagai **"Segment 0: No Recorded Interests"**.
3. **Fokus Populasi Aktif:** Algoritma clustering pada `03_model_development.ipynb` difokuskan pada **13.762 profil aktif** (`X_active`) yang memiliki sinyal preferensi nyata.
4. **Penyelesaian Masalah Duplikat:** Pemisahan ini secara otomatis mengisolasi lebih dari 91% kasus duplikat tanpa perlu menghapus profil dari dataset populasi awal.

---

## 5. Rekayasa Representasi Fitur Minat (Feature Engineering)

Untuk memastikan pengujian algoritma clustering berjalan optimal, disiapkan dua bentuk representasi fitur pada 36 variabel minat:

```
                            [36 Fitur Minat Mentah]
                         (Median=0, Sparsity=87.8%)
                                     |
             +-----------------------+-----------------------+
             |                                               |
     [Representasi A]                                [Representasi B]
       Count-Based                                     Binary-Based
             |                                               |
      Transformasi log1p                               Konversi Biner
        y = ln(1 + x)                                y = 1 jika x > 0
(Reduksi skewness hingga 69.5%)                  (Mengisolasi preferensi)
             |                                               |
     StandardScaler (Z-score)                                V
      z = (y - mu) / sigma                         X_active_binary.csv
             |                                     (Siap untuk Jaccard /
             V                                      Cosine / Binary K-Means)
  X_active_log1p_scaled.csv
   (Siap untuk Euclidean /
     K-Means / Ward's HC)
```

### 5.1. Representasi A: Count-Based (`log1p` + `StandardScaler`)
- **Tujuan:** Mempertahankan informasi intensitas minat (seberapa sering kata kunci muncul) sambil menyeimbangkan skala fitur dan menekan pengaruh nilai ekstrem.
- **Transformasi Logaritmik ($y = \ln(1 + x)$):**
  - Mengatasi right-skewness ekstrem tanpa memotong data (*clipping/winsorizing*).
  - Rata-rata koefisien skewness turun dari **12,38** menjadi **3,77** (**penurunan sebesar 69,5%**).
  - Kasus ekstrem seperti fitur `blonde` (nilai maksimum 327) berhasil ditekan skewness-nya dari **117,24** menjadi **8,24**.
- **Standardisasi Z-Score ($z = \frac{y - \mu}{\sigma}$):**
  - Seluruh 36 fitur memiliki $\text{mean} = 0,000$ dan $\text{std} = 1,000$.
  - Mencegah fitur populer (seperti `music` dan `shopping`) mendominasi perhitungan jarak Euclidean dibandingkan fitur spesifik (seperti `marching` atau `bible`).

### 5.2. Representasi B: Binary-Based (Presence/Absence)
- **Tujuan:** Menangkap sinyal keberadaan minat ($1$ jika count $> 0$, $0$ jika count $= 0$) tanpa memedulikan seberapa sering kata kunci diulang.
- **Rasionalitas:** Dalam konteks media sosial, siswa yang menyebut kata "soccer" 1 kali dan siswa yang menyebutnya 5 kali sama-sama menunjukkan preferensi terhadap sepak bola; perbedaan frekuensi sering kali hanya mencerminkan kebiasaan mengetik atau keaktifan akun.
- **Metrik Relevan:** Sangat ideal untuk dievaluasi menggunakan metrik jarak Jaccard, Cosine Similarity, atau K-Means pada ruang biner.

---

## 6. Daftar Artefak yang Dihasilkan

Seluruh artefak telah disimpan dan siap digunakan:

### 6.1. Dataset Hasil Preprocessing (`outputs/preprocessing/`)

| File | Baris x Kolom | Ukuran File | Keterangan & Penggunaan |
|---|:---:|:---:|---|
| [`student_profiles_clean.csv`](file:///c:/Users/boyze/OneDrive/Desktop/ML/Student-Social-Profile-Segmentation-Using-Clustering/outputs/preprocessing/student_profiles_clean.csv) | $15.000 \times 8$ | 575 KB | Metadata profil lengkap 15.000 siswa (`gender_clean`, `age_clean`, `age_imputed`, `is_all_zero`, dll.) |
| [`profiles_active.csv`](file:///c:/Users/boyze/OneDrive/Desktop/ML/Student-Social-Profile-Segmentation-Using-Clustering/outputs/preprocessing/profiles_active.csv) | $13.762 \times 8$ | 530 KB | Profil siswa aktif yang memiliki minimal 1 minat tercatat |
| [`profiles_all_zero.csv`](file:///c:/Users/boyze/OneDrive/Desktop/ML/Student-Social-Profile-Segmentation-Using-Clustering/outputs/preprocessing/profiles_all_zero.csv) | $1.238 \times 8$ | 46 KB | Profil siswa tanpa minat (Segment 0) |
| [`X_active_log1p_scaled.csv`](file:///c:/Users/boyze/OneDrive/Desktop/ML/Student-Social-Profile-Segmentation-Using-Clustering/outputs/preprocessing/X_active_log1p_scaled.csv) | $13.762 \times 36$ | 9,60 MB | **Input Utama:** 36 fitur minat aktif (Representasi A: Log1p + Scaled) |
| [`X_active_binary.csv`](file:///c:/Users/boyze/OneDrive/Desktop/ML/Student-Social-Profile-Segmentation-Using-Clustering/outputs/preprocessing/X_active_binary.csv) | $13.762 \times 36$ | 1,03 MB | **Input Alternatif:** 36 fitur minat aktif (Representasi B: Binary) |
| [`X_full_log1p_scaled.csv`](file:///c:/Users/boyze/OneDrive/Desktop/ML/Student-Social-Profile-Segmentation-Using-Clustering/outputs/preprocessing/X_full_log1p_scaled.csv) | $15.000 \times 36$ | 10,49 MB | Matriks fitur Representasi A untuk seluruh 15.000 siswa |
| [`X_full_binary.csv`](file:///c:/Users/boyze/OneDrive/Desktop/ML/Student-Social-Profile-Segmentation-Using-Clustering/outputs/preprocessing/X_full_binary.csv) | $15.000 \times 36$ | 1,12 MB | Matriks fitur Representasi B untuk seluruh 15.000 siswa |
| [`X_interest_raw.csv`](file:///c:/Users/boyze/OneDrive/Desktop/ML/Student-Social-Profile-Segmentation-Using-Clustering/outputs/preprocessing/X_interest_raw.csv) | $15.000 \times 36$ | 1,12 MB | Matriks 36 fitur minat mentah |
| [`scaler_active.joblib`](file:///c:/Users/boyze/OneDrive/Desktop/ML/Student-Social-Profile-Segmentation-Using-Clustering/outputs/preprocessing/scaler_active.joblib) | - | 2,2 KB | Objek transformer `StandardScaler` terlatih untuk data aktif |
| [`scaler_full.joblib`](file:///c:/Users/boyze/OneDrive/Desktop/ML/Student-Social-Profile-Segmentation-Using-Clustering/outputs/preprocessing/scaler_full.joblib) | - | 2,2 KB | Objek transformer `StandardScaler` terlatih untuk data lengkap |

### 6.2. Visualisasi Preprocessing (`outputs/figures/`)

- [`05_skewness_comparison.png`](file:///c:/Users/boyze/OneDrive/Desktop/ML/Student-Social-Profile-Segmentation-Using-Clustering/outputs/figures/05_skewness_comparison.png): Grafik batang perbandingan koefisien skewness sebelum vs sesudah transformasi `log1p` pada seluruh 36 fitur.
- [`06_feature_scaling_distribution.png`](file:///c:/Users/boyze/OneDrive/Desktop/ML/Student-Social-Profile-Segmentation-Using-Clustering/outputs/figures/06_feature_scaling_distribution.png): Boxplot distribusi nilai Z-score 36 fitur setelah standardisasi.
- [`07_binary_prevalence_ranking.png`](file:///c:/Users/boyze/OneDrive/Desktop/ML/Student-Social-Profile-Segmentation-Using-Clustering/outputs/figures/07_binary_prevalence_ranking.png): Peringkat prevalensi kemunculan minat siswa aktif dari tertinggi (`music`: 48,4%) hingga terendah (`bible`: 1,7%).
- [`08_all_zero_profile_comparison.png`](file:///c:/Users/boyze/OneDrive/Desktop/ML/Student-Social-Profile-Segmentation-Using-Clustering/outputs/figures/08_all_zero_profile_comparison.png): Perbandingan karakteristik jumlah teman, proporsi gender, dan tahun kelulusan antara profil all-zero vs profil aktif.

### 6.3. Tabel Ringkasan Statistik (`outputs/tables/`)

- [`preprocessing_summary.csv`](file:///c:/Users/boyze/OneDrive/Desktop/ML/Student-Social-Profile-Segmentation-Using-Clustering/outputs/tables/preprocessing_summary.csv): Tabel komprehensif metrik perbandingan nilai rata-rata mentah, nilai maksimum, skewness raw vs log1p, parameter scaling, dan prevalensi biner per fitur.

---

## 7. Panduan untuk Tahap Pemodelan (`03_model_development.ipynb`)

Pada tahap pemodelan clustering berikutnya, disarankan mengikuti panduan berikut:

1. **Pemuatan Data Input Clustering:**
   ```python
   import pandas as pd
   
   # Memuat fitur minat aktif (Representasi A: baseline)
   X_active_scaled = pd.read_csv("../outputs/preprocessing/X_active_log1p_scaled.csv", index_col=0)
   
   # Memuat fitur minat aktif (Representasi B: alternatif)
   X_active_binary = pd.read_csv("../outputs/preprocessing/X_active_binary.csv", index_col=0)
   
   # Memuat metadata profil siswa untuk evaluasi profil segmen
   df_profiles = pd.read_csv("../outputs/preprocessing/student_profiles_clean.csv", index_col=0)
   ```

2. **Rencana Eksperimen:**
   - Evaluasi jumlah klaster optimal ($k = 2$ hingga $k = 10$) menggunakan metode *Elbow Method* (Inertia), *Silhouette Score*, *Davies-Bouldin Index*, dan *Calinski-Harabasz Index*.
   - Uji algoritma: K-Means (sebagai baseline), dilanjutkan dengan Agglomerative Hierarchical Clustering atau Gaussian Mixture Models (GMM) jika diperlukan.
   - Komparasikan performa metrik antara **Representasi A (Count Log1p Scaled)** vs **Representasi B (Binary)**.

3. **Penggabungan Kembali Segmen Akhir:**
   - Setelah label klaster terbentuk pada 13.762 siswa aktif (misal Klaster 1, 2, 3, 4, 5), petakan 1.238 siswa all-zero sebagai **Klaster 0 ("No Recorded Interest")**.
   - Analisis interpretasi bisnis profil tiap klaster menggunakan atribut `gender_clean`, `age_imputed`, dan `NumberOffriends`.

---
*Laporan ini disusun secara otomatis berdasarkan eksekusi notebook `data_preparation.ipynb`.*

