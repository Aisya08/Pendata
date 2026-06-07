---
title: Laporan Analisis Kesuburan Tanah Menggunakan K-Nearest Neighbors (KNN)

---

# Laporan Analisis Kesuburan Tanah Menggunakan K-Nearest Neighbors (KNN)

## 📋 Deskripsi Proyek
Proyek ini berfokus pada implementasi machine learning untuk mengklasifikasikan kondisi kesuburan tanah berdasarkan parameter agronomis. Dengan menggunakan platform **KNIME Analytics Platform**, dataset sebanyak **2.000 sampel** dianalisis untuk menentukan apakah tanah masuk ke dalam kategori **Subur** atau **Tidak Subur**.

---

## 1. Tahapan Pemrosesan Data (Preprocessing)
Dalam analisis ini, data mentah melalui beberapa tahap agar siap diproses oleh algoritma KNN:
* **Penanganan Missing Values**: Menggunakan node `Missing Value` untuk memastikan tidak ada data hara yang kosong yang dapat mengganggu perhitungan jarak.
* **Normalisasi Data**: Menggunakan node `Normalizer` dengan metode **Min-Max (0 ke 1)**. Hal ini dilakukan karena fitur seperti P Tersedia (ppm) dan N Total (%) memiliki skala yang berbeda jauh.
* **Pembagian Data (Partitioning)**: Data dibagi menjadi **80% Data Latih** dan **20% Data Uji** untuk memvalidasi performa model.


![image](https://hackmd.io/_uploads/BJZvGpraWx.png)


## 🛠️ Penjelasan Komponen Workflow (Node Details)

Dalam perancangan model ini, setiap **node** pada KNIME memiliki fungsi spesifik untuk memastikan data diolah dengan standar *Data Science* yang benar:

### 1. Excel Reader (Input Data)
* **Fungsi:** Titik masuk utama dataset.
* **Peran:** Node ini membaca file spreadsheet yang berisi **2.000 sampel data tanah**. Data ini mencakup parameter penting seperti pH, kandungan Nitrogen (N), Fosfor (P), Kalium (K), hingga tekstur tanah.

### 2. Missing Value (Data Cleaning)
* **Fungsi:** Menangani data yang tidak lengkap atau kosong ($NaN$).
* **Peran:** Algoritma KNN berbasis jarak sangat sensitif terhadap data kosong. Node ini melakukan **Mean Imputation** (mengisi nilai kosong dengan rata-rata) agar dataset tetap utuh tanpa harus menghapus baris data.

### 3. Normalizer (Feature Scaling)
* **Fungsi:** Menyamakan rentang nilai fitur menggunakan **Min-Max Scaling**.
* **Peran:** Karena parameter tanah memiliki satuan berbeda (misal: pH 0-14, sedangkan P-Tersedia bisa mencapai 60), normalisasi mengubah semuanya ke rentang **0.0 - 1.0**. Tanpa tahap ini, fitur dengan angka besar akan mendominasi hasil prediksi secara tidak akurat.

### 4. Table Partitioner (Data Validation)
* **Fungsi:** Memisahkan data menjadi dua set (**Split Data 80:20**).
* **Peran:** * **80% (Training Set):** Digunakan model untuk mempelajari pola tanah subur.
    * **20% (Testing Set):** Digunakan sebagai data "asing" untuk menguji sejauh mana model mampu melakukan prediksi dengan benar.

### 5. K-Nearest Neighbor (Machine Learning Engine)
* **Fungsi:** Melakukan klasifikasi berdasarkan tingkat kemiripan tetangga terdekat.
* **Peran:** Menggunakan perhitungan **Jarak Euclidean** dengan nilai **$k=5$**. Jika mayoritas dari 5 tetangga terdekat suatu data berlabel "Subur", maka data tersebut akan diklasifikasikan sebagai "Subur".

### 6. Scorer (Evaluasi Akhir)
* **Fungsi:** Mengukur tingkat keberhasilan model.
* **Peran:** Node ini membandingkan jawaban asli (Ground Truth) dengan jawaban prediksi model. Dari sini dihasilkan metrik krusial seperti:
    - [x] **Confusion Matrix**
    - [x] **Accuracy**
    - [x] **Precision & Recall**
    - [x] **F1-Score**


## 2. Implementasi Algoritma KNN
Analisis dilakukan dengan menggunakan algoritma **K-Nearest Neighbors** dengan parameter sebagai berikut:
* **Nilai K**: 5 (Lima tetangga terdekat).
* **Fitur Utama**: pH Tanah, N Total, P Tersedia, K Tersedia, C Organik, KTK, Kejenuhan Basa, Kadar Air, dan Bulk Density.
* **Target**: Klasifikasi (Subur / Tidak Subur).

## 3. Hasil dan Metrik Evaluasi
Berdasarkan pengujian pada data uji, diperoleh hasil melalui node `Scorer` sebagai berikut:

| Metrik | Nilai | Keterangan |
| :--- | :--- | :--- |
| **Accuracy** | 1.0 (100%) | Seluruh data uji terklasifikasi dengan benar. |
| **Precision** | 1.0 | Ketepatan prediksi untuk kelas positif sangat sempurna. |
| **Recall** | 1.0 | Model mampu mendeteksi seluruh sampel tanah subur tanpa ada yang terlewat. |
| **F1-Score** | 1.0 | Keseimbangan antara Precision dan Recall sangat ideal. |

### Confusion Matrix
Hasil visualisasi menunjukkan:
* **True Positive (Subur)**: 163 sampel terdeteksi benar.
* **True Negative (Tidak Subur)**: 179 sampel terdeteksi benar.
* **Errors**: 0 (Tidak ada salah prediksi).
* ![image](https://hackmd.io/_uploads/HkhJZTSaZe.png)


## 4. Kesimpulan
Model KNN terbukti sangat efektif untuk melakukan klasifikasi kesuburan tanah pada dataset ini dengan tingkat akurasi mencapai 100%. Hal ini menunjukkan bahwa fitur-fitur agronomis yang digunakan memiliki batas pemisah (decision boundary) yang sangat jelas antar kelas.






#