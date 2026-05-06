---
title: Laporan Perhitungan Decision Tree Menggunakan Gain Ratio

---

# Laporan Perhitungan Decision Tree Menggunakan Gain Ratio

## 1. Tujuan

Menentukan atribut terbaik sebagai akar dan cabang pohon keputusan menggunakan **perhitungan Gain, SplitInfo, dan Gain Ratio** pada dataset *Play Tennis*.

![Cuplikan layar 2026-05-06 111407](https://hackmd.io/_uploads/HklLxH_Cbl.png)



### Penjelasan Alur Workflow

Workflow di atas menunjukkan proses pembuatan model klasifikasi menggunakan metode **Decision Tree** dengan pendekatan **Gain Ratio**.

### 1. Excel Reader

Node ini digunakan untuk membaca dataset dari file Excel yang berisi data *Play Tennis*. Data ini akan menjadi input utama dalam proses analisis.

---

### 2. Table Partition

Node ini berfungsi untuk membagi dataset menjadi dua bagian:

* **Training data** → digunakan untuk membangun model
* **Testing data** → digunakan untuk menguji model

Biasanya pembagian dilakukan dengan rasio 70:30.

---

### 3. Color Manager

Node ini digunakan untuk memberikan warna pada kelas (label) seperti *Yes* dan *No*.
Tujuannya agar visualisasi pada pohon keputusan lebih mudah dipahami.

---

### 4. Decision Tree Learner

Node ini merupakan inti dari workflow, digunakan untuk:

* Membangun model pohon keputusan
* Menggunakan metode **Gain Ratio** untuk menentukan atribut terbaik

Output dari node ini adalah model pohon keputusan.

---

### 5. Color Appender

Node ini berfungsi untuk menambahkan kembali informasi warna ke data testing, sehingga hasil prediksi dapat divisualisasikan dengan warna yang sesuai.

---

### 6. Decision Tree Predictor

Node ini digunakan untuk:

* Menerapkan model yang telah dibuat
* Melakukan prediksi terhadap data testing

Hasilnya berupa kolom baru yaitu **Prediction (PlayTennis)**.

---

### 7. Scorer

Node ini digunakan untuk mengevaluasi hasil prediksi dengan cara:

* Membandingkan hasil prediksi dengan data asli
* Menghasilkan **confusion matrix**
* Menghitung **akurasi model**

![Cuplikan layar 2026-05-06 111844](https://hackmd.io/_uploads/HyjWfrORbe.png)


## Analisis Visualisasi Decision Tree (Interactive View)

Berdasarkan hasil eksekusi pada node **Decision Tree Learner**, berikut adalah penjelasan mengenai struktur pohon yang terbentuk:

### 1. Root Node (Simpul Akar)
*   **Total Data**: Terdapat 10 data yang diolah (n=10).
*   **Distribusi Target**: 
    *   **Yes**: 60% (6 data)
    *   **No**: 30% (3 data)
    *   **Lainnya**: 10% (1 data)
*   Simpul akar ini merepresentasikan kondisi awal seluruh dataset sebelum dilakukan pemisahan (splitting) berdasarkan atribut dengan Gain Ratio tertinggi.

### 2. Proses Splitting (Pemisahan)
Gambar di atas menunjukkan bagaimana pohon keputusan melakukan klasifikasi spesifik hingga mencapai *leaf node* (daun):
*   **Atribut Pemisah**: Pohon memecah data berdasarkan identitas record (seperti `= D3`, `= D6`, `= D7`, dst.). 
*   **Karakteristik Leaf Node**: Setiap cabang menghasilkan kesimpulan tunggal dengan tingkat keyakinan (confidence) **100%**. 
    *   Contoh: Pada baris data **D6**, **D8**, dan **D14**, hasil klasifikasi akhirnya adalah **No** (Indikator warna merah/hijau penuh di chart).
    *   Contoh: Pada baris data **D3**, **D7**, **D9**, dan **D11**, hasil klasifikasi akhirnya adalah **Yes**.

### 3. Interpretasi Gain Ratio pada Visualisasi
Pohon ini menggunakan **Gain Index** untuk memastikan bahwa setiap percabangan memiliki efektivitas maksimal dalam mengurangi entropi. 
*   Visualisasi batang (bar chart) di setiap kotak menunjukkan seberapa dominan kelas target pada cabang tersebut. 
*   Warna hijau yang memenuhi bar chart pada *leaf node* menandakan bahwa atribut tersebut berhasil memisahkan data secara murni (*pure*) tanpa tercampur kelas lain.

---
*Catatan: Pastikan urutan kolom pada Excel sudah sesuai agar pohon keputusan bisa memilih atribut 'Outlook' sebagai pemisah utama di level pertama jika ingin melihat struktur yang lebih general.*



## 2. Dataset

Atribut yang digunakan:

* Outlook
* Humidity
* Wind
* PlayTennis (target)

---

## 3. Konsep Perhitungan

### Entropy

Entropy(S)=-\sum p_i\log_2 p_i

---

### Information Gain

Gain(S,A)=Entropy(S)-\sum_v \frac{|S_v|}{|S|}Entropy(S_v)

---

### Split Info

SplitInfo(A)=-\sum_v \frac{|S_v|}{|S|}\log_2\left(\frac{|S_v|}{|S|}\right)

---

### Gain Ratio

GainRatio(A)=\frac{Gain(S,A)}{SplitInfo(A)}

---

## 4. Langkah Perhitungan

### 4.1 Hitung Entropy Awal

Misal:

* Yes = 9
* No = 5

```text
Entropy(S) = -(9/14 log2 9/14) - (5/14 log2 5/14)
           ≈ 0.940
```

---

### 4.2 Hitung Gain tiap atribut

#### Atribut Outlook

* Memiliki nilai: Sunny, Overcast, Rain
* Hitung entropy masing-masing subset
* Hitung Gain

Hasil:

```text
Gain(Outlook) ≈ 0.246
```

---

### 4.3 Hitung Split Info Outlook

```text
SplitInfo(Outlook) ≈ 1.577
```

---

### 4.4 Hitung Gain Ratio Outlook

```text
GainRatio(Outlook) = 0.246 / 1.577 ≈ 0.156
```

---

## 5. Penentuan Akar Pohon

Dari perhitungan semua atribut:

* Outlook memiliki Gain Ratio tertinggi
  ➡️ Maka dipilih sebagai **akar pohon**

---

## 6. Pembentukan Pohon Keputusan

Hasil akhir pohon:

```text
Outlook
├── Sunny → Humidity
│   ├── High → No
│   └── Normal → Yes
├── Overcast → Yes
└── Rain → Wind
    ├── False → Yes
    └── True → No
```

---

## 7. Interpretasi

* Jika **Outlook = Overcast** → langsung **Yes**
* Jika **Sunny** → lihat **Humidity**
* Jika **Rain** → lihat **Wind**

---

![image](https://hackmd.io/_uploads/HJoOfHdC-x.png)
## Evaluasi Model: Confusion Matrix & Scorer

Setelah model Decision Tree dilatih dan dijalankan pada dataset, kita menggunakan node **Scorer** untuk mengevaluasi performa prediksi melalui **Confusion Matrix**.

### 1. Analisis Confusion Matrix
Berdasarkan tampilan hasil Scorer pada gambar di atas, dapat dilihat beberapa poin penting:
*   **Struktur Matrix**: Matrix membandingkan kolom **PlayTennis** (Data Aktual/Referensi) dengan kolom **Prediction (PlayTennis)** (Hasil Prediksi Model).
*   **Nilai 0 (Zero Values)**: Saat ini, matrix menunjukkan angka 0 pada semua sel (No/Yes). Hal ini mengindikasikan bahwa data belum terproses secara sempurna atau terdapat ketidakcocokan pada konfigurasi kolom.
*   **Peringatan (Warning)**: Muncul notifikasi *"There were missing values in the reference or in the prediction class columns"*. Ini menunjukkan adanya data kosong (*Missing Values*) atau kolom yang dipilih tidak berisi data yang valid untuk dibandingkan.

### 2. Troubleshooting & Solusi
Untuk mendapatkan hasil akurasi yang benar pada Confusion Matrix, langkah-langkah berikut perlu diperiksa kembali pada workflow:

1.  **Konfigurasi Node Scorer**:
    *   Pastikan **First Column** diarahkan ke kolom target asli (`PlayTennis`).
    *   Pastikan **Second Column** diarahkan ke kolom hasil prediksi model (biasanya bernama `Prediction (PlayTennis)`).
2.  **Pemeriksaan Data Kosong**: 
    *   Gunakan node **Missing Value** jika terdapat data yang tidak terisi pada file Excel asal.
    *   Pastikan baris header pada *Excel Reader* sudah tersetting dengan benar sehingga tidak ada sel "null" yang terbaca sebagai data.
3.  **Eksekusi Predictor**: 
    *   Pastikan node **Decision Tree Predictor** sudah terhubung ke data uji dan sudah dieksekusi (berlampu hijau) sebelum membuka hasil pada node Scorer.

### 3. Tujuan Akhir
Hasil akhir yang diharapkan adalah munculnya angka pada diagonal matrix, yang menunjukkan:
*   **True Positive (TP)**: Jumlah 'Yes' yang diprediksi benar sebagai 'Yes'.
*   **True Negative (TN)**: Jumlah 'No' yang diprediksi benar sebagai 'No'.
*   **Accuracy**: Persentase total prediksi benar dibandingkan dengan total seluruh data.

## 9. Kesimpulan

1. Gain Ratio digunakan untuk memilih atribut terbaik.
2. Outlook menjadi akar karena nilai Gain Ratio tertinggi.
3. Pohon keputusan dapat digunakan untuk klasifikasi PlayTennis.
