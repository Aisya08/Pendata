# **Data Preparation**

Tahap **Data Preparation** merupakan bagian penting dalam metodologi CRISP-DM yang dilakukan setelah Data Understanding. Pada tahap ini, data yang telah dianalisis karakteristiknya dipersiapkan agar siap digunakan pada tahap pemodelan (modeling).

Tujuan utama tahap ini adalah memastikan data dalam kondisi bersih, konsisten, dan berkualitas sehingga model yang dibangun dapat menghasilkan performa yang optimal.

---

## **1. Pemilihan Data (Data Selection)**

Dataset yang digunakan adalah **Iris Dataset**, yang terdiri dari:

* 150 data
* 4 atribut numerik:

  * `sepal_length`
  * `sepal_width`
  * `petal_length`
  * `petal_width`
* 1 atribut kategorikal:

  * `species`

Seluruh atribut digunakan dalam proses analisis dan pemodelan karena semuanya relevan untuk klasifikasi spesies bunga iris.

---

## **2. Pembersihan Data (Data Cleaning)**

### a. Pengecekan Missing Value

Hasil pengecekan menunjukkan:

| Atribut      | Missing Value |
| ------------ | ------------- |
| sepal_length | 0             |
| sepal_width  | 0             |
| petal_length | 0             |
| petal_width  | 0             |
| species      | 0             |

Kesimpulan:
Tidak terdapat missing value, sehingga tidak diperlukan proses imputasi atau penghapusan data.

---

### b. Pengecekan dan Penghapusan Data Duplikat

Berdasarkan tahap Data Understanding, ditemukan **3 data duplikat**.

Untuk menampilkan data duplikat:

```python
duplikat = df[df.duplicated()]
print(duplikat)
```

Data duplikat terdapat pada indeks:

* 34
* 37
* 142

Untuk menghapus data duplikat:

```python
df = df.drop_duplicates()
```

Penghapusan data duplikat penting dilakukan karena:

* Mencegah bias dalam proses pemodelan
* Menghindari perhitungan data yang sama lebih dari satu kali
* Meningkatkan kualitas dataset

Setelah penghapusan, jumlah data berkurang menjadi **147 data** dan tidak terdapat lagi duplikasi.

---

### c. Kesimpulan Tahap Pembersihan Data

Berdasarkan proses yang telah dilakukan:

* Tidak terdapat missing value
* Data duplikat telah dihapus
* Dataset dalam kondisi bersih

Dataset siap digunakan untuk tahap transformasi dan pemodelan.

---

## **3. Integrasi Data (Data Integration)**

Tahap Data Integration bertujuan untuk menggabungkan data dari berbagai sumber agar menjadi dataset yang konsisten.

Pada penelitian ini:

* Dataset diperoleh dari satu sumber (Kaggle)
* Seluruh atribut sudah tersedia dalam satu file
* Tidak ada data tambahan dari sumber lain

Kesimpulan:
Tidak diperlukan proses integrasi data karena dataset sudah terstruktur dan lengkap dalam satu file.

---

# **Kesimpulan Tahap Data Preparation**

Pada tahap Data Preparation telah dilakukan:

1. Pemilihan seluruh atribut yang relevan
2. Pengecekan missing value (tidak ditemukan)
3. Penghapusan 3 data duplikat
4. Verifikasi kesiapan dataset

Dataset akhir berjumlah **147 data** dan siap digunakan untuk tahap **Modeling**.

