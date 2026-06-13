
**Nama : Aisya**
**Nim : 240411100025**
**Mata Kuliah : Penambangan Data IF4A**



# Analisa Data (A)

# 1. Analisa prediksi tentang apa?

Pada dokumentasi **Skforecast Explainability**, analisis yang dilakukan adalah prediksi permintaan listrik (*electricity demand forecasting*) menggunakan data time series.

Time series adalah data yang tersusun berdasarkan waktu, misalnya per jam, per hari, atau per bulan. Pada kasus ini, model digunakan untuk memprediksi jumlah permintaan listrik di masa mendatang berdasarkan data historis sebelumnya.

Dataset yang digunakan memiliki beberapa variabel seperti:

| Variabel | Keterangan |
|---|---|
| Time | Waktu pengamatan |
| Date | Tanggal data |
| Demand | Jumlah permintaan listrik |
| Temperature | Suhu lingkungan |
| Holiday | Penanda hari libur |

Target utama prediksi adalah:
- **Demand (permintaan listrik)**

Tujuan analisis:
1. Memprediksi kebutuhan listrik di masa depan.
2. Mengetahui pola penggunaan listrik.
3. Melihat pengaruh suhu terhadap penggunaan listrik.
4. Mengetahui fitur yang paling mempengaruhi model menggunakan explainability.

Model machine learning yang digunakan adalah:
LGBMRegressor()

# 2. Bagaimana bentuk data trainingnya (apa saja input dan outputnya)?

## A. Input (X)

Input model terdiri dari:
1. Data historis permintaan listrik sebelumnya (*lag*).
2. Variabel tambahan yaitu suhu (*Temperature*).

Contoh fitur input:
- `lag_1`
- `lag_2`
- `lag_3`
- `lag_4`
- `lag_5`
- `lag_6`
- `lag_7`
- `Temperature`

Contoh bentuk data training:

| lag_1 | lag_2 | lag_3 | lag_4 | lag_5 | lag_6 | lag_7 | Temperature |
|---|---|---|---|---|---|---|---|
| 205338 | 211066 | 213792 | 258955 | 275490 | 227778 | 82531 | 24.09 |

Penjelasan:
- `lag_1` = data permintaan listrik 1 periode sebelumnya.
- `lag_2` = data 2 periode sebelumnya.
- dan seterusnya.

---

## B. Output (y)

Output atau target prediksi adalah:
- `Demand`

Contoh:

| y |
|---|
| 200693 |

Artinya model mencoba memprediksi jumlah permintaan listrik berikutnya berdasarkan data sebelumnya.

---

# 3. Apa itu lag?

Lag adalah data dari periode sebelumnya yang digunakan sebagai referensi untuk memprediksi data berikutnya.

Dalam forecasting, lag sangat penting karena data masa lalu biasanya memiliki hubungan dengan data masa depan.

Contoh sederhana:

| Hari | Demand |
|---|---|
| Senin | 100 |
| Selasa | 120 |
| Rabu | 130 |

Untuk memprediksi hari Kamis:
- `lag_1` = 130
- `lag_2` = 120
- `lag_3` = 100

Artinya model melihat pola dari data sebelumnya untuk memprediksi data berikutnya.

Pada dokumentasi ini digunakan:

```python
lags = 7
```

Artinya model menggunakan 7 periode sebelumnya untuk melakukan prediksi.

Fungsi lag:
1. Menangkap pola historis.
2. Mengetahui tren naik atau turun.
3. Membantu model memahami hubungan waktu.
4. Membuat prediksi lebih akurat.

---

# 4. Jelaskan proses analisis yang dilakukan dari kasus di atas

## A. Pengambilan dan Persiapan Data

Pertama dataset dimuat menggunakan library pandas.

Data kemudian diproses dengan:
- mengubah format tanggal,
- membersihkan data kosong,
- memilih variabel yang diperlukan,
- serta menyiapkan indeks waktu.

Contoh:

```python
data['Date'] = pd.to_datetime(data['Date'])
```

Tujuannya agar data siap digunakan dalam forecasting.

---

## B. Membagi Data Training dan Testing

Data dibagi menjadi:
1. Data training → digunakan untuk melatih model.
2. Data testing → digunakan untuk menguji model.

Tujuan pembagian data adalah agar model dapat dievaluasi menggunakan data yang belum pernah dipelajari sebelumnya.

---

## C. Membuat Fitur Lag

Skforecast otomatis membuat fitur lag menggunakan:

```python
lags = 7
```

Artinya:
- model menggunakan 7 data sebelumnya sebagai input prediksi.

Contoh:
- data hari ke-1 sampai hari ke-7 digunakan untuk memprediksi hari ke-8.

---

## D. Melatih Model Forecasting

Model yang digunakan:

```python
LGBMRegressor()
```

Kemudian dimasukkan ke:

```python
ForecasterRecursive
```

Fungsi model:
- mempelajari pola historis,
- memahami hubungan suhu dengan permintaan listrik,
- menghasilkan prediksi masa depan.


## E. Melakukan Prediksi

Setelah model selesai dilatih:
- model digunakan untuk memprediksi data testing.

Hasil prediksi dibandingkan dengan data asli untuk melihat tingkat akurasi model.


## F. Explainability Model

Bagian utama dokumentasi ini adalah explainability.

Explainability digunakan untuk memahami:
- bagaimana model bekerja,
- fitur mana yang paling penting,
- alasan model menghasilkan prediksi tertentu.

Metode explainability yang digunakan:

### 1. Feature Importance

Digunakan untuk melihat fitur paling berpengaruh terhadap hasil prediksi.

Hasilnya menunjukkan bahwa:
- `Temperature`
- `lag_1`
- `lag_3`

merupakan fitur yang paling penting.

Artinya suhu dan data historis sangat mempengaruhi permintaan listrik.


### 2. SHAP Values

SHAP digunakan untuk menjelaskan kontribusi setiap fitur terhadap hasil prediksi.

Kelebihan SHAP:
- dapat menjelaskan prediksi individual,
- menunjukkan apakah fitur menaikkan atau menurunkan hasil prediksi.

Contoh:
- suhu tinggi menyebabkan penggunaan AC meningkat,
- sehingga permintaan listrik ikut meningkat.



### 3. Partial Dependence Plot

Metode ini digunakan untuk melihat hubungan antara suatu fitur dengan hasil prediksi model.

Contoh:
- ketika suhu meningkat,
- maka permintaan listrik juga meningkat.

Grafik ini membantu memahami perilaku model terhadap perubahan fitur tertentu.



# Kesimpulan


Berdasarkan analisis pada dokumentasi Skforecast Explainability dapat disimpulkan bahwa:

1. Model digunakan untuk memprediksi permintaan listrik.
2. Data training menggunakan data historis dan suhu.
3. Lag adalah data periode sebelumnya yang digunakan sebagai input prediksi.
4. Model forecasting menggunakan `LGBMRegressor`.
5. Explainability digunakan untuk memahami cara kerja model.
6. Fitur paling berpengaruh adalah:
   - Temperature
   - lag_1
   - lag_3
7. Model mampu mempelajari pola historis sehingga menghasilkan prediksi yang cukup akurat.

Dengan explainability, pengguna tidak hanya mengetahui hasil prediksi, tetapi juga memahami alasan model menghasilkan prediksi tersebut.
