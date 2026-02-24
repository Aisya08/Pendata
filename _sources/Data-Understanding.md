# **Data Understanding**

## **1. Sumber Data**

Dataset yang digunakan adalah **Iris Flower Dataset** yang diperoleh dari platform Kaggle. Dataset ini merupakan dataset klasik dalam bidang data mining dan machine learning yang sering digunakan sebagai studi kasus klasifikasi.

Dataset dapat diakses melalui:
[https://www.kaggle.com/datasets/arshid/iris-flower-dataset](https://www.kaggle.com/datasets/arshid/iris-flower-dataset)

Dataset ini berisi data pengukuran bunga iris dari tiga spesies:

* *Iris-setosa*
* *Iris-versicolor*
* *Iris-virginica*

---

## **2. Deskripsi Dataset**

Dataset terdiri dari **150 data** dengan **5 atribut**, yaitu:

1. **Sepal Length (cm)** – panjang kelopak luar (numerik)
2. **Sepal Width (cm)** – lebar kelopak luar (numerik)
3. **Petal Length (cm)** – panjang mahkota bunga (numerik)
4. **Petal Width (cm)** – lebar mahkota bunga (numerik)
5. **Species** – jenis bunga (kategorikal)

Dataset ini digunakan untuk mengklasifikasikan spesies bunga iris berdasarkan ukuran morfologinya.

---

## **3. Eksplorasi Dataset Menggunakan Python**

### a. Struktur Dataset

Dataset dibaca menggunakan library **Pandas**:

```python
import pandas as pd
df = pd.read_csv("IRIS.csv")
df.head()
```

Hasil menunjukkan dataset memiliki 150 baris dan 5 kolom.

---

### b. Statistik Deskriptif

```python
df.describe()
```

Hasil statistik deskriptif menunjukkan:

* Seluruh atribut numerik memiliki jumlah data **150**, sehingga tidak terdapat missing value.
* Rata-rata:

  * Sepal length: 5.84 cm
  * Sepal width: 3.05 cm
  * Petal length: 3.76 cm
  * Petal width: 1.20 cm
* Standar deviasi terbesar terdapat pada **petal length (1.76)**, menunjukkan variasi yang lebih tinggi dibanding atribut lain.
* Rentang data masih dalam batas wajar tanpa penyimpangan ekstrem.

Secara umum, dataset memiliki distribusi yang baik dan layak digunakan untuk proses pemodelan.

---

### c. Pengecekan Data Duplikat

```python
df.duplicated().sum()
```

Hasil menunjukkan terdapat **3 data duplikat**. Data ini perlu ditangani pada tahap Data Preparation agar tidak memengaruhi model.

---

### d. Pengecekan Data Null

```python
df.isnull().sum()
```

Hasil menunjukkan seluruh atribut memiliki nilai **0 null**, sehingga tidak terdapat missing value.

---

## **4. Verifikasi Data**

Berdasarkan proses verifikasi:

* Dataset terdiri dari **150 data**
* Memiliki **4 atribut numerik** dan **1 atribut kategorikal**
* Tidak terdapat missing value
* Ditemukan 3 data duplikat
* Tipe data sudah sesuai (numerik dan kategorikal)

Dataset dinyatakan layak untuk tahap pemodelan setelah penanganan duplikat.

---

## **5. Visualisasi Data**

### a. Distribusi Jumlah Data per Species

Grafik menunjukkan setiap spesies memiliki **50 data**, sehingga dataset bersifat **balanced**. Kondisi ini sangat baik untuk proses klasifikasi karena mencegah bias terhadap kelas tertentu.

---

### b. Distribusi Fitur Numerik (Histogram)

Histogram menunjukkan bahwa:

* Seluruh fitur memiliki distribusi yang bervariasi.
* **Petal length** dan **petal width** menunjukkan pola yang lebih jelas dalam membedakan spesies.
* Tidak ditemukan anomali ekstrem.

Fitur petal berpotensi menjadi fitur utama dalam proses klasifikasi.

---

### c. Deteksi Outlier (Boxplot)

Boxplot menunjukkan:

* Terdapat beberapa outlier pada **sepal width**.
* Fitur lain relatif stabil.
* Variasi terbesar terlihat pada fitur petal.

---

### d. Analisis Korelasi

#### Korelasi Sepal Length dan Sepal Width

Scatter plot menunjukkan hubungan yang relatif lemah karena titik data tersebar tanpa pola yang jelas.

#### Korelasi Petal Length dan Petal Width

Scatter plot menunjukkan pola linear positif yang jelas, menandakan korelasi yang kuat antara kedua atribut. Hal ini menunjukkan bahwa fitur petal sangat relevan untuk klasifikasi.

---

# **Eksplorasi Dataset Menggunakan Orange**

Selain Python, analisis juga dilakukan menggunakan aplikasi **Orange Data Mining**.

### a. Statistik Deskriptif

Melalui widget *Column Statistics*, diperoleh nilai minimum, maksimum, rata-rata, dan standar deviasi yang konsisten dengan hasil analisis Python.

### b. Analisis Korelasi

Melalui widget *Scatter Plot*, terlihat bahwa:

* Atribut petal memiliki korelasi kuat.
* Atribut sepal memiliki korelasi lebih lemah.

Hasil ini memperkuat temuan bahwa fitur petal lebih signifikan dalam membedakan spesies.

---

# **Kesimpulan Tahap Data Understanding**

1. Dataset bersih (tanpa missing value).
2. Dataset seimbang (50 data per kelas).
3. Terdapat 3 data duplikat.
4. Fitur **petal length** dan **petal width** memiliki korelasi kuat dan berpotensi menjadi fitur utama dalam klasifikasi.

