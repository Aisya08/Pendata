---
title: Analisa Data Menggunakan Naive Bayes (A)

---

# Analisa Data Menggunakan Naive Bayes (A)

## Latar Belakang

Dalam data mining, klasifikasi merupakan salah satu metode yang digunakan untuk mengelompokkan data ke dalam kelas tertentu berdasarkan pola dari data sebelumnya.

Pada tugas ini digunakan metode **Naive Bayes** untuk melakukan klasifikasi data. Metode ini dipilih karena sederhana, cepat, dan cukup efektif digunakan pada data numerik.


## Dataset

Dataset yang digunakan adalah **IRIS dataset**.

Dataset ini merupakan dataset klasifikasi yang sangat umum digunakan dalam pembelajaran machine learning.

Jumlah data: **150 baris**

Jumlah atribut: **5 kolom**

### Atribut Dataset

| No | Atribut | Keterangan |
|:--:|---------|------------|
| 1 | SepalLengthCm | Panjang sepal |
| 2 | SepalWidthCm | Lebar sepal |
| 3 | PetalLengthCm | Panjang petal |
| 4 | PetalWidthCm | Lebar petal |
| 5 | Species | Jenis bunga iris |

---

## Tujuan Analisis

Analisis ini bertujuan untuk mengklasifikasikan jenis bunga iris berdasarkan ukuran sepal dan petal.

Kelas yang diprediksi adalah:

- **:contentReference[oaicite:1]{index=1}**
- **:contentReference[oaicite:2]{index=2}**
- **:contentReference[oaicite:3]{index=3}**

---

## Teori Naive Bayes

Naive Bayes merupakan metode klasifikasi probabilistik yang didasarkan pada **:contentReference[oaicite:4]{index=4}**.

Metode ini menghitung kemungkinan suatu data masuk ke dalam kelas tertentu berdasarkan probabilitas.

:contentReference[oaicite:5]{index=5}

### Keterangan

- **P(H|X)** = probabilitas posterior
- **P(X|H)** = likelihood
- **P(H)** = prior
- **P(X)** = evidence

Metode ini disebut **naive** karena mengasumsikan setiap atribut saling independen.

---

## Langkah Analisis

Tahapan analisis yang dilakukan adalah:

1. Membaca dataset **IRIS.csv**
2. Menentukan fitur input dan label kelas
3. Membagi data menjadi data training dan testing
4. Melatih model Naive Bayes
5. Melakukan prediksi data testing
6. Mengevaluasi hasil klasifikasi

---

## Implementasi Python

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

df = pd.read_csv("IRIS.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Akurasi:", accuracy_score(y_test, y_pred))