# Data Preparation

Tahap Data Preparation merupakan salah satu tahapan penting dalam metodologi CRISP-DM yang dilakukan setelah Data Understanding. Pada tahap ini, data yang telah dipahami karakteristiknya kemudian dipersiapkan agar siap digunakan pada tahap pemodelan (modeling).
Data yang diperoleh dari berbagai sumber biasanya masih dalam kondisi mentah dan belum siap untuk langsung dianalisis. Oleh karena itu, tahap Data Preparation dilakukan untuk memperbaiki dan menyiapkan data agar lebih rapi dan berkualitas, sehingga proses analisis dan pemodelan dapat berjalan dengan baik dan menghasilkan hasil yang lebih akurat.

### 1. Memilih Data

Pada tahap ini dilakukan pemilihan data yang akan digunakan dalam proses analisis dan pemodelan.
Dataset yang digunakan adalah dataset Iris, yang terdiri dari 150 data dengan 5 atribut, yaitu:

- sepal_length
- sepal_width
- petal_length
- petal_width
- species


### 2. Membersihkan Data

##### - Cek Missing Value

|  |  |
| :-- | :-- |
| sepal_length | 0 |
| sepal_width | 0 |
| petal_length | 0 |
| petal_width | 0 |
| species | 0 |

Berdasarkan hasil pemeriksaan pada data understanding, seluruh atribut memiliki nilai 0 pada missing value. Hal ini menunjukkan bahwa dataset sudah lengkap dan tidak memerlukan penanganan missing value.

##### - Cek Data Duplikat

Berdasarkan hasil pemeriksaan pada data understanding, terdapat 3 data duplikat pada dataset. Data tersebut kemudian akan dihapus pada tahap pembersihan data untuk memastikan kualitas dataset tetap baik.

```
duplikat = df[df.duplicated()]
print(duplikat)
```

| sepal_length | sepal_width | petal_length | petal_length | petal_width | species |
| :-- | :-- | :-- | :-- | :-- | :-- |
| 34 | 4.9 | 3.1 | 1.5 | 0.1 | Iris-setosa |
| 37 | 4.9 | 3.1 | 1.5 | 0.1 | Iris-setosa |
| 142 | 5.8 | 2.7 | 5.1 | 0.9 | Iris-virginica |

Data duplikat ditampilkan menggunakan fungsi duplicated(). Data duplikat terdapat pada baris  34, 37, dan 142.

```
df = df.drop_duplicates()
```

Data duplikat dapat menyebabkan analisis menjadi kurang akurat karena data yang sama dihitung lebih dari satu kali. Oleh karena itu, dilakukan proses pembersihan data dengan menghapus data duplikat menggunakan fungsi drop_duplicates() pada Python.

Setelah dilakukan penghapusan, jumlah data berkurang dan tidak terdapat lagi data duplikat, sehingga dataset menjadi lebih bersih dan siap digunakan untuk proses analisis dan pemodelan.

![original image](https://cdn.mathpix.com/snip/images/n9MAjgsSondmmBsZjTsThSPfzJmFd8hVKyvfmxiaNLY.original.fullsize.png)

##### - Kesimpulan

Berdasarkan proses pembersihan data yang telah dilakukan, diketahui bahwa:

- Tidak terdapat missing value
- Tidak terdapat data duplikat

Dengan demikian, dataset sudah bersih dan siap digunakan untuk tahap selanjutnya, yaitu transformasi data dan pemodelan.

### 3. Integrasi Data

Tahap data integration bertujuan untuk menggabungkan data dari berbagai sumber menjadi satu dataset yang konsisten dan terstruktur.

Pada penelitian ini, dataset Iris diperoleh dari satu sumber dan seluruh atribut yang dibutuhkan sudah tersedia dalam satu file. Atribut tersebut meliputi sepal_length, sepal_width, petal_length, petal_width, dan species.

Karena seluruh data sudah terintegrasi dalam satu dataset dan tidak terdapat sumber data lain yang perlu digabungkan, maka tidak diperlukan proses integrasi tambahan.