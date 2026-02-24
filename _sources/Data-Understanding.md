# Data Understanding

### 1. Sumber Data

Dataset yang digunakan adalah Iris Flower Dataset yang diperoleh dari platform Kaggle. Dataset ini merupakan dataset klasik dalam bidang data mining dan machine learning yang berisi data pengukuran bunga iris dari tiga spesies berbeda, yaitu Setosa, Versicolor, dan Virginica. Dataset ini digunakan sebagai studi kasus untuk penerapan metode klasifikasi dalam penambangan data.
Berikut link dari dataset 
[Dataset Iris](https://www.kaggle.com/datasets/arshid/iris-flower-dataset?resource=download)

### 2. Deskripsi Dataset

Dataset yang digunakan dalam penelitian ini adalah Iris Flower Dataset yang terdiri dari 150 data dengan 5 atribut. Dataset ini digunakan untuk mengklasifikasikan bunga iris ke dalam tiga spesies berdasarkan ukuran bagian bunganya. 
### 3. Eksplorasi Dataset

Dalam proses mengidentifikasi dataset ini, python membantu untuk mempermudah pengidentifikasiannya.
#### Eksplorasi Dataset dengan menggunakan Python:
##### - Upload file CSV
```
from google.colab import files
files.upload()
```
Proses upload file CSV dilakukan untuk memasukkan dataset ke dalam lingkungan Google Colaboratory agar dapat diproses menggunakan bahasa pemrograman Python. Dataset diunggah secara manual dari perangkat pengguna menggunakan fungsi files.upload() yang disediakan oleh Google Colab.

##### - Struktur Dataset

```
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("IRIS.csv")
df.head()
```

Berikut tampilan dari code di atas setelah di jalankan:

|index|sepal\_length|sepal\_width|petal\_length|petal\_width|species|
|---|---|---|---|---|---|
|0|5\.1|3\.5|1\.4|0\.2|Iris-setosa|
|1|4\.9|3\.0|1\.4|0\.2|Iris-setosa|
|2|4\.7|3\.2|1\.3|0\.2|Iris-setosa|
|3|4\.6|3\.1|1\.5|0\.2|Iris-setosa|
|4|5\.0|3\.6|1\.4|0\.2|Iris-setosa|

 Dataset memiliki 5 kolom, yaitu:
 1. Sepal Length (cm) (Numeric)
    Menunjukkan panjang kelopak luar bunga iris dalam satuan sentimeter.
 3. Sepal Width (cm) (Numeric)
    Menunjukkan lebar kelopak luar bunga iris dalam satuan sentimeter.
 5. Petal Length (cm) (Numeric)
    Menunjukkan panjang mahkota bunga iris dalam satuan sentimeter.
 7. Petal Width (cm) (Numeric)
    Menunjukkan lebar mahkota bunga iris dalam satuan sentimeter.
 9. Species (Kategorikal - String)
     Menunjukkan jenis bunga iris, yang terdiri dari tiga kategori:

    - Setosa
    - Versicolor
    - Virginica

##### -Statistik Deskriptif Awal

```
df.describe()
```
Statistik deskriptif digunakan untuk mengetahui gambaran umum data seperti nilai rata-rata, nilai minimum, dan maksimum dari setiap atribut numerik.

Dataset Iris memiliki 4 atribut numerik, yaitu:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

Berikut contoh statistik deskriptifnya:

|index|sepal\_length|sepal\_width|petal\_length|petal\_width|
|---|---|---|---|---|
|count|150\.0|150\.0|150\.0|150\.0|
|mean|5\.843333333333334|3\.0540000000000003|3\.758666666666666|1\.1986666666666668|
|std|0\.8280661279778629|0\.4335943113621737|1\.7644204199522617|0\.7631607417008414|
|min|4\.3|2\.0|1\.0|0\.1|
|25%|5\.1|2\.8|1\.6|0\.3|
|50%|5\.8|3\.0|4\.35|1\.3|
|75%|6\.4|3\.3|5\.1|1\.8|
|max|7\.9|4\.4|6\.9|2\.5|

Berdasarkan hasil analisis statistik deskriptif, dataset Iris terdiri dari 150 data pada setiap atribut numerik, sehingga dapat disimpulkan tidak terdapat missing value. Nilai rata-rata (mean) menunjukkan bahwa panjang sepal bunga iris adalah 5.84 cm, lebar sepal 3.05 cm, panjang petal 3.76 cm, dan lebar petal 1.20 cm. Standar deviasi tertinggi terdapat pada atribut petal length sebesar 1.76, yang menunjukkan bahwa variasi panjang petal lebih besar dibandingkan atribut lainnya. Nilai minimum dan maksimum menunjukkan rentang data yang masih dalam batas wajar, yaitu sepal length antara 4.3 hingga 7.9 cm dan petal width antara 0.1 hingga 2.5 cm. Selain itu, nilai kuartil (25%, 50%, dan 75%) menunjukkan distribusi data yang relatif stabil tanpa adanya penyimpangan ekstrem yang signifikan. Secara umum, dataset memiliki distribusi yang baik dan layak digunakan untuk proses analisis dan pemodelan pada tahap selanjutnya.

##### - Pengecekan Data Duplikat

```
df.duplicated().sum()
```
Code di atas merupakan code untuk pengecekan data duplikat, berikut untuk hasil dari pengecekan data duplikat:

```
np.int64(3)
```
Jadi setelah dilakukan pengecekan, terdapat data duplikat yaitu ada 3 data duplikat.

##### - Pengecekan Data Null

```
df.isnull().sum()
```

Berikut untuk tampilan hasil dari pengecekan data Null:

|||
|---|---|
|sepal_length|0|
|sepal_width|0|
|petal_length|0|
|petal_width|0|
|species|0|


### 4. Verifikasi Data

Berdasarkan proses verifikasi, dataset Iris terdiri dari 150 data dengan 4 atribut numerik dan 1 atribut kategorikal (species). Hasil pengecekan menunjukkan bahwa seluruh atribut memiliki jumlah data yang sama (count = 150), sehingga dapat disimpulkan tidak terdapat missing value pada dataset. Tipe data juga telah sesuai, di mana atribut sepal length, sepal width, petal length, dan petal width bertipe numerik, sedangkan species bertipe kategorikal. Berdasarkan hasil pengecekan menggunakan Pandas, ditemukan sebanyak 3 data duplikat pada dataset. Keberadaan data duplikat ini tidak menunjukkan kesalahan pada dataset, namun perlu ditangani pada tahap data preparation agar tidak memengaruhi proses pemodelan. 

### 5. Visualisasi Data

##### - Distribusi Jumlah Data per Species

![original image](https://cdn.mathpix.com/snip/images/Gt4lANFXJ1tvApce6vXv0vo4hu08VXJb7_bTkbfgn-M.original.fullsize.png)

  Grafik bar digunakan untuk melihat jumlah data pada setiap species. Berdasarkan grafik, diketahui bahwa setiap species yaitu Iris-setosa, Iris-versicolor, dan Iris-virginica masing-masing memiliki 50 data. Hal ini menunjukkan bahwa dataset dalam kondisi seimbang (balanced dataset), sehingga tidak terdapat ketimpangan jumlah data antar kelas. Kondisi ini sangat baik untuk proses modeling karena dapat membantu menghasilkan model klasifikasi yang lebih akurat dan tidak bias terhadap kelas tertentu.

##### - Distribusi Data Fitur Numerik pada Dataset Iris

![original image](https://cdn.mathpix.com/snip/images/6u595jYczu5oHxd1yDMvEGkfvzQYsyvFrhOICQnLnGY.original.fullsize.png)


Berdasarkan histogram, dapat disimpulkan bahwa seluruh fitur numerik memiliki distribusi yang bervariasi dan tidak terdapat anomali yang ekstrem. Fitur petal_length dan petal_width menunjukkan pola distribusi yang lebih jelas dalam membedakan kelompok data, sehingga kedua fitur ini sangat berpotensi menjadi fitur penting dalam proses klasifikasi pada tahap modeling. Visualisasi ini membantu dalam memahami karakteristik dan penyebaran data pada tahap Data Understanding dalam metodologi CRISP-DM.

##### - Analisis Penyebaran Data dan Deteksi Outlier Menggunakan Boxplot

![original image](https://cdn.mathpix.com/snip/images/YfGQj5-WjGnpPQF6mWpdtoRSXTPrJIw5sXZ6l769Xo0.original.fullsize.png)

Berdasarkan boxplot, dapat disimpulkan bahwa setiap fitur memiliki penyebaran data yang berbeda. Fitur sepal_width menunjukkan adanya beberapa outlier, sedangkan fitur lainnya memiliki distribusi yang relatif normal tanpa outlier yang signifikan. Fitur petal_length dan petal_width memiliki variasi data yang cukup besar, sehingga berpotensi menjadi fitur penting dalam membedakan species pada tahap modeling. Visualisasi boxplot ini membantu dalam memahami distribusi data dan mendeteksi outlier pada tahap Data Understanding dalam metodologi CRISP-DM.

#### - Analisa Korelasi

- Analisis korelasi sepal length dan sepal width:

![original image](https://cdn.mathpix.com/snip/images/sRKmyd2sVULbo8caUWfLT9eqt_lCk_W0yBKY_EifcEA.original.fullsize.png)


Analisis korelasi juga dilakukan pada atribut sepal length dan sepal width menggunakan scatter plot dengan bantuan Python. Berdasarkan hasil visualisasi, terlihat bahwa sebaran data tidak membentuk pola yang jelas, yang menunjukkan bahwa hubungan antara kedua atribut tersebut relatif lemah dibandingkan dengan atribut pada bagian petal.

- Analisis korelasi petal length dan petal width

![original image](https://cdn.mathpix.com/snip/images/8bCKkvYBfRcBPq1RZkXrOMc8sxvl-fqDQ-RhbE9lwRo.original.fullsize.png)

Analisis korelasi dilakukan secara visual menggunakan scatter plot dengan bantuan bahasa pemrograman Python. Scatter plot digunakan untuk melihat hubungan antara atribut petal length dan petal width. Berdasarkan hasil visualisasi, terlihat bahwa titik-titik data membentuk pola yang jelas, yang menunjukkan adanya hubungan atau korelasi yang kuat antara kedua atribut tersebut.


### Eksplorasi Dataset dengan menggunakan Aplikasi Orange:

##### - Statistik Deskriptif 

Statistik deskriptif digunakan untuk memahami karakteristik dasar dataset Iris Flower. Analisis ini dilakukan menggunakan Column Statistics pada aplikasi Orange Data Mining. Hasil statistik deskriptif menunjukkan nilai minimum, maksimum, rata-rata, dan standar deviasi dari setiap atribut, sehingga memberikan gambaran umum mengenai sebaran dan variasi data.

Berikut contoh statistik deskriptifnya:

![original image](https://cdn.mathpix.com/snip/images/1QfYry6Ge2WhH-6mPvVdIRTrstVxnt-bPx3SCKn8EtY.original.fullsize.png)

##### - Analisa kolerasi

Analisis korelasi antar atribut dilakukan secara visual menggunakan widget Scatter Plot pada aplikasi Orange Data Mining. Scatter Plot digunakan untuk melihat hubungan antara dua atribut numerik berdasarkan pola sebaran data. Berdasarkan hasil visualisasi, terlihat bahwa beberapa atribut, khususnya pada bagian petal, memiliki hubungan yang kuat, sedangkan atribut lainnya menunjukkan hubungan yang lebih lemah. Analisis ini membantu dalam memahami keterkaitan antar atribut sebelum masuk ke tahap pemodelan.

##### Berikut contoh Analisa Kolerasinya:

- Analisis korelasi sepal length dan sepal width:

![original image](https://cdn.mathpix.com/snip/images/iaM0IQD8x8ELhZrhLL7d07iobX8siwpqS55Ud7nF7uA.original.fullsize.png)


Berdasarkan hasil visualisasi scatter plot antara sepal length dan sepal width, terlihat bahwa titik-titik data tersebar secara acak dan tidak membentuk pola linear yang jelas. Hal ini menunjukkan bahwa hubungan antara kedua atribut tersebut relatif lemah. Dengan kata lain, perubahan nilai sepal length tidak secara konsisten diikuti oleh perubahan sepal width.

- Analisis korelasi petal length dan petal width

![original image](https://cdn.mathpix.com/snip/images/iT4Xj8fRcAwT4AI_oDVDJKNmDaNI3cv0BFxkJoB2VBw.original.fullsize.png)

Berbeda dengan atribut sepal, scatter plot antara petal length dan petal width menunjukkan pola sebaran yang lebih teratur dan cenderung membentuk hubungan linear positif. Titik-titik data terlihat mengikuti arah tertentu, yang menandakan adanya korelasi yang kuat antara kedua atribut petal tersebut.