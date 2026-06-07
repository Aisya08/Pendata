
# Analisa Data Menggunakan Regresi LInier 

Proyek ini  digunakan untuk membuat analisis data menggunakan Regresi Linier  
- membuat program menghitung koefisien regresi dengan  libarary dari sklearn from sklearn.linear_model import LinearRegression
- Menghitung secara analitik mencari koefisien regresi

     
![image](https://hackmd.io/_uploads/SyvGP5cJGe.png)

---

# Pendahuluan

Regresi linier merupakan metode analisis data yang digunakan untuk mengetahui hubungan antara variabel independen (x) dan variabel dependen (y).  
Metode ini digunakan untuk mencari garis terbaik yang mewakili pola data.

# Dataset

Data titik yang digunakan:

| Titik | x | y |
|---|---|---|
| A | 2 | 2 |
| B | 4 | 3 |
| C | 5 | 5 |
| D | 3 | 4 |
| E | 3 | 3 |
| F | 4 | 5 |
| G | 5 | 6 |

---

# Visualisasi Data Menggunakan GeoGebra

## Langkah Memasukkan Data

Masukkan titik satu per satu pada GeoGebra:

```python
A=(2,2)
B=(4,3)
C=(5,5)
D=(3,4)
E=(3,3)
F=(4,5)
G=(5,6)
```

![image](https://hackmd.io/_uploads/rkr-uq5yze.png)


---

# Rumus Regresi Linier

Persamaan regresi linier:

![image](https://hackmd.io/_uploads/B1BVO9ckMl.png)


Rumus mencari slope:
![image](https://hackmd.io/_uploads/HJkvd55Jfg.png)


Rumus mencari intercept:

![image](https://hackmd.io/_uploads/Bk3wd59kzx.png)

Rumus matriks regresi linier:

![image](https://hackmd.io/_uploads/S1Cdd59kGe.png)


---

# Program Python Menggunakan sklearn
![image](https://hackmd.io/_uploads/Bynr3qq1Ge.png)

---

# Perhitungan Manual

## Tabel Perhitungan

| x | y | x² | xy |
|---|---|---|---|
| 2 | 2 | 4 | 4 |
| 4 | 3 | 16 | 12 |
| 5 | 5 | 25 | 25 |
| 3 | 4 | 9 | 12 |
| 3 | 3 | 9 | 9 |
| 4 | 5 | 16 | 20 |
| 5 | 6 | 25 | 30 |

---

## Menghitung Jumlah

$$
\sum x = 26
$$

$$
\sum y = 28
$$

$$
\sum x^2 = 104
$$

$$
\sum xy = 112
$$

$$
n = 7
$$
---

# Menghitung Slope (b)

$$
b=\frac{n\sum xy-(\sum x)(\sum y)}{n\sum x^2-(\sum x)^2}
$$

$$
b=\frac{7(112)-(26)(28)}{7(104)-(26)^2}
=\frac{784-728}{728-676}
=\frac{56}{52}
=1.076923
\approx 1.08
$$

---

# Menghitung Intercept (a)

$$
a=\frac{\sum y-b\sum x}{n}
$$

$$
a=\frac{28-(1.076923)(26)}{7}
=\frac{28-28}{7}
=0
$$

---

# Persamaan Regresi

$$
y = 1.08x
$$
# Perhitungan Menggunakan Excel

## Rumus Excel

Menghitung \(x^2\):

```excel
=A2^2
```

Menghitung \(xy\):

```excel
=A2*B2
```

Menghitung jumlah:

```excel
=SUM(A2:A8)
```

Menghitung slope:

```excel
=((7*112)-(26*28))/((7*104)-(26^2))
```

Menghitung intercept:

```excel
=(28-(1.076923*26))/7
```

---

# Kesimpulan

Berdasarkan hasil perhitungan regresi linier sederhana, diperoleh nilai slope (**b**) sebesar **1.08** dan nilai intercept (**a**) sebesar **0**. Dengan demikian, persamaan regresi yang terbentuk adalah:

$$
y = 1.08x
$$

Persamaan tersebut menunjukkan bahwa setiap kenaikan 1 satuan pada variabel **x** akan meningkatkan nilai **y** sebesar **1.08** satuan. Karena nilai intercept sama dengan 0, garis regresi melewati titik asal (0,0).

Hasil tersebut menunjukkan bahwa setiap kenaikan 1 nilai x akan meningkatkan nilai y sebesar 1.08.