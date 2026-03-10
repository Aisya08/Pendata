# Panduan Cepat Jupyter Book (materi-pendat)

> Panduan ini menggunakan **Jupyter Book < 2.0.0** agar perintah `jupyter-book create` tetap tersedia.

---

## 1. Persiapan Environment (PowerShell)

```powershell
# Dari folder Pendata
..\.venv\Scripts\Activate.ps1

# Jika terkendala execution policy
# Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force
```

---

## 2. Instalasi Dependensi

```powershell
pip install "jupyter-book<2.0.0" ghp-import
```

**Catatan:**
- `jupyter-book<2.0.0` dipakai agar perintah `jupyter-book create` tetap tersedia.
- `ghp-import` dipakai untuk deploy GitHub Pages dari hasil build HTML.

---

## 3. Membuat Struktur Buku Baru

```powershell
jupyter-book create materi-pendat
```

Perintah ini akan membuat folder `materi-pendat/` dengan struktur:

```
materi-pendat/
├── _config.yml
├── _toc.yml
├── intro.md
├── markdown.md
├── markdown-notebooks.md
├── notebooks.ipynb
├── references.bib
├── requirements.txt
└── logo.png
```

---

## 4. Menambahkan Halaman Baru

1. Buat file markdown, misalnya `materi-pendat/logika.md`.
2. Daftarkan ke daftar isi di `materi-pendat/_toc.yml`.

**Contoh `_toc.yml`:**

```yaml
format: jb-book
root: intro
chapters:
  - file: pertemuan1
  - file: Penambangan_Data_A_Pertemuan_2
  - file: pertemuan3
  - file: logika
```

---

## 5. Konfigurasi `_config.yml`

Pastikan menggunakan `myst_enable_extensions` (bukan `myst_extensions` yang merupakan sintaks JB2):

```yaml
# Book settings
title: Penambangan Data - Materi
author: "Nama (NIM: XXXXXXXXX)"

execute:
  execute_notebooks: force

bibtex_bibfiles:
  - references.bib

repository:
  url: https://github.com/username/nama-repo
  path_to_book: Materi
  branch: main

html:
  use_issues_button: true
  use_repository_button: true

parse:
  myst_enable_extensions:
    - html_admonition
    - html_image
    - dollarmath
    - linkify
    - substitution
    - deflist
    - colon_fence
    - smartquotes
    - replacements
```

> **Penting:** Di JB1 gunakan `myst_enable_extensions`, bukan `myst_extensions`.

---

## 6. Build Website

```powershell
jupyter-book build Materi
```

Hasil website ada di:

```
Materi/_build/html/index.html
```

Buka file tersebut di browser untuk preview.

Jika ingin **clean build** (hapus cache lama):

```powershell
jupyter-book clean Materi --all
jupyter-book build Materi
```

---

## 7. Panduan Git (Push ke Origin)

### 7.1 Cek perubahan

```powershell
git status
```

### 7.2 Stage semua perubahan

```powershell
git add .
```

### 7.3 Commit

```powershell
git commit -m "update materi pertemuan 3"
```

### 7.4 Push ke branch utama

```powershell
git push origin main
```

---

## 8. Deploy GitHub Pages

Setelah build berhasil, deploy folder HTML ke branch `gh-pages`:

```powershell
ghp-import -n -p -f Materi/_build/html
```

**Keterangan opsi:**
| Opsi | Keterangan |
|------|------------|
| `-n` | Membuat file `.nojekyll` |
| `-p` | Langsung push ke remote |
| `-f` | Force overwrite branch `gh-pages` |

---

## 9. Setting GitHub Pages di Repository

1. Buka repository di GitHub → **Settings** → **Pages**
2. Di bagian **Source**, pilih:
   - **Branch:** `gh-pages`
   - **Folder:** `/ (root)`
3. Klik **Save**
4. Tunggu beberapa menit, situs akan tersedia di:

```
https://username-kamu.github.io/nama-repo-kamu/
```

---

## 10. Workflow Lengkap (Ringkasan)

```powershell
# 1. Aktifkan virtual environment
..\.venv\Scripts\Activate.ps1

# 2. Edit file markdown di folder Materi/

# 3. Build buku
jupyter-book build Materi

# 4. Preview di browser (buka Materi/_build/html/index.html)

# 5. Commit & push source code
git add .
git commit -m "update materi"
git push origin main

# 6. Deploy ke GitHub Pages
ghp-import -n -p -f Materi/_build/html
```

---

## Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `jupyter-book: command not found` | Pastikan virtual environment sudah aktif |
| Build error pada MyST extension | Gunakan `myst_enable_extensions` di `_config.yml` (bukan `myst_extensions`) |
| Halaman tidak muncul | Pastikan file sudah didaftarkan di `_toc.yml` |
| GitHub Pages 404 | Cek Settings → Pages, pastikan branch `gh-pages` dan folder `/ (root)` |
| Execution timeout | Tambah `nb_execution_timeout: 300` di `sphinx.config` pada `_config.yml` |
| Clean build diperlukan | `jupyter-book clean Materi --all` lalu build ulang |
