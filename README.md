# CitraGambar

### Richie Pranata
### 312410451
# Dokumentasi Perbaikan Code: Pencocokan Fitur menggunakan Algoritma ORB

Repository ini berisi implementasi perbaikan kode untuk melakukan pencocokan fitur (*feature matching*) antara dua gambar menggunakan algoritma **ORB (Oriented FAST and Rotated BRIEF)** di Python menggunakan OpenCV lokal (VS Code).

---

## 🛠️ Library yang Digunakan

Untuk menjalankan proyek ini di lingkungan lokal (seperti VS Code), beberapa library utama yang wajib diinstal adalah:
* **`opencv-python`**: Library utama untuk pemrosesan citra (Computer Vision) yang menyediakan fungsi ORB, pengubahan warna, dan visualisasi garis cocok.
* **`numpy`**: Digunakan untuk mendukung OpenCV dalam memproses gambar sebagai matriks angka.
* **`matplotlib`**: Digunakan untuk menampilkan grafik dan hasil gambar akhir dalam bentuk jendela pop-up di komputer lokal.

---

## 📝 Penjelasan Langkah-Langkah Perbaikan Kode

Terdapat beberapa perbaikan krusial dari kode draf awal agar program dapat berjalan tanpa eror di lingkungan lokal:

### 1. Perbaikan Sintaksis dan Typo Variabel
* **Masalah:** Pada draf awal, terdapat banyak kesalahan pengetikan spasi pada variabel, seperti `query Keypoints`, `query Descriptors`, dan `draw Matches`. Di Python, spasi di tengah nama variabel akan menyebabkan *SyntaxError*.
* **Solusi:** Semua nama variabel disatukan secara konsisten menggunakan standar *camelCase* atau *snake_case* (misal: `queryKeypoints` dan `queryDescriptors`).

### 2. Penanganan Masalah Jalur File (Absolute Path) & Ekstensi Ganda
* **Masalah:** Program sering kali memunculkan eror `Gambar tidak ditemukan!` karena OpenCV (`cv2.imread`) gagal membaca file gambar. Hal ini disebabkan oleh sistem operasi Windows yang sering kali menyembunyikan ekstensi asli file, sehingga nama file yang dikira `a.jpg` sebenarnya terbaca sebagai `a.jpg.jpg` oleh sistem. Selain itu, menjalankan terminal di direktori yang salah membuat jalur relatif tidak bekerja.
* **Solusi:** Kode diperbaiki menggunakan fungsi `os.path.join()` dan `os.path.dirname()` untuk mengunci jalur folder secara absolut berdasarkan posisi file `main.py`. Nama file gambar juga disesuaikan secara eksplisit menjadi `'a.jpg.jpg'` dan `'b.jpg.jpg'` untuk menyesuaikan dengan sifat pembacaan ekstensi tersembunyi di Windows.

### 3. Konsistensi Variabel Visualisasi dan Resize
* **Masalah:** Pada draf awal, fungsi penggambaran menggunakan variabel output `l_img`, namun pada baris pengubahan ukuran (*resize*), kode justru memanggil `final_img`. Ketidakcocokan ini menyebabkan eror *NameError* (variabel belum didefinisikan).
* **Solusi:** Nama variabel diseragamkan sepenuhnya menjadi `final_img` mulai dari proses `cv2.drawMatches()`, `cv2.resize()`, hingga penampilan di `plt.imshow()`.

### 4. Optimalisasi Parameter Matcher dan Pengurutan Jarak
* **Masalah:** Draf kode awal langsung mengambil potongan `matches[:20]` tanpa melakukan pengurutan (*sorting*) terlebih dahulu. Hal ini membuat 20 titik yang diambil belum tentu titik pencocokan yang paling akurat.
* **Solusi:** Ditambahkan fungsi pencocokan dengan parameter biner `cv2.NORM_HAMMING` (yang sangat direkomendasikan untuk ORB) serta fungsi `sorted(matches, key=lambda x: x.distance)` untuk mengurutkan kecocokan dari jarak terpendek (paling akurat) ke tertinggi sebelum divisualisasikan.

---
## 💻 Kode Python yang Sudah Diperbaiki (`main.py`)

```python
# ==========================================
# 1. IMPORT LIBRARY
# ==========================================
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 2. MEMBACA GAMBAR & KONVERSI KE GRAYSCALE
# ==========================================
# Mengambil jalur folder tempat file main.py ini berada
current_dir = os.path.dirname(os.path.abspath(__file__))

# Membaca gambar menggunakan nama file dengan ekstensi tersembunyi Windows
query_img = cv2.imread(os.path.join(current_dir, 'a.jpg.jpg'))
train_img = cv2.imread(os.path.join(current_dir, 'b.jpg.jpg'))

# Pengecekan apakah gambar berhasil dimuat
if query_img is None or train_img is None:
    print("Error: Gambar masih tidak terbaca. Pastikan nama file di kiri sudah benar-benar 'a.jpg' dan 'b.jpg'.")
    exit()

# Mengubah gambar ke skala abu-abu (Grayscale) karena ORB bekerja pada satu saluran
query_img_bw = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

# ==========================================
# 3. DETEKSI TITIK KUNCI & DESKRIPTOR (ORB)
# ==========================================
# Inisialisasi detektor ORB (Oriented FAST and Rotated BRIEF)
orb = cv2.ORB_create()

# Mendeteksi titik kunci (keypoints) dan menghitung deskriptor (descriptors)
queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw, None)
trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw, None)

# ==========================================
# 4. PENCOCOKAN DESKRIPTOR (BFMatcher)
# ==========================================
# Inisialisasi Brute-Force Matcher dengan NORM_HAMMING yang cocok untuk algoritma berbasis biner seperti ORB
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Mencocokkan deskriptor dari kedua gambar
matches = matcher.match(queryDescriptors, trainDescriptors)

# Mengurutkan kecocokan berdasarkan jarak terkecil (akurasi terbaik)
matches = sorted(matches, key=lambda x: x.distance)

# ==========================================
# 5. VISUALISASI HASIL KECOCOKAN
# ==========================================
# Menggambar 20 titik kecocokan terbaik agar visualisasi tetap rapi dan informatif
final_img = cv2.drawMatches(
    query_img, queryKeypoints, 
    train_img, trainKeypoints, 
    matches[:20], None, 
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# Mengubah ukuran gambar hasil agar pas saat ditampilkan di layar komputer
final_img = cv2.resize(final_img, (1000, 650))

# Menampilkan hasil akhir menggunakan jendela pop-up Matplotlib
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
plt.title("Feature Matches menggunakan ORB di VS Code")
plt.axis('off')
plt.show()  # Menampilkan jendela pop-up grafis hasil pencocokan

