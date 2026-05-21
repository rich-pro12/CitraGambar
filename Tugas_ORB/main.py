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
current_dir = os.path.dirname(os.path.abspath(__file__))

# Membaca gambar menggunakan nama file baru yang super bersih
query_img = cv2.imread(os.path.join(current_dir, 'a.jpg'))
train_img = cv2.imread(os.path.join(current_dir, 'b.jpg'))

# Pengecekan apakah gambar berhasil dimuat
if query_img is None or train_img is None:
    print("Error: Gambar masih tidak terbaca. Pastikan nama file di kiri sudah benar-benar 'a.jpg' dan 'b.jpg'.")
    exit()

# Mengubah gambar ke skala abu-abu (Grayscale)
query_img_bw = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

# ==========================================
# 3. DETEKSI TITIK KUNCI & DESKRIPTOR (ORB)
# ==========================================
orb = cv2.ORB_create()
queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw, None)
trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw, None)

# ==========================================
# 4. PENCOCOKAN DESKRIPTOR (BFMatcher)
# ==========================================
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(queryDescriptors, trainDescriptors)
matches = sorted(matches, key=lambda x: x.distance)

# ==========================================
# 5. VISUALISASI HASIL KECOCOKAN
# ==========================================
final_img = cv2.drawMatches(
    query_img, queryKeypoints, 
    train_img, trainKeypoints, 
    matches[:20], None, 
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

final_img = cv2.resize(final_img, (1000, 650))

plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
plt.title("Feature Matches menggunakan ORB di VS Code")
plt.axis('off')
plt.show()