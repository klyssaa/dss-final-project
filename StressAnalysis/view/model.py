import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle
import sys

# *************************************************************************
# FITUR KUESIONER YANG SUDAH DIPERBAIKI BERDASARKAN OUTPUT DATA ANDA
# Kolom yang dipilih fokus pada masalah tidur, konsentrasi, beban akademik, dan relaksasi.
# *************************************************************************
FITUR_AKTIVITAS = [
    'Do you face any sleep problems or difficulties falling asleep?', 
    'Do you have trouble concentrating on your academic tasks?',     
    'Do you feel overwhelmed with your academic workload?',
    'Do you struggle to find time for relaxation and leisure activities?'
] 
# *************************************************************************

# 1. Load dataset
try:
    # Nama file data Anda
    data = pd.read_csv('Stress_Dataset.csv') 
except FileNotFoundError:
    print("Error: File 'Stress_Dataset.csv' tidak ditemukan. Pastikan file ada di folder yang sama.")
    sys.exit()

# 2. Pilih fitur
try:
    X = data[FITUR_AKTIVITAS]
except KeyError:
    print(f"\nFATAL ERROR: KeyError terjadi.")
    print("Pastikan nama kolom di variabel FITUR_AKTIVITAS SAMA PERSIS dengan yang ada di CSV.")
    sys.exit()

# 3. Normalisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Model K-Means (K=3: Rendah, Sedang, Tinggi)
# n_init=10 ditambahkan untuk menghindari warning pada scikit-learn terbaru
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10) 
kmeans.fit(X_scaled)

# 5. Simpan model dan scaler
# Model disimpan di folder induk (../) agar mudah diakses oleh app.py
try:
    with open('../kmeans_model.pkl', 'wb') as f:
        pickle.dump(kmeans, f)
    with open('../scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("\n*** SUCCESS ***")
    print("Pelatihan selesai. Model (kmeans_model.pkl) dan Scaler (scaler.pkl) berhasil disimpan.")
    print("File disimpan di folder induk (C:\\StressAnalysis).")
except Exception as e:
    print(f"Gagal menyimpan file: {e}")