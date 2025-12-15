import streamlit as st
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt

# --- 0. Path file model & scaler ---
MODEL_PATH = 'kmeans_model.pkl'
SCALER_PATH = 'scaler.pkl'

# --- 1. Load Model dan Scaler ---
try:
    with open(MODEL_PATH, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(SCALER_PATH, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError:
    st.error(
        "File model/scaler tidak ditemukan. "
        "Pastikan 'kmeans_model.pkl' dan 'scaler.pkl' ada di folder yang sama dengan app.py."
    )
    sys.exit()

# --- 2. Konfigurasi halaman Streamlit ---
st.set_page_config(
    page_title="DSS Klasterisasi Stres Mahasiswa",
    layout="wide"
)

st.title("💡 Sistem Pendukung Keputusan Klasterisasi Tingkat Stres")
st.subheader("Berdasarkan Gejala dan Aktivitas Harian (Metode K-Means)")

st.markdown("""
Masukkan jawaban Anda pada skala 1 hingga 5.  
Angka 1 = Sangat Jarang/Tidak Sama Sekali, Angka 5 = Sering/Sangat Sering.
""")

# --- 3. Input Pengguna ---
st.header("Input Gejala & Aktivitas")
col1, col2 = st.columns(2)

with col1:
    input_tidur = st.slider("1. Kesulitan tidur atau masalah tidur lainnya?", 1, 5, 3)
    input_beban = st.slider("3. Terbebani tugas akademik?", 1, 5, 3)

with col2:
    input_konsen = st.slider("2. Kesulitan konsentrasi belajar?", 1, 5, 3)
    input_relaks = st.slider("4. Kesulitan mencari waktu santai dan relaksasi?", 1, 5, 3)

# --- 4. Visualisasi input gejala ---
data_labels = ['Tidur', 'Konsentrasi', 'Beban Akademik', 'Relaksasi']
data_values = [input_tidur, input_konsen, input_beban, input_relaks]

fig, ax = plt.subplots()
ax.bar(data_labels, data_values, color=['#4CAF50', '#2196F3', '#FFC107', '#FF5722'])
ax.set_ylim(0, 5)
ax.set_ylabel("Skor")
ax.set_title("Skor Gejala & Aktivitas")
st.pyplot(fig)

# --- 5. Tombol Prediksi ---
if st.button("📈 Prediksi Tingkat Stres"):
    # Siapkan data input dan normalisasi
    data_input = np.array([[input_tidur, input_konsen, input_beban, input_relaks]])
    X_scaled = scaler.transform(data_input)
    cluster = model.predict(X_scaled)[0]

    # Interpretasi cluster
    if cluster == 0:
        tingkat = "RENDAH"
        emoji = "😊"
        saran = "Pertahankan keseimbangan hidup Anda. Stres Anda terkendali."
        st.success(f"Cluster RENDAH {emoji}")
    elif cluster == 1:
        tingkat = "SEDANG"
        emoji = "😐"
        saran = "Perlu penyesuaian! Atur ulang jadwal akademik dan cari waktu relaksasi terencana."
        st.warning(f"Cluster SEDANG {emoji}")
    else:
        tingkat = "TINGGI"
        emoji = "😨"
        saran = "Tingkat stres tinggi. Segera istirahat, lakukan aktivitas fisik, dan pertimbangkan bantuan konseling."
        st.error(f"Cluster TINGGI {emoji}")

    # Tampilkan hasil metric
    st.divider()
    st.metric(
        label="Hasil Klasterisasi Tingkat Stres",
        value=f"{tingkat} {emoji}",
        delta=f"Klaster {cluster}"
    )

    # Saran lengkap di expander
    with st.expander("📋 Lihat saran lengkap"):
        st.write("""
        - Tidur cukup 7–8 jam per hari  
        - Atur jadwal belajar & relaksasi  
        - Olahraga ringan 20–30 menit  
        - Teknik relaksasi (meditasi/pernapasan)  
        - Konsultasi jika stres tinggi
        """)

    st.info(f"**Saran Pendukung Keputusan:** {saran}")
