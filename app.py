"""
app.py - Streamlit churn prediction UI (contoh referensi untuk Sesi 14).
Jalankan dari terminal (folder ini) dengan:
    streamlit run app.py
"""
import joblib
import pandas as pd
import streamlit as st

# Load model + encoder yang sudah dilatih dan disimpan lewat joblib.dump() (lihat Sesi 9/10).
# bundle berisi 4 hal sekaligus: model, encoder tiap kolom kategorikal, dan daftar nama kolom.
bundle = joblib.load("churn_model.pkl")
model = bundle["model"]
encoders = bundle["encoders"]
feature_cols = bundle["feature_cols"]
cat_cols = bundle["cat_cols"]

# st.title() dan st.write() menampilkan teks di halaman - tanpa perlu HTML sama sekali.
st.title("Prediksi Customer Churn")
st.write("Masukkan data pelanggan di bawah ini untuk memprediksi kemungkinan churn.")

# st.form() mengelompokkan semua input di dalamnya jadi 1 kesatuan - nilai input
# baru diproses saat tombol submit ditekan, bukan tiap kali 1 input diubah.
with st.form("form_prediksi"):
    # st.number_input(): kolom isian angka, dengan batas minimal/maksimal dan nilai default.
    tenure = st.number_input("Lama Berlangganan (bulan)", min_value=0, max_value=100, value=12)
    # st.slider(): geser untuk memilih angka dalam rentang tertentu.
    monthly = st.slider("Monthly Charges", 0, 200, 70)
    total = st.number_input("Total Charges", min_value=0.0, value=800.0)
    # st.selectbox(): dropdown pilihan - list-nya diambil langsung dari kelas yang
    # dikenal encoder, supaya pilihan di UI selalu cocok dengan yang dipelajari model.
    senior = st.selectbox("Senior Citizen?", ["Tidak", "Ya"])
    internet = st.selectbox("Internet Service", list(encoders["internet_service"].classes_))
    contract = st.selectbox("Jenis Kontrak", list(encoders["contract"].classes_))
    payment = st.selectbox("Metode Pembayaran", list(encoders["payment_method"].classes_))
    # st.form_submit_button(): tombol yang memicu form diproses.
    submit = st.form_submit_button("Prediksi")

# Blok ini hanya jalan SETELAH tombol "Prediksi" ditekan.
if submit:
    # Susun input jadi 1 dict, ubah kategori teks (mis. "Ya"/"Tidak") jadi angka
    # lewat encoder yang sama persis dipakai saat melatih model - kalau encoder beda,
    # angka yang dihasilkan bisa tidak sesuai dan prediksi jadi salah.
    row = {
        "senior_citizen": 1 if senior == "Ya" else 0,
        "tenure_bulan": tenure,
        "internet_service": encoders["internet_service"].transform([internet])[0],
        "contract": encoders["contract"].transform([contract])[0],
        "payment_method": encoders["payment_method"].transform([payment])[0],
        "monthly_charges": monthly,
        "total_charges": total,
    }
    # Bungkus jadi DataFrame 1 baris, urutan kolom harus sama dengan feature_cols
    # (urutan yang dipakai model saat dilatih).
    X = pd.DataFrame([row])[feature_cols]
    # model.predict() mengembalikan label (0 = Tidak Churn, 1 = Churn).
    # model.predict_proba() mengembalikan probabilitas untuk tiap kelas.
    hasil = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    hasil_text = "Churn" if hasil == 1 else "Tidak Churn"
    st.write("### Hasil Prediksi:", hasil_text)

    # st.bar_chart() menampilkan grafik batang dari data Pandas Series/DataFrame -
    # di sini dipakai untuk memvisualisasikan probabilitas tiap kelas.
    proba_df = pd.Series(proba, index=["Tidak Churn", "Churn"])
    st.bar_chart(proba_df)
