"""
main.py - FastAPI churn prediction API (contoh referensi untuk Sesi 14).
Jalankan dari terminal (folder ini) dengan:
    uvicorn main:app --reload
Lalu buka http://127.0.0.1:8000/docs untuk Swagger UI.
"""
import sqlite3

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# FastAPI() membuat instance aplikasi - variabel "app" inilah yang dicari uvicorn
# saat dijalankan dengan "uvicorn main:app" (main = nama file, app = nama variabel ini).
app = FastAPI(title="Churn Prediction API")

# Load model + encoder sekali saja saat server pertama kali dijalankan (bukan tiap
# ada request masuk) - supaya prediksi jadi cepat, tidak perlu load ulang file tiap saat.
bundle = joblib.load("churn_model.pkl")
model = bundle["model"]
encoders = bundle["encoders"]
feature_cols = bundle["feature_cols"]
cat_cols = bundle["cat_cols"]

DB_PATH = "riwayat.db"


# Membuat tabel SQLite untuk menyimpan riwayat prediksi kalau belum ada
# (reuse pengetahuan SQL dari Sesi 6 - CREATE TABLE IF NOT EXISTS supaya aman
# dipanggil berulang tanpa error kalau tabelnya sudah ada).
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prediksi (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tenure_bulan INTEGER,
            contract TEXT,
            monthly_charges REAL,
            hasil TEXT
        )
        """
    )
    conn.commit()
    conn.close()


# Dipanggil sekali saat file ini di-import/dijalankan, supaya tabel sudah siap
# sebelum ada request yang mencoba insert data ke dalamnya.
init_db()


# Pydantic BaseModel mendefinisikan bentuk data yang diharapkan endpoint di bawah.
# FastAPI otomatis MEMVALIDASI request masuk sesuai tipe data ini - kalau ada field
# yang salah tipe/hilang, request ditolak otomatis sebelum masuk ke fungsi endpoint.
class DataPelanggan(BaseModel):
    senior_citizen: int
    tenure_bulan: int
    internet_service: str
    contract: str
    payment_method: str
    monthly_charges: float
    total_charges: float


# @app.get("/") mendaftarkan endpoint GET di alamat root ("/").
# Endpoint sederhana ini berguna untuk cek server sudah aktif atau belum.
@app.get("/")
def read_root():
    return {"message": "API churn prediction siap dipakai"}


# @app.post("/prediksi-churn") mendaftarkan endpoint POST - dipakai karena kita
# MENGIRIM data (bukan cuma mengambil), sesuai konsep GET vs POST dari materi Sesi 14.
# Parameter "data: DataPelanggan" otomatis divalidasi FastAPI sesuai schema di atas.
@app.post("/prediksi-churn")
def prediksi_churn(data: DataPelanggan):
    try:
        # Ubah data Pydantic jadi dict biasa, lalu encode kolom kategorikal
        # (teks) jadi angka pakai encoder yang SAMA PERSIS dipakai saat training -
        # supaya angka yang dihasilkan konsisten dengan yang dipelajari model.
        row = data.dict()
        for col in cat_cols:
            row[col] = encoders[col].transform([row[col]])[0]

        # Bungkus jadi DataFrame 1 baris dengan urutan kolom sesuai feature_cols,
        # lalu panggil model.predict() untuk dapat hasil prediksi (0 atau 1).
        X = pd.DataFrame([row])[feature_cols]
        hasil = model.predict(X)[0]
        hasil_text = "Churn" if hasil == 1 else "Tidak Churn"

        # Simpan hasil prediksi ke SQLite sebagai riwayat - setiap kali endpoint
        # ini dipanggil, 1 baris baru ditambahkan ke tabel "prediksi".
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO prediksi (tenure_bulan, contract, monthly_charges, hasil) VALUES (?, ?, ?, ?)",
            (data.tenure_bulan, data.contract, data.monthly_charges, hasil_text),
        )
        conn.commit()
        conn.close()

        return {"prediksi": hasil_text}
    except Exception as e:
        # Kalau ada error apapun di atas (data aneh, model gagal prediksi, dst),
        # tangkap di sini dan kembalikan sebagai response error yang rapi (status 500)
        # alih-alih membuat server crash.
        raise HTTPException(status_code=500, detail=str(e))


# @app.get("/riwayat") - endpoint GET untuk membaca kembali semua riwayat prediksi
# yang sudah tersimpan di SQLite, dikembalikan dalam format JSON.
@app.get("/riwayat")
def get_riwayat():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT * FROM prediksi").fetchall()
    conn.close()
    return {"riwayat": rows}
