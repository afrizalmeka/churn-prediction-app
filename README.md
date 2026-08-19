# Churn Prediction App

Contoh aplikasi model serving untuk Sesi 14 (Python Programming for AI - Batch 8). Model machine learning (prediksi customer churn) di-deploy jadi dua bentuk:

- **FastAPI** (`main.py`) - REST API dengan endpoint prediksi, otomatis punya dokumentasi interaktif (Swagger UI).
- **Streamlit** (`app.py`) - antarmuka visual sederhana, tanpa perlu menulis HTML/CSS.

Model (`churn_model.pkl`) adalah Decision Tree yang dilatih dari dataset `telco_churn.csv` (Sesi 10), memprediksi apakah seorang pelanggan berpotensi churn (berhenti berlangganan) atau tidak.

## Struktur Folder

```
churn-prediction-app/
├── main.py              # FastAPI - REST API model serving
├── app.py                # Streamlit - antarmuka visual
├── churn_model.pkl        # Model terlatih (Decision Tree + encoder)
├── requirements.txt       # Daftar dependency Python
├── .gitignore
└── README.md
```

## Cara Kerja Tiap File

Kalau kamu terbiasa kerja di 1 file `.ipynb`, project ini mungkin terasa asing karena kodenya terpisah jadi beberapa file. Ini pola umum yang dipakai hampir semua project Python di dunia nyata:

- **`main.py`** - kode FastAPI. Isinya endpoint (alamat) yang bisa diakses lewat HTTP, misalnya `/prediksi-churn`. File ini yang dijalankan `uvicorn` untuk jadi server API yang terus aktif.
- **`app.py`** - kode Streamlit. Isinya UI (form, tombol, grafik) yang tampil di browser. Dijalankan dengan `streamlit run`.
- **`churn_model.pkl`** - bukan kode, ini model ML yang sudah dilatih dan disimpan (`joblib.dump()`). `main.py` dan `app.py` sama-sama membaca file ini untuk prediksi, tanpa perlu latih ulang tiap kali dijalankan.
- **`requirements.txt`** - daftar library yang dibutuhkan beserta versinya, pengganti `!pip install` yang biasa diketik di Colab. Dengan 1 file ini, siapapun bisa `pip install -r requirements.txt` dan mendapat environment yang sama persis.
- **`.gitignore`** - daftar file/folder yang sengaja tidak ikut di-commit ke Git (misalnya `riwayat.db`, isinya data hasil testing, bukan kode).
- **`README.md`** - file ini sendiri: penjelasan project untuk siapapun yang membukanya pertama kali.

Penjelasan lebih lengkap (termasuk kenapa strukturnya beda dari notebook) ada di dokumen `Panduan_Setup_dan_Troubleshooting_Sesi14.docx` bagian A.

## Instalasi

Pastikan Python 3.9+ sudah terpasang. Dari dalam folder ini, jalankan:

```bash
pip install -r requirements.txt
```

## Menjalankan FastAPI

```bash
uvicorn main:app --reload
```

Buka `http://127.0.0.1:8000/docs` di browser untuk mengakses Swagger UI - dokumentasi interaktif yang otomatis dibuat FastAPI, bisa langsung dipakai untuk mencoba endpoint tanpa menulis kode client.

### Endpoint yang tersedia

| Method | Path | Deskripsi |
|---|---|---|
| GET | `/` | Cek API sudah aktif |
| POST | `/prediksi-churn` | Kirim data pelanggan, dapatkan hasil prediksi |
| GET | `/riwayat` | Lihat riwayat prediksi yang tersimpan di SQLite |

Contoh request ke `/prediksi-churn`:

```json
{
  "senior_citizen": 0,
  "tenure_bulan": 5,
  "internet_service": "Fiber optic",
  "contract": "Month-to-month",
  "payment_method": "Electronic check",
  "monthly_charges": 85.5,
  "total_charges": 420.0
}
```

Setiap prediksi otomatis disimpan ke `riwayat.db` (SQLite, dibuat otomatis saat pertama kali dijalankan).

## Menjalankan Streamlit

```bash
streamlit run app.py
```

Browser akan terbuka otomatis menampilkan form input data pelanggan dan hasil prediksi dalam bentuk visual (termasuk grafik probabilitas).

## Troubleshooting

Lihat dokumen `Panduan_Setup_dan_Troubleshooting_Sesi14.docx` di folder induk (`Batch 8/code/Sesi14_Model_Serving_dan_AI_Deployment/`) untuk panduan setup lengkap dan solusi masalah umum (port bentrok, module not found, model tidak ditemukan, dsb).

## Konteks

Bagian dari materi Sesi 14 - Model Serving & AI Deployment, kurikulum Python Programming for AI Batch 8 (rubythalib.ai).
