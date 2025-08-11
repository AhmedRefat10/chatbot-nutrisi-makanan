
# Chatbot Gizi Berbasis Gambar

Aplikasi **Chatbot Gizi Berbasis Streamlit** ini memungkinkan pengguna untuk mengunggah foto makanan dan mendapatkan informasi kandungan gizi secara otomatis menggunakan model machine learning (TensorFlow Lite). Selain itu, pengguna juga dapat berinteraksi dengan chatbot untuk bertanya lebih lanjut mengenai nutrisi makanan, baik berdasarkan gambar yang diunggah maupun secara manual.

## 📌 Fitur Utama
- **Upload Foto Makanan** → Sistem akan memprediksi jenis makanan menggunakan model TFLite.
- **Kalkulasi Gizi Otomatis** → Menghitung kalori, protein, lemak, dan karbohidrat berdasarkan data JSON nutrisi per 100 gram.
- **Chatbot Interaktif** → Pengguna dapat mengajukan pertanyaan seperti *"Berapa kalori dalam 2 porsi?"* atau *"Berapa protein ayam goreng per 200 gram?"*.
- **UI Chat Bubble** → Tampilan percakapan layaknya aplikasi chat.
- **Histori Chat** → Menyimpan riwayat percakapan selama sesi berjalan.

## 🛠️ Teknologi yang Digunakan
- **Python 3.x**
- **Streamlit**
- **TensorFlow Lite**
- **Pandas & NumPy** untuk pengolahan data
- **JSON** sebagai basis data nutrisi

## 📂 Struktur Proyek
```
├── app.py               # Script utama aplikasi
├── nutrition.json       # Data nutrisi per 100 gram
├── model.tflite         # Model TensorFlow Lite
├── labels.txt           # Label kelas makanan
├── README.md            # Dokumentasi proyek
```

## 🚀 Cara Menjalankan
1. **Clone repository** ini atau salin semua file ke folder lokal.
2. Pastikan Python dan pip sudah terinstal.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Jalankan aplikasi Streamlit:
   ```bash
   streamlit run app.py
   ```

## 📌 Cara Menggunakan
1. **Unggah gambar makanan** melalui panel upload.
2. **Lihat prediksi makanan** dan informasi gizi otomatis.
3. **Gunakan chatbot** untuk bertanya lebih detail tentang gizi makanan.
4. Histori percakapan akan muncul di layar.

## 📝 Catatan
- Data nutrisi dalam `nutrition.json` berbasis **per 100 gram**. Jika ingin menghitung porsi, chatbot akan mengalikannya otomatis.
- Model TFLite harus sesuai dengan label di `labels.txt`.

## 📜 Lisensi
Proyek ini dibuat untuk tujuan pembelajaran dan penelitian.
