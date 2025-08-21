# 🍜 Food Tourism Assistant Indonesia

Aplikasi **Food Tourism Assistant** berbasis **Streamlit** yang membantu wisatawan mengenal kuliner Indonesia.  
Pengguna dapat mengunggah foto makanan, aplikasi akan mengenali jenis makanan menggunakan model **TensorFlow Lite**, lalu chatbot akan memberikan penjelasan budaya, bahan utama, serta tips menikmati makanan tersebut berdasarkan basis data JSON.

## ✨ Fitur Utama

- 📷 **Upload Foto Makanan** → Prediksi jenis makanan khas Indonesia dengan model TFLite.
- 🧑‍🍳 **Penjelasan Kuliner** → Menampilkan deskripsi, bahan utama, cita rasa, dan konteks budaya dari JSON.
- 💬 **Chatbot Interaktif** → Wisatawan bisa bertanya seputar makanan, cara penyajian, hingga rekomendasi kuliner.
- 💡 **UI Chat Bubble** → Tampilan percakapan mirip aplikasi chat.
- 🕒 **Histori Chat** → Menyimpan riwayat percakapan selama sesi.

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
