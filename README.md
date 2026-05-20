# 😄😠 AI Pendeteksi Ekspresi Wajah

Halo guys 👋  
Ini adalah project AI sederhana yang bisa ngedeteksi ekspresi wajah manusia secara real-time lewat kamera 🎥

AI ini cuma fokus ke 2 ekspresi aja:
- 😄 Senyum (Happy)
- 😠 Marah (Angry)

Cocok banget buat belajar AI, computer vision, atau sekadar iseng bikin project keren 😎

---

## 🚀 Fitur

- Deteksi wajah secara real-time
- Klasifikasi ekspresi: Senyum & Marah
- Jalan langsung dari webcam
- Ringan dan bisa dijalankan di browser / local server
- Dan Juga Di Handphone/Iphone

---

## 🧠 Cara Kerja

1. Kamera nyala dan ambil frame video
2. AI model (ONNX) baca wajah
3. Model nentuin ekspresi:
   - 😄 Happy kalau wajah terlihat senyum
   - 😠 Angry kalau wajah terlihat tegang / marah
4. Hasilnya ditampilin di layar secara real-time

---

## 🛠️ Teknologi yang Dipakai

- HTML, CSS, JavaScript
- OpenCV / TensorFlow.js (opsional)
- ONNX Model (hasil training dari Roboflow / dataset sendiri)

---

## 📂 Struktur Project
/project-root
│
├── index.html
├── app.js
├── style.css
├── model/
│ └── best.onnx
└── assets/
└── (optional: icon, image, dll)

---

## 📊 Dataset & Training Model

Model dilatih pakai dataset ekspresi wajah dengan variasi:

- Wajah orang berbeda-beda
- Lighting terang & gelap
- Sudut kamera berbeda
- Ekspresi senyum & marah

Tahapan training:
1. Kumpulin dataset ekspresi wajah
2. Labeling: `Senang` & `Marah`
3. Training model AI (misalnya di Roboflow / Python)
4. Export model ke ONNX
5. Integrasi ke web app

---

## ▶️ Cara Menjalankan

1. Clone project ini
2. Buka `index.html` di browser
   atau pakai local server (disarankan)
3. Izinkan akses kamera
4. Tampilkan ekspresi kamu 😿😾

---

## 🌐 Deployment

Project ini sudah kami deploy menggunakan **GitHub Pages** 🚀

👉 kami mendeploy project ini di GitHub agar bisa diakses secara online melalui browser.

Langkah deploy:
- Upload project ke GitHub repository
- Aktifkan GitHub Pages di settings
- Jalankan dari branch `main`

---

## ⚠️ Catatan

- Hasil AI tergantung kualitas dataset
- Kadang bisa salah deteksi kalau lighting jelek
- Ini masih project learning, bukan AI super canggih ya 😆

---

## 💡 Pengembangan Selanjutnya

Kalau mau upgrade, bisa banget:
- Tambah ekspresi lain (sedih, kaget, dll)
- Improve accuracy model
- Tambah UI yang lebih keren
- Deploy ke web hosting

---

## 👨‍💻 Dibuat oleh:
- Tri Nur Raafi  
- Ragga Hafidz Fianugra  
- Ziddan Mulki Akbar

---

## ❤️ Penutup

Project ini cocok banget buat belajar dasar AI + web.  
Kalau mau ngembangin lagi, tinggal improvisasi aja 🔥

Happy coding bro 👨‍💻✨
