# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding

**Jaya Jaya Institut** merupakan perguruan tinggi yang telah berdiri sejak tahun 2000. Selama lebih dari dua dekade, institusi ini berhasil melahirkan banyak lulusan yang berprestasi di berbagai bidang. Namun demikian, seperti banyak institusi pendidikan lainnya, Jaya Jaya Institut menghadapi tantangan serius terkait tingginya jumlah siswa yang tidak menyelesaikan pendidikannya (dropout).

Masalah **dropout** ini memiliki dampak besar, mulai dari merusak citra institusi, menurunkan tingkat kelulusan, hingga memengaruhi minat calon mahasiswa baru. Tingginya angka dropout juga dapat mengindikasikan adanya masalah dalam proses akademik, penerimaan, atau dukungan yang diberikan kepada mahasiswa.

### Permasalahan Bisnis

Beberapa pertanyaan utama yang ingin dijawab melalui proyek ini antara lain:

1. Bagaimana cara mengidentifikasi siswa yang berisiko dropout sejak awal?
2. Faktor apa saja yang paling memengaruhi keputusan siswa untuk berhenti kuliah?
3. Apa langkah konkret yang dapat dilakukan untuk meningkatkan tingkat kelulusan dan mengurangi dropout?

### Ruang Lingkup Proyek

* **Analisis Data:** Menggali data historis untuk menemukan faktor kunci penyebab dropout.
* **Visualisasi & Pelaporan:** Mengembangkan dashboard interaktif untuk memantau dan menganalisis indikator dropout.
* **Rekomendasi Tindakan:** Memberikan saran intervensi yang berbasis data untuk menekan angka dropout.

### Persiapan Proyek

Sumber data berasal dari dataset internal Jaya Jaya Institut. Proses setup dilakukan melalui:

* Pembuatan lingkungan kerja menggunakan conda
* Instalasi dependensi
* Setup database dan Metabase untuk analisis lanjutan
* Pengiriman dataset ke database melalui SQLAlchemy

## Dashboard Bisnis

Dashboard yang dikembangkan dalam proyek ini memberikan wawasan menyeluruh terkait faktor-faktor yang mempengaruhi status siswa (dropout, enrolled, graduated). Dengan menggunakan dashboard ini, institusi dapat:

1. **Memantau Tren Dropout Secara Real-Time:** Mengidentifikasi lonjakan angka dropout dan segera mengambil tindakan.
2. **Menganalisis Penyebab Dropout:** Melihat hubungan antara status siswa dan berbagai faktor akademik maupun sosial-ekonomi.

🔗 **Lihat Dashboard di Looker Studio:**
[https://lookerstudio.google.com/reporting/aee31d18-8d2e-4986-88ee-521dec85993b](https://lookerstudio.google.com/reporting/aee31d18-8d2e-4986-88ee-521dec85993b)

## Pengoperasian Sistem Machine Learning

Prototipe sistem machine learning telah dikembangkan untuk memprediksi status siswa berdasarkan data yang tersedia. Untuk menjalankannya secara lokal:

```bash
streamlit run app.py
```

Atau gunakan versi daring yang telah disiapkan pada platform streamlit community.

## Kesimpulan

Proyek ini menjawab berbagai tantangan yang dihadapi oleh Jaya Jaya Institut dalam menurunkan angka dropout mahasiswa.

1. **Identifikasi Dini Siswa Berisiko**
   Dengan memanfaatkan model Random Forest, institusi mampu memprediksi siswa yang berisiko tinggi dropout berdasarkan data historis dan variabel yang relevan.

2. **Faktor Utama Penyebab Dropout**
   Analisis menunjukkan bahwa nilai akademik, jumlah kredit yang diambil, serta kondisi ekonomi seperti beasiswa dan status sosial memegang peran penting dalam keputusan siswa untuk berhenti kuliah.

3. **Strategi Meningkatkan Retensi**
   Institusi dapat meningkatkan tingkat kelulusan dengan menyediakan dukungan akademik yang lebih baik, menyesuaikan kurikulum agar lebih fleksibel, dan memperluas akses beasiswa bagi siswa yang membutuhkan.

### Rekomendasi Tindak Lanjut

1. **Implementasi Sistem Pemantauan Berbasis Data:** Gunakan model prediksi untuk pemantauan rutin terhadap risiko dropout.
2. **Perkuat Program Dukungan Akademik dan Psikologis:** Sediakan layanan bimbingan belajar dan konseling untuk siswa rentan.
3. **Revisi dan Optimalisasi Kurikulum:** Tinjau ulang kurikulum di program studi dengan tingkat dropout tinggi dan sesuaikan beban belajar siswa.

---

# deploy-streamlit
