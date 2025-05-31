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

---

## Persiapan

Berikut adalah versi terbaru bagian **Persiapan**, khususnya bagian **Sumber Data**, yang telah dilengkapi dengan tautan dan deskripsi dataset sesuai permintaan:

---

## Persiapan

### Sumber Data

Dataset yang digunakan dalam proyek ini merupakan kumpulan data dari institusi pendidikan tinggi, yang mencakup informasi mahasiswa dari berbagai jurusan sarjana seperti agronomi, desain, pendidikan, keperawatan, jurnalisme, manajemen, layanan sosial, dan teknologi.

Dataset ini mencakup:

* Informasi saat pendaftaran mahasiswa (jalur akademik, demografi, dan faktor sosial-ekonomi)
* Kinerja akademik mahasiswa di akhir semester pertama dan kedua

Tujuan dari dataset ini adalah untuk membangun model klasifikasi guna memprediksi potensi dropout mahasiswa dan kesuksesan akademiknya.

📄 **Nama Dataset:** Students' Performance
🔗 **Link Dataset:** [https://github.com/dicodingacademy/dicoding\_dataset/tree/main/students\_performance](https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance)

---

### Setup Environment

Berikut tahapan setup environment untuk menjalankan proyek ini.

#### Setup Environment - Anaconda

```bash
conda create --name dropout-predictor python=3.9
conda activate dropout-predictor
pip install -r requirements.txt
````

#### Setup Environment - Shell/Terminal

```bash
pip install pipenv
pipenv install
pipenv shell
pip install -r requirements.txt
```

---

## Menjalankan Sistem Machine Learning

Sistem prediksi dikembangkan menggunakan algoritma Random Forest dan ditampilkan dalam bentuk aplikasi Streamlit.

### Cara Menjalankan Secara Lokal

```bash
streamlit run app.py
```

### Link Deployment

🚀 Sistem prediksi juga telah dideploy secara online dan dapat diakses melalui tautan berikut:

🔗 [https://deploy-app.streamlit.app/](https://deploy-app.streamlit.app/)

---

## Dashboard Bisnis

Dashboard yang dikembangkan memberikan wawasan menyeluruh terhadap status mahasiswa (Dropout, Enrolled, Graduate). Dashboard ini membantu pengambilan keputusan berdasarkan tren dan faktor risiko yang teridentifikasi.

🔗 **Lihat Dashboard Looker Studio:**
[https://lookerstudio.google.com/reporting/aee31d18-8d2e-4986-88ee-521dec85993b](https://lookerstudio.google.com/reporting/aee31d18-8d2e-4986-88ee-521dec85993b)

![syfa_oktapiani02-dashboard](https://github.com/user-attachments/assets/83420d40-65cb-4cdc-87f8-056451eab1b7)

---

## Kesimpulan

Proyek ini menjawab berbagai tantangan yang dihadapi oleh Jaya Jaya Institut dalam menurunkan angka dropout mahasiswa.

1. **Identifikasi Dini Siswa Berisiko**
   Dengan memanfaatkan model Random Forest, institusi mampu memprediksi siswa yang berisiko tinggi dropout berdasarkan data historis dan variabel yang relevan.

2. **Faktor Utama Penyebab Dropout**
   Analisis menunjukkan bahwa nilai akademik, jumlah kredit yang diambil, serta kondisi ekonomi seperti beasiswa dan status sosial memegang peran penting dalam keputusan siswa untuk berhenti kuliah.

3. **Strategi Meningkatkan Retensi**
   Institusi dapat meningkatkan tingkat kelulusan dengan menyediakan dukungan akademik yang lebih baik, menyesuaikan kurikulum agar lebih fleksibel, dan memperluas akses beasiswa bagi siswa yang membutuhkan.

---

## Rekomendasi Tindak Lanjut

1. **Implementasi Sistem Pemantauan Berbasis Data:**
   Gunakan model prediksi untuk pemantauan rutin terhadap risiko dropout.

2. **Perkuat Program Dukungan Akademik dan Psikologis:**
   Sediakan layanan bimbingan belajar dan konseling untuk siswa rentan.

3. **Revisi dan Optimalisasi Kurikulum:**
   Tinjau ulang kurikulum di program studi dengan tingkat dropout tinggi dan sesuaikan beban belajar siswa.

---
