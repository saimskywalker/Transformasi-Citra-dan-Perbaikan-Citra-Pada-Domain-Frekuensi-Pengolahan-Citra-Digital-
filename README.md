# Transformasi dan Perbaikan Citra pada Domain Frekuensi

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aisyhnabilaaa/Transformasi-Citra-dan-Perbaikan-Citra-Pada-Domain-Frekuensi-Pengolahan-Citra-Digital-/blob/main/domain_frekuensi.ipynb)

Tugas mata kuliah Pengolahan Citra Digital. Notebook `domain_frekuensi.ipynb` mengubah gambar `oyencat.jpg` (skala abu-abu) ke domain frekuensi dengan transformasi Fourier, lalu menerapkan beberapa filter untuk menghaluskan, menajamkan, dan mengurangi noise. Setiap sel menampilkan gambar asli berdampingan dengan hasilnya.

## Intuisi singkat

Setelah transformasi Fourier dan `fftshift`, frekuensi rendah berada di tengah spektrum dan frekuensi tinggi di pinggir. Frekuensi rendah mewakili area yang halus dan perubahan warna yang pelan, sedangkan frekuensi tinggi mewakili tepi, tekstur, dan detail halus. Cutoff di notebook diukur dalam piksel dari titik tengah spektrum.

## Teknik di notebook

1. **Transformasi Fourier (FFT/DFT)**: gambar diubah ke domain frekuensi dan ditampilkan sebagai spektrum magnitudo `20 * log(|F|)`, sehingga terlihat seberapa banyak energi di tiap frekuensi.
2. **Ideal lowpass dan highpass**: semua frekuensi di dalam (atau di luar) cutoff diloloskan penuh dan sisanya dibuang; potongan yang tajam ini memunculkan efek *ringing*, yaitu riak bergelombang di sekitar tepi.
3. **Butterworth lowpass dan highpass**: transisinya landai `1 / (1 + (D/D0)^(2n))`, sehingga ringing jauh berkurang dan makin besar orde `n` makin mirip filter Ideal.
4. **Gaussian lowpass dan highpass**: transisinya paling halus (kurva Gaussian), jadi hampir tidak ada ringing; di notebook `sigma` dipilih agar nilai filter tepat 0,5 pada cutoff.
5. **Unsharp masking**: versi buram (Gaussian blur) dikurangkan dari gambar asli untuk mendapatkan detail, lalu detail itu ditambahkan kembali dengan faktor `strength` agar tepi lebih tegas.
6. **Laplacian**: operator turunan kedua yang menonjolkan tepi; di notebook dihitung dengan `cv2.Laplacian` di domain spasial.
7. **Selective filtering (bilateral)**: `cv2.bilateralFilter` menghaluskan noise sambil menjaga tepi tetap tajam, juga di domain spasial.

Catatan kecil untuk yang membaca kodenya: sel Ideal lowpass memakai mask kotak (bukan lingkaran), dan fungsi `butterworth_lowpass_filter` serta `butterworth_highpass_filter` juga memakai mask kotak berbatas tegas. Filter Butterworth dengan rumus lengkap ada di sel yang membuat variabel `butterworth_filter`.

## Cara menjalankan

### Google Colab

1. Klik badge **Open in Colab** di atas.
2. Unggah `oyencat.jpg` ke panel *Files* di Colab, karena notebook membaca gambar dari `/content/oyencat.jpg`.
3. Jalankan sel dari atas ke bawah (*Runtime > Run all*).

### Lokal

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install numpy opencv-python matplotlib scipy jupyter
jupyter notebook domain_frekuensi.ipynb
```

## Tentang file gambar

`oyencat.jpg` belum ada di repositori ini, jadi siapkan sendiri (atau pakai gambar lain dengan nama yang sama). Kalau dijalankan lokal, ganti path `/content/oyencat.jpg` di setiap sel menjadi `oyencat.jpg` dan taruh gambarnya satu folder dengan notebook. Gambar dibaca dalam skala abu-abu; kalau path salah, `cv2.imread` mengembalikan `None` dan sel berikutnya akan error.

## Pengembangan selanjutnya

Materi ini bisa dikembangkan menjadi **Frekuensi Playground**, aplikasi web interaktif berbasis React + Vite + shadcn/ui yang berjalan sepenuhnya di browser (canvas, tanpa server) untuk mencoba cutoff, orde, dan jenis filter secara langsung.
Library Python `frekuensi` dipakai sebagai implementasi acuan agar hasil di browser bisa dicek kebenarannya.
