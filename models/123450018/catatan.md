# Catatan model 123450018

## Alasan optimizer
Saya memakai Adam dengan learning rate awal 1e-3. Dari Modul 03, SGD sangat bergantung
pada learning rate, sedangkan Adam menyesuaikan besar langkah untuk tiap parameter dan
bisa berjalan pada learning rate yang jauh lebih kecil. Jadi lebih mudah dipakai, apalagi
waktu pelatihan kami terbatas (maksimal 20 menit per run di CPU). Ini baru pilihan awal.
Nanti pilihan akhirnya saya bandingkan seperti di Modul 03: SGD, Momentum, dan Adam
masing-masing memakai learning rate terbaiknya, dengan model, split, dan jumlah update
yang sama, lalu dipilih dari validation loss.

## Regularisasi
Saya memakai Batch Normalization setelah tiap lapisan konvolusi dan Dropout 0,3 sebelum
lapisan linear terakhir. Pengaruhnya akan saya uji di ablation.

## Pencarian hyperparameter (tabel)
| Konfigurasi | lr | batch | dropout | epoch | catatan |
|---|---|---|---|---|---|
| awal | 1e-3 | 32 | 0,3 | 8 | seed 1 data primer: val macro-F1 0,261364, 37 sampel |

## Ablation (>= 2)
Belum dikerjakan. Rencananya: (1) dengan dan tanpa Dropout, (2) dengan dan tanpa
BatchNorm.

## Hasil 3 seed konfigurasi akhir
Run awal seed 1 pada data primer: akurasi validasi 0,243243, macro-F1 0,261364, durasi 32,9
detik. Masukan `[N, 3, 224, 224]`, keluaran 4 kelas. Data uji tidak dievaluasi. Tiga seed final
dan ablation masih dikerjakan pada Sprint 2.
