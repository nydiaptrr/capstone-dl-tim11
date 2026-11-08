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
| awal | 1e-3 | 32 | 0,3 | 8 | belum diuji di data meja (data baru direkam 6 Nov) |

## Ablation (>= 2)
Belum dikerjakan. Rencananya: (1) dengan dan tanpa Dropout, (2) dengan dan tanpa
BatchNorm.

## Hasil 3 seed konfigurasi akhir
Belum ada. Menunggu data meja dan pelatihan di Sprint 1 dan 2.