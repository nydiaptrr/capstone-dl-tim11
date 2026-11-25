# Catatan model Transformer — 123450029

## Arsitektur dan ukuran masukan/keluaran

`run.py` memakai Transformer encoder citra kecil. Citra RGB 224×224 dibagi oleh convolutional
patch embedding menjadi 196 token berukuran 16×16, ditambah satu token klasifikasi. Encoder
memiliki 2 lapisan, 4 attention head, embedding 64, dan feed-forward 128. Head linear menghasilkan
logit untuk 4 kelas bukaan. Bentuk masukan `[N, 3, 224, 224]`; bentuk keluaran `[N, 4]`.

## Alasan optimizer

AdamW (`lr=3e-4`, `weight_decay=1e-4`) dipakai untuk mengoptimalkan encoder dan classifier.
Pemisahan weight decay dari pembaruan gradien membantu regularisasi bobot Transformer.

## Regularisasi

Dropout 0,1 pada encoder dan 0,2 pada head; weight decay AdamW sebesar `1e-4`. Checkpoint dengan
loss validasi terendah dipakai untuk prediksi validasi.

## Konfigurasi uji awal

| Parameter | Nilai |
|---|---:|
| Patch | 16×16 |
| Dimensi embedding | 64 |
| Lapisan encoder / attention head | 2 / 4 |
| Dimensi feed-forward | 128 |
| Batch / epoch maksimum | 16 / 8 |
| Learning rate / weight decay | 0,0003 / 0,0001 |

## Hasil awal

Perintah `python models/123450029/run.py --seed 1` selesai pada data tersedia dan menulis
`results/123450029/metrics.json` serta `pred_val.npy`. Hasil validasi seed 1: akurasi 0,324324,
macro-F1 0,248696, durasi 59,3 detik, 37 sampel validasi. Ini run awal, belum hasil tiga seed.
Data uji tidak dievaluasi.

Catatan tugas menyebut pengujian simulasi, tetapi checkout ini tidak memiliki sesi simulasi.
Run di atas memakai data yang saat ini dipakai `src.protokol.data.muat()`.

## Pencarian hyperparameter dan ablation

Belum dilakukan; dicatat untuk Sprint 2 bersama tiga seed konfigurasi final.
