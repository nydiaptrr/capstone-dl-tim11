# Catatan model 122450079

Jalur: TL (transfer learning), ResNet18 ImageNet pralatih dibekukan dan head linear dilatih.
Run primer seed 1 sudah selesai. Bobot resmi ResNet18 disimpan pada cache PyTorch lokal, tidak
di-commit ke repositori.

## Alasan optimizer

Adam dengan lr 1e-3 untuk melatih head linear di atas fitur ResNet18 yang dibekukan. Adam dipilih karena tidak terlalu sensitif terhadap skala fitur dan cepat konvergen pada head sederhana; SGD biasa butuh penyetelan lr dan momentum lebih teliti. Nilai lr ini masih awal dan akan diuji (lihat tabel).

## Regularisasi

Saat ini belum ada regularisasi eksplisit (weight decay 0, tanpa dropout). Backbone yang dibekukan sudah membatasi jumlah parameter yang dilatih (hanya head 512 -> jumlah kelas). Akan dicoba weight decay pada head setelah data meja ada.

## Pencarian hyperparameter (tabel)

| Konfigurasi | lr | epoch | batch | weight decay | val acc | val macro-F1 |
|---|---|---|---|---|---|---|
| Awal (backbone beku) | 1e-3 | 20 | 128 | 0 | 0,270270 | 0,236957 |
| Rencana | 1e-4 | 20 | 128 | 0 | - | - |
| Rencana | 1e-3 | 40 | 128 | 1e-4 | - | - |

## Ablation (>= 2)

Direncanakan, dijalankan setelah data meja ada:

1. Backbone beku vs membuka `layer4` (fine-tuning sebagian).
2. Dengan vs tanpa weight decay pada head.

Hasil: belum dilakukan.

## Hasil 3 seed konfigurasi akhir

Konfigurasi awal pada data primer meja. Belum hasil konfigurasi final tiga seed.

| Seed | val acc | val macro-F1 |
|---|---|---|
| 1 | 0,270270 | 0,236957 |
| 2 | - | - |
| 3 | - | - |

## Praproses dan ukuran model

- Masukan model: citra 224x224, 3 kanal (1 kanal diulang jadi 3, resize bilinear), normalisasi mean/std ImageNet.
- Keluaran: jumlah kelas dari label latih (klasifikasi; 4 kelas pada data primer).
- Fitur diekstrak sekali dari backbone beku, lalu hanya head linear yang dilatih (cepat di CPU laptop).

Run `python models/122450079/run.py --seed 1` selesai pada data primer: 37 sampel validasi,
macro-F1 0,236957, durasi 28,1 detik. Data uji tidak dievaluasi. Citra dari pemuat data
distandardisasi memakai latih saja; sebelum backbone, dibalik ke RGB 0–1 memakai statistik latih
yang sama, lalu dinormalisasi sesuai ImageNet.
