# Catatan model 122450079

Jalur: TL (transfer learning). Status: kerangka `run.py` sudah berjalan pada data simulasi; pelatihan pada data meja menunggu data (diambil minggu depan).

## Alasan optimizer

Adam dengan lr 1e-3 untuk melatih head linear di atas fitur ResNet18 yang dibekukan. Adam dipilih karena tidak terlalu sensitif terhadap skala fitur dan cepat konvergen pada head sederhana; SGD biasa butuh penyetelan lr dan momentum lebih teliti. Nilai lr ini masih awal dan akan diuji (lihat tabel).

## Regularisasi

Saat ini belum ada regularisasi eksplisit (weight decay 0, tanpa dropout). Backbone yang dibekukan sudah membatasi jumlah parameter yang dilatih (hanya head 512 -> jumlah kelas). Akan dicoba weight decay pada head setelah data meja ada.

## Pencarian hyperparameter (tabel)

| Konfigurasi | lr | epoch | batch | weight decay | val acc | val macro-F1 |
|---|---|---|---|---|---|---|
| Awal (backbone beku) | 1e-3 | 20 | 128 | 0 | menunggu data meja | menunggu data meja |
| Rencana | 1e-4 | 20 | 128 | 0 | - | - |
| Rencana | 1e-3 | 40 | 128 | 1e-4 | - | - |

## Ablation (>= 2)

Direncanakan, dijalankan setelah data meja ada:

1. Backbone beku vs membuka `layer4` (fine-tuning sebagian).
2. Dengan vs tanpa weight decay pada head.

Hasil: menunggu data meja.

## Hasil 3 seed konfigurasi akhir

Menunggu data meja. Uji pada data simulasi (acak) hanya membuktikan `run.py` berjalan sampai akhir untuk seed 1 dan 2; angkanya tidak bermakna dan tidak dilaporkan.

| Seed | val acc | val macro-F1 |
|---|---|---|
| 1 | - | - |
| 2 | - | - |
| 3 | - | - |

## Praproses dan ukuran model

- Masukan model: citra 224x224, 3 kanal (1 kanal diulang jadi 3, resize bilinear), normalisasi mean/std ImageNet.
- Keluaran: jumlah kelas dari label latih (klasifikasi; 10 kelas pada data simulasi).
- Fitur diekstrak sekali dari backbone beku, lalu hanya head linear yang dilatih (cepat di CPU laptop).