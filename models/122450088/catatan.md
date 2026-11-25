# Catatan Model Autoencoder

## Jalur
Autoencoder (AE)

## Arsitektur
Model menggunakan Convolutional Autoencoder yang terdiri dari encoder dan decoder. Encoder mempelajari representasi fitur citra melalui proses kompresi, sedangkan decoder merekonstruksi citra dari representasi laten.

Representasi laten digunakan sebagai masukan classifier sederhana untuk memprediksi empat kondisi bukaan pintu.

## Optimizer
Adam dengan learning rate 0.001.

## Fungsi Loss
Total loss merupakan penjumlahan Mean Squared Error (MSE) untuk rekonstruksi citra dan Cross Entropy untuk klasifikasi.

## Konfigurasi Awal
- Ukuran citra: 224 × 224
- Dimensi latent: 128
- Batch size: 32
- Epoch: 20
- Optimizer: Adam
- Learning rate: 0.001

## Keterbatasan
Run awal pada data primer seed 1 menghasilkan akurasi validasi 0,243243 dan macro-F1 0,241228
pada 37 sampel validasi (durasi 88,7 detik). Masukan model `[N, 3, 224, 224]` dan head
menghasilkan logits untuk empat kelas. Data uji tidak dievaluasi. Tiga seed final, tuning, dan
ablation dikerjakan pada Sprint 2.
