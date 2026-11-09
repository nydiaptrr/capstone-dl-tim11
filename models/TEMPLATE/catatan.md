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
Implementasi masih menggunakan konfigurasi awal dan perlu diuji kembali setelah dataset simulasi tersedia.
