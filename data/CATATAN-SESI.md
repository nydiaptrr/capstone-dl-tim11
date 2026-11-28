# Catatan sesi dan pemeriksaan mutu — Tim 11

## Sesi yang digunakan

- Sesi: `sesi-20261117-174412`, direkam 17 November 2026.
- Sumber: arsip eksperimen dan rekaman web yang waktunya berjalan bersamaan; rincian berkas tercantum di `data/MANIFEST.json`.
- Status perekaman: `simulasi=False`. Log mencatat rekaman kamera web berjalan selama eksperimen dan 256/256 kondisi selesai.
- Rencana mencatat jeda persiapan 0,5 detik dan kamera rencana 1280×720 setiap 0,5 detik. Arsip kamera web aktual yang dipakai berukuran 640×480 dengan interval 1 detik. Perbedaan ini dicatat; pemuat data mengikuti karakteristik arsip aktual.

## Pemeriksaan jumlah dan kelengkapan

- Total 256 kondisi, sesuai rancangan.
- P01, P02, P03, dan P04 masing-masing memiliki 64 kondisi.
- Tiap label desain (`buka penuh`, `buka setengah`, `buka sedikit`, `tutup`) memiliki 64 kondisi.
- Pencocokan waktu label dengan indeks kamera dan aturan membuang satu detik awal menghasilkan 513 frame: 348 latih, 37 validasi, dan 128 uji. Pemanggilan `src.protokol.data.muat()` berhasil; bentuk citra setiap subset `[N, 3, 224, 224]`.
- Tidak ada kondisi yang dikeluarkan: seluruh 256 kondisi memiliki sekurangnya satu frame setelah pemotongan waktu. Log mencatat putusnya sambungan Wi-Fi ESP32 sebentar, lalu tersambung kembali; log juga mencatat semua kondisi selesai.
- Uji dikelompokkan berdasarkan partisipan P04. Validasi dipisahkan per kondisi dari partisipan selain P04. Normalisasi memakai statistik latih.

## Pengecualian dan batas pemeriksaan

- Kolom `pintu_awal` dan `pintu_akhir` kosong karena meja tidak memakai sensor pintu. Secara khusus, `pintu_akhir` kosong pada seluruh 64 baris berlabel `tutup`; karena itu pemeriksaan bahwa nilainya 0 tidak dapat dilakukan pada data ini. Label `tutup` berasal dari rancangan kondisi (`pintu`), bukan pengukuran sensor. Tidak ada nilai yang diisi atau diubah secara artifisial.
- Pemeriksaan jumlah, label, kecocokan waktu/frame, dan keberhasilan pemuatan dilakukan secara programatis. Pemeriksaan visual acak bahwa pintu pada citra tengah kondisi `tutup` benar-benar rapat belum tercatat di sini; jangan anggap butir visual tersebut sudah dikonfirmasi.
- Tidak ada kondisi yang dikecualikan dari dataset. Perbedaan resolusi/interval kamera dan kolom sensor pintu kosong di atas merupakan keterbatasan/pengecualian yang dicatat.

## `data_hash` yang digunakan

Hash dihitung dari berkas yang tercantum di `data/MANIFEST.json` beserta parameter split `2026`. Nilainya:

`b87eda5a3a7d5da25ecc14b3a603dfbd`

Nilai pada `results/<nim>/metrics.json` per 28 November 2026:

| NIM/artefak | `data_hash` |
|---|---|
| 122450079 | `b87eda5a3a7d5da25ecc14b3a603dfbd` |
| 122450088 | `b87eda5a3a7d5da25ecc14b3a603dfbd` |
| 123450018 | `b87eda5a3a7d5da25ecc14b3a603dfbd` |
| 123450029 | `b87eda5a3a7d5da25ecc14b3a603dfbd` |
| baseline tim | `b87eda5a3a7d5da25ecc14b3a603dfbd` |

Semua anggota yang memiliki artefak metrik di repo saat pemeriksaan memakai hash yang sama. Untuk menjaga kesamaan, setiap anggota perlu memastikan berkas lokalnya berasal dari manifest dan revisi data tim yang sama.
