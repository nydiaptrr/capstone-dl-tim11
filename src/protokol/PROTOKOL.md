# Protokol data dan evaluasi — Tim 11

## Kasus dan sumber data

Klasifikasi empat tingkat bukaan pintu lemari/kotak penyimpanan dari citra kamera meja. Label
berasal dari rancangan kondisi (`pintu` di `label.csv`), bukan sensor pintu: `buka_penuh` (sekitar
90°), `buka_setengah` (sekitar 45°), `buka_sedikit` (celah beberapa sentimeter), dan `tutup`.
Kolom `pintu_awal` dan `pintu_akhir` memang kosong karena meja tidak memakai sensor pintu.

Data primer yang dipakai adalah sesi `data/eksperimen/sesi-20261117-174412` tanggal 17 Nov 2026,
dipasangkan dengan rekaman web `data/eksperimen/kamera/` dan `kamera_index.csv`. Log eksperimen
menyatakan rekaman web Tim 11 berjalan saat sesi. `label.csv` berisi 256 kondisi: P01–P04
masing-masing 64 kondisi, dan tiap kelas memiliki 64 kondisi.
Data simulasi hanya untuk
menyiapkan dan memeriksa alur pemotongan/pemuatan; data simulasi tidak masuk tabel komparasi
utama, pemilihan jalur, atau evaluasi purwarupa. Data sekunder boleh dipakai untuk pralatih,
augmentasi, atau evaluasi tambahan, tetapi hasil utama wajib memakai data uji meja.
Lokasi sesi, versi, dan daftar berkas yang dipakai dicatat di `data/MANIFEST.json`.

## Akuisisi dan praproses

- Setiap partisipan adalah satu blok; rancangan terdiri atas 4 partisipan × 64 kondisi.
- Tiap partisipan menjalani 4 kondisi pintu × 2 taraf lampu (0 dan 255) × 8 ulangan. Satu
  kondisi berlangsung 3 detik dengan jeda antarkondisi 1,5 detik agar partisipan sempat menarik
  tangan keluar dari bingkai. Jeda 1,5 detik dikonfirmasi oleh Penjaga Data pada 23 Nov 2026;
  persetujuan pembimbing tetap dicatat lewat PR protokol. Catatan `rencana.yaml` sesi 17 Nov yang
  sudah direkam justru mencantumkan `jeda_detik: 0.5`; protokol 1,5 detik tidak boleh dianggap
  sebagai pengaturan sesi tersebut.
- Rekaman web aktual adalah 640×480 setiap 1 detik. Untuk setiap kondisi, buang citra pada detik
  pertama (semua frame dengan waktu sebelum `t_mulai_host + 1`); cocokkan waktu citra dari
  `kamera_index.csv` dengan `t_mulai_host`/`t_selesai_host` pada `label.csv`.
- Crop tetap ROI `(280, 80, 545, 395)` pada citra kamera web (format kiri, atas, kanan, bawah),
  lalu ubah ke RGB 224×224. ROI dipusatkan pada kotak dan sama untuk semua citra/model.
- Selama kondisi, pintu diatur sebelum kondisi dimulai dan tangan/badan/bayangan partisipan berada
  di luar bingkai. Pantulan lampu pada pintu diperiksa sebagai potensi petunjuk palsu.
- Pada hari pengambilan data, pastikan setiap partisipan memiliki 64 kondisi. Periksa beberapa
  citra tengah kondisi `tutup` secara acak untuk memastikan pintu tampak tertutup rapat.

## Pembagian data dan evaluasi

`TUGAS = "klasifikasi"`. Indeks label tetap: `buka_penuh` = 0, `buka_setengah` = 1,
`buka_sedikit` = 2, `tutup` = 3.

- Tahan seluruh blok partisipan `P04` sebagai data uji; tidak satu pun kondisi/frame dari P04
  boleh masuk latih atau validasi. ID partisipan untuk ketiga subset dicatat agar irisan ID
  latih–uji dapat diperiksa otomatis.
- Dari tiga partisipan lain, pisahkan 10% **kondisi** (bukan frame secara terpisah) sebagai
  validasi menggunakan `src.common.split.indeks` dengan seed 2026. Semua frame dari satu kondisi
  harus berada pada subset yang sama. Sisa kondisi menjadi data latih.
- Setelah mencocokkan `kamera_index.csv` dengan label dan membuang detik pertama, tersedia 513
  frame: 348 latih, 37 validasi, dan 128 uji. Kondisi dibagi 173 latih, 19 validasi, dan 64 uji.
  Jumlah ini berdasarkan arsip saat ini; muat ulang/hitung dari arsip yang sama bila berkas berubah.
- Hitung statistik normalisasi kanal hanya dari frame latih setelah crop dan resize; terapkan
  statistik yang sama tanpa menghitung ulang pada validasi maupun uji.
- Metrik utama komparasi adalah F1-makro. Laporkan rerata ± simpangan baku dari tiga seed pada
  konfigurasi final, serta ukuran ONNX (KB) dan latensi (ms) di laptop meja. Data uji meja
  dievaluasi sekali oleh Penjaga Data pada sprint terakhir (`--final`).
- Pembanding non-DL: histogram warna + HOG + Random Forest. Jalur tim: CNN, Transfer Learning,
  Transformer, dan Autoencoder.

| Butir | Ketetapan |
|---|---|
| Ukuran latih / validasi / uji | 348 / 37 / 128 frame (173 / 19 / 64 kondisi) |
| Anggaran komputasi | Maksimal 20 menit CPU laptop per run, kecuali disetujui lain oleh pembimbing |
| Hash data saat dibekukan | Diisi setelah manifest final dan baseline dijalankan; ambil dari `results/baseline/metrics.json` |
| Jeda antarkondisi | Sesi aktual: 0,5 detik menurut `rencana.yaml`. Protokol untuk akuisisi berikutnya: 1,5 detik, dikonfirmasi Penjaga Data 23 Nov 2026; persetujuan pembimbing dicatat lewat PR. |
| Persetujuan pembimbing / PR | Belum tercatat; lengkapi tautan PR dan tanggal setelah diajukan/disetujui |

## Batas implementasi saat ini

`src/protokol/data.py` membaca frame web dan label eksperimen berdasarkan waktu, membuang detik
pertama, menerapkan ROI tetap, membagi uji per partisipan dan validasi per kondisi, lalu
menstandarkan kanal menggunakan statistik data latih saja. Rekaman web yang digunakan memang
640×480 tiap 1 detik; ini karakteristik data yang dikumpulkan, sementara konfigurasi rencana meja
menyebut 1280×720 tiap 0,5 detik. `data_hash` final dihasilkan setelah baseline berjalan.
Dokumen ini masih draf dan belum diajukan lewat pull request.
