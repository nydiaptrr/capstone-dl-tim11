"""Pemuat citra Tim 11 dari arsip eksperimen dan rekaman kamera web yang tersinkron.

Kontrak penilai:
    TUGAS = "klasifikasi"
    muat() -> (X_train, y_train, X_val, y_val, X_test, y_test)

X dikembalikan sebagai float32 [N, C, H, W] yang dinormalisasi memakai rerata/simpangan
baku kanal dari data latih saja. Split validasi dilakukan per kondisi; P04 menjadi uji.
"""
from __future__ import annotations

import bisect
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from src.common import manifest, split

TUGAS = "klasifikasi"

# Arsip eksperimen lengkap dan rekaman web yang berjalan bersamaan (lihat log sesi).
SESI = "sesi-20261117-174412"
FOLDER_DATA = manifest.AKAR / "data" / "eksperimen"
FOLDER_SESI = FOLDER_DATA / SESI
FOLDER_KAMERA = FOLDER_DATA / "kamera"
INDEKS_KAMERA = FOLDER_DATA / "kamera_index.csv"

# ROI tetap pada citra rekaman web 640x480: kotak/pintu di tengah bingkai.
# Koordinat Pillow: (kiri, atas, kanan, bawah). Crop membuang sebagian besar area latar.
ROI = (280, 80, 545, 395)
UKURAN = (224, 224)
PARTISIPAN_UJI = "P04"

KELAS = {
    "buka penuh": 0,
    "buka setengah": 1,
    "buka sedikit": 2,
    "tutup": 3,
}

_MEAN_LATIH: np.ndarray | None = None
_STD_LATIH: np.ndarray | None = None


def _baca_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Berkas data tidak ditemukan: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _baca_gambar(nama: str) -> np.ndarray:
    path = FOLDER_KAMERA / nama
    if not path.is_file():
        raise FileNotFoundError(f"Citra {nama} yang tercatat di indeks kamera tidak ditemukan: {path}")
    with Image.open(path) as sumber:
        citra = sumber.convert("RGB")
        if citra.size != (640, 480):
            raise ValueError(f"Resolusi {nama} {citra.size}, diharapkan 640x480 sesuai rekaman web")
        kiri, atas, kanan, bawah = ROI
        if not (0 <= kiri < kanan <= citra.width and 0 <= atas < bawah <= citra.height):
            raise ValueError(f"ROI {ROI} berada di luar ukuran citra {citra.size}")
        potong = citra.crop(ROI).resize(UKURAN, Image.Resampling.BILINEAR)
        return np.asarray(potong, dtype=np.float32).transpose(2, 0, 1) / 255.0


def _normalisasi_latih(X_train: np.ndarray, *lainnya: np.ndarray) -> tuple[np.ndarray, ...]:
    """Standarkan tiap kanal memakai statistik piksel data latih saja, secara in-place."""
    mean = X_train.mean(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)[None, :, None, None]
    std = X_train.std(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)[None, :, None, None]
    std = np.maximum(std, 1e-6)
    global _MEAN_LATIH, _STD_LATIH
    _MEAN_LATIH, _STD_LATIH = mean.copy(), std.copy()
    X_train -= mean
    X_train /= std
    hasil = [X_train]
    for X in lainnya:
        X -= mean
        X /= std
        hasil.append(X)
    return tuple(hasil)


def balik_normalisasi(X: np.ndarray) -> np.ndarray:
    """Kembalikan citra ke skala RGB 0–1 memakai statistik latih dari pemanggilan ``muat`` terakhir.

    Dipakai TL untuk menerapkan normalisasi ImageNet yang diharapkan backbone pralatih.
    """
    if _MEAN_LATIH is None or _STD_LATIH is None:
        raise RuntimeError("Panggil muat() sebelum balik_normalisasi()")
    return np.clip(np.asarray(X, dtype=np.float32) * _STD_LATIH + _MEAN_LATIH, 0.0, 1.0)


def muat():
    """Baca, crop, labeli, dan split frame kamera yang waktunya berada di kondisi eksperimen."""
    label = _baca_csv(FOLDER_SESI / "label.csv")
    indeks = _baca_csv(INDEKS_KAMERA)
    if not label or not indeks:
        raise ValueError("label.csv atau indeks kamera kosong")

    wajib_label = {"no", "partisipan", "pintu", "t_mulai_host", "t_selesai_host"}
    wajib_indeks = {"berkas", "t_host"}
    if not wajib_label.issubset(label[0]) or not wajib_indeks.issubset(indeks[0]):
        raise ValueError("Kolom label/indeks tidak sesuai format sesi meja")

    # Setiap kondisi utuh berada di satu subset, agar frame berdekatan tidak bocor.
    kondisi_latihval = [r for r in label if r["partisipan"] != PARTISIPAN_UJI]
    if not kondisi_latihval or not any(r["partisipan"] == PARTISIPAN_UJI for r in label):
        raise ValueError(f"label.csv harus memuat partisipan uji {PARTISIPAN_UJI} dan partisipan latih")
    tr_idx, va_idx = split.indeks(len(kondisi_latihval), porsi_val=0.1, seed=split.SEED_SPLIT)
    id_latih = {kondisi_latihval[i]["no"] for i in tr_idx}
    id_val = {kondisi_latihval[i]["no"] for i in va_idx}
    id_uji = {r["no"] for r in label if r["partisipan"] == PARTISIPAN_UJI}

    # Tegaskan tidak ada kondisi/partisipan uji yang masuk subset pengembangan.
    if id_latih & id_val or id_latih & id_uji or id_val & id_uji:
        raise AssertionError("ID kondisi bocor antarsubset")
    part_latih = {r["partisipan"] for r in kondisi_latihval if r["no"] in id_latih}
    part_val = {r["partisipan"] for r in kondisi_latihval if r["no"] in id_val}
    part_uji = {r["partisipan"] for r in label if r["no"] in id_uji}
    if (part_latih | part_val) & part_uji:
        raise AssertionError("Partisipan muncul di data latih/validasi dan uji sekaligus")

    # Cari label kondisi berdasar waktu epoch kamera. Dua detik awal setiap kondisi dibuang
    # (laju rekaman aktual 1 citra/detik); jendela yang dipakai adalah [mulai+1, selesai).
    label_urut = sorted(label, key=lambda r: float(r["t_mulai_host"]))
    mulai = [float(r["t_mulai_host"]) for r in label_urut]
    akhir = [float(r["t_selesai_host"]) for r in label_urut]
    kumpulan: dict[str, tuple[list[np.ndarray], list[int]]] = {
        "train": ([], []), "val": ([], []), "test": ([], [])
    }
    kondisi_dengan_frame: set[str] = set()

    for frame in indeks:
        t = float(frame["t_host"])
        i = bisect.bisect_right(mulai, t) - 1
        if i < 0:
            continue
        row = label_urut[i]
        if not (t >= mulai[i] + 1.0 and t < akhir[i]):
            continue
        nama_kelas = row["pintu"].strip().lower()
        if nama_kelas not in KELAS:
            raise ValueError(f"Label pintu tidak dikenal: {row['pintu']!r}")

        if row["partisipan"] == PARTISIPAN_UJI:
            subset = "test"
        elif row["no"] in id_val:
            subset = "val"
        else:
            subset = "train"
        xs, ys = kumpulan[subset]
        xs.append(_baca_gambar(frame["berkas"]))
        ys.append(KELAS[nama_kelas])
        kondisi_dengan_frame.add(row["no"])

    kondisi_tanpa_frame = {r["no"] for r in label} - kondisi_dengan_frame
    if kondisi_tanpa_frame:
        contoh = ", ".join(sorted(kondisi_tanpa_frame)[:5])
        raise ValueError(f"Kondisi tanpa citra setelah buang 1 detik: {contoh}")

    if not all(kumpulan[k][0] for k in kumpulan):
        jumlah = {k: len(v[0]) for k, v in kumpulan.items()}
        raise ValueError(f"Ada subset tanpa citra cocok; jumlah frame: {jumlah}")

    X_train, y_train = np.stack(kumpulan["train"][0]), np.asarray(kumpulan["train"][1], dtype=np.int64)
    X_val, y_val = np.stack(kumpulan["val"][0]), np.asarray(kumpulan["val"][1], dtype=np.int64)
    X_test, y_test = np.stack(kumpulan["test"][0]), np.asarray(kumpulan["test"][1], dtype=np.int64)
    del kumpulan

    X_train, X_val, X_test = _normalisasi_latih(X_train, X_val, X_test)
    return X_train, y_train, X_val, y_val, X_test, y_test
