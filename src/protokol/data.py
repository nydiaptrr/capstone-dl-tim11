"""Pemuat data kasus tim — MILIK TIM, dibekukan setelah Sprint 0 dengan persetujuan pembimbing.

Isi oleh Penjaga Data & Evaluasi. Kontrak yang dipakai penilai dan baseline starter:

    TUGAS = "klasifikasi"  # atau "regresi"
    muat() -> (X_train, y_train, X_val, y_val, X_test, y_test)
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from pathlib import Path

from src.common import manifest, split

TUGAS = "klasifikasi"

def muat():
    """Memuat citra klasifikasi pintu terbuka (4 kelas) dan membaginya sesuai protokol."""
    m = manifest.baca()
    akar_data = manifest.AKAR / "data" / "raw"
    
    # 1. Peta label klasifikasi
    kelas_map = {
        "buka_penuh": 0, 
        "buka_setengah": 1, 
        "buka_sedikit": 2, 
        "tutup": 3
    }
    
    X_all, y_all, groups = [], [], []
    
    # 2. Iterasi folder dan memuat citra
    for kelas_nama, kelas_idx in kelas_map.items():
        folder_kelas = akar_data / kelas_nama
        if not folder_kelas.exists():
            continue
            
        for file_path in folder_kelas.glob("*.jpg"):
            nama_file = file_path.stem
            
            # TODO: Sesuaikan logika pemotongan string dengan format penamaan riil
            # Asumsi nama file: P01_buka_penuh_001.jpg
            parts = nama_file.split("_")
            partisipan = parts[0]           # Contoh: "P01"
            frame_idx = int(parts[-1])      # Contoh: 1, 2, 3...
            
            # Sesuai batasan: 1 detik pertama tiap kondisi dibuang (asumsi 2 frame awal jika 0.5s/frame)
            if frame_idx <= 2:
                continue
                
            # Muat citra, paksa ukuran 224x224, ubah ke format [C, H, W] PyTorch
            img = Image.open(file_path).convert("RGB").resize((224, 224))
            arr = np.array(img, dtype=np.float32).transpose((2, 0, 1))
            
            X_all.append(arr)
            y_all.append(kelas_idx)
            groups.append(partisipan)
            
    # Gabungkan ke array NumPy
    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.int64)
    groups = np.array(groups)
    
    if len(X_all) == 0:
        raise ValueError("Data tidak ditemukan. Pastikan citra mentah ada di data/raw/nama_kelas/")
    
    # 3. Split Data Uji (Berdasarkan Partisipan/Group)
    # Menjadikan P04 murni sebagai data uji (tidak pernah dilihat saat training)
    mask_test = (groups == "P04")
    X_test, y_test = X_all[mask_test], y_all[mask_test]
    
    # Sisa data untuk Latih dan Validasi
    X_sisa, y_sisa = X_all[~mask_test], y_all[~mask_test]
    
    # 4. Split Latih & Validasi (WAJIB dari src.common.split)
    tr_idx, va_idx = split.indeks(len(X_sisa))
    
    X_train, y_train = X_sisa[tr_idx], y_sisa[tr_idx]
    X_val, y_val = X_sisa[va_idx], y_sisa[va_idx]
    
    return X_train, y_train, X_val, y_val, X_test, y_test