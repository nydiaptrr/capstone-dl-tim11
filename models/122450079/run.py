
"""Transfer Learning ResNet18 pralatih ImageNet + head linear.

Jalankan:
    python models/122450079/run.py --seed 1
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))

from src.protokol import data

NIM = Path(__file__).resolve().parent.name
JALUR = "tl"

UKURAN = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def ke_nchw(X):
    """Pastikan data berbentuk NCHW dan float32."""
    X = np.asarray(X, dtype=np.float32)

    if X.ndim == 3:
        X = X[:, None, :, :]
    elif X.ndim == 4 and X.shape[-1] in (1, 3):
        X = X.transpose(0, 3, 1, 2)

    if X.ndim != 4:
        raise ValueError(f"Bentuk data tidak dikenali: {X.shape}")

    return X


def siapkan_batch(xb, torch):
    """Ubah citra menjadi format input ResNet18 ImageNet."""
    F = torch.nn.functional

    if xb.shape[1] == 1:
        xb = xb.repeat(1, 3, 1, 1)

    if tuple(xb.shape[-2:]) != (UKURAN, UKURAN):
        xb = F.interpolate(
            xb,
            size=(UKURAN, UKURAN),
            mode="bilinear",
            align_corners=False,
        )

    mean = torch.tensor(MEAN, dtype=xb.dtype).view(1, 3, 1, 1)
    std = torch.tensor(STD, dtype=xb.dtype).view(1, 3, 1, 1)

    return (xb - mean) / std


def bangun_backbone(torch):
    """Memuat ResNet18 dengan bobot ImageNet, tanpa fallback bobot acak."""
    from torchvision import models

    bobot = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=bobot)
    model.fc = torch.nn.Identity()

    for parameter in model.parameters():
        parameter.requires_grad = False

    return model.eval()


def ekstrak(model, X, torch, batch_size=32):
    """Ekstraksi fitur menggunakan backbone yang dibekukan."""
    hasil = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i + batch_size])
            xb = siapkan_batch(xb, torch)
            fitur = model(xb)
            hasil.append(fitur.cpu())

    return torch.cat(hasil, dim=0)


def latih_dan_prediksi(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Memuat data sesuai protokol resmi tim.
    Xtr, ytr, Xva, _, _, _ = data.muat()

    ytr = np.asarray(ytr).astype(np.int64).ravel()
    kelas = np.unique(ytr)

    # Mengembalikan citra ke RGB 0-1.
    # Fungsi ini menggunakan statistik normalisasi dari data train.
    Xtr = data.balik_normalisasi(Xtr)
    Xva = data.balik_normalisasi(Xva)

    Xtr = ke_nchw(Xtr)
    Xva = ke_nchw(Xva)

    print("Data primer berhasil dimuat.")
    print("Train:", Xtr.shape)
    print("Validation:", Xva.shape)
    print("Kelas:", kelas)

    # Label dibuat berurutan untuk CrossEntropy.
    y_encoded = np.searchsorted(kelas, ytr)

    backbone = bangun_backbone(torch)

    print("\nMengekstrak fitur ResNet18...")
    Ftr = ekstrak(backbone, Xtr, torch)
    Fva = ekstrak(backbone, Xva, torch)

    print("Fitur train:", tuple(Ftr.shape))
    print("Fitur validation:", tuple(Fva.shape))

    # Hanya head linear yang dilatih.
    head = torch.nn.Linear(Ftr.shape[1], len(kelas))

    optimizer = torch.optim.Adam(
        head.parameters(),
        lr=1e-3,
    )

    y = torch.from_numpy(y_encoded).long()

    epochs = 20
    batch_size = 128

    print("\nMelatih classification head...")
    head.train()

    for epoch in range(epochs):
        urutan = torch.randperm(len(Ftr))
        total_loss = 0.0

        for i in range(0, len(Ftr), batch_size):
            idx = urutan[i:i + batch_size]

            optimizer.zero_grad()
            output = head(Ftr[idx])
            loss = torch.nn.functional.cross_entropy(output, y[idx])
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch + 1:02d}/{epochs} "
            f"- Loss: {total_loss:.4f}"
        )

    head.eval()

    with torch.no_grad():
        pred_encoded = head(Fva).argmax(dim=1).numpy()

    # Mengembalikan prediksi ke label asli.
    pred_val = kelas[pred_encoded]

    return pred_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--perangkat", default="cpu laptop")
    args = parser.parse_args()

    mulai = time.time()

    pred_val = latih_dan_prediksi(args.seed)

    durasi = time.time() - mulai

    output = AKAR / "results" / NIM
    output.mkdir(parents=True, exist_ok=True)

    path_prediksi = output / "pred_val.npy"
    np.save(path_prediksi, pred_val)

    print("\nEvaluasi validation set...")

    perintah = [
        sys.executable,
        "-m",
        "src.common.evaluate",
        "--nim", NIM,
        "--model", JALUR,
        "--seed", str(args.seed),
        "--pred-val", str(path_prediksi),
        "--durasi", f"{durasi:.2f}",
        "--perangkat", args.perangkat,
    ]

    subprocess.run(perintah, check=True, cwd=AKAR)

    print(f"\nWaktu keseluruhan: {durasi:.2f} detik")
    print("Hasil tersimpan di:", output / "metrics.json")


if __name__ == "__main__":
    main()
