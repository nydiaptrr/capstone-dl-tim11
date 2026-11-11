"""Jalur TL: ResNet18 pralatih (beku) + head linear. Masukan model 224x224x3.

    python models/<nim>/run.py --seed 1 [--final]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))
from src.protokol import data  # noqa: E402  (protokol tim yang dibekukan)

NIM = Path(__file__).resolve().parent.name     # nama folder = NIM
JALUR = "tl"

UKURAN = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def ke_nchw(X) -> np.ndarray:
    """Ubah ke float32 (N, C, H, W) berskala 0-1."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 3:                      # (N, H, W) -> 1 kanal
        X = X[:, None, :, :]
    elif X.ndim == 4 and X.shape[-1] in (1, 3) and X.shape[1] not in (1, 3):
        X = X.transpose(0, 3, 1, 2)      # (N, H, W, C) -> (N, C, H, W)
    if X.ndim != 4:
        raise ValueError(f"Bentuk X tidak dikenali: {X.shape}")
    if X.max() > 1.5:                    # skala 0-255 -> 0-1
        X = X / 255.0
    return X


def siapkan_batch(xb, torch):
    """Resize ke 224x224, jadikan 3 kanal, normalisasi ImageNet."""
    F = torch.nn.functional
    if xb.shape[1] == 1:
        xb = xb.repeat(1, 3, 1, 1)
    if tuple(xb.shape[-2:]) != (UKURAN, UKURAN):
        xb = F.interpolate(xb, size=(UKURAN, UKURAN), mode="bilinear", align_corners=False)
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    return (xb - mean) / std


def bangun_backbone(torch):
    from torchvision import models
    try:
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)   # pralatih ImageNet
    except Exception as e:                                             # tanpa internet
        print(f"PERINGATAN: bobot pralatih gagal diunduh ({e}). "
              "Memakai bobot acak, HANYA untuk uji kode, bukan untuk hasil asli.")
        m = models.resnet18(weights=None)
    m.fc = torch.nn.Identity()           # keluaran = fitur 512-dim
    for p in m.parameters():
        p.requires_grad = False          # bekukan backbone
    return m.eval()


def ekstrak(model, X, torch, bs=64):
    hasil = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs])
            hasil.append(model(siapkan_batch(xb, torch)))
    return torch.cat(hasil)


def latih_dan_prediksi(seed: int, final: bool) -> tuple[np.ndarray, np.ndarray | None]:
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, ytr, Xva, _, Xte, _ = data.muat()
    ytr = np.asarray(ytr).astype(np.int64)
    n_kelas = int(ytr.max()) + 1         # keluaran mengikuti jenis tugas (klasifikasi)

    backbone = bangun_backbone(torch)
    Ftr = ekstrak(backbone, ke_nchw(Xtr), torch)
    Fva = ekstrak(backbone, ke_nchw(Xva), torch)
    y = torch.from_numpy(ytr)
    print(f"masukan model: (N, 3, {UKURAN}, {UKURAN}) | fitur: {tuple(Ftr.shape)} | keluaran: {n_kelas} kelas")

    # ---- HEAD (satu-satunya bagian yang dilatih) ----
    head = torch.nn.Linear(Ftr.shape[1], n_kelas)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)   # tulis alasan pilihan di catatan.md
    EPOCH, BS = 20, 128
    for _ in range(EPOCH):
        urut = torch.randperm(len(Ftr))
        for i in range(0, len(Ftr), BS):
            idx = urut[i:i + BS]
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(head(Ftr[idx]), y[idx])
            loss.backward()
            opt.step()

    head.eval()
    with torch.no_grad():
        pred_val = head(Fva).argmax(1).numpy()
        pred_test = None
        if final and Xte is not None:
            pred_test = head(ekstrak(backbone, ke_nchw(Xte), torch)).argmax(1).numpy()
    return pred_val, pred_test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--final", action="store_true", help="sprint terakhir: evaluasi data uji sekali")
    ap.add_argument("--perangkat", default="cpu laptop")
    a = ap.parse_args()
    t0 = time.time()
    pred_val, pred_test = latih_dan_prediksi(a.seed, a.final)
    durasi = time.time() - t0
    out = AKAR / "results" / NIM
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_val.npy", np.asarray(pred_val))
    perintah = [sys.executable, "-m", "src.common.evaluate", "--nim", NIM, "--model", JALUR, "--seed", str(a.seed),
                "--pred-val", str(out / "pred_val.npy"), "--durasi", f"{durasi:.1f}", "--perangkat", a.perangkat]
    if pred_test is not None:
        np.save(out / "pred_test.npy", np.asarray(pred_test))
        perintah += ["--pred-test", str(out / "pred_test.npy")]
    subprocess.run(perintah, check=True, cwd=AKAR)


if __name__ == "__main__":
    main()