"""run.py CNN dari nol (Modul 05): klasifikasi 4 bukaan pintu kotak dari citra 224x224.

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
JALUR = "cnn"                                   # cnn | tl | rnn | trf | ae

# Hyperparameter awal (catat pencarian dan alasannya di catatan.md)
N_KELAS = 4
EPOCH = 8
BATCH = 32
LR = 1e-3
DROPOUT = 0.3
BATAS_DETIK = 18 * 60    # anggaran komputasi: <= 20 menit per run di CPU


def bangun_model(dropout: float):
    import torch.nn as nn

    def blok(c_in, c_out, stride=1, pool=True):
        lapisan = [nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False),
                   nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
        if pool:
            lapisan.append(nn.MaxPool2d(2))
        return lapisan

    return nn.Sequential(
        *blok(3, 16, stride=2),        # 224 -> 112 -> 56
        *blok(16, 32),                 # 56 -> 28
        *blok(32, 64),                 # 28 -> 14
        *blok(64, 128, pool=False),    # 14
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Dropout(dropout),
        nn.Linear(128, N_KELAS),
    )


def statistik_kanal(X):
    """Rerata dan simpangan baku per kanal, dihitung dari DATA LATIH saja."""
    s = np.zeros(3, dtype=np.float64)
    s2 = np.zeros(3, dtype=np.float64)
    for i in range(0, len(X), 64):
        b = X[i:i + 64].astype(np.float64)
        s += b.sum(axis=(0, 2, 3))
        s2 += (b ** 2).sum(axis=(0, 2, 3))
    cnt = len(X) * X.shape[2] * X.shape[3]
    mean = s / cnt
    std = np.sqrt(np.maximum(s2 / cnt - mean ** 2, 1e-12))
    return mean, std


def latih_dan_prediksi(seed: int, final: bool) -> tuple[np.ndarray, np.ndarray | None]:
    """Kembalikan (pred_val, pred_test|None)."""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, ytr, Xva, _, Xte, _ = data.muat()

    mean, std = statistik_kanal(Xtr)
    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)

    def siap(X):
        return (torch.from_numpy(np.asarray(X, dtype=np.float32)) - mean_t) / std_t

    ytr_t = torch.from_numpy(np.asarray(ytr, dtype=np.int64))
    model = bangun_model(DROPOUT)
    opt = torch.optim.Adam(model.parameters(), lr=LR)   # TULIS alasan pilihan optimizer di catatan.md

    t0 = time.time()
    n = len(Xtr)
    for ep in range(EPOCH):
        model.train()
        urutan = torch.randperm(n)
        total = 0.0
        for i in range(0, n, BATCH):
            idx = urutan[i:i + BATCH]
            xb, yb = siap(Xtr[idx.numpy()]), ytr_t[idx]
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        print(f"epoch {ep + 1}/{EPOCH}  loss={total / n:.4f}", flush=True)
        if time.time() - t0 > BATAS_DETIK:
            print("Batas waktu tercapai, pelatihan dihentikan.", flush=True)
            break

    def prediksi(X):
        model.eval()
        hasil = []
        with torch.no_grad():
            for i in range(0, len(X), 64):
                hasil.append(model(siap(X[i:i + 64])).argmax(1).numpy())
        return np.concatenate(hasil).astype(np.int64)

    pred_val = prediksi(Xva)
    pred_test = prediksi(Xte) if final and Xte is not None and len(Xte) > 0 else None
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